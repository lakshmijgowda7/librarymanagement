"""Tests for transactional SQLite persistence and circulation rules."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import database
import library_operations
import library_management


class DatabaseIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        database_path = Path(self.temporary_directory.name) / "library.sqlite"
        database_path_patch = patch.object(database, "DB_PATH", database_path)
        database_path_patch.start()
        self.addCleanup(database_path_patch.stop)
        database.init_db()

    def add_student(self, name="Sample Student", email="student@example.edu"):
        return library_operations.add_student(name, email, ["Computing"])

    def add_book(self, copies=1):
        return library_operations.add_book(
            "Discrete Structures", "A. Author", "Computing", copies
        )

    def test_issue_return_and_persistence(self):
        student = self.add_student()
        book = self.add_book()

        loan = library_operations.issue_book(student["student_id"], book["book_id"])
        self.assertEqual(database.get_books()[0]["available_copies"], 0)
        with self.assertRaisesRegex(ValueError, "No copies"):
            library_operations.issue_book(student["student_id"], book["book_id"])

        returned = library_operations.return_book(loan["loan_id"])
        self.assertEqual(returned["status"], "returned")
        self.assertEqual(database.get_books()[0]["available_copies"], 1)
        with self.assertRaisesRegex(ValueError, "already been returned"):
            library_operations.return_book(loan["loan_id"])

        database.init_db()
        self.assertEqual(database.get_loans()[0]["status"], "returned")

    def test_reservations_preserve_queue_order(self):
        borrower = self.add_student("Borrower", "borrower@example.edu")
        first = self.add_student("First", "first@example.edu")
        second = self.add_student("Second", "second@example.edu")
        book = self.add_book()
        loan = library_operations.issue_book(borrower["student_id"], book["book_id"])

        first_reservation = library_operations.reserve_book(
            first["student_id"], book["book_id"]
        )
        second_reservation = library_operations.reserve_book(
            second["student_id"], book["book_id"]
        )
        self.assertEqual(first_reservation["queue_position"], 1)
        self.assertEqual(second_reservation["queue_position"], 2)
        with self.assertRaisesRegex(ValueError, "already has a reservation"):
            library_operations.reserve_book(first["student_id"], book["book_id"])

        library_operations.return_book(loan["loan_id"])
        with self.assertRaisesRegex(ValueError, "earlier student"):
            library_operations.issue_book(second["student_id"], book["book_id"])
        first_loan = library_operations.issue_book(first["student_id"], book["book_id"])
        self.assertEqual(
            database.get_loan(first_loan["loan_id"])["status"], "active"
        )
        library_operations.return_book(first_loan["loan_id"])
        second_loan = library_operations.issue_book(
            second["student_id"], book["book_id"]
        )
        self.assertEqual(database.get_loan(second_loan["loan_id"])["status"], "active")

    def test_concurrent_issues_cannot_overdraw_stock(self):
        students = [
            self.add_student("Student One", "one@example.edu"),
            self.add_student("Student Two", "two@example.edu"),
        ]
        book = self.add_book()

        def try_issue(student_id):
            try:
                return library_operations.issue_book(student_id, book["book_id"])
            except ValueError:
                return None

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    try_issue, [student["student_id"] for student in students]
                )
            )

        self.assertEqual(sum(result is not None for result in results), 1)
        saved_book = database.get_books()[0]
        self.assertEqual(saved_book["available_copies"], 0)

    def test_stock_cannot_be_reduced_below_active_loans(self):
        student = self.add_student()
        book = self.add_book(copies=2)
        library_operations.issue_book(student["student_id"], book["book_id"])

        with self.assertRaisesRegex(ValueError, "less than active loans"):
            library_management.update_book(book["book_id"], {"total_copies": 0})

        updated = library_management.update_book(
            book["book_id"], {"total_copies": 1}
        )
        self.assertEqual(updated["available_copies"], 0)

    def test_search_treats_wildcards_as_literal_text(self):
        matching = library_operations.add_book(
            "Book 100% Complete", "A. Author", "Computing", 1
        )
        library_operations.add_book("Ordinary Book", "B. Author", "Math", 1)

        results = database.search_books("%")

        self.assertEqual([book["book_id"] for book in results], [matching["book_id"]])


if __name__ == "__main__":
    unittest.main()
