"""Focused tests for library operations and propositional logic."""

import unittest
from unittest.mock import Mock, patch

import library_management
from logic_module import (
    are_logically_equivalent,
    check_eligibility,
    disjunctive_syllogism,
    equivalent,
    exclusive_or,
    generate_truth_table,
    implies,
    logical_and,
    logical_not,
    logical_or,
    modus_ponens,
    modus_tollens,
    truth_table,
)


class LogicTests(unittest.TestCase):
    def test_issue_eligibility_truth_table(self):
        rows = generate_truth_table()
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            sum(row["I"] for row in rows),
            1,
        )
        permitted = next(row for row in rows if row["I"])
        self.assertEqual(
            (permitted["R"], permitted["E"], permitted["A"], permitted["O"]),
            (True, True, True, False),
        )
        self.assertFalse(check_eligibility(True, True, True, True))

    def test_connectives_and_logical_equivalence(self):
        self.assertTrue(logical_not(False))
        self.assertTrue(logical_and(True, True))
        self.assertFalse(logical_or(False, False))
        self.assertTrue(exclusive_or(True, False))
        self.assertTrue(implies(False, False))
        self.assertTrue(equivalent(True, True))
        self.assertTrue(
            are_logically_equivalent(
                lambda values: implies(values["P"], values["Q"]),
                lambda values: logical_or(
                    logical_not(values["P"]), values["Q"]
                ),
                ("P", "Q"),
            )
        )
        self.assertEqual(
            truth_table(lambda values: values["P"], ("P",)),
            [
                {"P": False, "result": False},
                {"P": True, "result": True},
            ],
        )

    def test_rules_of_inference(self):
        self.assertTrue(modus_ponens(True, True))
        self.assertFalse(modus_ponens(False, True))
        self.assertTrue(modus_tollens(True, True))
        self.assertTrue(disjunctive_syllogism(True, True))

    def test_rejects_non_boolean_propositions_and_bad_variables(self):
        with self.assertRaises(ValueError):
            check_eligibility(1, True, True, False)
        with self.assertRaises(ValueError):
            truth_table(lambda values: True, ("P", "P"))


class LibraryManagementTests(unittest.TestCase):
    def setUp(self):
        self.db = Mock()
        self.db.add_book.return_value = {
            "book_id": "B1",
            "title": "Discrete Math",
            "author": "A. Author",
            "total_copies": 2,
            "available_copies": 2,
        }
        self.db.get_books.return_value = [self.db.add_book.return_value]
        self.db.search_books.return_value = [self.db.add_book.return_value]
        self.db.update_book.return_value = self.db.add_book.return_value
        self.db.add_student.return_value = {
            "student_id": "S1",
            "name": "Sample Student",
            "registered": True,
            "eligible": True,
        }
        self.db.get_student.return_value = self.db.add_student.return_value
        self.db.get_students.return_value = [self.db.add_student.return_value]
        self.db.issue_book.return_value = {"student_id": "S1", "book_id": "B1"}
        self.db.return_book.return_value = {
            "student_id": "S1",
            "book_id": "B1",
            "status": "returned",
        }
        self.db.reserve_book.return_value = {
            "student_id": "S1",
            "book_id": "B1",
            "queue_position": 1,
        }
        self.database_patch = patch("library_management._database", return_value=self.db)
        self.database_patch.start()
        self.addCleanup(self.database_patch.stop)

    def test_book_add_defaults_available_stock(self):
        result = library_management.add_book(
            {
                "book_id": "B1",
                "title": "Discrete Math",
                "author": "A. Author",
                "total_copies": 2,
            }
        )
        self.assertEqual(result["available_copies"], 2)
        self.db.add_book.assert_called_once()
        self.assertEqual(
            self.db.add_book.call_args.args[0]["available_copies"], 2
        )

    def test_book_read_search_and_update(self):
        self.assertEqual(len(library_management.get_books()), 1)
        self.assertEqual(len(library_management.search_books("math")), 1)
        library_management.update_book("B1", {"title": "Discrete Structures"})
        self.db.update_book.assert_called_once_with(
            "B1", {"title": "Discrete Structures"}
        )
        with self.assertRaises(ValueError):
            library_management.update_book("B1", {"available_copies": 99})

    def test_student_add_and_read(self):
        student = {
            "student_id": "S1",
            "name": "Sample Student",
            "registered": True,
            "eligible": True,
        }
        self.assertEqual(library_management.add_student(student), student)
        self.assertEqual(library_management.get_student("S1"), student)
        self.assertEqual(library_management.get_students(), [student])

    def test_issue_return_reservation_delegate_transactional_operations(self):
        self.assertEqual(library_management.issue_book("S1", "B1")["book_id"], "B1")
        self.assertEqual(
            library_management.return_book("S1", "B1")["status"], "returned"
        )
        self.assertEqual(
            library_management.reserve_book("S1", "B1")["queue_position"], 1
        )
        self.db.issue_book.assert_called_once_with("S1", "B1")
        self.db.return_book.assert_called_once_with("S1", "B1")
        self.db.reserve_book.assert_called_once_with("S1", "B1")

    def test_rejects_invalid_input_before_database_call(self):
        with self.assertRaises(ValueError):
            library_management.add_book(
                {
                    "book_id": "B1",
                    "title": "Discrete Math",
                    "author": "A. Author",
                    "total_copies": -1,
                }
            )
        with self.assertRaises(ValueError):
            library_management.add_student(
                {"student_id": "S1", "name": "Sample Student", "registered": 1,
                 "eligible": True}
            )
        with self.assertRaises(ValueError):
            library_management.issue_book("", "B1")
        self.db.add_book.assert_not_called()
        self.db.add_student.assert_not_called()
        self.db.issue_book.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
