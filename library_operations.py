"""Application-facing library actions adapted to the library business layer."""

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import database
import library_management


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


def add_student(name: str, email: str, interests: Sequence[str]) -> dict[str, Any]:
    student_name = _required_text(name, "name")
    student_email = _required_text(email, "email")
    if isinstance(interests, (str, bytes)) or not isinstance(interests, Sequence):
        raise ValueError("interests must be a sequence of strings.")
    normalized_interests = [_required_text(value, "interest") for value in interests]
    return library_management.add_student(
        {
            "student_id": f"S-{uuid4().hex}",
            "name": student_name,
            "email": student_email,
            "interests": normalized_interests,
            "registered": True,
            "eligible": True,
        }
    )


def add_book(
    title: str, author: str, category: str, total_copies: int
) -> dict[str, Any]:
    book_title = _required_text(title, "title")
    book_author = _required_text(author, "author")
    book_category = _required_text(category, "category")
    if isinstance(total_copies, bool) or not isinstance(total_copies, int):
        raise ValueError("total_copies must be a positive integer.")
    if total_copies < 1:
        raise ValueError("total_copies must be a positive integer.")
    return library_management.add_book(
        {
            "book_id": f"B-{uuid4().hex}",
            "title": book_title,
            "author": book_author,
            "category": book_category,
            "total_copies": total_copies,
        }
    )


def issue_book(student_id: str, book_id: str) -> dict[str, Any]:
    return library_management.issue_book(
        _required_text(student_id, "student_id"),
        _required_text(book_id, "book_id"),
    )


def return_book(loan_id: str) -> dict[str, Any]:
    identifier = _required_text(loan_id, "loan_id")
    loan = database.get_loan(identifier)
    if loan is None:
        raise ValueError(f"Unknown loan ID: {identifier}.")
    if loan["status"] != "active":
        raise ValueError(f"Loan {identifier} has already been returned.")
    return library_management.return_book(loan["student_id"], loan["book_id"])


def reserve_book(student_id: str, book_id: str) -> dict[str, Any]:
    return library_management.reserve_book(
        _required_text(student_id, "student_id"),
        _required_text(book_id, "book_id"),
    )
