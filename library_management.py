"""Library operations backed by the shared ``database.py`` interface.

The database module is owned by the UI/database teammate and is intentionally
not created here. It must provide ``add_book(data)``, ``get_books()``,
``search_books(query)``, ``update_book(book_id, updates)``,
``add_student(data)``, ``get_student(student_id)``, ``get_students()``,
``issue_book(student_id, book_id)``, ``return_book(student_id, book_id)``, and
``reserve_book(student_id, book_id)``. Data operations return row dictionaries
(or lists of them; ``get_student`` may return ``None``) and raise ``ValueError``
for invalid or conflicting operations.

Issue, return, stock updates, and reservations must be transactional. The
database must enforce issue eligibility and availability, keep available stock
between zero and total stock, reject unmatched returns and duplicate
reservations, and append reservations in queue order.
"""

from collections.abc import Mapping
from importlib import import_module
from typing import Any

from logic_module import check_eligibility


def _database() -> Any:
    try:
        return import_module("database")
    except ModuleNotFoundError as exc:
        if exc.name != "database":
            raise
        raise ImportError(
            "Library operations require the shared database.py module."
        ) from exc


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _record(value: Any, operation: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"database.{operation} must return a row dictionary.")
    return dict(value)


def _records(value: Any, operation: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise TypeError(f"database.{operation} must return a list of row dictionaries.")
    return [_record(item, operation) for item in value]


def _book_data(book_data: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(book_data, Mapping):
        raise ValueError("book_data must be a mapping.")
    data = dict(book_data)
    for field in ("book_id", "title", "author"):
        data[field] = _identifier(data.get(field), field)

    total = data.get("total_copies")
    available = data.get("available_copies", total)
    for field, value in (("total_copies", total), ("available_copies", available)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer.")
    if available > total:
        raise ValueError("available_copies cannot exceed total_copies.")
    data["available_copies"] = available
    return data


def add_book(book_data: Mapping[str, Any]) -> dict[str, Any]:
    """Add a book; ``available_copies`` defaults to ``total_copies``."""
    data = _book_data(book_data)
    return _record(_database().add_book(data), "add_book")


def get_books() -> list[dict[str, Any]]:
    """Return all books as row dictionaries."""
    return _records(_database().get_books(), "get_books")


def search_books(query: str) -> list[dict[str, Any]]:
    """Search books using the shared database module's search behavior."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    return _records(_database().search_books(query.strip()), "search_books")


def update_book(book_id: str, updates: Mapping[str, Any]) -> dict[str, Any]:
    """Update book metadata or total stock without directly setting availability.

    The database implementation must adjust total and available copies atomically
    and reject a stock reduction that would put total copies below active loans.
    """
    identifier = _identifier(book_id, "book_id")
    if not isinstance(updates, Mapping) or not updates:
        raise ValueError("updates must be a non-empty mapping.")
    changes = dict(updates)
    allowed_fields = {"title", "author", "total_copies"}
    unknown_fields = set(changes) - allowed_fields
    if unknown_fields:
        raise ValueError(f"Unsupported book fields: {sorted(unknown_fields)!r}.")
    for field in ("title", "author"):
        if field in changes:
            changes[field] = _identifier(changes[field], field)
    if "total_copies" in changes:
        total = changes["total_copies"]
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("total_copies must be a non-negative integer.")
    return _record(_database().update_book(identifier, changes), "update_book")


def add_student(student_data: Mapping[str, Any]) -> dict[str, Any]:
    """Add a student with explicit registration and eligibility flags."""
    if not isinstance(student_data, Mapping):
        raise ValueError("student_data must be a mapping.")
    data = dict(student_data)
    data["student_id"] = _identifier(data.get("student_id"), "student_id")
    data["name"] = _identifier(data.get("name"), "name")
    for field in ("registered", "eligible"):
        if field not in data or not isinstance(data[field], bool):
            raise ValueError(f"{field} must be a boolean.")
    return _record(_database().add_student(data), "add_student")


def get_student(student_id: str) -> dict[str, Any] | None:
    """Return a student row, or ``None`` if the ID is unknown."""
    identifier = _identifier(student_id, "student_id")
    result = _database().get_student(identifier)
    return None if result is None else _record(result, "get_student")


def get_students() -> list[dict[str, Any]]:
    """Return all students as row dictionaries."""
    return _records(_database().get_students(), "get_students")


def issue_book(student_id: str, book_id: str) -> dict[str, Any]:
    """Issue a book through an atomic database transaction.

    The database operation must enforce registration, eligibility, availability,
    and the no-overdue-loan rule, then decrement available stock exactly once.
    """
    student = _identifier(student_id, "student_id")
    book = _identifier(book_id, "book_id")
    return _record(_database().issue_book(student, book), "issue_book")


def return_book(student_id: str, book_id: str) -> dict[str, Any]:
    """Return an active loan through an atomic database transaction.

    The database operation must reject unmatched returns, update the loan once,
    increment available stock once, and preserve reservation queue order.
    """
    student = _identifier(student_id, "student_id")
    book = _identifier(book_id, "book_id")
    return _record(_database().return_book(student, book), "return_book")


def reserve_book(student_id: str, book_id: str) -> dict[str, Any]:
    """Join the end of the reservation queue for an unavailable book."""
    student = _identifier(student_id, "student_id")
    book = _identifier(book_id, "book_id")
    return _record(_database().reserve_book(student, book), "reserve_book")
