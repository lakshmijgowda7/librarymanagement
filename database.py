"""SQLite persistence for students, books, loans, and reservations."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from logic_module import check_eligibility

DB_PATH = Path(
    os.environ.get(
        "LIBRARY_DB_PATH", str(Path.home() / ".librarymanagement" / "library.db")
    )
)

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        name TEXT NOT NULL CHECK (length(trim(name)) > 0),
        email TEXT,
        interests TEXT NOT NULL DEFAULT '[]',
        registered INTEGER NOT NULL CHECK (registered IN (0, 1)),
        eligible INTEGER NOT NULL CHECK (eligible IN (0, 1)),
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS students_email_unique
    ON students(lower(email))
    WHERE email IS NOT NULL AND email <> ''
    """,
    """
    CREATE TABLE IF NOT EXISTS books (
        book_id TEXT PRIMARY KEY,
        title TEXT NOT NULL CHECK (length(trim(title)) > 0),
        author TEXT NOT NULL CHECK (length(trim(author)) > 0),
        category TEXT NOT NULL DEFAULT '',
        total_copies INTEGER NOT NULL CHECK (total_copies >= 0),
        available_copies INTEGER NOT NULL CHECK (
            available_copies >= 0 AND available_copies <= total_copies
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS loans (
        loan_id TEXT PRIMARY KEY,
        student_id TEXT NOT NULL REFERENCES students(student_id),
        book_id TEXT NOT NULL REFERENCES books(book_id),
        issued_at TEXT NOT NULL,
        due_at TEXT NOT NULL,
        returned_at TEXT,
        status TEXT NOT NULL CHECK (status IN ('active', 'returned')),
        CHECK (
            (status = 'active' AND returned_at IS NULL)
            OR (status = 'returned' AND returned_at IS NOT NULL)
        )
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS loans_one_active_book_per_student
    ON loans(student_id, book_id)
    WHERE status = 'active'
    """,
    """
    CREATE TABLE IF NOT EXISTS reservations (
        reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL REFERENCES students(student_id),
        book_id TEXT NOT NULL REFERENCES books(book_id),
        created_at TEXT NOT NULL,
        UNIQUE (student_id, book_id)
    )
    """,
)


@contextmanager
def _connect(
    *, write: bool = False
) -> Generator[sqlite3.Connection, None, None]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=30, isolation_level=None)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if write:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
        else:
            yield connection
    finally:
        connection.close()


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


def _non_negative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return value


def _record(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    if "interests" in result:
        try:
            result["interests"] = json.loads(result["interests"])
        except json.JSONDecodeError as error:
            raise RuntimeError("Stored student interests are not valid JSON.") from error
    return result


def _required_record(row: sqlite3.Row | None, entity: str) -> dict[str, Any]:
    result = _record(row)
    if result is None:
        raise RuntimeError(f"Could not read the newly saved {entity} record.")
    return result


def _records(rows: Sequence[sqlite3.Row]) -> list[dict[str, Any]]:
    return [_required_record(row, "database") for row in rows]


def init_db() -> None:
    """Create the schema and enable SQLite's write-ahead log."""
    with _connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
    with _connect(write=True) as connection:
        for statement in _SCHEMA:
            connection.execute(statement)


def add_book(book_data: Mapping[str, Any]) -> dict[str, Any]:
    book_id = _required_text(book_data.get("book_id"), "book_id")
    title = _required_text(book_data.get("title"), "title")
    author = _required_text(book_data.get("author"), "author")
    category_value = book_data.get("category", "")
    if not isinstance(category_value, str):
        raise ValueError("category must be a string.")
    category = category_value.strip()
    total = _non_negative_integer(book_data.get("total_copies"), "total_copies")
    available = _non_negative_integer(
        book_data.get("available_copies", total), "available_copies"
    )
    if available > total:
        raise ValueError("available_copies cannot exceed total_copies.")

    with _connect(write=True) as connection:
        connection.execute(
            """
            INSERT INTO books (
                book_id, title, author, category, total_copies, available_copies
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (book_id, title, author, category, total, available),
        )
        return _required_record(
            connection.execute(
                "SELECT * FROM books WHERE book_id = ?", (book_id,)
            ).fetchone(),
            "book",
        )


def get_books() -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM books ORDER BY lower(title), book_id"
        ).fetchall()
    return _records(rows)


def search_books(query: str) -> list[dict[str, Any]]:
    search = _required_text(query, "query")
    pattern = (
        "%"
        + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        + "%"
    )
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM books
            WHERE title LIKE ? ESCAPE '\\'
               OR author LIKE ? ESCAPE '\\'
               OR category LIKE ? ESCAPE '\\'
            ORDER BY lower(title), book_id
            """,
            (pattern, pattern, pattern),
        ).fetchall()
    return _records(rows)


def update_book(book_id: str, updates: Mapping[str, Any]) -> dict[str, Any]:
    identifier = _required_text(book_id, "book_id")
    if not updates:
        raise ValueError("updates must not be empty.")
    allowed = {"title", "author", "total_copies"}
    unknown = set(updates) - allowed
    if unknown:
        raise ValueError(f"Unsupported book fields: {sorted(unknown)!r}.")

    title = (
        _required_text(updates["title"], "title")
        if "title" in updates
        else None
    )
    author = (
        _required_text(updates["author"], "author")
        if "author" in updates
        else None
    )
    total = (
        _non_negative_integer(updates["total_copies"], "total_copies")
        if "total_copies" in updates
        else None
    )

    with _connect(write=True) as connection:
        book = connection.execute(
            "SELECT * FROM books WHERE book_id = ?", (identifier,)
        ).fetchone()
        if book is None:
            raise ValueError(f"Unknown book ID: {identifier}.")
        active_loans = connection.execute(
            "SELECT count(*) FROM loans WHERE book_id = ? AND status = 'active'",
            (identifier,),
        ).fetchone()[0]
        new_total = book["total_copies"] if total is None else total
        if new_total < active_loans:
            raise ValueError("total_copies cannot be less than active loans.")
        new_available = new_total - active_loans
        connection.execute(
            """
            UPDATE books
            SET title = ?, author = ?, total_copies = ?, available_copies = ?
            WHERE book_id = ?
            """,
            (
                book["title"] if title is None else title,
                book["author"] if author is None else author,
                new_total,
                new_available,
                identifier,
            ),
        )
        return _required_record(
            connection.execute(
                "SELECT * FROM books WHERE book_id = ?", (identifier,)
            ).fetchone(),
            "book",
        )


def add_student(student_data: Mapping[str, Any]) -> dict[str, Any]:
    student_id = _required_text(student_data.get("student_id"), "student_id")
    name = _required_text(student_data.get("name"), "name")
    email_value = student_data.get("email")
    email = (
        None
        if email_value is None
        else _required_text(email_value, "email")
    )
    interests_value = student_data.get("interests", [])
    if isinstance(interests_value, (str, bytes)) or not isinstance(
        interests_value, Sequence
    ):
        raise ValueError("interests must be a sequence of strings.")
    interests = [
        _required_text(interest, "interest") for interest in interests_value
    ]
    registered = student_data.get("registered")
    eligible = student_data.get("eligible")
    if not isinstance(registered, bool) or not isinstance(eligible, bool):
        raise ValueError("registered and eligible must be booleans.")

    with _connect(write=True) as connection:
        connection.execute(
            """
            INSERT INTO students (
                student_id, name, email, interests, registered, eligible, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                name,
                email,
                json.dumps(interests),
                int(registered),
                int(eligible),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        return _required_record(
            connection.execute(
                "SELECT * FROM students WHERE student_id = ?", (student_id,)
            ).fetchone(),
            "student",
        )


def get_student(student_id: str) -> dict[str, Any] | None:
    identifier = _required_text(student_id, "student_id")
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM students WHERE student_id = ?", (identifier,)
        ).fetchone()
    return _record(row)


def get_students() -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM students ORDER BY lower(name), student_id"
        ).fetchall()
    return _records(rows)


def get_loan(loan_id: str) -> dict[str, Any] | None:
    identifier = _required_text(loan_id, "loan_id")
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM loans WHERE loan_id = ?", (identifier,)
        ).fetchone()
    return _record(row)


def get_loans() -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM loans ORDER BY issued_at DESC, loan_id"
        ).fetchall()
    return _records(rows)


def issue_book(student_id: str, book_id: str) -> dict[str, Any]:
    student_identifier = _required_text(student_id, "student_id")
    book_identifier = _required_text(book_id, "book_id")
    now = datetime.now(timezone.utc)
    today = now.date()
    loan_id = f"L-{uuid4().hex}"

    with _connect(write=True) as connection:
        student = connection.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_identifier,)
        ).fetchone()
        if student is None:
            raise ValueError(f"Unknown student ID: {student_identifier}.")
        book = connection.execute(
            "SELECT * FROM books WHERE book_id = ?", (book_identifier,)
        ).fetchone()
        if book is None:
            raise ValueError(f"Unknown book ID: {book_identifier}.")
        overdue = connection.execute(
            """
            SELECT 1 FROM loans
            WHERE student_id = ? AND status = 'active' AND date(due_at) < date('now')
            LIMIT 1
            """,
            (student_identifier,),
        ).fetchone() is not None
        if overdue:
            raise ValueError("Student has an overdue loan.")
        if book["available_copies"] == 0:
            raise ValueError("No copies of this book are currently available.")
        if not check_eligibility(
            bool(student["registered"]),
            bool(student["eligible"]),
            book["available_copies"] > 0,
            overdue,
        ):
            raise ValueError("Student is not eligible to issue a book.")
        active = connection.execute(
            """
            SELECT 1 FROM loans
            WHERE student_id = ? AND book_id = ? AND status = 'active'
            """,
            (student_identifier, book_identifier),
        ).fetchone()
        if active is not None:
            raise ValueError("Student already has an active loan for this book.")

        reservation = connection.execute(
            """
            SELECT reservation_id, student_id FROM reservations
            WHERE book_id = ? ORDER BY reservation_id LIMIT 1
            """,
            (book_identifier,),
        ).fetchone()
        if reservation is not None and reservation["student_id"] != student_identifier:
            raise ValueError("This book is reserved for an earlier student in the queue.")

        due_at = (today + timedelta(days=14)).isoformat()
        connection.execute(
            """
            INSERT INTO loans (
                loan_id, student_id, book_id, issued_at, due_at, status
            ) VALUES (?, ?, ?, ?, ?, 'active')
            """,
            (
                loan_id,
                student_identifier,
                book_identifier,
                now.isoformat(timespec="seconds"),
                due_at,
            ),
        )
        updated = connection.execute(
            """
            UPDATE books SET available_copies = available_copies - 1
            WHERE book_id = ? AND available_copies > 0
            """,
            (book_identifier,),
        )
        if updated.rowcount != 1:
            raise RuntimeError("Book availability changed during an issue transaction.")
        if reservation is not None:
            connection.execute(
                "DELETE FROM reservations WHERE reservation_id = ?",
                (reservation["reservation_id"],),
            )
        return _required_record(
            connection.execute(
                "SELECT * FROM loans WHERE loan_id = ?", (loan_id,)
            ).fetchone(),
            "loan",
        )


def return_book(student_id: str, book_id: str) -> dict[str, Any]:
    student_identifier = _required_text(student_id, "student_id")
    book_identifier = _required_text(book_id, "book_id")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _connect(write=True) as connection:
        loan = connection.execute(
            """
            SELECT * FROM loans
            WHERE student_id = ? AND book_id = ? AND status = 'active'
            """,
            (student_identifier, book_identifier),
        ).fetchone()
        if loan is None:
            raise ValueError("No active loan matches this student and book.")
        connection.execute(
            """
            UPDATE loans SET status = 'returned', returned_at = ?
            WHERE loan_id = ? AND status = 'active'
            """,
            (now, loan["loan_id"]),
        )
        updated = connection.execute(
            """
            UPDATE books SET available_copies = available_copies + 1
            WHERE book_id = ? AND available_copies < total_copies
            """,
            (book_identifier,),
        )
        if updated.rowcount != 1:
            raise RuntimeError("Book stock is inconsistent with its active loans.")
        return _required_record(
            connection.execute(
                "SELECT * FROM loans WHERE loan_id = ?", (loan["loan_id"],)
            ).fetchone(),
            "loan",
        )


def reserve_book(student_id: str, book_id: str) -> dict[str, Any]:
    student_identifier = _required_text(student_id, "student_id")
    book_identifier = _required_text(book_id, "book_id")
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _connect(write=True) as connection:
        student = connection.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_identifier,)
        ).fetchone()
        if student is None:
            raise ValueError(f"Unknown student ID: {student_identifier}.")
        book = connection.execute(
            "SELECT * FROM books WHERE book_id = ?", (book_identifier,)
        ).fetchone()
        if book is None:
            raise ValueError(f"Unknown book ID: {book_identifier}.")
        if book["total_copies"] == 0:
            raise ValueError("A book with no copies cannot be reserved.")
        if not student["registered"] or not student["eligible"]:
            raise ValueError("Student is not eligible to reserve a book.")
        overdue = connection.execute(
            """
            SELECT 1 FROM loans
            WHERE student_id = ? AND status = 'active' AND date(due_at) < date('now')
            LIMIT 1
            """,
            (student_identifier,),
        ).fetchone()
        if overdue is not None:
            raise ValueError("Student has an overdue loan.")
        if book["available_copies"] > 0:
            raise ValueError("This book is available; issue it instead of reserving it.")
        active = connection.execute(
            """
            SELECT 1 FROM loans
            WHERE student_id = ? AND book_id = ? AND status = 'active'
            """,
            (student_identifier, book_identifier),
        ).fetchone()
        if active is not None:
            raise ValueError("Student already has an active loan for this book.")
        existing = connection.execute(
            """
            SELECT 1 FROM reservations
            WHERE student_id = ? AND book_id = ?
            """,
            (student_identifier, book_identifier),
        ).fetchone()
        if existing is not None:
            raise ValueError("Student already has a reservation for this book.")

        cursor = connection.execute(
            """
            INSERT INTO reservations (student_id, book_id, created_at)
            VALUES (?, ?, ?)
            """,
            (student_identifier, book_identifier, created_at),
        )
        position = connection.execute(
            "SELECT count(*) FROM reservations WHERE book_id = ?",
            (book_identifier,),
        ).fetchone()[0]
        reservation_id = cursor.lastrowid
        if reservation_id is None:
            raise RuntimeError("Reservation was saved without an ID.")
        return {
            "reservation_id": reservation_id,
            "student_id": student_identifier,
            "book_id": book_identifier,
            "queue_position": position,
            "created_at": created_at,
        }
