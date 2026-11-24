"""Focused tests for the Streamlit app's integrated module interfaces."""

import os
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import database


class AppIntegrationTests(unittest.TestCase):
    def test_logic_and_database_workflows_render_saved_output(self):
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "app.sqlite"
            with (
                patch.object(database, "DB_PATH", database_path),
                patch.dict(os.environ, {"LIBRARY_DB_PATH": str(database_path)}),
            ):
                app = AppTest.from_file(
                    str(Path(__file__).with_name("app.py"))
                ).run(timeout=60)
                app.sidebar.radio[0].set_value("Mathematical Logic")
                app.run(timeout=60)
                app.button[0].click()
                app.run(timeout=60)

                self.assertFalse(app.exception, app.exception)
                self.assertEqual(len(app.dataframe), 1)
                table = app.dataframe[0].value
                self.assertEqual(list(table.columns), ["R", "E", "A", "O", "I"])
                self.assertEqual(len(table), 16)

                app.sidebar.radio[0].set_value("Student and Book Management")
                app.run(timeout=60)
                self.assertEqual(len(app.dataframe[0].value), 0)
                app.text_input[0].set_value("Sample Student")
                app.text_input[1].set_value("sample@example.edu")
                app.text_input[2].set_value("Computing, Mathematics")
                app.button[0].click()
                app.run(timeout=60)
                self.assertFalse(app.exception, app.exception)
                self.assertEqual(
                    app.dataframe[0].value.iloc[0]["name"], "Sample Student"
                )
                self.assertTrue(
                    any("Student added:" in item.value for item in app.success)
                )

                app.text_input[3].set_value("Discrete Structures")
                app.text_input[4].set_value("A. Author")
                app.text_input[5].set_value("Computing")
                app.button[1].click()
                app.run(timeout=60)
                self.assertFalse(app.exception, app.exception)
                self.assertEqual(
                    app.dataframe[1].value.iloc[0]["title"],
                    "Discrete Structures",
                )
                self.assertTrue(
                    any("Book added:" in item.value for item in app.success)
                )

                student_id = app.dataframe[0].value.iloc[0]["student_id"]
                book_id = app.dataframe[1].value.iloc[0]["book_id"]
                app.sidebar.radio[0].set_value("Issue, Return, and Reservation")
                app.run(timeout=60)
                app.text_input[0].set_value(student_id)
                app.text_input[1].set_value(book_id)
                app.button[0].click()
                app.run(timeout=60)
                self.assertFalse(app.exception, app.exception)

                with closing(sqlite3.connect(database_path)) as connection:
                    loan_id = connection.execute(
                        "SELECT loan_id FROM loans WHERE status = 'active'"
                    ).fetchone()[0]
                app.selectbox[0].set_value("Return book")
                app.text_input[2].set_value(loan_id)
                app.button[0].click()
                app.run(timeout=60)
                self.assertFalse(app.exception, app.exception)
                with closing(sqlite3.connect(database_path)) as connection:
                    status = connection.execute(
                        "SELECT status FROM loans WHERE loan_id = ?", (loan_id,)
                    ).fetchone()[0]
                self.assertEqual(status, "returned")


if __name__ == "__main__":
    unittest.main()
