"""Focused tests for the Streamlit app's integrated module interfaces."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppIntegrationTests(unittest.TestCase):
    def test_logic_page_generates_eligibility_truth_table(self):
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(
            timeout=20
        )
        app.sidebar.radio[0].set_value("Mathematical Logic")
        app.run(timeout=20)
        app.button[0].click()
        app.run(timeout=20)

        self.assertFalse(app.exception, app.exception)
        self.assertEqual(len(app.dataframe), 1)
        table = app.dataframe[0].value
        self.assertEqual(list(table.columns), ["R", "E", "A", "O", "I"])
        self.assertEqual(len(table), 16)


if __name__ == "__main__":
    unittest.main()
