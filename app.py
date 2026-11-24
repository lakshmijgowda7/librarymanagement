"""Streamlit entry point. See README.md for the Member 1/2 integration contract."""

from __future__ import annotations

import importlib
import sqlite3
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

import ai_recommendation
import optimization_module


st.set_page_config(page_title="Intelligent Library Management System", page_icon="📚", layout="wide")

NAV_ITEMS = [
    "Dashboard",
    "Student and Book Management",
    "Issue, Return, and Reservation",
    "Mathematical Logic",
    "Counting and Combinatorics",
    "Relations and Functions",
    "Recurrence Relations",
    "Graph Theory",
    "AI Recommendations",
    "Optimization",
]

# These are the proposed agreed interfaces. A missing module is intentionally
# shown as an integration status, rather than replaced with a second implementation.
INTEGRATIONS = {
    "database": ("init_db", "get_students", "get_books", "get_loans"),
    "library_operations": ("add_student", "add_book", "issue_book", "return_book", "reserve_book"),
    "logic_module": ("generate_truth_table",),
    "counting_module": ("calculate_combinations", "calculate_permutations", "pigeonhole_analysis"),
    "relations_module": ("zero_one_matrix", "analyze_function", "equivalence_classes"),
    "recurrence_module": ("generate_first_order_sequence", "generate_second_order_sequence"),
    "graph_module": ("build_library_graph",),
}


@st.cache_resource
def initialize_database() -> tuple[Any | None, str | None]:
    try:
        database = importlib.import_module("database")
    except ModuleNotFoundError as error:
        if error.name == "database":
            return None, "database.py has not been added yet."
        return None, f"database.py could not load dependency '{error.name}'."
    try:
        database.init_db()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
        return database, f"database.init_db() failed: {error}"
    return database, None


def optional_module(name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(name), None
    except ModuleNotFoundError as error:
        if error.name == name:
            return None, f"{name}.py is not present."
        return None, f"{name}.py could not load dependency '{error.name}'."
    except ImportError as error:
        return None, f"{name}.py could not be imported: {error}"


def dataframe_from(database: Any, function_name: str) -> pd.DataFrame:
    function = getattr(database, function_name)
    value = function()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"database.{function_name}() must return a pandas DataFrame.")
    return value


def render_integration_status() -> None:
    st.subheader("Module integration status")
    rows = []
    for module_name, functions in INTEGRATIONS.items():
        module, error = optional_module(module_name)
        if module is None:
            status = error
        else:
            missing = [name for name in functions if not callable(getattr(module, name, None))]
            status = "Ready" if not missing else f"Missing functions: {', '.join(missing)}"
        rows.append({"module": f"{module_name}.py", "status": status})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("The exact signatures and return types are documented in README.md.")


def render_dashboard(database: Any | None, database_error: str | None) -> None:
    st.title("Intelligent Library Management System")
    st.write("Discrete Structures for Computing (UE25CS2301) · Team PBL")
    if database is None or database_error:
        st.info(database_error or "Library database is not integrated; showing illustrative sample data.")
        books = ai_recommendation.SAMPLE_BOOKS
        columns = st.columns(3)
        columns[0].metric("Sample titles", len(books))
        columns[1].metric("Available copies", int(books["available_copies"].sum()))
        columns[2].metric("Database", "Not integrated")
    else:
        try:
            students = dataframe_from(database, "get_students")
            books = dataframe_from(database, "get_books")
            loans = dataframe_from(database, "get_loans")
            columns = st.columns(3)
            columns[0].metric("Students", len(students))
            columns[1].metric("Books", len(books))
            columns[2].metric("Loans", len(loans))
            st.subheader("Book catalog")
            st.dataframe(books, hide_index=True, width="stretch")
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
            st.error(f"Could not load dashboard data: {error}")
    render_integration_status()


def render_management(database: Any | None) -> None:
    st.title("Student and Book Management")
    if database is None:
        st.info("Add Member 1's database.py and library_operations.py to enable catalog management.")
        return
    tabs = st.tabs(["Students", "Books"])
    for tab, getter in zip(tabs, ("get_students", "get_books")):
        with tab:
            try:
                st.dataframe(dataframe_from(database, getter), hide_index=True, width="stretch")
            except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
                st.error(f"Could not load {getter}(): {error}")
    operations, error = optional_module("library_operations")
    if operations is None:
        st.info(error or "Add library_operations.py to enable student and book creation.")
        return

    st.subheader("Add a student or book")
    student_tab, book_tab = st.tabs(["Add student", "Add book"])
    with student_tab:
        with st.form("add_student_form"):
            name = st.text_input("Student name")
            email = st.text_input("Email")
            interests_text = st.text_input("Interests (comma-separated)")
            submitted = st.form_submit_button("Add student")
        if submitted:
            interests = [value.strip() for value in interests_text.split(",") if value.strip()]
            if not name.strip() or not email.strip():
                st.error("Student name and email are required.")
            else:
                try:
                    result = operations.add_student(name.strip(), email.strip(), interests)
                    st.success(str(result) if result is not None else "Student added.")
                except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
                    st.error(f"Could not add student: {error}")
    with book_tab:
        with st.form("add_book_form"):
            title = st.text_input("Book title")
            author = st.text_input("Author")
            category = st.text_input("Category")
            total_copies = st.number_input("Total copies", min_value=1, value=1, step=1)
            submitted = st.form_submit_button("Add book")
        if submitted:
            if not title.strip() or not author.strip() or not category.strip():
                st.error("Title, author, and category are required.")
            else:
                try:
                    result = operations.add_book(
                        title.strip(), author.strip(), category.strip(), int(total_copies)
                    )
                    st.success(str(result) if result is not None else "Book added.")
                except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
                    st.error(f"Could not add book: {error}")


def render_circulation(database: Any | None) -> None:
    st.title("Issue, Return, and Reservation")
    operations, error = optional_module("library_operations")
    if database is None or operations is None:
        st.info(error or "Add database.py and library_operations.py using the contract in README.md.")
        return
    st.caption("Actions use Member 1's library_operations functions; no circulation logic is duplicated here.")
    with st.form("circulation_form"):
        action = st.selectbox("Action", ["Issue book", "Return book", "Reserve book"])
        student_id = st.text_input("Student ID")
        book_id = st.text_input("Book ID")
        loan_id = st.text_input("Loan ID (for returns)")
        submitted = st.form_submit_button("Run action")
    if submitted:
        function_name = {
            "Issue book": "issue_book",
            "Return book": "return_book",
            "Reserve book": "reserve_book",
        }[action]
        args = (loan_id,) if action == "Return book" else (student_id, book_id)
        try:
            result = getattr(operations, function_name)(*args)
            st.success(str(result) if result is not None else f"{action} completed.")
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
            st.error(f"{action} failed: {error}")


def render_member2_page(title: str, module_name: str) -> None:
    st.title(title)
    module, error = optional_module(module_name)
    if module is None:
        st.info(f"{error} Member 2's implementation will appear here after it is added.")
        st.caption("Do not copy the other member's module into app.py; use the agreed interface in README.md.")
        return
    try:
        if module_name == "logic_module":
            st.write("Issue eligibility rule: I = R AND E AND A AND NOT O")
            if st.button("Generate truth table"):
                st.dataframe(
                    pd.DataFrame(module.generate_truth_table()),
                    hide_index=True,
                    width="stretch",
                )
        elif module_name == "counting_module":
            operation = st.selectbox(
                "Counting operation",
                ["Combinations", "Permutations", "Binomial expansion", "Pigeonhole principle"],
            )
            n = st.number_input("n / number of objects", min_value=0, value=5, step=1)
            r = st.number_input("r / number of boxes", min_value=0, value=2, step=1)
            if operation == "Binomial expansion":
                a = st.number_input("a", value=1.0)
                b = st.number_input("b", value=2.0)
            if st.button("Calculate"):
                if operation == "Combinations":
                    st.write(module.calculate_combinations(int(n), int(r)))
                elif operation == "Permutations":
                    st.write(module.calculate_permutations(int(n), int(r)))
                elif operation == "Binomial expansion":
                    st.json(module.binomial_expansion(a, b, int(n)))
                else:
                    st.json(module.pigeonhole_analysis(int(n), int(r)))
        elif module_name == "relations_module":
            operation = st.selectbox(
                "Relations operation",
                ["Zero-one relation matrix", "Function properties", "Equivalence classes", "Divisibility Hasse diagram"],
            )
            domain_text = st.text_input("Domain (comma-separated)", "1, 2, 3")
            pairs_text = st.text_input("Ordered pairs (e.g. 1:1, 2:2, 3:3)", "1:1, 2:2, 3:3")
            if operation == "Function properties":
                codomain_text = st.text_input("Codomain (comma-separated)", "a, b, c")
            if operation == "Divisibility Hasse diagram":
                hasse_number = st.number_input("Positive integer", min_value=1, value=12, step=1)
            if st.button("Analyze"):
                domain = [value.strip() for value in domain_text.split(",") if value.strip()]
                pairs = [
                    tuple(part.strip() for part in item.split(":", 1))
                    for item in pairs_text.split(",")
                    if ":" in item
                ]
                if operation == "Zero-one relation matrix":
                    st.json(module.zero_one_matrix(domain, domain, pairs))
                elif operation == "Function properties":
                    codomain = [value.strip() for value in codomain_text.split(",") if value.strip()]
                    st.json(module.analyze_function(domain, codomain, pairs))
                elif operation == "Equivalence classes":
                    st.write(module.equivalence_classes(domain, pairs))
                else:
                    st.json(module.divisibility_hasse_diagram(int(hasse_number)))
        elif module_name == "recurrence_module":
            recurrence_type = st.selectbox(
                "Recurrence type",
                ["Additive first order", "First-order linear", "Second-order linear"],
            )
            periods = st.number_input("Transitions / periods", min_value=0, value=10, step=1)
            if recurrence_type == "Additive first order":
                initial = st.number_input("Initial value", value=0.0)
                changes_text = st.text_input("Per-period changes (comma-separated)", "1")
            elif recurrence_type == "First-order linear":
                initial = st.number_input("Initial value", value=1.0)
                multiplier = st.number_input("Multiplier", value=2.0)
                constant = st.number_input("Constant", value=1.0)
            else:
                initial_0 = st.number_input("x[0]", value=0.0)
                initial_1 = st.number_input("x[1]", value=1.0)
                coefficient_1 = st.number_input("Coefficient of x[n-1]", value=1.0)
                coefficient_2 = st.number_input("Coefficient of x[n-2]", value=1.0)
            if st.button("Generate sequence"):
                if recurrence_type == "Additive first order":
                    change_values = [float(value.strip()) for value in changes_text.split(",") if value.strip()]
                    changes = change_values[0] if len(change_values) == 1 else change_values
                    st.write(module.generate_first_order_sequence(initial, changes, int(periods)))
                elif recurrence_type == "First-order linear":
                    st.write(module.solve_first_order_recurrence(initial, multiplier, constant, int(periods)))
                else:
                    st.write(module.generate_second_order_sequence(
                        initial_0, initial_1, int(periods), coefficient_1, coefficient_2
                    ))
        elif module_name == "graph_module":
            students_text = st.text_input("Student IDs (comma-separated)", "S1, S2")
            books_text = st.text_input("Book IDs (comma-separated)", "B1, B2")
            edges_text = st.text_input("Borrowing records (e.g. S1:B1, S2:B1)", "S1:B1, S2:B2")
            if st.button("Build graph"):
                students = [value.strip() for value in students_text.split(",") if value.strip()]
                book_ids = [value.strip() for value in books_text.split(",") if value.strip()]
                books = [{"id": book_id} for book_id in book_ids]
                records = [
                    tuple(part.strip() for part in item.split(":", 1))
                    for item in edges_text.split(",")
                    if ":" in item
                ]
                graph = module.build_library_graph(books, students, records)
                if not isinstance(graph, nx.Graph):
                    raise TypeError("graph_module.build_library_graph() must return a networkx.Graph.")
                figure, axis = plt.subplots()
                nx.draw_networkx(graph, ax=axis, with_labels=True, node_color="#b7d7f0")
                axis.axis("off")
                st.pyplot(figure)
                plt.close(figure)
                st.json(module.analyze_graph(graph))
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
        st.error(f"{title} integration failed: {error}")


def render_recommendations(database: Any | None) -> None:
    st.title("AI Recommendations")
    st.write("A transparent, rule-based scoring demonstration; this is not a trained or evaluated AI model.")
    st.latex(r"\mathrm{score}(b)=100\left(0.45I_b+0.35H_b+0.20P_b\right)")
    st.caption(
        "I, H, and P are category interest, borrowing-history, and preference signals, "
        "each normalized to [0, 1]. Availability is reported separately."
    )
    books = ai_recommendation.SAMPLE_BOOKS
    if database is not None:
        try:
            candidate = dataframe_from(database, "get_books")
            if {"book_id", "title", "category"}.issubset(candidate.columns) and (
                "available_copies" in candidate.columns or "available" in candidate.columns
            ):
                books = candidate
            else:
                st.warning("Database book columns do not match the recommendation input contract; using sample catalog.")
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
            st.warning(f"Could not read database catalog ({error}); using sample catalog.")
    interests = st.text_input("Student interests", "Computing, Mathematics")
    history = st.text_input("Borrowing history categories", "Computing, Computing, Mathematics")
    preferences = st.text_input("Preferred categories", "Mathematics, Computing")
    try:
        recommendations = ai_recommendation.recommend_books(
            books,
            interests=[value.strip() for value in interests.split(",") if value.strip()],
            borrowing_history=[value.strip() for value in history.split(",") if value.strip()],
            category_preferences=[value.strip() for value in preferences.split(",") if value.strip()],
        )
        st.dataframe(recommendations, hide_index=True, width="stretch")
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
        st.error(f"Could not calculate recommendations: {error}")


def render_optimization() -> None:
    st.title("Mathematics for AI and Optimization")
    st.markdown(f"**Objective:** `{optimization_module.OPTIMIZATION_FORMULA}`")
    st.latex(r"\nabla f=\begin{bmatrix}2(x-2)+0.5(y-1)\\4(y-1)+0.5(x-2)\end{bmatrix}")
    st.latex(r"\nabla^2 f=\begin{bmatrix}2&0.5\\0.5&4\end{bmatrix}")
    summary = optimization_module.convexity_summary()
    maximum = optimization_module.concave_maximum_summary()
    columns = st.columns(3)
    columns[0].metric("Global minimum", "(2, 1)")
    columns[1].metric("Minimum objective", f"{summary['minimum_value']:.3g}")
    columns[2].metric("Strictly convex", "Yes" if summary["strictly_convex"] else "No")
    st.write("Treat x and y as two continuous recommendation-calibration controls in a synthetic loss. The positive Hessian eigenvalues make this loss strictly convex, with a unique global minimum at (2, 1).")
    st.write(f"For the concave objective g(x, y) = -f(x, y), the unique global maximum is {tuple(maximum['global_maximum'])} with value {maximum['maximum_value']:.3g}; its Hessian is negative definite.")
    st.subheader("Lagrange multipliers: staff-hour allocation")
    allocation = optimization_module.allocation_lagrange_solution()
    st.write("Minimize 1/2[(x - 4)^2 + (y - 3)^2], subject to x + y = 5 and x, y >= 0.")
    st.latex(r"\mathcal{L}=\frac12[(x-4)^2+(y-3)^2]+\lambda(x+y-5)")
    st.json(allocation)
    st.caption("The solution is x=3, y=2, lambda=1. This continuous allocation demo is distinct from selecting discrete book copies.")
    st.subheader("Gradient descent")
    learning_rates = st.multiselect("Learning rates to compare", [0.05, 0.2, 0.4], default=[0.05, 0.2, 0.4])
    if learning_rates:
        comparison, histories = optimization_module.compare_learning_rates(learning_rates)
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.pyplot(optimization_module.convergence_figure(histories))
        plt.close("all")
    st.subheader("Convex resource-allocation set")
    st.write("For non-negative desk hours, { (x, y) : x >= 0, y >= 0, x + y <= 5 } is a convex simplex: every line segment between feasible allocations stays feasible.")


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", NAV_ITEMS)
    database, database_error = initialize_database()
    if database_error:
        st.sidebar.warning(database_error)

    if page == "Dashboard":
        render_dashboard(database, database_error)
    elif page == "Student and Book Management":
        render_management(database)
    elif page == "Issue, Return, and Reservation":
        render_circulation(database)
    elif page == "Mathematical Logic":
        render_member2_page(page, "logic_module")
    elif page == "Counting and Combinatorics":
        render_member2_page(page, "counting_module")
    elif page == "Relations and Functions":
        render_member2_page(page, "relations_module")
    elif page == "Recurrence Relations":
        render_member2_page(page, "recurrence_module")
    elif page == "Graph Theory":
        render_member2_page(page, "graph_module")
    elif page == "AI Recommendations":
        render_recommendations(database)
    elif page == "Optimization":
        render_optimization()


if __name__ == "__main__":
    main()
