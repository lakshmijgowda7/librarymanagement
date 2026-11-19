# Intelligent Library Management System

College PBL project for Discrete Structures for Computing (UE25CS2301).

## Architecture and ownership

- `app.py` owns Streamlit navigation, presentation, input validation at the UI boundary, and composition only.
- `database.py` and `library_operations.py` belong to Member 1. Keep schema, database access, and circulation rules there.
- Member 2 owns `logic_module.py`, `counting_module.py`, `relations_module.py`, `recurrence_module.py`, and `graph_module.py`.
- `ai_recommendation.py` owns the explainable category-scoring demo.
- `optimization_module.py` owns the Unit 6 objective, exact derivatives, constrained allocation example, and gradient-descent experiment.

The app starts when the other members' files are absent and reports the missing integrations. Do not copy their business logic into `app.py`; agree on these interfaces and adapt the owning module if its implementation uses different internal names.

## Integration contract

Every function below should return values (not print output). Database collection functions return `pandas.DataFrame`. `init_db()` creates/opens the shared SQLite schema and can raise an exception if initialization fails.

```python
# database.py — Member 1
init_db() -> None
get_students() -> pandas.DataFrame  # columns include student_id and name
get_books() -> pandas.DataFrame     # book_id, title, category, available_copies
get_loans() -> pandas.DataFrame    # loan_id, student_id, book_id, status

# library_operations.py — Member 1; each returns a useful value or None
add_student(name: str, email: str, interests: list[str])
add_book(title: str, author: str, category: str, total_copies: int)
issue_book(student_id: str, book_id: str)
return_book(loan_id: str)
reserve_book(student_id: str, book_id: str)

# Member 2 — functions available on `feature/math-modules`
logic_module.generate_truth_table(expression: str) -> pandas.DataFrame
counting_module.calculate_combinations(n: int, r: int) -> int
counting_module.calculate_permutations(n: int, r: int) -> int
counting_module.binomial_expansion(a, b, n) -> dict
counting_module.pigeonhole_analysis(objects, boxes, assignment=None) -> dict
relations_module.zero_one_matrix(rows, columns, relation) -> dict
relations_module.analyze_function(domain, codomain, mapping) -> dict
relations_module.equivalence_classes(domain, relation) -> list[set]
relations_module.divisibility_hasse_diagram(number) -> dict
recurrence_module.generate_first_order_sequence(initial_value, changes, periods) -> list
recurrence_module.solve_first_order_recurrence(initial_value, multiplier, constant, periods) -> list
recurrence_module.generate_second_order_sequence(initial_0, initial_1, periods, coefficient_1=1, coefficient_2=1) -> list
graph_module.build_library_graph(books, students, records) -> networkx.DiGraph
graph_module.analyze_graph(graph) -> dict
```

The existing remote branch did not contain `logic_module.py` or Member 1's database/operation modules when checked. Their navigation pages display integration status until those modules are provided. The Streamlit calls for the four present Member 2 modules match their current public interfaces. Relations form inputs are passed as string labels. Graph demo inputs are converted to `books`, `students`, and borrowing `records` as expected by `build_library_graph`.

## Recommendation model

`recommend_books()` accepts a catalog with `book_id`, `title`, `category`, and either `available_copies` or `available`. Interest, borrowing-history, and category-preference inputs may be category lists or non-negative category weights. Each signal is normalized by its largest input value to `[0, 1]`.

```text
score(book) = 100 * (0.45 * interest_match
                   + 0.35 * borrowing_history_match
                   + 0.20 * category_preference_match)
```

This is a transparent rule-based baseline, not a trained model. Availability does not change relevance; it is displayed explicitly and available titles sort first. The sample catalog is illustrative only.

## Unit 6 mathematical notes

The optimization demo uses a synthetic loss over two continuous recommendation-calibration controls (illustrative, not learned from data):

```text
f(x,y) = (x-2)^2 + 2(y-1)^2 + 0.5(x-2)(y-1)
df/dx = 2(x-2) + 0.5(y-1)
df/dy = 4(y-1) + 0.5(x-2)
H = [[2, 0.5], [0.5, 4]]
```

The Hessian eigenvalues are positive, so `f` is strictly convex; its unique stationary/global minimizing point is `(2,1)` with value `0`. The concave objective `g=-f` has a unique global maximum at the same point with value `0`. The demo verifies the minimum from the same objective used by gradient descent. The displayed convex feasible set is `{(x,y): x>=0, y>=0, x+y<=5}`.

For Lagrange multipliers, the example minimizes `1/2[(x-4)^2+(y-3)^2]` subject to `x+y=5`. With `L=f+lambda(x+y-5)`, stationarity gives `(x,y,lambda)=(3,2,1)`. It illustrates continuous allocation of staff hours; choosing integer book copies is a separate discrete optimization problem.

The gradient-descent page compares learning rates `0.05`, `0.2`, and `0.4` from the same initial point and shows the actual loss histories and stopping status. This objective has largest Hessian eigenvalue approximately `4.118`, so these rates are below the quadratic gradient-descent stability bound `2/lambda_max`.

## Windows setup and run

From the project directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m py_compile app.py ai_recommendation.py optimization_module.py counting_module.py relations_module.py recurrence_module.py graph_module.py
python -m streamlit run app.py
```

If PowerShell blocks activation, use the venv interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m py_compile app.py ai_recommendation.py optimization_module.py counting_module.py relations_module.py recurrence_module.py graph_module.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Team Git workflow and integration checks

Use the team's shared repository and do not overwrite or recreate a teammate's module. The team math modules are based on `feature/math-modules`; create Megha's branch from that branch:

```powershell
git status
git fetch origin
git switch -c megha --track origin/feature/math-modules
git add app.py ai_recommendation.py optimization_module.py requirements.txt README.md
git commit -m "Add AI recommendations and optimization dashboard"
git push -u origin megha
```

If `megha` already exists remotely, use `git switch --track origin/megha` instead. Ask Member 1 to confirm the database/operation signatures before merging. Then install from a fresh virtual environment, test each operation and each mathematical page, and open/review a pull request before merging to `main`.

For a quick numerical smoke test after installing dependencies:

```powershell
python -c "import numpy as np; from optimization_module import *; assert np.allclose(quadratic_gradient([2,1]), [0,0]); assert np.allclose(quadratic_hessian(), [[2,.5],[.5,4]]); assert concave_maximum_summary()['maximum_value'] == 0; assert allocation_lagrange_solution()['lambda'] == 1; report, histories = compare_learning_rates(); print(report.to_string(index=False)); assert report['converged'].all(); from ai_recommendation import sample_recommendations; print(sample_recommendations()[['title','recommendation_score','availability_status']].to_string(index=False))"
```
