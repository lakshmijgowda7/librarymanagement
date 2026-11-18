"""Counting and combinatorics helpers for library-management examples.

Run ``python counting_module.py`` for a small, manually checkable demonstration.
All counting functions use exact integer arithmetic.
"""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from math import comb, isclose, perm
from typing import Any


def _non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def sum_rule(*counts: int) -> int:
    """Return the number of choices across mutually exclusive alternatives."""
    for count in counts:
        _non_negative_integer(count, "Each count")
    return sum(counts)


def product_rule(*counts: int) -> int:
    """Return the number of outcomes from successive independent choices."""
    for count in counts:
        _non_negative_integer(count, "Each count")
    result = 1
    for count in counts:
        result *= count
    return result


def calculate_combinations(n: int, r: int) -> int:
    """Return C(n, r), the number of unordered selections of size ``r``."""
    _non_negative_integer(n, "n")
    _non_negative_integer(r, "r")
    if r > n:
        raise ValueError("r cannot be greater than n.")
    return comb(n, r)


def calculate_permutations(n: int, r: int) -> int:
    """Return P(n, r), the number of ordered selections of size ``r``."""
    _non_negative_integer(n, "n")
    _non_negative_integer(r, "r")
    if r > n:
        raise ValueError("r cannot be greater than n.")
    return perm(n, r)


def binomial_expansion(a: int | float, b: int | float, n: int) -> dict[str, Any]:
    """Expand ``(a + b)**n`` using the binomial theorem.

    The result contains ``terms`` (coefficient and powers), ``lhs``, and ``rhs``.
    Each term is ``C(n, k) * a**(n-k) * b**k``.
    """
    _non_negative_integer(n, "n")
    terms = [
        {
            "coefficient": comb(n, k),
            "a_power": n - k,
            "b_power": k,
            "value": comb(n, k) * a ** (n - k) * b**k,
        }
        for k in range(n + 1)
    ]
    lhs = (a + b) ** n
    rhs = sum(term["value"] for term in terms)
    verified = (
        isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)
        if isinstance(lhs, float) or isinstance(rhs, float)
        else lhs == rhs
    )
    return {"terms": terms, "lhs": lhs, "rhs": rhs, "verified": verified}


def verify_sum_induction_example(n: int) -> dict[str, int | bool]:
    """Check the induction identity ``1 + ... + n = n(n+1)/2``.

    The function checks the statement for one input. The induction proof is:
    the identity holds at 0; assuming it at n, adding n+1 gives
    n(n+1)/2 + (n+1) = (n+1)(n+2)/2.
    """
    _non_negative_integer(n, "n")
    total = sum(range(1, n + 1))
    formula = n * (n + 1) // 2
    return {"n": n, "sum": total, "formula": formula, "verified": total == formula}


def well_ordering_demo(values: Iterable[int]) -> int:
    """Return the least positive integer in a non-empty candidate collection."""
    candidates = list(values)
    if not candidates:
        raise ValueError("The candidate collection must not be empty.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in candidates):
        raise ValueError("Every candidate must be a positive integer.")
    return min(candidates)


def pigeonhole_analysis(
    objects: int | Sequence[Hashable],
    boxes: int | Sequence[Hashable],
    assignment: Mapping[Hashable, Hashable] | None = None,
) -> dict[str, Any]:
    """Apply the pigeonhole bound and optionally summarize an allocation.

    ``objects`` and ``boxes`` may be counts or collections of unique labels.
    If both are collections and ``assignment`` is omitted, objects are assigned
    cyclically to boxes for a reproducible illustration. With counts only, the
    result reports the guaranteed minimum but does not invent an allocation.
    """
    if isinstance(objects, bool):
        raise ValueError("objects must be a count or a sequence of labels.")
    if isinstance(objects, int):
        object_count = objects
        object_labels: list[Hashable] | None = None
    elif isinstance(objects, Sequence) and not isinstance(objects, (str, bytes)):
        object_labels = list(objects)
        object_count = len(object_labels)
    else:
        raise ValueError("objects must be a count or a sequence of labels.")
    if isinstance(boxes, bool):
        raise ValueError("boxes must be a count or a sequence of labels.")
    if isinstance(boxes, int):
        box_count = boxes
        box_labels: list[Hashable] | None = None
    elif isinstance(boxes, Sequence) and not isinstance(boxes, (str, bytes)):
        box_labels = list(boxes)
        box_count = len(box_labels)
    else:
        raise ValueError("boxes must be a count or a sequence of labels.")
    _non_negative_integer(object_count, "The number of objects")
    _non_negative_integer(box_count, "The number of boxes")
    if box_count == 0:
        raise ValueError("At least one box is required.")
    if object_labels is not None and len(set(object_labels)) != object_count:
        raise ValueError("Object labels must be unique.")
    if box_labels is not None and len(set(box_labels)) != box_count:
        raise ValueError("Box labels must be unique.")

    if assignment is not None and object_labels is None:
        raise ValueError("An explicit assignment requires labeled objects.")
    if assignment is not None and box_labels is None:
        raise ValueError("An explicit assignment requires labeled boxes.")
    if object_labels is not None and box_labels is not None:
        if assignment is None:
            allocation = {
                obj: box_labels[index % box_count]
                for index, obj in enumerate(object_labels)
            }
        else:
            if set(assignment) != set(object_labels):
                raise ValueError("The assignment must specify every object exactly once.")
            if not set(assignment.values()).issubset(set(box_labels)):
                raise ValueError("Every assigned box must be in the boxes collection.")
            allocation = dict(assignment)
        counts = {box: 0 for box in box_labels}
        for box in allocation.values():
            counts[box] += 1
        allocated_maximum = max(counts.values(), default=0)
    else:
        allocation = None
        counts = None
        allocated_maximum = None

    guaranteed_minimum = (object_count + box_count - 1) // box_count
    return {
        "object_count": object_count,
        "box_count": box_count,
        "guaranteed_minimum_in_some_box": guaranteed_minimum,
        "allocation": allocation,
        "counts_by_box": counts,
        "maximum_in_allocated_box": allocated_maximum,
    }


def count_book_selections(book_count: int, choose_count: int) -> int:
    """Count ways to select an unordered set of books, without replacement."""
    return calculate_combinations(book_count, choose_count)


if __name__ == "__main__":
    print("Choose 3 of 10 books:", calculate_combinations(10, 3))
    print("Assign 2 of 5 distinct books to two ordered slots:",
          calculate_permutations(5, 2))
    print("Reservation options (3 time slots, 4 sections):",
          product_rule(3, 4))
    print("Binomial theorem, (x + 2)^3 at x=1:",
          binomial_expansion(1, 2, 3))
    print("Induction identity at n=10:", verify_sum_induction_example(10))
    print("Least positive candidate:", well_ordering_demo([12, 4, 9]))
    print("Pigeonhole result for 11 books in 4 shelves:",
          pigeonhole_analysis(11, 4))
    print("Floating-point binomial check:",
          binomial_expansion(0.1, 0.2, 3)["verified"])
