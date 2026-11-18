"""Recurrence relations for illustrative inventory and reservation examples.

Generated sequences are mathematical demonstrations, not empirical forecasts.
"""

from collections.abc import Sequence
from cmath import sqrt as complex_sqrt
from math import isclose
from typing import Any


def _period_count(periods: int) -> None:
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 0:
        raise ValueError("periods must be a non-negative integer.")


def generate_first_order_sequence(
    initial_value: int | float,
    changes: int | float | Sequence[int | float],
    periods: int,
) -> list[int | float]:
    """Generate ``x[t+1] = x[t] + changes[t]`` for ``periods`` transitions.

    ``changes`` may be one constant change or a sequence with at least one
    change per transition. The result includes the initial value.
    """
    _period_count(periods)
    if isinstance(changes, (int, float)):
        increments = [changes] * periods
    else:
        increments = list(changes)
        if len(increments) < periods:
            raise ValueError("changes must contain at least one value per period.")
    values: list[int | float] = [initial_value]
    for change in increments[:periods]:
        values.append(values[-1] + change)
    return values


def solve_first_order_recurrence(
    initial_value: int | float,
    multiplier: int | float,
    constant: int | float,
    periods: int,
) -> list[int | float]:
    """Generate the solution of ``x[n] = multiplier*x[n-1] + constant``."""
    _period_count(periods)
    values: list[int | float] = [initial_value]
    for _ in range(periods):
        values.append(multiplier * values[-1] + constant)
    return values


def solve_first_order_closed_form(
    initial_value: int | float,
    multiplier: int | float,
    constant: int | float,
    periods: int,
) -> list[int | float]:
    """Evaluate the closed form for ``x[n] = a*x[n-1] + c``.

    For ``a != 1``, ``x[n] = a**n*x[0] + c*(a**n - 1)/(a - 1)``.
    For ``a == 1``, ``x[n] = x[0] + n*c``. The returned list includes
    ``x[0]`` through ``x[periods]``.
    """
    _period_count(periods)
    if multiplier == 1:
        return [initial_value + n * constant for n in range(periods + 1)]
    return [
        multiplier**n * initial_value
        + constant * (multiplier**n - 1) / (multiplier - 1)
        for n in range(periods + 1)
    ]


def generate_inventory_sequence(
    initial_inventory: int,
    deliveries: Sequence[int],
    borrowings: Sequence[int],
) -> list[int]:
    """Compute inventory by period: ``I[t+1] = I[t] + deliveries - loans``."""
    for name, value in [("initial_inventory", initial_inventory)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    if len(deliveries) != len(borrowings):
        raise ValueError("deliveries and borrowings must have equal lengths.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in [*deliveries, *borrowings]):
        raise ValueError("Deliveries and borrowings must be non-negative integers.")
    inventory = [initial_inventory]
    for period, (delivered, borrowed) in enumerate(zip(deliveries, borrowings), 1):
        next_inventory = inventory[-1] + delivered - borrowed
        if next_inventory < 0:
            raise ValueError(f"Inventory becomes negative after period {period}.")
        inventory.append(next_inventory)
    return inventory


def generate_second_order_sequence(
    initial_0: int | float,
    initial_1: int | float,
    periods: int,
    coefficient_1: int | float = 1,
    coefficient_2: int | float = 1,
) -> list[int | float]:
    """Generate ``x[n] = coefficient_1*x[n-1] + coefficient_2*x[n-2]``.

    ``periods`` is the number of terms after the two initial terms.
    Defaults produce the Fibonacci recurrence.
    """
    _period_count(periods)
    values: list[int | float] = [initial_0, initial_1]
    for _ in range(periods):
        values.append(coefficient_1 * values[-1] + coefficient_2 * values[-2])
    return values


def _clean_number(value: complex, tolerance: float = 1e-10) -> int | float | complex:
    if abs(value.imag) < tolerance:
        real = value.real
        if isclose(real, round(real), abs_tol=tolerance):
            return int(round(real))
        return real
    return value


def solve_second_order_characteristic(
    initial_0: int | float,
    initial_1: int | float,
    coefficient_1: int | float,
    coefficient_2: int | float,
    periods: int,
) -> dict[str, Any]:
    """Solve a second-order recurrence from its characteristic roots.

    For ``x[n] = p*x[n-1] + q*x[n-2]``, the characteristic equation is
    ``r**2 - p*r - q = 0``. Distinct roots use ``A*r1**n + B*r2**n``;
    a repeated root uses ``(A + B*n)*r**n``.
    """
    _period_count(periods)
    discriminant = coefficient_1**2 + 4 * coefficient_2
    root = complex_sqrt(complex(discriminant))
    root_1 = (coefficient_1 + root) / 2
    root_2 = (coefficient_1 - root) / 2
    tolerance = 1e-12

    if abs(root_1 - root_2) <= tolerance:
        repeated_root = root_1
        if abs(repeated_root) <= tolerance:
            a, b = None, None
            formula_values = [initial_0, initial_1]
            for _ in range(periods):
                formula_values.append(
                    coefficient_1 * formula_values[-1]
                    + coefficient_2 * formula_values[-2]
                )
        else:
            a = initial_0
            b = initial_1 / repeated_root - a
            formula_values = [
                (a + b * n) * repeated_root**n for n in range(periods + 2)
            ]
        root_description: tuple[Any, ...] = (root_1,)
        root_case = "repeated"
    else:
        a = (initial_1 - initial_0 * root_2) / (root_1 - root_2)
        b = initial_0 - a
        formula_values = [
            a * root_1**n + b * root_2**n for n in range(periods + 2)
        ]
        root_description = (root_1, root_2)
        root_case = "distinct"

    values = [_clean_number(complex(value)) for value in formula_values]
    return {
        "root_case": root_case,
        "characteristic_roots": tuple(_clean_number(complex(value))
                                      for value in root_description),
        "formula_constants": (
            None if a is None else _clean_number(complex(a)),
            None if b is None else _clean_number(complex(b)),
        ),
        "sequence": values,
    }


if __name__ == "__main__":
    print("Inventory changes:", generate_first_order_sequence(120, [-5, 8, -3], 3))
    print("Inventory with deliveries and borrowing:",
          generate_inventory_sequence(20, [4, 0, 10], [3, 6, 8]))
    print("First-order closed recurrence:",
          solve_first_order_recurrence(2, 2, 1, 5))
    print("First-order closed-form check:",
          solve_first_order_closed_form(2, 2, 1, 5))
    print("Fibonacci recurrence:",
          generate_second_order_sequence(0, 1, 8))
    print("Fibonacci characteristic solution:",
          solve_second_order_characteristic(0, 1, 1, 1, 8))
    print("Repeated-root example:",
          solve_second_order_characteristic(2, 3, 2, -1, 5))
