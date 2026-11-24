"""Analytic optimization demonstrations for the library project."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OPTIMIZATION_FORMULA = (
    "f(x, y) = (x - 2)^2 + 2(y - 1)^2 + 0.5(x - 2)(y - 1)"
)
OPTIMUM = np.array([2.0, 1.0])
HESSIAN = np.array([[2.0, 0.5], [0.5, 4.0]])


def quadratic_objective(point: Sequence[float]) -> float:
    """Evaluate the chosen strictly convex two-variable objective."""
    x, y = _point(point)
    return float((x - 2.0) ** 2 + 2.0 * (y - 1.0) ** 2 + 0.5 * (x - 2.0) * (y - 1.0))


def quadratic_gradient(point: Sequence[float]) -> np.ndarray:
    """Return the exact gradient: [2(x-2)+0.5(y-1), 4(y-1)+0.5(x-2)]."""
    x, y = _point(point)
    return np.array([2.0 * (x - 2.0) + 0.5 * (y - 1.0), 4.0 * (y - 1.0) + 0.5 * (x - 2.0)])


def quadratic_hessian() -> np.ndarray:
    """Return the constant Hessian of ``quadratic_objective``."""
    return HESSIAN.copy()


def _point(point: Sequence[float]) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError("A two-dimensional point (x, y) is required.")
    x, y = float(point[0]), float(point[1])
    if not np.isfinite([x, y]).all():
        raise ValueError("Point coordinates must be finite.")
    return x, y


def convexity_summary() -> dict[str, object]:
    """Return a positive-definiteness check for the objective Hessian."""
    eigenvalues = np.linalg.eigvalsh(HESSIAN)
    return {
        "hessian": HESSIAN.copy(),
        "eigenvalues": eigenvalues,
        "strictly_convex": bool(np.all(eigenvalues > 0)),
        "global_minimum": OPTIMUM.copy(),
        "minimum_value": quadratic_objective(OPTIMUM),
    }


def concave_maximum_summary() -> dict[str, object]:
    """For g=-f, return its unique global maximum and negative-definite Hessian."""
    return {
        "hessian": -HESSIAN.copy(),
        "eigenvalues": -np.linalg.eigvalsh(HESSIAN),
        "global_maximum": OPTIMUM.copy(),
        "maximum_value": -quadratic_objective(OPTIMUM),
    }


def allocation_lagrange_solution(
    target_desk_hours: Sequence[float] = (4.0, 3.0),
    total_hours: float = 5.0,
) -> dict[str, float]:
    """Minimize 1/2 sum((allocated-target)^2), subject to x+y=total.

    The defaults model allocating five staff-hours between two library desks
    whose preferred targets are four and three hours. They give x=3, y=2 and
    Lagrange multiplier lambda=1 for L=f+lambda(x+y-total).
    """
    targets = np.asarray(target_desk_hours, dtype=float)
    total = float(total_hours)
    if targets.shape != (2,) or not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("Two finite, non-negative target allocations are required.")
    if not np.isfinite(total) or total < 0:
        raise ValueError("Total hours must be finite and non-negative.")

    x = (total + targets[0] - targets[1]) / 2.0
    y = total - x
    if x < 0 or y < 0:
        raise ValueError("This demo requires an interior solution with non-negative allocations.")
    lagrange_multiplier = targets[0] - x
    objective = 0.5 * ((x - targets[0]) ** 2 + (y - targets[1]) ** 2)
    return {
        "desk_1_hours": float(x),
        "desk_2_hours": float(y),
        "lambda": float(lagrange_multiplier),
        "objective_value": float(objective),
    }


def gradient_descent(
    learning_rate: float,
    initial_point: Sequence[float] = (8.0, 5.0),
    tolerance: float = 1e-6,
    max_iterations: int = 10000,
) -> tuple[pd.DataFrame, bool]:
    """Minimize the selected objective and return the full iterate history."""
    rate = float(learning_rate)
    tol = float(tolerance)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    stability_limit = 2.0 / float(np.linalg.eigvalsh(HESSIAN).max())
    if rate >= stability_limit:
        raise ValueError(f"Learning rate must be less than {stability_limit:.6g} for this objective to converge.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("Tolerance must be finite and positive.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    point = np.asarray(_point(initial_point), dtype=float)
    rows = []
    converged = False
    for iteration in range(max_iterations + 1):
        gradient = quadratic_gradient(point)
        rows.append(
            {
                "iteration": iteration,
                "x": point[0],
                "y": point[1],
                "loss": quadratic_objective(point),
                "gradient_norm": float(np.linalg.norm(gradient)),
            }
        )
        if rows[-1]["gradient_norm"] <= tol:
            converged = True
            break
        if iteration < max_iterations:
            point = point - rate * gradient
    return pd.DataFrame(rows), converged


def compare_learning_rates(
    learning_rates: Sequence[float] = (0.05, 0.2, 0.4),
    initial_point: Sequence[float] = (8.0, 5.0),
    tolerance: float = 1e-6,
    max_iterations: int = 10000,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    """Compare convergence histories using the same initial point and target."""
    summary_rows = []
    histories: dict[float, pd.DataFrame] = {}
    for raw_rate in learning_rates:
        rate = float(raw_rate)
        history, converged = gradient_descent(rate, initial_point, tolerance, max_iterations)
        histories[rate] = history
        final = history.iloc[-1]
        summary_rows.append(
            {
                "learning_rate": rate,
                "iterations": int(final["iteration"]),
                "final_loss": float(final["loss"]),
                "final_gradient_norm": float(final["gradient_norm"]),
                "converged": converged,
            }
        )
    return pd.DataFrame(summary_rows), histories


def convergence_figure(histories: dict[float, pd.DataFrame]):
    """Plot loss against iteration for a set of gradient-descent histories."""
    figure, axis = plt.subplots()
    for rate, history in histories.items():
        axis.plot(history["iteration"], history["loss"], label=f"learning rate = {rate:g}")
    axis.set(title="Gradient descent convergence", xlabel="Iteration", ylabel="Objective value")
    axis.set_yscale("symlog", linthresh=1e-10)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure
