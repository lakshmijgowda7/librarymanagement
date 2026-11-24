"""Transparent, rule-based book recommendations for the library demo."""

from collections import Counter
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


RECOMMENDATION_FORMULA = (
    "score = 100 * (0.45 * interest_match + 0.35 * borrowing_history_match "
    "+ 0.20 * category_preference_match); each input is normalized to [0, 1]."
)

SAMPLE_BOOKS = pd.DataFrame(
    [
        {"book_id": 1, "title": "Introduction to Algorithms", "category": "Computing", "available_copies": 2},
        {"book_id": 2, "title": "Discrete Mathematics", "category": "Mathematics", "available_copies": 1},
        {"book_id": 3, "title": "The Hidden Life of Trees", "category": "Science", "available_copies": 0},
        {"book_id": 4, "title": "Data Science from Scratch", "category": "Computing", "available_copies": 3},
        {"book_id": 5, "title": "A Brief History of Time", "category": "Science", "available_copies": 1},
        {"book_id": 6, "title": "Mathematical Thinking", "category": "Mathematics", "available_copies": 0},
    ]
)


def _normalized_weights(values: Mapping[str, float] | Iterable[str] | None) -> dict[str, float]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        weights: dict[str, float] = {}
        for key, raw_value in values.items():
            value = float(raw_value)
            if not np.isfinite(value) or value < 0:
                raise ValueError("Interest and preference weights must be finite and non-negative.")
            weights[str(key).strip().casefold()] = value
    else:
        if isinstance(values, (str, bytes)):
            values = [values]
        weights = dict(Counter(str(value).strip().casefold() for value in values))
    maximum = max(weights.values(), default=0.0)
    if maximum == 0:
        return {}
    return {key: value / maximum for key, value in weights.items()}


def recommend_books(
    books: pd.DataFrame,
    interests: Mapping[str, float] | Iterable[str] | None = None,
    borrowing_history: Mapping[str, float] | Iterable[str] | None = None,
    category_preferences: Mapping[str, float] | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Rank books using normalized category signals and report availability separately.

    Required columns are ``book_id``, ``title``, ``category``, and either
    ``available_copies`` (a non-negative count) or ``available`` (a boolean).
    Availability does not change relevance scores; available titles are listed
    first so unavailable titles remain visible as future recommendations.
    """
    required = {"book_id", "title", "category"}
    missing = required.difference(books.columns)
    if missing:
        raise ValueError(f"Books data is missing required columns: {', '.join(sorted(missing))}.")
    if "available_copies" not in books.columns and "available" not in books.columns:
        raise ValueError("Books data must include 'available_copies' or 'available'.")

    result = books.copy()
    result["category"] = result["category"].fillna("").astype(str)
    category_keys = result["category"].str.strip().str.casefold()

    interest_weights = _normalized_weights(interests)
    history_weights = _normalized_weights(borrowing_history)
    preference_weights = _normalized_weights(category_preferences)

    result["interest_match"] = category_keys.map(interest_weights).fillna(0.0)
    result["borrowing_history_match"] = category_keys.map(history_weights).fillna(0.0)
    result["category_preference_match"] = category_keys.map(preference_weights).fillna(0.0)
    result["recommendation_score"] = 100.0 * (
        0.45 * result["interest_match"]
        + 0.35 * result["borrowing_history_match"]
        + 0.20 * result["category_preference_match"]
    )

    if "available_copies" in result.columns:
        copies = pd.to_numeric(result["available_copies"], errors="raise")
        if copies.isna().any() or not np.isfinite(copies).all() or (copies < 0).any() or (copies % 1 != 0).any():
            raise ValueError("'available_copies' must contain non-negative whole numbers.")
        result["available_copies"] = copies.astype(int)
        is_available = result["available_copies"] > 0
    else:
        if result["available"].isna().any() or not result["available"].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError("'available' must contain boolean values.")
        is_available = result["available"].astype(bool)
        result["available_copies"] = is_available.astype(int)

    result["availability_status"] = is_available.map({True: "Available", False: "Unavailable"})
    result["_availability_sort"] = (~is_available).astype(int)
    result = result.sort_values(
        ["_availability_sort", "recommendation_score", "title"],
        ascending=[True, False, True],
        kind="stable",
    )
    return result.drop(columns="_availability_sort").reset_index(drop=True)


def sample_recommendations() -> pd.DataFrame:
    """Return recommendations over the built-in illustrative catalog."""
    return recommend_books(
        SAMPLE_BOOKS,
        interests={"Computing": 1.0, "Mathematics": 0.8, "Science": 0.3},
        borrowing_history={"Computing": 3, "Mathematics": 1},
        category_preferences={"Mathematics": 1.0, "Computing": 0.7},
    )
