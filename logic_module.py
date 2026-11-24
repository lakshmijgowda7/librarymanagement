"""Propositional logic helpers and the library issue-eligibility model."""

from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import TypeAlias

TruthFunction: TypeAlias = Callable[[Mapping[str, bool]], bool]


def _proposition(value: bool, name: str = "proposition") -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def logical_not(value: bool) -> bool:
    """Return the negation of a proposition."""
    return not _proposition(value)


def logical_and(*values: bool) -> bool:
    """Return the conjunction of zero or more propositions."""
    propositions = [_proposition(value) for value in values]
    return all(propositions)


def logical_or(*values: bool) -> bool:
    """Return the disjunction of zero or more propositions."""
    propositions = [_proposition(value) for value in values]
    return any(propositions)


def exclusive_or(left: bool, right: bool) -> bool:
    """Return true when exactly one proposition is true."""
    return _proposition(left, "left") != _proposition(right, "right")


def implies(antecedent: bool, consequent: bool) -> bool:
    """Evaluate material implication, ``antecedent -> consequent``."""
    return not _proposition(antecedent, "antecedent") or _proposition(
        consequent, "consequent"
    )


def equivalent(left: bool, right: bool) -> bool:
    """Evaluate the biconditional for two truth values."""
    return _proposition(left, "left") == _proposition(right, "right")


def check_eligibility(
    registered: bool, eligible: bool, available: bool, overdue: bool
) -> bool:
    """Evaluate ``I = R AND E AND A AND NOT O``."""
    return logical_and(
        _proposition(registered, "registered"),
        _proposition(eligible, "eligible"),
        _proposition(available, "available"),
        logical_not(_proposition(overdue, "overdue")),
    )


def _variables(variables: Sequence[str]) -> list[str]:
    if isinstance(variables, (str, bytes)) or not isinstance(variables, Sequence):
        raise ValueError("variables must be a sequence of unique names.")
    names = list(variables)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Each variable name must be a non-empty string.")
    if len(set(names)) != len(names):
        raise ValueError("Variable names must be unique.")
    return names


def truth_table(
    expression: TruthFunction, variables: Sequence[str]
) -> list[dict[str, bool]]:
    """Return all assignments and the result of a Boolean expression."""
    if not callable(expression):
        raise ValueError("expression must be callable.")
    names = _variables(variables)
    rows: list[dict[str, bool]] = []
    for values in product((False, True), repeat=len(names)):
        assignment = dict(zip(names, values))
        result = expression(assignment)
        row = dict(assignment)
        row["result"] = _proposition(result, "expression result")
        rows.append(row)
    return rows


def generate_truth_table() -> list[dict[str, bool]]:
    """Generate the 16-row truth table for the issue rule ``I = R & E & A & ~O``."""
    generic_rows = truth_table(
        lambda values: check_eligibility(
            values["R"], values["E"], values["A"], values["O"]
        ),
        ("R", "E", "A", "O"),
    )
    return [
        {
            "R": row["R"],
            "E": row["E"],
            "A": row["A"],
            "O": row["O"],
            "I": row["result"],
        }
        for row in generic_rows
    ]


def are_logically_equivalent(
    left: TruthFunction, right: TruthFunction, variables: Sequence[str]
) -> bool:
    """Check whether two Boolean expressions agree for every assignment."""
    names = _variables(variables)
    left_table = truth_table(left, names)
    right_table = truth_table(right, names)
    return all(
        left_row["result"] == right_row["result"]
        for left_row, right_row in zip(left_table, right_table)
    )


def modus_ponens(antecedent: bool, implication: bool) -> bool:
    """Apply modus ponens to premises ``P`` and ``P -> Q``."""
    return logical_and(
        _proposition(antecedent, "antecedent"),
        _proposition(implication, "implication"),
    )


def modus_tollens(not_consequent: bool, implication: bool) -> bool:
    """Apply modus tollens to premises ``P -> Q`` and ``NOT Q``."""
    return logical_and(
        _proposition(not_consequent, "not_consequent"),
        _proposition(implication, "implication"),
    )


def disjunctive_syllogism(disjunction: bool, not_left: bool) -> bool:
    """Apply disjunctive syllogism to ``P OR Q`` and ``NOT P``."""
    return logical_and(
        _proposition(disjunction, "disjunction"),
        _proposition(not_left, "not_left"),
    )


if __name__ == "__main__":
    print("Eligibility rule: I = R AND E AND A AND NOT O")
    print("R E A O | I")
    for row in generate_truth_table():
        print(
            f"{int(row['R'])} {int(row['E'])} {int(row['A'])} "
            f"{int(row['O'])} | {int(row['I'])}"
        )
    print(
        "Equivalent to (R AND E AND A) -> NOT O:",
        are_logically_equivalent(
            lambda v: check_eligibility(v["R"], v["E"], v["A"], v["O"]),
            lambda v: implies(
                logical_and(v["R"], v["E"], v["A"]), logical_not(v["O"])
            ),
            ("R", "E", "A", "O"),
        ),
    )
