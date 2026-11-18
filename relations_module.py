"""Relations, functions, matrices, equivalence classes, and partial orders."""

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any

Pair = tuple[Hashable, Hashable]


def cartesian_product(
    first: Iterable[Hashable], second: Iterable[Hashable]
) -> set[Pair]:
    """Return the Cartesian product ``first × second`` as a set of pairs."""
    return {(left, right) for left in first for right in second}


def _relation_pairs(relation: Iterable[Sequence[Hashable]]) -> set[Pair]:
    pairs: set[Pair] = set()
    for pair in relation:
        if len(pair) != 2:
            raise ValueError("Every relation entry must contain exactly two values.")
        pairs.add((pair[0], pair[1]))
    return pairs


def create_borrowing_matrix(
    students: Sequence[Hashable],
    books: Sequence[Hashable],
    records: Iterable[Mapping[str, Any] | Sequence[Hashable]],
) -> dict[str, Any]:
    """Create a zero-one student-by-book matrix.

    A record may be ``(student_id, book_id)`` or a mapping with
    ``student_id`` and ``book_id`` keys. Repeated borrowing records remain 1.
    The returned matrix is a list of row lists in the supplied label order.
    """
    student_labels = list(students)
    book_labels = list(books)
    if len(set(student_labels)) != len(student_labels):
        raise ValueError("Student IDs must be unique.")
    if len(set(book_labels)) != len(book_labels):
        raise ValueError("Book IDs must be unique.")
    student_index = {student: index for index, student in enumerate(student_labels)}
    book_index = {book: index for index, book in enumerate(book_labels)}
    matrix = [[0 for _ in book_labels] for _ in student_labels]

    for record in records:
        if isinstance(record, Mapping):
            try:
                student, book = record["student_id"], record["book_id"]
            except KeyError as exc:
                raise ValueError(
                    "Borrowing mappings require student_id and book_id."
                ) from exc
        else:
            if len(record) != 2:
                raise ValueError("Borrowing tuple records must have two values.")
            student, book = record
        if student not in student_index:
            raise ValueError(f"Unknown student ID: {student!r}")
        if book not in book_index:
            raise ValueError(f"Unknown book ID: {book!r}")
        matrix[student_index[student]][book_index[book]] = 1

    return {"students": student_labels, "books": book_labels, "matrix": matrix}


def zero_one_matrix(
    rows: Sequence[Hashable],
    columns: Sequence[Hashable],
    relation: Iterable[Sequence[Hashable]],
) -> dict[str, Any]:
    """Represent a relation as a zero-one matrix in the supplied label order."""
    row_labels, column_labels = list(rows), list(columns)
    if len(set(row_labels)) != len(row_labels):
        raise ValueError("Row labels must be unique.")
    if len(set(column_labels)) != len(column_labels):
        raise ValueError("Column labels must be unique.")
    row_index = {value: index for index, value in enumerate(row_labels)}
    column_index = {value: index for index, value in enumerate(column_labels)}
    matrix = [[0 for _ in column_labels] for _ in row_labels]
    for pair in _relation_pairs(relation):
        left, right = pair
        if left not in row_index or right not in column_index:
            raise ValueError(f"Relation pair {pair!r} is outside the supplied sets.")
        matrix[row_index[left]][column_index[right]] = 1
    return {"rows": row_labels, "columns": column_labels, "matrix": matrix}


def analyze_function(
    domain: Iterable[Hashable],
    codomain: Iterable[Hashable],
    mapping: Mapping[Hashable, Hashable] | Iterable[Sequence[Hashable]],
) -> dict[str, Any]:
    """Check whether a mapping is a total function and its injectivity/surjectivity.

    Pair iterables are accepted to detect duplicate inputs that a dictionary
    would otherwise hide. ``is_bijective`` is true exactly when the relation
    defines a function that is both one-to-one and onto.
    """
    domain_set, codomain_set = set(domain), set(codomain)
    pairs = list(mapping.items()) if isinstance(mapping, Mapping) else list(mapping)
    normalized: dict[Hashable, Hashable] = {}
    duplicate_inputs: set[Hashable] = set()
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("Each mapping entry must contain an input and output.")
        source, target = pair
        if source in normalized:
            duplicate_inputs.add(source)
        normalized[source] = target

    missing_inputs = domain_set - set(normalized)
    extra_inputs = set(normalized) - domain_set
    outputs_outside_codomain = set(normalized.values()) - codomain_set
    preimages: dict[Hashable, list[Hashable]] = {}
    for source, target in normalized.items():
        preimages.setdefault(target, []).append(source)
    collisions = {
        target: sources for target, sources in preimages.items() if len(sources) > 1
    }
    is_function = not (
        missing_inputs or extra_inputs or outputs_outside_codomain or duplicate_inputs
    )
    is_injective = is_function and not collisions
    is_surjective = is_function and codomain_set.issubset(set(normalized.values()))
    return {
        "is_function": is_function,
        "is_injective": is_injective,
        "is_surjective": is_surjective,
        "is_bijective": is_injective and is_surjective,
        "missing_inputs": missing_inputs,
        "extra_inputs": extra_inputs,
        "outputs_outside_codomain": outputs_outside_codomain,
        "duplicate_inputs": duplicate_inputs,
        "collisions": collisions,
        "unreached_codomain": codomain_set - set(normalized.values()),
    }


def _function_mapping(
    mapping: Mapping[Hashable, Hashable] | Iterable[Sequence[Hashable]],
) -> dict[Hashable, Hashable]:
    pairs = list(mapping.items()) if isinstance(mapping, Mapping) else list(mapping)
    result: dict[Hashable, Hashable] = {}
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("Each mapping entry must contain an input and output.")
        source, target = pair
        if source in result:
            raise ValueError(f"Input {source!r} has more than one output.")
        result[source] = target
    return result


def compose_functions(
    first: Mapping[Hashable, Hashable] | Iterable[Sequence[Hashable]],
    second: Mapping[Hashable, Hashable] | Iterable[Sequence[Hashable]],
) -> dict[Hashable, Hashable]:
    """Return ``second ∘ first``; every first output must be a second input."""
    first_map, second_map = _function_mapping(first), _function_mapping(second)
    result: dict[Hashable, Hashable] = {}
    for source, intermediate in first_map.items():
        if intermediate not in second_map:
            raise ValueError(
                f"First function output {intermediate!r} is not in the "
                "second function's domain."
            )
        result[source] = second_map[intermediate]
    return result


def inverse_function(
    mapping: Mapping[Hashable, Hashable] | Iterable[Sequence[Hashable]],
) -> dict[Hashable, Hashable]:
    """Return the inverse mapping, requiring the supplied mapping to be injective."""
    function = _function_mapping(mapping)
    if len(set(function.values())) != len(function):
        raise ValueError("An inverse function exists only for a one-to-one mapping.")
    return {target: source for source, target in function.items()}


def equivalence_classes(
    domain: Iterable[Hashable], relation: Iterable[Sequence[Hashable]]
) -> list[set[Hashable]]:
    """Validate an equivalence relation and return its disjoint classes."""
    elements = set(domain)
    pairs = _relation_pairs(relation)
    if any(left not in elements or right not in elements for left, right in pairs):
        raise ValueError("Relation pairs must belong to the supplied domain.")
    if any((element, element) not in pairs for element in elements):
        raise ValueError("Relation is not reflexive.")
    if any((right, left) not in pairs for left, right in pairs):
        raise ValueError("Relation is not symmetric.")
    if any((left, third) not in pairs
           for left, middle in pairs
           for middle2, third in pairs
           if middle == middle2):
        raise ValueError("Relation is not transitive.")

    unseen = set(elements)
    classes: list[set[Hashable]] = []
    while unseen:
        representative = next(iter(unseen))
        members = {right for left, right in pairs if left == representative}
        classes.append(members)
        unseen -= members
    return classes


def generate_hasse_diagram(
    elements: Iterable[Hashable],
    less_equal: Callable[[Hashable, Hashable], bool],
) -> dict[str, Any]:
    """Return the cover edges of a finite partially ordered set.

    Edges are oriented from the smaller element to the larger element.
    """
    nodes = list(elements)
    if len(set(nodes)) != len(nodes):
        raise ValueError("Poset elements must be unique.")
    relation = {(left, right) for left in nodes for right in nodes
                if less_equal(left, right)}
    if any((node, node) not in relation for node in nodes):
        raise ValueError("The supplied order is not reflexive.")
    if any(
        left != right and (right, left) in relation
        for left, right in relation
    ):
        raise ValueError("The supplied order is not antisymmetric.")
    if any(
        (left, third) not in relation
        for left, middle in relation
        for middle2, third in relation
        if middle == middle2
    ):
        raise ValueError("The supplied order is not transitive.")

    covers = []
    for lower, upper in relation:
        if lower == upper:
            continue
        has_intermediate = any(
            middle not in (lower, upper)
            and (lower, middle) in relation
            and (middle, upper) in relation
            for middle in nodes
        )
        if not has_intermediate:
            covers.append((lower, upper))
    return {"elements": nodes, "cover_edges": sorted(covers, key=repr)}


def divisibility_hasse_diagram(number: int) -> dict[str, Any]:
    """Return the Hasse diagram for positive divisors ordered by divisibility."""
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise ValueError("number must be a positive integer.")
    divisors = [candidate for candidate in range(1, number + 1)
                if number % candidate == 0]
    return generate_hasse_diagram(
        divisors, lambda left, right: right % left == 0
    )


if __name__ == "__main__":
    students = ["S1", "S2", "S3"]
    books = ["B1", "B2"]
    records = [("S1", "B1"), ("S2", "B1"), ("S3", "B2")]
    print("Borrowing matrix:", create_borrowing_matrix(students, books, records))
    print("Function analysis:", analyze_function(
        {1, 2, 3}, {"a", "b"}, {1: "a", 2: "b", 3: "a"}
    ))
    print("Partitions modulo 2:", equivalence_classes(
        {0, 1, 2, 3},
        {(value, other) for value in range(4) for other in range(4)
         if value % 2 == other % 2},
    ))
    print("Divisibility Hasse diagram for 12:", divisibility_hasse_diagram(12))
