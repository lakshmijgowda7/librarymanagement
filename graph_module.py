"""Library graph construction, analysis, visualization, and prefix coding.

NetworkX and Matplotlib are loaded only by functions that need them, so importing
this module does not prevent use of the other mathematical modules.
"""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from importlib import import_module
from itertools import combinations
from math import isfinite
from numbers import Real
from typing import Any


def _networkx() -> Any:
    try:
        return import_module("networkx")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Graph algorithms require NetworkX. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc


def _pyplot() -> Any:
    try:
        return import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Graph visualization requires Matplotlib. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc


def _entity_id(value: Any, entity_type: str) -> Hashable:
    identifier = value.get("id") if isinstance(value, Mapping) else value
    if identifier is None:
        raise ValueError(f"Each {entity_type} mapping must contain a non-null 'id'.")
    try:
        hash(identifier)
    except TypeError as exc:
        raise ValueError(f"{entity_type} IDs must be hashable.") from exc
    return identifier


def _related_values(value: Any) -> list[Hashable]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return [value]
    return list(value)


def build_library_graph(
    books: Sequence[Any],
    students: Sequence[Any],
    records: Iterable[Mapping[str, Any] | Sequence[Hashable]],
) -> Any:
    """Build a directed graph of student loans and book metadata.

    A student node points to a book node for each borrowing record. Book
    mappings may additionally contain ``author``, ``category``, or ``section``
    values (single values or sequences); corresponding typed nodes and edges
    represent those relationships. IDs are wrapped as ``(kind, id)`` tuples.
    Records may be ``(student_id, book_id)`` pairs or mappings with
    ``student_id`` and ``book_id`` keys. Repeated loans increment edge ``count``.
    """
    nx = _networkx()
    graph = nx.DiGraph()
    book_ids: set[Hashable] = set()
    student_ids: set[Hashable] = set()

    for student in students:
        identifier = _entity_id(student, "student")
        if identifier in student_ids:
            raise ValueError(f"Duplicate student ID: {identifier!r}")
        student_ids.add(identifier)
        graph.add_node(("student", identifier), node_type="student", label=str(identifier))

    metadata_fields = {
        "author": "written_by",
        "category": "categorized_as",
        "section": "located_in",
    }
    for book in books:
        identifier = _entity_id(book, "book")
        if identifier in book_ids:
            raise ValueError(f"Duplicate book ID: {identifier!r}")
        book_ids.add(identifier)
        title = book.get("title", identifier) if isinstance(book, Mapping) else identifier
        book_node = ("book", identifier)
        graph.add_node(book_node, node_type="book", label=str(title), book_id=identifier)
        if isinstance(book, Mapping):
            for field, relationship in metadata_fields.items():
                for related in _related_values(book.get(field)):
                    related_node = (field, related)
                    graph.add_node(
                        related_node, node_type=field, label=str(related)
                    )
                    graph.add_edge(book_node, related_node, relation=relationship)

    for record in records:
        if isinstance(record, Mapping):
            try:
                student_id, book_id = record["student_id"], record["book_id"]
            except KeyError as exc:
                raise ValueError(
                    "Borrowing mappings require student_id and book_id."
                ) from exc
        else:
            if len(record) != 2:
                raise ValueError("Borrowing tuple records must have two values.")
            student_id, book_id = record
        if student_id not in student_ids:
            raise ValueError(f"Unknown student ID: {student_id!r}")
        if book_id not in book_ids:
            raise ValueError(f"Unknown book ID: {book_id!r}")
        source, target = ("student", student_id), ("book", book_id)
        if graph.has_edge(source, target):
            graph[source][target]["count"] += 1
        else:
            graph.add_edge(source, target, relation="borrowed", count=1)
    return graph


def _undirected_projection(graph: Any) -> Any:
    nx = _networkx()
    if graph.is_directed():
        projection = nx.Graph()
        projection.add_nodes_from(graph.nodes(data=True))
        projection.add_edges_from(graph.edges())
        return projection
    return graph.copy()


def analyze_graph(graph: Any) -> dict[str, Any]:
    """Return graph size, degree, connectivity, block, cut-point, and bridge data.

    For directed graphs, connected components, blocks, cut-points, and bridges
    are calculated on the undirected projection; directed in/out-degrees remain
    separately available. ``degree_sequence`` uses total degree for digraphs.
    """
    nx = _networkx()
    projection = _undirected_projection(graph)
    nodes = list(graph.nodes)
    degree_by_node = dict(graph.degree())
    components = [set(component) for component in nx.connected_components(projection)]
    blocks = [set(block) for block in nx.biconnected_components(projection)]
    cut_points = set(nx.articulation_points(projection))
    bridges = list(nx.bridges(projection))
    result = {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "is_directed": graph.is_directed(),
        "has_self_loops": nx.number_of_selfloops(graph) > 0,
        "is_connected": len(components) == 1 if nodes else True,
        "connected_components": components,
        "degree_by_node": degree_by_node,
        "degree_sequence": sorted(degree_by_node.values(), reverse=True),
        "is_complete": (
            projection.number_of_edges()
            == projection.number_of_nodes() * (projection.number_of_nodes() - 1) // 2
        ),
        "is_multigraph": graph.is_multigraph(),
        "is_simple": not graph.is_multigraph() and not nx.number_of_selfloops(graph),
        "blocks": blocks,
        "cut_points": cut_points,
        "bridges": bridges,
        "is_tree": nx.is_tree(projection) if nodes else False,
        "is_bipartite": nx.is_bipartite(projection),
    }
    if graph.is_directed():
        result["in_degree_by_node"] = dict(graph.in_degree())
        result["out_degree_by_node"] = dict(graph.out_degree())
    return result


def create_subgraph(graph: Any, nodes: Iterable[Hashable]) -> Any:
    """Return a copy of the induced subgraph on the selected existing nodes."""
    selected = list(nodes)
    missing = set(selected) - set(graph.nodes)
    if missing:
        raise ValueError(f"Subgraph contains unknown nodes: {missing!r}")
    return graph.subgraph(selected).copy()


def analyze_walk(graph: Any, vertices: Sequence[Hashable]) -> dict[str, Any]:
    """Validate a vertex sequence as a walk and report path/cycle properties."""
    if not vertices:
        raise ValueError("A walk must contain at least one vertex.")
    missing = set(vertices) - set(graph.nodes)
    if missing:
        raise ValueError(f"Walk contains unknown nodes: {missing!r}")
    for left, right in zip(vertices, vertices[1:]):
        if not graph.has_edge(left, right):
            raise ValueError(f"There is no edge from {left!r} to {right!r}.")
    is_cycle = len(vertices) > 1 and vertices[0] == vertices[-1]
    return {
        "vertices": list(vertices),
        "edge_count": len(vertices) - 1,
        "is_walk": True,
        "is_path": len(set(vertices)) == len(vertices),
        "is_cycle": is_cycle,
    }


def calculate_shortest_path(
    graph: Any, source: Hashable, destination: Hashable
) -> dict[str, Any]:
    """Return a shortest path and its number of edges.

    Directed graphs respect edge direction. A missing node or unreachable
    destination raises NetworkX's standard ``NodeNotFound`` or ``NetworkXNoPath``.
    """
    nx = _networkx()
    path = nx.shortest_path(graph, source=source, target=destination)
    return {"path": path, "distance": len(path) - 1}


def self_complementary_graph_example() -> Any:
    """Return C5, a valid graph isomorphic to its own complement."""
    nx = _networkx()
    graph = nx.cycle_graph(5)
    if not nx.is_isomorphic(graph, nx.complement(graph)):
        raise RuntimeError("The C5 self-complementarity check unexpectedly failed.")
    return graph


def build_block_graph(graph: Any) -> Any:
    """Construct the block-cut forest: block nodes linked to cut-point nodes."""
    nx = _networkx()
    projection = _undirected_projection(graph)
    block_graph = nx.Graph()
    cut_points = set(nx.articulation_points(projection))
    for index, block in enumerate(nx.biconnected_components(projection)):
        block_node = ("block", index)
        block_graph.add_node(block_node, node_type="block", members=set(block))
        for cut_point in cut_points.intersection(block):
            cut_node = ("cut_point", cut_point)
            block_graph.add_node(cut_node, node_type="cut_point", label=cut_point)
            block_graph.add_edge(block_node, cut_node)
    return block_graph


def build_cutpoint_graph(graph: Any) -> Any:
    """Connect cut-points that occur together in at least one biconnected block."""
    nx = _networkx()
    projection = _undirected_projection(graph)
    cut_points = set(nx.articulation_points(projection))
    cut_graph = nx.Graph()
    cut_graph.add_nodes_from(cut_points)
    for block in nx.biconnected_components(projection):
        members = sorted(cut_points.intersection(block), key=repr)
        cut_graph.add_edges_from(combinations(members, 2))
    return cut_graph


def draw_graph(
    graph: Any,
    *,
    title: str = "Library relationship graph",
    save_path: str | None = None,
    show: bool = True,
) -> Any:
    """Draw a graph with labels; optionally save it and/or display the figure."""
    nx = _networkx()
    plt = _pyplot()
    labels = {
        node: str(data.get("label", node))
        for node, data in graph.nodes(data=True)
    }
    positions = nx.spring_layout(graph, seed=7)
    figure, axis = plt.subplots(figsize=(10, 7))
    nx.draw_networkx(
        graph, positions, labels=labels, with_labels=True, ax=axis,
        node_size=1400, font_size=8, arrows=graph.is_directed(),
    )
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def draw_hasse_diagram(
    elements: Sequence[Hashable],
    cover_edges: Sequence[tuple[Hashable, Hashable]],
    *,
    title: str = "Hasse diagram",
    save_path: str | None = None,
    show: bool = True,
) -> Any:
    """Draw a Hasse diagram from the smaller-to-larger cover edges."""
    nx = _networkx()
    plt = _pyplot()
    graph = nx.DiGraph()
    graph.add_nodes_from(elements)
    graph.add_edges_from(cover_edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Hasse cover edges must form a directed acyclic graph.")
    generations = list(nx.topological_generations(graph))
    positions: dict[Hashable, tuple[float, float]] = {}
    for level, generation in enumerate(generations):
        ordered = sorted(generation, key=repr)
        center = (len(ordered) - 1) / 2
        for index, node in enumerate(ordered):
            positions[node] = (index - center, -float(level))
    figure, axis = plt.subplots(figsize=(8, 6))
    nx.draw_networkx(
        graph, positions, labels={node: str(node) for node in graph},
        with_labels=True, arrows=False, node_size=1200, ax=axis,
    )
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def build_huffman_codes(frequencies: Mapping[Hashable, Real]) -> dict[Hashable, str]:
    """Build a binary Huffman prefix code from positive symbol frequencies."""
    if not frequencies:
        raise ValueError("At least one symbol frequency is required.")
    heap: list[tuple[float, int, Any]] = []
    for order, (symbol, frequency) in enumerate(frequencies.items()):
        if isinstance(frequency, bool) or not isinstance(frequency, Real):
            raise ValueError("Frequencies must be positive finite numbers.")
        numeric_frequency = float(frequency)
        if not isfinite(numeric_frequency) or numeric_frequency <= 0:
            raise ValueError("Frequencies must be positive finite numbers.")
        heap.append((numeric_frequency, order, ("leaf", symbol)))
    import heapq

    heapq.heapify(heap)
    if len(heap) == 1:
        return {heap[0][2][1]: "0"}
    next_order = len(heap)
    while len(heap) > 1:
        left_frequency, _, left = heapq.heappop(heap)
        right_frequency, _, right = heapq.heappop(heap)
        heapq.heappush(
            heap,
            (left_frequency + right_frequency, next_order,
             ("branch", left, right)),
        )
        next_order += 1

    codes: dict[Hashable, str] = {}

    def visit(node: Any, prefix: str) -> None:
        if node[0] == "leaf":
            codes[node[1]] = prefix
            return
        visit(node[1], prefix + "0")
        visit(node[2], prefix + "1")

    visit(heap[0][2], "")
    return codes


def encode_message(message: Iterable[Hashable], codes: Mapping[Hashable, str]) -> str:
    """Encode symbols with a prefix-code mapping."""
    encoded: list[str] = []
    for symbol in message:
        if symbol not in codes:
            raise ValueError(f"No code is defined for symbol {symbol!r}.")
        code = codes[symbol]
        if not code or any(bit not in "01" for bit in code):
            raise ValueError(f"Invalid binary code for symbol {symbol!r}.")
        encoded.append(code)
    return "".join(encoded)


def decode_message(encoded: str, codes: Mapping[Hashable, str]) -> list[Hashable]:
    """Decode a bit string using a prefix-free binary code mapping."""
    if any(bit not in "01" for bit in encoded):
        raise ValueError("The encoded message must contain only 0 and 1.")
    if not codes or any(not code for code in codes.values()):
        raise ValueError("Codes must be a non-empty mapping of non-empty bit strings.")
    code_to_symbol = {code: symbol for symbol, code in codes.items()}
    if len(code_to_symbol) != len(codes):
        raise ValueError("Each symbol must have a unique code.")
    if any(
        other.startswith(code)
        for code in code_to_symbol
        for other in code_to_symbol
        if other != code
    ):
        raise ValueError("The code mapping must be prefix-free.")
    result: list[Hashable] = []
    buffer = ""
    for bit in encoded:
        buffer += bit
        if buffer in code_to_symbol:
            result.append(code_to_symbol[buffer])
            buffer = ""
        elif not any(code.startswith(buffer) for code in code_to_symbol):
            raise ValueError("The encoded message contains an invalid code prefix.")
    if buffer:
        raise ValueError("The encoded message ends with an incomplete code.")
    return result


if __name__ == "__main__":
    sample_books = [
        {"id": "B1", "title": "Discrete Math", "author": "A1",
         "category": "Computing", "section": "S1"},
        {"id": "B2", "title": "Graph Theory", "author": "A2",
         "category": "Computing", "section": "S1"},
    ]
    sample_graph = build_library_graph(
        sample_books, ["U1", "U2"], [("U1", "B1"), ("U2", "B1"), ("U2", "B2")]
    )
    print("Graph analysis:", analyze_graph(sample_graph))
    print("Loan path:", calculate_shortest_path(
        sample_graph, ("student", "U1"), ("section", "S1")
    ))
    print("C5 self-complementary:", self_complementary_graph_example().number_of_edges())
    codes = build_huffman_codes({"B1": 8, "B2": 4, "B3": 2, "B4": 1})
    encoded = encode_message(["B1", "B2", "B4"], codes)
    print("Prefix codes:", codes, "round trip:", decode_message(encoded, codes))
    draw_graph(sample_graph)
    draw_hasse_diagram([1, 2, 4], [(1, 2), (2, 4)],
                       title="Divisibility order on {1, 2, 4}")
