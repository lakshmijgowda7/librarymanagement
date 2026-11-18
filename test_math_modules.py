"""Focused standard-library tests for the four mathematical modules."""

import importlib.util
import unittest

from counting_module import (
    binomial_expansion,
    calculate_combinations,
    calculate_permutations,
    pigeonhole_analysis,
    product_rule,
    sum_rule,
    verify_sum_induction_example,
    well_ordering_demo,
)
from graph_module import (
    analyze_graph,
    analyze_walk,
    build_block_graph,
    build_cutpoint_graph,
    build_huffman_codes,
    build_library_graph,
    calculate_shortest_path,
    create_subgraph,
    decode_message,
    encode_message,
    self_complementary_graph_example,
)
from recurrence_module import (
    generate_first_order_sequence,
    generate_inventory_sequence,
    generate_second_order_sequence,
    solve_first_order_closed_form,
    solve_first_order_recurrence,
    solve_second_order_characteristic,
)
from relations_module import (
    analyze_function,
    compose_functions,
    create_borrowing_matrix,
    divisibility_hasse_diagram,
    equivalence_classes,
    inverse_function,
    zero_one_matrix,
)


HAS_NETWORKX = importlib.util.find_spec("networkx") is not None


class CountingTests(unittest.TestCase):
    def test_rules_and_selection_counts(self):
        self.assertEqual(sum_rule(3, 4), 7)
        self.assertEqual(product_rule(3, 4), 12)
        self.assertEqual(calculate_combinations(5, 2), 10)
        self.assertEqual(calculate_permutations(5, 2), 20)

    def test_binomial_induction_and_pigeonhole(self):
        self.assertEqual(binomial_expansion(1, 2, 3)["rhs"], 27)
        self.assertTrue(binomial_expansion(0.1, 0.2, 3)["verified"])
        self.assertTrue(verify_sum_induction_example(100)["verified"])
        self.assertEqual(pigeonhole_analysis(11, 4)[
            "guaranteed_minimum_in_some_box"], 3)
        result = pigeonhole_analysis(
            ["B1", "B2", "B3"], ["Shelf A", "Shelf B"]
        )
        self.assertEqual(result["maximum_in_allocated_box"], 2)

    def test_well_ordering(self):
        self.assertEqual(well_ordering_demo([8, 2, 5]), 2)
        with self.assertRaises(ValueError):
            well_ordering_demo([])


class RelationTests(unittest.TestCase):
    def test_borrowing_matrix(self):
        result = create_borrowing_matrix(
            ["S1", "S2"], ["B1", "B2"],
            [("S1", "B1"), {"student_id": "S2", "book_id": "B2"}],
        )
        self.assertEqual(result["matrix"], [[1, 0], [0, 1]])
        self.assertEqual(zero_one_matrix(
            ["S1", "S2"], ["B1", "B2"], [("S1", "B2")]
        )["matrix"], [[0, 1], [0, 0]])

    def test_function_properties_composition_inverse(self):
        result = analyze_function({1, 2}, {"a", "b"}, {1: "a", 2: "b"})
        self.assertTrue(result["is_bijective"])
        self.assertEqual(compose_functions({1: "a", 2: "b"},
                                           {"a": 10, "b": 20}),
                         {1: 10, 2: 20})
        self.assertEqual(inverse_function({1: "a", 2: "b"}), {"a": 1, "b": 2})

    def test_equivalence_classes_and_hasse_covers(self):
        relation = {(left, right) for left in range(4) for right in range(4)
                    if left % 2 == right % 2}
        classes = equivalence_classes(range(4), relation)
        self.assertEqual({frozenset(group) for group in classes},
                         {frozenset({0, 2}), frozenset({1, 3})})
        hasse = divisibility_hasse_diagram(12)
        self.assertEqual(set(hasse["cover_edges"]),
                         {(1, 2), (1, 3), (2, 4), (2, 6), (3, 6), (4, 12), (6, 12)})


class RecurrenceTests(unittest.TestCase):
    def test_first_order_and_inventory(self):
        self.assertEqual(generate_first_order_sequence(10, [2, -1], 2), [10, 12, 11])
        self.assertEqual(solve_first_order_recurrence(1, 2, 1, 3), [1, 3, 7, 15])
        self.assertEqual(solve_first_order_closed_form(1, 2, 1, 3),
                         [1, 3, 7, 15])
        self.assertEqual(solve_first_order_closed_form(1, 1, 2, 3),
                         [1, 3, 5, 7])
        self.assertEqual(generate_inventory_sequence(10, [3, 0], [2, 4]),
                         [10, 11, 7])

    def test_second_order_distinct_and_repeated_roots(self):
        expected = [0, 1, 1, 2, 3, 5, 8]
        self.assertEqual(generate_second_order_sequence(0, 1, 5), expected)
        self.assertEqual(solve_second_order_characteristic(0, 1, 1, 1, 5)[
            "sequence"], expected)
        repeated = solve_second_order_characteristic(2, 3, 2, -1, 4)
        self.assertEqual(repeated["root_case"], "repeated")
        self.assertEqual(repeated["sequence"], [2, 3, 4, 5, 6, 7])
        zero_root = solve_second_order_characteristic(0, 1, 0, 0, 2)
        self.assertEqual(zero_root["sequence"], [0, 1, 0, 0])


@unittest.skipUnless(HAS_NETWORKX, "Install NetworkX to run graph tests.")
class GraphTests(unittest.TestCase):
    def test_library_graph_paths_and_analysis(self):
        graph = build_library_graph(
            [{"id": "B1", "section": "S1"}], ["U1", "U2"],
            [("U1", "B1"), ("U2", "B1")],
        )
        path = calculate_shortest_path(
            graph, ("student", "U1"), ("section", "S1")
        )
        self.assertEqual(path["distance"], 2)
        self.assertEqual(analyze_graph(graph)["node_count"], 4)
        walk = analyze_walk(graph, [
            ("student", "U1"), ("book", "B1"), ("section", "S1")
        ])
        self.assertTrue(walk["is_path"])
        self.assertEqual(create_subgraph(graph, walk["vertices"]).number_of_nodes(), 3)

    def test_complement_and_block_graphs(self):
        self.assertEqual(self_complementary_graph_example().number_of_edges(), 5)
        graph = build_library_graph(["B1", "B2"], ["U1", "U2"], [
            ("U1", "B1"), ("U1", "B2"), ("U2", "B2")
        ])
        analysis = analyze_graph(graph)
        self.assertTrue(analysis["cut_points"])
        self.assertGreater(build_block_graph(graph).number_of_nodes(), 0)
        self.assertGreaterEqual(build_cutpoint_graph(graph).number_of_nodes(), 0)

    def test_prefix_code_round_trip(self):
        codes = build_huffman_codes({"A": 5, "B": 2, "C": 1})
        message = ["A", "B", "C", "A"]
        self.assertEqual(decode_message(encode_message(message, codes), codes), message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
