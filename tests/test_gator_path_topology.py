import unittest

from gator_path.topology import (
    FIXED_RING_CELL_OFFSETS,
    VALIDATED_NODE_ADJACENCY,
    FactorizedAction,
    PathAction,
    PathTopology,
    decode_joint_index,
    encode_joint_indices,
    fixed_ring_topology,
    legal_destinations_from_adjacency,
    validate_cell_offsets,
    validate_node_adjacency,
)


class FixedTopologyTests(unittest.TestCase):
    def test_fixed_ring_offsets_include_stay_in_stable_action_order(self) -> None:
        self.assertEqual(FIXED_RING_CELL_OFFSETS, (0, -1, 1, -2, 2))

    def test_ring_destination_table_wraps_in_both_directions(self) -> None:
        topology = fixed_ring_topology(10)

        self.assertEqual(
            topology.cell_destination_table[0],
            (0, 9, 1, 8, 2),
        )
        self.assertEqual(
            topology.cell_destination_table[9],
            (9, 8, 0, 7, 1),
        )
        self.assertEqual(topology.destination_cell(0, -1), 9)
        self.assertEqual(topology.destination_cell(9, 2), 1)

    def test_small_rings_keep_actions_distinct_even_when_destinations_alias(
        self,
    ) -> None:
        topology = fixed_ring_topology(1)

        self.assertEqual(topology.num_cell_actions, 5)
        self.assertEqual(topology.cell_destination_table[0], (0, 0, 0, 0, 0))

    def test_validated_adjacency_builds_expected_legal_destinations(self) -> None:
        topology = fixed_ring_topology(10)

        self.assertEqual(
            topology.node_adjacency,
            VALIDATED_NODE_ADJACENCY,
        )
        self.assertEqual(
            topology.legal_node_destinations,
            ((0, 1), (1, 2, 3), (0, 2, 3), (0, 1, 3)),
        )
        self.assertEqual(
            legal_destinations_from_adjacency(VALIDATED_NODE_ADJACENCY),
            topology.legal_node_destinations,
        )

    def test_source_local_joint_tables_have_only_legal_actions(self) -> None:
        topology = fixed_ring_topology(10)

        self.assertEqual(
            tuple(topology.num_joint_actions(node) for node in range(4)),
            (10, 15, 15, 15),
        )
        for source_node, actions in enumerate(topology.joint_action_tables):
            expected = {
                PathAction(offset, destination)
                for offset in FIXED_RING_CELL_OFFSETS
                for destination in topology.legal_node_destinations[
                    source_node
                ]
            }
            self.assertEqual(set(actions), expected)
            self.assertEqual(len(actions), len(expected))


class ActionEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = fixed_ring_topology(10)

    def test_factorized_encoding_uses_head_local_indices(self) -> None:
        action = PathAction(cell_offset=1, destination_node=3)

        encoded = self.topology.encode_factorized_action(1, action)

        self.assertEqual(encoded, FactorizedAction(cell_action=2, node_action=2))
        self.assertEqual(
            self.topology.decode_factorized_action(1, encoded),
            action,
        )

    def test_joint_encoding_is_cell_major(self) -> None:
        action = PathAction(cell_offset=1, destination_node=3)

        action_id = self.topology.encode_joint_action(1, action)

        # Source node 1 has three node actions.  Offset +1 is cell action 2.
        self.assertEqual(action_id, 2 * 3 + 2)
        self.assertEqual(
            self.topology.decode_joint_action(1, action_id),
            action,
        )

    def test_every_legal_action_round_trips_in_both_encodings(self) -> None:
        for source_node in range(self.topology.num_nodes):
            for expected_id, action in enumerate(
                self.topology.joint_action_tables[source_node]
            ):
                factorized = self.topology.encode_factorized_action(
                    source_node, action
                )
                self.assertEqual(
                    self.topology.decode_factorized_action(
                        source_node, factorized
                    ),
                    action,
                )
                self.assertEqual(
                    self.topology.encode_joint_action(source_node, action),
                    expected_id,
                )
                self.assertEqual(
                    self.topology.decode_joint_action(
                        source_node, expected_id
                    ),
                    action,
                )

    def test_low_level_joint_index_helpers_round_trip(self) -> None:
        action = FactorizedAction(cell_action=4, node_action=1)

        action_id = encode_joint_indices(action, node_action_count=3)

        self.assertEqual(action_id, 13)
        self.assertEqual(
            decode_joint_index(
                action_id,
                cell_action_count=5,
                node_action_count=3,
            ),
            action,
        )

    def test_resolve_action_validates_edge_and_wraps_cell(self) -> None:
        self.assertEqual(
            self.topology.resolve_action(
                source_cell=9,
                source_node=0,
                action=PathAction(cell_offset=2, destination_node=1),
            ),
            (1, 1),
        )

    def test_illegal_node_edge_is_rejected(self) -> None:
        action = PathAction(cell_offset=0, destination_node=2)

        with self.assertRaisesRegex(ValueError, "0->2"):
            self.topology.encode_factorized_action(0, action)
        with self.assertRaisesRegex(ValueError, "0->2"):
            self.topology.resolve_action(0, 0, action)

    def test_out_of_range_action_indices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 5\)"):
            self.topology.decode_cell_action(5)
        with self.assertRaisesRegex(ValueError, r"\[0, 2\)"):
            self.topology.decode_node_action(0, 2)
        with self.assertRaisesRegex(ValueError, r"\[0, 10\)"):
            self.topology.decode_joint_action(0, 10)


class TopologyValidationTests(unittest.TestCase):
    def test_sequences_are_canonicalized_to_immutable_tuples(self) -> None:
        topology = PathTopology(
            num_cells=3,
            cell_offsets=[0, -1, 1],
            node_adjacency=[[True, False], [True, True]],
        )

        self.assertEqual(topology.cell_offsets, (0, -1, 1))
        self.assertEqual(
            topology.node_adjacency,
            ((True, False), (True, True)),
        )
        self.assertEqual(topology.legal_node_destinations, ((0,), (0, 1)))

    def test_cell_offset_validation_rejects_missing_stay_and_duplicates(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "stay"):
            validate_cell_offsets((-1, 1))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_cell_offsets((0, 1, 1))

    def test_adjacency_validation_rejects_non_square_or_dead_end_graphs(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "square"):
            validate_node_adjacency(((True, False),))
        with self.assertRaisesRegex(ValueError, "no legal destination"):
            validate_node_adjacency(((True, False), (False, False)))

    def test_adjacency_requires_actual_booleans(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            validate_node_adjacency(((1,),))

    def test_counts_and_coordinates_require_plain_integers(self) -> None:
        with self.assertRaisesRegex(TypeError, "num_cells"):
            fixed_ring_topology(True)
        with self.assertRaisesRegex(TypeError, "source_cell"):
            fixed_ring_topology(3).destination_cell(False, 0)

    def test_invalid_topology_coordinates_are_rejected(self) -> None:
        topology = fixed_ring_topology(3)

        with self.assertRaisesRegex(ValueError, r"\[0, 3\)"):
            topology.destination_cell(3, 0)
        with self.assertRaisesRegex(ValueError, r"\[0, 4\)"):
            topology.num_node_actions(4)
        with self.assertRaisesRegex(ValueError, "not a legal action"):
            topology.destination_cell(0, 3)


if __name__ == "__main__":
    unittest.main()
