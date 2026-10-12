"""Fixed graph topology and route-action helpers for Gator-Path.

The topology layer is deliberately independent of PyTorch.  It describes the
small, static action spaces that tensor routing code will consume later:

* cell actions are indices into a fixed ring-offset table;
* node actions are indices into the legal destinations of the source node;
* joint actions flatten those two factorized indices in cell-major order.

Joint action identifiers are local to a source node because the validated
internal graph has different out-degrees for different nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# This order is part of the routing contract.  In particular, action zero is
# always "stay", which gives future routers a stable no-movement action.
FIXED_RING_CELL_OFFSETS: tuple[int, ...] = (0, -1, 1, -2, 2)

# Rows are source nodes and columns are destination nodes.
VALIDATED_NODE_ADJACENCY: tuple[tuple[bool, ...], ...] = (
    (True, True, False, False),
    (False, True, True, True),
    (True, False, True, True),
    (True, True, False, True),
)


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def validate_cell_offsets(
    offsets: Sequence[int],
) -> tuple[int, ...]:
    """Return a canonical cell-offset tuple after validating its action space."""

    if isinstance(offsets, (str, bytes)) or not isinstance(offsets, Sequence):
        raise TypeError("cell_offsets must be a sequence of integers")
    canonical = tuple(
        _require_int(offset, f"cell_offsets[{index}]")
        for index, offset in enumerate(offsets)
    )
    if not canonical:
        raise ValueError("cell_offsets must contain at least one action")
    if len(set(canonical)) != len(canonical):
        raise ValueError("cell_offsets must not contain duplicate actions")
    if canonical.count(0) != 1:
        raise ValueError("cell_offsets must contain exactly one stay action (0)")
    return canonical


def validate_node_adjacency(
    adjacency: Sequence[Sequence[bool]],
) -> tuple[tuple[bool, ...], ...]:
    """Return an immutable, validated square node-adjacency matrix."""

    if isinstance(adjacency, (str, bytes)) or not isinstance(
        adjacency, Sequence
    ):
        raise TypeError("node_adjacency must be a sequence of rows")
    rows = tuple(adjacency)
    if not rows:
        raise ValueError("node_adjacency must contain at least one node")

    node_count = len(rows)
    canonical_rows: list[tuple[bool, ...]] = []
    for source_node, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(
                f"node_adjacency[{source_node}] must be a sequence of booleans"
            )
        canonical_row = tuple(row)
        if len(canonical_row) != node_count:
            raise ValueError("node_adjacency must be square")
        for destination_node, connected in enumerate(canonical_row):
            if type(connected) is not bool:
                raise TypeError(
                    "node_adjacency"
                    f"[{source_node}][{destination_node}] must be a boolean"
                )
        if not any(canonical_row):
            raise ValueError(
                f"source node {source_node} has no legal destination"
            )
        canonical_rows.append(canonical_row)
    return tuple(canonical_rows)


def legal_destinations_from_adjacency(
    adjacency: Sequence[Sequence[bool]],
) -> tuple[tuple[int, ...], ...]:
    """Build source-indexed legal destination tables from an adjacency matrix."""

    canonical = validate_node_adjacency(adjacency)
    return tuple(
        tuple(
            destination
            for destination, connected in enumerate(row)
            if connected
        )
        for row in canonical
    )


@dataclass(frozen=True, slots=True)
class PathAction:
    """A legal action expressed in graph coordinates."""

    cell_offset: int
    destination_node: int

    def __post_init__(self) -> None:
        _require_int(self.cell_offset, "cell_offset")
        destination_node = _require_int(
            self.destination_node, "destination_node"
        )
        if destination_node < 0:
            raise ValueError("destination_node must be non-negative")


@dataclass(frozen=True, slots=True)
class FactorizedAction:
    """Cell-head and source-local node-head action indices."""

    cell_action: int
    node_action: int

    def __post_init__(self) -> None:
        cell_action = _require_int(self.cell_action, "cell_action")
        node_action = _require_int(self.node_action, "node_action")
        if cell_action < 0:
            raise ValueError("cell_action must be non-negative")
        if node_action < 0:
            raise ValueError("node_action must be non-negative")


def encode_joint_indices(
    action: FactorizedAction,
    *,
    node_action_count: int,
) -> int:
    """Flatten factorized indices in cell-major order.

    Bounds that depend on the number of cell actions are checked by
    :class:`PathTopology`; this low-level helper validates the node component.
    """

    if not isinstance(action, FactorizedAction):
        raise TypeError("action must be a FactorizedAction")
    node_action_count = _require_int(
        node_action_count, "node_action_count"
    )
    if node_action_count <= 0:
        raise ValueError("node_action_count must be positive")
    if action.node_action >= node_action_count:
        raise ValueError(
            f"node_action must be less than {node_action_count}"
        )
    return action.cell_action * node_action_count + action.node_action


def decode_joint_index(
    action_id: int,
    *,
    cell_action_count: int,
    node_action_count: int,
) -> FactorizedAction:
    """Unflatten a joint action identifier into its factorized indices."""

    action_id = _require_int(action_id, "action_id")
    cell_action_count = _require_int(
        cell_action_count, "cell_action_count"
    )
    node_action_count = _require_int(
        node_action_count, "node_action_count"
    )
    if cell_action_count <= 0:
        raise ValueError("cell_action_count must be positive")
    if node_action_count <= 0:
        raise ValueError("node_action_count must be positive")
    joint_action_count = cell_action_count * node_action_count
    if not 0 <= action_id < joint_action_count:
        raise ValueError(
            f"action_id must be in [0, {joint_action_count})"
        )
    cell_action, node_action = divmod(action_id, node_action_count)
    return FactorizedAction(
        cell_action=cell_action,
        node_action=node_action,
    )


@dataclass(frozen=True, slots=True)
class PathTopology:
    """Validated ring cells combined with a fixed internal node graph."""

    num_cells: int
    cell_offsets: tuple[int, ...] = FIXED_RING_CELL_OFFSETS
    node_adjacency: tuple[tuple[bool, ...], ...] = (
        VALIDATED_NODE_ADJACENCY
    )
    cell_destination_table: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False
    )
    legal_node_destinations: tuple[tuple[int, ...], ...] = field(
        init=False
    )
    joint_action_tables: tuple[tuple[PathAction, ...], ...] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        num_cells = _require_int(self.num_cells, "num_cells")
        if num_cells <= 0:
            raise ValueError("num_cells must be positive")
        cell_offsets = validate_cell_offsets(self.cell_offsets)
        node_adjacency = validate_node_adjacency(self.node_adjacency)
        legal_node_destinations = legal_destinations_from_adjacency(
            node_adjacency
        )

        object.__setattr__(self, "num_cells", num_cells)
        object.__setattr__(self, "cell_offsets", cell_offsets)
        object.__setattr__(self, "node_adjacency", node_adjacency)
        object.__setattr__(
            self,
            "cell_destination_table",
            tuple(
                tuple(
                    (source_cell + offset) % num_cells
                    for offset in cell_offsets
                )
                for source_cell in range(num_cells)
            ),
        )
        object.__setattr__(
            self, "legal_node_destinations", legal_node_destinations
        )
        object.__setattr__(
            self,
            "joint_action_tables",
            tuple(
                tuple(
                    PathAction(offset, destination_node)
                    for offset in cell_offsets
                    for destination_node in destinations
                )
                for destinations in legal_node_destinations
            ),
        )

    @property
    def num_nodes(self) -> int:
        return len(self.node_adjacency)

    @property
    def num_cell_actions(self) -> int:
        return len(self.cell_offsets)

    def _validate_source_cell(self, source_cell: int) -> int:
        source_cell = _require_int(source_cell, "source_cell")
        if not 0 <= source_cell < self.num_cells:
            raise ValueError(
                f"source_cell must be in [0, {self.num_cells})"
            )
        return source_cell

    def _validate_source_node(self, source_node: int) -> int:
        source_node = _require_int(source_node, "source_node")
        if not 0 <= source_node < self.num_nodes:
            raise ValueError(
                f"source_node must be in [0, {self.num_nodes})"
            )
        return source_node

    def num_node_actions(self, source_node: int) -> int:
        source_node = self._validate_source_node(source_node)
        return len(self.legal_node_destinations[source_node])

    def num_joint_actions(self, source_node: int) -> int:
        return self.num_cell_actions * self.num_node_actions(source_node)

    def encode_cell_action(self, cell_offset: int) -> int:
        cell_offset = _require_int(cell_offset, "cell_offset")
        try:
            return self.cell_offsets.index(cell_offset)
        except ValueError as error:
            raise ValueError(
                f"cell offset {cell_offset} is not a legal action"
            ) from error

    def decode_cell_action(self, cell_action: int) -> int:
        cell_action = _require_int(cell_action, "cell_action")
        if not 0 <= cell_action < self.num_cell_actions:
            raise ValueError(
                f"cell_action must be in [0, {self.num_cell_actions})"
            )
        return self.cell_offsets[cell_action]

    def encode_node_action(
        self,
        source_node: int,
        destination_node: int,
    ) -> int:
        source_node = self._validate_source_node(source_node)
        destination_node = _require_int(
            destination_node, "destination_node"
        )
        destinations = self.legal_node_destinations[source_node]
        try:
            return destinations.index(destination_node)
        except ValueError as error:
            raise ValueError(
                f"node edge {source_node}->{destination_node} is not legal"
            ) from error

    def decode_node_action(
        self,
        source_node: int,
        node_action: int,
    ) -> int:
        source_node = self._validate_source_node(source_node)
        node_action = _require_int(node_action, "node_action")
        destinations = self.legal_node_destinations[source_node]
        if not 0 <= node_action < len(destinations):
            raise ValueError(
                f"node_action must be in [0, {len(destinations)}) "
                f"for source node {source_node}"
            )
        return destinations[node_action]

    def encode_factorized_action(
        self,
        source_node: int,
        action: PathAction,
    ) -> FactorizedAction:
        """Encode graph coordinates for the two independent route heads."""

        if not isinstance(action, PathAction):
            raise TypeError("action must be a PathAction")
        return FactorizedAction(
            cell_action=self.encode_cell_action(action.cell_offset),
            node_action=self.encode_node_action(
                source_node, action.destination_node
            ),
        )

    def decode_factorized_action(
        self,
        source_node: int,
        action: FactorizedAction,
    ) -> PathAction:
        """Decode route-head indices into graph coordinates."""

        if not isinstance(action, FactorizedAction):
            raise TypeError("action must be a FactorizedAction")
        return PathAction(
            cell_offset=self.decode_cell_action(action.cell_action),
            destination_node=self.decode_node_action(
                source_node, action.node_action
            ),
        )

    def encode_joint_action(
        self,
        source_node: int,
        action: PathAction,
    ) -> int:
        """Encode a legal graph action as a source-local joint identifier."""

        source_node = self._validate_source_node(source_node)
        factorized = self.encode_factorized_action(source_node, action)
        return encode_joint_indices(
            factorized,
            node_action_count=self.num_node_actions(source_node),
        )

    def decode_joint_action(
        self,
        source_node: int,
        action_id: int,
    ) -> PathAction:
        """Decode a source-local joint identifier into graph coordinates."""

        source_node = self._validate_source_node(source_node)
        factorized = decode_joint_index(
            action_id,
            cell_action_count=self.num_cell_actions,
            node_action_count=self.num_node_actions(source_node),
        )
        return self.decode_factorized_action(source_node, factorized)

    def destination_cell(
        self,
        source_cell: int,
        cell_offset: int,
    ) -> int:
        """Resolve one legal ring action with wraparound."""

        source_cell = self._validate_source_cell(source_cell)
        cell_action = self.encode_cell_action(cell_offset)
        return self.cell_destination_table[source_cell][cell_action]

    def resolve_action(
        self,
        source_cell: int,
        source_node: int,
        action: PathAction,
    ) -> tuple[int, int]:
        """Validate and resolve a complete action to ``(cell, node)``."""

        if not isinstance(action, PathAction):
            raise TypeError("action must be a PathAction")
        source_node = self._validate_source_node(source_node)
        self.encode_node_action(source_node, action.destination_node)
        return (
            self.destination_cell(source_cell, action.cell_offset),
            action.destination_node,
        )


def fixed_ring_topology(num_cells: int) -> PathTopology:
    """Construct the canonical degree-four Gator-Path ring."""

    return PathTopology(
        num_cells=num_cells,
        cell_offsets=FIXED_RING_CELL_OFFSETS,
        node_adjacency=VALIDATED_NODE_ADJACENCY,
    )


__all__ = [
    "FIXED_RING_CELL_OFFSETS",
    "VALIDATED_NODE_ADJACENCY",
    "FactorizedAction",
    "PathAction",
    "PathTopology",
    "decode_joint_index",
    "encode_joint_indices",
    "fixed_ring_topology",
    "legal_destinations_from_adjacency",
    "validate_cell_offsets",
    "validate_node_adjacency",
]
