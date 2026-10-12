"""Bounded local routing primitives for Gator-Path.

The router in this module never allocates or scores an action for every cell.
It scores the fixed ring-offset action table and the source node's local legal
edge table.  The resulting joint action space is therefore bounded by
``num_cell_actions * max_node_actions`` regardless of population size.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor

from .topology import PathTopology


def _require_long_matrix(name: str, value: Tensor) -> tuple[int, int]:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape [B, L]; got {tuple(value.shape)}"
        )
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long")
    batch_size, slot_count = value.shape
    if batch_size <= 0 or slot_count <= 0:
        raise ValueError(f"{name} dimensions B and L must be positive")
    return batch_size, slot_count


def _require_floating_logits(
    name: str,
    value: Tensor,
    *,
    prefix: tuple[int, int],
    action_count: int,
    device: torch.device,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected = (*prefix, action_count)
    if tuple(value.shape) != expected:
        raise ValueError(
            f"{name} must have shape [B, L, {action_count}]; "
            f"got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.device != device:
        raise ValueError(
            f"{name} must be on source device {device}; got {value.device}"
        )


def _contains_out_of_range(
    value: Tensor,
    *,
    lower: int,
    upper: int,
) -> bool:
    return bool(torch.any((value < lower) | (value >= upper)).item())


def _local_node_tables(
    topology: PathTopology,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return padded destinations, legal mask, and source-local degrees."""

    max_node_actions = max(
        len(destinations)
        for destinations in topology.legal_node_destinations
    )
    destinations = torch.zeros(
        topology.num_nodes,
        max_node_actions,
        dtype=torch.long,
        device=device,
    )
    legal = torch.zeros(
        topology.num_nodes,
        max_node_actions,
        dtype=torch.bool,
        device=device,
    )
    degrees = torch.empty(
        topology.num_nodes,
        dtype=torch.long,
        device=device,
    )
    for source_node, row in enumerate(
        topology.legal_node_destinations
    ):
        width = len(row)
        destinations[source_node, :width] = torch.tensor(
            row,
            dtype=torch.long,
            device=device,
        )
        legal[source_node, :width] = True
        degrees[source_node] = width
    return destinations, legal, degrees


@dataclass(frozen=True)
class LocalRouteResult:
    """Sparse destinations plus bounded local routing diagnostics.

    Destination and selected-action tensors have shape ``[B, L, K]``.
    Probability tensors use only fixed local action axes:

    * ``cell_probabilities``: ``[B, L, A_c]``
    * ``node_probabilities``: ``[B, L, A_n_max]``
    * ``joint_probabilities``: ``[B, L, A_c, A_n_max]``

    No result tensor contains an axis of length ``num_cells``.
    """

    destination_cell: Tensor
    destination_node: Tensor
    cell_action: Tensor
    node_action: Tensor
    joint_action: Tensor
    weights: Tensor
    selected_probability: Tensor
    cell_logits: Tensor
    node_logits: Tensor
    joint_logits: Tensor
    cell_probabilities: Tensor
    node_probabilities: Tensor
    joint_probabilities: Tensor
    node_legal_mask: Tensor

    @property
    def top_k(self) -> int:
        return self.destination_cell.shape[-1]


def route_local_actions(
    *,
    topology: PathTopology,
    source_cell: Tensor,
    source_node: Tensor,
    cell_logits: Tensor,
    node_logits: Tensor,
    top_k: int = 1,
    temperature: float = 1.0,
    straight_through_hard: bool = False,
    cell_destination_table: Tensor | None = None,
    validate_coordinates: bool = True,
) -> LocalRouteResult:
    """Select deterministic top-k routes from factorized local logits.

    ``node_logits`` uses a padded source-local action axis whose width is the
    maximum internal-node out-degree.  Actions beyond a source row's degree
    are masked before normalization and selection.

    Soft routing renormalizes retained top-k probability mass.  Hard routing
    uses a uniform k-hot value in the forward pass and the selected
    factorized probabilities in the backward pass.  In particular, hard
    top-1 has forward weight one while retaining gradients to both heads.
    """

    if not isinstance(topology, PathTopology):
        raise TypeError("topology must be a PathTopology")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not isinstance(temperature, Real) or isinstance(temperature, bool):
        raise TypeError("temperature must be a real number")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if type(straight_through_hard) is not bool:
        raise TypeError("straight_through_hard must be a boolean")
    if type(validate_coordinates) is not bool:
        raise TypeError("validate_coordinates must be a boolean")
    packet_shape = _require_long_matrix("source_cell", source_cell)
    if _require_long_matrix("source_node", source_node) != packet_shape:
        raise ValueError(
            "source_node must have the same [B, L] shape as source_cell"
        )
    if source_node.device != source_cell.device:
        raise ValueError(
            "source_node and source_cell must be on the same device"
        )
    if cell_destination_table is not None:
        expected_shape = (
            topology.num_cells,
            topology.num_cell_actions,
        )
        if (
            not isinstance(cell_destination_table, Tensor)
            or cell_destination_table.dtype != torch.long
            or tuple(cell_destination_table.shape) != expected_shape
        ):
            raise TypeError(
                "cell_destination_table must be a torch.long tensor with "
                f"shape {expected_shape}"
            )
        if cell_destination_table.device != source_cell.device:
            raise ValueError(
                "cell_destination_table and source_cell must share a device"
            )
        if _contains_out_of_range(
            cell_destination_table,
            lower=0,
            upper=topology.num_cells,
        ):
            raise ValueError(
                "cell_destination_table contains an invalid cell"
            )
    if validate_coordinates:
        if _contains_out_of_range(
            source_cell,
            lower=0,
            upper=topology.num_cells,
        ):
            raise ValueError(
                f"source_cell values must be in [0, {topology.num_cells})"
            )
        if _contains_out_of_range(
            source_node,
            lower=0,
            upper=topology.num_nodes,
        ):
            raise ValueError(
                f"source_node values must be in [0, {topology.num_nodes})"
            )

    node_destinations, node_legal_rows, node_degrees = (
        _local_node_tables(topology, device=source_cell.device)
    )
    max_node_actions = node_destinations.shape[-1]
    _require_floating_logits(
        "cell_logits",
        cell_logits,
        prefix=packet_shape,
        action_count=topology.num_cell_actions,
        device=source_cell.device,
    )
    _require_floating_logits(
        "node_logits",
        node_logits,
        prefix=packet_shape,
        action_count=max_node_actions,
        device=source_cell.device,
    )
    if cell_logits.dtype != node_logits.dtype:
        raise TypeError("cell_logits and node_logits must have the same dtype")

    packet_node_degrees = node_degrees[source_node]
    minimum_joint_count = min(
        topology.num_joint_actions(node)
        for node in range(topology.num_nodes)
    )
    if top_k > minimum_joint_count:
        raise ValueError(
            "top_k exceeds the legal local joint-action count for at "
            f"least one packet ({minimum_joint_count})"
        )

    scaled_cell_logits = cell_logits / temperature
    scaled_node_logits = node_logits / temperature
    node_legal_mask = node_legal_rows[source_node]
    masked_node_logits = scaled_node_logits.masked_fill(
        ~node_legal_mask,
        -torch.inf,
    )
    cell_probabilities = torch.softmax(scaled_cell_logits, dim=-1)
    node_probabilities = torch.softmax(masked_node_logits, dim=-1)

    joint_logits = (
        scaled_cell_logits.unsqueeze(-1)
        + masked_node_logits.unsqueeze(-2)
    )
    joint_probabilities = (
        cell_probabilities.unsqueeze(-1)
        * node_probabilities.unsqueeze(-2)
    )
    flat_probabilities = joint_probabilities.flatten(start_dim=-2)

    # Stable sorting makes ties choose the earliest cell-major local action.
    # The action table is tiny (currently at most 15 entries), so sorting the
    # complete bounded local table is preferable to backend-dependent topk
    # tie behavior.
    _, ordered_actions = torch.sort(
        flat_probabilities,
        dim=-1,
        descending=True,
        stable=True,
    )
    selected_rectangular = ordered_actions[..., :top_k]
    selected_probability = torch.gather(
        flat_probabilities,
        dim=-1,
        index=selected_rectangular,
    )
    cell_action = torch.div(
        selected_rectangular,
        max_node_actions,
        rounding_mode="floor",
    )
    node_action = selected_rectangular.remainder(max_node_actions)

    if cell_destination_table is None:
        cell_offsets = torch.tensor(
            topology.cell_offsets,
            dtype=torch.long,
            device=source_cell.device,
        )
        selected_offsets = cell_offsets[cell_action]
        destination_cell = (
            source_cell.unsqueeze(-1) + selected_offsets
        ).remainder(topology.num_cells)
    else:
        packet_destinations = cell_destination_table[source_cell]
        destination_cell = torch.gather(
            packet_destinations,
            dim=-1,
            index=cell_action,
        )
    packet_node_destinations = node_destinations[source_node]
    destination_node = torch.gather(
        packet_node_destinations,
        dim=-1,
        index=node_action,
    )
    joint_action = (
        cell_action * packet_node_degrees.unsqueeze(-1) + node_action
    )

    if straight_through_hard:
        hard_weights = torch.full_like(
            selected_probability,
            1.0 / top_k,
        )
        weights = (
            hard_weights
            + selected_probability
            - selected_probability.detach()
        )
    else:
        retained_mass = selected_probability.sum(
            dim=-1,
            keepdim=True,
        )
        weights = selected_probability / retained_mass.clamp_min(
            torch.finfo(selected_probability.dtype).tiny
        )

    return LocalRouteResult(
        destination_cell=destination_cell,
        destination_node=destination_node,
        cell_action=cell_action,
        node_action=node_action,
        joint_action=joint_action,
        weights=weights,
        selected_probability=selected_probability,
        cell_logits=cell_logits,
        node_logits=node_logits,
        joint_logits=joint_logits,
        cell_probabilities=cell_probabilities,
        node_probabilities=node_probabilities,
        joint_probabilities=joint_probabilities,
        node_legal_mask=node_legal_mask,
    )


__all__ = [
    "LocalRouteResult",
    "route_local_actions",
]
