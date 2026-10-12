"""Persistent receiver-specific node memory for carried Gator packets.

The dense Arrow reference rebuilds every receiver's node memory from every
writer at every relay.  This module stores the sufficient statistics of that
memory and updates them from packet departures and arrivals instead.

The update work is proportional to ``batch * receivers * changed_packets``.
It does not materialize a ``batch * receiver * writer * cell * node`` tensor.
The persistent state is still dense in cell/node locations for the correctness
milestone; a later packed state can omit empty locations without changing the
update equations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class PersistentRelativeMemoryState:
    """Carried normalized-memory sufficient statistics.

    ``numerator`` has shape ``[B, I, C, N, D]`` and ``denominator`` has shape
    ``[B, I, C, N]``.  ``I`` is the receiver slot coordinate.
    """

    numerator: Tensor
    denominator: Tensor

    @property
    def value(self) -> Tensor:
        safe = torch.where(
            self.denominator > 0,
            self.denominator,
            torch.ones_like(self.denominator),
        )
        value = self.numerator / safe.unsqueeze(-1)
        return torch.where(
            self.denominator.unsqueeze(-1) > 0,
            value,
            torch.zeros_like(value),
        )


def relative_code(
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return Arrow's fixed four-channel receiver/writer relation code."""

    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    coordinate = torch.arange(length, device=device, dtype=dtype)
    displacement = (
        coordinate[:, None] - coordinate[None, :]
    ) / max(length - 1, 1)
    return torch.stack(
        (
            displacement,
            displacement.abs(),
            (math.pi * displacement).sin(),
            (math.pi * displacement).cos(),
        ),
        dim=-1,
    )


def _validate_packets(
    payload: Tensor,
    cell_id: Tensor,
    node_id: Tensor,
    write_gate: Tensor,
    relation_bias: Tensor,
) -> tuple[int, int, int, int, int]:
    if payload.ndim != 3:
        raise ValueError("payload must have shape [B, L, D]")
    batch, length, width = payload.shape
    expected = (batch, length)
    for name, value in (
        ("cell_id", cell_id),
        ("node_id", node_id),
        ("write_gate", write_gate),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape [B, L]")
        if value.device != payload.device:
            raise ValueError(f"{name} must share payload's device")
    if cell_id.dtype != torch.long or node_id.dtype != torch.long:
        raise TypeError("cell_id and node_id must have dtype torch.long")
    if write_gate.dtype != payload.dtype:
        raise TypeError("write_gate and payload must share dtype")
    if relation_bias.ndim != 3 or relation_bias.shape[-1] != 4:
        raise ValueError("relation_bias must have shape [C, N, 4]")
    if relation_bias.device != payload.device:
        raise ValueError("relation_bias must share payload's device")
    if relation_bias.dtype != payload.dtype:
        raise TypeError("relation_bias and payload must share dtype")
    cells, nodes, _ = relation_bias.shape
    if bool(((cell_id < 0) | (cell_id >= cells)).any()):
        raise ValueError("cell_id contains an invalid cell")
    if bool(((node_id < 0) | (node_id >= nodes)).any()):
        raise ValueError("node_id contains an invalid node")
    if bool((write_gate < 0).any()):
        raise ValueError("write_gate cannot be negative")
    return batch, length, cells, nodes, width


def empty_relative_memory(
    *,
    batch: int,
    length: int,
    cells: int,
    nodes: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> PersistentRelativeMemoryState:
    """Create an empty carried memory state."""

    dimensions = {
        "batch": batch,
        "length": length,
        "cells": cells,
        "nodes": nodes,
        "width": width,
    }
    for name, value in dimensions.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return PersistentRelativeMemoryState(
        numerator=torch.zeros(
            batch,
            length,
            cells,
            nodes,
            width,
            device=device,
            dtype=dtype,
        ),
        denominator=torch.zeros(
            batch,
            length,
            cells,
            nodes,
            device=device,
            dtype=dtype,
        ),
    )


def _selected_relative_weights(
    *,
    cell_id: Tensor,
    node_id: Tensor,
    write_gate: Tensor,
    relation_bias: Tensor,
) -> Tensor:
    """Return ``[B, receiver, writer]`` weights for selected locations."""

    batch, length = cell_id.shape
    relations = relative_code(
        length,
        device=relation_bias.device,
        dtype=relation_bias.dtype,
    )
    selected_bias = relation_bias[cell_id, node_id]
    score = torch.einsum(
        "ijr,bjr->bij",
        relations,
        selected_bias,
    )

    # A fixed receiver/location shift makes exponentiation stable while
    # leaving each normalized memory value unchanged.  Unlike a dynamic
    # per-packet softmax maximum, this scale remains valid across incremental
    # departures and arrivals.
    all_score = torch.einsum(
        "ijr,cnr->ijcn",
        relations,
        relation_bias,
    )
    location_max = all_score.amax(dim=1)
    selected_max = location_max[:, cell_id, node_id].permute(1, 0, 2)
    return (
        (score - selected_max).exp()
        * write_gate[:, None, :]
    )


def _apply_packet_contributions(
    state: PersistentRelativeMemoryState,
    *,
    payload: Tensor,
    cell_id: Tensor,
    node_id: Tensor,
    write_gate: Tensor,
    relation_bias: Tensor,
    sign: float,
    active: Tensor | None,
) -> PersistentRelativeMemoryState:
    batch, length, cells, nodes, width = _validate_packets(
        payload,
        cell_id,
        node_id,
        write_gate,
        relation_bias,
    )
    expected_numerator = (batch, length, cells, nodes, width)
    expected_denominator = (batch, length, cells, nodes)
    if tuple(state.numerator.shape) != expected_numerator:
        raise ValueError("state numerator shape does not match packets")
    if tuple(state.denominator.shape) != expected_denominator:
        raise ValueError("state denominator shape does not match packets")
    if active is not None:
        if active.dtype != torch.bool or tuple(active.shape) != (
            batch,
            length,
        ):
            raise TypeError("active must be boolean with shape [B, L]")
        if active.device != payload.device:
            raise ValueError("active must share payload's device")
        write_gate = write_gate * active.to(write_gate.dtype)

    weight = _selected_relative_weights(
        cell_id=cell_id,
        node_id=node_id,
        write_gate=write_gate,
        relation_bias=relation_bias,
    )
    location = cell_id * nodes + node_id
    flat_numerator = state.numerator.reshape(
        batch,
        length,
        cells * nodes,
        width,
    )
    flat_denominator = state.denominator.reshape(
        batch,
        length,
        cells * nodes,
    )
    numerator_index = location[:, None, :, None].expand(
        batch,
        length,
        length,
        width,
    )
    denominator_index = location[:, None, :].expand(
        batch,
        length,
        length,
    )
    contribution = (
        sign
        * weight.unsqueeze(-1)
        * payload[:, None]
    )
    flat_numerator = flat_numerator.scatter_add(
        2,
        numerator_index,
        contribution,
    )
    flat_denominator = flat_denominator.scatter_add(
        2,
        denominator_index,
        sign * weight,
    )
    return PersistentRelativeMemoryState(
        numerator=flat_numerator.reshape(expected_numerator),
        denominator=flat_denominator.reshape(expected_denominator),
    )


def build_relative_memory(
    *,
    payload: Tensor,
    cell_id: Tensor,
    node_id: Tensor,
    write_gate: Tensor,
    relation_bias: Tensor,
    active: Tensor | None = None,
) -> PersistentRelativeMemoryState:
    """Build initial memory once from the current packet population."""

    batch, length, cells, nodes, width = _validate_packets(
        payload,
        cell_id,
        node_id,
        write_gate,
        relation_bias,
    )
    state = empty_relative_memory(
        batch=batch,
        length=length,
        cells=cells,
        nodes=nodes,
        width=width,
        device=payload.device,
        dtype=payload.dtype,
    )
    return _apply_packet_contributions(
        state,
        payload=payload,
        cell_id=cell_id,
        node_id=node_id,
        write_gate=write_gate,
        relation_bias=relation_bias,
        sign=1.0,
        active=active,
    )


def update_relative_memory(
    state: PersistentRelativeMemoryState,
    *,
    old_payload: Tensor,
    old_cell_id: Tensor,
    old_node_id: Tensor,
    old_write_gate: Tensor,
    new_payload: Tensor,
    new_cell_id: Tensor,
    new_node_id: Tensor,
    new_write_gate: Tensor,
    relation_bias: Tensor,
    active: Tensor | None = None,
) -> PersistentRelativeMemoryState:
    """Subtract departing packet state and add carried arrivals."""

    without_departures = _apply_packet_contributions(
        state,
        payload=old_payload,
        cell_id=old_cell_id,
        node_id=old_node_id,
        write_gate=old_write_gate,
        relation_bias=relation_bias,
        sign=-1.0,
        active=active,
    )
    return _apply_packet_contributions(
        without_departures,
        payload=new_payload,
        cell_id=new_cell_id,
        node_id=new_node_id,
        write_gate=new_write_gate,
        relation_bias=relation_bias,
        sign=1.0,
        active=active,
    )


def _validate_path_packets(
    payload: Tensor,
    position_id: Tensor,
    cell_id: Tensor,
    node_id: Tensor,
    write_gate: Tensor,
    relation_bias: Tensor,
    receiver_length: int,
) -> tuple[int, int, int, int, int]:
    """Validate flattened cooperative traces with repeated slot positions."""

    if payload.ndim != 3:
        raise ValueError("payload must have shape [B, P, D]")
    batch, packets, width = payload.shape
    expected = (batch, packets)
    for name, value in (
        ("position_id", position_id),
        ("cell_id", cell_id),
        ("node_id", node_id),
        ("write_gate", write_gate),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape [B, P]")
        if value.device != payload.device:
            raise ValueError(f"{name} must share payload's device")
    for name, value in (
        ("position_id", position_id),
        ("cell_id", cell_id),
        ("node_id", node_id),
    ):
        if value.dtype != torch.long:
            raise TypeError(f"{name} must have dtype torch.long")
    if write_gate.dtype != payload.dtype:
        raise TypeError("write_gate and payload must share dtype")
    if (
        isinstance(receiver_length, bool)
        or not isinstance(receiver_length, int)
        or receiver_length <= 0
    ):
        raise ValueError("receiver_length must be a positive integer")
    if bool(((position_id < 0) | (position_id >= receiver_length)).any()):
        raise ValueError("position_id contains an invalid receiver position")
    if relation_bias.ndim != 3 or relation_bias.shape[-1] != 4:
        raise ValueError("relation_bias must have shape [C, N, 4]")
    if relation_bias.device != payload.device:
        raise ValueError("relation_bias must share payload's device")
    if relation_bias.dtype != payload.dtype:
        raise TypeError("relation_bias and payload must share dtype")
    cells, nodes, _ = relation_bias.shape
    if bool(((cell_id < 0) | (cell_id >= cells)).any()):
        raise ValueError("cell_id contains an invalid cell")
    if bool(((node_id < 0) | (node_id >= nodes)).any()):
        raise ValueError("node_id contains an invalid node")
    if bool((write_gate < 0).any()):
        raise ValueError("write_gate cannot be negative")
    return batch, packets, cells, nodes, width


def _selected_path_relative_weights(
    *,
    position_id: Tensor,
    cell_id: Tensor,
    node_id: Tensor,
    write_gate: Tensor,
    relation_bias: Tensor,
    receiver_length: int,
) -> Tensor:
    """Return ``[B, receiver, trace]`` selected-location weights."""

    relations = relative_code(
        receiver_length,
        device=relation_bias.device,
        dtype=relation_bias.dtype,
    )
    selected_relation = relations[:, position_id].permute(1, 0, 2, 3)
    selected_bias = relation_bias[cell_id, node_id]
    score = torch.einsum(
        "bipr,bpr->bip",
        selected_relation,
        selected_bias,
    )
    all_score = torch.einsum(
        "ijr,cnr->ijcn",
        relations,
        relation_bias,
    )
    location_max = all_score.amax(dim=1)
    selected_max = location_max[:, cell_id, node_id].permute(1, 0, 2)
    return (
        (score - selected_max).exp()
        * write_gate[:, None, :]
    )


def _apply_path_packet_contributions(
    state: PersistentRelativeMemoryState,
    *,
    payload: Tensor,
    position_id: Tensor,
    cell_id: Tensor,
    node_id: Tensor,
    write_gate: Tensor,
    relation_bias: Tensor,
    receiver_length: int,
    sign: float,
    active: Tensor | None,
) -> PersistentRelativeMemoryState:
    batch, packets, cells, nodes, width = _validate_path_packets(
        payload,
        position_id,
        cell_id,
        node_id,
        write_gate,
        relation_bias,
        receiver_length,
    )
    expected_numerator = (
        batch,
        receiver_length,
        cells,
        nodes,
        width,
    )
    expected_denominator = (
        batch,
        receiver_length,
        cells,
        nodes,
    )
    if tuple(state.numerator.shape) != expected_numerator:
        raise ValueError("state numerator shape does not match path packets")
    if tuple(state.denominator.shape) != expected_denominator:
        raise ValueError(
            "state denominator shape does not match path packets"
        )
    if active is not None:
        if active.dtype != torch.bool or tuple(active.shape) != (
            batch,
            packets,
        ):
            raise TypeError("active must be boolean with shape [B, P]")
        if active.device != payload.device:
            raise ValueError("active must share payload's device")
        write_gate = write_gate * active.to(write_gate.dtype)

    weight = _selected_path_relative_weights(
        position_id=position_id,
        cell_id=cell_id,
        node_id=node_id,
        write_gate=write_gate,
        relation_bias=relation_bias,
        receiver_length=receiver_length,
    )
    location = cell_id * nodes + node_id
    flat_numerator = state.numerator.reshape(
        batch,
        receiver_length,
        cells * nodes,
        width,
    )
    flat_denominator = state.denominator.reshape(
        batch,
        receiver_length,
        cells * nodes,
    )
    numerator_index = location[:, None, :, None].expand(
        batch,
        receiver_length,
        packets,
        width,
    )
    denominator_index = location[:, None, :].expand(
        batch,
        receiver_length,
        packets,
    )
    contribution = (
        sign
        * weight.unsqueeze(-1)
        * payload[:, None]
    )
    flat_numerator = flat_numerator.scatter_add(
        2,
        numerator_index,
        contribution,
    )
    flat_denominator = flat_denominator.scatter_add(
        2,
        denominator_index,
        sign * weight,
    )
    return PersistentRelativeMemoryState(
        numerator=flat_numerator.reshape(expected_numerator),
        denominator=flat_denominator.reshape(expected_denominator),
    )


def build_path_relative_memory(
    *,
    payload: Tensor,
    position_id: Tensor,
    cell_id: Tensor,
    node_id: Tensor,
    write_gate: Tensor,
    relation_bias: Tensor,
    receiver_length: int,
    active: Tensor | None = None,
) -> PersistentRelativeMemoryState:
    """Build memory for several cooperative traces of every sequence slot."""

    batch, _, cells, nodes, width = _validate_path_packets(
        payload,
        position_id,
        cell_id,
        node_id,
        write_gate,
        relation_bias,
        receiver_length,
    )
    state = empty_relative_memory(
        batch=batch,
        length=receiver_length,
        cells=cells,
        nodes=nodes,
        width=width,
        device=payload.device,
        dtype=payload.dtype,
    )
    return _apply_path_packet_contributions(
        state,
        payload=payload,
        position_id=position_id,
        cell_id=cell_id,
        node_id=node_id,
        write_gate=write_gate,
        relation_bias=relation_bias,
        receiver_length=receiver_length,
        sign=1.0,
        active=active,
    )


def update_path_relative_memory(
    state: PersistentRelativeMemoryState,
    *,
    position_id: Tensor,
    old_payload: Tensor,
    old_cell_id: Tensor,
    old_node_id: Tensor,
    old_write_gate: Tensor,
    new_payload: Tensor,
    new_cell_id: Tensor,
    new_node_id: Tensor,
    new_write_gate: Tensor,
    relation_bias: Tensor,
    receiver_length: int,
    active: Tensor | None = None,
) -> PersistentRelativeMemoryState:
    """Incrementally move a cooperative packet-trace population."""

    without_departures = _apply_path_packet_contributions(
        state,
        payload=old_payload,
        position_id=position_id,
        cell_id=old_cell_id,
        node_id=old_node_id,
        write_gate=old_write_gate,
        relation_bias=relation_bias,
        receiver_length=receiver_length,
        sign=-1.0,
        active=active,
    )
    return _apply_path_packet_contributions(
        without_departures,
        payload=new_payload,
        position_id=position_id,
        cell_id=new_cell_id,
        node_id=new_node_id,
        write_gate=new_write_gate,
        relation_bias=relation_bias,
        receiver_length=receiver_length,
        sign=1.0,
        active=active,
    )


__all__ = [
    "PersistentRelativeMemoryState",
    "build_path_relative_memory",
    "build_relative_memory",
    "empty_relative_memory",
    "relative_code",
    "update_path_relative_memory",
    "update_relative_memory",
]
