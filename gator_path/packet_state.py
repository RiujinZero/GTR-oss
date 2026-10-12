"""Persistent packet state for Gator-Path.

Gator-Path keeps one mutable payload per batch item and output slot.  Graph
locations are compact integer tensors; there is deliberately no cell axis on
the payload or any other per-packet state.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace as dataclass_replace
from typing import Any, ClassVar

import torch
from torch import Tensor


def _require_tensor(name: str, value: Any) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_shape(
    name: str,
    value: Tensor,
    expected: tuple[int, ...],
    notation: str,
) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(
            f"{name} must have shape {notation}; "
            f"got {tuple(value.shape)}"
        )


def _require_floating(name: str, value: Tensor) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


def _require_long(name: str, value: Tensor) -> None:
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long")


@dataclass(frozen=True)
class PacketState:
    """A synchronous snapshot of persistent Gator-Path packets.

    ``payload`` has shape ``[B, L, D]`` and is never replicated over cells.
    ``address`` normally has the shared shape ``[L, D_a]``.  A caller that
    requires batch-specific addresses may instead provide ``[B, L, D_a]``;
    its leading dimensions are checked against the payload.

    ``cell_id`` and ``node_id`` are the committed top-1 location.  The three
    ``support_*`` tensors optionally retain a sparse top-k route beam without
    changing the single persistent payload representation.

    The dataclass is frozen so relay updates are functional.  PyTorch tensors
    themselves remain mutable, so :meth:`snapshot` clones every mutable field
    when an isolated synchronous read snapshot is required.  Addresses are
    intentionally shared across snapshots and cannot be changed by
    :meth:`replace`.
    """

    payload: Tensor
    address: Tensor
    cell_id: Tensor
    node_id: Tensor
    active: Tensor
    write_gate: Tensor
    trace: Tensor | None = None
    support_cell: Tensor | None = None
    support_node: Tensor | None = None
    support_weight: Tensor | None = None

    _FIELD_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "payload",
            "address",
            "cell_id",
            "node_id",
            "active",
            "write_gate",
            "trace",
            "support_cell",
            "support_node",
            "support_weight",
        }
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the structural packet-state contract.

        Validation is intentionally structural: it checks ranks, shapes,
        dtypes, and devices without reducing tensor values.  State creation
        therefore does not add a device synchronization to a relay.
        """

        payload = _require_tensor("payload", self.payload)
        if payload.ndim != 3:
            raise ValueError(
                "payload must have shape [B, L, D] with no cell axis; "
                f"got {tuple(payload.shape)}"
            )
        batch_size, slot_count, payload_width = payload.shape
        if batch_size <= 0 or slot_count <= 0 or payload_width <= 0:
            raise ValueError("payload dimensions B, L, and D must be positive")
        _require_floating("payload", payload)

        address = _require_tensor("address", self.address)
        if address.ndim == 2:
            if address.shape[0] != slot_count:
                raise ValueError(
                    "shared address must have shape [L, D_a] with "
                    f"L={slot_count}; got {tuple(address.shape)}"
                )
        elif address.ndim == 3:
            if tuple(address.shape[:2]) != (batch_size, slot_count):
                raise ValueError(
                    "batched address must have shape [B, L, D_a] with "
                    f"(B, L)=({batch_size}, {slot_count}); "
                    f"got {tuple(address.shape)}"
                )
        else:
            raise ValueError(
                "address must have shared shape [L, D_a] or batched "
                f"shape [B, L, D_a]; got {tuple(address.shape)}"
            )
        if address.shape[-1] <= 0:
            raise ValueError("address width D_a must be positive")
        _require_floating("address", address)
        if address.dtype != payload.dtype:
            raise TypeError("address and payload must have the same dtype")

        packet_shape = (batch_size, slot_count)
        cell_id = _require_tensor("cell_id", self.cell_id)
        node_id = _require_tensor("node_id", self.node_id)
        active = _require_tensor("active", self.active)
        write_gate = _require_tensor("write_gate", self.write_gate)
        _require_shape("cell_id", cell_id, packet_shape, "[B, L]")
        _require_shape("node_id", node_id, packet_shape, "[B, L]")
        _require_shape("active", active, packet_shape, "[B, L]")
        _require_shape("write_gate", write_gate, packet_shape, "[B, L]")
        _require_long("cell_id", cell_id)
        _require_long("node_id", node_id)
        if active.dtype != torch.bool:
            raise TypeError("active must have dtype torch.bool")
        _require_floating("write_gate", write_gate)

        if self.trace is not None:
            trace = _require_tensor("trace", self.trace)
            if (
                trace.ndim != 3
                or tuple(trace.shape[:2]) != packet_shape
                or trace.shape[-1] <= 0
            ):
                raise ValueError(
                    "trace must have shape [B, L, D_t] with positive "
                    f"D_t; got {tuple(trace.shape)}"
                )
            _require_floating("trace", trace)
            if trace.dtype != payload.dtype:
                raise TypeError("trace and payload must have the same dtype")

        support_values = (
            self.support_cell,
            self.support_node,
            self.support_weight,
        )
        present_support = tuple(value is not None for value in support_values)
        if any(present_support) and not all(present_support):
            raise ValueError(
                "support_cell, support_node, and support_weight must be "
                "provided together"
            )
        if all(present_support):
            support_cell = _require_tensor(
                "support_cell", self.support_cell
            )
            support_node = _require_tensor(
                "support_node", self.support_node
            )
            support_weight = _require_tensor(
                "support_weight", self.support_weight
            )
            if support_cell.ndim != 3:
                raise ValueError(
                    "support_cell must have shape [B, L, K]; "
                    f"got {tuple(support_cell.shape)}"
                )
            support_shape = tuple(support_cell.shape)
            if support_shape[:2] != packet_shape or support_shape[-1] <= 0:
                raise ValueError(
                    "support_cell must have shape [B, L, K] with positive "
                    f"K; got {support_shape}"
                )
            _require_shape(
                "support_node",
                support_node,
                support_shape,
                "[B, L, K]",
            )
            _require_shape(
                "support_weight",
                support_weight,
                support_shape,
                "[B, L, K]",
            )
            _require_long("support_cell", support_cell)
            _require_long("support_node", support_node)
            _require_floating("support_weight", support_weight)

        for name, value in self._tensor_fields():
            if value.device != payload.device:
                raise ValueError(
                    f"{name} must be on payload device {payload.device}; "
                    f"got {value.device}"
                )

    @property
    def batch_size(self) -> int:
        return self.payload.shape[0]

    @property
    def slot_count(self) -> int:
        return self.payload.shape[1]

    @property
    def payload_width(self) -> int:
        return self.payload.shape[2]

    @property
    def address_width(self) -> int:
        return self.address.shape[-1]

    @property
    def trace_width(self) -> int | None:
        return None if self.trace is None else self.trace.shape[-1]

    @property
    def route_k(self) -> int:
        """Return the number of represented routes per packet."""

        return (
            1
            if self.support_cell is None
            else self.support_cell.shape[-1]
        )

    @property
    def has_topk_support(self) -> bool:
        return self.support_cell is not None

    @property
    def batched_address(self) -> Tensor:
        """Return addresses as ``[B, L, D_a]`` without copying shared data."""

        if self.address.ndim == 3:
            return self.address
        return self.address.unsqueeze(0).expand(self.batch_size, -1, -1)

    def snapshot(self, *, detach: bool = False) -> PacketState:
        """Clone mutable packet tensors for an isolated synchronous read.

        The immutable address tensor is shared.  With ``detach=False`` the
        clones remain connected to autograd; ``detach=True`` creates a
        history-free diagnostic/checkpoint snapshot.
        """

        self.validate()

        def copy_tensor(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            if detach:
                return value.detach().clone()
            return value.clone()

        return PacketState(
            payload=copy_tensor(self.payload),
            address=self.address,
            cell_id=copy_tensor(self.cell_id),
            node_id=copy_tensor(self.node_id),
            active=copy_tensor(self.active),
            write_gate=copy_tensor(self.write_gate),
            trace=copy_tensor(self.trace),
            support_cell=copy_tensor(self.support_cell),
            support_node=copy_tensor(self.support_node),
            support_weight=copy_tensor(self.support_weight),
        )

    def replace(self, **changes: Any) -> PacketState:
        """Build a validated functional update while preserving addresses."""

        unknown = set(changes).difference(self._FIELD_NAMES)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unknown packet-state field(s): {names}")
        if (
            "address" in changes
            and changes["address"] is not self.address
        ):
            raise ValueError(
                "address is immutable across packet-state replacements"
            )
        return dataclass_replace(self, **changes)

    def _tensor_fields(self) -> tuple[tuple[str, Tensor], ...]:
        values: list[tuple[str, Tensor]] = []
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                values.append((field.name, value))
        return tuple(values)
