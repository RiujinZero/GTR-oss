"""Sparse occupied-location mailboxes for Gator-Path.

The mailbox is a segmented reduction over active packet route support.  It
stores one row per unique ``(batch, cell, node)`` key and never allocates a
``[B, C, N, ...]`` table.  Packet readers recover their current location with
``searchsorted`` over the sorted occupied keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .packet_state import PacketState


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_active_locations(
    state: PacketState,
    *,
    num_cells: int,
    num_nodes: int,
) -> None:
    """Reject invalid committed locations for active packets.

    Inactive padding packets may use sentinel locations such as ``-1`` because
    they neither write nor read.  Active packets must always occupy a legal
    graph location.
    """

    active_cells = state.cell_id[state.active]
    active_nodes = state.node_id[state.active]
    if active_cells.numel() == 0:
        return
    invalid_cell = (active_cells < 0) | (active_cells >= num_cells)
    if bool(invalid_cell.any().item()):
        raise ValueError(
            f"active cell_id values must be in [0, {num_cells})"
        )
    invalid_node = (active_nodes < 0) | (active_nodes >= num_nodes)
    if bool(invalid_node.any().item()):
        raise ValueError(
            f"active node_id values must be in [0, {num_nodes})"
        )


def _validate_active_coordinate_tensors(
    cell_id: Tensor,
    node_id: Tensor,
    active: Tensor,
    *,
    num_cells: int,
    num_nodes: int,
) -> None:
    active_cells = cell_id[active]
    active_nodes = node_id[active]
    if active_cells.numel() == 0:
        return
    if bool(((active_cells < 0) | (active_cells >= num_cells)).any().item()):
        raise ValueError(
            f"active cell_id values must be in [0, {num_cells})"
        )
    if bool(((active_nodes < 0) | (active_nodes >= num_nodes)).any().item()):
        raise ValueError(
            f"active node_id values must be in [0, {num_nodes})"
        )


def encode_location_keys(
    batch_id: Tensor,
    cell_id: Tensor,
    node_id: Tensor,
    *,
    num_cells: int,
    num_nodes: int,
) -> Tensor:
    """Encode equal-shaped graph coordinates as sortable integer keys."""

    num_cells = _require_positive_int(num_cells, "num_cells")
    num_nodes = _require_positive_int(num_nodes, "num_nodes")
    if not isinstance(batch_id, Tensor):
        raise TypeError("batch_id must be a torch.Tensor")
    if not isinstance(cell_id, Tensor):
        raise TypeError("cell_id must be a torch.Tensor")
    if not isinstance(node_id, Tensor):
        raise TypeError("node_id must be a torch.Tensor")
    if batch_id.dtype != torch.long:
        raise TypeError("batch_id must have dtype torch.long")
    if cell_id.dtype != torch.long:
        raise TypeError("cell_id must have dtype torch.long")
    if node_id.dtype != torch.long:
        raise TypeError("node_id must have dtype torch.long")
    if batch_id.shape != cell_id.shape or batch_id.shape != node_id.shape:
        raise ValueError(
            "batch_id, cell_id, and node_id must have equal shapes"
        )
    if batch_id.device != cell_id.device or batch_id.device != node_id.device:
        raise ValueError(
            "batch_id, cell_id, and node_id must share a device"
        )
    return (
        batch_id * (num_cells * num_nodes)
        + cell_id * num_nodes
        + node_id
    )


@dataclass(frozen=True)
class MailboxRead:
    """Per-packet read of sparse mailboxes at committed locations."""

    mailbox_index: Tensor
    found: Tensor
    mass: Tensor
    payload_mean: Tensor
    address_mean: Tensor
    feature_sum: Tensor | None = None
    feature_payload_sum: Tensor | None = None


@dataclass(frozen=True)
class SparseMailbox:
    """Compact mailbox rows sorted by encoded occupied-location key."""

    occupied_keys: Tensor
    batch_id: Tensor
    cell_id: Tensor
    node_id: Tensor
    mass: Tensor
    payload_sum: Tensor
    address_sum: Tensor
    payload_mean: Tensor
    address_mean: Tensor
    batch_size: int
    num_cells: int
    num_nodes: int
    feature_sum: Tensor | None = None
    feature_payload_sum: Tensor | None = None

    def __post_init__(self) -> None:
        occupied_keys = self.occupied_keys
        if not isinstance(occupied_keys, Tensor):
            raise TypeError("occupied_keys must be a torch.Tensor")
        if occupied_keys.ndim != 1 or occupied_keys.dtype != torch.long:
            raise TypeError(
                "occupied_keys must be a rank-1 torch.long tensor"
            )
        occupied_count = occupied_keys.shape[0]
        for name in ("batch_id", "cell_id", "node_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.shape != (occupied_count,)
                or value.dtype != torch.long
            ):
                raise TypeError(
                    f"{name} must be a rank-1 torch.long tensor with "
                    "one entry per occupied key"
                )
        if (
            not isinstance(self.mass, Tensor)
            or self.mass.shape != (occupied_count,)
            or not self.mass.is_floating_point()
        ):
            raise TypeError(
                "mass must be a rank-1 floating tensor with one entry "
                "per occupied key"
            )
        for name in (
            "payload_sum",
            "address_sum",
            "payload_mean",
            "address_mean",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.ndim != 2
                or value.shape[0] != occupied_count
                or not value.is_floating_point()
            ):
                raise TypeError(
                    f"{name} must be a rank-2 floating tensor with one "
                    "row per occupied key"
                )
        reference_device = occupied_keys.device
        for name in (
            "batch_id",
            "cell_id",
            "node_id",
            "mass",
            "payload_sum",
            "address_sum",
            "payload_mean",
            "address_mean",
        ):
            if getattr(self, name).device != reference_device:
                raise ValueError(
                    f"{name} must be on occupied_keys device "
                    f"{reference_device}"
                )
        _require_positive_int(self.batch_size, "batch_size")
        _require_positive_int(self.num_cells, "num_cells")
        _require_positive_int(self.num_nodes, "num_nodes")
        feature_values = (
            self.feature_sum,
            self.feature_payload_sum,
        )
        if any(value is not None for value in feature_values):
            if not all(value is not None for value in feature_values):
                raise ValueError(
                    "feature_sum and feature_payload_sum must be "
                    "provided together"
                )
            if (
                self.feature_sum.ndim != 2
                or self.feature_sum.shape[0] != occupied_count
                or not self.feature_sum.is_floating_point()
            ):
                raise TypeError(
                    "feature_sum must have shape [occupied, features]"
                )
            expected = (
                occupied_count,
                self.feature_sum.shape[1],
                self.payload_width,
            )
            if (
                tuple(self.feature_payload_sum.shape) != expected
                or not self.feature_payload_sum.is_floating_point()
            ):
                raise TypeError(
                    "feature_payload_sum must have shape "
                    "[occupied, features, payload]"
                )
            if (
                self.feature_sum.device != reference_device
                or self.feature_payload_sum.device != reference_device
            ):
                raise ValueError(
                    "feature mailbox tensors must share the mailbox device"
                )

    @property
    def occupied_count(self) -> int:
        """Number of materialized mailbox rows."""

        return self.occupied_keys.shape[0]

    @property
    def payload_width(self) -> int:
        return self.payload_mean.shape[-1]

    @property
    def address_width(self) -> int:
        return self.address_mean.shape[-1]

    def lookup(self, keys: Tensor) -> Tensor:
        """Return row indices for keys, with ``-1`` for absent locations."""

        if not isinstance(keys, Tensor):
            raise TypeError("keys must be a torch.Tensor")
        if keys.dtype != torch.long:
            raise TypeError("keys must have dtype torch.long")
        if keys.device != self.occupied_keys.device:
            raise ValueError(
                "keys must be on the same device as occupied_keys"
            )
        if self.occupied_count == 0:
            return torch.full_like(keys, -1)

        insertion = torch.searchsorted(self.occupied_keys, keys)
        safe = insertion.clamp(max=self.occupied_count - 1)
        found = (
            (insertion < self.occupied_count)
            & (self.occupied_keys[safe] == keys)
        )
        return torch.where(found, insertion, torch.full_like(insertion, -1))

    def read_packets(
        self,
        state: PacketState,
        *,
        validate_locations: bool = True,
    ) -> MailboxRead:
        """Read each active packet's current location from this mailbox."""

        if not isinstance(state, PacketState):
            raise TypeError("state must be a PacketState")
        if type(validate_locations) is not bool:
            raise TypeError("validate_locations must be a boolean")
        state.validate()
        if state.batch_size != self.batch_size:
            raise ValueError(
                "state batch size does not match mailbox batch size"
            )
        if state.payload_width != self.payload_width:
            raise ValueError(
                "state payload width does not match mailbox payload width"
            )
        if state.address_width != self.address_width:
            raise ValueError(
                "state address width does not match mailbox address width"
            )
        if state.payload.device != self.occupied_keys.device:
            raise ValueError(
                "state and mailbox must be on the same device"
            )
        if validate_locations:
            _validate_active_locations(
                state,
                num_cells=self.num_cells,
                num_nodes=self.num_nodes,
            )

        return self._read_coordinates(
            state,
            cell_id=state.cell_id,
            node_id=state.node_id,
            active=state.active,
        )

    def read_support(
        self,
        state: PacketState,
        *,
        validate_locations: bool = True,
    ) -> MailboxRead:
        """Read every compact top-k support location of each packet."""

        if not isinstance(state, PacketState):
            raise TypeError("state must be a PacketState")
        if not state.has_topk_support:
            raise ValueError("state does not contain top-k support")
        if type(validate_locations) is not bool:
            raise TypeError("validate_locations must be a boolean")
        state.validate()
        if state.batch_size != self.batch_size:
            raise ValueError(
                "state batch size does not match mailbox batch size"
            )
        if state.payload_width != self.payload_width:
            raise ValueError(
                "state payload width does not match mailbox payload width"
            )
        if state.address_width != self.address_width:
            raise ValueError(
                "state address width does not match mailbox address width"
            )
        if state.payload.device != self.occupied_keys.device:
            raise ValueError("state and mailbox must be on the same device")
        active = state.active.unsqueeze(-1).expand_as(state.support_cell)
        if validate_locations:
            _validate_active_coordinate_tensors(
                state.support_cell,
                state.support_node,
                active,
                num_cells=self.num_cells,
                num_nodes=self.num_nodes,
            )
        return self._read_coordinates(
            state,
            cell_id=state.support_cell,
            node_id=state.support_node,
            active=active,
        )

    def _read_coordinates(
        self,
        state: PacketState,
        *,
        cell_id: Tensor,
        node_id: Tensor,
        active: Tensor,
    ) -> MailboxRead:
        prefix = tuple(cell_id.shape)
        batch_shape = (state.batch_size,) + (1,) * (cell_id.ndim - 1)
        batch_ids = torch.arange(
            state.batch_size,
            device=state.payload.device,
            dtype=torch.long,
        ).view(batch_shape).expand(prefix)
        query_keys = encode_location_keys(
            batch_ids,
            cell_id,
            node_id,
            num_cells=self.num_cells,
            num_nodes=self.num_nodes,
        )
        mailbox_index = self.lookup(query_keys)
        found = (mailbox_index >= 0) & active
        safe_index = mailbox_index.clamp_min(0)

        if self.occupied_count == 0:
            mass = state.payload.new_zeros(prefix)
            mass = mass + state.write_gate.sum() * 0.0
            payload_mean = state.payload.new_zeros(
                *prefix, state.payload_width
            )
            payload_mean = payload_mean + state.payload.sum() * 0.0
            address_mean = state.address.new_zeros(
                *prefix, state.address_width
            )
            address_mean = address_mean + state.address.sum() * 0.0
            if self.feature_sum is None:
                feature_sum = None
                feature_payload_sum = None
            else:
                feature_width = self.feature_sum.shape[-1]
                feature_sum = state.payload.new_zeros(
                    *prefix, feature_width
                )
                feature_payload_sum = state.payload.new_zeros(
                    *prefix,
                    feature_width,
                    state.payload_width,
                )
        else:
            mass = self.mass[safe_index]
            payload_mean = self.payload_mean[safe_index]
            address_mean = self.address_mean[safe_index]
            mass = torch.where(found, mass, torch.zeros_like(mass))
            payload_mean = torch.where(
                found.unsqueeze(-1),
                payload_mean,
                torch.zeros_like(payload_mean),
            )
            address_mean = torch.where(
                found.unsqueeze(-1),
                address_mean,
                torch.zeros_like(address_mean),
            )
            if self.feature_sum is None:
                feature_sum = None
                feature_payload_sum = None
            else:
                feature_sum = self.feature_sum[safe_index]
                feature_payload_sum = self.feature_payload_sum[safe_index]
                feature_sum = torch.where(
                    found.unsqueeze(-1),
                    feature_sum,
                    torch.zeros_like(feature_sum),
                )
                feature_payload_sum = torch.where(
                    found.unsqueeze(-1).unsqueeze(-1),
                    feature_payload_sum,
                    torch.zeros_like(feature_payload_sum),
                )

        mailbox_index = torch.where(
            found,
            mailbox_index,
            torch.full_like(mailbox_index, -1),
        )
        return MailboxRead(
            mailbox_index=mailbox_index,
            found=found,
            mass=mass,
            payload_mean=payload_mean,
            address_mean=address_mean,
            feature_sum=feature_sum,
            feature_payload_sum=feature_payload_sum,
        )


def build_sparse_mailbox(
    state: PacketState,
    *,
    num_cells: int,
    num_nodes: int,
    epsilon: float = 1.0e-6,
    validate_locations: bool = True,
    writer_features: Tensor | None = None,
) -> SparseMailbox:
    """Aggregate active packet support into compact occupied-location rows."""

    if not isinstance(state, PacketState):
        raise TypeError("state must be a PacketState")
    state.validate()
    num_cells = _require_positive_int(num_cells, "num_cells")
    num_nodes = _require_positive_int(num_nodes, "num_nodes")
    epsilon = _require_positive_float(epsilon, "epsilon")
    if type(validate_locations) is not bool:
        raise TypeError("validate_locations must be a boolean")
    if writer_features is not None:
        if not isinstance(writer_features, Tensor):
            raise TypeError("writer_features must be a torch.Tensor or None")
        if writer_features.ndim == 2:
            if writer_features.shape[0] != state.slot_count:
                raise ValueError(
                    "shared writer_features must have shape [L, F]"
                )
        elif writer_features.ndim == 3:
            if tuple(writer_features.shape[:2]) != (
                state.batch_size,
                state.slot_count,
            ):
                raise ValueError(
                    "batched writer_features must have shape [B, L, F]"
                )
        else:
            raise ValueError(
                "writer_features must have shape [L, F] or [B, L, F]"
            )
        if writer_features.shape[-1] <= 0:
            raise ValueError("writer feature width must be positive")
        if not writer_features.is_floating_point():
            raise TypeError("writer_features must be floating point")
        if writer_features.device != state.payload.device:
            raise ValueError("writer_features and state must share a device")
        if writer_features.dtype != state.payload.dtype:
            raise TypeError(
                "writer_features and payload must have the same dtype"
            )
    if validate_locations:
        if state.has_topk_support:
            support_active = state.active.unsqueeze(-1).expand_as(
                state.support_cell
            )
            _validate_active_coordinate_tensors(
                state.support_cell,
                state.support_node,
                support_active,
                num_cells=num_cells,
                num_nodes=num_nodes,
            )
        else:
            _validate_active_locations(
                state,
                num_cells=num_cells,
                num_nodes=num_nodes,
            )

    if state.has_topk_support:
        support_active = state.active.unsqueeze(-1).expand_as(
            state.support_cell
        )
        active_batch, active_slot, active_branch = torch.nonzero(
            support_active,
            as_tuple=True,
        )
    else:
        active_batch, active_slot = torch.nonzero(
            state.active,
            as_tuple=True,
        )
        active_branch = None
    if active_batch.numel() == 0:
        empty_long = state.cell_id.reshape(-1)[:0]
        empty_mass = state.write_gate.to(state.payload.dtype).reshape(-1)[:0]
        empty_payload = state.payload.reshape(
            -1, state.payload_width
        )[:0]
        if state.address.ndim == 2:
            empty_address = state.address[:0]
        else:
            empty_address = state.address.reshape(
                -1, state.address_width
            )[:0]
        return SparseMailbox(
            occupied_keys=empty_long,
            batch_id=empty_long,
            cell_id=empty_long,
            node_id=empty_long,
            mass=empty_mass,
            payload_sum=empty_payload,
            address_sum=empty_address,
            payload_mean=empty_payload,
            address_mean=empty_address,
            batch_size=state.batch_size,
            num_cells=num_cells,
            num_nodes=num_nodes,
            feature_sum=(
                None
                if writer_features is None
                else state.payload.new_zeros(
                    0, writer_features.shape[-1]
                )
            ),
            feature_payload_sum=(
                None
                if writer_features is None
                else state.payload.new_zeros(
                    0,
                    writer_features.shape[-1],
                    state.payload_width,
                )
            ),
        )

    if active_branch is None:
        active_cell = state.cell_id[active_batch, active_slot]
        active_node = state.node_id[active_batch, active_slot]
        branch_weight = torch.ones_like(
            state.write_gate[active_batch, active_slot]
        )
    else:
        active_cell = state.support_cell[
            active_batch, active_slot, active_branch
        ]
        active_node = state.support_node[
            active_batch, active_slot, active_branch
        ]
        branch_weight = state.support_weight[
            active_batch, active_slot, active_branch
        ]
    keys = encode_location_keys(
        active_batch,
        active_cell,
        active_node,
        num_cells=num_cells,
        num_nodes=num_nodes,
    )
    sorted_keys, order = torch.sort(keys)
    occupied_keys, segment = torch.unique_consecutive(
        sorted_keys,
        return_inverse=True,
    )
    occupied_count = occupied_keys.shape[0]

    gate = (
        state.write_gate[active_batch, active_slot]
        * branch_weight.to(state.write_gate.dtype)
    ).to(state.payload.dtype)[order]
    payload = state.payload[active_batch, active_slot][order]
    if state.address.ndim == 2:
        address = state.address[active_slot][order]
    else:
        address = state.address[active_batch, active_slot][order]
    if writer_features is not None:
        if writer_features.ndim == 2:
            selected_features = writer_features[active_slot][order]
        else:
            selected_features = writer_features[
                active_batch, active_slot
            ][order]

    mass = state.payload.new_zeros(occupied_count)
    mass = mass.index_add(0, segment, gate)
    payload_sum = state.payload.new_zeros(
        occupied_count, state.payload_width
    )
    payload_sum = payload_sum.index_add(
        0,
        segment,
        payload * gate.unsqueeze(-1),
    )
    address_sum = state.address.new_zeros(
        occupied_count, state.address_width
    )
    address_sum = address_sum.index_add(
        0,
        segment,
        address * gate.unsqueeze(-1),
    )
    denominator = mass.clamp_min(epsilon).unsqueeze(-1)
    payload_mean = payload_sum / denominator
    address_mean = address_sum / denominator
    if writer_features is None:
        feature_sum = None
        feature_payload_sum = None
    else:
        feature_width = selected_features.shape[-1]
        feature_sum = state.payload.new_zeros(
            occupied_count,
            feature_width,
        ).index_add(
            0,
            segment,
            selected_features * gate.unsqueeze(-1),
        )
        feature_payload_sum = state.payload.new_zeros(
            occupied_count,
            feature_width,
            state.payload_width,
        ).index_add(
            0,
            segment,
            selected_features.unsqueeze(-1)
            * payload.unsqueeze(-2)
            * gate.unsqueeze(-1).unsqueeze(-1),
        )

    locations_per_batch = num_cells * num_nodes
    occupied_batch = torch.div(
        occupied_keys,
        locations_per_batch,
        rounding_mode="floor",
    )
    within_batch = occupied_keys.remainder(locations_per_batch)
    occupied_cell = torch.div(
        within_batch,
        num_nodes,
        rounding_mode="floor",
    )
    occupied_node = within_batch.remainder(num_nodes)

    return SparseMailbox(
        occupied_keys=occupied_keys,
        batch_id=occupied_batch,
        cell_id=occupied_cell,
        node_id=occupied_node,
        mass=mass,
        payload_sum=payload_sum,
        address_sum=address_sum,
        payload_mean=payload_mean,
        address_mean=address_mean,
        batch_size=state.batch_size,
        num_cells=num_cells,
        num_nodes=num_nodes,
        feature_sum=feature_sum,
        feature_payload_sum=feature_payload_sum,
    )


__all__ = [
    "MailboxRead",
    "SparseMailbox",
    "build_sparse_mailbox",
    "encode_location_keys",
]
