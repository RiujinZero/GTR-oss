"""One synchronous packet relay for Gator-Path.

The relay reads an immutable packet snapshot, builds sparse occupied-location
mailboxes, chooses bounded local routes, evaluates only the selected private
path transforms, and commits every packet movement together.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mailbox import (
    MailboxRead,
    SparseMailbox,
    build_sparse_mailbox,
)
from .packet_state import PacketState
from .routing import LocalRouteResult, route_local_actions
from .topology import PathTopology
from .transforms import SelectedPathTransform


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class RelayResult:
    """Committed state plus the old-snapshot relay diagnostics."""

    state: PacketState
    mailbox: SparseMailbox
    read: MailboxRead
    route: LocalRouteResult
    source_cell: Tensor
    source_node: Tensor


class GatorPathRelay(nn.Module):
    """A bounded local Gator-Path routing and transformation step."""

    def __init__(
        self,
        *,
        topology: PathTopology,
        payload_width: int,
        address_width: int,
        top_k: int = 1,
        temperature: float = 1.0,
        straight_through_hard: bool = False,
        local_policy: bool = False,
        norm_epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if not isinstance(topology, PathTopology):
            raise TypeError("topology must be a PathTopology")
        self.topology = topology
        self.payload_width = _positive_int(
            payload_width, "payload_width"
        )
        self.address_width = _positive_int(
            address_width, "address_width"
        )
        self.top_k = _positive_int(top_k, "top_k")
        minimum_joint_actions = min(
            topology.num_joint_actions(node)
            for node in range(topology.num_nodes)
        )
        if self.top_k > minimum_joint_actions:
            raise ValueError(
                "top_k exceeds the legal local joint-action count for "
                f"at least one source node ({minimum_joint_actions})"
            )
        self.temperature = _positive_finite_float(
            temperature, "temperature"
        )
        self.straight_through_hard = _boolean(
            straight_through_hard, "straight_through_hard"
        )
        self.local_policy = _boolean(local_policy, "local_policy")
        self.norm_epsilon = _positive_finite_float(
            norm_epsilon, "norm_epsilon"
        )

        self.max_node_actions = max(
            len(destinations)
            for destinations in topology.legal_node_destinations
        )
        router_width = (
            2 * self.payload_width + 2 * self.address_width + 1
        )
        self.router_width = router_width
        self.mailbox_address_weight = nn.Parameter(
            torch.empty(self.payload_width, self.address_width)
        )
        self.mailbox_key_weight = nn.Parameter(
            torch.empty(self.address_width, self.address_width)
        )
        self.mailbox_query_weight = nn.Parameter(
            torch.empty(self.address_width, self.address_width)
        )

        # The parameters are private to the packet's committed source cell.
        # Only selected source rows are gathered during the forward pass.
        self.cell_router_weight = nn.Parameter(
            torch.empty(
                topology.num_cells,
                topology.num_cell_actions,
                router_width,
            )
        )
        self.cell_router_bias = nn.Parameter(
            torch.zeros(
                topology.num_cells,
                topology.num_cell_actions,
            )
        )
        self.node_router_weight = nn.Parameter(
            torch.empty(
                topology.num_cells,
                topology.num_nodes,
                self.max_node_actions,
                router_width,
            )
        )
        self.node_router_bias = nn.Parameter(
            torch.zeros(
                topology.num_cells,
                topology.num_nodes,
                self.max_node_actions,
            )
        )

        self.path_transform = SelectedPathTransform(
            num_cells=topology.num_cells,
            num_nodes=topology.num_nodes,
            payload_width=self.payload_width,
            address_width=self.address_width,
            num_cell_actions=topology.num_cell_actions,
        )
        self.destination_norm_weight = nn.Parameter(
            torch.ones(topology.num_cells, self.payload_width)
        )
        self.destination_norm_bias = nn.Parameter(
            torch.zeros(topology.num_cells, self.payload_width)
        )
        self.reset_router_parameters()

    def reset_router_parameters(self) -> None:
        nn.init.xavier_uniform_(self.mailbox_address_weight)
        nn.init.xavier_uniform_(self.mailbox_key_weight)
        nn.init.xavier_uniform_(self.mailbox_query_weight)
        cell_bound = math.sqrt(
            6.0
            / (self.router_width + self.topology.num_cell_actions)
        )
        node_bound = math.sqrt(
            6.0 / (self.router_width + self.max_node_actions)
        )
        nn.init.uniform_(
            self.cell_router_weight, -cell_bound, cell_bound
        )
        nn.init.zeros_(self.cell_router_bias)
        nn.init.uniform_(
            self.node_router_weight, -node_bound, node_bound
        )
        nn.init.zeros_(self.node_router_bias)

    def _controls(
        self,
        *,
        top_k: int | None,
        temperature: float | None,
        straight_through_hard: bool | None,
    ) -> tuple[int, float, bool]:
        selected_top_k = (
            self.top_k
            if top_k is None
            else _positive_int(top_k, "top_k")
        )
        minimum_joint_actions = min(
            self.topology.num_joint_actions(node)
            for node in range(self.topology.num_nodes)
        )
        if selected_top_k > minimum_joint_actions:
            raise ValueError(
                "top_k exceeds the legal local joint-action count for "
                f"at least one source node ({minimum_joint_actions})"
            )
        selected_temperature = (
            self.temperature
            if temperature is None
            else _positive_finite_float(temperature, "temperature")
        )
        selected_hard = (
            self.straight_through_hard
            if straight_through_hard is None
            else _boolean(
                straight_through_hard, "straight_through_hard"
            )
        )
        return selected_top_k, selected_temperature, selected_hard

    def _validate_state(self, state: PacketState) -> None:
        if not isinstance(state, PacketState):
            raise TypeError("state must be a PacketState")
        state.validate()
        if state.payload_width != self.payload_width:
            raise ValueError(
                f"state payload width must be {self.payload_width}; "
                f"got {state.payload_width}"
            )
        if state.address_width != self.address_width:
            raise ValueError(
                f"state address width must be {self.address_width}; "
                f"got {state.address_width}"
            )
        if state.payload.device != self.cell_router_weight.device:
            raise ValueError(
                "state and relay parameters must be on the same device"
            )
        if state.payload.dtype != self.cell_router_weight.dtype:
            raise TypeError(
                "state payload and relay parameters must have the same dtype"
            )

    def _router_logits(
        self,
        features: Tensor,
        source_cell: Tensor,
        source_node: Tensor,
    ) -> tuple[Tensor, Tensor]:
        selected_cell_weight = self.cell_router_weight[source_cell]
        selected_cell_bias = self.cell_router_bias[source_cell]
        cell_logits = (
            torch.einsum(
                "blai,bli->bla",
                selected_cell_weight,
                features,
            )
            + selected_cell_bias
        )

        selected_node_weight = self.node_router_weight[
            source_cell, source_node
        ]
        selected_node_bias = self.node_router_bias[
            source_cell, source_node
        ]
        node_logits = (
            torch.einsum(
                "blai,bli->bla",
                selected_node_weight,
                features,
            )
            + selected_node_bias
        )
        return cell_logits, node_logits

    def _selected_payload_update(
        self,
        *,
        state: PacketState,
        read: MailboxRead,
        route: LocalRouteResult,
    ) -> Tensor:
        active_batch, active_slot = torch.nonzero(
            state.active, as_tuple=True
        )
        if active_batch.numel() == 0:
            return state.payload

        top_k = route.top_k
        payload = state.payload[active_batch, active_slot]
        mailbox_payload = (
            read.payload_mean
            + F.linear(
                read.address_mean,
                self.mailbox_address_weight,
            )
        )[active_batch, active_slot]
        address = state.batched_address[active_batch, active_slot]

        branch_payload = payload.unsqueeze(1).expand(
            -1, top_k, -1
        )
        branch_mailbox = mailbox_payload.unsqueeze(1).expand(
            -1, top_k, -1
        )
        branch_address = address.unsqueeze(1).expand(
            -1, top_k, -1
        )
        branch_source_node = state.node_id[
            active_batch, active_slot
        ].unsqueeze(-1).expand(-1, top_k)
        branch_source_cell = state.cell_id[
            active_batch, active_slot
        ].unsqueeze(-1).expand(-1, top_k)

        selected_delta = self.path_transform(
            branch_payload,
            branch_mailbox,
            branch_address,
            route.destination_cell[active_batch, active_slot],
            branch_source_node,
            route.destination_node[active_batch, active_slot],
            route.cell_action[active_batch, active_slot],
            steward_cell=(
                branch_source_cell if self.local_policy else None
            ),
            steward_node=(
                branch_source_node if self.local_policy else None
            ),
            validate_indices=False,
        )
        selected_weights = route.weights[active_batch, active_slot]
        soft_delta = (
            selected_delta * selected_weights.unsqueeze(-1)
        ).sum(dim=1)
        if self.local_policy:
            # The forward pass is one actual stewardship decision.  The
            # soft local mixture exists only as a surrogate gradient so
            # alternate neighboring cells can become useful before they
            # are selected.  No route beam survives this relay.
            hard_delta = selected_delta[:, 0]
            mixed_delta = (
                hard_delta + soft_delta - soft_delta.detach()
            )
        else:
            mixed_delta = soft_delta
        residual = payload + mixed_delta

        # The persistent payload follows the top-ranked committed path.  K>1
        # alternatives remain compact support metadata for the next stage.
        committed_cell = route.destination_cell[
            active_batch, active_slot, 0
        ]
        normalization_cell = (
            state.cell_id[active_batch, active_slot]
            if self.local_policy
            else committed_cell
        )
        mean = residual.mean(dim=-1, keepdim=True)
        variance = residual.var(
            dim=-1, unbiased=False, keepdim=True
        )
        normalized = (
            (residual - mean)
            * torch.rsqrt(variance + self.norm_epsilon)
        )
        normalized = (
            normalized * self.destination_norm_weight[normalization_cell]
            + self.destination_norm_bias[normalization_cell]
        )

        flat_packet_index = (
            active_batch * state.slot_count + active_slot
        )
        update = state.payload.new_zeros(
            state.batch_size * state.slot_count,
            self.payload_width,
        )
        update = update.index_add(
            0,
            flat_packet_index,
            normalized - payload,
        )
        return state.payload + update.view_as(state.payload)

    def forward(
        self,
        state: PacketState,
        *,
        top_k: int | None = None,
        temperature: float | None = None,
        straight_through_hard: bool | None = None,
        cell_access_mask: Tensor | None = None,
        cell_destination_table: Tensor | None = None,
        validate_coordinates: bool = True,
    ) -> RelayResult:
        """Read, route, transform, and synchronously commit one relay."""

        self._validate_state(state)
        validate_coordinates = _boolean(
            validate_coordinates, "validate_coordinates"
        )
        selected_top_k, selected_temperature, selected_hard = (
            self._controls(
                top_k=top_k,
                temperature=temperature,
                straight_through_hard=straight_through_hard,
            )
        )
        if cell_access_mask is not None:
            expected_access_shape = (
                self.topology.num_cells,
                self.topology.num_cells,
            )
            if (
                not isinstance(cell_access_mask, Tensor)
                or cell_access_mask.dtype != torch.bool
                or tuple(cell_access_mask.shape) != expected_access_shape
            ):
                raise TypeError(
                    "cell_access_mask must be a boolean tensor with shape "
                    f"{expected_access_shape}"
                )
            if cell_access_mask.device != state.payload.device:
                raise ValueError(
                    "cell_access_mask and state must share a device"
                )
        if cell_destination_table is not None:
            expected_destination_shape = (
                self.topology.num_cells,
                self.topology.num_cell_actions,
            )
            if (
                not isinstance(cell_destination_table, Tensor)
                or cell_destination_table.dtype != torch.long
                or tuple(cell_destination_table.shape)
                != expected_destination_shape
            ):
                raise TypeError(
                    "cell_destination_table must be a torch.long tensor "
                    f"with shape {expected_destination_shape}"
                )
            if cell_destination_table.device != state.payload.device:
                raise ValueError(
                    "cell_destination_table and state must share a device"
                )

        # This is the only value-range validation in the relay.  Inactive
        # sentinel coordinates are excluded by the mailbox validator.
        mailbox = build_sparse_mailbox(
            state,
            num_cells=self.topology.num_cells,
            num_nodes=self.topology.num_nodes,
            validate_locations=validate_coordinates,
            writer_features=(
                F.elu(
                    F.linear(
                        state.batched_address,
                        self.mailbox_key_weight,
                    )
                )
                + 1.0
            ),
        )
        if state.has_topk_support:
            support_read = mailbox.read_support(
                state,
                validate_locations=False,
            )
            support_weight = torch.where(
                state.active.unsqueeze(-1),
                state.support_weight,
                torch.zeros_like(state.support_weight),
            )
            support_weight = support_weight / support_weight.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(torch.finfo(support_weight.dtype).tiny)
            read = MailboxRead(
                mailbox_index=support_read.mailbox_index[..., 0],
                found=support_read.found.any(dim=-1),
                mass=(
                    support_weight * support_read.mass
                ).sum(dim=-1),
                payload_mean=(
                    support_weight.unsqueeze(-1)
                    * support_read.payload_mean
                ).sum(dim=-2),
                address_mean=(
                    support_weight.unsqueeze(-1)
                    * support_read.address_mean
                ).sum(dim=-2),
                feature_sum=(
                    support_weight.unsqueeze(-1)
                    * support_read.feature_sum
                ).sum(dim=-2),
                feature_payload_sum=(
                    support_weight.unsqueeze(-1).unsqueeze(-1)
                    * support_read.feature_payload_sum
                ).sum(dim=-3),
            )
        else:
            read = mailbox.read_packets(
                state, validate_locations=False
            )

        # A sparse linear-address read gives packets sharing one physical
        # mailbox different views of its contents. This restores the
        # receiver-relative information that made ARW useful without
        # constructing an O(L^2) packet-attention tensor.
        receiver_query = (
            F.elu(
                F.linear(
                    state.batched_address,
                    self.mailbox_query_weight,
                )
            )
            + 1.0
        )
        relative_mass = (
            receiver_query * read.feature_sum
        ).sum(dim=-1)
        relative_payload = (
            receiver_query.unsqueeze(-1)
            * read.feature_payload_sum
        ).sum(dim=-2) / relative_mass.clamp_min(
            torch.finfo(state.payload.dtype).tiny
        ).unsqueeze(-1)
        relative_payload = torch.where(
            read.found.unsqueeze(-1),
            relative_payload,
            torch.zeros_like(relative_payload),
        )
        read = MailboxRead(
            mailbox_index=read.mailbox_index,
            found=read.found,
            mass=read.mass,
            payload_mean=relative_payload,
            address_mean=read.address_mean,
            feature_sum=read.feature_sum,
            feature_payload_sum=read.feature_payload_sum,
        )

        # Local routing requires valid gather coordinates even for inactive
        # padding.  Sanitize those coordinates, then restore the sentinels at
        # commit so inactive packets remain exactly where they were.
        safe_cell = torch.where(
            state.active,
            state.cell_id,
            torch.zeros_like(state.cell_id),
        )
        safe_node = torch.where(
            state.active,
            state.node_id,
            torch.zeros_like(state.node_id),
        )
        router_features = torch.cat(
            (
                state.payload,
                read.payload_mean,
                state.batched_address,
                read.address_mean,
                torch.log1p(read.mass.clamp_min(0.0)).unsqueeze(-1),
            ),
            dim=-1,
        )
        cell_logits, node_logits = self._router_logits(
            router_features,
            safe_cell,
            safe_node,
        )
        if cell_access_mask is not None:
            destination_table = (
                torch.tensor(
                    self.topology.cell_destination_table,
                    device=state.payload.device,
                    dtype=torch.long,
                )
                if cell_destination_table is None
                else cell_destination_table
            )
            local_destinations = destination_table[safe_cell]
            permitted_actions = cell_access_mask[
                safe_cell.unsqueeze(-1),
                local_destinations,
            ]
            node_degrees = torch.tensor(
                tuple(
                    self.topology.num_node_actions(node)
                    for node in range(self.topology.num_nodes)
                ),
                device=state.payload.device,
                dtype=torch.long,
            )
            available_joint_actions = (
                permitted_actions.sum(dim=-1)
                * node_degrees[safe_node]
            )
            if bool(
                (
                    state.active
                    & (available_joint_actions < selected_top_k)
                ).any().item()
            ):
                raise ValueError(
                    "top_k exceeds the routes permitted by "
                    "cell_access_mask for at least one active packet"
                )
            cell_logits = cell_logits.masked_fill(
                ~permitted_actions,
                -torch.inf,
            )
        route = route_local_actions(
            topology=self.topology,
            source_cell=safe_cell,
            source_node=safe_node,
            cell_logits=cell_logits,
            node_logits=node_logits,
            top_k=selected_top_k,
            temperature=selected_temperature,
            straight_through_hard=selected_hard,
            cell_destination_table=cell_destination_table,
            validate_coordinates=False,
        )

        payload = self._selected_payload_update(
            state=state,
            read=read,
            route=route,
        )
        committed_cell = torch.where(
            state.active,
            route.destination_cell[..., 0],
            state.cell_id,
        )
        committed_node = torch.where(
            state.active,
            route.destination_node[..., 0],
            state.node_id,
        )

        if selected_top_k > 1 and not self.local_policy:
            inactive_cell = state.cell_id.unsqueeze(-1).expand(
                -1, -1, selected_top_k
            )
            inactive_node = state.node_id.unsqueeze(-1).expand(
                -1, -1, selected_top_k
            )
            support_cell = torch.where(
                state.active.unsqueeze(-1),
                route.destination_cell,
                inactive_cell,
            )
            support_node = torch.where(
                state.active.unsqueeze(-1),
                route.destination_node,
                inactive_node,
            )
            support_weight = torch.where(
                state.active.unsqueeze(-1),
                route.weights,
                torch.zeros_like(route.weights),
            )
        else:
            support_cell = None
            support_node = None
            support_weight = None

        committed = state.replace(
            payload=payload,
            cell_id=committed_cell,
            node_id=committed_node,
            support_cell=support_cell,
            support_node=support_node,
            support_weight=support_weight,
        )
        return RelayResult(
            state=committed,
            mailbox=mailbox,
            read=read,
            route=route,
            source_cell=state.cell_id,
            source_node=state.node_id,
        )


__all__ = ["GatorPathRelay", "RelayResult"]
