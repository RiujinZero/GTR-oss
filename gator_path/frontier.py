"""Dense-to-deterministic local packet stewardship for Gator-Path.

This module carries only the *current* packet frontier.  It never stores,
scores, or replays a complete route.  At every relay, each source cell:

1. reads packets currently under its stewardship;
2. applies its own private transform;
3. chooses the next cell and node from a bounded local action space; and
4. hands the transformed packet state to the next frontier.

Soft frontier mass gives every plausible steward real forward participation
early in training.  Lower routing temperature, or straight-through hard
routing at deployment, turns the same local policies into deterministic
handoffs without introducing a route-memory subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .topology import PathTopology


@dataclass(frozen=True)
class FrontierState:
    """The current packet locations and carried payloads."""

    payload: Tensor
    mass: Tensor
    address: Tensor


@dataclass(frozen=True)
class FrontierRelayResult:
    """Small diagnostic summary of one local handoff."""

    route_entropy: Tensor
    move_fraction: Tensor
    occupied_locations: Tensor


@dataclass(frozen=True)
class FrontierResult:
    """Final current frontier plus per-relay summaries."""

    state: FrontierState
    relay_results: tuple[FrontierRelayResult, ...]

    @property
    def relay_steps(self) -> int:
        return len(self.relay_results)


def _straight_through_one_hot(probabilities: Tensor) -> Tensor:
    index = probabilities.argmax(dim=-1)
    hard = F.one_hot(index, probabilities.shape[-1]).to(
        probabilities.dtype
    )
    return hard + probabilities - probabilities.detach()


class LocalPacketFrontier(nn.Module):
    """Repeated local cell policies over a persistent packet frontier."""

    def __init__(
        self,
        *,
        topology: PathTopology,
        payload_width: int,
        address_width: int,
        relay_steps: int,
        relation_aware: bool = False,
    ) -> None:
        super().__init__()
        self.topology = topology
        self.payload_width = payload_width
        self.address_width = address_width
        self.relay_steps = relay_steps
        self.relation_aware = relation_aware
        self.router_width = (
            2 * payload_width + address_width + 1
        )
        cells = topology.num_cells
        nodes = topology.num_nodes
        cell_actions = topology.num_cell_actions

        # The cell chooses its next neighboring cell.  This is deliberately
        # private per cell but shared across that cell's internal nodes.
        self.cell_router_weight = nn.Parameter(
            torch.empty(cells, cell_actions, self.router_width)
        )
        self.cell_router_bias = nn.Parameter(
            torch.zeros(cells, cell_actions)
        )

        # Internal-node choices are private to each cell and source node.
        self.node_router_weight = nn.Parameter(
            torch.empty(cells, nodes, nodes, self.router_width)
        )
        self.node_router_bias = nn.Parameter(
            torch.zeros(cells, nodes, nodes)
        )

        # The source steward transforms the packet before handing it off.
        self.transform_weight = nn.Parameter(
            torch.empty(cells, nodes, payload_width, payload_width)
        )
        self.transform_bias = nn.Parameter(
            torch.zeros(cells, nodes, payload_width)
        )
        self.address_to_state = nn.Linear(
            address_width,
            payload_width,
            bias=False,
        )
        self.step_to_state = nn.Linear(
            address_width,
            payload_width,
            bias=False,
        )
        if relation_aware:
            # Each private (cell, node) policy learns which relative sequence
            # displacements are useful to a receiving packet. This restores
            # Arrow's receiver-specific node reads without introducing global
            # attention or any remembered route object.
            self.relative_node_bias = nn.Parameter(
                torch.zeros(cells, nodes, 4)
            )
        else:
            self.register_parameter("relative_node_bias", None)
        self.norm = nn.LayerNorm(payload_width)

        cell_destination = torch.tensor(
            topology.cell_destination_table,
            dtype=torch.long,
        )
        node_mask = torch.tensor(
            topology.node_adjacency,
            dtype=torch.bool,
        )
        self.register_buffer(
            "cell_destination",
            cell_destination,
            persistent=False,
        )
        self.register_buffer("node_mask", node_mask, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.cell_router_weight)
        nn.init.xavier_uniform_(self.node_router_weight)
        nn.init.xavier_uniform_(self.transform_weight)
        nn.init.zeros_(self.cell_router_bias)
        nn.init.zeros_(self.node_router_bias)
        nn.init.zeros_(self.transform_bias)
        nn.init.xavier_uniform_(self.address_to_state.weight)
        nn.init.xavier_uniform_(self.step_to_state.weight)
        self.norm.reset_parameters()

    def _relative_code(self, length: int, state: FrontierState) -> Tensor:
        receiver = torch.arange(
            length,
            device=state.payload.device,
            dtype=state.payload.dtype,
        )
        displacement = (
            receiver[:, None] - receiver[None, :]
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

    def _step_code(self, relay: int, state: FrontierState) -> Tensor:
        position = torch.tensor(
            relay / max(self.relay_steps - 1, 1),
            device=state.payload.device,
            dtype=state.payload.dtype,
        )
        frequencies = torch.arange(
            1,
            self.address_width // 2 + 1,
            device=state.payload.device,
            dtype=state.payload.dtype,
        )
        angle = math.pi * position * frequencies
        code = torch.stack((angle.sin(), angle.cos()), dim=-1).flatten()
        if code.numel() < self.address_width:
            code = F.pad(code, (0, self.address_width - code.numel()))
        return code[: self.address_width]

    def _mailbox(
        self,
        state: FrontierState,
        write_gate: Tensor,
    ) -> tuple[Tensor, Tensor]:
        write_mass = (
            state.mass
            * write_gate[:, :, None, None]
        )
        location_mass = write_mass.sum(dim=1)
        numerator = (
            write_mass.unsqueeze(-1) * state.payload
        ).sum(dim=1)
        if self.relative_node_bias is None:
            memory = (
                numerator
                / location_mass.clamp_min(1.0e-6).unsqueeze(-1)
            )
            return memory, location_mass

        # scores[b, receiver, writer, cell, node]. A receiving packet gets a
        # different view of the same local mailbox according to relative slot
        # displacement. Packets still interact only inside the cell/node that
        # currently stewards them.
        relative_bias = torch.einsum(
            "ijr,cnr->ijcn",
            self._relative_code(state.payload.shape[1], state),
            self.relative_node_bias,
        )
        tiny = torch.finfo(state.payload.dtype).tiny
        scores = (
            state.mass.clamp_min(tiny).log()[:, None]
            + relative_bias[None]
            + write_gate.clamp_min(tiny).log()[:, None, :, None, None]
        )
        attention = torch.softmax(scores, dim=2)
        memory = torch.einsum(
            "bijcn,bjcnd->bicnd",
            attention,
            state.payload,
        )
        return memory, location_mass

    def _route(
        self,
        state: FrontierState,
        memory: Tensor,
        location_mass: Tensor,
        *,
        temperature: float,
        hard_routes: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, cells, nodes, _ = state.payload.shape
        if memory.ndim == 4:
            context = memory[:, None].expand(
                -1, length, -1, -1, -1
            )
        elif memory.ndim == 5:
            context = memory
        else:
            raise RuntimeError("mailbox memory has an invalid rank")
        address = state.address[None, :, None, None].expand(
            batch, -1, cells, nodes, -1
        )
        occupancy = (
            location_mass / max(length, 1)
        )[:, None, :, :, None].expand(-1, length, -1, -1, -1)
        features = torch.cat(
            (state.payload, context, address, occupancy),
            dim=-1,
        )

        cell_logits = torch.einsum(
            "blcnf,caf->blcna",
            features,
            self.cell_router_weight,
        ) + self.cell_router_bias[None, None, :, None]
        node_logits = torch.einsum(
            "blcnf,cndf->blcnd",
            features,
            self.node_router_weight,
        ) + self.node_router_bias[None, None]
        node_logits = node_logits.masked_fill(
            ~self.node_mask[None, None, None],
            torch.finfo(node_logits.dtype).min,
        )
        cell_probability = torch.softmax(
            cell_logits / temperature,
            dim=-1,
        )
        node_probability = torch.softmax(
            node_logits / temperature,
            dim=-1,
        )
        if hard_routes:
            cell_probability = _straight_through_one_hot(cell_probability)
            node_probability = _straight_through_one_hot(node_probability)
        return features, cell_probability, node_probability

    def _transform(
        self,
        state: FrontierState,
        memory: Tensor,
        relay: int,
    ) -> Tensor:
        address_state = self.address_to_state(state.address)
        step_state = self.step_to_state(self._step_code(relay, state))
        if memory.ndim == 4:
            context = memory[:, None]
        elif memory.ndim == 5:
            context = memory
        else:
            raise RuntimeError("mailbox memory has an invalid rank")
        base = self.norm(
            state.payload
            + context
            + address_state[None, :, None, None]
            + step_state
        )
        delta = torch.tanh(
            torch.einsum(
                "blcni,cnoi->blcno",
                base,
                self.transform_weight,
            )
            + self.transform_bias[None, None]
        )
        return self.norm(state.payload + delta)

    def _handoff(
        self,
        state: FrontierState,
        transformed: Tensor,
        cell_probability: Tensor,
        node_probability: Tensor,
    ) -> FrontierState:
        batch, length, cells, nodes = state.mass.shape
        next_mass = state.mass.new_zeros(batch, length, cells, nodes)
        next_payload_numerator = state.payload.new_zeros(
            batch,
            length,
            cells,
            nodes,
            self.payload_width,
        )

        # Build the bounded local action frontier once.  This is still only
        # the current handoff; it is discarded after the synchronous commit.
        joint_share = (
            state.mass[..., None, None]
            * cell_probability[..., :, None]
            * node_probability[..., None, :]
        )
        source_mass = joint_share.sum(dim=3)
        source_payload = torch.einsum(
            "blcnak,blcnp->blcakp",
            joint_share,
            transformed,
        )

        # Vectorize all destination nodes together and loop only over the five
        # local cell offsets.
        for cell_action, offset in enumerate(self.topology.cell_offsets):
            next_mass += torch.roll(
                source_mass[..., cell_action, :],
                shifts=offset,
                dims=2,
            )
            next_payload_numerator += torch.roll(
                source_payload[..., cell_action, :, :],
                shifts=offset,
                dims=2,
            )

        next_payload = next_payload_numerator / next_mass.clamp_min(
            1.0e-8
        ).unsqueeze(-1)
        # Empty locations carry exact zeros, making the frontier state explicit
        # and preventing normalization noise from creating phantom packets.
        next_payload = torch.where(
            next_mass.unsqueeze(-1) > 0,
            next_payload,
            torch.zeros_like(next_payload),
        )
        next_mass = next_mass / next_mass.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(1.0e-8)
        return FrontierState(
            payload=next_payload,
            mass=next_mass,
            address=state.address,
        )

    def forward(
        self,
        state: FrontierState,
        *,
        write_gate_schedule: Tensor,
        temperature: float,
        hard_routes: bool = False,
        retain_diagnostics: bool = False,
    ) -> FrontierResult:
        relay_results: list[FrontierRelayResult] = []
        current = state
        for relay in range(self.relay_steps):
            memory, location_mass = self._mailbox(
                current,
                write_gate_schedule[relay],
            )
            _, cell_probability, node_probability = self._route(
                current,
                memory,
                location_mass,
                temperature=temperature,
                hard_routes=hard_routes,
            )
            transformed = self._transform(current, memory, relay)

            if retain_diagnostics:
                tiny = torch.finfo(current.payload.dtype).tiny
                cell_entropy = -(
                    cell_probability.clamp_min(tiny)
                    * cell_probability.clamp_min(tiny).log()
                ).sum(dim=-1)
                node_entropy = -(
                    node_probability.clamp_min(tiny)
                    * node_probability.clamp_min(tiny).log()
                ).sum(dim=-1)
                weighted_entropy = (
                    current.mass * (cell_entropy + node_entropy)
                ).sum(dim=(2, 3)).mean()
                stay_probability = cell_probability[..., 0]
                same_node_probability = torch.diagonal(
                    node_probability,
                    dim1=3,
                    dim2=4,
                )
                stay = stay_probability * same_node_probability
                move_fraction = (
                    current.mass * (1.0 - stay)
                ).sum(dim=(2, 3)).mean()
                occupied = current.mass.gt(1.0e-4).float().sum(
                    dim=(2, 3)
                ).mean()
                relay_results.append(
                    FrontierRelayResult(
                        route_entropy=weighted_entropy,
                        move_fraction=move_fraction,
                        occupied_locations=occupied,
                    )
                )

            current = self._handoff(
                current,
                transformed,
                cell_probability,
                node_probability,
            )
        return FrontierResult(
            state=current,
            relay_results=tuple(relay_results),
        )


__all__ = [
    "FrontierRelayResult",
    "FrontierResult",
    "FrontierState",
    "LocalPacketFrontier",
]
