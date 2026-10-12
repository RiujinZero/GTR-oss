"""Simultaneous modular-prefix task shell for Gator-Path."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .core import CoreResult, GatorPathCore
from .frontier import (
    FrontierResult,
    FrontierState,
    LocalPacketFrontier,
)
from .packet_state import PacketState
from .relay import GatorPathRelay
from .topology import fixed_ring_topology


def fixed_address_code(
    length: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a fixed continuous coordinate code without task modulus hints."""

    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    if isinstance(width, bool) or not isinstance(width, int) or width < 2:
        raise ValueError("width must be an integer of at least two")
    position = torch.arange(length, device=device, dtype=dtype)
    angle = position / max(length - 1, 1) * math.pi
    channels: list[Tensor] = []
    frequency = 1
    while len(channels) < width:
        channels.append(torch.sin(frequency * angle))
        if len(channels) < width:
            channels.append(torch.cos(frequency * angle))
        frequency += 1
    return torch.stack(channels, dim=-1)


@dataclass(frozen=True)
class GatorPathPrefixConfig:
    """Architecture settings for the first Gator-Path learning proof."""

    vocab_size: int = 16
    num_cells: int = 1
    payload_width: int = 8
    address_width: int = 4
    relay_steps: int = 4
    top_k: int = 2
    straight_through_hard: bool = False
    local_policy: bool = False
    learned_ingress: bool = False
    frontier_policy: bool = False
    private_cell_io: bool = False
    relation_aware: bool = False

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "num_cells",
            "payload_width",
            "address_width",
            "relay_steps",
            "top_k",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.vocab_size < 16:
            raise ValueError("vocab_size must be at least 16 for the prefix task")
        if self.address_width < 2:
            raise ValueError("address_width must be at least two")
        if type(self.straight_through_hard) is not bool:
            raise TypeError("straight_through_hard must be a boolean")
        if type(self.local_policy) is not bool:
            raise TypeError("local_policy must be a boolean")
        if type(self.learned_ingress) is not bool:
            raise TypeError("learned_ingress must be a boolean")
        if type(self.frontier_policy) is not bool:
            raise TypeError("frontier_policy must be a boolean")
        if type(self.private_cell_io) is not bool:
            raise TypeError("private_cell_io must be a boolean")
        if type(self.relation_aware) is not bool:
            raise TypeError("relation_aware must be a boolean")
        if self.frontier_policy and not self.learned_ingress:
            raise ValueError(
                "frontier_policy requires learned_ingress"
            )
        if (self.private_cell_io or self.relation_aware) and not (
            self.frontier_policy
        ):
            raise ValueError(
                "private_cell_io and relation_aware require frontier_policy"
            )


class GatorPathPrefixModel(nn.Module):
    """Form every modular-prefix output through one simultaneous Path pass."""

    def __init__(self, config: GatorPathPrefixConfig) -> None:
        super().__init__()
        if not isinstance(config, GatorPathPrefixConfig):
            raise TypeError("config must be a GatorPathPrefixConfig")
        self.config = config
        self.topology = fixed_ring_topology(config.num_cells)
        if config.private_cell_io:
            self.token_embedding = None
            self.cell_token_embedding = nn.Parameter(
                torch.empty(
                    config.num_cells,
                    config.vocab_size,
                    config.payload_width,
                )
            )
        else:
            self.token_embedding = nn.Embedding(
                config.vocab_size,
                config.payload_width,
            )
            self.register_parameter("cell_token_embedding", None)
        if config.frontier_policy:
            self.relay = None
            self.core = None
            self.frontier = LocalPacketFrontier(
                topology=self.topology,
                payload_width=config.payload_width,
                address_width=config.address_width,
                relay_steps=config.relay_steps,
                relation_aware=config.relation_aware,
            )
        else:
            self.relay = GatorPathRelay(
                topology=self.topology,
                payload_width=config.payload_width,
                address_width=config.address_width,
                top_k=config.top_k,
                straight_through_hard=config.straight_through_hard,
                local_policy=config.local_policy,
            )
            self.core = GatorPathCore(
                self.relay,
                relay_steps=config.relay_steps,
            )
            self.frontier = None
        self.address_to_readout = nn.Linear(
            config.address_width,
            config.payload_width,
            bias=False,
        )
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        if config.learned_ingress:
            ingress_width = config.payload_width + config.address_width
            self.ingress_router = nn.Linear(
                ingress_width,
                config.num_cells * self.topology.num_nodes,
            )
            self.ingress_delta = nn.Parameter(
                torch.empty(
                    config.num_cells,
                    self.topology.num_nodes,
                    config.payload_width,
                )
            )
        else:
            self.ingress_router = None
            self.register_parameter("ingress_delta", None)
        self.reset_task_parameters()

    def reset_task_parameters(self) -> None:
        if self.cell_token_embedding is not None:
            nn.init.normal_(
                self.cell_token_embedding,
                std=1.0 / math.sqrt(self.config.payload_width),
            )
        elif self.token_embedding is not None:
            nn.init.normal_(
                self.token_embedding.weight,
                std=1.0 / math.sqrt(self.config.payload_width),
            )
        else:
            raise RuntimeError("no token embedding is configured")
        nn.init.xavier_uniform_(self.address_to_readout.weight)
        nn.init.zeros_(self.lm_bias)
        if self.ingress_router is not None:
            nn.init.xavier_uniform_(self.ingress_router.weight)
            nn.init.zeros_(self.ingress_router.bias)
            nn.init.normal_(
                self.ingress_delta,
                std=1.0 / math.sqrt(self.config.payload_width),
            )

    def _learned_initial_stewards(
        self,
        payload: Tensor,
        addresses: Tensor,
        *,
        temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Choose one first steward with a dense local surrogate gradient."""

        if self.ingress_router is None or self.ingress_delta is None:
            raise RuntimeError("learned ingress is not configured")
        batched_address = addresses.unsqueeze(0).expand(
            payload.shape[0], -1, -1
        )
        features = torch.cat((payload, batched_address), dim=-1)
        logits = self.ingress_router(features) / temperature
        probabilities = torch.softmax(logits, dim=-1)
        selected = probabilities.argmax(dim=-1)

        candidate_delta = torch.tanh(
            self.ingress_delta.reshape(
                self.config.num_cells * self.topology.num_nodes,
                self.config.payload_width,
            )
        )
        hard_delta = candidate_delta[selected]
        soft_delta = torch.einsum(
            "blq,qd->bld",
            probabilities,
            candidate_delta,
        )
        payload = payload + hard_delta + soft_delta - soft_delta.detach()
        mean = payload.mean(dim=-1, keepdim=True)
        variance = payload.var(dim=-1, unbiased=False, keepdim=True)
        payload = (payload - mean) * torch.rsqrt(variance + 1.0e-5)
        cell_id = torch.div(
            selected,
            self.topology.num_nodes,
            rounding_mode="floor",
        )
        node_id = selected.remainder(self.topology.num_nodes)
        return payload, cell_id, node_id

    def _ingress_probabilities(
        self,
        payload: Tensor,
        addresses: Tensor,
        *,
        temperature: float,
        hard_routes: bool,
    ) -> Tensor:
        """Return the current first-steward distribution, not a route."""

        if self.ingress_router is None:
            raise RuntimeError("learned ingress is not configured")
        batched_address = addresses.unsqueeze(0).expand(
            payload.shape[0], -1, -1
        )
        features = torch.cat((payload, batched_address), dim=-1)
        probabilities = torch.softmax(
            self.ingress_router(features) / temperature,
            dim=-1,
        )
        if hard_routes:
            selected = probabilities.argmax(dim=-1)
            hard = torch.nn.functional.one_hot(
                selected,
                probabilities.shape[-1],
            ).to(probabilities.dtype)
            probabilities = (
                hard + probabilities - probabilities.detach()
            )
        return probabilities

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_state(
        self,
        observed: Tensor,
        *,
        output_slots: int,
        initial_cell_id: Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[PacketState, Tensor]:
        """Create one packet for every input token and blank output slot."""

        if not isinstance(observed, Tensor) or observed.ndim != 2:
            raise ValueError("observed must have shape [B, observed_length]")
        if observed.dtype != torch.long:
            raise TypeError("observed must have dtype torch.long")
        if (
            isinstance(output_slots, bool)
            or not isinstance(output_slots, int)
            or output_slots <= 0
        ):
            raise ValueError("output_slots must be a positive integer")
        batch_size, observed_length = observed.shape
        if batch_size <= 0 or observed_length <= 0:
            raise ValueError("observed dimensions must be positive")

        total_slots = observed_length + output_slots
        addresses = fixed_address_code(
            total_slots,
            self.config.address_width,
            device=observed.device,
            dtype=self.token_embedding.weight.dtype,
        )
        if self.token_embedding is None:
            raise RuntimeError(
                "private cell I/O is available only through frontier policy"
            )
        observed_payload = self.token_embedding(observed)
        blank_payload = self.token_embedding.weight.mean(dim=0).view(
            1, 1, -1
        ).expand(batch_size, output_slots, -1)
        payload = torch.cat((observed_payload, blank_payload), dim=1)

        # All packets begin in one shared workspace. They can read one another
        # before local routing disperses them through the cell population.
        if initial_cell_id is None and self.config.learned_ingress:
            payload, cell_id, node_id = self._learned_initial_stewards(
                payload,
                addresses,
                temperature=temperature,
            )
        elif initial_cell_id is None:
            cell_id = torch.zeros(
                batch_size,
                total_slots,
                device=observed.device,
                dtype=torch.long,
            )
            node_id = torch.zeros_like(cell_id)
        else:
            if (
                not isinstance(initial_cell_id, Tensor)
                or initial_cell_id.dtype != torch.long
                or tuple(initial_cell_id.shape)
                != (batch_size, total_slots)
            ):
                raise TypeError(
                    "initial_cell_id must be a torch.long tensor with "
                    "shape [B, observed_length + output_slots]"
                )
            if initial_cell_id.device != observed.device:
                raise ValueError(
                    "initial_cell_id and observed must share a device"
                )
            if bool(
                (
                    (initial_cell_id < 0)
                    | (initial_cell_id >= self.config.num_cells)
                ).any().item()
            ):
                raise ValueError("initial_cell_id contains an invalid cell")
            cell_id = initial_cell_id
            node_id = torch.zeros_like(cell_id)
        active = torch.ones_like(cell_id, dtype=torch.bool)
        write_gate = torch.ones(
            batch_size,
            total_slots,
            device=observed.device,
            dtype=payload.dtype,
        )
        return (
            PacketState(
                payload=payload,
                address=addresses,
                cell_id=cell_id,
                node_id=node_id,
                active=active,
                write_gate=write_gate,
            ),
            addresses,
        )

    def initial_frontier(
        self,
        observed: Tensor,
        *,
        output_slots: int,
        temperature: float,
        hard_routes: bool = False,
    ) -> FrontierState:
        """Create a genuine dense current frontier for every packet."""

        if self.frontier is None:
            raise RuntimeError("frontier policy is not configured")
        if self.ingress_delta is None:
            raise RuntimeError("frontier policy requires learned ingress")
        batch_size, observed_length = observed.shape
        total_slots = observed_length + output_slots
        addresses = fixed_address_code(
            total_slots,
            self.config.address_width,
            device=observed.device,
            dtype=(
                self.cell_token_embedding.dtype
                if self.cell_token_embedding is not None
                else self.token_embedding.weight.dtype
            ),
        )
        if self.cell_token_embedding is not None:
            # [C,B,O,D] -> [B,O,C,D]. Every original Arrow cell owned a
            # private lexical table; preserving that capacity lets each first
            # steward interpret the same packet differently.
            observed_payload = self.cell_token_embedding[:, observed].permute(
                1, 2, 0, 3
            )
            blank_payload = self.cell_token_embedding.mean(dim=1)[
                None, None
            ].expand(batch_size, output_slots, -1, -1)
            cell_payload = torch.cat(
                (observed_payload, blank_payload),
                dim=1,
            )
            base_payload = cell_payload.mean(dim=2)
        elif self.token_embedding is not None:
            observed_payload = self.token_embedding(observed)
            blank_payload = self.token_embedding.weight.mean(dim=0).view(
                1, 1, -1
            ).expand(batch_size, output_slots, -1)
            base_payload = torch.cat(
                (observed_payload, blank_payload),
                dim=1,
            )
            cell_payload = base_payload[:, :, None].expand(
                -1, -1, self.config.num_cells, -1
            )
        else:
            raise RuntimeError("no token embedding is configured")
        ingress = self._ingress_probabilities(
            base_payload,
            addresses,
            temperature=temperature,
            hard_routes=hard_routes,
        )
        mass = ingress.view(
            batch_size,
            total_slots,
            self.config.num_cells,
            self.topology.num_nodes,
        )
        address_state = self.frontier.address_to_state(addresses)
        payload = (
            cell_payload[:, :, :, None]
            + address_state[None, :, None, None]
            + self.ingress_delta[None, None]
        )
        payload = self.frontier.norm(payload)
        return FrontierState(
            payload=payload,
            mass=mass,
            address=addresses,
        )

    def write_gate_schedule(
        self,
        *,
        batch_size: int,
        observed_length: int,
        output_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Use the validated context-first controlled-writing schedule."""

        total_slots = observed_length + output_slots
        schedule = torch.ones(
            self.config.relay_steps,
            batch_size,
            total_slots,
            device=device,
            dtype=dtype,
        )
        denominator = max(self.config.relay_steps - 1, 1)
        fractions = (
            torch.arange(
                self.config.relay_steps,
                device=device,
                dtype=dtype,
            )
            / denominator
        )
        schedule[:, :, observed_length:] = fractions.view(-1, 1, 1)
        return schedule

    def forward(
        self,
        observed: Tensor,
        *,
        output_slots: int,
        temperature: float = 1.0,
        straight_through_hard: bool | None = None,
        hard_ingress: bool | None = None,
        hard_handoffs: bool | None = None,
        route_top_k: int | None = None,
        initial_cell_id: Tensor | None = None,
        cell_access_mask: Tensor | None = None,
        cell_destination_table: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, CoreResult | FrontierResult]:
        if self.config.frontier_policy:
            if initial_cell_id is not None:
                raise ValueError(
                    "frontier policy does not accept explicit initial_cell_id"
                )
            if cell_access_mask is not None or cell_destination_table is not None:
                raise ValueError(
                    "frontier policy uses its fixed local topology"
                )
            hard_default = (
                self.config.straight_through_hard
                if straight_through_hard is None
                else straight_through_hard
            )
            ingress_is_hard = (
                hard_default if hard_ingress is None else hard_ingress
            )
            handoffs_are_hard = (
                hard_default if hard_handoffs is None else hard_handoffs
            )
            if type(ingress_is_hard) is not bool:
                raise TypeError("hard_ingress must be a boolean or None")
            if type(handoffs_are_hard) is not bool:
                raise TypeError("hard_handoffs must be a boolean or None")
            frontier = self.initial_frontier(
                observed,
                output_slots=output_slots,
                temperature=temperature,
                hard_routes=ingress_is_hard,
            )
            observed_length = observed.shape[1]
            schedule = self.write_gate_schedule(
                batch_size=observed.shape[0],
                observed_length=observed_length,
                output_slots=output_slots,
                device=observed.device,
                dtype=frontier.payload.dtype,
            )
            frontier_result = self.frontier(
                frontier,
                write_gate_schedule=schedule,
                temperature=temperature,
                hard_routes=handoffs_are_hard,
                retain_diagnostics=return_diagnostics,
            )
            output_address = self.address_to_readout(
                frontier.address[observed_length:]
            )
            if self.cell_token_embedding is not None:
                readout = (
                    frontier_result.state.payload[
                        :, observed_length:, :, :, :
                    ]
                    + output_address[None, :, None, None]
                )
                location_logits = torch.einsum(
                    "blcnd,cvd->blcnv",
                    readout,
                    self.cell_token_embedding,
                )
                logits = (
                    frontier_result.state.mass[
                        :, observed_length:, :, :
                    ].unsqueeze(-1)
                    * location_logits
                ).sum(dim=(2, 3)) + self.lm_bias
            elif self.token_embedding is not None:
                output_payload = (
                    frontier_result.state.mass[
                        :, observed_length:, :, :
                    ].unsqueeze(-1)
                    * frontier_result.state.payload[
                        :, observed_length:, :, :, :
                    ]
                ).sum(dim=(2, 3))
                readout = output_payload + output_address.unsqueeze(0)
                logits = (
                    readout @ self.token_embedding.weight.transpose(0, 1)
                    + self.lm_bias
                )
            else:
                raise RuntimeError("no token embedding is configured")
            if return_diagnostics:
                return logits, frontier_result
            return logits

        state, addresses = self.initial_state(
            observed,
            output_slots=output_slots,
            initial_cell_id=initial_cell_id,
            temperature=temperature,
        )
        observed_length = observed.shape[1]
        schedule = self.write_gate_schedule(
            batch_size=observed.shape[0],
            observed_length=observed_length,
            output_slots=output_slots,
            device=observed.device,
            dtype=state.payload.dtype,
        )
        if self.core is None:
            raise RuntimeError("legacy path core is not configured")
        result = self.core(
            state,
            write_gate_schedule=schedule,
            top_k=route_top_k,
            temperature=temperature,
            straight_through_hard=straight_through_hard,
            cell_access_mask=cell_access_mask,
            cell_destination_table=cell_destination_table,
            retain_diagnostics=return_diagnostics,
        )
        output_payload = result.state.payload[:, observed_length:]
        output_address = self.address_to_readout(
            addresses[observed_length:]
        )
        readout = output_payload + output_address.unsqueeze(0)
        if self.token_embedding is None:
            raise RuntimeError("legacy path requires shared token embedding")
        logits = (
            readout @ self.token_embedding.weight.transpose(0, 1)
            + self.lm_bias
        )
        if return_diagnostics:
            return logits, result
        return logits


__all__ = [
    "GatorPathPrefixConfig",
    "GatorPathPrefixModel",
    "fixed_address_code",
]
