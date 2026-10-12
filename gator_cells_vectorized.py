"""Vectorized 10-cell Gator ring with checkpoint compatibility."""

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from gtr_a_experiments import GTRFixedSlots, fixed_code, relative_code
from gator import OptimizationEngine, StepResult
from gator.cli import add_optimization_arguments, optimization_config_from_args
from gator.diagnostics import AdamWWeightTracer
from gator.models.memory import linear_scan_receiver_context
from gator_path.persistent_memory import (
    build_path_relative_memory,
    build_relative_memory,
    update_path_relative_memory,
    update_relative_memory,
)
from parallel_prefix_benchmark import evaluate, make_prefix_batch


class VectorizedGatorCellRing(nn.Module):
    def __init__(
        self,
        n_cells=10,
        n_nodes=4,
        relays=4,
        neighbors=4,
        *,
        d_model=8,
        routing_dim=None,
        receiver_memory_backend="attention",
        carry_state=False,
        destination_carry=False,
        source_carry=False,
        persistent_active=False,
        active_paths=None,
        reference_dense=False,
    ):
        super().__init__()
        if (
            isinstance(n_cells, bool)
            or not isinstance(n_cells, int)
            or n_cells <= 0
        ):
            raise ValueError("n_cells must be a positive integer")
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model < 4:
            raise ValueError("d_model must be an integer of at least four")
        if routing_dim is not None and (
            isinstance(routing_dim, bool)
            or not isinstance(routing_dim, int)
            or routing_dim < 1
        ):
            raise ValueError("routing_dim must be a positive integer or None")
        if receiver_memory_backend not in {"attention", "linear_scan"}:
            raise ValueError(
                "receiver_memory_backend must be 'attention' or 'linear_scan'"
            )
        if (
            isinstance(n_nodes, bool)
            or not isinstance(n_nodes, int)
            or n_nodes < 2
        ):
            raise ValueError("n_nodes must be an integer of at least two")
        if neighbors not in {2, 4, 6, 8}:
            raise ValueError("neighbors must be one of 2, 4, 6, 8")
        if n_cells <= neighbors:
            raise ValueError("n_cells must exceed the neighbor count")
        if type(carry_state) is not bool:
            raise TypeError("carry_state must be a boolean")
        if type(destination_carry) is not bool:
            raise TypeError("destination_carry must be a boolean")
        if type(source_carry) is not bool:
            raise TypeError("source_carry must be a boolean")
        if source_carry and (carry_state or destination_carry):
            raise ValueError(
                "source-owned and destination-owned carry are exclusive"
            )
        if type(persistent_active) is not bool:
            raise TypeError("persistent_active must be a boolean")
        if persistent_active and n_nodes != 4:
            raise ValueError(
                "persistent-active execution currently requires four nodes"
            )
        if type(reference_dense) is not bool:
            raise TypeError("reference_dense must be a boolean")
        if receiver_memory_backend == "linear_scan" and (
            persistent_active or reference_dense
        ):
            raise NotImplementedError(
                "receiver_memory_backend='linear_scan' is only wired into "
                "the default vectorized forward path; persistent_active "
                "and reference_dense use separate memory-read einsums that "
                "don't yet call it"
            )
        if routing_dim is not None and (persistent_active or reference_dense):
            raise NotImplementedError(
                "routing_dim is only wired into the default vectorized "
                "forward path; persistent_active and reference_dense use "
                "separate router einsums that don't yet apply the "
                "bottleneck"
            )
        if active_paths is None:
            active_paths = n_cells
        if (
            isinstance(active_paths, bool)
            or not isinstance(active_paths, int)
            or active_paths <= 0
            or active_paths > n_cells
        ):
            raise ValueError("active_paths must be in [1, n_cells]")
        self.n_cells = n_cells
        self.d_model = d_model
        self.routing_dim = routing_dim
        self.receiver_memory_backend = receiver_memory_backend
        self.n_nodes = n_nodes
        self.relays = relays
        self.neighbors = neighbors
        self.carry_state = carry_state or destination_carry
        self.destination_carry = self.carry_state
        self.source_carry = source_carry
        self.persistent_active = persistent_active
        self.active_paths = active_paths
        self.reference_dense = reference_dense
        self.cells = nn.ModuleList(
            GTRFixedSlots(
                "gtrarw",
                relays,
                d_model,
                n_nodes=n_nodes,
                routing_dim=routing_dim,
            )
            for _ in range(n_cells)
        )
        self.left_projection = nn.Linear(d_model, d_model)
        self.right_projection = nn.Linear(d_model, d_model)
        self.coupling_norm = nn.LayerNorm(d_model)
        self.neighbor_gates = nn.Parameter(torch.zeros(n_cells, 2))
        self.self_gates = nn.Parameter(torch.ones(n_cells))
        self.output_votes = nn.Parameter(torch.zeros(n_cells))
        # The selected internal-node edge doubles as the local cell handoff.
        # No complete route is represented or stored. Each source cell makes
        # this one decision from the packet it currently stewards.
        self.register_buffer(
            "active_cell_offsets",
            torch.tensor(
                [
                    [0, 1, 0, 0],
                    [0, 0, 1, -1],
                    [2, 0, 0, -2],
                    [-1, 1, 0, 0],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        if n_cells == 10 and n_nodes == 4 and d_model == 8 and routing_dim is None:
            assert self.parameter_count == 10_000

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def relay_receptive_field(self):
        """Maximum cells that one final lane can depend on through the ring.

        Each relay can move information by ``neighbors // 2`` cells in both
        ring directions.  Growing the population without also growing relay
        depth or neighborhood size therefore adds cells outside a terminal
        lane's causal field.
        """

        return min(
            self.n_cells,
            1 + self.neighbors * self.relays,
        )

    @property
    def relay_population_coverage(self):
        """Fraction of the cell population visible to one terminal lane."""

        return self.relay_receptive_field / self.n_cells

    def _cell_norm(self, x):
        weights = torch.stack([cell.norm.weight for cell in self.cells])[None, :, None]
        biases = torch.stack([cell.norm.bias for cell in self.cells])[None, :, None]
        return F.layer_norm(x, (self.d_model,)) * weights + biases

    def _selected_cell_norm(self, value, cell_id):
        """Apply only the private norm of each packet's current steward."""

        weights = torch.stack([cell.norm.weight for cell in self.cells])
        biases = torch.stack([cell.norm.bias for cell in self.cells])
        return (
            F.layer_norm(value, (self.d_model,)) * weights[cell_id]
            + biases[cell_id]
        )

    def _hard_sample(self, logits, temperature):
        """One deterministic choice with a dense surrogate gradient."""

        if self.training:
            probabilities = F.softmax(logits / temperature, dim=-1)
            index = probabilities.argmax(dim=-1)
            hard = F.one_hot(index, logits.shape[-1]).to(
                probabilities.dtype
            )
            return hard + probabilities - probabilities.detach()
        index = logits.argmax(dim=-1)
        return F.one_hot(index, logits.shape[-1]).to(logits.dtype)

    def _cooperative_active_forward(
        self,
        observed,
        *,
        output_slots,
        temperature,
    ):
        """Carry K complementary traces per token through local handoffs."""

        batch, observed_length = observed.shape
        length = observed_length + output_slots
        paths = self.active_paths
        device = observed.device
        addresses = fixed_code(length, 4, device)
        token_weights = torch.stack(
            [cell.token_embedding.weight for cell in self.cells]
        )
        address_state_weights = torch.stack(
            [cell.address_to_state.weight for cell in self.cells]
        )
        address_state = torch.einsum(
            "ta,cda->tcd",
            addresses,
            address_state_weights,
        )

        # Spread fewer-than-C traces evenly; K=C gives every private Arrow
        # cell one complementary interpretation of every token.
        branch_cell = torch.div(
            torch.arange(paths, device=device) * self.n_cells,
            paths,
            rounding_mode="floor",
        )
        cell_id = branch_cell.view(1, 1, paths).expand(
            batch,
            length,
            -1,
        )
        observed_candidates = token_weights[branch_cell][:, observed].permute(
            1,
            2,
            0,
            3,
        )
        blank_candidates = token_weights[branch_cell].mean(dim=1)[
            None,
            None,
        ].expand(batch, output_slots, -1, -1)
        payload = torch.cat(
            (observed_candidates, blank_candidates),
            dim=1,
        )
        selected_address_state = address_state[:, branch_cell][None].expand(
            batch,
            -1,
            -1,
            -1,
        )
        payload = self._selected_cell_norm(
            payload + selected_address_state,
            cell_id,
        )

        starts = torch.stack([cell.start_logits for cell in self.cells])
        node_choice = self._hard_sample(starts[cell_id], temperature)
        node_id = node_choice.argmax(dim=-1)
        selected_start_gate = node_choice.gather(
            -1,
            node_id.unsqueeze(-1),
        ).squeeze(-1)
        payload = payload * selected_start_gate.unsqueeze(-1)

        packets = length * paths
        payload = payload.reshape(batch, packets, self.d_model)
        cell_id = cell_id.reshape(batch, packets)
        node_id = node_id.reshape(batch, packets)
        position_id = torch.arange(
            length,
            device=device,
        ).repeat_interleave(paths)[None].expand(batch, -1)
        packet_address = addresses[position_id]

        relative_bias = torch.stack(
            [cell.relative_node_bias for cell in self.cells]
        )
        slot_write = torch.ones(
            self.relays,
            length,
            device=device,
            dtype=payload.dtype,
        )
        if output_slots:
            fractions = torch.arange(
                self.relays,
                device=device,
                dtype=payload.dtype,
            ) / max(self.relays - 1, 1)
            slot_write[:, observed_length:] = fractions[:, None]
        write_schedule = slot_write[
            :,
            position_id[0],
        ][:, None].expand(-1, batch, -1)

        memory = build_path_relative_memory(
            payload=payload,
            position_id=position_id,
            cell_id=cell_id,
            node_id=node_id,
            write_gate=write_schedule[0],
            relation_bias=relative_bias,
            receiver_length=length,
        )
        adjacency = self.cells[0].adjacency
        router_w = torch.stack([cell.router.weight for cell in self.cells])
        router_b = torch.stack([cell.router.bias for cell in self.cells])
        address_edge_w = torch.stack(
            [cell.address_to_edges.weight for cell in self.cells]
        )
        step_w = torch.stack(
            [cell.step_projection.weight for cell in self.cells]
        )
        node_embeddings = torch.stack(
            [cell.node_embedding.weight for cell in self.cells]
        )
        node_w = torch.stack(
            [
                torch.stack(
                    [transform.weight for transform in cell.node_transforms]
                )
                for cell in self.cells
            ]
        )
        node_b = torch.stack(
            [
                torch.stack(
                    [transform.bias for transform in cell.node_transforms]
                )
                for cell in self.cells
            ]
        )
        batch_index = torch.arange(device=device, end=batch)[:, None]
        ones = payload.new_ones(batch, packets)

        for relay in range(self.relays):
            old_payload = payload
            old_cell_id = cell_id
            old_node_id = node_id
            context = memory.value[
                batch_index,
                position_id,
                cell_id,
                node_id,
            ]

            flat_location = cell_id * 4 + node_id
            flat_occupancy = payload.new_zeros(batch, self.n_cells * 4)
            flat_occupancy.scatter_add_(1, flat_location, ones)
            occupancy = (
                flat_occupancy.view(batch, self.n_cells, 4)
                / max(length, 1)
            )[batch_index, cell_id]
            router_input = torch.cat(
                (payload, context, occupancy),
                dim=-1,
            )
            edge_logits = torch.einsum(
                "bpod,bpd->bpo",
                router_w[cell_id],
                router_input,
            ) + router_b[cell_id]
            edge_logits = (
                edge_logits
                + torch.einsum(
                    "bpoa,bpa->bpo",
                    address_edge_w[cell_id],
                    packet_address,
                )
            ).view(batch, packets, 4, 4)
            source_logits = edge_logits.gather(
                2,
                node_id[..., None, None].expand(-1, -1, 1, 4),
            ).squeeze(2)
            source_logits = source_logits.masked_fill(
                ~adjacency[node_id],
                torch.finfo(source_logits.dtype).min,
            )
            destination_choice = self._hard_sample(
                source_logits,
                temperature,
            )
            next_node_id = destination_choice.argmax(dim=-1)
            selected_route_gate = destination_choice.gather(
                -1,
                next_node_id.unsqueeze(-1),
            ).squeeze(-1)

            step_code = fixed_code(self.relays, 4, device)[relay]
            step_state = torch.einsum(
                "bpda,a->bpd",
                step_w[cell_id],
                step_code,
            )
            selected_address_state = torch.einsum(
                "bpda,bpa->bpd",
                address_state_weights[cell_id],
                packet_address,
            )
            base = (
                self._selected_cell_norm(
                    payload + context,
                    cell_id,
                )
                + selected_address_state
                + step_state
            )
            selected_node_input = (
                base + node_embeddings[cell_id, next_node_id]
            )
            selected_delta = torch.tanh(
                torch.einsum(
                    "bpdi,bpi->bpd",
                    node_w[cell_id, next_node_id],
                    selected_node_input,
                )
                + node_b[cell_id, next_node_id]
            )
            local_payload = self._selected_cell_norm(
                payload
                + selected_route_gate.unsqueeze(-1) * selected_delta,
                cell_id,
            )

            offset = self.active_cell_offsets[
                node_id,
                next_node_id,
            ]
            next_cell_id = (cell_id + offset).remainder(self.n_cells)
            self_scale = self.self_gates[cell_id].unsqueeze(-1)
            left_message = torch.sigmoid(
                self.neighbor_gates[cell_id, 0]
            ).unsqueeze(-1) * self.left_projection(local_payload)
            right_message = torch.sigmoid(
                self.neighbor_gates[cell_id, 1]
            ).unsqueeze(-1) * self.right_projection(local_payload)
            moved_message = torch.where(
                (offset > 0).unsqueeze(-1),
                left_message,
                right_message,
            )
            payload = self.coupling_norm(
                self_scale * local_payload
                + torch.where(
                    (offset != 0).unsqueeze(-1),
                    moved_message,
                    torch.zeros_like(moved_message),
                )
            )
            cell_id = next_cell_id
            node_id = next_node_id
            next_write = write_schedule[
                min(relay + 1, self.relays - 1)
            ]
            memory = update_path_relative_memory(
                memory,
                position_id=position_id,
                old_payload=old_payload,
                old_cell_id=old_cell_id,
                old_node_id=old_node_id,
                old_write_gate=write_schedule[relay],
                new_payload=payload,
                new_cell_id=cell_id,
                new_node_id=node_id,
                new_write_gate=next_write,
                relation_bias=relative_bias,
                receiver_length=length,
            )

        output_payload = payload.view(batch, length, paths, self.d_model)[
            :,
            observed_length:,
        ]
        output_cell = cell_id.view(batch, length, paths)[
            :,
            observed_length:,
        ]
        readout_w = torch.stack(
            [cell.address_to_readout.weight for cell in self.cells]
        )
        output_address = addresses[observed_length:][
            None,
            :,
            None,
        ].expand(batch, -1, paths, -1)
        address_out = torch.einsum(
            "blpda,blpa->blpd",
            readout_w[output_cell],
            output_address,
        )
        branch_logits = torch.einsum(
            "blpd,blpvd->blpv",
            output_payload + address_out,
            token_weights[output_cell],
        )
        branch_votes = F.softmax(self.output_votes[branch_cell], dim=0)
        return torch.einsum("p,blpv->blv", branch_votes, branch_logits)

    def _active_forward(
        self,
        observed,
        *,
        output_slots,
        temperature,
    ):
        """Carry one owned packet and persistent memory through every relay.

        Only the current steward cell and selected destination node transform
        each packet. The local decision is made afresh at every relay; no
        complete route history is stored.
        """

        batch, observed_length = observed.shape
        length = observed_length + output_slots
        device = observed.device
        addresses = fixed_code(length, 4, device)
        token_weights = torch.stack(
            [cell.token_embedding.weight for cell in self.cells]
        )
        address_state_weights = torch.stack(
            [cell.address_to_state.weight for cell in self.cells]
        )
        address_state = torch.einsum(
            "ta,cda->tcd",
            addresses,
            address_state_weights,
        )

        # Ingress is broad during training through the straight-through
        # surrogate, while the actual forward state has one steward. This is
        # the only all-cell packet view; relays below are selected/active-only.
        ingress_logits = self.output_votes.view(1, 1, self.n_cells).expand(
            batch,
            length,
            -1,
        )
        ingress = self._hard_sample(ingress_logits, temperature)
        cell_id = ingress.argmax(dim=-1)
        observed_candidates = token_weights[:, observed].permute(1, 2, 0, 3)
        blank_candidates = token_weights.mean(dim=1)[None, None].expand(
            batch,
            output_slots,
            -1,
            -1,
        )
        content_candidates = torch.cat(
            (observed_candidates, blank_candidates),
            dim=1,
        )
        payload = (
            ingress.unsqueeze(-1) * content_candidates
        ).sum(dim=2)
        selected_address_state = (
            ingress.unsqueeze(-1) * address_state[None]
        ).sum(dim=2)
        payload = self._selected_cell_norm(
            payload + selected_address_state,
            cell_id,
        )

        starts = torch.stack([cell.start_logits for cell in self.cells])
        start_logits = starts[cell_id]
        node_choice = self._hard_sample(start_logits, temperature)
        node_id = node_choice.argmax(dim=-1)
        selected_start_gate = node_choice.gather(
            -1,
            node_id.unsqueeze(-1),
        ).squeeze(-1)
        # Preserve a loss path into the selected cell's start-node router.
        payload = payload * selected_start_gate.unsqueeze(-1)

        relative_bias = torch.stack(
            [cell.relative_node_bias for cell in self.cells]
        )
        write_schedule = torch.ones(
            self.relays,
            batch,
            length,
            device=device,
            dtype=payload.dtype,
        )
        if output_slots:
            fractions = torch.arange(
                self.relays,
                device=device,
                dtype=payload.dtype,
            ) / max(self.relays - 1, 1)
            write_schedule[
                :,
                :,
                observed_length:,
            ] = fractions[:, None, None]

        memory = build_relative_memory(
            payload=payload,
            cell_id=cell_id,
            node_id=node_id,
            write_gate=write_schedule[0],
            relation_bias=relative_bias,
        )
        adjacency = self.cells[0].adjacency
        router_w = torch.stack([cell.router.weight for cell in self.cells])
        router_b = torch.stack([cell.router.bias for cell in self.cells])
        address_edge_w = torch.stack(
            [cell.address_to_edges.weight for cell in self.cells]
        )
        step_w = torch.stack(
            [cell.step_projection.weight for cell in self.cells]
        )
        node_embeddings = torch.stack(
            [cell.node_embedding.weight for cell in self.cells]
        )
        node_w = torch.stack(
            [
                torch.stack(
                    [transform.weight for transform in cell.node_transforms]
                )
                for cell in self.cells
            ]
        )
        node_b = torch.stack(
            [
                torch.stack(
                    [transform.bias for transform in cell.node_transforms]
                )
                for cell in self.cells
            ]
        )
        batch_index = torch.arange(device=device, end=batch)[:, None]
        slot_index = torch.arange(device=device, end=length)[None, :]
        ones = payload.new_ones(batch, length)

        for relay in range(self.relays):
            old_payload = payload
            old_cell_id = cell_id
            old_node_id = node_id
            context = memory.value[
                batch_index,
                slot_index,
                cell_id,
                node_id,
            ]

            flat_location = cell_id * 4 + node_id
            flat_occupancy = payload.new_zeros(batch, self.n_cells * 4)
            flat_occupancy.scatter_add_(1, flat_location, ones)
            occupancy = (
                flat_occupancy.view(batch, self.n_cells, 4)
                / max(length, 1)
            )[batch_index, cell_id]

            router_input = torch.cat(
                (payload, context, occupancy),
                dim=-1,
            )
            selected_router_w = router_w[cell_id]
            edge_logits = torch.einsum(
                "blod,bld->blo",
                selected_router_w,
                router_input,
            ) + router_b[cell_id]
            selected_address_edges = torch.einsum(
                "bloa,bla->blo",
                address_edge_w[cell_id],
                addresses[None].expand(batch, -1, -1),
            )
            edge_logits = (
                edge_logits + selected_address_edges
            ).view(batch, length, 4, 4)
            source_logits = edge_logits.gather(
                2,
                node_id[..., None, None].expand(-1, -1, 1, 4),
            ).squeeze(2)
            source_adjacency = adjacency[node_id]
            source_logits = source_logits.masked_fill(
                ~source_adjacency,
                torch.finfo(source_logits.dtype).min,
            )
            destination_choice = self._hard_sample(
                source_logits,
                temperature,
            )
            next_node_id = destination_choice.argmax(dim=-1)
            selected_route_gate = destination_choice.gather(
                -1,
                next_node_id.unsqueeze(-1),
            ).squeeze(-1)

            step_code = fixed_code(self.relays, 4, device)[relay]
            step_state = torch.einsum(
                "bcda,a->bcd",
                step_w[cell_id],
                step_code,
            )
            selected_address_state = address_state[
                slot_index,
                cell_id,
            ]
            base = (
                self._selected_cell_norm(
                    payload + context,
                    cell_id,
                )
                + selected_address_state
                + step_state
            )
            selected_node_input = (
                base + node_embeddings[cell_id, next_node_id]
            )
            selected_delta = torch.tanh(
                torch.einsum(
                    "bldi,bli->bld",
                    node_w[cell_id, next_node_id],
                    selected_node_input,
                )
                + node_b[cell_id, next_node_id]
            )
            local_payload = self._selected_cell_norm(
                payload
                + selected_route_gate.unsqueeze(-1) * selected_delta,
                cell_id,
            )

            offset = self.active_cell_offsets[
                node_id,
                next_node_id,
            ]
            next_cell_id = (cell_id + offset).remainder(self.n_cells)
            self_scale = self.self_gates[cell_id].unsqueeze(-1)
            left_gate = torch.sigmoid(
                self.neighbor_gates[cell_id, 0]
            ).unsqueeze(-1)
            right_gate = torch.sigmoid(
                self.neighbor_gates[cell_id, 1]
            ).unsqueeze(-1)
            left_message = left_gate * self.left_projection(local_payload)
            right_message = right_gate * self.right_projection(local_payload)
            moved_message = torch.where(
                (offset > 0).unsqueeze(-1),
                left_message,
                right_message,
            )
            mixed = self_scale * local_payload + torch.where(
                (offset != 0).unsqueeze(-1),
                moved_message,
                torch.zeros_like(moved_message),
            )
            payload = self.coupling_norm(mixed)
            cell_id = next_cell_id
            node_id = next_node_id

            next_write = write_schedule[
                min(relay + 1, self.relays - 1)
            ]
            memory = update_relative_memory(
                memory,
                old_payload=old_payload,
                old_cell_id=old_cell_id,
                old_node_id=old_node_id,
                old_write_gate=write_schedule[relay],
                new_payload=payload,
                new_cell_id=cell_id,
                new_node_id=node_id,
                new_write_gate=next_write,
                relation_bias=relative_bias,
            )

        readout_w = torch.stack(
            [cell.address_to_readout.weight for cell in self.cells]
        )
        output_cell = cell_id[:, observed_length:]
        output_payload = payload[:, observed_length:]
        output_address = addresses[observed_length:][None].expand(
            batch,
            -1,
            -1,
        )
        address_out = torch.einsum(
            "blda,bla->bld",
            readout_w[output_cell],
            output_address,
        )
        selected_token_weights = token_weights[output_cell]
        return torch.einsum(
            "bld,blvd->blv",
            output_payload + address_out,
            selected_token_weights,
        )

    def _reference_dense_forward(
        self,
        observed,
        *,
        output_slots,
        temperature,
        return_trace=False,
        node_top_k=None,
        active_cells=None,
        relay_limit=None,
    ):
        """Pre-optimization Arrow execution retained as a locked control."""

        batch, observed_length = observed.shape
        length = observed_length + output_slots
        executed_relays = (
            self.relays if relay_limit is None else relay_limit
        )
        addresses = fixed_code(length, 4, observed.device)

        token_weights = torch.stack(
            [cell.token_embedding.weight for cell in self.cells]
        )
        observed_h = token_weights[:, observed].permute(1, 0, 2, 3)
        mask = token_weights.mean(1)[None, :, None].expand(
            batch,
            -1,
            output_slots,
            -1,
        )
        content = torch.cat([observed_h, mask], 2)
        address_state_weights = torch.stack(
            [cell.address_to_state.weight for cell in self.cells]
        )
        address_state = torch.einsum(
            "ta,cda->ctd",
            addresses,
            address_state_weights,
        )
        h = self._cell_norm(content + address_state[None])
        cell_mask = None
        if active_cells is not None:
            cell_mask = torch.zeros(
                self.n_cells,
                device=h.device,
                dtype=h.dtype,
            )
            cell_mask[list(active_cells)] = 1.0
            if not bool(cell_mask.any()):
                raise ValueError("active_cells must retain at least one cell")
            cell_scale = cell_mask.view(1, self.n_cells, 1, 1)
            h = h * cell_scale

        starts = torch.stack([cell.start_logits for cell in self.cells])
        q = F.softmax(starts / 0.35, -1)[None, :, None].expand(
            batch,
            -1,
            length,
            -1,
        )
        locations = [q] if return_trace else None
        transition_entropies = [] if return_trace else None
        adjacency = self.cells[0].adjacency.view(
            1,
            1,
            1,
            self.n_nodes,
            self.n_nodes,
        )
        rel = relative_code(length, h.device)

        for relay in range(executed_relays):
            write = torch.ones(
                length,
                device=h.device,
                dtype=h.dtype,
            )
            write[observed_length:] = (
                relay / max(self.relays - 1, 1)
            )
            relative_biases = torch.stack(
                [cell.relative_node_bias for cell in self.cells]
            )
            rel_bias = torch.einsum(
                "ijr,cnr->cijn",
                rel,
                relative_biases,
            )
            scores = (
                q.clamp_min(1e-9).log()[:, :, None, :, :]
                + rel_bias[None]
            )
            scores = scores + write.clamp_min(1e-9).log()[
                None,
                None,
                None,
                :,
                None,
            ]
            alpha = scores.softmax(3)
            node_memory = torch.einsum(
                "bcijn,bcjd->bcind",
                alpha,
                h,
            )
            context = (q.unsqueeze(-1) * node_memory).sum(3)
            occupancy = q.mean(2, keepdim=True).expand_as(q)

            router_w = torch.stack(
                [cell.router.weight for cell in self.cells]
            )
            router_b = torch.stack(
                [cell.router.bias for cell in self.cells]
            )
            router_input = torch.cat(
                [h, context, occupancy],
                -1,
            )
            edge_logits = torch.einsum(
                "bcti,coi->bcto",
                router_input,
                router_w,
            ) + router_b[None, :, None]
            address_edge_w = torch.stack(
                [cell.address_to_edges.weight for cell in self.cells]
            )
            address_edges = torch.einsum(
                "ta,coa->cto",
                addresses,
                address_edge_w,
            )
            edge_logits = (
                edge_logits + address_edges[None]
            ).view(
                batch,
                self.n_cells,
                length,
                self.n_nodes,
                self.n_nodes,
            )
            transition = F.softmax(
                edge_logits.masked_fill(~adjacency, -1e9) / temperature,
                -1,
            )
            if return_trace:
                edge_entropy = -(
                    transition.clamp_min(1e-9)
                    * transition.clamp_min(1e-9).log()
                ).sum(-1)
                transition_entropies.append(
                    (q * edge_entropy).sum(-1).mean()
                )
            q_next = torch.einsum(
                "bcts,bctsd->bctd",
                q,
                transition,
            )
            if node_top_k is not None and node_top_k < q_next.shape[-1]:
                retained, retained_index = q_next.topk(
                    node_top_k,
                    dim=-1,
                )
                sparse_q = torch.zeros_like(q_next).scatter(
                    -1,
                    retained_index,
                    retained,
                )
                q_next = sparse_q / sparse_q.sum(
                    -1,
                    keepdim=True,
                ).clamp_min(1e-9)

            step_code = fixed_code(
                self.relays,
                4,
                h.device,
            )[relay]
            step_w = torch.stack(
                [cell.step_projection.weight for cell in self.cells]
            )
            step_state = torch.einsum(
                "a,cda->cd",
                step_code,
                step_w,
            )
            base = (
                self._cell_norm(h + context)
                + address_state[None]
                + step_state[None, :, None]
            )
            node_embeddings = torch.stack(
                [cell.node_embedding.weight for cell in self.cells]
            )
            node_inputs = (
                base.unsqueeze(3)
                + node_embeddings[None, :, None]
            )
            node_w = torch.stack(
                [
                    torch.stack(
                        [
                            transform.weight
                            for transform in cell.node_transforms
                        ]
                    )
                    for cell in self.cells
                ]
            )
            node_b = torch.stack(
                [
                    torch.stack(
                        [
                            transform.bias
                            for transform in cell.node_transforms
                        ]
                    )
                    for cell in self.cells
                ]
            )
            destination_h = torch.tanh(
                torch.einsum(
                    "bctni,cnoi->bctno",
                    node_inputs,
                    node_w,
                )
                + node_b[None, :, None]
            )
            local_h = self._cell_norm(
                h
                + (
                    q_next.unsqueeze(-1) * destination_h
                ).sum(3)
            )
            if cell_mask is not None:
                local_h = local_h * cell_scale
            q = q_next
            if return_trace:
                locations.append(q)

            radius = self.neighbors // 2
            gates = torch.sigmoid(
                self.neighbor_gates
            )[None, :, None, :, None]
            mixed = (
                self.self_gates[None, :, None, None] * local_h
            )
            if self.destination_carry:
                # Sources compute each projected message once and carry it to
                # neighboring destinations. The receiver still owns the gate
                # that accepts the arrival. This is algebraically Arrow's
                # original destination-specialization rule.
                left_source = self.left_projection(local_h)
                right_source = self.right_projection(local_h)
                left_arrival = torch.stack(
                    [
                        torch.roll(
                            left_source,
                            shifts=distance,
                            dims=1,
                        )
                        for distance in range(1, radius + 1)
                    ]
                ).mean(0)
                right_arrival = torch.stack(
                    [
                        torch.roll(
                            right_source,
                            shifts=-distance,
                            dims=1,
                        )
                        for distance in range(1, radius + 1)
                    ]
                ).mean(0)
                mixed = (
                    mixed
                    + gates[..., 0, :] * left_arrival
                    + gates[..., 1, :] * right_arrival
                )
            elif self.source_carry:
                left_source = (
                    gates[..., 0, :]
                    * self.left_projection(local_h)
                )
                right_source = (
                    gates[..., 1, :]
                    * self.right_projection(local_h)
                )
                mixed = mixed + torch.stack(
                    [
                        torch.roll(
                            left_source,
                            shifts=distance,
                            dims=1,
                        )
                        for distance in range(1, radius + 1)
                    ]
                ).mean(0)
                mixed = mixed + torch.stack(
                    [
                        torch.roll(
                            right_source,
                            shifts=-distance,
                            dims=1,
                        )
                        for distance in range(1, radius + 1)
                    ]
                ).mean(0)
            else:
                left_state = torch.stack(
                    [
                        torch.roll(
                            local_h,
                            shifts=distance,
                            dims=1,
                        )
                        for distance in range(1, radius + 1)
                    ]
                ).mean(0)
                right_state = torch.stack(
                    [
                        torch.roll(
                            local_h,
                            shifts=-distance,
                            dims=1,
                        )
                        for distance in range(1, radius + 1)
                    ]
                ).mean(0)
                left = self.left_projection(left_state)
                right = self.right_projection(right_state)
                mixed = (
                    mixed
                    + gates[..., 0, :] * left
                    + gates[..., 1, :] * right
                )
            h = self.coupling_norm(mixed)
            if cell_mask is not None:
                h = h * cell_scale

        readout_w = torch.stack(
            [cell.address_to_readout.weight for cell in self.cells]
        )
        address_out = torch.einsum(
            "ta,cda->ctd",
            addresses,
            readout_w,
        )
        read_h = (
            h[:, :, observed_length:]
            + address_out[None, :, observed_length:]
        )
        if cell_mask is not None:
            read_h = read_h * cell_scale
        logits = torch.einsum(
            "bctd,cvd->bctv",
            read_h,
            token_weights,
        )
        votes = F.softmax(self.output_votes, 0)
        if cell_mask is not None:
            votes = votes * cell_mask
            votes = votes / votes.sum().clamp_min(1e-9)
        combined_logits = torch.einsum("c,bctv->btv", votes, logits)
        if not return_trace:
            return combined_logits
        return combined_logits, {
            "locations": torch.stack(locations, dim=1),
            "transition_entropy": torch.stack(transition_entropies),
            "cell_logits": logits,
            "output_votes": votes,
            "output_states": read_h,
            "neighbor_gates": torch.sigmoid(self.neighbor_gates),
            "self_gates": self.self_gates,
            "active_cells": cell_mask,
        }

    def forward(
        self,
        observed,
        *,
        output_slots=7,
        temperature=0.35,
        return_trace=False,
        node_top_k=None,
        active_cells=None,
        relay_limit=None,
    ):
        if relay_limit is not None and (
            isinstance(relay_limit, bool)
            or not isinstance(relay_limit, int)
            or not 1 <= relay_limit <= self.relays
        ):
            raise ValueError("relay_limit must be in [1, relays]")
        if (return_trace or active_cells is not None) and self.persistent_active:
            raise ValueError(
                "dense Arrow trace is unavailable for persistent-active mode"
            )
        if return_trace or relay_limit is not None:
            return self._reference_dense_forward(
                observed,
                output_slots=output_slots,
                temperature=temperature,
                return_trace=return_trace,
                node_top_k=node_top_k,
                active_cells=active_cells,
                relay_limit=relay_limit,
            )
        if node_top_k is not None or active_cells is not None:
            return self._reference_dense_forward(
                observed,
                output_slots=output_slots,
                temperature=temperature,
                return_trace=False,
                node_top_k=node_top_k,
                active_cells=active_cells,
                relay_limit=None,
            )
        if self.persistent_active:
            if self.active_paths > 1:
                return self._cooperative_active_forward(
                    observed,
                    output_slots=output_slots,
                    temperature=temperature,
                )
            return self._active_forward(
                observed,
                output_slots=output_slots,
                temperature=temperature,
            )
        if self.reference_dense:
            return self._reference_dense_forward(
                observed,
                output_slots=output_slots,
                temperature=temperature,
                return_trace=False,
                node_top_k=None,
                active_cells=None,
                relay_limit=None,
            )
        batch, observed_length = observed.shape
        length = observed_length + output_slots
        addresses = fixed_code(length, 4, observed.device)

        token_weights = torch.stack([cell.token_embedding.weight for cell in self.cells])
        observed_h = token_weights[:, observed].permute(1, 0, 2, 3)
        mask = token_weights.mean(1)[None, :, None].expand(batch, -1, output_slots, -1)
        content = torch.cat([observed_h, mask], 2)
        address_state_weights = torch.stack([cell.address_to_state.weight for cell in self.cells])
        address_state = torch.einsum("ta,cda->ctd", addresses, address_state_weights)
        h = self._cell_norm(content + address_state[None])

        starts = torch.stack([cell.start_logits for cell in self.cells])
        q = F.softmax(starts / 0.35, -1)[None, :, None].expand(batch, -1, length, -1)
        adjacency = self.cells[0].adjacency.view(
            1,
            1,
            1,
            self.n_nodes,
            self.n_nodes,
        )
        rel = relative_code(length, h.device)
        relative_biases = torch.stack(
            [cell.relative_node_bias for cell in self.cells]
        )
        rel_bias = torch.einsum(
            "ijr,cnr->cijn",
            rel,
            relative_biases,
        )
        router_w = torch.stack(
            [cell.router.weight for cell in self.cells]
        )
        router_b = torch.stack(
            [cell.router.bias for cell in self.cells]
        )
        if self.routing_dim is not None:
            bottleneck_w = torch.stack(
                [cell.routing_bottleneck.weight for cell in self.cells]
            )
            bottleneck_b = torch.stack(
                [cell.routing_bottleneck.bias for cell in self.cells]
            )
        else:
            bottleneck_w = None
            bottleneck_b = None
        address_edge_w = torch.stack(
            [cell.address_to_edges.weight for cell in self.cells]
        )
        address_edges = torch.einsum(
            "ta,coa->cto",
            addresses,
            address_edge_w,
        )
        step_w = torch.stack(
            [cell.step_projection.weight for cell in self.cells]
        )
        step_codes = fixed_code(self.relays, 4, h.device)
        step_states = torch.einsum(
            "ra,cda->rcd",
            step_codes,
            step_w,
        )
        node_embeddings = torch.stack(
            [cell.node_embedding.weight for cell in self.cells]
        )
        node_w = torch.stack(
            [
                torch.stack(
                    [transform.weight for transform in cell.node_transforms]
                )
                for cell in self.cells
            ]
        )
        node_b = torch.stack(
            [
                torch.stack(
                    [transform.bias for transform in cell.node_transforms]
                )
                for cell in self.cells
            ]
        )
        write_schedule = torch.ones(
            self.relays,
            length,
            device=h.device,
            dtype=h.dtype,
        )
        write_schedule[
            :,
            observed_length:,
        ] = (
            torch.arange(
                self.relays,
                device=h.device,
                dtype=h.dtype,
            )[:, None]
            / max(self.relays - 1, 1)
        )
        radius = self.neighbors // 2
        gates = torch.sigmoid(
            self.neighbor_gates
        )[None, :, None, :, None]

        for relay in range(self.relays):
            write = write_schedule[relay]
            if self.receiver_memory_backend == "linear_scan":
                writer_mass = write[None, None, :].expand(batch, self.n_cells, length)
                context = linear_scan_receiver_context(
                    h, q, writer_mass, relative_biases
                )
            else:
                scores = q.clamp_min(1e-9).log()[:, :, None, :, :] + rel_bias[None]
                scores = scores + write.clamp_min(1e-9).log()[None, None, None, :, None]
                alpha = scores.softmax(3)
                node_memory = torch.einsum("bcijn,bcjd->bcind", alpha, h)
                context = (q.unsqueeze(-1) * node_memory).sum(3)
            occupancy = q.mean(2, keepdim=True).expand_as(q)

            router_input = torch.cat([h, context, occupancy], -1)
            if bottleneck_w is not None:
                router_input = torch.tanh(
                    torch.einsum("bcti,coi->bcto", router_input, bottleneck_w)
                    + bottleneck_b[None, :, None]
                )
            edge_logits = torch.einsum("bcti,coi->bcto", router_input, router_w) + router_b[None, :, None]
            edge_logits = (
                edge_logits + address_edges[None]
            ).view(
                batch,
                self.n_cells,
                length,
                self.n_nodes,
                self.n_nodes,
            )
            transition = F.softmax(edge_logits.masked_fill(~adjacency, -1e9) / temperature, -1)
            q_next = torch.einsum("bcts,bctsd->bctd", q, transition)

            step_state = step_states[relay]
            base = self._cell_norm(h + context) + address_state[None] + step_state[None, :, None]
            node_inputs = base.unsqueeze(3) + node_embeddings[None, :, None]
            destination_h = torch.tanh(
                torch.einsum("bctni,cnoi->bctno", node_inputs, node_w)
                + node_b[None, :, None]
            )
            local_h = self._cell_norm(h + (q_next.unsqueeze(-1) * destination_h).sum(3))
            q = q_next

            mixed = self.self_gates[None, :, None, None] * local_h
            if self.destination_carry:
                left_source = self.left_projection(local_h)
                right_source = self.right_projection(local_h)
                left_arrival = torch.stack(
                    [
                        torch.roll(
                            left_source,
                            shifts=distance,
                            dims=1,
                        )
                        for distance in range(1, radius + 1)
                    ]
                ).mean(0)
                right_arrival = torch.stack(
                    [
                        torch.roll(
                            right_source,
                            shifts=-distance,
                            dims=1,
                        )
                        for distance in range(1, radius + 1)
                    ]
                ).mean(0)
                mixed = (
                    mixed
                    + gates[..., 0, :] * left_arrival
                    + gates[..., 1, :] * right_arrival
                )
            elif self.source_carry:
                # Source-owned carry: each cell transforms the state it owns,
                # applies its own outgoing gate, and hands that exact state to
                # neighboring cells. Destination states are the synchronous
                # commit of those source handoffs. No token state is rebuilt
                # from an embedding or an independently recomputed cell copy.
                left_source = (
                    gates[..., 0, :] * self.left_projection(local_h)
                )
                right_source = (
                    gates[..., 1, :] * self.right_projection(local_h)
                )
                mixed = mixed + torch.stack([
                    torch.roll(
                        left_source,
                        shifts=distance,
                        dims=1,
                    )
                    for distance in range(1, radius + 1)
                ]).mean(0)
                mixed = mixed + torch.stack([
                    torch.roll(
                        right_source,
                        shifts=-distance,
                        dims=1,
                    )
                    for distance in range(1, radius + 1)
                ]).mean(0)
            else:
                # Validated Arrow reference: every destination reconstructs
                # its next cell state by reading a dense neighbor aggregate
                # and applying the destination's gates.
                left_state = torch.stack([
                    torch.roll(
                        local_h,
                        shifts=distance,
                        dims=1,
                    )
                    for distance in range(1, radius + 1)
                ]).mean(0)
                right_state = torch.stack([
                    torch.roll(
                        local_h,
                        shifts=-distance,
                        dims=1,
                    )
                    for distance in range(1, radius + 1)
                ]).mean(0)
                left = self.left_projection(left_state)
                right = self.right_projection(right_state)
                mixed = (
                    mixed
                    + gates[..., 0, :] * left
                    + gates[..., 1, :] * right
                )
            h = self.coupling_norm(mixed)

        readout_w = torch.stack([cell.address_to_readout.weight for cell in self.cells])
        address_out = torch.einsum("ta,cda->ctd", addresses, readout_w)
        read_h = h[:, :, observed_length:] + address_out[None, :, observed_length:]
        logits = torch.einsum("bctd,cvd->bctv", read_h, token_weights)
        votes = F.softmax(self.output_votes, 0)
        return torch.einsum("c,bctv->btv", votes, logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--cells", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument(
        "--d-model",
        type=int,
        default=8,
        help=(
            "per-node feature width; widens each node's own representation "
            "without changing node count or relative_node_bias's (nodes, "
            "nodes) shape -- a different scaling axis than --nodes"
        ),
    )
    parser.add_argument(
        "--routing-dim",
        type=int,
        default=None,
        help=(
            "bottleneck width of the router's input, applied via an extra "
            "tanh(Linear) projection before the final edge-logit layer; "
            "decouples routing's decision bandwidth from d_model (whose "
            "growth otherwise widens the router's input for free without "
            "changing its n_nodes*n_nodes output). default None preserves "
            "the original direct router with no bottleneck"
        ),
    )
    parser.add_argument(
        "--receiver-memory-backend",
        choices=["attention", "linear_scan"],
        default="attention",
        help=(
            "attention: the original O(length^2) relation-aware memory "
            "read (unchanged default, byte-identical to every prior run). "
            "linear_scan: gator.models.memory.linear_scan_receiver_context, "
            "a genuinely O(length) bidirectional decayed-scan stand-in "
            "(reinterprets relative_node_bias as decay-rate/mix "
            "parameters instead of a relative-position code -- a "
            "different computation through the same parameter slot, not "
            "a numerically-equivalent drop-in). Only wired into the "
            "default vectorized forward path (incompatible with "
            "--persistent-active/--reference-dense)"
        ),
    )
    parser.add_argument(
        "--relays",
        type=int,
        default=4,
        help="shared recurrent relay depth; changes compute, not parameters",
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--portable-batches",
        action="store_true",
        help=(
            "sample training/evaluation examples with a dedicated CPU RNG "
            "so CPU and CUDA runs receive the same batch stream"
        ),
    )
    parser.add_argument(
        "--carry-state",
        action="store_true",
        help=(
            "carry source-projected messages while preserving the receiving "
            "destination's original Arrow gate ownership"
        ),
    )
    parser.add_argument(
        "--destination-carry",
        action="store_true",
        help=(
            "deprecated alias for --carry-state"
        ),
    )
    parser.add_argument(
        "--source-carry",
        action="store_true",
        help=(
            "diagnostic rejected variant: source cells own outgoing gates; "
            "this changes Arrow's credit assignment"
        ),
    )
    parser.add_argument(
        "--persistent-active",
        action="store_true",
        help=(
            "carry packet/node memory across relays and evaluate only each "
            "packet's selected steward cell and destination node"
        ),
    )
    parser.add_argument(
        "--active-paths",
        type=int,
        default=None,
        help=(
            "cooperative carried traces per token; defaults to one trace "
            "per cell, while 1 reproduces the winner-take-all ablation"
        ),
    )
    parser.add_argument(
        "--reference-dense",
        action="store_true",
        help=(
            "run the locked pre-optimization dense Arrow operation order "
            "for same-machine learning and throughput controls"
        ),
    )
    parser.add_argument(
        "--node-top-k",
        type=int,
        default=None,
        help=(
            "retain and renormalize only this many internal node locations "
            "after every relay; intended for dense-to-sparse continuation"
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="load model and available optimizer state from a checkpoint",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="when resuming, load model weights but start fresh optimizer state",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="optional checkpoint path; defaults to the historical filename",
    )
    parser.add_argument(
        "--adamw-weight-trace-every",
        type=int,
        default=0,
        help=(
            "record AdamW weight/update geometry at this step interval; "
            "zero disables tracing"
        ),
    )
    parser.add_argument(
        "--adamw-weight-trace-dir",
        type=Path,
        default=None,
        help=(
            "trace output directory; when omitted, an enabled trace uses "
            "<checkpoint parent>/adamw_weight_trace"
        ),
    )
    parser.add_argument(
        "--adamw-weight-trace-save-tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "save full weights, interval deltas, clipped-gradient summaries, "
            "per-step update RMS, and Adam moments at every trace interval"
        ),
    )
    parser.add_argument(
        "--adamw-weight-trace-step-statistics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "also aggregate exact gradients and update-path geometry after "
            "every optimizer step; disabled by default because the Python "
            "tensor walk can roughly halve CPU training throughput"
        ),
    )
    add_optimization_arguments(parser)
    args = parser.parse_args()
    if args.node_top_k is not None and not 1 <= args.node_top_k <= args.nodes:
        parser.error("--node-top-k must be in [1, --nodes]")
    if args.node_top_k is not None and args.persistent_active:
        parser.error("--node-top-k and --persistent-active are exclusive")
    if args.source_carry and (args.carry_state or args.destination_carry):
        parser.error(
            "--source-carry is exclusive with corrected destination carry"
        )
    if args.adamw_weight_trace_every < 0:
        parser.error("--adamw-weight-trace-every must be nonnegative")
    if args.adamw_weight_trace_every > 0 and args.optimizer != "adamw":
        parser.error("--adamw-weight-trace-every requires --optimizer adamw")
    # Every flag, fully resolved, as the first line of every log -- so a
    # run's exact reproduction recipe is never dependent on a separately-kept
    # launch script surviving. See Papers/GRANT_EVIDENCE_STANDARDS.md rule 2.
    print(
        "config="
        + json.dumps(
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in sorted(vars(args).items())
            },
            sort_keys=True,
        ),
        flush=True,
    )
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorizedGatorCellRing(
        n_cells=args.cells,
        n_nodes=args.nodes,
        relays=args.relays,
        neighbors=args.neighbors,
        d_model=args.d_model,
        routing_dim=args.routing_dim,
        receiver_memory_backend=args.receiver_memory_backend,
        carry_state=args.carry_state,
        destination_carry=args.destination_carry,
        source_carry=args.source_carry,
        persistent_active=args.persistent_active,
        active_paths=args.active_paths,
        reference_dense=args.reference_dense,
    ).to(device)
    train_model = torch.compile(model) if args.compile else model
    sampling_device = (
        torch.device("cpu") if args.portable_batches else device
    )
    batch_generator = torch.Generator(
        device=sampling_device
    ).manual_seed(args.seed + 1_000_003)
    optimization = optimization_config_from_args(args)
    engine = OptimizationEngine(model, optimization)
    resumed_step = 0
    if args.resume is not None:
        resume_payload = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(resume_payload.get("model", resume_payload))
        resumed_step = int(resume_payload.get("step", 0))
        if "batch_generator_state" in resume_payload:
            batch_generator.set_state(
                resume_payload["batch_generator_state"].cpu()
            )
        if not args.reset_optimizer:
            if "training" in resume_payload:
                engine.load_state_dict(
                    resume_payload["training"],
                    load_scheduler=False,
                    learning_rate=optimization.learning_rate,
                )
            elif "optimizer" in resume_payload:
                engine.optimizer.load_state_dict(
                    resume_payload["optimizer"]
                )
                for parameter_group in engine.optimizer.param_groups:
                    parameter_group["lr"] = optimization.learning_rate
                engine.step_number = resumed_step
    checkpoint_path = args.checkpoint or Path(
        f"gator_vectorized_c{args.cells}_n{args.nodes}"
        f"_d{args.neighbors}"
        f"{'_source_carry' if args.source_carry else ''}"
        f"{'_carry' if args.carry_state or args.destination_carry else ''}"
        f"{'_persistent_active' if args.persistent_active else ''}"
        f"{f'_k{model.active_paths}' if args.persistent_active else ''}"
        f"{'_reference' if args.reference_dense else ''}"
        "_checkpoint.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    adamw_tracer = None
    if args.adamw_weight_trace_every > 0:
        trace_dir = (
            args.adamw_weight_trace_dir
            or checkpoint_path.parent / "adamw_weight_trace"
        )
        adamw_tracer = AdamWWeightTracer(
            model,
            engine.optimizer,
            trace_dir,
            interval=args.adamw_weight_trace_every,
            save_tensors=args.adamw_weight_trace_save_tensors,
            step_statistics=args.adamw_weight_trace_step_statistics,
        )
    print(f"device={device} parameters={model.parameter_count} "
          f"cells={args.cells} nodes={args.nodes} d_model={args.d_model} "
          f"routing_dim={args.routing_dim} "
          f"receiver_memory_backend={args.receiver_memory_backend} "
          f"relays={args.relays} neighbors={args.neighbors} "
          f"compiled={args.compile} optimizer={optimization.name} "
          f"carry_state={args.carry_state or args.destination_carry} "
          f"carry_owner="
          f"{'source' if args.source_carry else ('destination' if args.carry_state or args.destination_carry else 'none')} "
          f"persistent_active={args.persistent_active} "
          f"active_paths={model.active_paths} "
          f"reference_dense={args.reference_dense} "
          f"portable_batches={args.portable_batches} "
          f"node_top_k={args.node_top_k} resumed_step={resumed_step} "
          f"lr={optimization.learning_rate} "
          f"optimizer_epsilon={optimization.epsilon:g} "
          f"scheduler={optimization.scheduler.name}")
    if device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    interval_started = started
    interval_step = 0
    for step in range(args.steps):
        observed, targets = make_prefix_batch(
            args.batch_size,
            7,
            device,
            generator=batch_generator,
            sampling_device=sampling_device,
        )
        total_step = resumed_step + step
        temperature = max(0.35, 1.25 * (0.997**total_step))
        logits = train_model(
            observed,
            output_slots=7,
            temperature=temperature,
            node_top_k=args.node_top_k,
        )
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        if adamw_tracer is None:
            step_result = engine.step(loss)
        else:
            # Bypass engine.step()'s bundled zero_grad/backward/clip/step so
            # the tracer can sit between clipping and the actual AdamW
            # update, matching gator_linear_prefix_train.py's pattern.
            engine.zero_grad()
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    optimization.gradient_clip,
                    error_if_nonfinite=True,
                )
            )
            adamw_tracer.before_optimizer_step(
                step=engine.step_number + 1,
                loss=float(loss.detach()),
                preclip_gradient_norm=gradient_norm,
                gradient_clip=optimization.gradient_clip,
                force_snapshot=(engine.step_number + 1 == args.steps),
            )
            engine.optimizer.step()
            adamw_tracer.after_optimizer_step()
            if engine.scheduler is not None:
                engine.scheduler.step()
            engine.step_number += 1
            step_result = StepResult(
                step=engine.step_number,
                loss=float(loss.detach()),
                learning_rate=engine.learning_rate,
                gradient_norm=gradient_norm,
            )
            trace = adamw_tracer.maybe_snapshot(
                engine.step_number,
                context={
                    "learning_rate": engine.learning_rate,
                    "temperature": float(temperature),
                },
            )
            if trace is not None:
                print(trace.log_line(), flush=True)
        if (step + 1) % 1000 == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            interval_rate = (step + 1 - interval_step) / (
                now - interval_started
            )
            result = evaluate(
                model,
                7,
                device,
                batch_size=args.log_eval_batch_size,
                model_kwargs={"node_top_k": args.node_top_k},
                sampling_device=sampling_device,
            )
            print(f"step={total_step + 1} phase_step={step+1} "
                  f"steps_per_second={interval_rate:.2f} "
                  f"correct={result['correct']:.2f}/7 acc={result['accuracy']:.3f} "
                  f"loss={result['loss']:.4f} lr={step_result.learning_rate:.6g} "
                  f"grad_norm={step_result.gradient_norm:.4f}", flush=True)
            interval_started = time.perf_counter()
            interval_step = step + 1
            torch.save({"step": total_step + 1, "model": model.state_dict(),
                        "training": engine.state_dict(),
                        "batch_generator_state": batch_generator.get_state()},
                       checkpoint_path)
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = evaluate(
        model,
        7,
        device,
        batch_size=args.eval_batch_size,
        model_kwargs={"node_top_k": args.node_top_k},
        sampling_device=sampling_device,
    )
    print(f"final correct={result['correct']:.2f}/7 acc={result['accuracy']:.3f} "
          f"exact={result['exact']:.3f} loss={result['loss']:.4f}")
    print(f"runtime_seconds={elapsed:.2f} steps_per_second={args.steps / elapsed:.2f}")


if __name__ == "__main__":
    main()
