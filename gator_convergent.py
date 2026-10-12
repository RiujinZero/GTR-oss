"""Flat 40-node Gator with ten persistent convergent payload lanes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from gator import OptimizationEngine
from gator.cli import (
    add_optimization_arguments,
    optimization_config_from_args,
)
from gator.models.dense_kernels import (
    cell_linear,
    folded_cell_node_linear,
)
from gator.models.adaptive_frontier import adaptive_uncertainty_top_k
from gator.models.active_node_compute import SharedTrunkNodeAdapters
from gator.models.hierarchical_ingress import HierarchicalPackedIngress
from gator.models.packed_frontier import (
    PackedNodeFrontier,
    packed_node_occupancy,
    transport_packed_frontier,
)
from gator.models.packed_convolution_memory import (
    packed_convolution_node_context,
)
from gator.models.opaque_fft_memory import (
    opaque_packed_fft_node_context,
)
from gator.models.packed_memory import packed_convergent_node_context
from gtr_a_experiments import fixed_code, relative_code
from parallel_prefix_benchmark import evaluate, make_prefix_batch


def _padded_grouped_mm_width(size: int, dtype: torch.dtype) -> int:
    """Return the next feature width with a 16-byte grouped-MM stride."""

    if size <= 0:
        raise ValueError("grouped-MM feature width must be positive")
    alignment = 16 // dtype.itemsize
    return ((size + alignment - 1) // alignment) * alignment


class _Fp8TiedReadout(torch.autograd.Function):
    """Tied-embedding vocab readout (readout @ weight.T + bias) with an fp8
    forward matmul (dynamic per-tensor scaling via torchao) and a bf16
    backward pass. fp8 only pays off on large matmuls (see the 2026-07-28
    feasibility benchmarks: ~0.6x at Gator's node-sized matmuls, ~1.4x at
    the vocab-sized readout) and carries a real ~13x-worse relative-error
    cost vs bf16 - backward stays bf16 here since gradient precision
    sensitivity wasn't benchmarked and this is a forward-speed experiment,
    not a validated production path.
    """

    @staticmethod
    def forward(ctx, readout: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated

        orig_shape = readout.shape
        flat = readout.reshape(-1, orig_shape[-1]).contiguous()

        # weight is [vocab_size, embedding_dim]; the matmul needs weight.T
        # ([embedding_dim, vocab_size]) - transpose BEFORE casting to fp8,
        # not after (casting weight itself and column-major-tricking it
        # keeps the wrong [vocab_size, embedding_dim] shape).
        weight_t = weight.t()
        r_scale = tensor_to_scale(flat, torch.float8_e4m3fn)
        w_scale = tensor_to_scale(weight_t, torch.float8_e4m3fn)
        r_fp8 = to_fp8_saturated(flat.float() * r_scale, torch.float8_e4m3fn)
        # torch._scaled_mm requires the second operand column-major.
        w_fp8 = to_fp8_saturated(weight_t.float() * w_scale, torch.float8_e4m3fn)
        w_fp8_colmajor = w_fp8.t().contiguous().t()

        out = torch._scaled_mm(
            r_fp8, w_fp8_colmajor,
            scale_a=(1.0 / r_scale).detach().clone(),
            scale_b=(1.0 / w_scale).detach().clone(),
            out_dtype=torch.bfloat16,
        )
        logits = out.reshape(*orig_shape[:-1], weight.shape[0]).float() + bias
        ctx.save_for_backward(readout, weight)
        return logits

    @staticmethod
    def backward(ctx, grad_output):
        readout, weight = ctx.saved_tensors
        grad_output = grad_output.to(torch.bfloat16)
        readout_bf16 = readout.to(torch.bfloat16)
        weight_bf16 = weight.to(torch.bfloat16)
        grad_readout = (grad_output @ weight_bf16).to(readout.dtype)
        grad_weight = torch.einsum(
            "...o,...e->oe", grad_output, readout_bf16
        ).to(weight.dtype)
        grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0).to(torch.float32)
        return grad_readout, grad_weight, grad_bias


def fp8_tied_readout(readout: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """--fp8-vocab-readout entry point: see _Fp8TiedReadout for caveats."""

    return _Fp8TiedReadout.apply(readout, weight, bias)


def connection_destinations(
    *,
    n_nodes: int,
    degree: int,
    layout: str,
    seed: int,
    rewire_probability: float = 1.0,
) -> torch.Tensor:
    """Build a deterministic table of legal outgoing graph connections."""

    if degree <= 0 or degree > n_nodes:
        raise ValueError("degree must be in [1, n_nodes]")
    if not 0.0 <= rewire_probability <= 1.0:
        raise ValueError("rewire_probability must be in [0, 1]")
    if layout == "dense":
        if degree != n_nodes:
            raise ValueError("dense layout requires degree == n_nodes")
        return torch.arange(n_nodes).expand(n_nodes, -1).clone()
    if layout == "ring":
        offsets = [0]
        distance = 1
        while len(offsets) < degree:
            offsets.append(distance)
            if len(offsets) < degree:
                offsets.append(-distance)
            distance += 1
        source = torch.arange(n_nodes)[:, None]
        return (
            source + torch.tensor(offsets, dtype=torch.long)[None]
        ).remainder(n_nodes)
    if layout == "small_world":
        if degree < 3:
            raise ValueError("small_world layout requires degree >= 3")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        ring = connection_destinations(
            n_nodes=n_nodes,
            degree=degree,
            layout="ring",
            seed=seed,
            rewire_probability=rewire_probability,
        )
        rows = []
        for source in range(n_nodes):
            # Self preserves local computation and +1 guarantees a directed
            # backbone. Other ring edges are independently replaced by fixed
            # nonlocal shortcuts, giving a controllable Watts-Strogatz-like
            # continuum without changing parameter count.
            selected = ring[source, :2].tolist()
            for original in ring[source, 2:].tolist():
                should_rewire = (
                    original in selected
                    or bool(
                        torch.rand((), generator=generator)
                        < rewire_probability
                    )
                )
                if not should_rewire:
                    selected.append(original)
                    continue
                candidates = torch.tensor(
                    [
                        node
                        for node in range(n_nodes)
                        if node not in selected
                    ],
                    dtype=torch.long,
                )
                choice = torch.randint(
                    candidates.numel(),
                    (),
                    generator=generator,
                )
                selected.append(int(candidates[choice]))
            rows.append(selected)
        return torch.tensor(rows, dtype=torch.long)
    raise ValueError(
        "connection_layout must be ring, small_world, or dense"
    )


def graph_reachability(
    destinations: torch.Tensor,
    start_nodes: torch.Tensor,
    relays: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return padded exact reachable sets before and after every relay."""

    n_nodes = destinations.shape[0]
    lane_sets = [{int(node)} for node in start_nodes.tolist()]
    levels: list[list[list[int]]] = []
    for relay in range(relays + 1):
        levels.append(
            [sorted(nodes) for nodes in lane_sets]
        )
        if relay == relays:
            break
        lane_sets = [
            {
                int(destination)
                for source in nodes
                for destination in destinations[source].tolist()
            }
            for nodes in lane_sets
        ]

    nodes = torch.zeros(
        relays + 1,
        len(lane_sets),
        n_nodes,
        dtype=torch.long,
    )
    mask = torch.zeros_like(nodes, dtype=torch.bool)
    counts = torch.zeros(relays + 1, dtype=torch.long)
    for relay, lane_nodes in enumerate(levels):
        maximum = max(len(reachable) for reachable in lane_nodes)
        counts[relay] = maximum
        for lane, reachable in enumerate(lane_nodes):
            count = len(reachable)
            nodes[relay, lane, :count] = torch.tensor(reachable)
            mask[relay, lane, :count] = True
    return nodes, mask, counts


class ConvergentGator(nn.Module):
    """Carry distinct semantic lanes through one shared sparse node graph.

    A lane is a packet branch, not a cell replica. Lanes share graph
    parameters but retain separate payload and node-location state across
    relays. Co-located lanes interact through receiver-specific node memory,
    and learned terminal weights fuse full token-score vectors.
    """

    def __init__(
        self,
        *,
        n_nodes: int = 40,
        d_model: int = 8,
        degree: int = 3,
        router_mode: Literal[
            "private_full",
            "shared_low_rank",
        ] = "private_full",
        router_rank: int = 16,
        ingress_mode: Literal["fixed", "learned"] = "fixed",
        route_backend: Literal["dense", "packed"] = "dense",
        packed_memory_backend: Literal[
            "matrix",
            "convolution",
            "fft",
            "fft_opaque",
        ] = "matrix",
        packed_memory_reader_backend: Literal[
            "torch",
            "triton",
        ] = "torch",
        packed_memory_writer_backend: Literal[
            "torch",
            "triton",
        ] = "torch",
        packed_memory_feature_chunk: int = 32,
        packed_memory_compact_nodes: bool = False,
        adaptive_route_min_k: int | None = None,
        adaptive_route_mode: Literal[
            "entropy",
            "margin",
            "hybrid",
        ] = "hybrid",
        adaptive_route_entropy_mix: float = 0.5,
        compact_node_dispatch: bool = False,
        node_compute_mode: Literal[
            "full_private",
            "shared_adapter",
        ] = "full_private",
        node_adapter_rank: int = 8,
        ingress_groups: int | None = None,
        ingress_top_groups: int = 2,
        ingress_learned_groups: bool = True,
        active_node_top_k: int | None = None,
        active_node_mass_threshold: float = 0.0,
        relays: int = 4,
        lanes: int = 10,
        reachable_node_execution: bool = True,
        relay_code_span: int | None = None,
        write_warmup_relays: int | None = None,
        bootstrap_context_mode: str = "shared",
        connection_layout: str = "ring",
        connection_seed: int = 0,
        connection_rewire: float = 1.0,
        terminal_vote_mode: Literal[
            "global",
            "prompt",
            "slot",
        ] = "prompt",
        slot_isolation_slots: int | None = None,
        lane_vote_noise_std: float = 0.0,
        vocabulary_mode: Literal[
            "private_lane",
            "shared_projected",
        ] = "private_lane",
        vocab_size: int = 16,
        embedding_dim: int | None = None,
        blank_token_id: int | None = None,
        forbidden_output_token_ids: tuple[int, ...] = (),
        position_norm_length: int | None = None,
        fp8_vocab_readout: bool = False,
    ) -> None:
        super().__init__()
        if n_nodes < 2 or d_model <= 0 or relays <= 0 or lanes <= 0:
            raise ValueError("nodes, width, relays, and lanes must be positive")
        if position_norm_length is not None and position_norm_length <= 1:
            raise ValueError("position_norm_length must be greater than 1")
        if degree <= 0 or degree > n_nodes:
            raise ValueError("degree must be in [1, n_nodes]")
        if router_mode not in {"private_full", "shared_low_rank"}:
            raise ValueError(
                "router_mode must be private_full or shared_low_rank"
            )
        if router_rank <= 0:
            raise ValueError("router_rank must be positive")
        if ingress_mode not in {"fixed", "learned"}:
            raise ValueError("ingress_mode must be fixed or learned")
        if ingress_mode == "learned" and router_mode != "shared_low_rank":
            raise ValueError(
                "learned ingress requires the shared_low_rank router"
            )
        if route_backend not in {"dense", "packed"}:
            raise ValueError("route_backend must be dense or packed")
        if packed_memory_backend not in {
            "matrix",
            "convolution",
            "fft",
            "fft_opaque",
        }:
            raise ValueError(
                "packed_memory_backend must be matrix, convolution, fft, "
                "or fft_opaque"
            )
        if packed_memory_reader_backend not in {"torch", "triton"}:
            raise ValueError(
                "packed_memory_reader_backend must be torch or triton"
            )
        if (
            packed_memory_reader_backend == "triton"
            and packed_memory_backend == "matrix"
        ):
            raise ValueError(
                "the Triton packed reader requires convolution, fft, or "
                "fft_opaque receiver memory"
            )
        if packed_memory_writer_backend not in {"torch", "triton"}:
            raise ValueError(
                "packed_memory_writer_backend must be torch or triton"
            )
        if (
            packed_memory_writer_backend == "triton"
            and (
                route_backend != "packed"
                or packed_memory_backend != "fft_opaque"
            )
        ):
            raise ValueError(
                "the Triton packed writer currently requires "
                "route_backend=packed and packed_memory_backend=fft_opaque"
            )
        if (
            packed_memory_writer_backend == "triton"
            and packed_memory_feature_chunk > 128
        ):
            raise ValueError(
                "the Triton packed writer supports feature chunks up to 128"
            )
        if packed_memory_feature_chunk <= 0:
            raise ValueError("packed_memory_feature_chunk must be positive")
        if route_backend != "packed" and packed_memory_backend != "matrix":
            raise ValueError(
                "convolutional packed memory requires route_backend=packed"
            )
        if adaptive_route_mode not in {"entropy", "margin", "hybrid"}:
            raise ValueError(
                "adaptive_route_mode must be entropy, margin, or hybrid"
            )
        if not 0.0 <= adaptive_route_entropy_mix <= 1.0:
            raise ValueError("adaptive_route_entropy_mix must be in [0, 1]")
        if adaptive_route_min_k is not None:
            if route_backend != "packed" or active_node_top_k is None:
                raise ValueError(
                    "adaptive_route_min_k requires a fixed-cap packed route"
                )
            if not 1 <= adaptive_route_min_k <= active_node_top_k:
                raise ValueError(
                    "adaptive_route_min_k must be in [1, active_node_top_k]"
                )
        if node_compute_mode not in {"full_private", "shared_adapter"}:
            raise ValueError(
                "node_compute_mode must be full_private or shared_adapter"
            )
        if node_adapter_rank <= 0:
            raise ValueError("node_adapter_rank must be positive")
        if node_compute_mode == "shared_adapter" and route_backend != "packed":
            raise ValueError(
                "shared_adapter node compute currently requires packed routes"
            )
        if ingress_top_groups <= 0:
            raise ValueError("ingress_top_groups must be positive")
        if ingress_groups is not None:
            if not 1 <= ingress_groups <= n_nodes:
                raise ValueError("ingress_groups must be in [1, n_nodes]")
            if route_backend != "packed" or ingress_mode != "learned":
                raise ValueError(
                    "hierarchical ingress requires packed learned ingress"
                )
            if ingress_top_groups > ingress_groups:
                raise ValueError(
                    "ingress_top_groups cannot exceed ingress_groups"
                )
            group_capacity = (n_nodes + ingress_groups - 1) // ingress_groups
            if active_node_top_k is None or active_node_top_k > min(
                n_nodes,
                ingress_top_groups * group_capacity,
            ):
                raise ValueError(
                    "active_node_top_k exceeds hierarchical ingress candidates"
                )
        if route_backend == "packed" and active_node_top_k is None:
            raise ValueError(
                "packed route_backend requires active_node_top_k"
            )
        if route_backend == "packed" and (
            router_mode != "shared_low_rank" or ingress_mode != "learned"
        ):
            raise ValueError(
                "packed route_backend currently requires shared_low_rank "
                "routing with learned ingress"
            )
        if active_node_top_k is not None and not (
            1 <= active_node_top_k <= n_nodes
        ):
            raise ValueError("active_node_top_k must be in [1, n_nodes]")
        if not 0.0 <= active_node_mass_threshold < 1.0:
            raise ValueError(
                "active_node_mass_threshold must be in [0, 1)"
            )
        if lanes > n_nodes:
            raise ValueError("lanes cannot exceed graph nodes")
        relay_code_span = (
            relays if relay_code_span is None else relay_code_span
        )
        write_warmup_relays = (
            relays
            if write_warmup_relays is None
            else write_warmup_relays
        )
        if relay_code_span < relays:
            raise ValueError("relay_code_span cannot be shorter than relays")
        if write_warmup_relays <= 0:
            raise ValueError("write_warmup_relays must be positive")
        if bootstrap_context_mode not in {"shared", "self"}:
            raise ValueError(
                "bootstrap_context_mode must be shared or self"
            )
        if terminal_vote_mode not in {"global", "prompt", "slot"}:
            raise ValueError(
                "terminal_vote_mode must be global, prompt, or slot"
            )
        if slot_isolation_slots is not None:
            if not 1 <= slot_isolation_slots <= n_nodes:
                raise ValueError(
                    "slot_isolation_slots must be in [1, n_nodes]"
                )
            if terminal_vote_mode != "slot":
                raise ValueError(
                    "slot-isolated computation requires slot terminal voting"
                )
            if route_backend != "dense" or active_node_top_k is None:
                raise ValueError(
                    "slot-isolated computation requires dense routing with "
                    "active_node_top_k"
                )
            if 2 * slot_isolation_slots > d_model:
                raise ValueError(
                    "slot-isolated juries require d_model >= 2 * slots"
                )
        if lane_vote_noise_std < 0:
            raise ValueError("lane_vote_noise_std must be non-negative")
        if vocabulary_mode not in {
            "private_lane",
            "shared_projected",
        }:
            raise ValueError(
                "vocabulary_mode must be private_lane or shared_projected"
            )
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        embedding_dim = (
            d_model if embedding_dim is None else embedding_dim
        )
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if vocabulary_mode == "private_lane" and (
            vocab_size != 16 or embedding_dim != d_model
        ):
            raise ValueError(
                "private_lane mode preserves the validated 16-token, "
                "embedding_dim=d_model contract"
            )
        if blank_token_id is not None and not (
            0 <= blank_token_id < vocab_size
        ):
            raise ValueError("blank_token_id must be inside the vocabulary")
        if len(set(forbidden_output_token_ids)) != len(
            forbidden_output_token_ids
        ):
            raise ValueError("forbidden output token IDs must be unique")
        if any(
            token_id < 0 or token_id >= vocab_size
            for token_id in forbidden_output_token_ids
        ):
            raise ValueError(
                "forbidden output token IDs must be inside the vocabulary"
            )
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.degree = degree
        self.router_mode = router_mode
        self.router_rank = router_rank
        self.ingress_mode = ingress_mode
        self.route_backend = route_backend
        self.packed_memory_backend = packed_memory_backend
        self.packed_memory_reader_backend = packed_memory_reader_backend
        self.packed_memory_writer_backend = packed_memory_writer_backend
        self.packed_memory_feature_chunk = packed_memory_feature_chunk
        self.packed_memory_compact_nodes = packed_memory_compact_nodes
        self.adaptive_route_min_k = adaptive_route_min_k
        self.adaptive_route_mode = adaptive_route_mode
        self.adaptive_route_entropy_mix = adaptive_route_entropy_mix
        self.compact_node_dispatch = compact_node_dispatch
        self.node_compute_mode = node_compute_mode
        self.node_adapter_rank = node_adapter_rank
        self.ingress_groups = ingress_groups
        self.ingress_top_groups = ingress_top_groups
        self.ingress_learned_groups = ingress_learned_groups
        self.active_node_top_k = active_node_top_k
        self.active_node_mass_threshold = active_node_mass_threshold
        self.relays = relays
        self.lanes = lanes
        self.reachable_node_execution = reachable_node_execution
        self.relay_code_span = relay_code_span
        self.write_warmup_relays = write_warmup_relays
        self.bootstrap_context_mode = bootstrap_context_mode
        self.connection_layout = connection_layout
        self.connection_seed = connection_seed
        self.connection_rewire = connection_rewire
        self.terminal_vote_mode = terminal_vote_mode
        self.slot_isolation_slots = slot_isolation_slots
        self.lane_vote_noise_std = lane_vote_noise_std
        self.vocabulary_mode = vocabulary_mode
        self.position_norm_length = position_norm_length
        self.fp8_vocab_readout = fp8_vocab_readout
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.blank_token_id = blank_token_id

        if slot_isolation_slots is None:
            self.register_buffer(
                "slot_compute_nodes",
                None,
                persistent=False,
            )
            self.register_buffer(
                "slot_compute_sizes",
                None,
                persistent=False,
            )
        else:
            base_size, remainder = divmod(n_nodes, slot_isolation_slots)
            sizes = torch.full(
                (slot_isolation_slots,),
                base_size,
                dtype=torch.long,
            )
            sizes[:remainder] += 1
            pools = torch.zeros(
                slot_isolation_slots,
                int(sizes.max()),
                dtype=torch.long,
            )
            offset = 0
            for slot, size in enumerate(sizes.tolist()):
                pools[slot, :size] = torch.arange(offset, offset + size)
                offset += size
            self.register_buffer(
                "slot_compute_nodes",
                pools,
                persistent=False,
            )
            self.register_buffer(
                "slot_compute_sizes",
                sizes,
                persistent=False,
            )

        # Lanes are persistent complementary feature views, not permanent
        # graph sites.  Their private lexical/address transforms preserve the
        # functional width supplied accidentally by OG Arrow's private cells.
        if vocabulary_mode == "private_lane":
            self.token_embedding = nn.Parameter(
                torch.empty(lanes, vocab_size, d_model)
            )
            self.register_parameter("input_projection", None)
            self.register_parameter("readout_projection", None)
            self.register_parameter("lm_bias", None)
        else:
            # Language keeps one tied lexical table. Each persistent lane
            # learns its own input and output view, preserving complementary
            # traces without multiplying the 2,048-row vocabulary by K.
            self.token_embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
            )
            self.input_projection = nn.Parameter(
                torch.empty(lanes, d_model, embedding_dim)
            )
            self.readout_projection = nn.Parameter(
                torch.empty(lanes, embedding_dim, d_model)
            )
            self.lm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.register_buffer(
            "_forbidden_output_token_ids",
            torch.tensor(
                forbidden_output_token_ids,
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.address_to_state = nn.Parameter(
            torch.empty(lanes, d_model, 4)
        )
        self.address_to_edges = nn.Linear(
            4,
            n_nodes * degree,
            bias=False,
        )
        self.address_to_readout = nn.Parameter(
            torch.empty(lanes, d_model, 4)
        )
        if node_compute_mode == "full_private":
            self.node_embedding = nn.Embedding(n_nodes, d_model)
            self.node_weight = nn.Parameter(
                torch.empty(n_nodes, d_model, d_model)
            )
            self.node_bias = nn.Parameter(torch.empty(n_nodes, d_model))
            self.register_module("shared_node_adapters", None)
        else:
            self.register_module("node_embedding", None)
            self.register_parameter("node_weight", None)
            self.register_parameter("node_bias", None)
            self.shared_node_adapters = SharedTrunkNodeAdapters(
                n_nodes=n_nodes,
                width=d_model,
                adapter_rank=node_adapter_rank,
            )
        self.step_projection = nn.Parameter(
            torch.empty(lanes, d_model, 4)
        )
        self.lane_adapter = nn.Parameter(
            torch.empty(lanes, d_model, d_model)
        )
        self.lane_adapter_out = nn.Parameter(
            torch.empty(lanes, d_model, d_model)
        )
        self.lane_adapter_bias = nn.Parameter(
            torch.empty(lanes, d_model)
        )
        self.lane_norm_weight = nn.Parameter(
            torch.ones(lanes, d_model)
        )
        self.lane_norm_bias = nn.Parameter(
            torch.zeros(lanes, d_model)
        )
        if router_mode == "private_full":
            self.router_weight = nn.Parameter(
                torch.empty(
                    n_nodes,
                    degree,
                    2 * d_model + degree,
                )
            )
            self.register_parameter("router_trunk_weight", None)
            self.register_parameter("router_trunk_bias", None)
            self.register_parameter("router_edge_key", None)
            self.register_parameter("router_occupancy_weight", None)
            self.register_parameter("ingress_node_key", None)
            self.register_parameter("ingress_node_bias", None)
        else:
            # One packet feature extraction is shared by every source node.
            # Private node-edge keys retain the local "where do I send this
            # shape?" policy without repeating a full 2D-wide router at every
            # graph site.
            self.register_parameter("router_weight", None)
            self.router_trunk_weight = nn.Parameter(
                torch.empty(router_rank, 2 * d_model)
            )
            self.router_trunk_bias = nn.Parameter(
                torch.zeros(router_rank)
            )
            self.router_edge_key = nn.Parameter(
                torch.empty(n_nodes, degree, router_rank)
            )
            self.router_occupancy_weight = nn.Parameter(
                torch.empty(degree, degree)
            )
            if ingress_mode == "learned":
                # The shared trunk's hidden-state half produces an ingress
                # query. Every node owns a private key and can therefore win
                # some packet types without dense node-to-node connectivity.
                self.ingress_node_key = nn.Parameter(
                    torch.empty(n_nodes, router_rank)
                )
                self.ingress_node_bias = nn.Parameter(
                    torch.zeros(n_nodes)
                )
            else:
                self.register_parameter("ingress_node_key", None)
                self.register_parameter("ingress_node_bias", None)
        if ingress_groups is None:
            self.register_module("hierarchical_ingress", None)
        else:
            if active_node_top_k is None:
                raise RuntimeError("hierarchical ingress lost its route capacity")
            self.hierarchical_ingress = HierarchicalPackedIngress(
                n_nodes=n_nodes,
                rank=router_rank,
                n_groups=ingress_groups,
                top_groups=ingress_top_groups,
                top_k=active_node_top_k,
                learned_groups=ingress_learned_groups,
            )
        self.router_bias = nn.Parameter(
            torch.empty(n_nodes, degree)
        )
        self.relative_node_bias = nn.Parameter(torch.zeros(n_nodes, 4))
        self.terminal_votes = nn.Parameter(torch.zeros(lanes))
        if terminal_vote_mode != "global":
            # Prompt voting starts as the exact global-vote model. Slot voting
            # also receives the separately initialized immutable-address path
            # below so its juries are distinct from the first update.
            self.terminal_vote_query = nn.Parameter(
                torch.zeros(lanes, d_model)
            )
        else:
            self.register_parameter("terminal_vote_query", None)
        if terminal_vote_mode == "slot":
            # Give immutable slot identity a short path to the verdict.  The
            # previous slot controller could only recover the address through
            # the repeatedly mixed final lane state.  A small direct key lets
            # each fixed four-number address immediately choose a different
            # lane coalition while remaining length-general.
            self.terminal_address_key = nn.Parameter(
                torch.empty(lanes, 4)
            )
        else:
            self.register_parameter("terminal_address_key", None)

        if vocabulary_mode == "private_lane":
            nn.init.normal_(self.token_embedding)
        else:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
            nn.init.kaiming_uniform_(self.input_projection, a=5**0.5)
            nn.init.kaiming_uniform_(
                self.readout_projection,
                a=5**0.5,
            )
        for parameter in (
            self.address_to_state,
            self.address_to_readout,
            self.step_projection,
            self.lane_adapter,
            self.lane_adapter_out,
        ):
            nn.init.kaiming_uniform_(parameter, a=5**0.5)
        bound = 1 / d_model**0.5
        nn.init.uniform_(
            self.lane_adapter_bias,
            -bound,
            bound,
        )
        if self.node_compute_mode == "full_private":
            nn.init.kaiming_uniform_(self.node_weight, a=5**0.5)
            nn.init.uniform_(self.node_bias, -bound, bound)
        if router_mode == "private_full":
            nn.init.kaiming_uniform_(self.router_weight, a=5**0.5)
            router_bound = 1 / (2 * d_model + degree) ** 0.5
        else:
            nn.init.kaiming_uniform_(
                self.router_trunk_weight,
                a=5**0.5,
            )
            nn.init.kaiming_uniform_(
                self.router_edge_key,
                a=5**0.5,
            )
            nn.init.kaiming_uniform_(
                self.router_occupancy_weight,
                a=5**0.5,
            )
            if self.ingress_node_key is not None:
                nn.init.kaiming_uniform_(
                    self.ingress_node_key,
                    a=5**0.5,
                )
            router_bound = 1 / router_rank**0.5
        nn.init.uniform_(
            self.router_bias,
            -router_bound,
            router_bound,
        )
        # Initialize new optional parameters only after every historical
        # shared tensor so a fixed seed remains an exact architectural A/B.
        if self.terminal_address_key is not None:
            nn.init.kaiming_uniform_(
                self.terminal_address_key,
                a=5**0.5,
            )
        if (
            self.hierarchical_ingress is not None
            and self.hierarchical_ingress.group_keys is not None
        ):
            nn.init.kaiming_uniform_(
                self.hierarchical_ingress.group_keys,
                a=5**0.5,
            )
        self.register_buffer(
            "destinations",
            connection_destinations(
                n_nodes=n_nodes,
                degree=degree,
                layout=connection_layout,
                seed=connection_seed,
                rewire_probability=connection_rewire,
            ),
        )
        start_nodes = torch.div(
            torch.arange(lanes) * n_nodes,
            lanes,
            rounding_mode="floor",
        )
        self.register_buffer("start_nodes", start_nodes)
        reachable_nodes, reachable_mask, reachable_counts = graph_reachability(
            self.destinations,
            start_nodes,
            relays,
        )
        self.register_buffer(
            "reachable_nodes",
            reachable_nodes,
            persistent=False,
        )
        self.register_buffer(
            "reachable_mask",
            reachable_mask,
            persistent=False,
        )
        self.register_buffer(
            "reachable_counts",
            reachable_counts,
            persistent=False,
        )
        self._reachable_count_values = tuple(
            int(value)
            for value in reachable_counts.tolist()
        )
        self.register_load_state_dict_pre_hook(
            self._upgrade_legacy_slot_voter_state
        )
        self.register_load_state_dict_post_hook(
            self._refresh_reachability_after_load
        )
        if vocabulary_mode == "private_lane":
            assert self.parameter_count == self.parameter_count_for(
                n_nodes=n_nodes,
                d_model=d_model,
                lanes=lanes,
                degree=degree,
                router_mode=router_mode,
                router_rank=router_rank,
                ingress_mode=ingress_mode,
                node_compute_mode=node_compute_mode,
                node_adapter_rank=node_adapter_rank,
                ingress_groups=ingress_groups,
                ingress_learned_groups=ingress_learned_groups,
                terminal_vote_mode=terminal_vote_mode,
            )
            if (
                n_nodes,
                d_model,
                degree,
                relays,
                lanes,
            ) == (40, 8, 3, 4, 10):
                assert self.parameter_count == 10_010
        else:
            assert (
                self.parameter_count
                == self.shared_language_parameter_count_for(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    n_nodes=n_nodes,
                    d_model=d_model,
                    lanes=lanes,
                    degree=degree,
                    router_mode=router_mode,
                    router_rank=router_rank,
                    ingress_mode=ingress_mode,
                    node_compute_mode=node_compute_mode,
                    node_adapter_rank=node_adapter_rank,
                    ingress_groups=ingress_groups,
                    ingress_learned_groups=ingress_learned_groups,
                    terminal_vote_mode=terminal_vote_mode,
                )
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _refresh_reachability_after_load(
        self,
        _module: nn.Module,
        _incompatible_keys,
    ) -> None:
        nodes, mask, counts = graph_reachability(
            self.destinations.detach().cpu(),
            self.start_nodes.detach().cpu(),
            self.relays,
        )
        self.reachable_nodes = nodes.to(self.destinations.device)
        self.reachable_mask = mask.to(self.destinations.device)
        self.reachable_counts = counts.to(self.destinations.device)
        self._reachable_count_values = tuple(
            int(value)
            for value in counts.tolist()
        )

    def _upgrade_legacy_slot_voter_state(
        self,
        _module: nn.Module,
        state_dict: dict,
        prefix: str,
        _local_metadata: dict,
        _strict: bool,
        _missing_keys: list[str],
        _unexpected_keys: list[str],
        _error_messages: list[str],
    ) -> None:
        """Load pre-address slot checkpoints without changing their logits."""

        if self.terminal_address_key is None:
            return
        key = prefix + "terminal_address_key"
        if key not in state_dict:
            state_dict[key] = torch.zeros_like(self.terminal_address_key)

    @property
    def architecture_config(
        self,
    ) -> dict[str, int | float | str | bool | None]:
        return {
            "n_nodes": self.n_nodes,
            "d_model": self.d_model,
            "degree": self.degree,
            "router_mode": self.router_mode,
            "router_rank": self.router_rank,
            "ingress_mode": self.ingress_mode,
            "route_backend": self.route_backend,
            "packed_memory_backend": self.packed_memory_backend,
            "packed_memory_reader_backend": (
                self.packed_memory_reader_backend
            ),
            "packed_memory_writer_backend": (
                self.packed_memory_writer_backend
            ),
            "packed_memory_feature_chunk": self.packed_memory_feature_chunk,
            "packed_memory_compact_nodes": self.packed_memory_compact_nodes,
            "adaptive_route_min_k": self.adaptive_route_min_k,
            "adaptive_route_mode": self.adaptive_route_mode,
            "adaptive_route_entropy_mix": self.adaptive_route_entropy_mix,
            "compact_node_dispatch": self.compact_node_dispatch,
            "node_compute_mode": self.node_compute_mode,
            "node_adapter_rank": self.node_adapter_rank,
            "ingress_groups": self.ingress_groups,
            "ingress_top_groups": self.ingress_top_groups,
            "ingress_learned_groups": self.ingress_learned_groups,
            "active_node_top_k": self.active_node_top_k,
            "active_node_mass_threshold": (
                self.active_node_mass_threshold
            ),
            "relays": self.relays,
            "lanes": self.lanes,
            "relay_code_span": self.relay_code_span,
            "write_warmup_relays": self.write_warmup_relays,
            "bootstrap_context_mode": self.bootstrap_context_mode,
            "connection_layout": self.connection_layout,
            "connection_seed": self.connection_seed,
            "connection_rewire": self.connection_rewire,
            "terminal_vote_mode": self.terminal_vote_mode,
            "slot_isolation_slots": self.slot_isolation_slots,
            "lane_vote_noise_std": self.lane_vote_noise_std,
            "vocabulary_mode": self.vocabulary_mode,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "position_norm_length": self.position_norm_length,
            "fp8_vocab_readout": self.fp8_vocab_readout,
        }

    @staticmethod
    def parameter_count_for(
        *,
        n_nodes: int,
        d_model: int,
        lanes: int,
        degree: int = 3,
        router_mode: Literal[
            "private_full",
            "shared_low_rank",
        ] = "private_full",
        router_rank: int = 16,
        ingress_mode: Literal["fixed", "learned"] = "fixed",
        node_compute_mode: Literal[
            "full_private",
            "shared_adapter",
        ] = "full_private",
        node_adapter_rank: int = 8,
        ingress_groups: int | None = None,
        ingress_learned_groups: bool = True,
        terminal_vote_mode: Literal[
            "global",
            "prompt",
            "slot",
        ] = "prompt",
    ) -> int:
        """Return the trainable count for the selected graph degree."""

        lane_parameters = lanes * (
            2 * d_model**2
            + 31 * d_model
            + 1
            + (d_model if terminal_vote_mode != "global" else 0)
            + (4 if terminal_vote_mode == "slot" else 0)
        )
        graph_parameters = ConvergentGator._graph_parameter_count(
            n_nodes=n_nodes,
            d_model=d_model,
            degree=degree,
            router_mode=router_mode,
            router_rank=router_rank,
            ingress_mode=ingress_mode,
            node_compute_mode=node_compute_mode,
            node_adapter_rank=node_adapter_rank,
            ingress_groups=ingress_groups,
            ingress_learned_groups=ingress_learned_groups,
        )
        return lane_parameters + graph_parameters

    @staticmethod
    def shared_language_parameter_count_for(
        *,
        vocab_size: int,
        embedding_dim: int,
        n_nodes: int,
        d_model: int,
        lanes: int,
        degree: int = 3,
        router_mode: Literal[
            "private_full",
            "shared_low_rank",
        ] = "private_full",
        router_rank: int = 16,
        ingress_mode: Literal["fixed", "learned"] = "fixed",
        node_compute_mode: Literal[
            "full_private",
            "shared_adapter",
        ] = "full_private",
        node_adapter_rank: int = 8,
        ingress_groups: int | None = None,
        ingress_learned_groups: bool = True,
        terminal_vote_mode: Literal[
            "global",
            "prompt",
            "slot",
        ] = "prompt",
    ) -> int:
        """Count tied-vocabulary language parameters exactly."""

        # Remove each lane's private 16*D table from the validated cell,
        # then add lane-specific E->D and D->E lexical views.
        lane_parameters = lanes * (
            2 * d_model**2
            + 15 * d_model
            + 1
            + 2 * d_model * embedding_dim
            + (d_model if terminal_vote_mode != "global" else 0)
            + (4 if terminal_vote_mode == "slot" else 0)
        )
        graph_parameters = ConvergentGator._graph_parameter_count(
            n_nodes=n_nodes,
            d_model=d_model,
            degree=degree,
            router_mode=router_mode,
            router_rank=router_rank,
            ingress_mode=ingress_mode,
            node_compute_mode=node_compute_mode,
            node_adapter_rank=node_adapter_rank,
            ingress_groups=ingress_groups,
            ingress_learned_groups=ingress_learned_groups,
        )
        vocabulary_parameters = (
            vocab_size * embedding_dim + vocab_size
        )
        return (
            lane_parameters
            + graph_parameters
            + vocabulary_parameters
        )

    @staticmethod
    def _graph_parameter_count(
        *,
        n_nodes: int,
        d_model: int,
        degree: int,
        router_mode: str,
        router_rank: int,
        ingress_mode: str,
        node_compute_mode: str,
        node_adapter_rank: int,
        ingress_groups: int | None,
        ingress_learned_groups: bool,
    ) -> int:
        """Count private node compute and the selected routing machinery."""

        if ingress_mode not in {"fixed", "learned"}:
            raise ValueError("ingress_mode must be fixed or learned")
        if ingress_mode == "learned" and router_mode != "shared_low_rank":
            raise ValueError(
                "learned ingress requires the shared_low_rank router"
            )
        if node_compute_mode == "full_private":
            node_parameters = n_nodes * (
                d_model**2 + 2 * d_model + 4
            )
        elif node_compute_mode == "shared_adapter":
            if node_adapter_rank <= 0:
                raise ValueError("node_adapter_rank must be positive")
            node_parameters = (
                d_model**2
                + d_model
                + n_nodes
                * (
                    2 * d_model
                    + 2 * d_model * node_adapter_rank
                    + 4
                )
            )
        else:
            raise ValueError(
                "node_compute_mode must be full_private or shared_adapter"
            )
        if ingress_groups is not None:
            if not 1 <= ingress_groups <= n_nodes:
                raise ValueError("ingress_groups must be in [1, n_nodes]")
            if ingress_learned_groups:
                node_parameters += ingress_groups * (router_rank + 1)
        address_parameters = 4 * n_nodes * degree
        if router_mode == "private_full":
            router_parameters = n_nodes * degree * (
                2 * d_model + degree + 1
            )
        elif router_mode == "shared_low_rank":
            router_parameters = (
                router_rank * (2 * d_model)
                + router_rank
                + n_nodes * degree * router_rank
                + degree**2
                + n_nodes * degree
            )
            if ingress_mode == "learned":
                router_parameters += n_nodes * router_rank + n_nodes
        else:
            raise ValueError(
                "router_mode must be private_full or shared_low_rank"
            )
        return node_parameters + address_parameters + router_parameters

    def _lane_norm(self, value: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(value, (self.d_model,))
        return (
            normalized * self.lane_norm_weight[None, None]
            + self.lane_norm_bias[None, None]
        )

    def _receiver_kernel(
        self,
        length: int,
        device: torch.device,
        *,
        causal: bool = False,
        norm_length: int | None = None,
    ) -> torch.Tensor:
        relation_bias = torch.einsum(
            "ijr,nr->ijn",
            relative_code(length, device, norm_length=norm_length).to(
                dtype=self.relative_node_bias.dtype
            ),
            self.relative_node_bias,
        )
        # Use the same per-node constant as the convolutional form.  The
        # constant cancels exactly in numerator / denominator, while sharing
        # the stabilization contract prevents AdamW from amplifying tiny
        # backend-only gradient differences.
        relation_bias = relation_bias - relation_bias.amax(
            dim=(0, 1),
            keepdim=True,
        ).detach()
        kernel = relation_bias.exp()
        if causal:
            # Axis 0 is the receiver and axis 1 is the writer. A receiver may
            # read itself and earlier positions, never a future writer.
            kernel = kernel * torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=device,
            ).tril().unsqueeze(-1)
        return kernel

    @staticmethod
    def _factored_router_logits(
        semantic_input: torch.Tensor,
        local_occupancy: torch.Tensor,
        router_weight: torch.Tensor,
        router_bias: torch.Tensor,
        address_edges: torch.Tensor,
    ) -> torch.Tensor:
        semantic_width = semantic_input.size(-1)
        batch, tokens, lanes, _ = semantic_input.shape
        sources, degree = router_weight.shape[1:3]
        semantic_weight = router_weight[
            ..., :semantic_width
        ].reshape(
            lanes,
            sources * degree,
            semantic_width,
        )
        semantic_logits = cell_linear(
            semantic_input.permute(0, 2, 1, 3),
            semantic_weight,
        ).reshape(
            batch,
            lanes,
            tokens,
            sources,
            degree,
        ).permute(0, 2, 1, 3, 4)
        if local_occupancy.ndim == 4:
            occupancy_logits = torch.einsum(
                "bksj,ksdj->bksd",
                local_occupancy,
                router_weight[..., semantic_width:],
            )[:, None]
        elif local_occupancy.ndim == 5:
            occupancy_logits = torch.einsum(
                "btksj,ksdj->btksd",
                local_occupancy,
                router_weight[..., semantic_width:],
            )
        else:
            raise ValueError(
                "local occupancy must have [B,K,S,D] or [B,T,K,S,D] shape"
            )
        return (
            semantic_logits
            + occupancy_logits
            + router_bias[None, None]
            + address_edges[None]
        )

    def _shared_router_packet_features(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Project packet state without materializing ``cat(hidden, context)``."""

        if (
            self.router_trunk_weight is None
            or self.router_trunk_bias is None
        ):
            raise RuntimeError("shared low-rank router is not configured")
        hidden_weight = self.router_trunk_weight[:, : self.d_model]
        context_weight = self.router_trunk_weight[:, self.d_model :]
        projected = F.linear(
            hidden,
            hidden_weight,
            self.router_trunk_bias,
        )
        projected = projected + F.linear(
            context,
            context_weight,
        )
        return F.gelu(projected)

    def _shared_low_rank_router_logits(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        local_occupancy: torch.Tensor,
        source_nodes: torch.Tensor,
        address_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Score private node edges from one shared packet representation."""

        if (
            self.router_trunk_weight is None
            or self.router_trunk_bias is None
            or self.router_edge_key is None
            or self.router_occupancy_weight is None
        ):
            raise RuntimeError("shared low-rank router is not configured")
        packet_features = self._shared_router_packet_features(
            hidden,
            context,
        )
        edge_keys = self.router_edge_key[source_nodes]
        semantic_logits = torch.einsum(
            "btkh,ksdh->btksd",
            packet_features,
            edge_keys,
        )
        if local_occupancy.ndim == 4:
            occupancy_logits = torch.einsum(
                "bksj,dj->bksd",
                local_occupancy,
                self.router_occupancy_weight,
            )[:, None]
        elif local_occupancy.ndim == 5:
            occupancy_logits = torch.einsum(
                "btksj,dj->btksd",
                local_occupancy,
                self.router_occupancy_weight,
            )
        else:
            raise ValueError(
                "local occupancy must have [B,K,S,D] or [B,T,K,S,D] shape"
            )
        return (
            semantic_logits
            + occupancy_logits
            + self.router_bias[source_nodes][None, None]
            + address_edges[None]
        )

    def _shared_low_rank_dynamic_router_logits(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        local_occupancy: torch.Tensor,
        source_nodes: torch.Tensor,
        address_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Score only the packet-specific source nodes carrying route mass."""

        if (
            self.router_trunk_weight is None
            or self.router_trunk_bias is None
            or self.router_edge_key is None
            or self.router_occupancy_weight is None
        ):
            raise RuntimeError("shared low-rank router is not configured")
        packet_features = self._shared_router_packet_features(
            hidden,
            context,
        )
        all_source_logits = self._shared_router_all_source_logits(
            packet_features,
        )
        semantic_logits = all_source_logits.gather(
            3,
            source_nodes.unsqueeze(-1).expand(
                -1,
                -1,
                -1,
                -1,
                self.degree,
            ),
        )
        occupancy_logits = torch.einsum(
            "btkmj,dj->btkmd",
            local_occupancy,
            self.router_occupancy_weight,
        )
        return (
            semantic_logits
            + occupancy_logits
            + self.router_bias[source_nodes]
            + address_edges
        )

    def _shared_router_all_source_logits(
        self,
        packet_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score every source edge before gathering packet-active nodes.

        Keeping router rank inside one GEMM avoids materializing indexed edge
        keys with ``[B,T,L,K,degree,rank]`` shape. The retained score tensor is
        only ``[B,T,L,N,degree]``.
        """

        if self.router_edge_key is None:
            raise RuntimeError("shared low-rank router is not configured")
        flat_edge_keys = self.router_edge_key.reshape(
            self.n_nodes * self.degree,
            self.router_rank,
        )
        flat_scores = F.linear(packet_features, flat_edge_keys)
        return flat_scores.view(
            *packet_features.shape[:-1],
            self.n_nodes,
            self.degree,
        )

    @staticmethod
    def _sparsify_locations(
        locations: torch.Tensor,
        *,
        top_k: int | None,
        mass_threshold: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Keep a fixed-cap set of meaningful nodes per packet.

        Returning a fixed-width top-k tensor keeps ``torch.compile`` shapes
        stable. Thresholded entries carry zero mass; the strongest node is
        always retained so every packet remains normalized.
        """

        n_nodes = locations.shape[-1]
        if top_k is None and mass_threshold <= 0.0:
            return locations, None, None
        if top_k is None:
            # A normalized distribution can contain at most floor(1/tau)
            # entries with mass >= tau. The +1 handles boundary rounding
            # conservatively while remaining shape-stable.
            top_k = min(
                n_nodes,
                int(1.0 / mass_threshold) + 1,
            )
        retained_mass, retained_nodes = locations.topk(
            min(top_k, n_nodes),
            dim=-1,
        )
        if mass_threshold > 0.0:
            retained = retained_mass.ge(mass_threshold)
            strongest = retained_mass.argmax(
                dim=-1,
                keepdim=True,
            )
            retained = retained.scatter(-1, strongest, True)
            retained_mass = retained_mass * retained.to(
                retained_mass.dtype
            )
        retained_mass = retained_mass / retained_mass.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-9)
        sparse = torch.zeros_like(locations).scatter(
            -1,
            retained_nodes,
            retained_mass,
        )
        return sparse, retained_nodes, retained_mass

    def _learned_ingress(
        self,
        hidden: torch.Tensor,
        *,
        temperature: float | torch.Tensor,
        top_k: int | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Choose packet-conditioned entry nodes with global graph coverage."""

        if (
            self.router_trunk_weight is None
            or self.router_trunk_bias is None
            or self.ingress_node_key is None
            or self.ingress_node_bias is None
        ):
            raise RuntimeError("learned ingress is not configured")
        ingress_features = F.gelu(
            F.linear(
                hidden,
                self.router_trunk_weight[:, : self.d_model],
                self.router_trunk_bias,
            )
        )
        ingress_logits = (
            torch.einsum(
                "btkh,nh->btkn",
                ingress_features,
                self.ingress_node_key,
            )
            + self.ingress_node_bias[None, None, None]
        )
        locations = F.softmax(ingress_logits / temperature, dim=-1)
        return self._sparsify_locations(
            locations,
            top_k=top_k,
            mass_threshold=self.active_node_mass_threshold,
        )

    def _learned_ingress_frontier(
        self,
        hidden: torch.Tensor,
        *,
        temperature: float | torch.Tensor,
    ) -> PackedNodeFrontier:
        """Choose a globally reachable ingress without a dense passport.

        Global node logits are still required so every node remains
        reachable. Only the configured top-k logits become persistent route
        state. With no threshold, softmax over those retained logits is
        exactly the dense-top-k result after its required renormalization.
        """

        if (
            self.router_trunk_weight is None
            or self.router_trunk_bias is None
            or self.ingress_node_key is None
            or self.ingress_node_bias is None
            or self.active_node_top_k is None
        ):
            raise RuntimeError("packed learned ingress is not configured")
        ingress_features = F.gelu(
            F.linear(
                hidden,
                self.router_trunk_weight[:, : self.d_model],
                self.router_trunk_bias,
            )
        )
        if self.hierarchical_ingress is not None:
            return self.hierarchical_ingress(
                ingress_features,
                self.ingress_node_key,
                self.ingress_node_bias,
                temperature=temperature,
                mass_threshold=self.active_node_mass_threshold,
            )
        scaled_logits = (
            torch.einsum(
                "btkh,nh->btkn",
                ingress_features,
                self.ingress_node_key,
            )
            + self.ingress_node_bias[None, None, None]
        ) / temperature
        retained_logits, retained_nodes = scaled_logits.topk(
            self.active_node_top_k,
            dim=-1,
        )
        if self.active_node_mass_threshold > 0.0:
            normalizer = torch.logsumexp(
                scaled_logits,
                dim=-1,
                keepdim=True,
            )
            retained_mass = (retained_logits - normalizer).exp()
            retained = retained_mass.ge(
                self.active_node_mass_threshold
            )
            retained[..., 0] = True
            retained_mass = retained_mass * retained.to(
                retained_mass.dtype
            )
        else:
            retained = torch.ones_like(
                retained_nodes,
                dtype=torch.bool,
            )
            retained_mass = F.softmax(retained_logits, dim=-1)
        retained_mass = retained_mass / retained_mass.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-9)
        return PackedNodeFrontier(
            nodes=torch.where(
                retained,
                retained_nodes,
                torch.zeros_like(retained_nodes),
            ),
            mass=retained_mass,
            valid=retained,
            n_nodes=self.n_nodes,
        )

    def _slot_owned_compute_nodes(
        self,
        active_nodes: torch.Tensor,
        *,
        observed_length: int,
        output_slots: int,
    ) -> torch.Tensor:
        """Map output routes onto disjoint slot-owned compute parameters.

        Routing locations remain unchanged and continue to drive shared node
        memory, so causal packets can still observe one another. Only the
        private transform selected for each output packet is remapped. This
        separates communication from trainable compute ownership without
        changing active route mass or parameter count.
        """

        if self.slot_isolation_slots is None:
            return active_nodes
        if active_nodes.ndim != 4:
            raise ValueError(
                "slot-owned compute expects [batch,tokens,lanes,routes] nodes"
            )
        if active_nodes.shape[1] != observed_length + output_slots:
            raise ValueError("active-node token count does not match formation")
        if output_slots > self.slot_isolation_slots:
            raise ValueError(
                "output slots exceed configured slot-isolation capacity"
            )
        if self.slot_compute_nodes is None or self.slot_compute_sizes is None:
            raise RuntimeError("slot-owned compute buffers are missing")

        output_nodes = active_nodes[:, observed_length:]
        batch, _, lanes, routes = output_nodes.shape
        lane_index = torch.arange(
            lanes,
            device=output_nodes.device,
            dtype=output_nodes.dtype,
        )[None, None, :, None]
        route_index = torch.arange(
            routes,
            device=output_nodes.device,
            dtype=output_nodes.dtype,
        )[None, None, None, :]
        sizes = self.slot_compute_sizes[:output_slots][None, :, None, None]
        selector = torch.remainder(
            output_nodes + 17 * lane_index + 31 * route_index,
            sizes,
        )
        pools = self.slot_compute_nodes[:output_slots][None, :, None].expand(
            batch,
            -1,
            lanes,
            -1,
        )
        mapped_output_nodes = pools.gather(-1, selector)
        return torch.cat(
            (active_nodes[:, :observed_length], mapped_output_nodes),
            dim=1,
        )

    def _sparse_node_update(
        self,
        base: torch.Tensor,
        active_nodes: torch.Tensor,
        active_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Execute private transforms only for packet-selected graph nodes."""

        if self.node_compute_mode == "shared_adapter":
            if self.shared_node_adapters is None:
                raise RuntimeError("shared node adapters are not configured")
            return self.shared_node_adapters(
                base,
                active_nodes,
                active_mass,
                active_mass.gt(0),
            )
        if (
            self.node_embedding is None
            or self.node_weight is None
            or self.node_bias is None
        ):
            raise RuntimeError("full private node compute is not configured")
        batch, tokens, lanes, width = base.shape
        active_width = active_nodes.shape[-1]
        packet_count = batch * tokens * lanes
        packet_base = base.reshape(packet_count, width)
        flat_nodes = active_nodes.reshape(-1)
        flat_mass = active_mass.reshape(-1)
        packet_ids = torch.arange(
            packet_count,
            device=base.device,
            dtype=torch.long,
        ).repeat_interleave(active_width)
        if self.compact_node_dispatch:
            live = flat_mass.gt(0)
            dispatch_nodes = flat_nodes[live]
            dispatch_mass = flat_mass[live]
            dispatch_packets = packet_ids[live]
        else:
            dispatch_nodes = flat_nodes
            dispatch_mass = flat_mass
            dispatch_packets = packet_ids
        can_group = (
            hasattr(torch, "_grouped_mm")
            and base.device.type == "cuda"
            and torch.cuda.get_device_capability(base.device) >= (8, 0)
        )
        if can_group:
            order = dispatch_nodes.argsort(stable=True)
            sorted_nodes = dispatch_nodes[order]
            unique_nodes, counts = torch.unique_consecutive(
                sorted_nodes,
                return_counts=True,
            )
            sorted_input = (
                packet_base[dispatch_packets[order]]
                + self.node_embedding.weight[sorted_nodes]
            )
            offsets = counts.cumsum(0).to(torch.int32)
            # CUDA grouped-MM currently exposes a BF16 kernel on Blackwell
            # even when the surrounding trainer keeps FP32 master weights and
            # activations. Cast only this private-node GEMM, then restore the
            # payload dtype. Gradients flow through both casts into the FP32
            # node parameters. Zero-pad unaligned feature widths to the
            # grouped kernel's 16-byte stride rather than falling back to a
            # Python loop over graph nodes; padding adds no trainable values
            # and the extra output lanes are discarded.
            execution_dtype = torch.bfloat16
            padded_width = _padded_grouped_mm_width(
                width,
                execution_dtype,
            )
            feature_padding = padded_width - width
            grouped_input = (
                sorted_input
                if feature_padding == 0
                else F.pad(sorted_input, (0, feature_padding))
            )
            grouped_weight = self.node_weight[unique_nodes]
            if feature_padding:
                grouped_weight = F.pad(
                    grouped_weight,
                    (0, feature_padding, 0, feature_padding),
                )
            projected = torch._grouped_mm(
                grouped_input.to(execution_dtype),
                grouped_weight.to(execution_dtype),
                offs=offsets,
            )[:, :width].to(base.dtype)
            weighted = torch.tanh(
                projected + self.node_bias[sorted_nodes]
            ) * dispatch_mass[order, None]
            packet_update = torch.zeros_like(packet_base).scatter_add(
                0,
                dispatch_packets[order, None].expand(-1, width),
                weighted,
            )
            return packet_update.view(batch, tokens, lanes, width)

        # Portable numerical fallback. It avoids both the per-packet
        # [B,T,L,active,D,D] weight materialization and computation for nodes
        # with no retained packets. The host checks make this an eager
        # correctness path; aligned builds use grouped MM above.
        packet_update = torch.zeros_like(packet_base)
        for node in range(self.n_nodes):
            selected = dispatch_nodes.eq(node) & dispatch_mass.gt(0)
            if not bool(selected.any()):
                continue
            destination = torch.tanh(
                torch.matmul(
                    packet_base[dispatch_packets[selected]]
                    + self.node_embedding.weight[node],
                    self.node_weight[node],
                )
                + self.node_bias[node]
            ) * dispatch_mass[selected, None]
            packet_update = packet_update.scatter_add(
                0,
                dispatch_packets[selected, None].expand(-1, width),
                destination,
            )
        return packet_update.view(batch, tokens, lanes, width)

    def _node_memory(
        self,
        h: torch.Tensor,
        q: torch.Tensor,
        write: torch.Tensor,
        *,
        receiver_kernel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return receiver-specific memory shared by co-located lanes."""

        _, length, _, _ = h.shape
        # softmax_j,k(log(q_jkn) + relative_bias_ijn + log(write_j))
        # factorizes because the receiver-relative term does not depend on
        # lane k.  Aggregate the distinct lane payloads per writer/node first,
        # then apply the receiver kernel.  This avoids materializing
        # [B, receiver, writer, lane, node].
        if receiver_kernel is None:
            receiver_kernel = self._receiver_kernel(length, h.device)
        if write.ndim == 1:
            write = write[None].expand(h.shape[0], -1)
        if write.shape != h.shape[:2]:
            raise ValueError(
                "write mask must have [tokens] or [batch,tokens] shape"
            )
        writer_mass = (
            q.clamp_min(1e-9)
            * write[:, :, None, None]
        )
        writer_node_mass = writer_mass.sum(dim=2)
        writer_node_state = torch.einsum(
            "bjkn,bjkd->bjnd",
            writer_mass,
            h,
        )
        # Fold mass and state into one feature matrix so receiver mixing needs
        # one grouped matrix multiply instead of separate numerator and
        # denominator contractions.
        writer_features = torch.cat(
            (
                writer_node_mass.unsqueeze(-1),
                writer_node_state,
            ),
            dim=-1,
        ).permute(0, 2, 1, 3)
        mixed = torch.matmul(
            receiver_kernel.permute(2, 0, 1).unsqueeze(0),
            writer_features,
        )
        denominator = mixed[..., 0].permute(0, 2, 1)
        numerator = mixed[..., 1:].permute(0, 2, 1, 3)
        return numerator / denominator.clamp_min(1e-9).unsqueeze(-1)

    def _slot_vote_logits(
        self,
        read_h: torch.Tensor,
        output_addresses: torch.Tensor,
    ) -> torch.Tensor:
        """Return ordinary or slot-private jury logits.

        The isolated A/B reuses the existing terminal query parameter as two
        private scalars per slot and lane (bias and evidence gain). Reusing
        those values keeps the total parameter count identical to the shared
        slot-jury baseline.
        """

        if self.terminal_vote_query is None or self.terminal_address_key is None:
            raise RuntimeError("slot terminal voting parameters are missing")
        output_slots = read_h.shape[1]
        address_vote_logits = torch.einsum(
            "oa,ka->ok",
            output_addresses,
            self.terminal_address_key,
        ) / 4**0.5
        if self.slot_isolation_slots is None:
            return (
                self.terminal_votes[None, None]
                + torch.einsum(
                    "bokd,kd->bok",
                    read_h,
                    self.terminal_vote_query,
                )
                / self.d_model**0.5
                + address_vote_logits[None]
            )
        if output_slots > self.slot_isolation_slots:
            raise ValueError(
                "output slots exceed configured slot-isolation capacity"
            )
        private_values = self.terminal_vote_query.reshape(-1)[
            : self.slot_isolation_slots * self.lanes * 2
        ].view(self.slot_isolation_slots, self.lanes, 2)
        private_bias = private_values[:output_slots, :, 0]
        private_gain = private_values[:output_slots, :, 1]
        lane_evidence = read_h.mean(dim=-1) / self.d_model**0.5
        return (
            self.terminal_votes[None, None]
            + address_vote_logits[None]
            + private_bias[None]
            + private_gain[None] * lane_evidence
        )

    def _packed_terminal_output(
        self,
        hidden: torch.Tensor,
        *,
        observed_length: int,
        output_slots: int,
        address_out: torch.Tensor,
        return_trace: bool,
        locations: list[torch.Tensor] | None,
        transition_entropy: list[torch.Tensor] | None,
        ingress_entropy: torch.Tensor | None,
        active_route_k: list[torch.Tensor] | None,
        route_uncertainty: list[torch.Tensor] | None,
    ):
        """Apply the unchanged terminal jury to packed-relay hidden state."""

        read_h = (
            hidden[:, observed_length:]
            + address_out[None, observed_length:]
        )
        if self.terminal_vote_mode == "global":
            terminal_weights = F.softmax(self.terminal_votes, dim=0)
        elif self.terminal_vote_mode == "prompt":
            vote_features = read_h.mean(dim=1)
            vote_logits = (
                self.terminal_votes[None]
                + torch.einsum(
                    "bkd,kd->bk",
                    vote_features,
                    self.terminal_vote_query,
                )
                / self.d_model**0.5
            )
            if self.training and self.lane_vote_noise_std > 0:
                vote_logits = vote_logits + (
                    torch.randn_like(vote_logits) * self.lane_vote_noise_std
                )
            terminal_weights = F.softmax(vote_logits, dim=-1)
        else:
            output_addresses = fixed_code(
                output_slots,
                4,
                hidden.device,
            )
            vote_logits = self._slot_vote_logits(
                read_h,
                output_addresses,
            )
            if self.training and self.lane_vote_noise_std > 0:
                vote_logits = vote_logits + (
                    torch.randn_like(vote_logits) * self.lane_vote_noise_std
                )
            terminal_weights = F.softmax(vote_logits, dim=-1)
        if self.vocabulary_mode == "private_lane":
            lane_logits = torch.einsum(
                "bokd,kvd->bokv",
                read_h,
                self.token_embedding,
            )
            if terminal_weights.ndim == 1:
                logits = torch.einsum(
                    "k,bokv->bov",
                    terminal_weights,
                    lane_logits,
                )
            elif terminal_weights.ndim == 2:
                logits = torch.einsum(
                    "bk,bokv->bov",
                    terminal_weights,
                    lane_logits,
                )
            else:
                logits = torch.einsum(
                    "bok,bokv->bov",
                    terminal_weights,
                    lane_logits,
                )
            lane_readout = read_h
        else:
            lane_readout = torch.einsum(
                "bokd,ked->boke",
                read_h,
                self.readout_projection,
            )
            if terminal_weights.ndim == 1:
                readout = torch.einsum(
                    "k,boke->boe",
                    terminal_weights,
                    lane_readout,
                )
            elif terminal_weights.ndim == 2:
                readout = torch.einsum(
                    "bk,boke->boe",
                    terminal_weights,
                    lane_readout,
                )
            else:
                readout = torch.einsum(
                    "bok,boke->boe",
                    terminal_weights,
                    lane_readout,
                )
            logits = (
                fp8_tied_readout(readout, self.token_embedding.weight, self.lm_bias)
                if self.fp8_vocab_readout
                else readout @ self.token_embedding.weight.t() + self.lm_bias
            )
        if self._forbidden_output_token_ids.numel():
            logits = logits.index_fill(
                -1,
                self._forbidden_output_token_ids,
                -1e9,
            )
        if not return_trace:
            return logits
        if locations is None or transition_entropy is None:
            raise RuntimeError("packed trace state was not retained")
        stacked_locations = torch.stack(locations, dim=1)
        trace = {
            "locations": stacked_locations,
            "transition_entropy": torch.stack(transition_entropy),
            "ingress_entropy": ingress_entropy,
            "terminal_weights": terminal_weights,
            "start_nodes": self.start_nodes,
            "active_node_union_fraction": stacked_locations.gt(0).any(
                dim=(0, 1, 2, 3)
            ).to(hidden.dtype).mean(),
        }
        if active_route_k is not None:
            trace["active_route_k"] = torch.stack(active_route_k)
        if route_uncertainty is not None:
            trace["route_uncertainty"] = torch.stack(route_uncertainty)
        if self.vocabulary_mode == "private_lane":
            trace["lane_logits"] = lane_logits
        else:
            trace["lane_readout"] = lane_readout
        return logits, trace

    def _forward_packed_from_hidden(
        self,
        hidden: torch.Tensor,
        *,
        observed_length: int,
        output_slots: int,
        valid_positions: torch.Tensor,
        address_state: torch.Tensor,
        address_edges: torch.Tensor,
        address_out: torch.Tensor,
        causal: bool,
        temperature: float | torch.Tensor,
        route_top_k: int | None,
        route_top_k_start: int,
        route_top_k_schedule: tuple[int | None, ...] | None,
        executed_relays: int,
        return_trace: bool,
    ):
        """Execute the relay loop with fixed-cap packed node passports."""

        batch, length, _, _ = hidden.shape
        frontier = self._learned_ingress_frontier(
            hidden,
            temperature=temperature,
        )
        ingress_adaptive = None
        if self.adaptive_route_min_k is not None:
            ingress_adaptive = adaptive_uncertainty_top_k(
                frontier,
                min_k=self.adaptive_route_min_k,
                max_k=frontier.capacity,
                mode=self.adaptive_route_mode,
                entropy_mix=self.adaptive_route_entropy_mix,
            )
            frontier = ingress_adaptive.frontier
        locations = (
            [frontier.to_dense()] if return_trace else None
        )
        ingress_entropy = (
            -(
                frontier.masked_mass.clamp_min(1e-9)
                * frontier.masked_mass.clamp_min(1e-9).log()
            ).sum(-1).mean()
            if return_trace
            else None
        )
        transition_entropy = [] if return_trace else None
        active_route_k = (
            [frontier.valid.sum(-1).to(hidden.dtype).mean()]
            if return_trace
            else None
        )
        route_uncertainty = (
            [ingress_adaptive.uncertainty.mean()]
            if return_trace and ingress_adaptive is not None
            else None
        )
        receiver_kernel = None
        step_states = torch.einsum(
            "ra,kda->rkd",
            fixed_code(self.relay_code_span, 4, hidden.device)[
                : self.relays
            ],
            self.step_projection,
        )
        write_schedule = torch.ones(
            batch,
            self.relays,
            length,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        write_schedule[:, :, observed_length:] = (
            torch.arange(
                self.relays,
                device=hidden.device,
                dtype=hidden.dtype,
            )[None, :, None]
            / max(self.write_warmup_relays - 1, 1)
        ).clamp_max(1)
        write_schedule = (
            write_schedule
            * valid_positions[:, None].to(hidden.dtype)
        )
        address_edges_by_node = address_edges.view(
            length,
            self.n_nodes,
            self.degree,
        )

        for relay in range(executed_relays):
            if relay == 0 and self.bootstrap_context_mode == "self":
                context = hidden
            else:
                if self.packed_memory_backend == "matrix":
                    if receiver_kernel is None:
                        receiver_kernel = self._receiver_kernel(
                            length,
                            hidden.device,
                            causal=causal,
                        )
                    context = packed_convergent_node_context(
                        hidden,
                        frontier.nodes,
                        frontier.mass,
                        frontier.valid,
                        write_schedule[:, relay],
                        receiver_kernel,
                    )
                elif self.packed_memory_backend == "fft_opaque":
                    context = opaque_packed_fft_node_context(
                        hidden,
                        frontier.nodes,
                        frontier.mass,
                        frontier.valid,
                        write_schedule[:, relay],
                        self.relative_node_bias,
                        causal=causal,
                        feature_chunk_size=(
                            self.packed_memory_feature_chunk
                        ),
                        compact_active_nodes=(
                            self.packed_memory_compact_nodes
                        ),
                        assume_unique_routes=True,
                        reader_backend=(
                            self.packed_memory_reader_backend
                        ),
                        writer_backend=(
                            self.packed_memory_writer_backend
                        ),
                    )
                else:
                    context = packed_convolution_node_context(
                        hidden,
                        frontier.nodes,
                        frontier.mass,
                        frontier.valid,
                        write_schedule[:, relay],
                        self.relative_node_bias,
                        causal=causal,
                        feature_chunk_size=(
                            self.packed_memory_feature_chunk
                        ),
                        compact_active_nodes=(
                            self.packed_memory_compact_nodes
                        ),
                        convolution_backend=(
                            "fft"
                            if self.packed_memory_backend == "fft"
                            else "direct"
                        ),
                        reader_backend=(
                            self.packed_memory_reader_backend
                        ),
                        assume_unique_routes=True,
                    )
            destination_candidates = self.destinations[frontier.nodes]
            local_occupancy = packed_node_occupancy(
                frontier,
                position_valid=valid_positions,
                causal=causal,
                query_nodes=destination_candidates,
                query_valid=(
                    frontier.valid.unsqueeze(-1).expand_as(
                        destination_candidates
                    )
                ),
            )
            address_source = address_edges_by_node[
                None,
                :,
                None,
                :,
                :,
            ].expand(
                batch,
                -1,
                self.lanes,
                -1,
                -1,
            )
            selected_address_edges = address_source.gather(
                3,
                frontier.nodes.unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    -1,
                    self.degree,
                ),
            )
            edge_logits = self._shared_low_rank_dynamic_router_logits(
                hidden,
                context,
                local_occupancy,
                frontier.nodes,
                selected_address_edges,
            )
            transition = F.softmax(
                edge_logits / temperature,
                dim=-1,
            )
            if return_trace:
                edge_entropy = -(
                    transition.clamp_min(1e-9)
                    * transition.clamp_min(1e-9).log()
                ).sum(-1)
                transition_entropy.append(
                    (
                        frontier.masked_mass * edge_entropy
                    ).sum(-1).mean()
                )
            relay_top_k = (
                route_top_k_schedule[relay]
                if route_top_k_schedule is not None
                else (
                    route_top_k
                    if relay >= route_top_k_start
                    else None
                )
            )
            configured_top_k = (
                relay_top_k
                if relay_top_k is not None
                else self.active_node_top_k
            )
            if configured_top_k is None:
                raise RuntimeError("packed relay has no frontier capacity")
            next_capacity = min(
                configured_top_k,
                frontier.capacity * self.degree,
                self.n_nodes,
            )
            next_frontier = transport_packed_frontier(
                frontier,
                destination_candidates,
                transition,
                top_k=next_capacity,
                mass_threshold=self.active_node_mass_threshold,
            )
            relay_adaptive = None
            if self.adaptive_route_min_k is not None:
                relay_adaptive = adaptive_uncertainty_top_k(
                    next_frontier,
                    min_k=min(
                        self.adaptive_route_min_k,
                        next_frontier.capacity,
                    ),
                    max_k=next_frontier.capacity,
                    mode=self.adaptive_route_mode,
                    entropy_mix=self.adaptive_route_entropy_mix,
                )
                next_frontier = relay_adaptive.frontier

            base = (
                self._lane_norm(hidden + context)
                + address_state[None]
                + step_states[relay][None, None]
            )
            adapter_input = base.permute(0, 2, 1, 3)
            adapter_hidden = F.gelu(
                cell_linear(adapter_input, self.lane_adapter)
            )
            adapter_output = cell_linear(
                adapter_hidden,
                self.lane_adapter_out,
            ).permute(0, 2, 1, 3)
            base = self._lane_norm(
                base
                + torch.tanh(
                    adapter_output
                    + self.lane_adapter_bias[None, None]
                )
            )
            update = self._sparse_node_update(
                base,
                next_frontier.nodes,
                next_frontier.masked_mass,
            )
            hidden = self._lane_norm(hidden + update)
            frontier = next_frontier
            if return_trace:
                locations.append(frontier.to_dense())
                if active_route_k is None:
                    raise RuntimeError("packed route-width trace was lost")
                active_route_k.append(
                    frontier.valid.sum(-1).to(hidden.dtype).mean()
                )
                if relay_adaptive is not None:
                    if route_uncertainty is None:
                        raise RuntimeError(
                            "adaptive uncertainty trace was not initialized"
                        )
                    route_uncertainty.append(
                        relay_adaptive.uncertainty.mean()
                    )

        return self._packed_terminal_output(
            hidden,
            observed_length=observed_length,
            output_slots=output_slots,
            address_out=address_out,
            return_trace=return_trace,
            locations=locations,
            transition_entropy=transition_entropy,
            ingress_entropy=ingress_entropy,
            active_route_k=active_route_k,
            route_uncertainty=route_uncertainty,
        )

    def forward(
        self,
        observed: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        *,
        output_slots: int = 7,
        output_token_ids: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
        causal: bool = False,
        temperature: float | torch.Tensor = 0.35,
        route_top_k: int | None = None,
        route_top_k_start: int = 0,
        route_top_k_schedule: tuple[int | None, ...] | None = None,
        relay_limit: int | None = None,
        return_trace: bool = False,
        position_norm_length: int | None = None,
    ):
        if relay_limit is not None and (
            isinstance(relay_limit, bool)
            or not isinstance(relay_limit, int)
            or not 1 <= relay_limit <= self.relays
        ):
            raise ValueError("relay_limit must be in [1, relays]")
        executed_relays = (
            self.relays if relay_limit is None else relay_limit
        )
        if route_top_k is not None and route_top_k_schedule is not None:
            raise ValueError(
                "route_top_k and route_top_k_schedule are mutually exclusive"
            )
        if (
            route_top_k is not None
            and (
                isinstance(route_top_k, bool)
                or route_top_k <= 0
            )
        ):
            raise ValueError("route_top_k must be a positive integer or None")
        if (
            isinstance(route_top_k_start, bool)
            or route_top_k_start < 0
            or route_top_k_start >= self.relays
        ):
            raise ValueError(
                "route_top_k_start must select a configured relay"
            )
        if route_top_k_schedule is not None:
            if len(route_top_k_schedule) != self.relays:
                raise ValueError(
                    "route_top_k_schedule must have one entry per relay"
                )
            if any(
                value is not None
                and (
                    isinstance(value, bool)
                    or value <= 0
                )
                for value in route_top_k_schedule
            ):
                raise ValueError(
                    "route schedule entries must be positive integers or None"
                )
        batch, observed_length = observed.shape
        position_norm_length = (
            self.position_norm_length
            if position_norm_length is None
            else position_norm_length
        )
        if causal and self.terminal_vote_mode == "prompt":
            raise ValueError(
                "causal execution requires global or slot terminal voting"
            )
        if output_token_ids is not None and (
            output_token_ids.shape != (batch, output_slots)
            or output_token_ids.dtype != torch.long
        ):
            raise ValueError(
                "output_token_ids must be long with [batch, output_slots] shape"
            )
        if output_mask is None:
            output_mask = torch.ones(
                batch,
                output_slots,
                dtype=torch.bool,
                device=observed.device,
            )
        elif (
            output_mask.shape != (batch, output_slots)
            or output_mask.dtype != torch.bool
        ):
            raise ValueError(
                "output_mask must be boolean with [batch, output_slots] shape"
            )
        if context_mask is None:
            context_mask = torch.ones(
                batch,
                observed_length,
                dtype=torch.bool,
                device=observed.device,
            )
        elif (
            context_mask.shape != observed.shape
            or context_mask.dtype != torch.bool
        ):
            raise ValueError(
                "context_mask must be boolean with observed shape"
            )
        length = observed_length + output_slots
        valid_positions = torch.cat(
            (
                context_mask,
                output_mask,
            ),
            dim=1,
        )
        addresses = fixed_code(
            length, 4, observed.device, norm_length=position_norm_length
        )
        address_state = torch.einsum(
            "ta,kda->tkd",
            addresses,
            self.address_to_state,
        )
        address_edges = self.address_to_edges(addresses)
        address_out = torch.einsum(
            "ta,kda->tkd",
            addresses,
            self.address_to_readout,
        )

        if self.vocabulary_mode == "private_lane":
            if output_token_ids is None:
                output_h = self.token_embedding.mean(1).view(
                    1,
                    1,
                    self.lanes,
                    self.d_model,
                ).expand(batch, output_slots, self.lanes, -1)
            else:
                output_h = self.token_embedding[
                    :, output_token_ids
                ].permute(1, 2, 0, 3)
            observed_h = self.token_embedding[:, observed].permute(
                1,
                2,
                0,
                3,
            )
        else:
            observed_embedding = self.token_embedding(observed)
            observed_h = torch.einsum(
                "bte,kde->btkd",
                observed_embedding,
                self.input_projection,
            )
            if output_token_ids is not None:
                output_embedding = self.token_embedding(output_token_ids)
                output_h = torch.einsum(
                    "bte,kde->btkd",
                    output_embedding,
                    self.input_projection,
                )
            else:
                if self.blank_token_id is None:
                    blank_embedding = self.token_embedding.weight.mean(0)
                else:
                    blank_embedding = self.token_embedding.weight[
                        self.blank_token_id
                    ]
                blank_lane = torch.einsum(
                    "e,kde->kd",
                    blank_embedding,
                    self.input_projection,
                )
                output_h = blank_lane[None, None].expand(
                    batch,
                    output_slots,
                    -1,
                    -1,
                )
        h = torch.cat((observed_h, output_h), dim=1)
        h = self._lane_norm(h + address_state[None])

        if self.route_backend == "packed":
            return self._forward_packed_from_hidden(
                h,
                observed_length=observed_length,
                output_slots=output_slots,
                valid_positions=valid_positions,
                address_state=address_state,
                address_edges=address_edges,
                address_out=address_out,
                causal=causal,
                temperature=temperature,
                route_top_k=route_top_k,
                route_top_k_start=route_top_k_start,
                route_top_k_schedule=route_top_k_schedule,
                executed_relays=executed_relays,
                return_trace=return_trace,
            )

        if self.ingress_mode == "learned":
            q, current_active_nodes, _ = (
                self._learned_ingress(
                    h,
                    temperature=temperature,
                    top_k=self.active_node_top_k,
                )
            )
        else:
            initial = F.one_hot(
                self.start_nodes,
                num_classes=self.n_nodes,
            ).to(h.dtype)
            q = initial[None, None].expand(
                batch,
                length,
                -1,
                -1,
            )
            current_active_nodes = None
        locations = [q] if return_trace else None
        ingress_entropy = (
            -(
                q.clamp_min(1e-9)
                * q.clamp_min(1e-9).log()
            ).sum(-1).mean()
            if return_trace
            else None
        )
        transition_entropy = [] if return_trace else None
        receiver_kernel = None
        step_states = torch.einsum(
            "ra,kda->rkd",
            fixed_code(self.relay_code_span, 4, h.device)[
                : self.relays
            ],
            self.step_projection,
        )
        write_schedule = torch.ones(
            batch,
            self.relays,
            length,
            device=h.device,
            dtype=h.dtype,
        )
        write_schedule[:, :, observed_length:] = (
            torch.arange(
                self.relays,
                device=h.device,
                dtype=h.dtype,
            )[None, :, None]
            / max(self.write_warmup_relays - 1, 1)
        ).clamp_max(1)
        write_schedule = (
            write_schedule
            * valid_positions[:, None].to(h.dtype)
        )
        address_edges_by_node = address_edges.view(
            length,
            self.n_nodes,
            self.degree,
        )

        for relay in range(executed_relays):
            if relay == 0 and self.bootstrap_context_mode == "self":
                context = h
            else:
                if receiver_kernel is None:
                    receiver_kernel = self._receiver_kernel(
                        length,
                        h.device,
                        causal=causal,
                        norm_length=position_norm_length,
                    )
                node_memory = self._node_memory(
                    h,
                    q,
                    write_schedule[:, relay],
                    receiver_kernel=receiver_kernel,
                )
                context = torch.einsum(
                    "btkn,btnd->btkd",
                    q,
                    node_memory,
                )
            occupancy_weight = valid_positions[
                :, :, None, None
            ].to(q.dtype)
            position_occupancy = (
                q * occupancy_weight
            ).sum(dim=2)
            if causal:
                occupancy = position_occupancy.cumsum(dim=1) / (
                    valid_positions.cumsum(dim=1).to(q.dtype)
                    * self.lanes
                ).clamp_min(1).unsqueeze(-1)
            else:
                occupancy = position_occupancy.sum(dim=1) / (
                    valid_positions.sum(1, keepdim=True).to(q.dtype)
                    * self.lanes
                ).clamp_min(1)
            source_width = self._reachable_count_values[relay]
            dynamic_sources = (
                current_active_nodes is not None
                and self.router_mode == "shared_low_rank"
            )
            if dynamic_sources:
                source_nodes = current_active_nodes
                destination_candidates = self.destinations[source_nodes]
                if causal:
                    occupancy_source = occupancy[
                        :,
                        :,
                        None,
                        None,
                        :,
                    ].expand(
                        -1,
                        -1,
                        self.lanes,
                        source_nodes.shape[-1],
                        -1,
                    )
                else:
                    occupancy_source = occupancy[
                        :,
                        None,
                        None,
                        None,
                        :,
                    ].expand(
                        -1,
                        length,
                        self.lanes,
                        source_nodes.shape[-1],
                        -1,
                    )
                local_occupancy = occupancy_source.gather(
                    -1,
                    destination_candidates,
                )
                address_source = address_edges_by_node[
                    None,
                    :,
                    None,
                    :,
                    :,
                ].expand(
                    batch,
                    -1,
                    self.lanes,
                    -1,
                    -1,
                )
                selected_address_edges = address_source.gather(
                    3,
                    source_nodes.unsqueeze(-1).expand(
                        -1,
                        -1,
                        -1,
                        -1,
                        self.degree,
                    ),
                )
                edge_logits = (
                    self._shared_low_rank_dynamic_router_logits(
                        h,
                        context,
                        local_occupancy,
                        source_nodes,
                        selected_address_edges,
                    )
                )
                source_q = q.gather(-1, source_nodes)
                scatter_destinations = destination_candidates.flatten(-2)
            else:
                if (
                    self.ingress_mode == "fixed"
                    and self.reachable_node_execution
                    and source_width < self.n_nodes
                ):
                    source_nodes = self.reachable_nodes[
                        relay,
                        :,
                        :source_width,
                    ]
                    source_mask = self.reachable_mask[
                        relay,
                        :,
                        :source_width,
                    ]
                else:
                    source_nodes = torch.arange(
                        self.n_nodes,
                        device=h.device,
                    )[None].expand(self.lanes, -1)
                    source_mask = torch.ones_like(
                        source_nodes,
                        dtype=torch.bool,
                    )
                destination_candidates = self.destinations[source_nodes]
                if causal:
                    local_occupancy = occupancy[
                        :,
                        :,
                        destination_candidates,
                    ]
                else:
                    local_occupancy = occupancy[
                        :,
                        destination_candidates,
                    ]
                selected_address_edges = address_edges_by_node[
                    :,
                    source_nodes,
                ]
                if self.router_mode == "private_full":
                    semantic_router_input = torch.cat(
                        (h, context),
                        dim=-1,
                    )
                    router_weight = self.router_weight[source_nodes]
                    router_bias = self.router_bias[source_nodes]
                    edge_logits = self._factored_router_logits(
                        semantic_router_input,
                        local_occupancy,
                        router_weight,
                        router_bias,
                        selected_address_edges,
                    )
                else:
                    edge_logits = self._shared_low_rank_router_logits(
                        h,
                        context,
                        local_occupancy,
                        source_nodes,
                        selected_address_edges,
                    )
                source_q = q.gather(
                    -1,
                    source_nodes[None, None].expand(
                        batch,
                        length,
                        -1,
                        -1,
                    ),
                ) * source_mask[None, None]
                scatter_destinations = destination_candidates.flatten(
                    -2
                )[None, None].expand(
                    batch,
                    length,
                    self.lanes,
                    -1,
                )
            transition = F.softmax(
                edge_logits / temperature,
                dim=-1,
            )
            if return_trace:
                edge_entropy = -(
                    transition.clamp_min(1e-9)
                    * transition.clamp_min(1e-9).log()
                ).sum(-1)
                transition_entropy.append(
                    (source_q * edge_entropy).sum(-1).mean()
                )
            contributions = (
                source_q.unsqueeze(-1) * transition
            ).flatten(-2)
            q_next = torch.zeros_like(q).scatter_add(
                -1,
                scatter_destinations,
                contributions,
            )
            relay_top_k = (
                route_top_k_schedule[relay]
                if route_top_k_schedule is not None
                else (
                    route_top_k
                    if relay >= route_top_k_start
                    else None
                )
            )
            configured_top_k = (
                relay_top_k
                if relay_top_k is not None
                else self.active_node_top_k
            )
            if (
                configured_top_k is not None
                and self.ingress_mode == "fixed"
                and self.active_node_top_k is None
                and self.active_node_mass_threshold <= 0.0
            ):
                # Preserve the legacy optional top-k frontier exactly.
                configured_top_k = min(
                    configured_top_k,
                    2 * (relay + 1) + 1,
                )
            q_next, active_nodes, active_mass = (
                self._sparsify_locations(
                    q_next,
                    top_k=configured_top_k,
                    mass_threshold=self.active_node_mass_threshold,
                )
            )

            base = (
                self._lane_norm(h + context)
                + address_state[None]
                + step_states[relay][None, None]
            )
            adapter_input = base.permute(0, 2, 1, 3)
            adapter_hidden = F.gelu(
                cell_linear(
                    adapter_input,
                    self.lane_adapter,
                )
            )
            adapter_output = cell_linear(
                adapter_hidden,
                self.lane_adapter_out,
            ).permute(0, 2, 1, 3)
            base = self._lane_norm(
                base
                + torch.tanh(
                    adapter_output
                    + self.lane_adapter_bias[None, None]
                )
            )
            reachable_width = self._reachable_count_values[relay + 1]
            if active_nodes is not None and active_mass is not None:
                compute_active_nodes = self._slot_owned_compute_nodes(
                    active_nodes,
                    observed_length=observed_length,
                    output_slots=output_slots,
                )
                update = self._sparse_node_update(
                    base,
                    compute_active_nodes,
                    active_mass,
                )
            elif (
                self.ingress_mode == "fixed"
                and
                self.reachable_node_execution
                and reachable_width < self.n_nodes
            ):
                reachable = self.reachable_nodes[
                    relay + 1,
                    :,
                    :reachable_width,
                ]
                reachable_mask = self.reachable_mask[
                    relay + 1,
                    :,
                    :reachable_width,
                ]
                node_embedding = self.node_embedding.weight[reachable]
                node_weight = self.node_weight[reachable]
                node_bias = self.node_bias[reachable]
                destination_h = torch.tanh(
                    folded_cell_node_linear(
                        base.permute(0, 2, 1, 3),
                        node_embedding,
                        node_weight.transpose(-1, -2),
                        node_bias,
                    ).permute(0, 2, 1, 3, 4)
                )
                active_q = q_next.gather(
                    -1,
                    reachable[None, None].expand(
                        batch,
                        length,
                        -1,
                        -1,
                    ),
                ) * reachable_mask[None, None]
                update = torch.einsum(
                    "btks,btksd->btkd",
                    active_q,
                    destination_h,
                )
            else:
                node_embedding = self.node_embedding.weight[None].expand(
                    self.lanes,
                    -1,
                    -1,
                )
                node_weight = self.node_weight[None].expand(
                    self.lanes,
                    -1,
                    -1,
                    -1,
                )
                node_bias = self.node_bias[None].expand(
                    self.lanes,
                    -1,
                    -1,
                )
                destination_h = torch.tanh(
                    folded_cell_node_linear(
                        base.permute(0, 2, 1, 3),
                        node_embedding,
                        node_weight.transpose(-1, -2),
                        node_bias,
                    ).permute(0, 2, 1, 3, 4)
                )
                update = torch.einsum(
                    "btkn,btknd->btkd",
                    q_next,
                    destination_h,
                )
            h = self._lane_norm(h + update)
            q = q_next
            current_active_nodes = active_nodes
            if return_trace:
                locations.append(q)

        read_h = (
            h[:, observed_length:]
            + address_out[None, observed_length:]
        )
        if self.terminal_vote_mode == "global":
            terminal_weights = F.softmax(self.terminal_votes, dim=0)
        elif self.terminal_vote_mode == "prompt":
            # One prompt chooses one jury for the complete simultaneous
            # formation. Averaging final output-query states preserves a
            # shared sequence-level coalition rather than selecting an
            # unrelated voter set independently at every output slot.
            vote_features = read_h.mean(dim=1)
            vote_logits = (
                self.terminal_votes[None]
                + torch.einsum(
                    "bkd,kd->bk",
                    vote_features,
                    self.terminal_vote_query,
                )
                / self.d_model**0.5
            )
            if self.training and self.lane_vote_noise_std > 0:
                vote_logits = vote_logits + (
                    torch.randn_like(vote_logits) * self.lane_vote_noise_std
                )
            terminal_weights = F.softmax(vote_logits, dim=-1)
        else:
            # Each immutable output address chooses its own prompt-aware
            # coalition. read_h contains the final routed lane state plus a
            # freshly restored address.  The separate address key below also
            # injects the fixed coordinate directly into the lane logits, so
            # slot identity cannot be erased by relay mixing or ignored while
            # the mutable states are still collapsed.
            # Use a dedicated output-relative coordinate system here.  The
            # relay addresses span context + output, which makes adjacent
            # output coordinates almost identical at long contexts (for
            # example, pi/519 apart at C512/O8).  Jury addresses must span
            # the formation itself so all simultaneous slots remain clearly
            # distinguishable regardless of prompt-window length.
            output_addresses = fixed_code(
                output_slots,
                4,
                observed.device,
            )
            vote_logits = self._slot_vote_logits(
                read_h,
                output_addresses,
            )
            if self.training and self.lane_vote_noise_std > 0:
                vote_logits = vote_logits + (
                    torch.randn_like(vote_logits) * self.lane_vote_noise_std
                )
            terminal_weights = F.softmax(vote_logits, dim=-1)
        if self.vocabulary_mode == "private_lane":
            lane_logits = torch.einsum(
                "bokd,kvd->bokv",
                read_h,
                self.token_embedding,
            )
            if terminal_weights.ndim == 1:
                logits = torch.einsum(
                    "k,bokv->bov",
                    terminal_weights,
                    lane_logits,
                )
            elif terminal_weights.ndim == 2:
                logits = torch.einsum(
                    "bk,bokv->bov",
                    terminal_weights,
                    lane_logits,
                )
            else:
                logits = torch.einsum(
                    "bok,bokv->bov",
                    terminal_weights,
                    lane_logits,
                )
            lane_readout = read_h
        else:
            # Fuse complementary lane views before the large tied lexical
            # projection. This is algebraically the same as voting over K
            # vocabulary heads without materializing [B,O,K,V].
            lane_readout = torch.einsum(
                "bokd,ked->boke",
                read_h,
                self.readout_projection,
            )
            if terminal_weights.ndim == 1:
                readout = torch.einsum(
                    "k,boke->boe",
                    terminal_weights,
                    lane_readout,
                )
            elif terminal_weights.ndim == 2:
                readout = torch.einsum(
                    "bk,boke->boe",
                    terminal_weights,
                    lane_readout,
                )
            else:
                readout = torch.einsum(
                    "bok,boke->boe",
                    terminal_weights,
                    lane_readout,
                )
            logits = (
                fp8_tied_readout(readout, self.token_embedding.weight, self.lm_bias)
                if self.fp8_vocab_readout
                else readout @ self.token_embedding.weight.t() + self.lm_bias
            )
        if self._forbidden_output_token_ids.numel():
            logits = logits.index_fill(
                -1,
                self._forbidden_output_token_ids,
                -1e9,
            )
        if not return_trace:
            return logits
        trace = {
            "locations": torch.stack(locations, dim=1),
            "transition_entropy": torch.stack(transition_entropy),
            "ingress_entropy": ingress_entropy,
            "terminal_weights": terminal_weights,
            "start_nodes": self.start_nodes,
        }
        if self.vocabulary_mode == "private_lane":
            trace["lane_logits"] = lane_logits
        else:
            trace["lane_readout"] = lane_readout
        return logits, trace


@dataclass(frozen=True)
class ConvergentLanguageConfig:
    """Tied-vocabulary simultaneous language configuration."""

    vocab_size: int
    d_model: int = 20
    embedding_dim: int = 32
    n_nodes: int = 66
    degree: int = 3
    router_mode: Literal[
        "private_full",
        "shared_low_rank",
    ] = "private_full"
    router_rank: int = 16
    ingress_mode: Literal["fixed", "learned"] = "fixed"
    route_backend: Literal["dense", "packed"] = "dense"
    packed_memory_backend: Literal[
        "matrix",
        "convolution",
        "fft",
        "fft_opaque",
    ] = "matrix"
    packed_memory_reader_backend: Literal[
        "torch",
        "triton",
    ] = "torch"
    packed_memory_writer_backend: Literal[
        "torch",
        "triton",
    ] = "torch"
    packed_memory_feature_chunk: int = 32
    packed_memory_compact_nodes: bool = False
    adaptive_route_min_k: int | None = None
    adaptive_route_mode: Literal[
        "entropy",
        "margin",
        "hybrid",
    ] = "hybrid"
    adaptive_route_entropy_mix: float = 0.5
    compact_node_dispatch: bool = False
    node_compute_mode: Literal[
        "full_private",
        "shared_adapter",
    ] = "full_private"
    node_adapter_rank: int = 8
    ingress_groups: int | None = None
    ingress_top_groups: int = 2
    ingress_learned_groups: bool = True
    active_node_top_k: int | None = None
    active_node_mass_threshold: float = 0.0
    relays: int = 2
    lanes: int = 10
    mask_token_id: int = 3
    blank_token_id: int | None = None
    forbidden_output_token_ids: tuple[int, ...] = ()
    connection_layout: Literal[
        "ring",
        "small_world",
        "dense",
    ] = "small_world"
    connection_seed: int = 0
    connection_rewire: float = 0.5
    relay_code_span: int | None = None
    write_warmup_relays: int | None = None
    bootstrap_context_mode: Literal["shared", "self"] = "shared"
    terminal_vote_mode: Literal[
        "global",
        "prompt",
        "slot",
    ] = "global"
    slot_isolation_slots: int | None = None
    lane_vote_noise_std: float = 0.0
    reachable_node_execution: bool = True
    position_norm_length: int | None = None
    fp8_vocab_readout: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0 or self.embedding_dim <= 0:
            raise ValueError(
                "d_model and embedding_dim must be positive"
            )
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("mask_token_id must be inside vocabulary")
        if self.blank_token_id is not None and not (
            0 <= self.blank_token_id < self.vocab_size
        ):
            raise ValueError("blank_token_id must be inside vocabulary")
        if self.terminal_vote_mode not in {"global", "prompt", "slot"}:
            raise ValueError(
                "terminal_vote_mode must be global, prompt, or slot"
            )
        if self.slot_isolation_slots is not None:
            if not 1 <= self.slot_isolation_slots <= self.n_nodes:
                raise ValueError(
                    "slot_isolation_slots must be in [1, n_nodes]"
                )
            if self.terminal_vote_mode != "slot":
                raise ValueError(
                    "slot-isolated computation requires slot terminal voting"
                )
            if 2 * self.slot_isolation_slots > self.d_model:
                raise ValueError(
                    "slot-isolated juries require d_model >= 2 * slots"
                )
        if self.lane_vote_noise_std < 0:
            raise ValueError("lane_vote_noise_std must be non-negative")
        if self.router_mode not in {"private_full", "shared_low_rank"}:
            raise ValueError(
                "router_mode must be private_full or shared_low_rank"
            )
        if self.router_rank <= 0:
            raise ValueError("router_rank must be positive")
        if self.ingress_mode not in {"fixed", "learned"}:
            raise ValueError("ingress_mode must be fixed or learned")
        if self.route_backend not in {"dense", "packed"}:
            raise ValueError("route_backend must be dense or packed")
        if self.packed_memory_backend not in {
            "matrix",
            "convolution",
            "fft",
            "fft_opaque",
        }:
            raise ValueError(
                "packed_memory_backend must be matrix, convolution, fft, "
                "or fft_opaque"
            )
        if self.packed_memory_reader_backend not in {"torch", "triton"}:
            raise ValueError(
                "packed_memory_reader_backend must be torch or triton"
            )
        if (
            self.packed_memory_reader_backend == "triton"
            and self.packed_memory_backend == "matrix"
        ):
            raise ValueError(
                "the Triton packed reader requires convolution, fft, or "
                "fft_opaque receiver memory"
            )
        if self.packed_memory_writer_backend not in {"torch", "triton"}:
            raise ValueError(
                "packed_memory_writer_backend must be torch or triton"
            )
        if (
            self.packed_memory_writer_backend == "triton"
            and (
                self.route_backend != "packed"
                or self.packed_memory_backend != "fft_opaque"
            )
        ):
            raise ValueError(
                "the Triton packed writer currently requires "
                "route_backend=packed and packed_memory_backend=fft_opaque"
            )
        if (
            self.packed_memory_writer_backend == "triton"
            and self.packed_memory_feature_chunk > 128
        ):
            raise ValueError(
                "the Triton packed writer supports feature chunks up to 128"
            )
        if self.packed_memory_feature_chunk <= 0:
            raise ValueError("packed_memory_feature_chunk must be positive")
        if (
            self.route_backend != "packed"
            and self.packed_memory_backend != "matrix"
        ):
            raise ValueError(
                "convolutional packed memory requires route_backend=packed"
            )
        if self.adaptive_route_mode not in {
            "entropy",
            "margin",
            "hybrid",
        }:
            raise ValueError(
                "adaptive_route_mode must be entropy, margin, or hybrid"
            )
        if not 0.0 <= self.adaptive_route_entropy_mix <= 1.0:
            raise ValueError("adaptive_route_entropy_mix must be in [0, 1]")
        if self.adaptive_route_min_k is not None:
            if self.route_backend != "packed" or self.active_node_top_k is None:
                raise ValueError(
                    "adaptive_route_min_k requires a fixed-cap packed route"
                )
            if not 1 <= self.adaptive_route_min_k <= self.active_node_top_k:
                raise ValueError(
                    "adaptive_route_min_k must be in [1, active_node_top_k]"
                )
        if self.node_compute_mode not in {"full_private", "shared_adapter"}:
            raise ValueError(
                "node_compute_mode must be full_private or shared_adapter"
            )
        if self.node_adapter_rank <= 0:
            raise ValueError("node_adapter_rank must be positive")
        if (
            self.node_compute_mode == "shared_adapter"
            and self.route_backend != "packed"
        ):
            raise ValueError(
                "shared_adapter node compute currently requires packed routes"
            )
        if self.ingress_top_groups <= 0:
            raise ValueError("ingress_top_groups must be positive")
        if self.ingress_groups is not None:
            if not 1 <= self.ingress_groups <= self.n_nodes:
                raise ValueError("ingress_groups must be in [1, n_nodes]")
            if self.route_backend != "packed" or self.ingress_mode != "learned":
                raise ValueError(
                    "hierarchical ingress requires packed learned ingress"
                )
            if self.ingress_top_groups > self.ingress_groups:
                raise ValueError(
                    "ingress_top_groups cannot exceed ingress_groups"
                )
            group_capacity = (
                self.n_nodes + self.ingress_groups - 1
            ) // self.ingress_groups
            if self.active_node_top_k is None or self.active_node_top_k > min(
                self.n_nodes,
                self.ingress_top_groups * group_capacity,
            ):
                raise ValueError(
                    "active_node_top_k exceeds hierarchical ingress candidates"
                )
        if (
            self.route_backend == "packed"
            and self.active_node_top_k is None
        ):
            raise ValueError(
                "packed route_backend requires active_node_top_k"
            )
        if self.route_backend == "packed" and (
            self.router_mode != "shared_low_rank"
            or self.ingress_mode != "learned"
        ):
            raise ValueError(
                "packed route_backend currently requires shared_low_rank "
                "routing with learned ingress"
            )
        if (
            self.ingress_mode == "learned"
            and self.router_mode != "shared_low_rank"
        ):
            raise ValueError(
                "learned ingress requires the shared_low_rank router"
            )
        if self.active_node_top_k is not None and not (
            1 <= self.active_node_top_k <= self.n_nodes
        ):
            raise ValueError("active_node_top_k must be in [1, n_nodes]")
        if not 0.0 <= self.active_node_mass_threshold < 1.0:
            raise ValueError(
                "active_node_mass_threshold must be in [0, 1)"
            )

    def to_dict(self) -> dict:
        return asdict(self)


class ConvergentLanguageModel(nn.Module):
    """R2 connected Gator with one tied 2,048-piece lexical shell."""

    def __init__(self, config: ConvergentLanguageConfig) -> None:
        super().__init__()
        self.config = config
        self.core = ConvergentGator(
            n_nodes=config.n_nodes,
            d_model=config.d_model,
            degree=config.degree,
            router_mode=config.router_mode,
            router_rank=config.router_rank,
            ingress_mode=config.ingress_mode,
            route_backend=config.route_backend,
            packed_memory_backend=config.packed_memory_backend,
            packed_memory_reader_backend=(
                config.packed_memory_reader_backend
            ),
            packed_memory_writer_backend=(
                config.packed_memory_writer_backend
            ),
            packed_memory_feature_chunk=(
                config.packed_memory_feature_chunk
            ),
            packed_memory_compact_nodes=(
                config.packed_memory_compact_nodes
            ),
            adaptive_route_min_k=config.adaptive_route_min_k,
            adaptive_route_mode=config.adaptive_route_mode,
            adaptive_route_entropy_mix=(
                config.adaptive_route_entropy_mix
            ),
            compact_node_dispatch=config.compact_node_dispatch,
            node_compute_mode=config.node_compute_mode,
            node_adapter_rank=config.node_adapter_rank,
            ingress_groups=config.ingress_groups,
            ingress_top_groups=config.ingress_top_groups,
            ingress_learned_groups=config.ingress_learned_groups,
            active_node_top_k=config.active_node_top_k,
            active_node_mass_threshold=(
                config.active_node_mass_threshold
            ),
            relays=config.relays,
            lanes=config.lanes,
            reachable_node_execution=config.reachable_node_execution,
            relay_code_span=config.relay_code_span,
            write_warmup_relays=config.write_warmup_relays,
            bootstrap_context_mode=config.bootstrap_context_mode,
            connection_layout=config.connection_layout,
            connection_seed=config.connection_seed,
            connection_rewire=config.connection_rewire,
            terminal_vote_mode=config.terminal_vote_mode,
            slot_isolation_slots=config.slot_isolation_slots,
            lane_vote_noise_std=config.lane_vote_noise_std,
            vocabulary_mode="shared_projected",
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            blank_token_id=config.blank_token_id,
            forbidden_output_token_ids=(
                config.forbidden_output_token_ids
            ),
            position_norm_length=config.position_norm_length,
            fp8_vocab_readout=config.fp8_vocab_readout,
        )

    @property
    def parameter_count(self) -> int:
        return self.core.parameter_count

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        output_slots: int,
        output_token_ids: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
        causal: bool = False,
        temperature: float | torch.Tensor = 0.35,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.core(
            context_ids,
            context_mask,
            output_slots=output_slots,
            output_token_ids=output_token_ids,
            output_mask=output_mask,
            causal=causal,
            temperature=temperature,
            return_trace=return_trace,
        )


def default_convergent_checkpoint_path(
    *,
    seed: int,
    route_backend: str,
    packed_memory_backend: str,
    packed_memory_reader_backend: str,
    packed_memory_writer_backend: str = "torch",
) -> Path:
    """Return a collision-resistant default checkpoint for execution A/Bs."""

    return Path(
        "gator_convergent_"
        f"{route_backend}_{packed_memory_backend}_"
        f"reader-{packed_memory_reader_backend}_"
        f"writer-{packed_memory_writer_backend}_seed{seed}.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument(
        "--router-mode",
        choices=("private_full", "shared_low_rank"),
        default="private_full",
    )
    parser.add_argument("--router-rank", type=int, default=16)
    parser.add_argument(
        "--ingress-mode",
        choices=("fixed", "learned"),
        default="fixed",
    )
    parser.add_argument(
        "--route-backend",
        choices=("dense", "packed"),
        default="dense",
    )
    parser.add_argument(
        "--packed-memory-backend",
        choices=("matrix", "convolution", "fft", "fft_opaque"),
        default="matrix",
    )
    parser.add_argument(
        "--packed-memory-reader-backend",
        choices=("torch", "triton"),
        default="torch",
    )
    parser.add_argument(
        "--packed-memory-writer-backend",
        choices=("torch", "triton"),
        default="torch",
    )
    parser.add_argument(
        "--packed-memory-feature-chunk",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--packed-memory-compact-nodes",
        action="store_true",
        help="eager-only dynamic union of nodes used by packed receiver memory",
    )
    parser.add_argument("--adaptive-route-min-k", type=int, default=None)
    parser.add_argument(
        "--adaptive-route-mode",
        choices=("entropy", "margin", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--adaptive-route-entropy-mix",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--compact-node-dispatch",
        action="store_true",
        help="eager-only removal of zero-mass rows before grouped node GEMMs",
    )
    parser.add_argument(
        "--node-compute-mode",
        choices=("full_private", "shared_adapter"),
        default="full_private",
    )
    parser.add_argument("--node-adapter-rank", type=int, default=8)
    parser.add_argument("--ingress-groups", type=int, default=None)
    parser.add_argument("--ingress-top-groups", type=int, default=2)
    parser.add_argument(
        "--no-ingress-learned-groups",
        action="store_false",
        dest="ingress_learned_groups",
    )
    parser.add_argument("--active-node-top-k", type=int, default=None)
    parser.add_argument(
        "--active-node-mass-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--connection-layout",
        choices=("ring", "small_world", "dense"),
        default="ring",
        help=(
            "fixed outgoing connection graph; small_world keeps a directed "
            "backbone and replaces geometric neighbors with shortcuts"
        ),
    )
    parser.add_argument("--connection-seed", type=int, default=0)
    parser.add_argument(
        "--connection-rewire",
        type=float,
        default=1.0,
        help=(
            "fraction of non-backbone ring edges replaced by fixed shortcuts "
            "in the small_world layout"
        ),
    )
    parser.add_argument("--relays", type=int, default=4)
    parser.add_argument("--relay-code-span", type=int, default=None)
    parser.add_argument("--write-warmup-relays", type=int, default=None)
    parser.add_argument(
        "--bootstrap-context-mode",
        choices=("shared", "self"),
        default="shared",
        help=(
            "shared lets relay 0 read node memory; self requires packets "
            "to move and meet before information can transfer"
        ),
    )
    parser.add_argument("--lanes", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--portable-batches",
        action="store_true",
        help=(
            "use a dedicated CPU RNG so CPU and CUDA runs see identical "
            "training/evaluation examples"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "resume model, optimizer, and global step; --steps is the "
            "additional update count"
        ),
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="load resumed model weights but start fresh optimizer state",
    )
    add_optimization_arguments(parser)
    args = parser.parse_args()
    if min(
        args.steps,
        args.batch_size,
        args.eval_batch_size,
        args.log_every,
    ) <= 0:
        parser.error("training and logging sizes must be positive")
    if (
        args.route_backend != "packed"
        and args.packed_memory_backend != "matrix"
    ):
        parser.error(
            "convolution/fft/fft_opaque memory requires --route-backend packed"
        )
    if (
        args.packed_memory_reader_backend == "triton"
        and args.packed_memory_backend == "matrix"
    ):
        parser.error(
            "the Triton packed reader requires convolution, fft, or "
            "fft_opaque memory"
        )
    if (
        args.packed_memory_writer_backend == "triton"
        and (
            args.route_backend != "packed"
            or args.packed_memory_backend != "fft_opaque"
        )
    ):
        parser.error(
            "the Triton packed writer currently requires --route-backend "
            "packed and --packed-memory-backend fft_opaque"
        )
    if (
        args.packed_memory_writer_backend == "triton"
        and args.packed_memory_feature_chunk > 128
    ):
        parser.error(
            "the Triton packed writer supports feature chunks up to 128"
        )
    if args.compile and args.packed_memory_backend == "fft":
        parser.error(
            "raw FFT exposes complex tensors to Inductor; use fft_opaque "
            "with --compile or raw fft without --compile"
        )
    if args.compile and args.packed_memory_compact_nodes:
        parser.error("--packed-memory-compact-nodes is eager-only")
    if args.compile and args.compact_node_dispatch:
        parser.error("--compact-node-dispatch is eager-only")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvergentGator(
        n_nodes=args.nodes,
        d_model=args.width,
        degree=args.degree,
        router_mode=args.router_mode,
        router_rank=args.router_rank,
        ingress_mode=args.ingress_mode,
        route_backend=args.route_backend,
        packed_memory_backend=args.packed_memory_backend,
        packed_memory_reader_backend=args.packed_memory_reader_backend,
        packed_memory_writer_backend=args.packed_memory_writer_backend,
        packed_memory_feature_chunk=args.packed_memory_feature_chunk,
        packed_memory_compact_nodes=args.packed_memory_compact_nodes,
        adaptive_route_min_k=args.adaptive_route_min_k,
        adaptive_route_mode=args.adaptive_route_mode,
        adaptive_route_entropy_mix=args.adaptive_route_entropy_mix,
        compact_node_dispatch=args.compact_node_dispatch,
        node_compute_mode=args.node_compute_mode,
        node_adapter_rank=args.node_adapter_rank,
        ingress_groups=args.ingress_groups,
        ingress_top_groups=args.ingress_top_groups,
        ingress_learned_groups=args.ingress_learned_groups,
        active_node_top_k=args.active_node_top_k,
        active_node_mass_threshold=args.active_node_mass_threshold,
        relays=args.relays,
        lanes=args.lanes,
        relay_code_span=args.relay_code_span,
        write_warmup_relays=args.write_warmup_relays,
        bootstrap_context_mode=args.bootstrap_context_mode,
        connection_layout=args.connection_layout,
        connection_seed=args.connection_seed,
        connection_rewire=args.connection_rewire,
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
    best_accuracy = -1.0
    if args.resume is not None:
        payload = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(payload.get("model", payload))
        resumed_step = int(payload.get("step", 0))
        best_accuracy = float(payload.get("best_accuracy", -1.0))
        if "batch_generator_state" in payload:
            batch_generator.set_state(
                payload["batch_generator_state"].cpu()
            )
        if not args.reset_optimizer and "training" in payload:
            engine.load_state_dict(
                payload["training"],
                load_scheduler=False,
                learning_rate=optimization.learning_rate,
            )
    checkpoint = args.checkpoint or default_convergent_checkpoint_path(
        seed=args.seed,
        route_backend=args.route_backend,
        packed_memory_backend=args.packed_memory_backend,
        packed_memory_reader_backend=args.packed_memory_reader_backend,
        packed_memory_writer_backend=args.packed_memory_writer_backend,
    )
    best_checkpoint = checkpoint.with_name(
        f"{checkpoint.stem}.best{checkpoint.suffix}"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    final_reach = model.reachable_mask[-1].sum(dim=-1).float()
    print(
        f"device={device} parameters={model.parameter_count} "
        f"cells=0 nodes={model.n_nodes} lanes={model.lanes} "
        f"width={model.d_model} degree={model.degree} "
        f"relays={model.relays} compiled={args.compile} seed={args.seed} "
        f"route_backend={model.route_backend} "
        f"packed_memory_backend={model.packed_memory_backend} "
        f"packed_memory_reader_backend={model.packed_memory_reader_backend} "
        f"packed_memory_writer_backend={model.packed_memory_writer_backend} "
        f"relay_code_span={model.relay_code_span} "
        f"write_warmup_relays={model.write_warmup_relays} "
        f"bootstrap_context_mode={model.bootstrap_context_mode} "
        f"connection_layout={model.connection_layout} "
        f"connection_seed={model.connection_seed} "
        f"connection_rewire={model.connection_rewire:g} "
        f"final_reach_mean={float(final_reach.mean()):.2f} "
        f"final_reach_min={int(final_reach.min())} "
        f"final_reach_max={int(final_reach.max())} "
        f"resumed_step={resumed_step} "
        f"portable_batches={args.portable_batches} "
        f"optimizer={optimization.name} lr={optimization.learning_rate:g} "
        f"optimizer_epsilon={optimization.epsilon:g}",
        flush=True,
    )

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
        )
        loss = F.cross_entropy(
            logits.reshape(-1, 16),
            targets.reshape(-1),
        )
        step_result = engine.step(loss)
        phase_step = step + 1
        completed = resumed_step + phase_step
        if phase_step % args.log_every:
            continue
        now = time.perf_counter()
        speed = (phase_step - interval_step) / (now - interval_started)
        result = evaluate(
            model,
            7,
            device,
            batch_size=args.eval_batch_size,
            sampling_device=sampling_device,
        )
        with torch.no_grad():
            _, trace = model(
                observed[: min(32, args.batch_size)],
                output_slots=7,
                return_trace=True,
            )
            terminal = trace["terminal_weights"]
            terminal_entropy = -(
                terminal.clamp_min(1e-9)
                * terminal.clamp_min(1e-9).log()
            ).sum()
            final_q = trace["locations"][:, -1]
            route_entropy = -(
                final_q.clamp_min(1e-9)
                * final_q.clamp_min(1e-9).log()
            ).sum(-1).mean()
            node_mass = final_q.sum(dim=2)
            lane_pair_overlap = 0.5 * (
                node_mass.square()
                - final_q.square().sum(dim=2)
            ).sum(dim=-1).mean()
        print(
            f"step={completed} steps_per_second={speed:.2f} "
            f"correct={result['correct']:.2f}/7 "
            f"acc={result['accuracy']:.3f} exact={result['exact']:.3f} "
            f"loss={result['loss']:.4f} "
            f"terminal_effective={float(terminal_entropy.exp()):.2f} "
            f"node_effective={float(route_entropy.exp()):.2f} "
            f"lane_overlap={float(lane_pair_overlap):.3f} "
            f"lr={step_result.learning_rate:.6g} "
            f"grad_norm={step_result.gradient_norm:.4f}",
            flush=True,
        )
        interval_started = time.perf_counter()
        interval_step = phase_step
        improved = result["accuracy"] > best_accuracy
        best_accuracy = max(best_accuracy, result["accuracy"])
        checkpoint_payload = {
            "step": completed,
            "best_accuracy": best_accuracy,
            "model_config": model.architecture_config,
            "model": model.state_dict(),
            "training": engine.state_dict(),
            "batch_generator_state": batch_generator.get_state(),
        }
        torch.save(checkpoint_payload, checkpoint)
        if improved:
            torch.save(checkpoint_payload, best_checkpoint)

    result = evaluate(
        model,
        7,
        device,
        batch_size=args.eval_batch_size,
        sampling_device=sampling_device,
    )
    elapsed = time.perf_counter() - started
    print(
        f"final correct={result['correct']:.2f}/7 "
        f"acc={result['accuracy']:.3f} exact={result['exact']:.3f} "
        f"loss={result['loss']:.4f}",
        flush=True,
    )
    print(
        f"runtime_seconds={elapsed:.2f} "
        f"steps_per_second={args.steps / elapsed:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
