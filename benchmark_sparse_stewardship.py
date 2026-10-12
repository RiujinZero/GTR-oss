"""Benchmark Gator's sparse stewardship execution backends.

This benchmark deliberately starts from a synthetic, already-sparse ownership
layout.  It compares:

* ``repack``: the legacy globally padded receiver/local path, and
* ``persistent_bucketed``: bucketed receiver memory and cell-local kernels.

The suffix benchmark models the execution-policy difference as well.  The
legacy path packs and unpacks on every relay; the persistent path packs once,
keeps the ragged representation through every relay, and reads out from it
directly.  Handoffs are deterministically fixed to the self action so the
requested support remains constant.  This isolates sparse execution from
learned route widening and makes backend correctness directly measurable.

Representative GPU run:

    python benchmark_sparse_stewardship.py

Small CPU smoke:

    python benchmark_sparse_stewardship.py --device cpu --batch-size 1 \
        --cells 6 --tokens 6 --supports 2,1 --relays 2 \
        --warmup 0 --iterations 1 --suffix-iterations 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time
from typing import Callable, Literal

import torch
import torch.nn.functional as F

from gator.models import (
    GatorLanguageConfig,
    GatorLanguageModel,
    PackedStewardship,
    adaptive_nucleus_stewardship,
    build_packed_group_plan,
    pack_stewardship,
    packed_readout_hidden,
    prune_packed_stewardship,
    relative_position_code,
    transport_packed_stewardship,
    unpack_stewardship,
)


Layout = Literal["clustered", "balanced"]


@dataclass(frozen=True)
class SyntheticCase:
    """Dense and packed views of one deterministic sparse route layout."""

    ownership: torch.Tensor
    states: torch.Tensor
    nodes: torch.Tensor
    packed: PackedStewardship


@dataclass(frozen=True)
class RelayInputs:
    """Parameter-derived tensors shared by the two backend measurements."""

    write: torch.Tensor
    relative: torch.Tensor
    addresses: torch.Tensor
    address_state: torch.Tensor
    address_out: torch.Tensor
    step_states: torch.Tensor


@dataclass(frozen=True)
class Timing:
    milliseconds: float
    peak_allocation_mib: float | None


def _fixed_code(
    length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = torch.arange(length, device=device, dtype=torch.float32)
    angle = x / max(length - 1, 1) * math.pi
    code = torch.stack(
        [
            torch.sin(angle),
            torch.cos(angle),
            torch.sin(2 * angle),
            torch.cos(2 * angle),
        ],
        -1,
    )
    return code.to(dtype=dtype)


def _parse_int_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers"
        ) from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("every list value must be positive")
    return parsed


def _parse_layouts(value: str) -> tuple[Layout, ...]:
    layouts = tuple(item.strip() for item in value.split(","))
    valid = {"clustered", "balanced"}
    if not layouts or any(item not in valid for item in layouts):
        raise argparse.ArgumentTypeError(
            "layouts must contain clustered and/or balanced"
        )
    return layouts  # type: ignore[return-value]


def _device_from_argument(value: str) -> torch.device:
    if value == "auto":
        return (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _dtype_from_argument(value: str) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {value}")


def _build_models(
    *,
    cells: int,
    nodes: int,
    hidden: int,
    relays: int,
    neighbors: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[GatorLanguageModel, GatorLanguageModel]:
    common = dict(
        vocab_size=256,
        d_model=hidden,
        embedding_dim=hidden,
        n_cells=cells,
        n_nodes=nodes,
        relays=relays,
        neighbors=neighbors,
        lexical_mode="communal",
        cell_input_mode="shared_projection",
        cell_output_mode="shared_pool",
        cell_transport_mode="stewardship",
        stewardship_execution="packed_auto",
        stewardship_keep_mass=1.0,
        stewardship_handoff_keep_mass=1.0,
        stewardship_min_active=1,
        stewardship_max_active=cells,
        stewardship_exploration=0.0,
        stewardship_scouts=0,
        node_topology="symmetric",
        dense_kernel_padding=1,
    )
    legacy = GatorLanguageModel(
        GatorLanguageConfig(
            **common,
            packed_stewardship_backend="repack",
        )
    ).to(device=device, dtype=dtype)
    persistent = GatorLanguageModel(
        GatorLanguageConfig(
            **common,
            packed_stewardship_backend="persistent_bucketed",
        )
    ).to(device=device, dtype=dtype)
    persistent.load_state_dict(legacy.state_dict())
    return legacy.eval(), persistent.eval()


def _build_case(
    *,
    batch: int,
    cells: int,
    tokens: int,
    hidden: int,
    nodes: int,
    support: int,
    layout: Layout,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> SyntheticCase:
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)

    ownership = torch.zeros(
        batch, cells, tokens, device=device, dtype=dtype
    )
    batch_index = torch.arange(device=device, end=batch)[:, None, None]
    token_index = torch.arange(device=device, end=tokens)[None, :, None]
    rank = torch.arange(device=device, end=support)[None, None, :]
    if layout == "clustered":
        cell_index = rank.expand(batch, tokens, -1)
    else:
        # Rotate the live window by token and batch.  Each token still has
        # exactly ``support`` stewards, but work is spread across all cells.
        cell_index = (
            token_index + batch_index * tokens + rank
        ).remainder(cells)
    ownership[
        batch_index.expand(-1, tokens, support),
        cell_index,
        token_index.expand(batch, -1, support),
    ] = 1.0 / support

    states = torch.randn(
        batch,
        cells,
        tokens,
        hidden,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    node_logits = torch.randn(
        batch,
        cells,
        tokens,
        nodes,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    node_locations = F.softmax(node_logits.float(), -1).to(dtype)
    live = ownership.gt(0)
    states = states * live.unsqueeze(-1).to(dtype)
    node_locations = node_locations * live.unsqueeze(-1).to(dtype)
    packed = pack_stewardship(ownership, states, node_locations)
    return SyntheticCase(
        ownership=ownership,
        states=states,
        nodes=node_locations,
        packed=packed,
    )


def _relay_inputs(
    model: GatorLanguageModel,
    *,
    batch: int,
    tokens: int,
    relays: int,
    device: torch.device,
    dtype: torch.dtype,
) -> RelayInputs:
    cells = model.config.n_cells
    hidden = model.config.d_model
    addresses = _fixed_code(tokens, device, dtype)
    relay_codes = _fixed_code(relays, device, dtype)
    address_state = F.linear(
        addresses,
        model.address_to_state.reshape(cells * hidden, 4),
    ).view(tokens, cells, hidden).permute(1, 0, 2)
    address_out = F.linear(
        addresses,
        model.address_to_readout.reshape(cells * hidden, 4),
    ).view(tokens, cells, hidden).permute(1, 0, 2)
    step_states = F.linear(
        relay_codes,
        model.step_projection.reshape(cells * hidden, 4),
    ).view(relays, cells, hidden)
    return RelayInputs(
        write=torch.ones(batch, tokens, device=device, dtype=dtype),
        relative=relative_position_code(
            tokens, device, dtype=dtype
        ),
        addresses=addresses,
        address_state=address_state,
        address_out=address_out,
        step_states=step_states,
    )


def _local(
    model: GatorLanguageModel,
    packed: PackedStewardship,
    inputs: RelayInputs,
    *,
    relay: int = 0,
    temperature: float,
) -> tuple[PackedStewardship, torch.Tensor, torch.Tensor]:
    return model._packed_stewardship_local(
        packed,
        write=inputs.write,
        relative=inputs.relative,
        addresses=inputs.addresses,
        address_state=inputs.address_state,
        step_state=inputs.step_states[relay],
        temperature=temperature,
        stewardship_temperature=temperature,
    )


def _stay_handoffs(
    packed: PackedStewardship,
    actions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    handoffs = packed.mass.new_zeros(packed.size, actions)
    handoffs[:, 0] = 1.0
    return handoffs, handoffs.gt(0)


def _legacy_suffix(
    model: GatorLanguageModel,
    case: SyntheticCase,
    inputs: RelayInputs,
    *,
    relays: int,
    output_tokens: int,
    temperature: float,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    ownership = case.ownership
    states = case.states
    nodes = case.nodes
    batch, cells, tokens = ownership.shape
    actions = model.handoff_destinations.shape[-1]

    for relay in range(relays):
        # This is the legacy execution policy: every relay rebuilds the packed
        # rows from a globally shaped state and expands the result again.
        packed = pack_stewardship(ownership, states, nodes)
        processed, _, _ = _local(
            model,
            packed,
            inputs,
            relay=relay,
            temperature=temperature,
        )
        stay, stay_support = _stay_handoffs(processed, actions)
        transported = transport_packed_stewardship(
            processed,
            stay,
            model.handoff_destinations,
            batch_size=batch,
            cells=cells,
            tokens=tokens,
            support=stay_support,
        )
        ownership, states, nodes = unpack_stewardship(
            transported,
            batch_size=batch,
            cells=cells,
            tokens=tokens,
        )
        ownership, _ = adaptive_nucleus_stewardship(
            ownership,
            keep_mass=1.0,
            min_active=1,
            max_active=cells,
            scouts=0,
            exploration_mass=0.0,
            straight_through=False,
        )

    context_tokens = tokens - output_tokens
    target_hidden = (
        states[:, :, context_tokens:]
        + inputs.address_out[None, :, context_tokens:]
    )
    readout = torch.einsum(
        "bct,bctd->btd",
        ownership[:, :, context_tokens:],
        target_hidden,
    )
    return readout, (ownership, states, nodes)


def _persistent_suffix(
    model: GatorLanguageModel,
    case: SyntheticCase,
    inputs: RelayInputs,
    *,
    relays: int,
    output_tokens: int,
    temperature: float,
) -> tuple[torch.Tensor, PackedStewardship]:
    batch, cells, tokens = case.ownership.shape
    actions = model.handoff_destinations.shape[-1]
    # Persistent execution pays this conversion once at the dense/sparse
    # boundary, then carries the packed record through the complete suffix.
    packed = pack_stewardship(case.ownership, case.states, case.nodes)

    for relay in range(relays):
        processed, _, _ = _local(
            model,
            packed,
            inputs,
            relay=relay,
            temperature=temperature,
        )
        stay, stay_support = _stay_handoffs(processed, actions)
        transported = transport_packed_stewardship(
            processed,
            stay,
            model.handoff_destinations,
            batch_size=batch,
            cells=cells,
            tokens=tokens,
            support=stay_support,
        )
        packed, _ = prune_packed_stewardship(
            transported,
            keep_mass=1.0,
            batch_size=batch,
            cells=cells,
            tokens=tokens,
            min_active=1,
            max_active=cells,
            scouts=0,
            exploration_mass=0.0,
            straight_through=False,
        )

    readout = packed_readout_hidden(
        packed,
        inputs.address_out,
        batch_size=batch,
        context_tokens=tokens - output_tokens,
        output_tokens=output_tokens,
    )
    return readout, packed


def _timed(
    function: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> Timing:
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")
    with torch.inference_mode():
        for _ in range(warmup):
            function()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                function()
            end.record()
            torch.cuda.synchronize(device)
            milliseconds = start.elapsed_time(end) / iterations
            peak = torch.cuda.max_memory_allocated(device)
            peak_mib = max(peak - baseline, 0) / (1024**2)
        else:
            start_time = time.perf_counter()
            for _ in range(iterations):
                function()
            milliseconds = (
                time.perf_counter() - start_time
            ) * 1000.0 / iterations
            # PyTorch CPU tensor storage is not tracked by tracemalloc.
            peak_mib = None
    return Timing(milliseconds, peak_mib)


def _maximum_error(*pairs: tuple[torch.Tensor, torch.Tensor]) -> float:
    return max(
        float((actual.float() - expected.float()).abs().max().item())
        for actual, expected in pairs
    )


def _histogram_text(histogram: dict[int, int]) -> str:
    return ",".join(
        f"{capacity}:{groups}"
        for capacity, groups in sorted(histogram.items())
    )


def _memory_text(value: float | None) -> str:
    return "n/a(cpu)" if value is None else f"{value:.2f}"


def _benchmark_case(
    *,
    legacy: GatorLanguageModel,
    persistent: GatorLanguageModel,
    inputs: RelayInputs,
    case: SyntheticCase,
    layout: Layout,
    support: int,
    relays: int,
    output_tokens: int,
    temperature: float,
    device: torch.device,
    warmup: int,
    iterations: int,
    suffix_iterations: int,
    tolerance: float,
) -> None:
    batch, cells, tokens = case.ownership.shape
    packed = case.packed
    receiver_key = packed.batch * cells + packed.cell
    receiver_plan = build_packed_group_plan(
        receiver_key,
        maximum_group_size=tokens,
    )
    cell_plan = build_packed_group_plan(
        packed.cell,
        maximum_group_size=batch * tokens,
    )
    receiver_stats = receiver_plan.statistics()
    cell_stats = cell_plan.statistics()
    active_groups = int(receiver_key.unique().numel())
    active_fraction = packed.size / (batch * cells * tokens)
    theoretical_branch_reduction = cells / support

    legacy_local_timing = _timed(
        lambda: _local(
            legacy,
            packed,
            inputs,
            temperature=temperature,
        ),
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    persistent_local_timing = _timed(
        lambda: _local(
            persistent,
            packed,
            inputs,
            temperature=temperature,
        ),
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    legacy_suffix_timing = _timed(
        lambda: _legacy_suffix(
            legacy,
            case,
            inputs,
            relays=relays,
            output_tokens=output_tokens,
            temperature=temperature,
        ),
        device=device,
        warmup=warmup,
        iterations=suffix_iterations,
    )
    persistent_suffix_timing = _timed(
        lambda: _persistent_suffix(
            persistent,
            case,
            inputs,
            relays=relays,
            output_tokens=output_tokens,
            temperature=temperature,
        ),
        device=device,
        warmup=warmup,
        iterations=suffix_iterations,
    )

    with torch.inference_mode():
        expected_local, expected_handoffs, expected_entropy = _local(
            legacy,
            packed,
            inputs,
            temperature=temperature,
        )
        actual_local, actual_handoffs, actual_entropy = _local(
            persistent,
            packed,
            inputs,
            temperature=temperature,
        )
        local_error = _maximum_error(
            (actual_local.state, expected_local.state),
            (actual_local.node, expected_local.node),
            (actual_handoffs, expected_handoffs),
            (actual_entropy, expected_entropy),
        )

        expected_readout, expected_dense = _legacy_suffix(
            legacy,
            case,
            inputs,
            relays=relays,
            output_tokens=output_tokens,
            temperature=temperature,
        )
        actual_readout, actual_packed = _persistent_suffix(
            persistent,
            case,
            inputs,
            relays=relays,
            output_tokens=output_tokens,
            temperature=temperature,
        )
        actual_dense = unpack_stewardship(
            actual_packed,
            batch_size=batch,
            cells=cells,
            tokens=tokens,
        )
        suffix_error = _maximum_error(
            (actual_readout, expected_readout),
            *tuple(zip(actual_dense, expected_dense, strict=True)),
        )

    print(
        f"layout={layout} support={support} "
        f"live_rows={packed.size} active_fraction={active_fraction:.6f} "
        f"theoretical_dense_branch_reduction="
        f"{theoretical_branch_reduction:.3f}x "
        f"active_batch_cell_groups={active_groups} "
        f"receiver_buckets={_histogram_text(receiver_stats['bucket_histogram'])} "
        f"receiver_padding_efficiency={receiver_stats['padding_efficiency']:.4f} "
        f"cell_buckets={_histogram_text(cell_stats['bucket_histogram'])} "
        f"cell_padding_efficiency={cell_stats['padding_efficiency']:.4f}"
    )
    print(
        "  local "
        f"repack_padded_ms={legacy_local_timing.milliseconds:.3f} "
        f"persistent_bucketed_ms={persistent_local_timing.milliseconds:.3f} "
        f"speedup={legacy_local_timing.milliseconds / persistent_local_timing.milliseconds:.3f}x "
        f"repack_token_relays_per_second="
        f"{batch * tokens * 1000.0 / legacy_local_timing.milliseconds:.1f} "
        f"bucketed_token_relays_per_second="
        f"{batch * tokens * 1000.0 / persistent_local_timing.milliseconds:.1f} "
        f"repack_peak_alloc_mib={_memory_text(legacy_local_timing.peak_allocation_mib)} "
        f"bucketed_peak_alloc_mib={_memory_text(persistent_local_timing.peak_allocation_mib)} "
        f"correctness_max_error={local_error:.7g}"
    )
    print(
        f"  suffix relays={relays} "
        f"repack_padded_ms={legacy_suffix_timing.milliseconds:.3f} "
        f"persistent_bucketed_ms={persistent_suffix_timing.milliseconds:.3f} "
        f"speedup={legacy_suffix_timing.milliseconds / persistent_suffix_timing.milliseconds:.3f}x "
        f"repack_token_relays_per_second="
        f"{batch * tokens * relays * 1000.0 / legacy_suffix_timing.milliseconds:.1f} "
        f"bucketed_token_relays_per_second="
        f"{batch * tokens * relays * 1000.0 / persistent_suffix_timing.milliseconds:.1f} "
        f"repack_peak_alloc_mib={_memory_text(legacy_suffix_timing.peak_allocation_mib)} "
        f"bucketed_peak_alloc_mib={_memory_text(persistent_suffix_timing.peak_allocation_mib)} "
        f"correctness_max_error={suffix_error:.7g}"
    )
    if max(local_error, suffix_error) > tolerance:
        raise RuntimeError(
            f"backend error {max(local_error, suffix_error):.7g} "
            f"exceeds tolerance {tolerance:.7g}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy repacked/global-padded and persistent bucketed "
            "sparse stewardship execution."
        )
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cells", type=int, default=62)
    parser.add_argument("--tokens", type=int, default=104)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--relays", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument(
        "--supports",
        type=_parse_int_list,
        default=(62, 32, 16, 8, 4, 2, 1),
    )
    parser.add_argument(
        "--layouts",
        type=_parse_layouts,
        default=("clustered", "balanced"),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--suffix-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--correctness-tolerance",
        type=float,
        default=None,
        help="Defaults to 1e-4 for float32 and 0.1 for bfloat16.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = _device_from_argument(args.device)
    dtype = _dtype_from_argument(args.dtype)
    if args.batch_size <= 0 or args.tokens <= 0 or args.hidden <= 0:
        raise ValueError("batch size, tokens, and hidden width must be positive")
    if args.relays <= 0:
        raise ValueError("relays must be positive")
    if not 0 < args.output_tokens <= args.tokens:
        raise ValueError("output_tokens must be in [1, tokens]")
    if any(support > args.cells for support in args.supports):
        raise ValueError("every support value must be <= cells")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    tolerance = (
        args.correctness_tolerance
        if args.correctness_tolerance is not None
        else (0.1 if dtype == torch.bfloat16 else 1e-4)
    )
    if tolerance < 0:
        raise ValueError("correctness tolerance must be nonnegative")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    legacy, persistent = _build_models(
        cells=args.cells,
        nodes=args.nodes,
        hidden=args.hidden,
        relays=args.relays,
        neighbors=args.neighbors,
        device=device,
        dtype=dtype,
    )
    inputs = _relay_inputs(
        legacy,
        batch=args.batch_size,
        tokens=args.tokens,
        relays=args.relays,
        device=device,
        dtype=dtype,
    )

    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else str(device)
    )
    print(
        f"device={device_name} dtype={args.dtype} batch={args.batch_size} "
        f"cells={args.cells} tokens={args.tokens} hidden={args.hidden} "
        f"nodes={args.nodes} relays={args.relays} "
        f"actions={args.neighbors + 1} "
        f"supports={','.join(map(str, args.supports))} "
        f"layouts={','.join(args.layouts)}"
    )
    print(
        "scope=eager_sparse_forward_only compiled_dense_not_measured=true "
        "handoff=deterministic_stay support_is_constant=true"
    )

    for layout in args.layouts:
        for support in args.supports:
            case = _build_case(
                batch=args.batch_size,
                cells=args.cells,
                tokens=args.tokens,
                hidden=args.hidden,
                nodes=args.nodes,
                support=support,
                layout=layout,
                device=device,
                dtype=dtype,
                seed=args.seed + support,
            )
            _benchmark_case(
                legacy=legacy,
                persistent=persistent,
                inputs=inputs,
                case=case,
                layout=layout,
                support=support,
                relays=args.relays,
                output_tokens=args.output_tokens,
                temperature=args.temperature,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
                suffix_iterations=args.suffix_iterations,
                tolerance=tolerance,
            )


if __name__ == "__main__":
    main()
