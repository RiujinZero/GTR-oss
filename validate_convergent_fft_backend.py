"""Validate the complete Convergent model across the fused FFT backends.

This is the full-model correctness companion to
``validate_fused_fft_backend.py``.  It compares one seeded Convergent model
and one frozen synthetic language batch through:

* an eager packed raw-FFT receiver with the Torch reader; and
* a compiled packed ``fft_opaque`` receiver with Triton writer and reader.

The two executions start from an identical state dict.  The gate compares
logits, cross-entropy loss, every named parameter gradient, and every named
parameter delta after one AdamW step.  The models execute sequentially and
the oracle snapshot is moved to CPU before the compiled candidate is built,
so assistant-scale checks do not keep two complete CUDA training states
resident at once.

Small Blackwell smoke::

    python -u validate_convergent_fft_backend.py --device cuda:0 --compile

Historical 140M-shape check::

    python -u validate_convergent_fft_backend.py \
      --device cuda:0 --compile --batch-size 6 \
      --context-tokens 512 --output-slots 128 --vocab-size 32768 \
      --embedding-dim 768 --d-model 768 --nodes 154 --lanes 10 \
      --active-node-top-k 32 --degree 3 --relays 2 --router-rank 96 \
      --packed-memory-feature-chunk 32 --causal

Run causal and noncausal invocations as separate processes for a release
gate.  This script proves numerical equivalence; it is deliberately not a
throughput or peak-memory benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
import importlib.metadata
import time
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from gator.models.triton_packed_memory import (
    triton_packed_reader_available,
)
from gator.models.triton_packed_writer import (
    triton_packed_writer_available,
)
from gator_convergent import (
    ConvergentLanguageConfig,
    ConvergentLanguageModel,
)


TensorOrNone = torch.Tensor | None
TensorMap = dict[str, TensorOrNone]
ForwardModel = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class ModelShape:
    batch_size: int
    context_tokens: int
    output_slots: int
    vocab_size: int
    embedding_dim: int
    d_model: int
    nodes: int
    lanes: int
    routes: int
    degree: int
    relays: int
    router_rank: int
    feature_chunk: int


@dataclass(frozen=True)
class SyntheticLanguageBatch:
    context_ids: torch.Tensor
    context_mask: torch.Tensor
    targets: torch.Tensor
    output_token_ids: torch.Tensor
    output_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "SyntheticLanguageBatch":
        return replace(
            self,
            context_ids=self.context_ids.to(device),
            context_mask=self.context_mask.to(device),
            targets=self.targets.to(device),
            output_token_ids=self.output_token_ids.to(device),
            output_mask=self.output_mask.to(device),
        )


@dataclass(frozen=True)
class TrainingSnapshot:
    logits: torch.Tensor
    loss: torch.Tensor
    gradients: TensorMap
    parameter_deltas: TensorMap


@dataclass(frozen=True)
class DriftSummary:
    tensors_checked: int
    tensors_both_none: int
    maximum_absolute: float
    maximum_relative: float
    worst_absolute_tensor: str
    worst_relative_tensor: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-tokens", type=int, default=7)
    parser.add_argument("--output-slots", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=9)
    parser.add_argument(
        "--d-model",
        "--width",
        dest="d_model",
        type=int,
        default=8,
    )
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument(
        "--active-node-top-k",
        "--route-width",
        "--routes",
        dest="active_node_top_k",
        type=int,
        default=3,
    )
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--relays", type=int, default=2)
    parser.add_argument("--router-rank", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument(
        "--packed-memory-feature-chunk",
        "--feature-chunk",
        dest="packed_memory_feature_chunk",
        type=int,
        default=4,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--delta-atol", type=float, default=5e-6)
    parser.add_argument("--delta-rtol", type=float, default=5e-4)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="required; the candidate is executed through torch.compile",
    )
    parser.add_argument(
        "--packed-memory-writer-backend",
        choices=("triton",),
        default="triton",
        help="fixed candidate backend; accepted for release-runner parity",
    )
    parser.add_argument(
        "--fullgraph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "require one unbroken full-model Dynamo graph; the default "
            "matches the production trainer and permits unrelated graph breaks"
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "batch_size",
        "context_tokens",
        "output_slots",
        "nodes",
        "d_model",
        "embedding_dim",
        "lanes",
        "active_node_top_k",
        "degree",
        "relays",
        "router_rank",
        "vocab_size",
        "packed_memory_feature_chunk",
    )
    invalid = [name for name in positive if getattr(args, name) <= 0]
    if invalid:
        raise ValueError(
            "these arguments must be positive: " + ", ".join(invalid)
        )
    if args.vocab_size <= 4:
        raise ValueError("vocab_size must leave room above reserved token IDs")
    if args.lanes > args.nodes:
        raise ValueError("lanes cannot exceed nodes")
    if args.active_node_top_k > args.nodes:
        raise ValueError("active_node_top_k cannot exceed nodes")
    if args.degree > args.nodes:
        raise ValueError("degree cannot exceed nodes")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    for name in ("atol", "rtol", "delta_atol", "delta_rtol"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if not str(args.device).startswith("cuda"):
        raise ValueError("this full-model release gate requires CUDA")
    if not args.compile:
        raise ValueError("this full-model release gate requires --compile")


def shape_from_args(args: argparse.Namespace) -> ModelShape:
    return ModelShape(
        batch_size=args.batch_size,
        context_tokens=args.context_tokens,
        output_slots=args.output_slots,
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        d_model=args.d_model,
        nodes=args.nodes,
        lanes=args.lanes,
        routes=args.active_node_top_k,
        degree=args.degree,
        relays=args.relays,
        router_rank=args.router_rank,
        feature_chunk=args.packed_memory_feature_chunk,
    )


def build_synthetic_batch(
    shape: ModelShape,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> SyntheticLanguageBatch:
    """Build one deterministic batch with shifted teacher-prefix inputs."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    context_ids = torch.randint(
        4,
        shape.vocab_size,
        (shape.batch_size, shape.context_tokens),
        generator=generator,
        dtype=torch.long,
    )
    context_mask = torch.ones_like(context_ids, dtype=torch.bool)
    # Exercise padding semantics without producing an empty context row.
    if shape.context_tokens > 1:
        context_mask[::2, -1] = False
        context_ids = context_ids.masked_fill(~context_mask, 0)
    targets = torch.randint(
        4,
        shape.vocab_size,
        (shape.batch_size, shape.output_slots),
        generator=generator,
        dtype=torch.long,
    )
    output_token_ids = torch.full_like(targets, 1)
    if shape.output_slots > 1:
        output_token_ids[:, 1:] = targets[:, :-1]
    output_mask = torch.ones_like(targets, dtype=torch.bool)
    return SyntheticLanguageBatch(
        context_ids=context_ids,
        context_mask=context_mask,
        targets=targets,
        output_token_ids=output_token_ids,
        output_mask=output_mask,
    ).to(device)


def build_model_config(
    shape: ModelShape,
    *,
    seed: int,
    candidate: bool,
) -> ConvergentLanguageConfig:
    return ConvergentLanguageConfig(
        vocab_size=shape.vocab_size,
        d_model=shape.d_model,
        embedding_dim=shape.embedding_dim,
        n_nodes=shape.nodes,
        degree=shape.degree,
        router_mode="shared_low_rank",
        router_rank=shape.router_rank,
        ingress_mode="learned",
        route_backend="packed",
        packed_memory_backend="fft_opaque" if candidate else "fft",
        packed_memory_reader_backend="triton" if candidate else "torch",
        packed_memory_writer_backend="triton" if candidate else "torch",
        packed_memory_feature_chunk=shape.feature_chunk,
        active_node_top_k=shape.routes,
        relays=shape.relays,
        lanes=shape.lanes,
        mask_token_id=3,
        blank_token_id=1,
        connection_layout="small_world",
        connection_seed=seed,
        connection_rewire=0.5,
        terminal_vote_mode="slot",
    )


def state_dict_to_cpu(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def require_finite(label: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"{label} contains a non-finite value")


def assert_state_dict_exact(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> None:
    if actual.keys() != expected.keys():
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise AssertionError(
            f"state_dict key mismatch missing={missing} unexpected={unexpected}"
        )
    for name, expected_value in expected.items():
        actual_value = actual[name].detach().cpu()
        if actual_value.shape != expected_value.shape:
            raise AssertionError(
                f"state_dict shape mismatch for {name}: "
                f"{tuple(actual_value.shape)} != {tuple(expected_value.shape)}"
            )
        if actual_value.dtype != expected_value.dtype:
            raise AssertionError(
                f"state_dict dtype mismatch for {name}: "
                f"{actual_value.dtype} != {expected_value.dtype}"
            )
        if not torch.equal(actual_value, expected_value):
            raise AssertionError(f"state_dict value mismatch for {name}")


def capture_training_snapshot(
    model: torch.nn.Module,
    forward_model: ForwardModel,
    batch: SyntheticLanguageBatch,
    initial_state: Mapping[str, torch.Tensor],
    *,
    output_slots: int,
    vocab_size: int,
    causal: bool,
    temperature: float | torch.Tensor,
    learning_rate: float,
    weight_decay: float,
) -> TrainingSnapshot:
    """Run forward/backward/AdamW once and retain a CPU oracle snapshot."""

    named_parameters = dict(model.named_parameters())
    if not named_parameters:
        raise RuntimeError("model has no named parameters")
    missing_initial = sorted(set(named_parameters) - set(initial_state))
    if missing_initial:
        raise AssertionError(
            f"initial_state is missing parameters: {missing_initial}"
        )
    for name, parameter in named_parameters.items():
        require_finite(f"parameter_before[{name}]", parameter)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )
    optimizer.zero_grad(set_to_none=True)
    logits = forward_model(
        batch.context_ids,
        batch.context_mask,
        output_slots=output_slots,
        output_token_ids=batch.output_token_ids,
        output_mask=batch.output_mask,
        causal=causal,
        temperature=temperature,
    )
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model must return one logits tensor for the gate")
    expected_shape = (
        batch.targets.shape[0],
        output_slots,
        vocab_size,
    )
    if logits.shape != expected_shape:
        raise AssertionError(
            f"unexpected logits shape {tuple(logits.shape)}; "
            f"expected {expected_shape}"
        )
    require_finite("logits", logits)
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        batch.targets.reshape(-1),
    )
    require_finite("loss", loss)
    loss.backward()

    gradients: TensorMap = {}
    for name, parameter in named_parameters.items():
        gradient = parameter.grad
        if gradient is None:
            gradients[name] = None
            continue
        require_finite(f"gradient[{name}]", gradient)
        gradients[name] = gradient.detach().cpu().clone()

    optimizer.step()
    parameter_deltas: TensorMap = {}
    for name, parameter in named_parameters.items():
        require_finite(f"parameter_after[{name}]", parameter)
        after = parameter.detach().cpu()
        before = initial_state[name]
        delta = after - before
        require_finite(f"parameter_delta[{name}]", delta)
        parameter_deltas[name] = delta.clone()

    snapshot = TrainingSnapshot(
        logits=logits.detach().cpu().clone(),
        loss=loss.detach().cpu().clone(),
        gradients=gradients,
        parameter_deltas=parameter_deltas,
    )
    optimizer.zero_grad(set_to_none=True)
    return snapshot


def _tensor_drift(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    if actual.numel() == 0:
        return 0.0, 0.0
    difference = (actual.float() - expected.float()).abs()
    scale = torch.maximum(
        actual.float().abs(), expected.float().abs()
    ).clamp_min(1e-12)
    return float(difference.max()), float((difference / scale).max())


def compare_tensor_maps(
    label: str,
    actual: Mapping[str, TensorOrNone],
    expected: Mapping[str, TensorOrNone],
    *,
    atol: float,
    rtol: float,
) -> DriftSummary:
    if actual.keys() != expected.keys():
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise AssertionError(
            f"{label} key mismatch missing={missing} unexpected={unexpected}"
        )
    checked = 0
    both_none = 0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    worst_absolute = ""
    worst_relative = ""
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if actual_value is None or expected_value is None:
            if actual_value is None and expected_value is None:
                both_none += 1
                continue
            raise AssertionError(
                f"{label}[{name}] gradient/update presence mismatch"
            )
        require_finite(f"{label}.actual[{name}]", actual_value)
        require_finite(f"{label}.expected[{name}]", expected_value)
        torch.testing.assert_close(
            actual_value,
            expected_value,
            atol=atol,
            rtol=rtol,
            msg=lambda message, tensor_name=name: (
                f"{label}[{tensor_name}]: {message}"
            ),
        )
        checked += 1
        absolute, relative = _tensor_drift(actual_value, expected_value)
        if absolute >= maximum_absolute:
            maximum_absolute = absolute
            worst_absolute = name
        if relative >= maximum_relative:
            maximum_relative = relative
            worst_relative = name
    return DriftSummary(
        tensors_checked=checked,
        tensors_both_none=both_none,
        maximum_absolute=maximum_absolute,
        maximum_relative=maximum_relative,
        worst_absolute_tensor=worst_absolute,
        worst_relative_tensor=worst_relative,
    )


def compare_snapshots(
    actual: TrainingSnapshot,
    expected: TrainingSnapshot,
    *,
    atol: float,
    rtol: float,
    delta_atol: float,
    delta_rtol: float,
) -> dict[str, DriftSummary]:
    scalar_actual = {
        "logits": actual.logits,
        "loss": actual.loss,
    }
    scalar_expected = {
        "logits": expected.logits,
        "loss": expected.loss,
    }
    return {
        "outputs": compare_tensor_maps(
            "outputs",
            scalar_actual,
            scalar_expected,
            atol=atol,
            rtol=rtol,
        ),
        "gradients": compare_tensor_maps(
            "gradients",
            actual.gradients,
            expected.gradients,
            atol=atol,
            rtol=rtol,
        ),
        "parameter_deltas": compare_tensor_maps(
            "parameter_deltas",
            actual.parameter_deltas,
            expected.parameter_deltas,
            atol=delta_atol,
            rtol=delta_rtol,
        ),
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def require_cuda_triton(device_value: str) -> torch.device:
    if not hasattr(torch, "compile"):
        raise RuntimeError("full-model FFT validation requires torch.compile")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "full-model FFT validation requires CUDA; "
            "torch.cuda.is_available() is false"
        )
    if not triton_packed_reader_available():
        raise RuntimeError(
            "full-model FFT validation requires the Triton packed reader plus "
            "torch.library.triton_op/wrap_triton support"
        )
    if not triton_packed_writer_available():
        raise RuntimeError(
            "full-model FFT validation requires the Triton packed writer plus "
            "torch.library.triton_op/wrap_triton support"
        )
    device = torch.device(device_value)
    if device.type != "cuda":
        raise RuntimeError("full-model FFT validation requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    return device


def _release_cuda_state(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def _reset_compile_counters() -> Mapping[str, object] | None:
    """Reset Dynamo evidence counters when this PyTorch build exposes them."""

    if not hasattr(torch, "_dynamo"):
        return None
    torch._dynamo.reset()
    utils = getattr(torch._dynamo, "utils", None)
    counters = getattr(utils, "counters", None)
    if counters is None:
        return None
    counters.clear()
    return counters


def _compile_evidence(counters: Mapping[str, object] | None) -> tuple[int, int]:
    if counters is None:
        return -1, -1
    stats = counters.get("stats", {})
    if not isinstance(stats, Mapping):
        raise RuntimeError("unexpected torch.compile stats counter format")
    unique_graphs = int(stats.get("unique_graphs", 0))
    graph_breaks = counters.get("graph_break", {})
    if not isinstance(graph_breaks, Mapping):
        raise RuntimeError("unexpected torch.compile graph-break counter format")
    graph_break_count = int(sum(graph_breaks.values()))
    if unique_graphs <= 0:
        raise RuntimeError(
            "torch.compile produced no recorded Dynamo graph; refusing to "
            "claim compiled full-model equivalence"
        )
    return unique_graphs, graph_break_count


def run_validation(args: argparse.Namespace) -> None:
    device = require_cuda_triton(args.device)
    shape = shape_from_args(args)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    print(
        f"runtime torch={torch.__version__} cuda={torch.version.cuda} "
        f"triton={package_version('triton')} device={device} "
        f"gpu={torch.cuda.get_device_name(device)!r} "
        f"capability={torch.cuda.get_device_capability(device)} "
        f"total_vram_bytes={torch.cuda.get_device_properties(device).total_memory}",
        flush=True,
    )
    print(
        f"shape batch={shape.batch_size} context={shape.context_tokens} "
        f"output={shape.output_slots} vocab={shape.vocab_size} "
        f"embedding={shape.embedding_dim} width={shape.d_model} "
        f"nodes={shape.nodes} lanes={shape.lanes} routes={shape.routes} "
        f"degree={shape.degree} relays={shape.relays} "
        f"router_rank={shape.router_rank} feature_chunk={shape.feature_chunk} "
        f"causal={args.causal} seed={args.seed} teacher_prefix=True "
        f"reader_backend=triton writer_backend=triton "
        f"compile=True fullgraph={args.fullgraph}",
        flush=True,
    )

    cpu_batch = build_synthetic_batch(shape, seed=args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    reference = ConvergentLanguageModel(
        build_model_config(shape, seed=args.seed, candidate=False)
    )
    initial_state = state_dict_to_cpu(reference)
    parameter_count = sum(parameter.numel() for parameter in reference.parameters())
    reference = reference.to(device=device, dtype=torch.float32)
    assert_state_dict_exact(reference.state_dict(), initial_state)
    reference_batch = cpu_batch.to(device)
    temperature = torch.tensor(
        args.temperature,
        device=device,
        dtype=torch.float32,
    )
    reference_started = time.perf_counter()
    reference_snapshot = capture_training_snapshot(
        reference,
        reference,
        reference_batch,
        initial_state,
        output_slots=shape.output_slots,
        vocab_size=shape.vocab_size,
        causal=args.causal,
        temperature=temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    torch.cuda.synchronize(device)
    print(
        f"backend=eager_fft_torch parameters={parameter_count} "
        f"forward_backward_adamw_seconds="
        f"{time.perf_counter() - reference_started:.6f} status=complete",
        flush=True,
    )
    del reference_batch
    del reference
    del temperature
    _release_cuda_state(device)

    candidate = ConvergentLanguageModel(
        build_model_config(shape, seed=args.seed, candidate=True)
    )
    candidate.load_state_dict(initial_state, strict=True)
    assert_state_dict_exact(candidate.state_dict(), initial_state)
    candidate_count = sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    if candidate_count != parameter_count:
        raise AssertionError(
            f"parameter count mismatch {candidate_count} != {parameter_count}"
        )
    candidate = candidate.to(device=device, dtype=torch.float32)
    candidate_batch = cpu_batch.to(device)
    temperature = torch.tensor(
        args.temperature,
        device=device,
        dtype=torch.float32,
    )
    compile_counters = _reset_compile_counters()
    wrapper_started = time.perf_counter()
    compiled_candidate = torch.compile(
        candidate,
        fullgraph=args.fullgraph,
    )
    wrapper_seconds = time.perf_counter() - wrapper_started
    candidate_started = time.perf_counter()
    candidate_snapshot = capture_training_snapshot(
        candidate,
        compiled_candidate,
        candidate_batch,
        initial_state,
        output_slots=shape.output_slots,
        vocab_size=shape.vocab_size,
        causal=args.causal,
        temperature=temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    torch.cuda.synchronize(device)
    unique_graphs, graph_breaks = _compile_evidence(compile_counters)
    print(
        f"backend=compiled_fft_opaque_triton_reader_triton_writer "
        f"parameters={candidate_count} "
        f"compile_wrapper_seconds={wrapper_seconds:.6f} "
        f"first_forward_backward_adamw_seconds="
        f"{time.perf_counter() - candidate_started:.6f} "
        f"unique_graphs={unique_graphs} graph_breaks={graph_breaks} "
        f"status=complete",
        flush=True,
    )

    summaries = compare_snapshots(
        candidate_snapshot,
        reference_snapshot,
        atol=args.atol,
        rtol=args.rtol,
        delta_atol=args.delta_atol,
        delta_rtol=args.delta_rtol,
    )
    for label, summary in summaries.items():
        print(
            f"equivalence category={label} "
            f"checked={summary.tensors_checked} "
            f"both_none={summary.tensors_both_none} "
            f"max_abs={summary.maximum_absolute:.9g} "
            f"worst_abs={summary.worst_absolute_tensor!r} "
            f"max_rel={summary.maximum_relative:.9g} "
            f"worst_rel={summary.worst_relative_tensor!r}",
            flush=True,
        )
    print("validation=PASS scope=full_model_equivalence", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run_validation(args)
    except BaseException as error:
        print(
            f"validation=FAIL scope=full_model_equivalence "
            f"error_type={type(error).__name__} error={error}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
