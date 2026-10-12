"""Candidate-only CUDA training and resume smoke for fused FFT Gator.

This validator intentionally exercises only the production candidate:

* packed Convergent routing;
* opaque FFT receiver memory;
* the fused Triton packed writer and reader; and
* a compiled :class:`ConvergentLanguageModel` using production semantics.

It trains on deterministic synthetic language batches, writes one local
checkpoint containing the model, AdamW optimizer, RNG, and exact next batch,
then compares the next uninterrupted update with a freshly loaded and
compiled continuation.  The comparison covers the scalar loss, every named
parameter update, and every named AdamW state update.  Unsupported devices or
backends fail explicitly; this script never changes to a portable backend.

Small Blackwell smoke::

    python -u validate_convergent_fft_training.py --device cuda:0

Larger target shapes can be supplied with the same width/node/lane/context
knobs used by the full-model validator.  Checkpoints are local single-write
artifacts rather than periodic training checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

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


CHECKPOINT_FORMAT = "gator.convergent_fft_training"
CHECKPOINT_VERSION = 1
VALIDATION_SCOPE = "candidate_training_resume"

TensorMap = dict[str, torch.Tensor]
ForwardModel = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class ModelShape:
    """All tensor-capacity knobs that affect the compiled candidate."""

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
    """One serializable teacher-prefix language batch."""

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

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, torch.Tensor],
    ) -> "SyntheticLanguageBatch":
        expected = {
            "context_ids",
            "context_mask",
            "targets",
            "output_token_ids",
            "output_mask",
        }
        if set(state) != expected:
            missing = sorted(expected - set(state))
            unexpected = sorted(set(state) - expected)
            raise ValueError(
                "next-batch key mismatch "
                f"missing={missing} unexpected={unexpected}"
            )
        if not all(isinstance(state[name], torch.Tensor) for name in expected):
            raise TypeError("every next-batch value must be a tensor")
        return cls(**{name: state[name].detach().cpu().clone() for name in expected})


@dataclass(frozen=True)
class StepResult:
    loss: torch.Tensor
    gradient_norm: float
    gradients_checked: int


@dataclass(frozen=True)
class ContinuationSnapshot:
    loss: torch.Tensor
    parameter_updates: TensorMap
    optimizer_state_updates: TensorMap


@dataclass(frozen=True)
class DriftSummary:
    tensors_checked: int
    maximum_absolute: float
    maximum_relative: float
    worst_absolute_tensor: str
    worst_relative_tensor: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-tokens", type=int, default=7)
    parser.add_argument("--output-slots", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument(
        "--d-model",
        "--width",
        dest="d_model",
        type=int,
        default=8,
    )
    parser.add_argument("--nodes", type=int, default=9)
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
    parser.add_argument(
        "--packed-memory-feature-chunk",
        "--feature-chunk",
        dest="packed_memory_feature_chunk",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--packed-memory-writer-backend",
        choices=("triton",),
        default="triton",
        help="fixed candidate backend; accepted for release-runner parity",
    )
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer-epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("convergent_fft_training_smoke.pt"),
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="required; the candidate always runs through torch.compile",
    )
    parser.add_argument(
        "--fullgraph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "optionally require one full Dynamo graph; production-compatible "
            "graph capture is the default"
        ),
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
    )
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=2e-5)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "steps",
        "batch_size",
        "context_tokens",
        "output_slots",
        "vocab_size",
        "embedding_dim",
        "d_model",
        "nodes",
        "lanes",
        "active_node_top_k",
        "degree",
        "relays",
        "router_rank",
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
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    finite_positive = (
        "temperature",
        "learning_rate",
        "optimizer_epsilon",
    )
    for name in finite_positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    for name in ("atol", "rtol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not str(args.device).startswith("cuda"):
        raise ValueError("candidate training/resume validation requires CUDA")
    if not args.compile:
        raise ValueError("candidate training/resume validation requires --compile")


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


def build_model_config(
    shape: ModelShape,
    *,
    seed: int,
) -> ConvergentLanguageConfig:
    """Build the fixed candidate backend; no fallback knobs are accepted."""

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
        packed_memory_backend="fft_opaque",
        packed_memory_reader_backend="triton",
        packed_memory_writer_backend="triton",
        packed_memory_feature_chunk=shape.feature_chunk,
        packed_memory_compact_nodes=False,
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


def _batch_seed(seed: int, batch_index: int) -> int:
    if batch_index < 0:
        raise ValueError("batch_index must be nonnegative")
    return (seed * 1_000_003 + batch_index * 97_409 + 17) % (2**63 - 1)


def build_synthetic_batch(
    shape: ModelShape,
    *,
    seed: int,
    batch_index: int,
    device: torch.device | str = "cpu",
) -> SyntheticLanguageBatch:
    """Build a deterministic, index-addressable teacher-prefix batch."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_batch_seed(seed, batch_index))
    context_ids = torch.randint(
        4,
        shape.vocab_size,
        (shape.batch_size, shape.context_tokens),
        generator=generator,
        dtype=torch.long,
    )
    context_mask = torch.ones_like(context_ids, dtype=torch.bool)
    if shape.batch_size > 1 and shape.context_tokens > 1:
        context_mask[1::2, -1] = False
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
    batch = SyntheticLanguageBatch(
        context_ids=context_ids,
        context_mask=context_mask,
        targets=targets,
        output_token_ids=output_token_ids,
        output_mask=output_mask,
    )
    validate_batch(batch, shape)
    return batch.to(device)


def validate_batch(batch: SyntheticLanguageBatch, shape: ModelShape) -> None:
    expected_context = (shape.batch_size, shape.context_tokens)
    expected_output = (shape.batch_size, shape.output_slots)
    if batch.context_ids.shape != expected_context:
        raise ValueError("next-batch context_ids shape is incompatible")
    if batch.context_mask.shape != expected_context:
        raise ValueError("next-batch context_mask shape is incompatible")
    for name in ("targets", "output_token_ids", "output_mask"):
        if getattr(batch, name).shape != expected_output:
            raise ValueError(f"next-batch {name} shape is incompatible")
    if any(
        value.dtype != torch.long
        for value in (
            batch.context_ids,
            batch.targets,
            batch.output_token_ids,
        )
    ):
        raise TypeError("next-batch token tensors must use torch.long")
    if batch.context_mask.dtype != torch.bool or batch.output_mask.dtype != torch.bool:
        raise TypeError("next-batch masks must use torch.bool")
    devices = {
        value.device
        for value in (
            batch.context_ids,
            batch.context_mask,
            batch.targets,
            batch.output_token_ids,
            batch.output_mask,
        )
    }
    if len(devices) != 1:
        raise ValueError("every next-batch tensor must share a device")
    for name in ("context_ids", "targets", "output_token_ids"):
        value = getattr(batch, name)
        if not bool(value.ge(0).logical_and(value.lt(shape.vocab_size)).all()):
            raise ValueError(f"next-batch {name} contains an invalid token ID")
    if shape.output_slots > 1 and not torch.equal(
        batch.output_token_ids[:, 1:],
        batch.targets[:, :-1],
    ):
        raise ValueError("next-batch teacher prefix is not target-shifted")


def require_cuda_triton(device_value: str) -> torch.device:
    if not hasattr(torch, "compile"):
        raise RuntimeError("candidate training/resume validation requires torch.compile")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "candidate training/resume validation requires CUDA; "
            "torch.cuda.is_available() is false"
        )
    if not triton_packed_reader_available():
        raise RuntimeError(
            "candidate training/resume validation requires the Triton packed "
            "reader plus torch.library.triton_op/wrap_triton support"
        )
    if not triton_packed_writer_available():
        raise RuntimeError(
            "candidate training/resume validation requires the Triton packed "
            "writer plus torch.library.triton_op/wrap_triton support"
        )
    device = torch.device(device_value)
    if device.type != "cuda":
        raise RuntimeError("candidate training/resume validation requires CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    return device


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def emit_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True), flush=True)


def _tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    return value


def state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def named_parameters_to_cpu(model: torch.nn.Module) -> TensorMap:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def optimizer_parameter_names(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> list[list[str]]:
    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    result: list[list[str]] = []
    seen: set[str] = set()
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            name = name_by_id.get(id(parameter))
            if name is None:
                raise AssertionError("optimizer contains a parameter outside the model")
            if name in seen:
                raise AssertionError(f"optimizer contains duplicate parameter {name}")
            seen.add(name)
            names.append(name)
        result.append(names)
    expected = set(name_by_id.values())
    if seen != expected:
        raise AssertionError(
            "optimizer/model parameter mismatch "
            f"missing={sorted(expected - seen)} unexpected={sorted(seen - expected)}"
        )
    return result


def _state_value_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, bool):
        return torch.tensor(value, dtype=torch.bool)
    if isinstance(value, int):
        return torch.tensor(value, dtype=torch.int64)
    if isinstance(value, float):
        return torch.tensor(value, dtype=torch.float64)
    raise TypeError(f"unsupported optimizer state value {type(value).__name__}")


def named_optimizer_state_to_cpu(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> TensorMap:
    result: TensorMap = {}
    for groups, names in zip(
        optimizer.param_groups,
        optimizer_parameter_names(model, optimizer),
    ):
        for parameter, name in zip(groups["params"], names):
            state = optimizer.state.get(parameter, {})
            for state_name, value in sorted(state.items()):
                result[f"{name}::{state_name}"] = _state_value_tensor(value)
    return result


def serialized_named_optimizer_state(
    optimizer_state: Mapping[str, Any],
    parameter_names: Sequence[Sequence[str]],
) -> TensorMap:
    groups = optimizer_state.get("param_groups")
    state = optimizer_state.get("state")
    if not isinstance(groups, list) or not isinstance(state, dict):
        raise TypeError("optimizer checkpoint has an invalid state_dict")
    if len(groups) != len(parameter_names):
        raise ValueError("optimizer checkpoint parameter-group count mismatch")
    result: TensorMap = {}
    for group, group_names in zip(groups, parameter_names):
        parameter_ids = group.get("params")
        if not isinstance(parameter_ids, list) or len(parameter_ids) != len(group_names):
            raise ValueError("optimizer checkpoint parameter-group width mismatch")
        for parameter_id, name in zip(parameter_ids, group_names):
            parameter_state = state.get(parameter_id, {})
            if not isinstance(parameter_state, dict):
                raise TypeError("optimizer checkpoint parameter state must be a mapping")
            for state_name, value in sorted(parameter_state.items()):
                result[f"{name}::{state_name}"] = _state_value_tensor(value)
    return result


def require_named_tensors_finite(
    label: str,
    values: Mapping[str, torch.Tensor],
) -> None:
    if not values:
        raise AssertionError(f"{label} is empty")
    # Non-capturable AdamW normally keeps ``step`` on CPU while its moment
    # tensors live beside CUDA parameters.  Reduce once per device rather
    # than attempting to stack cross-device scalar checks.
    by_device: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
    for name, value in values.items():
        by_device.setdefault(value.device, []).append(
            (name, torch.isfinite(value.detach()).all())
        )
    failing: list[str] = []
    for checks in by_device.values():
        if bool(torch.stack([check for _, check in checks]).all()):
            continue
        failing.extend(name for name, check in checks if not bool(check))
    if failing:
        raise FloatingPointError(
            f"{label} contains non-finite tensors: {failing}"
        )


def _forward_loss(
    forward_model: ForwardModel,
    batch: SyntheticLanguageBatch,
    *,
    shape: ModelShape,
    causal: bool,
    temperature: torch.Tensor,
) -> torch.Tensor:
    logits = forward_model(
        batch.context_ids,
        batch.context_mask,
        output_slots=shape.output_slots,
        output_token_ids=batch.output_token_ids,
        output_mask=batch.output_mask,
        causal=causal,
        temperature=temperature,
    )
    if not isinstance(logits, torch.Tensor):
        raise TypeError("candidate must return one logits tensor")
    expected = (shape.batch_size, shape.output_slots, shape.vocab_size)
    if logits.shape != expected:
        raise AssertionError(
            f"candidate logits shape {tuple(logits.shape)} != {expected}"
        )
    if not bool(torch.isfinite(logits.detach()).all()):
        raise FloatingPointError("candidate logits contain a non-finite value")
    loss = F.cross_entropy(
        logits.float().reshape(-1, shape.vocab_size),
        batch.targets.reshape(-1),
    )
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("candidate loss is non-finite")
    return loss


def run_adamw_step(
    model: ConvergentLanguageModel,
    forward_model: ForwardModel,
    optimizer: torch.optim.AdamW,
    batch: SyntheticLanguageBatch,
    *,
    shape: ModelShape,
    causal: bool,
    temperature: torch.Tensor,
) -> StepResult:
    named_parameters = dict(model.named_parameters())
    if not named_parameters:
        raise RuntimeError("candidate model has no named parameters")
    require_named_tensors_finite("parameters_before", named_parameters)
    optimizer.zero_grad(set_to_none=True)
    loss = _forward_loss(
        forward_model,
        batch,
        shape=shape,
        causal=causal,
        temperature=temperature,
    )
    loss.backward()
    missing = [
        name for name, parameter in named_parameters.items() if parameter.grad is None
    ]
    if missing:
        raise AssertionError(f"candidate parameters have no gradient: {missing}")
    gradients = {
        name: parameter.grad
        for name, parameter in named_parameters.items()
        if parameter.grad is not None
    }
    require_named_tensors_finite("gradients", gradients)
    squared_norm = torch.stack(
        [gradient.detach().float().square().sum() for gradient in gradients.values()]
    ).sum()
    gradient_norm = float(squared_norm.sqrt().detach().cpu())
    optimizer.step()
    require_named_tensors_finite("parameters_after", named_parameters)
    optimizer_values = {
        f"state_{index}": value
        for index, value in enumerate(
            item
            for state in optimizer.state.values()
            for item in state.values()
            if isinstance(item, torch.Tensor)
        )
    }
    require_named_tensors_finite("optimizer_state_after", optimizer_values)
    return StepResult(
        loss=loss.detach().cpu().clone(),
        gradient_norm=gradient_norm,
        gradients_checked=len(gradients),
    )


def tensor_map_updates(
    after: Mapping[str, torch.Tensor],
    before: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> TensorMap:
    if after.keys() != before.keys():
        missing = sorted(set(before) - set(after))
        unexpected = sorted(set(after) - set(before))
        raise AssertionError(
            f"{label} key mismatch missing={missing} unexpected={unexpected}"
        )
    result: TensorMap = {}
    for name, before_value in before.items():
        after_value = after[name]
        if after_value.shape != before_value.shape:
            raise AssertionError(f"{label}[{name}] shape changed")
        if after_value.dtype != before_value.dtype:
            raise AssertionError(f"{label}[{name}] dtype changed")
        if after_value.is_floating_point() or after_value.is_complex():
            result[name] = after_value - before_value
        else:
            result[name] = after_value.to(torch.int64) - before_value.to(torch.int64)
    require_named_tensors_finite(label, result)
    return result


def capture_continuation(
    model: ConvergentLanguageModel,
    forward_model: ForwardModel,
    optimizer: torch.optim.AdamW,
    batch: SyntheticLanguageBatch,
    *,
    baseline_parameters: Mapping[str, torch.Tensor],
    baseline_optimizer_state: Mapping[str, torch.Tensor],
    shape: ModelShape,
    causal: bool,
    temperature: torch.Tensor,
) -> ContinuationSnapshot:
    result = run_adamw_step(
        model,
        forward_model,
        optimizer,
        batch,
        shape=shape,
        causal=causal,
        temperature=temperature,
    )
    parameter_updates = tensor_map_updates(
        named_parameters_to_cpu(model),
        baseline_parameters,
        label="parameter_updates",
    )
    optimizer_updates = tensor_map_updates(
        named_optimizer_state_to_cpu(model, optimizer),
        baseline_optimizer_state,
        label="optimizer_state_updates",
    )
    return ContinuationSnapshot(
        loss=result.loss,
        parameter_updates=parameter_updates,
        optimizer_state_updates=optimizer_updates,
    )


def training_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "causal": bool(args.causal),
        "temperature": float(args.temperature),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "optimizer_epsilon": float(args.optimizer_epsilon),
        "seed": int(args.seed),
        "compile_mode": str(args.compile_mode),
        "fullgraph": bool(args.fullgraph),
    }


def build_checkpoint_payload(
    model: ConvergentLanguageModel,
    optimizer: torch.optim.AdamW,
    next_batch: SyntheticLanguageBatch,
    *,
    shape: ModelShape,
    config: ConvergentLanguageConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    parameter_names = optimizer_parameter_names(model, optimizer)
    optimizer_state = _tree_to_cpu(optimizer.state_dict())
    completed_steps = int(args.steps)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "shape": asdict(shape),
        "model_config": config.to_dict(),
        "training_spec": training_spec(args),
        "completed_steps": completed_steps,
        "next_batch_index": completed_steps,
        "model_state": state_dict_to_cpu(model),
        "optimizer_state": optimizer_state,
        "optimizer_parameter_names": parameter_names,
        "next_batch": next_batch.state_dict(),
        "cpu_rng_state": torch.get_rng_state().cpu().clone(),
        "cuda_rng_state": torch.cuda.get_rng_state(device).cpu().clone(),
    }
    baseline_optimizer = serialized_named_optimizer_state(
        optimizer_state,
        parameter_names,
    )
    require_named_tensors_finite("checkpoint_model_state", payload["model_state"])
    require_named_tensors_finite("checkpoint_optimizer_state", baseline_optimizer)
    return payload


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()
    return resolved


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only keyword.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("training/resume checkpoint must contain a mapping")
    return payload


def _assert_batch_equal(
    actual: SyntheticLanguageBatch,
    expected: SyntheticLanguageBatch,
) -> None:
    for name in actual.state_dict():
        if not torch.equal(getattr(actual, name), getattr(expected, name)):
            raise AssertionError(f"next-batch mismatch for {name}")


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    shape: ModelShape,
    config: ConvergentLanguageConfig,
    args: argparse.Namespace,
) -> SyntheticLanguageBatch:
    required = {
        "format",
        "version",
        "shape",
        "model_config",
        "training_spec",
        "completed_steps",
        "next_batch_index",
        "model_state",
        "optimizer_state",
        "optimizer_parameter_names",
        "next_batch",
        "cpu_rng_state",
        "cuda_rng_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {missing}")
    if payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint format is not the candidate training smoke")
    if payload["version"] != CHECKPOINT_VERSION:
        raise ValueError("checkpoint version is unsupported")
    if payload["shape"] != asdict(shape):
        raise ValueError("checkpoint target shape does not match this invocation")
    if payload["model_config"] != config.to_dict():
        raise ValueError("checkpoint candidate model configuration does not match")
    if payload["training_spec"] != training_spec(args):
        raise ValueError("checkpoint training specification does not match")
    if payload["completed_steps"] != args.steps:
        raise ValueError("checkpoint completed_steps does not match")
    if payload["next_batch_index"] != args.steps:
        raise ValueError("checkpoint next_batch_index does not match")
    if not isinstance(payload["model_state"], dict):
        raise TypeError("checkpoint model_state must be a mapping")
    if not isinstance(payload["optimizer_state"], dict):
        raise TypeError("checkpoint optimizer_state must be a mapping")
    parameter_names = payload["optimizer_parameter_names"]
    if not isinstance(parameter_names, list):
        raise TypeError("checkpoint optimizer_parameter_names must be a list")
    baseline_optimizer = serialized_named_optimizer_state(
        payload["optimizer_state"],
        parameter_names,
    )
    require_named_tensors_finite("loaded_model_state", payload["model_state"])
    require_named_tensors_finite("loaded_optimizer_state", baseline_optimizer)
    next_batch = SyntheticLanguageBatch.from_state_dict(payload["next_batch"])
    validate_batch(next_batch, shape)
    expected_batch = build_synthetic_batch(
        shape,
        seed=args.seed,
        batch_index=args.steps,
    )
    _assert_batch_equal(next_batch, expected_batch)
    for name in ("cpu_rng_state", "cuda_rng_state"):
        value = payload[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
            raise TypeError(f"checkpoint {name} must be a uint8 tensor")
    return next_batch


def restore_rng_state(payload: Mapping[str, Any], device: torch.device) -> None:
    torch.set_rng_state(payload["cpu_rng_state"].cpu())
    torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), device=device)


def make_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.optimizer_epsilon,
        foreach=False,
        fused=False,
    )


def compile_candidate(
    model: ConvergentLanguageModel,
    args: argparse.Namespace,
) -> torch.nn.Module:
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
    return torch.compile(
        model,
        fullgraph=args.fullgraph,
        mode=args.compile_mode,
    )


def dynamo_metrics() -> dict[str, int]:
    """Return stable scalar graph-capture counters for one compile phase."""

    if not hasattr(torch, "_dynamo"):
        raise RuntimeError("Dynamo counters are unavailable")
    counters = torch._dynamo.utils.counters
    stats = counters.get("stats", {})
    frames = counters.get("frames", {})
    graph_break = counters.get("graph_break", {})
    metrics = {
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "calls_captured": int(stats.get("calls_captured", 0)),
        "frames_total": int(frames.get("total", 0)),
        "frames_ok": int(frames.get("ok", 0)),
        "graph_breaks": sum(int(value) for value in graph_break.values()),
    }
    if metrics["unique_graphs"] < 1:
        raise AssertionError(
            "torch.compile captured no Dynamo graph for the candidate"
        )
    return metrics


def _tensor_drift(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    if actual.numel() == 0:
        return 0.0, 0.0
    difference = (actual.float() - expected.float()).abs()
    scale = torch.maximum(actual.float().abs(), expected.float().abs()).clamp_min(
        1e-12
    )
    return float(difference.max()), float((difference / scale).max())


def compare_tensor_maps(
    label: str,
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
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
    maximum_absolute = 0.0
    maximum_relative = 0.0
    worst_absolute = ""
    worst_relative = ""
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if actual_value.shape != expected_value.shape:
            raise AssertionError(f"{label}[{name}] shape mismatch")
        torch.testing.assert_close(
            actual_value,
            expected_value,
            atol=atol,
            rtol=rtol,
            msg=lambda message, tensor_name=name: (
                f"{label}[{tensor_name}]: {message}"
            ),
        )
        absolute, relative = _tensor_drift(actual_value, expected_value)
        if absolute >= maximum_absolute:
            maximum_absolute = absolute
            worst_absolute = name
        if relative >= maximum_relative:
            maximum_relative = relative
            worst_relative = name
    return DriftSummary(
        tensors_checked=len(actual),
        maximum_absolute=maximum_absolute,
        maximum_relative=maximum_relative,
        worst_absolute_tensor=worst_absolute,
        worst_relative_tensor=worst_relative,
    )


def compare_continuations(
    resumed: ContinuationSnapshot,
    uninterrupted: ContinuationSnapshot,
    *,
    atol: float,
    rtol: float,
) -> dict[str, DriftSummary]:
    torch.testing.assert_close(
        resumed.loss,
        uninterrupted.loss,
        atol=atol,
        rtol=rtol,
        msg="resumed continuation loss differs from uninterrupted loss",
    )
    loss_absolute, loss_relative = _tensor_drift(
        resumed.loss,
        uninterrupted.loss,
    )
    return {
        "loss": DriftSummary(
            tensors_checked=1,
            maximum_absolute=loss_absolute,
            maximum_relative=loss_relative,
            worst_absolute_tensor="loss",
            worst_relative_tensor="loss",
        ),
        "parameter_updates": compare_tensor_maps(
            "parameter_updates",
            resumed.parameter_updates,
            uninterrupted.parameter_updates,
            atol=atol,
            rtol=rtol,
        ),
        "optimizer_state_updates": compare_tensor_maps(
            "optimizer_state_updates",
            resumed.optimizer_state_updates,
            uninterrupted.optimizer_state_updates,
            atol=atol,
            rtol=rtol,
        ),
    }


def _release_cuda_state(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def _load_fresh_candidate(
    payload: Mapping[str, Any],
    config: ConvergentLanguageConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ConvergentLanguageModel, torch.nn.Module, torch.optim.AdamW]:
    model = ConvergentLanguageModel(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = make_optimizer(model, args)
    optimizer.load_state_dict(payload["optimizer_state"])
    loaded_names = optimizer_parameter_names(model, optimizer)
    if loaded_names != payload["optimizer_parameter_names"]:
        raise AssertionError("loaded optimizer parameter order changed")
    expected_optimizer = serialized_named_optimizer_state(
        payload["optimizer_state"],
        payload["optimizer_parameter_names"],
    )
    actual_optimizer = named_optimizer_state_to_cpu(model, optimizer)
    compare_tensor_maps(
        "loaded_optimizer_state",
        actual_optimizer,
        expected_optimizer,
        atol=0.0,
        rtol=0.0,
    )
    compiled = compile_candidate(model, args)
    return model, compiled, optimizer


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    device = require_cuda_triton(args.device)
    shape = shape_from_args(args)
    config = build_model_config(shape, seed=args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    emit_json(
        {
            "event": "runtime",
            "scope": VALIDATION_SCOPE,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": package_version("triton"),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "shape": asdict(shape),
            "backend": "packed+fft_opaque+triton_reader+triton_writer",
            "compile_mode": args.compile_mode,
            "fullgraph": args.fullgraph,
        }
    )

    model = ConvergentLanguageModel(config).to(device=device, dtype=torch.float32)
    optimizer = make_optimizer(model, args)
    compiled = compile_candidate(model, args)
    temperature = torch.tensor(
        args.temperature,
        device=device,
        dtype=torch.float32,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.perf_counter()
    for step in range(args.steps):
        batch = build_synthetic_batch(
            shape,
            seed=args.seed,
            batch_index=step,
            device=device,
        )
        result = run_adamw_step(
            model,
            compiled,
            optimizer,
            batch,
            shape=shape,
            causal=args.causal,
            temperature=temperature,
        )
        torch.cuda.synchronize(device)
        emit_json(
            {
                "event": "train_step",
                "scope": VALIDATION_SCOPE,
                "step": step + 1,
                "loss": float(result.loss),
                "gradient_norm": result.gradient_norm,
                "gradients_checked": result.gradients_checked,
            }
        )

    next_batch = build_synthetic_batch(
        shape,
        seed=args.seed,
        batch_index=args.steps,
    )
    payload = build_checkpoint_payload(
        model,
        optimizer,
        next_batch,
        shape=shape,
        config=config,
        args=args,
        device=device,
    )
    checkpoint_path = save_checkpoint(args.checkpoint_path, payload)
    loaded_payload = load_checkpoint(checkpoint_path)
    loaded_next_batch = validate_checkpoint_payload(
        loaded_payload,
        shape=shape,
        config=config,
        args=args,
    )
    baseline_parameters = {
        name: loaded_payload["model_state"][name]
        for name, _ in model.named_parameters()
    }
    baseline_optimizer = serialized_named_optimizer_state(
        loaded_payload["optimizer_state"],
        loaded_payload["optimizer_parameter_names"],
    )
    emit_json(
        {
            "event": "checkpoint_saved",
            "scope": VALIDATION_SCOPE,
            "path": str(checkpoint_path),
            "completed_steps": args.steps,
            "next_batch_index": args.steps,
        }
    )

    restore_rng_state(loaded_payload, device)
    uninterrupted = capture_continuation(
        model,
        compiled,
        optimizer,
        loaded_next_batch.to(device),
        baseline_parameters=baseline_parameters,
        baseline_optimizer_state=baseline_optimizer,
        shape=shape,
        causal=args.causal,
        temperature=temperature,
    )
    torch.cuda.synchronize(device)
    uninterrupted_dynamo = dynamo_metrics()
    del batch
    del compiled
    del optimizer
    del model
    del temperature
    _release_cuda_state(device)

    # Cross the serialization boundary again after the uninterrupted model is
    # gone.  The resumed branch must not inherit a live Python payload from
    # the first branch.
    del loaded_payload
    resumed_payload = load_checkpoint(checkpoint_path)
    resumed_next_batch = validate_checkpoint_payload(
        resumed_payload,
        shape=shape,
        config=config,
        args=args,
    )
    resumed_model, resumed_compiled, resumed_optimizer = _load_fresh_candidate(
        resumed_payload,
        config,
        args,
        device,
    )
    resumed_temperature = torch.tensor(
        args.temperature,
        device=device,
        dtype=torch.float32,
    )
    restore_rng_state(resumed_payload, device)
    resumed = capture_continuation(
        resumed_model,
        resumed_compiled,
        resumed_optimizer,
        resumed_next_batch.to(device),
        baseline_parameters=baseline_parameters,
        baseline_optimizer_state=baseline_optimizer,
        shape=shape,
        causal=args.causal,
        temperature=resumed_temperature,
    )
    torch.cuda.synchronize(device)
    resumed_dynamo = dynamo_metrics()
    comparisons = compare_continuations(
        resumed,
        uninterrupted,
        atol=args.atol,
        rtol=args.rtol,
    )
    elapsed = time.perf_counter() - started
    return {
        "event": "validation",
        "status": "PASS",
        "scope": VALIDATION_SCOPE,
        "backend": "packed+fft_opaque+triton_reader+triton_writer",
        "compiled": True,
        "fullgraph": args.fullgraph,
        "parameter_count": parameter_count,
        "training_steps_before_checkpoint": args.steps,
        "continuation_steps_compared": 1,
        "checkpoint_path": str(checkpoint_path),
        "elapsed_seconds": elapsed,
        "shape": asdict(shape),
        "comparisons": {
            label: asdict(summary) for label, summary in comparisons.items()
        },
        "dynamo": {
            "uninterrupted": uninterrupted_dynamo,
            "resumed": resumed_dynamo,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        report = run_validation(args)
        emit_json(report)
    except BaseException as error:
        emit_json(
            {
                "event": "validation",
                "status": "FAIL",
                "scope": VALIDATION_SCOPE,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise


if __name__ == "__main__":
    main()
