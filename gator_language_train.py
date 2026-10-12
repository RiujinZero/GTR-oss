"""Train Gator to form a complete SmolTalk response window simultaneously."""

from __future__ import annotations

import argparse
import atexit
from dataclasses import replace
import hashlib
from itertools import chain
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Sequence

import torch
import torch.nn.functional as F

from gator import (
    LanguageBatchPipeline,
    OptimizationEngine,
    StepResult,
    StewardshipExecutionController,
    best_aligned_diagnostics,
    compile_module,
    configure_float32_runtime,
    mark_compiled_iteration_begin,
    monotonic_forward_sum_nll,
    sequence_quality_loss,
    viterbi_aligned_cross_entropy,
)
from gator.cli import (
    add_compute_arguments,
    add_data_pipeline_arguments,
    add_optimization_arguments,
    compute_config_from_args,
    data_pipeline_config_from_args,
    optimization_config_from_args,
    preparation_config_from_args,
)
from gator.data import (
    assistant_examples,
    conversation_ascii_letter_ratio,
    iter_tinystories,
    iter_jsonl_conversations,
    load_smoltalk,
    split_conversations,
    tinystories_conversations,
    tinystories_sentence_pairs,
)
from gator.models import (
    GatorLanguageConfig,
    GatorLanguageModel,
    OneShotTransformerLanguageConfig,
    OneShotTransformerLanguageModel,
)
from gator_cell_growth import grow_gator_state_dict
from gator.tasks import (
    EncodedAssistantExample,
    LanguageBatch,
    LanguageBatchConfig,
    LanguageBatcher,
    LanguageExampleFilterConfig,
    LanguageExampleFilterStats,
    MixtureLanguageBatcher,
    PackedLanguageSource,
    StreamingLanguageSource,
    WeightedLanguageSource,
    filter_language_examples,
    legal_response_token_mask,
    masked_cross_entropy,
    rigid_blank_tail_targets,
    stewardship_route_losses_from_statistics,
    whitespace_formation_penalties,
    whitespace_token_mask,
)
from gator.preparation import tokenize_assistant_examples
from gator.tokenization import CompactTokenizer
from gator_convergent import (
    ConvergentLanguageConfig,
    ConvergentLanguageModel,
)


_EXECUTION_PRESET_OVERRIDES: dict[str, dict[str, object]] = {
    "manual": {},
    "dense_gpu_safe": {
        "compile": False,
        "compile_mode": "default",
        "compile_shape_padding": None,
        "float32_matmul_precision": "highest",
        "cudnn_tf32": False,
        "receiver_memory_backend": "convolution",
        "dense_router_backend": "einsum",
        "dense_node_backend": "folded_bmm",
        "dense_kernel_padding": 1,
        "stewardship_execution": "dense",
        "packed_stewardship_backend": "persistent_bucketed",
        "stewardship_execution_check_every": 100,
        "batch_pipeline": "prefetch",
        "prefetch_batches": 2,
        "pin_memory": None,
        "non_blocking_transfer": True,
        "dedicated_copy_stream": True,
    },
    "dense_gpu": {
        "compile": True,
        "compile_mode": "default",
        "compile_shape_padding": None,
        "float32_matmul_precision": "high",
        "cudnn_tf32": True,
        "receiver_memory_backend": "convolution",
        "dense_router_backend": "einsum",
        "dense_node_backend": "folded_bmm",
        "dense_kernel_padding": 1,
        "stewardship_execution": "dense",
        "packed_stewardship_backend": "persistent_bucketed",
        "stewardship_execution_check_every": 100,
        "batch_pipeline": "prefetch",
        "prefetch_batches": 2,
        "pin_memory": None,
        "non_blocking_transfer": True,
        "dedicated_copy_stream": True,
    },
    "optimized_gpu_safe": {
        "compile": False,
        "compile_mode": "default",
        "compile_shape_padding": None,
        "float32_matmul_precision": "highest",
        "cudnn_tf32": False,
        "receiver_memory_backend": "convolution",
        "dense_router_backend": "einsum",
        "dense_node_backend": "folded_bmm",
        "dense_kernel_padding": 1,
        "stewardship_execution": "packed_auto",
        "stewardship_packed_threshold": 0.5,
        "packed_stewardship_backend": "persistent_bucketed",
        "stewardship_execution_check_every": 100,
        "batch_pipeline": "prefetch",
        "prefetch_batches": 2,
        "pin_memory": None,
        "non_blocking_transfer": True,
        "dedicated_copy_stream": True,
    },
    "optimized_gpu": {
        "compile": True,
        "compile_mode": "default",
        "compile_shape_padding": None,
        "float32_matmul_precision": "high",
        "cudnn_tf32": True,
        "receiver_memory_backend": "convolution",
        "dense_router_backend": "einsum",
        "dense_node_backend": "folded_bmm",
        "dense_kernel_padding": 1,
        "stewardship_execution": "packed_auto",
        "stewardship_packed_threshold": 0.5,
        "packed_stewardship_backend": "persistent_bucketed",
        "stewardship_execution_check_every": 100,
        "batch_pipeline": "prefetch",
        "prefetch_batches": 2,
        "pin_memory": None,
        "non_blocking_transfer": True,
        "dedicated_copy_stream": True,
    },
}


def serial_next_token_batch(
    batch: LanguageBatch,
    *,
    eos_token_id: int,
    prefix_lengths: torch.Tensor | None = None,
    positions_per_example: int = 1,
) -> LanguageBatch:
    """Convert full responses into teacher-forced next-token examples.

    The original context and a selected correct response prefix share the
    existing context window. Older prompt tokens fall off the left edge when
    the prefix fills that window, matching ordinary fixed-context
    autoregressive training. The returned batch always has one target slot
    per row.

    positions_per_example: when 1 (default), each original row contributes a
    single randomly-sampled prefix position, matching the original behavior
    (dense coverage across the response window only emerges over many steps).
    When > 1, each original row is expanded into this many rows, one per
    stratified prefix position spread across its response window (one
    random offset per stratum, not pure uniform random, for more even
    coverage than resampling the same random distribution N times) - giving
    denser per-step supervision across the full response at a
    positions_per_example x compute cost. Ignored when prefix_lengths is
    passed explicitly (deterministic validation path).
    """

    if positions_per_example < 1:
        raise ValueError("positions_per_example must be >= 1")

    batch_size, context_width = batch.context_ids.shape
    target_lengths = batch.target_mask.sum(dim=1)
    if bool(target_lengths.le(0).any()):
        raise ValueError("serial decoding requires one target token per row")
    if prefix_lengths is None:
        if positions_per_example == 1:
            prefix_lengths = (
                torch.rand(
                    batch_size,
                    device=target_lengths.device,
                )
                * target_lengths.to(torch.float32)
            ).to(torch.long)
        else:
            expand_index = torch.arange(
                batch_size, device=batch.context_ids.device
            ).repeat_interleave(positions_per_example)
            batch = LanguageBatch(
                context_ids=batch.context_ids[expand_index],
                context_mask=batch.context_mask[expand_index],
                target_ids=batch.target_ids[expand_index],
                target_mask=batch.target_mask[expand_index],
                target_complete=batch.target_complete[expand_index],
            )
            target_lengths = target_lengths[expand_index]
            batch_size = batch_size * positions_per_example
            strata = torch.arange(
                positions_per_example,
                device=target_lengths.device,
                dtype=torch.float32,
            ).repeat(batch_size // positions_per_example)
            fractions = (
                strata
                + torch.rand(batch_size, device=target_lengths.device)
            ) / positions_per_example
            prefix_lengths = (
                fractions * target_lengths.to(torch.float32)
            ).to(torch.long).clamp_min(0).clamp_max(
                (target_lengths - 1).clamp(min=0)
            )
    else:
        prefix_lengths = prefix_lengths.to(
            device=target_lengths.device,
            dtype=torch.long,
        )
    if prefix_lengths.shape != (batch_size,):
        raise ValueError("prefix_lengths must have shape [batch]")
    if bool(
        (
            prefix_lengths.lt(0)
            | prefix_lengths.ge(target_lengths)
        ).any()
    ):
        raise ValueError(
            "each prefix length must select an existing next token"
        )

    serial_context = torch.zeros_like(batch.context_ids)
    serial_mask = torch.zeros_like(batch.context_mask)
    for row in range(batch_size):
        context = batch.context_ids[row][batch.context_mask[row]]
        prefix_length = int(prefix_lengths[row])
        prefix = batch.target_ids[row, :prefix_length]
        combined = torch.cat((context, prefix))[-context_width:]
        start = context_width - combined.numel()
        serial_context[row, start:] = combined
        serial_mask[row, start:] = True

    next_tokens = batch.target_ids.gather(
        1,
        prefix_lengths[:, None],
    )
    return LanguageBatch(
        context_ids=serial_context,
        context_mask=serial_mask,
        target_ids=next_tokens,
        target_mask=torch.ones_like(next_tokens, dtype=torch.bool),
        target_complete=next_tokens[:, 0].eq(eos_token_id),
    )


def parallel_next_chunk_batch(
    batch: LanguageBatch,
    *,
    eos_token_id: int,
    chunk_size: int,
    chunk_indices: torch.Tensor | None = None,
) -> LanguageBatch:
    """Convert full responses into parallel next-chunk examples.

    Each row receives the original prompt plus every *completed* gold chunk
    preceding the selected chunk.  The selected chunk itself is returned as
    ``chunk_size`` target slots and is never exposed in the input.  This keeps
    tokens inside a chunk genuinely simultaneous while teaching the model to
    advance through long responses one committed formation at a time.

    Training samples one aligned chunk uniformly per row.  Callers can pass
    explicit ``chunk_indices`` for deterministic validation.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    batch_size, context_width = batch.context_ids.shape
    target_lengths = batch.target_mask.sum(dim=1)
    if bool(target_lengths.le(0).any()):
        raise ValueError("chunk decoding requires one target token per row")
    chunk_counts = torch.div(
        target_lengths + chunk_size - 1,
        chunk_size,
        rounding_mode="floor",
    )
    if chunk_indices is None:
        chunk_indices = (
            torch.rand(
                batch_size,
                device=target_lengths.device,
            )
            * chunk_counts.to(torch.float32)
        ).to(torch.long)
    else:
        chunk_indices = chunk_indices.to(
            device=target_lengths.device,
            dtype=torch.long,
        )
    if chunk_indices.shape != (batch_size,):
        raise ValueError("chunk_indices must have shape [batch]")
    if bool(
        (
            chunk_indices.lt(0)
            | chunk_indices.ge(chunk_counts)
        ).any()
    ):
        raise ValueError(
            "each chunk index must select an existing target chunk"
        )

    chunk_context = torch.zeros_like(batch.context_ids)
    chunk_context_mask = torch.zeros_like(batch.context_mask)
    chunk_target_ids = torch.zeros(
        batch_size,
        chunk_size,
        dtype=batch.target_ids.dtype,
        device=batch.target_ids.device,
    )
    chunk_target_mask = torch.zeros(
        batch_size,
        chunk_size,
        dtype=torch.bool,
        device=batch.target_mask.device,
    )
    for row in range(batch_size):
        chunk_start = int(chunk_indices[row]) * chunk_size
        target_length = int(target_lengths[row])
        chunk_end = min(chunk_start + chunk_size, target_length)

        context = batch.context_ids[row][batch.context_mask[row]]
        completed_chunks = batch.target_ids[row, :chunk_start]
        combined = torch.cat((context, completed_chunks))[-context_width:]
        context_start = context_width - combined.numel()
        chunk_context[row, context_start:] = combined
        chunk_context_mask[row, context_start:] = True

        active = chunk_end - chunk_start
        chunk_target_ids[row, :active] = batch.target_ids[
            row,
            chunk_start:chunk_end,
        ]
        chunk_target_mask[row, :active] = True

    return LanguageBatch(
        context_ids=chunk_context,
        context_mask=chunk_context_mask,
        target_ids=chunk_target_ids,
        target_mask=chunk_target_mask,
        target_complete=(
            chunk_target_ids.eq(eos_token_id) & chunk_target_mask
        ).any(dim=1),
    )


def teacher_forced_output_inputs(
    batch: LanguageBatch,
    *,
    blank_token_id: int,
) -> torch.Tensor:
    """Return BLANK + shifted reference tokens for causal output slots."""

    shifted = torch.full_like(batch.target_ids, blank_token_id)
    if shifted.shape[1] > 1:
        shifted[:, 1:] = torch.where(
            batch.target_mask[:, :-1],
            batch.target_ids[:, :-1],
            torch.full_like(
                batch.target_ids[:, :-1],
                blank_token_id,
            ),
        )
    return shifted


def tokenwise_teacher_output_inputs(
    batch: LanguageBatch,
    *,
    blank_token_id: int,
    removed_tokens: int,
    row_offset: int = 0,
    seed: int = 0,
    rotation: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Remove an exact number of causal teacher tokens from every row.

    Slot zero is always unsupported because causal teacher forcing begins with
    BLANK.  ``removed_tokens`` counts additional shifted reference tokens to
    replace with BLANK.  A deterministic integer hash rotates the removed
    positions across rows and training steps, so every address learns to work
    without support while each curriculum level removes exactly one more
    scaffold token per complete row.
    """

    if removed_tokens < 0:
        raise ValueError("removed_tokens cannot be negative")
    shifted = teacher_forced_output_inputs(
        batch,
        blank_token_id=blank_token_id,
    )
    eligible = torch.zeros_like(batch.target_mask)
    if eligible.shape[1] > 1:
        eligible[:, 1:] = batch.target_mask[:, :-1]

    rows = (
        torch.arange(
            shifted.shape[0],
            device=shifted.device,
            dtype=torch.int64,
        )
        + int(row_offset)
    )[:, None]
    slots = torch.arange(
        shifted.shape[1],
        device=shifted.device,
        dtype=torch.int64,
    )[None, :]
    priorities = (
        rows * 1_103_515_245
        + slots * 12_345
        + int(seed) * 2_654_435_761
        + int(rotation) * 97_531
    ) & 0x7FFFFFFF
    priorities = priorities.masked_fill(~eligible, 0x7FFFFFFF)
    order = priorities.argsort(dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(
        1,
        order,
        torch.arange(
            shifted.shape[1],
            device=shifted.device,
            dtype=torch.int64,
        )[None, :].expand_as(order),
    )
    row_eligible = eligible.sum(dim=1, keepdim=True)
    row_removed = torch.minimum(
        row_eligible,
        torch.full_like(row_eligible, int(removed_tokens)),
    )
    removed = eligible & rank.lt(row_removed)
    visible = eligible & ~removed
    inputs = torch.where(
        visible,
        shifted,
        torch.full_like(shifted, blank_token_id),
    )
    # These are the output positions whose immediate causal teacher input is
    # absent. Slot zero belongs here even at the fully supported level.
    unsupported_slots = batch.target_mask & ~visible
    eligible_count = int(eligible.sum())
    visible_count = int(visible.sum())
    removed_count = int(removed.sum())
    return inputs, unsupported_slots, {
        "requested_removed_tokens": float(removed_tokens),
        "eligible_tokens": float(eligible_count),
        "visible_tokens": float(visible_count),
        "removed_tokens": float(removed_count),
        "actual_retention": visible_count / max(eligible_count, 1),
        "unsupported_slots": float(unsupported_slots.sum()),
        "batch_rows": float(shifted.shape[0]),
    }


def prefix_denoising_output_inputs(
    batch: LanguageBatch,
    *,
    blank_token_id: int,
    retention: float,
    zero_prefix_fraction: float,
    short_prefix_fraction: float,
    short_prefix_max_tokens: int,
    span_fraction: float = 0.50,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Corrupt shifted response prefixes without exposing future targets.

    Ordinary rows use either independent token dropout or one contiguous
    missing span.  Zero-prefix rows remove every response writer, while
    short-prefix rows retain only a small leading scaffold.  The model remains
    causally masked, so a visible token can only support later output slots.
    """

    for name, value in (
        ("retention", retention),
        ("zero_prefix_fraction", zero_prefix_fraction),
        ("short_prefix_fraction", short_prefix_fraction),
        ("span_fraction", span_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if zero_prefix_fraction + short_prefix_fraction > 1.0:
        raise ValueError(
            "zero-prefix and short-prefix fractions cannot sum above one"
        )
    if short_prefix_max_tokens <= 0:
        raise ValueError("short_prefix_max_tokens must be positive")

    shifted = teacher_forced_output_inputs(
        batch,
        blank_token_id=blank_token_id,
    )
    eligible = torch.zeros_like(batch.target_mask)
    if eligible.shape[1] > 1:
        eligible[:, 1:] = batch.target_mask[:, :-1]
    visible = eligible.clone()
    batch_size = shifted.shape[0]
    row_draw = torch.rand(batch_size, device=shifted.device)
    zero_rows = row_draw.lt(zero_prefix_fraction)
    short_rows = row_draw.ge(zero_prefix_fraction) & row_draw.lt(
        zero_prefix_fraction + short_prefix_fraction
    )
    ordinary_rows = ~(zero_rows | short_rows)
    span_rows = ordinary_rows & torch.rand(
        batch_size, device=shifted.device
    ).lt(span_fraction)
    token_rows = ordinary_rows & ~span_rows

    if bool(token_rows.any()):
        token_keep = torch.rand(
            shifted.shape,
            device=shifted.device,
        ).lt(retention)
        visible[token_rows] &= token_keep[token_rows]

    for row in span_rows.nonzero(as_tuple=False).flatten().tolist():
        eligible_slots = eligible[row].nonzero(
            as_tuple=False
        ).flatten()
        eligible_count = int(eligible_slots.numel())
        drop_count = min(
            eligible_count,
            max(0, int(round((1.0 - retention) * eligible_count))),
        )
        if drop_count == 0:
            continue
        start_limit = eligible_count - drop_count + 1
        start = int(
            torch.randint(
                start_limit,
                (),
                device=shifted.device,
            )
        )
        visible[row, eligible_slots[start : start + drop_count]] = False

    visible[zero_rows] = False
    for row in short_rows.nonzero(as_tuple=False).flatten().tolist():
        eligible_slots = eligible[row].nonzero(
            as_tuple=False
        ).flatten()
        maximum = min(short_prefix_max_tokens, int(eligible_slots.numel()))
        if maximum <= 0:
            visible[row] = False
            continue
        keep_count = int(
            torch.randint(
                1,
                maximum + 1,
                (),
                device=shifted.device,
            )
        )
        visible[row] = False
        visible[row, eligible_slots[:keep_count]] = True

    inputs = torch.where(
        visible,
        shifted,
        torch.full_like(shifted, blank_token_id),
    )
    eligible_count = int(eligible.sum())
    visible_count = int(visible.sum())
    statistics = {
        "requested_retention": float(retention),
        "actual_retention": visible_count / max(eligible_count, 1),
        "eligible_tokens": float(eligible_count),
        "visible_tokens": float(visible_count),
        "zero_rows": float(zero_rows.sum()),
        "short_rows": float(short_rows.sum()),
        "span_rows": float(span_rows.sum()),
        "token_rows": float(token_rows.sum()),
        "batch_rows": float(batch_size),
    }
    return inputs, statistics


def deterministic_prefix_evaluation_inputs(
    batch: LanguageBatch,
    *,
    blank_token_id: int,
    retention: float,
    row_offset: int = 0,
    seed: int = 0,
) -> torch.Tensor:
    """Return a reproducible partial prefix for paired context probes."""

    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must be in [0, 1]")
    shifted = teacher_forced_output_inputs(
        batch,
        blank_token_id=blank_token_id,
    )
    eligible = torch.zeros_like(batch.target_mask)
    if eligible.shape[1] > 1:
        eligible[:, 1:] = batch.target_mask[:, :-1]
    rows = (
        torch.arange(
            shifted.shape[0],
            device=shifted.device,
            dtype=torch.int64,
        )
        + int(row_offset)
    )[:, None]
    slots = torch.arange(
        shifted.shape[1],
        device=shifted.device,
        dtype=torch.int64,
    )[None, :]
    # Integer hashing avoids touching the training RNG and produces the exact
    # same support mask for correct-context and shuffled-context evaluations.
    hashed = (
        rows * 1_103_515_245
        + slots * 12_345
        + int(seed) * 2_654_435_761
    ) & 0x7FFFFFFF
    threshold = int(round(retention * 0x80000000))
    visible = eligible & hashed.lt(threshold)
    return torch.where(
        visible,
        shifted,
        torch.full_like(shifted, blank_token_id),
    )


class PrefixDenoisingController:
    """Remove causal prefix support gradually and respond to context utility.

    A fixed warmup lowers retention from ``start_retention`` to
    ``floor_retention`` even before the model demonstrates semantic prompt
    use.  Further removal requires both teacher accuracy and a positive
    correct-prompt versus shuffled-prompt accuracy margin.  If either signal
    regresses, the adaptive portion of teacher support returns.
    """

    def __init__(
        self,
        *,
        start_retention: float = 0.95,
        floor_retention: float = 0.50,
        warmup_steps: int = 100_000,
        start_accuracy: float = 0.60,
        end_accuracy: float = 0.90,
        start_context_utility: float = 0.02,
        end_context_utility: float = 0.10,
        ema_decay: float = 0.50,
    ) -> None:
        if not 0.0 <= floor_retention <= start_retention <= 1.0:
            raise ValueError(
                "prefix retention must satisfy "
                "0 <= floor <= start <= 1"
            )
        if warmup_steps <= 0:
            raise ValueError("prefix denoising warmup steps must be positive")
        if not 0.0 <= start_accuracy < end_accuracy <= 1.0:
            raise ValueError(
                "prefix denoising accuracies must satisfy "
                "0 <= start < end <= 1"
            )
        if not (
            0.0
            <= start_context_utility
            < end_context_utility
            <= 1.0
        ):
            raise ValueError(
                "context utility thresholds must satisfy "
                "0 <= start < end <= 1"
            )
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("prefix denoising EMA decay must be in [0, 1)")
        self.start_retention = float(start_retention)
        self.floor_retention = float(floor_retention)
        self.warmup_steps = int(warmup_steps)
        self.start_accuracy = float(start_accuracy)
        self.end_accuracy = float(end_accuracy)
        self.start_context_utility = float(start_context_utility)
        self.end_context_utility = float(end_context_utility)
        self.ema_decay = float(ema_decay)
        self.content_accuracy_ema: float | None = None
        self.context_utility_ema: float | None = None

    @property
    def readiness(self) -> float:
        if (
            self.content_accuracy_ema is None
            or self.context_utility_ema is None
        ):
            return 0.0
        accuracy_readiness = (
            self.content_accuracy_ema - self.start_accuracy
        ) / (self.end_accuracy - self.start_accuracy)
        utility_readiness = (
            self.context_utility_ema - self.start_context_utility
        ) / (
            self.end_context_utility - self.start_context_utility
        )
        return max(
            0.0,
            min(1.0, accuracy_readiness, utility_readiness),
        )

    def retention(self, completed_step: int) -> float:
        if completed_step < 0:
            raise ValueError("completed_step cannot be negative")
        warmup = min(completed_step / self.warmup_steps, 1.0)
        scheduled = self.start_retention + warmup * (
            self.floor_retention - self.start_retention
        )
        return scheduled * (1.0 - self.readiness)

    def observe(
        self,
        *,
        content_accuracy: float,
        shuffled_context_accuracy: float,
        supported_context_accuracy: float | None = None,
    ) -> float:
        supported_context_accuracy = (
            content_accuracy
            if supported_context_accuracy is None
            else supported_context_accuracy
        )
        for name, value in (
            ("content accuracy", content_accuracy),
            ("supported-context accuracy", supported_context_accuracy),
            ("shuffled-context accuracy", shuffled_context_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        utility = max(
            -1.0,
            min(
                1.0,
                supported_context_accuracy
                - shuffled_context_accuracy,
            ),
        )
        if self.content_accuracy_ema is None:
            self.content_accuracy_ema = float(content_accuracy)
            self.context_utility_ema = float(utility)
        else:
            self.content_accuracy_ema = (
                self.ema_decay * self.content_accuracy_ema
                + (1.0 - self.ema_decay) * float(content_accuracy)
            )
            assert self.context_utility_ema is not None
            self.context_utility_ema = (
                self.ema_decay * self.context_utility_ema
                + (1.0 - self.ema_decay) * float(utility)
            )
        return self.readiness

    def state_dict(self) -> dict:
        return {
            "start_retention": self.start_retention,
            "floor_retention": self.floor_retention,
            "warmup_steps": self.warmup_steps,
            "start_accuracy": self.start_accuracy,
            "end_accuracy": self.end_accuracy,
            "start_context_utility": self.start_context_utility,
            "end_context_utility": self.end_context_utility,
            "ema_decay": self.ema_decay,
            "content_accuracy_ema": self.content_accuracy_ema,
            "context_utility_ema": self.context_utility_ema,
            "readiness": self.readiness,
        }

    def load_state_dict(self, state: dict) -> None:
        for key, expected in (
            ("start_retention", self.start_retention),
            ("floor_retention", self.floor_retention),
            ("warmup_steps", self.warmup_steps),
            ("start_accuracy", self.start_accuracy),
            ("end_accuracy", self.end_accuracy),
            ("start_context_utility", self.start_context_utility),
            ("end_context_utility", self.end_context_utility),
            ("ema_decay", self.ema_decay),
        ):
            observed = type(expected)(state.get(key, expected))
            if observed != expected:
                raise ValueError(
                    f"prefix denoising {key} differs from checkpoint: "
                    f"{observed} != {expected}"
                )
        observed_accuracy = state.get("content_accuracy_ema")
        observed_utility = state.get("context_utility_ema")
        self.content_accuracy_ema = (
            None
            if observed_accuracy is None
            else float(observed_accuracy)
        )
        self.context_utility_ema = (
            None if observed_utility is None else float(observed_utility)
        )


class AccuracyGatedTeacherHandoff:
    """Deterministically mix whole teacher and blank-parallel batches.

    The controller observes teacher-forced validation content accuracy. Below
    ``start_accuracy`` every training batch is causal teacher forcing. Between
    the two thresholds, teacher-batch frequency falls linearly. At or above
    ``end_accuracy`` every batch is blank-parallel. Evaluation continues in
    both modes, so a regression can restore teacher support.
    """

    def __init__(
        self,
        *,
        start_accuracy: float = 0.60,
        end_accuracy: float = 0.90,
        ema_decay: float = 0.50,
    ) -> None:
        if not 0.0 <= start_accuracy < end_accuracy <= 1.0:
            raise ValueError(
                "handoff accuracies must satisfy "
                "0 <= start < end <= 1"
            )
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("handoff EMA decay must be in [0, 1)")
        self.start_accuracy = float(start_accuracy)
        self.end_accuracy = float(end_accuracy)
        self.ema_decay = float(ema_decay)
        self.content_accuracy_ema: float | None = None
        self.batch_credit = 0.0
        self.teacher_batches = 0
        self.parallel_batches = 0

    @property
    def teacher_fraction(self) -> float:
        if self.content_accuracy_ema is None:
            return 1.0
        return max(
            0.0,
            min(
                1.0,
                (
                    self.end_accuracy
                    - self.content_accuracy_ema
                )
                / (self.end_accuracy - self.start_accuracy),
            ),
        )

    def observe(self, content_accuracy: float) -> float:
        """Update the validation EMA and return the next teacher fraction."""

        if not 0.0 <= content_accuracy <= 1.0:
            raise ValueError("content accuracy must be in [0, 1]")
        if self.content_accuracy_ema is None:
            self.content_accuracy_ema = float(content_accuracy)
        else:
            self.content_accuracy_ema = (
                self.ema_decay * self.content_accuracy_ema
                + (1.0 - self.ema_decay) * float(content_accuracy)
            )
        return self.teacher_fraction

    def choose_teacher_batch(self) -> bool:
        """Choose the next whole-batch mode using error diffusion."""

        teacher_fraction = self.teacher_fraction
        if teacher_fraction >= 1.0:
            teacher = True
        elif teacher_fraction <= 0.0:
            teacher = False
        else:
            self.batch_credit += teacher_fraction
            teacher = self.batch_credit >= 1.0
            if teacher:
                self.batch_credit -= 1.0
        if teacher:
            self.teacher_batches += 1
        else:
            self.parallel_batches += 1
        return teacher

    def state_dict(self) -> dict:
        return {
            "start_accuracy": self.start_accuracy,
            "end_accuracy": self.end_accuracy,
            "ema_decay": self.ema_decay,
            "content_accuracy_ema": self.content_accuracy_ema,
            "teacher_fraction": self.teacher_fraction,
            "batch_credit": self.batch_credit,
            "teacher_batches": self.teacher_batches,
            "parallel_batches": self.parallel_batches,
        }

    def load_state_dict(self, state: dict) -> None:
        for key, expected in (
            ("start_accuracy", self.start_accuracy),
            ("end_accuracy", self.end_accuracy),
            ("ema_decay", self.ema_decay),
        ):
            observed = float(state.get(key, expected))
            if not math.isclose(observed, expected):
                raise ValueError(
                    f"handoff {key} differs from checkpoint: "
                    f"{observed} != {expected}"
                )
        observed_accuracy = state.get("content_accuracy_ema")
        self.content_accuracy_ema = (
            None
            if observed_accuracy is None
            else float(observed_accuracy)
        )
        self.batch_credit = float(state.get("batch_credit", 0.0))
        self.teacher_batches = int(state.get("teacher_batches", 0))
        self.parallel_batches = int(state.get("parallel_batches", 0))


class GraduatedTeacherHandoff:
    """Increase unsupported runs as unsupported validation improves.

    The first ``baseline_fraction`` of the planned run alternates unsupported
    and teacher-forced batches. At the end of that window, unsupported
    content-accuracy EMA becomes the baseline. Every ``accuracy_step``
    percentage-point gain adds one more unsupported batch before the next
    teacher batch. Curriculum levels only move forward. Reaching
    ``release_accuracy`` permanently releases the teacher.
    """

    def __init__(
        self,
        *,
        planned_steps: int,
        baseline_fraction: float = 0.05,
        accuracy_step: float = 0.05,
        release_accuracy: float = 0.90,
        ema_decay: float = 0.50,
    ) -> None:
        if planned_steps <= 0:
            raise ValueError("graduated handoff planned steps must be positive")
        if not 0.0 < baseline_fraction < 1.0:
            raise ValueError(
                "graduated handoff baseline fraction must be in (0, 1)"
            )
        if not 0.0 < accuracy_step <= 1.0:
            raise ValueError(
                "graduated handoff accuracy step must be in (0, 1]"
            )
        if not 0.0 < release_accuracy <= 1.0:
            raise ValueError(
                "graduated handoff release accuracy must be in (0, 1]"
            )
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(
                "graduated handoff EMA decay must be in [0, 1)"
            )
        self.planned_steps = int(planned_steps)
        self.baseline_fraction = float(baseline_fraction)
        self.baseline_steps = max(
            1,
            math.ceil(self.planned_steps * self.baseline_fraction),
        )
        self.accuracy_step = float(accuracy_step)
        self.release_accuracy = float(release_accuracy)
        self.ema_decay = float(ema_decay)
        self.content_accuracy_ema: float | None = None
        self.baseline_accuracy: float | None = None
        self.max_gain_level = 0
        self.teacher_released = False
        self.cycle_position = 0
        self.teacher_batches = 0
        self.parallel_batches = 0

    @property
    def unsupported_burst(self) -> int:
        return 1 + self.max_gain_level

    @property
    def teacher_fraction(self) -> float:
        if self.teacher_released:
            return 0.0
        return 1.0 / (self.unsupported_burst + 1)

    def observe(
        self,
        content_accuracy: float,
        *,
        completed_step: int,
    ) -> float:
        """Observe fixed-validation unsupported content top-1 accuracy."""

        if not 0.0 <= content_accuracy <= 1.0:
            raise ValueError("content accuracy must be in [0, 1]")
        if completed_step < 0:
            raise ValueError("completed step cannot be negative")
        if self.content_accuracy_ema is None:
            self.content_accuracy_ema = float(content_accuracy)
        else:
            self.content_accuracy_ema = (
                self.ema_decay * self.content_accuracy_ema
                + (1.0 - self.ema_decay) * float(content_accuracy)
            )
        if (
            self.baseline_accuracy is None
            and completed_step >= self.baseline_steps
        ):
            self.baseline_accuracy = self.content_accuracy_ema
            self.cycle_position = 0
        if self.content_accuracy_ema >= self.release_accuracy:
            self.teacher_released = True
            self.cycle_position = 0
        elif self.baseline_accuracy is not None:
            gain = max(
                0.0,
                self.content_accuracy_ema - self.baseline_accuracy,
            )
            gain_level = int(
                math.floor((gain + 1e-12) / self.accuracy_step)
            )
            if gain_level > self.max_gain_level:
                self.max_gain_level = gain_level
                self.cycle_position = 0
        return self.teacher_fraction

    def choose_teacher_batch(self, *, completed_step: int) -> bool:
        """Choose one whole-batch mode using the current graduated cadence."""

        if completed_step <= 0:
            raise ValueError("completed step must be positive")
        if self.teacher_released:
            teacher = False
        elif completed_step <= self.baseline_steps:
            # Start with unsupported, then alternate exactly.
            teacher = completed_step % 2 == 0
        else:
            teacher = self.cycle_position >= self.unsupported_burst
            self.cycle_position = (
                0 if teacher else self.cycle_position + 1
            )
        if teacher:
            self.teacher_batches += 1
        else:
            self.parallel_batches += 1
        return teacher

    def state_dict(self) -> dict:
        return {
            "planned_steps": self.planned_steps,
            "baseline_fraction": self.baseline_fraction,
            "baseline_steps": self.baseline_steps,
            "accuracy_step": self.accuracy_step,
            "release_accuracy": self.release_accuracy,
            "ema_decay": self.ema_decay,
            "content_accuracy_ema": self.content_accuracy_ema,
            "baseline_accuracy": self.baseline_accuracy,
            "max_gain_level": self.max_gain_level,
            "unsupported_burst": self.unsupported_burst,
            "teacher_released": self.teacher_released,
            "teacher_fraction": self.teacher_fraction,
            "cycle_position": self.cycle_position,
            "teacher_batches": self.teacher_batches,
            "parallel_batches": self.parallel_batches,
        }

    def load_state_dict(self, state: dict) -> None:
        for key, expected in (
            ("planned_steps", self.planned_steps),
            ("baseline_fraction", self.baseline_fraction),
            ("baseline_steps", self.baseline_steps),
            ("accuracy_step", self.accuracy_step),
            ("release_accuracy", self.release_accuracy),
            ("ema_decay", self.ema_decay),
        ):
            observed = type(expected)(state.get(key, expected))
            if isinstance(expected, float):
                matches = math.isclose(observed, expected)
            else:
                matches = observed == expected
            if not matches:
                raise ValueError(
                    f"graduated handoff {key} differs from checkpoint: "
                    f"{observed} != {expected}"
                )
        observed_accuracy = state.get("content_accuracy_ema")
        observed_baseline = state.get("baseline_accuracy")
        self.content_accuracy_ema = (
            None
            if observed_accuracy is None
            else float(observed_accuracy)
        )
        self.baseline_accuracy = (
            None
            if observed_baseline is None
            else float(observed_baseline)
        )
        self.max_gain_level = int(state.get("max_gain_level", 0))
        self.teacher_released = bool(
            state.get("teacher_released", False)
        )
        self.cycle_position = int(state.get("cycle_position", 0))
        self.teacher_batches = int(state.get("teacher_batches", 0))
        self.parallel_batches = int(state.get("parallel_batches", 0))


class TokenwiseTeacherHandoff:
    """Remove one causal teacher token after sustained frontier mastery.

    Advancement is based on accuracy at the exact unsupported output slots,
    normalized by teacher-forced accuracy at those same addresses.  Separate
    advance and retreat thresholds provide hysteresis.  A small deterministic
    stream of fully unsupported batches keeps the final formation pathway
    alive throughout the curriculum.
    """

    def __init__(
        self,
        *,
        response_slots: int,
        start_removed_tokens: int = 1,
        maximum_removed_tokens: int | None = None,
        advance_ratio: float = 0.95,
        retreat_ratio: float = 0.75,
        minimum_teacher_accuracy: float = 0.15,
        minimum_frontier_accuracy: float = 0.0,
        advance_patience: int = 3,
        retreat_patience: int = 2,
        ema_decay: float = 0.50,
        full_probe_fraction: float = 0.02,
        maximum_teacher_perplexity_ratio: float | None = None,
        teacher_perplexity_retreat_ratio: float | None = None,
        teacher_perplexity_baseline: float | None = None,
    ) -> None:
        if response_slots <= 0:
            raise ValueError("response_slots must be positive")
        maximum = (
            response_slots - 1
            if maximum_removed_tokens is None
            else int(maximum_removed_tokens)
        )
        if not 0 <= start_removed_tokens <= maximum <= response_slots - 1:
            raise ValueError(
                "token handoff removals must satisfy "
                "0 <= start <= maximum <= response_slots - 1"
            )
        if not 0.0 < retreat_ratio < advance_ratio <= 1.0:
            raise ValueError(
                "token handoff ratios must satisfy "
                "0 < retreat < advance <= 1"
            )
        if not 0.0 <= minimum_teacher_accuracy <= 1.0:
            raise ValueError("minimum teacher accuracy must be in [0, 1]")
        if not 0.0 <= minimum_frontier_accuracy <= 1.0:
            raise ValueError("minimum frontier accuracy must be in [0, 1]")
        if advance_patience <= 0 or retreat_patience <= 0:
            raise ValueError("token handoff patience must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("token handoff EMA decay must be in [0, 1)")
        if not 0.0 <= full_probe_fraction <= 1.0:
            raise ValueError("full probe fraction must be in [0, 1]")
        guard_enabled = maximum_teacher_perplexity_ratio is not None
        if guard_enabled != (teacher_perplexity_retreat_ratio is not None):
            raise ValueError(
                "teacher perplexity advance and retreat ratios must be "
                "configured together"
            )
        if guard_enabled:
            maximum_ratio = float(maximum_teacher_perplexity_ratio)
            retreat_perplexity_ratio = float(
                teacher_perplexity_retreat_ratio
            )
            if not (
                math.isfinite(maximum_ratio)
                and math.isfinite(retreat_perplexity_ratio)
                and 1.0 <= maximum_ratio < retreat_perplexity_ratio
            ):
                raise ValueError(
                    "teacher perplexity ratios must satisfy "
                    "1 <= maximum < retreat"
                )
        else:
            maximum_ratio = None
            retreat_perplexity_ratio = None
        if teacher_perplexity_baseline is not None and not (
            math.isfinite(teacher_perplexity_baseline)
            and teacher_perplexity_baseline > 0.0
        ):
            raise ValueError(
                "teacher perplexity baseline must be finite and positive"
            )
        self.response_slots = int(response_slots)
        self.start_removed_tokens = int(start_removed_tokens)
        self.maximum_removed_tokens = maximum
        self.advance_ratio = float(advance_ratio)
        self.retreat_ratio = float(retreat_ratio)
        self.minimum_teacher_accuracy = float(minimum_teacher_accuracy)
        self.minimum_frontier_accuracy = float(minimum_frontier_accuracy)
        self.advance_patience = int(advance_patience)
        self.retreat_patience = int(retreat_patience)
        self.ema_decay = float(ema_decay)
        self.full_probe_fraction = float(full_probe_fraction)
        self.maximum_teacher_perplexity_ratio = maximum_ratio
        self.teacher_perplexity_retreat_ratio = retreat_perplexity_ratio
        self.teacher_perplexity_baseline = (
            None
            if teacher_perplexity_baseline is None
            else float(teacher_perplexity_baseline)
        )
        self.teacher_perplexity_ratio: float | None = None
        self.removed_tokens = self.start_removed_tokens
        self.accuracy_ratio_ema: float | None = None
        self.advance_streak = 0
        self.retreat_streak = 0
        self.advances = 0
        self.retreats = 0
        self.probe_credit = 0.0
        self.partial_batches = 0
        self.full_probe_batches = 0

    @property
    def visible_teacher_tokens(self) -> int:
        return max(0, self.response_slots - 1 - self.removed_tokens)

    @property
    def released(self) -> bool:
        return self.removed_tokens >= self.maximum_removed_tokens

    @property
    def retention_guard_enabled(self) -> bool:
        return self.maximum_teacher_perplexity_ratio is not None

    def training_removed_tokens(self) -> int:
        """Return this batch's removal level, including sparse full probes."""

        full_probe = False
        if self.full_probe_fraction >= 1.0:
            full_probe = True
        elif self.full_probe_fraction > 0.0:
            self.probe_credit += self.full_probe_fraction
            if self.probe_credit >= 1.0:
                self.probe_credit -= 1.0
                full_probe = True
        if full_probe:
            self.full_probe_batches += 1
            return self.response_slots - 1
        self.partial_batches += 1
        return self.removed_tokens

    def observe(
        self,
        *,
        unsupported_accuracy: float,
        teacher_accuracy: float,
        teacher_content_perplexity: float | None = None,
    ) -> str:
        """Update the frontier and return ``advance``, ``retreat``, or ``hold``."""

        for name, value in (
            ("unsupported_accuracy", unsupported_accuracy),
            ("teacher_accuracy", teacher_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if teacher_content_perplexity is not None and not (
            math.isfinite(teacher_content_perplexity)
            and teacher_content_perplexity > 0.0
        ):
            raise ValueError(
                "teacher_content_perplexity must be finite and positive"
            )
        if self.retention_guard_enabled:
            if teacher_content_perplexity is None:
                raise ValueError(
                    "teacher_content_perplexity is required when the "
                    "retention guard is enabled"
                )
            if self.teacher_perplexity_baseline is None:
                self.teacher_perplexity_baseline = float(
                    teacher_content_perplexity
                )
            self.teacher_perplexity_ratio = (
                float(teacher_content_perplexity)
                / self.teacher_perplexity_baseline
            )
        else:
            self.teacher_perplexity_ratio = None
        ratio = unsupported_accuracy / max(teacher_accuracy, 1e-9)
        ratio = min(ratio, 1.0)
        if self.accuracy_ratio_ema is None:
            self.accuracy_ratio_ema = ratio
        else:
            self.accuracy_ratio_ema = (
                self.ema_decay * self.accuracy_ratio_ema
                + (1.0 - self.ema_decay) * ratio
            )

        retention_ready = (
            not self.retention_guard_enabled
            or self.teacher_perplexity_ratio
            <= self.maximum_teacher_perplexity_ratio
        )
        retention_regressed = (
            self.retention_guard_enabled
            and self.teacher_perplexity_ratio
            >= self.teacher_perplexity_retreat_ratio
        )
        ready = (
            teacher_accuracy >= self.minimum_teacher_accuracy
            and unsupported_accuracy >= self.minimum_frontier_accuracy
            and self.accuracy_ratio_ema >= self.advance_ratio
            and retention_ready
        )
        regressed = (
            self.accuracy_ratio_ema < self.retreat_ratio
            or retention_regressed
        )
        self.advance_streak = self.advance_streak + 1 if ready else 0
        self.retreat_streak = self.retreat_streak + 1 if regressed else 0

        action = "hold"
        if (
            self.advance_streak >= self.advance_patience
            and self.removed_tokens < self.maximum_removed_tokens
        ):
            self.removed_tokens += 1
            self.advances += 1
            self.advance_streak = 0
            self.retreat_streak = 0
            self.accuracy_ratio_ema = None
            action = "advance"
        elif (
            self.retreat_streak >= self.retreat_patience
            and self.removed_tokens > self.start_removed_tokens
        ):
            self.removed_tokens -= 1
            self.retreats += 1
            self.advance_streak = 0
            self.retreat_streak = 0
            self.accuracy_ratio_ema = None
            action = "retreat"
        return action

    def state_dict(self) -> dict:
        return {
            "response_slots": self.response_slots,
            "start_removed_tokens": self.start_removed_tokens,
            "maximum_removed_tokens": self.maximum_removed_tokens,
            "advance_ratio": self.advance_ratio,
            "retreat_ratio": self.retreat_ratio,
            "minimum_teacher_accuracy": self.minimum_teacher_accuracy,
            "minimum_frontier_accuracy": self.minimum_frontier_accuracy,
            "advance_patience": self.advance_patience,
            "retreat_patience": self.retreat_patience,
            "ema_decay": self.ema_decay,
            "full_probe_fraction": self.full_probe_fraction,
            "maximum_teacher_perplexity_ratio": (
                self.maximum_teacher_perplexity_ratio
            ),
            "teacher_perplexity_retreat_ratio": (
                self.teacher_perplexity_retreat_ratio
            ),
            "teacher_perplexity_baseline": self.teacher_perplexity_baseline,
            "teacher_perplexity_ratio": self.teacher_perplexity_ratio,
            "removed_tokens": self.removed_tokens,
            "visible_teacher_tokens": self.visible_teacher_tokens,
            "accuracy_ratio_ema": self.accuracy_ratio_ema,
            "advance_streak": self.advance_streak,
            "retreat_streak": self.retreat_streak,
            "advances": self.advances,
            "retreats": self.retreats,
            "probe_credit": self.probe_credit,
            "partial_batches": self.partial_batches,
            "full_probe_batches": self.full_probe_batches,
        }

    def load_state_dict(
        self,
        state: dict,
        *,
        allow_policy_override: bool = False,
    ) -> None:
        for key, expected in (
            ("response_slots", self.response_slots),
            ("start_removed_tokens", self.start_removed_tokens),
            ("maximum_removed_tokens", self.maximum_removed_tokens),
            ("advance_ratio", self.advance_ratio),
            ("retreat_ratio", self.retreat_ratio),
            ("minimum_teacher_accuracy", self.minimum_teacher_accuracy),
            ("minimum_frontier_accuracy", self.minimum_frontier_accuracy),
            ("advance_patience", self.advance_patience),
            ("retreat_patience", self.retreat_patience),
            ("ema_decay", self.ema_decay),
            ("full_probe_fraction", self.full_probe_fraction),
        ):
            observed = type(expected)(state.get(key, expected))
            matches = (
                math.isclose(observed, expected)
                if isinstance(expected, float)
                else observed == expected
            )
            if not matches and not allow_policy_override:
                raise ValueError(
                    f"token handoff {key} differs from checkpoint: "
                    f"{observed} != {expected}"
                )
        for key, expected in (
            (
                "maximum_teacher_perplexity_ratio",
                self.maximum_teacher_perplexity_ratio,
            ),
            (
                "teacher_perplexity_retreat_ratio",
                self.teacher_perplexity_retreat_ratio,
            ),
        ):
            observed = state.get(key, expected)
            observed = None if observed is None else float(observed)
            matches = (
                observed is None
                if expected is None
                else observed is not None
                and math.isclose(observed, expected)
            )
            if not matches and not allow_policy_override:
                raise ValueError(
                    f"token handoff {key} differs from checkpoint: "
                    f"{observed} != {expected}"
                )
        observed_teacher_baseline = state.get(
            "teacher_perplexity_baseline",
            self.teacher_perplexity_baseline,
        )
        self.teacher_perplexity_baseline = (
            None
            if observed_teacher_baseline is None
            else float(observed_teacher_baseline)
        )
        observed_teacher_ratio = state.get("teacher_perplexity_ratio")
        self.teacher_perplexity_ratio = (
            None
            if observed_teacher_ratio is None
            else float(observed_teacher_ratio)
        )
        self.removed_tokens = int(
            state.get("removed_tokens", self.start_removed_tokens)
        )
        observed_ratio = state.get("accuracy_ratio_ema")
        self.accuracy_ratio_ema = (
            None if observed_ratio is None else float(observed_ratio)
        )
        self.advance_streak = int(state.get("advance_streak", 0))
        self.retreat_streak = int(state.get("retreat_streak", 0))
        self.advances = int(state.get("advances", 0))
        self.retreats = int(state.get("retreats", 0))
        self.probe_credit = float(state.get("probe_credit", 0.0))
        self.partial_batches = int(state.get("partial_batches", 0))
        self.full_probe_batches = int(state.get("full_probe_batches", 0))


def explicit_cli_destinations(
    parser: argparse.ArgumentParser,
    arguments: list[str],
) -> set[str]:
    """Return argparse destinations explicitly selected on the command line."""

    option_destinations = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    explicit: set[str] = set()
    for argument in arguments:
        option = argument.split("=", 1)[0]
        destination = option_destinations.get(option)
        if destination is not None:
            explicit.add(destination)
    return explicit


def apply_execution_preset(
    args: argparse.Namespace,
    *,
    explicit_destinations: set[str] | None = None,
) -> argparse.Namespace:
    """Apply one explicit parameter-free trainer execution policy.

    A preset supplies cohesive defaults. Explicit low-level CLI options win,
    allowing one runtime lever to be isolated without restating the policy.
    """

    preset = args.execution_preset
    try:
        overrides = _EXECUTION_PRESET_OVERRIDES[preset]
    except KeyError as error:
        raise ValueError(f"unsupported execution preset: {preset}") from error
    explicit_destinations = explicit_destinations or set()
    for name, value in overrides.items():
        if name not in explicit_destinations:
            setattr(args, name, value)
    return args


def slim_step_log_line(
    *,
    step: int,
    steps_per_second: float,
    result: dict,
) -> str:
    """Return the compact operator view of validation progress."""

    aligned = result["formation_objective"] in {"viterbi", "monotonic"}
    accuracy = (
        result["aligned_required_accuracy"]
        if aligned
        else result["accuracy"]
    )
    exact = result["aligned_exact"] if aligned else result["exact"]
    loss = result["selection_loss"]
    perplexity = math.exp(min(loss, 20.0))
    return (
        f"step={step} steps_per_second={steps_per_second:.2f} "
        f"loss={loss:.4f} ppl={perplexity:.2f} "
        f"acc={accuracy:.3f} top5={result['top5_accuracy']:.3f} "
        f"content_acc={result['content_accuracy']:.3f} "
        f"content_ppl={result['content_perplexity']:.2f} "
        f"content_top5={result['content_top5_accuracy']:.3f} "
        f"coverage={result['formation_coverage']:.3f} "
        f"ws_omission={result['whitespace_omission_rate']:.3f} "
        f"exact={exact:.3f}"
    )


def handoff_log_line(
    *,
    controller: AccuracyGatedTeacherHandoff | GraduatedTeacherHandoff,
    teacher_result: dict,
    parallel_result: dict,
) -> str:
    """Return the compact teacher-to-free formation controller state."""

    ema = controller.content_accuracy_ema
    graduated = ""
    if isinstance(controller, GraduatedTeacherHandoff):
        baseline = controller.baseline_accuracy
        graduated = (
            f" baseline_acc="
            f"{'pending' if baseline is None else f'{baseline:.3f}'} "
            f"unsupported_burst={controller.unsupported_burst} "
            f"released={controller.teacher_released}"
        )
    return (
        f"handoff teacher_fraction={controller.teacher_fraction:.3f} "
        f"teacher_content_acc={teacher_result['content_accuracy']:.3f} "
        f"teacher_content_ppl={teacher_result['content_perplexity']:.2f} "
        f"parallel_content_acc={parallel_result['content_accuracy']:.3f} "
        f"parallel_content_ppl={parallel_result['content_perplexity']:.2f} "
        f"accuracy_ema={0.0 if ema is None else ema:.3f} "
        f"batches={controller.teacher_batches}:"
        f"{controller.parallel_batches}"
        f"{graduated}"
    )


def token_handoff_log_line(
    *,
    controller: TokenwiseTeacherHandoff,
    partial_result: dict,
    teacher_result: dict,
    unsupported_result: dict,
    action: str,
) -> str:
    """Return the compact one-token-at-a-time curriculum state."""

    ratio = controller.accuracy_ratio_ema
    teacher_perplexity_ratio = controller.teacher_perplexity_ratio
    retention = "disabled"
    if controller.retention_guard_enabled:
        retention = (
            "pending"
            if teacher_perplexity_ratio is None
            else f"{teacher_perplexity_ratio:.3f}"
        )
    return (
        "token_handoff "
        f"teacher_visible={controller.visible_teacher_tokens}/"
        f"{controller.response_slots - 1} "
        f"removed={controller.removed_tokens} "
        f"action={action} "
        f"frontier_acc={partial_result['unsupported_content_accuracy']:.3f} "
        f"teacher_frontier_acc="
        f"{teacher_result['unsupported_content_accuracy']:.3f} "
        f"teacher_content_ppl={teacher_result['content_perplexity']:.2f} "
        f"retention_ratio={retention} "
        f"ratio_ema={'pending' if ratio is None else f'{ratio:.3f}'} "
        f"free_content_acc={unsupported_result['content_accuracy']:.3f} "
        f"free_content_top5="
        f"{unsupported_result['content_top5_accuracy']:.3f} "
        f"free_content_ppl={unsupported_result['content_perplexity']:.2f} "
        f"free_coverage={unsupported_result['formation_coverage']:.3f} "
        f"streaks={controller.advance_streak}:{controller.retreat_streak} "
        f"moves={controller.advances}:{controller.retreats} "
        f"batches={controller.partial_batches}:"
        f"{controller.full_probe_batches}"
    )


def prefix_denoising_log_line(
    *,
    controller: PrefixDenoisingController,
    completed_step: int,
    corruption: dict[str, float],
    teacher_result: dict,
    supported_context_result: dict,
    shuffled_context_result: dict,
) -> str:
    """Return the compact causal-support and context-utility state."""

    accuracy_ema = controller.content_accuracy_ema
    utility_ema = controller.context_utility_ema
    context_utility = (
        supported_context_result["content_accuracy"]
        - shuffled_context_result["content_accuracy"]
    )
    return (
        "prefix_denoise "
        f"retention={controller.retention(completed_step):.3f} "
        f"actual={corruption['actual_retention']:.3f} "
        f"rows=zero:{int(corruption['zero_rows'])}/"
        f"short:{int(corruption['short_rows'])}/"
        f"span:{int(corruption['span_rows'])}/"
        f"token:{int(corruption['token_rows'])} "
        f"teacher_content_acc={teacher_result['content_accuracy']:.3f} "
        f"teacher_content_ppl={teacher_result['content_perplexity']:.2f} "
        f"supported_content_acc="
        f"{supported_context_result['content_accuracy']:.3f} "
        f"supported_content_ppl="
        f"{supported_context_result['content_perplexity']:.2f} "
        f"shuffled_content_acc="
        f"{shuffled_context_result['content_accuracy']:.3f} "
        f"shuffled_content_ppl="
        f"{shuffled_context_result['content_perplexity']:.2f} "
        f"context_utility={context_utility:+.3f} "
        f"accuracy_ema={0.0 if accuracy_ema is None else accuracy_ema:.3f} "
        f"utility_ema={0.0 if utility_ema is None else utility_ema:+.3f} "
        f"readiness={controller.readiness:.3f}"
    )


def _compact_route_values(values, *, digits: int = 2) -> str:
    return "[" + ",".join(
        f"{float(value):.{digits}f}" for value in values
    ) + "]"


def slim_routing_log_line(stewardship: dict) -> str:
    """Return the compact routing view needed during ordinary training."""

    packed = "[" + ",".join(
        "1" if bool(value) else "0"
        for value in stewardship["packed_execution_by_relay"]
    ) + "]"
    return (
        "routing "
        f"support={_compact_route_values(stewardship['support_by_relay'])} "
        f"effective={_compact_route_values(stewardship['effective_cells_by_relay'])} "
        f"beam={_compact_route_values(stewardship['handoff_support_by_relay'])} "
        f"stay={_compact_route_values(stewardship['stay_rate_by_relay'])} "
        f"dead={_compact_route_values(stewardship['dead_cell_fraction_by_relay'])} "
        f"packed={packed}"
    )


def terminal_vote_summary(
    terminal_weights: torch.Tensor,
    *,
    batch_size: int,
    output_slots: int,
) -> dict[str, float]:
    """Summarize static, prompt, or per-slot terminal lane coalitions."""

    if terminal_weights.ndim == 1:
        expanded = terminal_weights[None, None].expand(
            batch_size,
            output_slots,
            -1,
        )
    elif terminal_weights.ndim == 2:
        if terminal_weights.shape[0] != batch_size:
            raise ValueError("prompt terminal weights have wrong batch axis")
        expanded = terminal_weights[:, None].expand(
            -1,
            output_slots,
            -1,
        )
    elif terminal_weights.ndim == 3:
        if terminal_weights.shape[:2] != (batch_size, output_slots):
            raise ValueError("slot terminal weights have wrong batch/slot axes")
        expanded = terminal_weights
    else:
        raise ValueError("terminal weights must have one, two, or three axes")

    probabilities = expanded.detach().to(dtype=torch.float64)
    entropy = -(
        probabilities.clamp_min(1e-12)
        * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    winners = probabilities.argmax(dim=-1)
    winner_support = F.one_hot(
        winners,
        num_classes=probabilities.shape[-1],
    ).any(dim=1).sum(dim=-1)
    adjacent_delta = (
        (probabilities[:, 1:] - probabilities[:, :-1]).abs().mean()
        if output_slots > 1
        else probabilities.new_zeros(())
    )
    return {
        "effective_voters": float(entropy.mean().exp()),
        "top1_mass": float(probabilities.amax(dim=-1).mean()),
        "adjacent_slot_delta": float(adjacent_delta),
        "winning_lanes_per_response": float(
            winner_support.to(torch.float64).mean()
        ),
    }


def lane_balance_loss_from_terminal_weights(
    terminal_weights: torch.Tensor,
) -> torch.Tensor:
    """Penalize terminal lane-vote collapse by maximizing vote entropy.

    Differentiable counterpart to the diagnostic entropy computed in
    ``terminal_vote_summary``. Loss is minimized (entropy maximized) as
    training pushes lane usage away from a permanent single-lane winner.
    """

    probabilities = terminal_weights.clamp_min(1e-9)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    return -entropy.mean()


def slim_terminal_vote_log_line(summary: dict[str, float]) -> str:
    """Return compact formation-controller diagnostics."""

    line = (
        "terminal "
        f"effective={summary['effective_voters']:.2f} "
        f"top1={summary['top1_mass']:.3f} "
        f"slot_delta={summary['adjacent_slot_delta']:.4f} "
        f"winning_lanes={summary['winning_lanes_per_response']:.2f} "
        f"query_rms={summary.get('query_rms', 0.0):.6f}"
    )
    if "route_k_by_relay" in summary:
        route_k = ",".join(
            f"{value:.2f}" for value in summary["route_k_by_relay"]
        )
        line += f" route_k=[{route_k}]"
    if "route_uncertainty_by_relay" in summary:
        uncertainty = ",".join(
            f"{value:.3f}"
            for value in summary["route_uncertainty_by_relay"]
        )
        line += f" route_uncertainty=[{uncertainty}]"
    if "active_node_union_fraction" in summary:
        line += (
            f" node_union={summary['active_node_union_fraction']:.3f}"
        )
    return line


def formation_loss_components(
    logits: torch.Tensor,
    batch,
    tokenizer: CompactTokenizer,
    *,
    valid_token_mask: torch.Tensor,
    semantic_token_mask: torch.Tensor,
    objective: str,
    alignment_mix: float,
    max_right_shift: int | None,
    sequence_composition_weight: float,
    sequence_length_weight: float,
    sequence_eos_weight: float,
    score_censored_blank_tail: bool = False,
    compute_all: bool = False,
    compute_sequence: bool = True,
) -> dict:
    """Compose stage-selectable formation losses from generic alignment APIs."""

    blank_id = tokenizer.blank_token_id
    if blank_id is None:
        raise ValueError("formation objectives require a tokenizer with BLANK")
    if valid_token_mask.ndim != 1 or valid_token_mask.shape[0] != logits.shape[-1]:
        raise ValueError("valid_token_mask must have one entry per output class")
    valid_token_mask = valid_token_mask.to(device=logits.device, dtype=torch.bool)
    log_probs = F.log_softmax(
        logits.masked_fill(~valid_token_mask[None, None], -torch.inf), dim=-1
    )
    slot_ce = (
        masked_cross_entropy(
            logits,
            batch.target_ids,
            batch.target_mask,
            precomputed_log_probs=log_probs,
        )
        if compute_all or objective == "slot_ce"
        else None
    )
    need_blank_ce = (
        compute_all
        or objective == "blank_ce"
        or (objective in {"viterbi", "monotonic"} and alignment_mix < 1.0)
    )
    blank_labels = blank_mask = blank_ce = None
    if need_blank_ce:
        blank_labels, blank_mask = rigid_blank_tail_targets(
            batch.target_ids,
            batch.target_mask,
            batch.target_complete,
            blank_token_id=blank_id,
            score_censored_tail=score_censored_blank_tail,
        )
        blank_ce = masked_cross_entropy(
            logits,
            blank_labels,
            blank_mask,
            precomputed_log_probs=log_probs,
        )

    viterbi_result = None
    monotonic_loss = None
    monotonic_per_example = None
    monotonic_required_per_example = None
    if compute_all or objective == "viterbi":
        viterbi_result = viterbi_aligned_cross_entropy(
            logits,
            batch.target_ids,
            batch.target_mask,
            blank_id=blank_id,
            eos_id=tokenizer.EOS_ID,
            complete=batch.target_complete,
            valid_token_mask=valid_token_mask,
            precomputed_log_probs=log_probs,
            max_right_shift=max_right_shift,
        )
    if compute_all or objective == "monotonic":
        monotonic_path_nll = monotonic_forward_sum_nll(
            logits,
            batch.target_ids,
            batch.target_mask,
            blank_id=blank_id,
            eos_id=tokenizer.EOS_ID,
            complete=batch.target_complete,
            valid_token_mask=valid_token_mask,
            precomputed_log_probs=log_probs,
            max_right_shift=max_right_shift,
            normalize=False,
            reduction="none",
        )
        monotonic_per_example = monotonic_path_nll / logits.shape[1]
        monotonic_required_per_example = monotonic_path_nll / batch.target_mask.sum(
            1
        ).clamp_min(1).to(monotonic_path_nll.dtype)
        monotonic_loss = monotonic_per_example.mean()

    if objective == "slot_ce":
        assert slot_ce is not None
        selected = slot_ce
    elif objective == "blank_ce":
        assert blank_ce is not None
        selected = blank_ce
    elif objective == "viterbi":
        assert viterbi_result is not None
        if alignment_mix < 1.0:
            assert blank_ce is not None
            selected = (
                (1.0 - alignment_mix) * blank_ce
                + alignment_mix * viterbi_result.loss
            )
        else:
            selected = viterbi_result.loss
    elif objective == "monotonic":
        assert monotonic_loss is not None
        if alignment_mix < 1.0:
            assert blank_ce is not None
            selected = (
                (1.0 - alignment_mix) * blank_ce
                + alignment_mix * monotonic_loss
            )
        else:
            selected = monotonic_loss
    else:
        raise ValueError(f"unknown formation objective: {objective}")

    sequence = None
    if compute_sequence:
        sequence = sequence_quality_loss(
            logits,
            batch.target_ids,
            batch.target_mask,
            blank_id=blank_id,
            eos_id=tokenizer.EOS_ID,
            complete=batch.target_complete,
            valid_token_mask=valid_token_mask,
            precomputed_log_probs=log_probs,
            semantic_token_mask=semantic_token_mask,
            composition_weight=sequence_composition_weight,
            length_weight=sequence_length_weight,
            eos_weight=sequence_eos_weight,
        )
    return {
        "slot_ce": slot_ce,
        "blank_ce": blank_ce,
        "viterbi": None if viterbi_result is None else viterbi_result.loss,
        "viterbi_result": viterbi_result,
        "monotonic": monotonic_loss,
        "monotonic_per_example": monotonic_per_example,
        "monotonic_required_per_example": monotonic_required_per_example,
        "selected": selected,
        "blank_labels": blank_labels,
        "blank_mask": blank_mask,
        "sequence": sequence,
        "log_probs": log_probs,
    }


def token_handoff_cross_entropy(
    log_probs: torch.Tensor,
    target_ids: torch.Tensor,
    unsupported_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    normalization: str = "released",
) -> torch.Tensor:
    """Score only released teacher slots without touching forbidden PAD IDs."""

    if normalization not in {"released", "active_slots"}:
        raise ValueError(
            "token handoff loss normalization must be released or active_slots"
        )
    slot_mask = unsupported_mask.bool() & target_mask.bool()
    token_nll = -log_probs.gather(2, target_ids[..., None]).squeeze(2)
    # PAD labels are intentionally forbidden and therefore have +inf NLL.
    # ``where`` must precede reduction: arithmetic masking would form 0 * inf.
    selected_nll = torch.where(
        slot_mask,
        token_nll,
        torch.zeros_like(token_nll),
    )
    denominator_mask = (
        slot_mask if normalization == "released" else target_mask.bool()
    )
    return selected_nll.sum() / denominator_mask.sum().clamp_min(1).to(
        selected_nll.dtype
    )


def optimizer_parameter_component(name: str) -> str:
    """Map one Convergent parameter to its architectural optimizer group."""

    if any(
        marker in name
        for marker in (
            "router",
            "ingress",
            "terminal_vote",
            "terminal_address",
            "address_to_edges",
        )
    ):
        return "route_gate"
    if any(
        marker in name
        for marker in (
            "node_embedding",
            "node_weight",
            "node_bias",
            "shared_node_adapters",
        )
    ):
        return "private_node"
    if any(
        marker in name
        for marker in (
            "token_embedding",
            "input_projection",
            "readout_projection",
            "lm_bias",
        )
    ):
        return "lexical"
    return "lane_core"


def is_route_transport_parameter(name: str) -> bool:
    """Return whether a parameter changes packet transport rather than verdicts."""

    return any(
        marker in name
        for marker in (
            "router",
            "ingress",
            "address_to_edges",
        )
    )


OPTIMIZER_COMPONENTS = (
    "lexical",
    "lane_core",
    "private_node",
    "route_gate",
)


def component_optimizer_parameter_groups(
    model: torch.nn.Module,
    optimization,
    multipliers: dict[str, float],
    *,
    train_components: tuple[str, ...] | None = None,
    retain_frozen_state: bool = False,
    freeze_route_transport: bool = False,
) -> list[dict]:
    """Build ratio-preserving AdamW groups for Convergent components."""

    components = OPTIMIZER_COMPONENTS
    if set(multipliers) != set(components):
        raise ValueError("component optimizer multipliers are incomplete")
    if any(value <= 0 or not math.isfinite(value) for value in multipliers.values()):
        raise ValueError("component optimizer multipliers must be finite and positive")
    active_components = set(
        components if train_components is None else train_components
    )
    if not active_components or not active_components.issubset(components):
        raise ValueError(
            "train_components must be a non-empty subset of optimizer components"
        )
    grouped = {component: [] for component in components}
    for name, parameter in model.named_parameters():
        component = optimizer_parameter_component(name)
        component_active = component in active_components
        parameter_active = component_active and not (
            freeze_route_transport
            and is_route_transport_parameter(name)
        )
        parameter.requires_grad_(parameter_active)
        # Keeping frozen parameters in their optimizer groups preserves their
        # existing AdamW moments without applying updates: grad remains None,
        # so AdamW neither advances moments nor applies weight decay.  Route
        # transport stays in the active route/gate group for the same reason,
        # while terminal-jury parameters remain trainable.
        if component_active or retain_frozen_state:
            grouped[component].append(parameter)
    parameter_groups = []
    for component in components:
        parameters = grouped[component]
        if not parameters:
            continue
        multiplier = float(multipliers[component])
        parameter_groups.append(
            {
                "params": parameters,
                "lr": optimization.learning_rate * multiplier,
                "lr_multiplier": multiplier,
                "weight_decay": optimization.weight_decay,
                "group_label": component,
            }
        )
    return parameter_groups


def staged_route_temperature(
    *,
    local_step: int,
    planned_steps: int,
    start: float,
    middle: float,
    end: float,
    hold_fraction: float,
    middle_fraction: float,
) -> float:
    """Return the branch-local geometric reheat/cool temperature."""

    progress = min(
        max((local_step - 1) / max(planned_steps - 1, 1), 0.0),
        1.0,
    )
    if progress <= hold_fraction:
        return start
    if progress <= middle_fraction:
        middle_progress = (
            (progress - hold_fraction)
            / (middle_fraction - hold_fraction)
        )
        return start * (middle / start) ** middle_progress
    end_progress = (
        (progress - middle_fraction)
        / (1.0 - middle_fraction)
    )
    return middle * (end / middle) ** end_progress


def _stratified_validation_slice(
    examples: Sequence[EncodedAssistantExample],
    limit: int,
) -> Sequence[EncodedAssistantExample]:
    """Return an even, deterministic per-source slice of examples.

    ``examples`` is typically built as several sources concatenated in
    sequence (e.g. TinyStories, then primary, then LMSYS) - a plain
    ``examples[:limit]`` truncation would silently draw entirely from
    whichever source happens to be listed first once that source alone
    exceeds ``limit``. This groups by ``.source`` and takes an even share
    from each group instead, so every contributing corpus is represented
    in evaluation regardless of how many examples it happens to supply.
    """

    if limit <= 0 or limit >= len(examples):
        return examples
    by_source: dict[str, list] = {}
    order: list[str] = []
    for example in examples:
        if example.source not in by_source:
            by_source[example.source] = []
            order.append(example.source)
        by_source[example.source].append(example)
    source_count = len(order)
    base_share, remainder = divmod(limit, source_count)
    selected: list = []
    for index, source in enumerate(order):
        take = base_share + (1 if index < remainder else 0)
        selected.extend(by_source[source][:take])
    return selected


@torch.no_grad()
def evaluate(
    model: GatorLanguageModel,
    batcher: LanguageBatcher,
    tokenizer: CompactTokenizer,
    device: torch.device,
    *,
    batch_size: int,
    limit: int,
    display_index: int,
    stewardship_temperature: float | None = None,
    stewardship_prune: bool = True,
    stewardship_packed_from_relay: int | None = None,
    formation_objective: str = "slot_ce",
    alignment_max_shift: int | None = 4,
    sequence_composition_weight: float = 1.0,
    sequence_length_weight: float = 0.25,
    sequence_eos_weight: float = 0.1,
    decode_mode: str = "parallel",
    chunk_size: int = 8,
    context_mode: str = "correct",
    teacher_prefix_retention: float = 1.0,
    teacher_removed_tokens: int = 0,
    reasoning_loops: int = 1,
    mixed_precision: str = "none",
) -> dict:
    if context_mode not in {"correct", "shuffled", "masked"}:
        raise ValueError(
            "context_mode must be correct, shuffled, or masked"
        )
    if not 0.0 <= teacher_prefix_retention <= 1.0:
        raise ValueError("teacher_prefix_retention must be in [0, 1]")
    if teacher_removed_tokens < 0:
        raise ValueError("teacher_removed_tokens cannot be negative")
    model.eval()
    examples = _stratified_validation_slice(batcher.examples, limit)
    loss_sum = correct_sum = top5_correct_sum = exact_sum = token_sum = 0.0
    content_correct_sum = content_top5_sum = content_token_sum = 0.0
    content_loss_sum = 0.0
    unsupported_correct_sum = unsupported_top5_sum = 0.0
    unsupported_token_sum = unsupported_content_sum = 0.0
    unsupported_content_correct_sum = unsupported_content_top5_sum = 0.0
    unsupported_loss_sum = unsupported_content_loss_sum = 0.0
    whitespace_omission_sum = whitespace_mass_sum = 0.0
    adjacent_whitespace_sum = adjacent_pair_sum = 0.0
    predicted_content_sum = 0.0
    blank_ce_sum = viterbi_loss_sum = monotonic_loss_sum = 0.0
    viterbi_required_loss_sum = monotonic_required_loss_sum = 0.0
    blank_token_sum = 0.0
    sequence_loss_sum = 0.0
    sequence_composition_sum = sequence_length_sum = sequence_eos_sum = 0.0
    aligned_formation_sum = aligned_required_sum = aligned_content_sum = 0.0
    aligned_content_precision_sum = aligned_content_recall_sum = 0.0
    aligned_content_f1_sum = 0.0
    aligned_edit_accuracy_sum = aligned_exact_sum = 0.0
    aligned_displacement_sum = aligned_max_displacement_sum = 0.0
    edit_insertions_sum = edit_deletions_sum = edit_substitutions_sum = 0.0
    predicted_blank_sum = forbidden_prediction_sum = 0.0
    complete_sum = missing_eos_sum = eos_position_error_sum = 0.0
    eos_position_observation_sum = 0.0
    post_eos_violation_sum = post_eos_slot_sum = 0.0
    predicted_length_sum = reference_length_sum = 0.0
    whitespace_mask = whitespace_token_mask(tokenizer, device=device)
    valid_token_mask = legal_response_token_mask(tokenizer, device=device)
    semantic_token_mask = valid_token_mask & ~whitespace_mask
    semantic_token_mask = semantic_token_mask.clone()
    semantic_token_mask[tokenizer.EOS_ID] = False
    if tokenizer.blank_token_id is not None:
        semantic_token_mask[tokenizer.blank_token_id] = False
    stewardship_enabled = (
        getattr(getattr(model, "config", None), "cell_transport_mode", None)
        == "stewardship"
    )
    route_support_sum = route_entropy_sum = route_uncertainty_sum = None
    route_mean_uncertainty_sum = route_handoff_support_sum = None
    route_stay_sum = route_load_sum = None
    route_widen_sum = route_narrow_sum = None
    packed_execution_sum = None
    route_observations = 0
    terminal_voting = None
    shown_context = shown_target = shown_prediction = shown_prediction_slots = None
    shown_example = examples[display_index % len(examples)]
    evaluation_output_slots = (
        1
        if decode_mode == "serial"
        else (
            chunk_size
            if decode_mode == "chunk_parallel"
            else batcher.config.response_tokens
        )
    )
    for offset in range(0, len(examples), batch_size):
        batch_examples = examples[offset : offset + batch_size]
        batch = batcher.collate(batch_examples, device)
        if decode_mode == "serial":
            target_lengths = batch.target_mask.sum(dim=1)
            prefix_lengths = (
                torch.arange(
                    offset,
                    offset + len(batch_examples),
                    device=device,
                )
                % target_lengths
            )
            batch = serial_next_token_batch(
                batch,
                eos_token_id=tokenizer.EOS_ID,
                prefix_lengths=prefix_lengths,
            )
        elif decode_mode == "chunk_parallel":
            target_lengths = batch.target_mask.sum(dim=1)
            chunk_counts = torch.div(
                target_lengths + chunk_size - 1,
                chunk_size,
                rounding_mode="floor",
            )
            chunk_indices = (
                torch.arange(
                    offset,
                    offset + len(batch_examples),
                    device=device,
                )
                % chunk_counts
            )
            batch = parallel_next_chunk_batch(
                batch,
                eos_token_id=tokenizer.EOS_ID,
                chunk_size=chunk_size,
                chunk_indices=chunk_indices,
            )
        model_context_ids = batch.context_ids
        model_context_mask = batch.context_mask
        if context_mode == "masked":
            model_context_mask = torch.zeros_like(batch.context_mask)
        elif context_mode == "shuffled" and len(examples) > 1:
            context_examples = [
                examples[(offset + index + 1) % len(examples)]
                for index in range(len(batch_examples))
            ]
            context_batch = batcher.collate(context_examples, device)
            model_context_ids = context_batch.context_ids
            model_context_mask = context_batch.context_mask
        teacher_forced_kwargs = {}
        unsupported_slot_positions = None
        if decode_mode in {"teacher_forced", "token_handoff"}:
            if tokenizer.blank_token_id is None:
                raise ValueError(
                    "teacher-forced decoding requires a BLANK token"
                )
            if decode_mode == "token_handoff":
                (
                    output_token_ids,
                    unsupported_slot_positions,
                    _,
                ) = tokenwise_teacher_output_inputs(
                    batch,
                    blank_token_id=tokenizer.blank_token_id,
                    removed_tokens=teacher_removed_tokens,
                    row_offset=offset,
                    seed=17,
                )
            else:
                output_token_ids = (
                    teacher_forced_output_inputs(
                        batch,
                        blank_token_id=tokenizer.blank_token_id,
                    )
                    if teacher_prefix_retention >= 1.0
                    else deterministic_prefix_evaluation_inputs(
                        batch,
                        blank_token_id=tokenizer.blank_token_id,
                        retention=teacher_prefix_retention,
                        row_offset=offset,
                        seed=17,
                    )
                )
                if teacher_removed_tokens > 0:
                    (
                        _,
                        unsupported_slot_positions,
                        _,
                    ) = tokenwise_teacher_output_inputs(
                        batch,
                        blank_token_id=tokenizer.blank_token_id,
                        removed_tokens=teacher_removed_tokens,
                        row_offset=offset,
                        seed=17,
                    )
            teacher_forced_kwargs = {
                "output_token_ids": output_token_ids,
                "output_mask": batch.target_mask,
                "causal": True,
            }
        convergent_probe = (
            isinstance(model, ConvergentLanguageModel)
            and offset == 0
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(mixed_precision == "bf16" and device.type == "cuda"),
        ):
            if stewardship_enabled:
                model_output = model(
                    model_context_ids,
                    model_context_mask,
                    output_slots=evaluation_output_slots,
                    **teacher_forced_kwargs,
                    stewardship_temperature=stewardship_temperature,
                    stewardship_prune=stewardship_prune,
                    stewardship_packed_from_relay=stewardship_packed_from_relay,
                    diagnostics="route_metrics",
                )
            elif convergent_probe:
                model_output = model(
                    model_context_ids,
                    model_context_mask,
                    output_slots=evaluation_output_slots,
                    **teacher_forced_kwargs,
                    return_trace=True,
                )
            elif reasoning_loops == 2:
                pad_ids = torch.zeros_like(model_context_ids[:, :1])
                pad_mask = torch.zeros_like(model_context_mask[:, :1])
                draft_context_ids = torch.cat([model_context_ids, pad_ids], dim=1)
                draft_context_mask = torch.cat(
                    [model_context_mask, pad_mask], dim=1
                )
                with torch.no_grad():
                    draft_output = model(
                        draft_context_ids,
                        draft_context_mask,
                        output_slots=evaluation_output_slots,
                        **teacher_forced_kwargs,
                    )
                    draft_token = draft_output.argmax(dim=-1)[:, :1]
                refine_context_ids = torch.cat(
                    [model_context_ids, draft_token], dim=1
                )
                refine_context_mask = torch.cat(
                    [model_context_mask, torch.ones_like(pad_mask)], dim=1
                )
                model_output = model(
                    refine_context_ids,
                    refine_context_mask,
                    output_slots=evaluation_output_slots,
                    **teacher_forced_kwargs,
                )
            else:
                model_output = model(
                    model_context_ids,
                    model_context_mask,
                    output_slots=evaluation_output_slots,
                    **teacher_forced_kwargs,
                )
        if stewardship_enabled:
            logits, route_diagnostics = model_output
            ownership = route_diagnostics["ownership"]
            handoffs = route_diagnostics["handoffs"]
            uncertainty = route_diagnostics["route_uncertainty"]
            mean_uncertainty = route_diagnostics["route_mean_uncertainty"]
            handoff_support = route_diagnostics["handoff_support"]
            route_valid = route_diagnostics["route_valid"]
            packed_execution = route_diagnostics["packed_execution"]
            support = ownership.gt(0).sum(2).to(torch.float32)
            entropy = -(
                ownership.clamp_min(1e-9) * ownership.clamp_min(1e-9).log()
            ).sum(2)
            route_weight = route_valid[:, None].to(torch.float32)
            observations = int(route_valid.sum())
            if route_support_sum is None:
                route_support_sum = torch.zeros(
                    ownership.shape[1], dtype=torch.float64, device=device
                )
                route_entropy_sum = torch.zeros_like(route_support_sum)
                route_uncertainty_sum = torch.zeros(
                    uncertainty.shape[1], dtype=torch.float64, device=device
                )
                route_mean_uncertainty_sum = torch.zeros_like(
                    route_uncertainty_sum
                )
                route_handoff_support_sum = torch.zeros_like(
                    route_uncertainty_sum
                )
                route_stay_sum = torch.zeros_like(route_uncertainty_sum)
                route_widen_sum = torch.zeros_like(route_uncertainty_sum)
                route_narrow_sum = torch.zeros_like(route_uncertainty_sum)
                route_load_sum = torch.zeros(
                    ownership.shape[1] - 1,
                    ownership.shape[2],
                    dtype=torch.float64,
                    device=device,
                )
                packed_execution_sum = torch.zeros(
                    packed_execution.shape[0], dtype=torch.float64, device=device
                )
            route_support_sum += (
                support * route_weight
            ).sum(dim=(0, 2)).to(torch.float64)
            route_entropy_sum += (
                entropy * route_weight
            ).sum(dim=(0, 2)).to(torch.float64)
            route_uncertainty_sum += (
                uncertainty * route_weight
            ).sum(dim=(0, 2)).to(torch.float64)
            route_mean_uncertainty_sum += (
                mean_uncertainty * route_weight
            ).sum(dim=(0, 2)).to(torch.float64)
            route_handoff_support_sum += (
                handoff_support * route_weight
            ).sum(dim=(0, 2)).to(torch.float64)
            route_stay_sum += (
                ownership[:, :-1]
                * handoffs[..., 0]
                * route_valid[:, None, None].to(ownership.dtype)
            ).sum(dim=(0, 2, 3)).to(torch.float64)
            route_widen_sum += (
                support[:, 1:].gt(support[:, :-1]) * route_weight
            ).sum(dim=(0, 2)).to(torch.float64)
            route_narrow_sum += (
                support[:, 1:].lt(support[:, :-1]) * route_weight
            ).sum(dim=(0, 2)).to(torch.float64)
            route_load_sum += (
                ownership[:, 1:]
                * route_valid[:, None, None].to(ownership.dtype)
            ).sum(dim=(0, 3)).to(torch.float64)
            packed_execution_sum += packed_execution.to(torch.float64)
            route_observations += observations
        elif convergent_probe:
            logits, convergent_trace = model_output
            terminal_voting = terminal_vote_summary(
                convergent_trace["terminal_weights"],
                batch_size=len(batch_examples),
                output_slots=evaluation_output_slots,
            )
            terminal_query = model.core.terminal_vote_query
            terminal_voting["query_rms"] = (
                float(terminal_query.square().mean().sqrt())
                if terminal_query is not None
                else 0.0
            )
            if "active_route_k" in convergent_trace:
                terminal_voting["route_k_by_relay"] = (
                    convergent_trace["active_route_k"].detach().cpu().tolist()
                )
            if "route_uncertainty" in convergent_trace:
                terminal_voting["route_uncertainty_by_relay"] = (
                    convergent_trace["route_uncertainty"]
                    .detach()
                    .cpu()
                    .tolist()
                )
            if "active_node_union_fraction" in convergent_trace:
                terminal_voting["active_node_union_fraction"] = float(
                    convergent_trace["active_node_union_fraction"]
                )
        else:
            logits = model_output
        components = formation_loss_components(
            logits,
            batch,
            tokenizer,
            valid_token_mask=valid_token_mask,
            semantic_token_mask=semantic_token_mask,
            objective=formation_objective,
            alignment_mix=1.0,
            max_right_shift=alignment_max_shift,
            sequence_composition_weight=sequence_composition_weight,
            sequence_length_weight=sequence_length_weight,
            sequence_eos_weight=sequence_eos_weight,
            compute_all=True,
        )
        loss = components["slot_ce"]
        log_probs = components["log_probs"]
        alignment = best_aligned_diagnostics(
            logits,
            batch.target_ids,
            batch.target_mask,
            blank_id=tokenizer.blank_token_id,
            eos_id=tokenizer.EOS_ID,
            complete=batch.target_complete,
            valid_token_mask=valid_token_mask,
            precomputed_log_probs=log_probs,
            content_token_mask=semantic_token_mask,
            max_right_shift=alignment_max_shift,
        )
        gathered_nll = -log_probs.gather(
            2, batch.target_ids[..., None]
        ).squeeze(2)
        token_losses = torch.where(
            batch.target_mask, gathered_nll, torch.zeros_like(gathered_nll)
        )
        prediction = log_probs.argmax(-1)
        correct = prediction.eq(batch.target_ids)
        target_whitespace = whitespace_mask[batch.target_ids]
        predicted_whitespace = whitespace_mask[prediction]
        content_positions = batch.target_mask & semantic_token_mask[batch.target_ids]
        if unsupported_slot_positions is not None:
            unsupported_positions = (
                unsupported_slot_positions & batch.target_mask
            )
            unsupported_content_positions = (
                unsupported_positions & content_positions
            )
        else:
            unsupported_positions = torch.zeros_like(batch.target_mask)
            unsupported_content_positions = unsupported_positions
        top5 = log_probs.topk(min(5, logits.shape[-1]), dim=-1).indices
        top5_correct = top5.eq(batch.target_ids.unsqueeze(-1)).any(dim=-1)
        whitespace_probability = log_probs.exp()[..., whitespace_mask].sum(-1)
        pair_mask = batch.target_mask[:, :-1] & batch.target_mask[:, 1:]
        pair_mask = pair_mask & ~(
            target_whitespace[:, :-1] & target_whitespace[:, 1:]
        )
        tokens = int(batch.target_mask.sum())
        batch_examples_count = len(batch_examples)
        loss_sum += float(loss) * tokens
        blank_mask = components["blank_mask"]
        blank_ce = components["blank_ce"]
        assert blank_mask is not None and blank_ce is not None
        blank_tokens = int(blank_mask.sum())
        blank_ce_sum += float(blank_ce) * blank_tokens
        blank_token_sum += blank_tokens
        viterbi_result = components["viterbi_result"]
        assert viterbi_result is not None
        viterbi_loss_sum += float(viterbi_result.per_example.sum())
        viterbi_required_loss_sum += float(
            viterbi_result.per_required_token.sum()
        )
        monotonic_per_example = components["monotonic_per_example"]
        assert monotonic_per_example is not None
        monotonic_loss_sum += float(monotonic_per_example.sum())
        monotonic_required_per_example = components[
            "monotonic_required_per_example"
        ]
        assert monotonic_required_per_example is not None
        monotonic_required_loss_sum += float(
            monotonic_required_per_example.sum()
        )
        sequence = components["sequence"]
        assert sequence is not None
        sequence_loss_sum += float(sequence.loss) * batch_examples_count
        sequence_composition_sum += (
            float(sequence.semantic_composition) * batch_examples_count
        )
        sequence_length_sum += (
            float(sequence.expected_length_huber) * batch_examples_count
        )
        sequence_eos_sum += float(sequence.complete_eos_count) * batch_examples_count
        correct_sum += int((correct & batch.target_mask).sum())
        top5_correct_sum += int((top5_correct & batch.target_mask).sum())
        exact_sum += int((correct | ~batch.target_mask).all(1).sum())
        token_sum += tokens
        content_correct_sum += int((correct & content_positions).sum())
        content_top5_sum += int((top5_correct & content_positions).sum())
        content_token_sum += int(content_positions.sum())
        content_loss_sum += float((token_losses * content_positions).sum())
        unsupported_correct_sum += int(
            (correct & unsupported_positions).sum()
        )
        unsupported_top5_sum += int(
            (top5_correct & unsupported_positions).sum()
        )
        unsupported_token_sum += int(unsupported_positions.sum())
        unsupported_loss_sum += float(
            (token_losses * unsupported_positions).sum()
        )
        unsupported_content_correct_sum += int(
            (correct & unsupported_content_positions).sum()
        )
        unsupported_content_top5_sum += int(
            (top5_correct & unsupported_content_positions).sum()
        )
        unsupported_content_sum += int(
            unsupported_content_positions.sum()
        )
        unsupported_content_loss_sum += float(
            (token_losses * unsupported_content_positions).sum()
        )
        whitespace_omission_sum += int(
            (predicted_whitespace & content_positions).sum()
        )
        whitespace_mass_sum += float(
            (whitespace_probability * content_positions).sum()
        )
        adjacent_whitespace_sum += int(
            (
                predicted_whitespace[:, :-1]
                & predicted_whitespace[:, 1:]
                & pair_mask
            ).sum()
        )
        adjacent_pair_sum += int(pair_mask.sum())
        predicted_content_sum += int(
            (semantic_token_mask[prediction] & content_positions).sum()
        )
        aligned_formation_sum += float(alignment.formation_accuracy.sum())
        aligned_required_sum += float(alignment.required_token_accuracy.sum())
        aligned_content_sum += float(alignment.content_accuracy.sum())
        aligned_content_precision_sum += float(alignment.content_precision.sum())
        aligned_content_recall_sum += float(alignment.content_recall.sum())
        aligned_content_f1_sum += float(alignment.content_f1.sum())
        aligned_edit_accuracy_sum += float(alignment.edit_accuracy.sum())
        aligned_exact_sum += int(alignment.rendered_exact.sum())
        aligned_displacement_sum += float(alignment.mean_displacement.sum())
        aligned_max_displacement_sum += float(alignment.max_displacement.sum())
        edit_insertions_sum += int(alignment.insertions.sum())
        edit_deletions_sum += int(alignment.deletions.sum())
        edit_substitutions_sum += int(alignment.substitutions.sum())
        predicted_length_sum += int(alignment.predicted_lengths.sum())
        reference_length_sum += int(alignment.reference_lengths.sum())

        blank_id = tokenizer.blank_token_id
        assert blank_id is not None
        predicted_blank_sum += int(prediction.eq(blank_id).sum())
        forbidden_prediction_sum += int((~valid_token_mask[prediction]).sum())
        complete = batch.target_complete
        complete_sum += int(complete.sum())
        eos_prediction = prediction.eq(tokenizer.EOS_ID)
        has_eos = eos_prediction.any(1)
        missing_eos_sum += int((complete & ~has_eos).sum())
        slot_index = torch.arange(prediction.shape[1], device=device)[None]
        first_eos = torch.where(
            eos_prediction,
            slot_index,
            torch.full_like(slot_index, prediction.shape[1]),
        ).min(1).values
        reference_eos = batch.target_mask.sum(1) - 1
        eos_observed = complete & has_eos
        eos_position_error_sum += float(
            (first_eos[eos_observed] - reference_eos[eos_observed]).abs().sum()
        )
        eos_position_observation_sum += int(eos_observed.sum())
        after_eos = has_eos[:, None] & slot_index.gt(first_eos[:, None])
        post_eos_violation_sum += int(
            (after_eos & prediction.ne(blank_id)).sum()
        )
        post_eos_slot_sum += int(after_eos.sum())
        if shown_example in batch_examples:
            local_index = batch_examples.index(shown_example)
            shown_context = batch.context_ids[local_index].tolist()
            shown_target = batch.target_ids[local_index].tolist()
            shown_prediction = prediction[local_index].tolist()
            shown_prediction_slots = tokenizer.decode(
                shown_prediction, diagnostic=True
            )
    model.train()
    mean_loss = loss_sum / max(token_sum, 1)
    example_count = max(len(examples), 1)
    output_slot_count = max(len(examples) * evaluation_output_slots, 1)
    mean_blank_ce = blank_ce_sum / max(blank_token_sum, 1)
    mean_viterbi_loss = viterbi_loss_sum / example_count
    mean_monotonic_loss = monotonic_loss_sum / example_count
    mean_viterbi_required_loss = viterbi_required_loss_sum / example_count
    mean_monotonic_required_loss = monotonic_required_loss_sum / example_count
    selection_loss = {
        "slot_ce": mean_loss,
        "blank_ce": mean_blank_ce,
        "viterbi": mean_viterbi_loss,
        "monotonic": mean_monotonic_loss,
    }[formation_objective]
    result = {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "blank_ce_loss": mean_blank_ce,
        "best_aligned_loss": mean_viterbi_loss,
        "aligned_loss": mean_monotonic_loss,
        "best_aligned_required_loss": mean_viterbi_required_loss,
        "aligned_required_loss": mean_monotonic_required_loss,
        "selection_loss": selection_loss,
        "formation_objective": formation_objective,
        "sequence_quality_loss": sequence_loss_sum / example_count,
        "sequence_composition_loss": sequence_composition_sum / example_count,
        "sequence_length_loss": sequence_length_sum / example_count,
        "sequence_eos_loss": sequence_eos_sum / example_count,
        "accuracy": correct_sum / max(token_sum, 1),
        "top5_accuracy": top5_correct_sum / max(token_sum, 1),
        "exact": exact_sum / example_count,
        "best_aligned_accuracy": aligned_formation_sum / example_count,
        "aligned_required_accuracy": aligned_required_sum / example_count,
        "aligned_content_accuracy": aligned_content_sum / example_count,
        "aligned_content_precision": aligned_content_precision_sum / example_count,
        "aligned_content_recall": aligned_content_recall_sum / example_count,
        "aligned_content_f1": aligned_content_f1_sum / example_count,
        "aligned_edit_accuracy": aligned_edit_accuracy_sum / example_count,
        "aligned_exact": aligned_exact_sum / max(complete_sum, 1),
        "mean_alignment_displacement": aligned_displacement_sum / example_count,
        "mean_max_alignment_displacement": (
            aligned_max_displacement_sum / example_count
        ),
        "edit_insertions_per_example": edit_insertions_sum / example_count,
        "edit_deletions_per_example": edit_deletions_sum / example_count,
        "edit_substitutions_per_example": edit_substitutions_sum / example_count,
        "predicted_length": predicted_length_sum / example_count,
        "reference_length": reference_length_sum / example_count,
        "complete_fraction": complete_sum / example_count,
        "missing_eos_rate": missing_eos_sum / max(complete_sum, 1),
        "eos_position_mae": (
            eos_position_error_sum / max(eos_position_observation_sum, 1)
        ),
        "post_eos_violation_rate": (
            post_eos_violation_sum / max(post_eos_slot_sum, 1)
        ),
        "blank_prediction_rate": predicted_blank_sum / output_slot_count,
        "forbidden_prediction_rate": forbidden_prediction_sum / output_slot_count,
        "content_accuracy": content_correct_sum / max(content_token_sum, 1),
        "content_loss": content_loss_sum / max(content_token_sum, 1),
        "content_perplexity": math.exp(
            min(content_loss_sum / max(content_token_sum, 1), 20.0)
        ),
        "content_top5_accuracy": content_top5_sum / max(content_token_sum, 1),
        "unsupported_accuracy": (
            unsupported_correct_sum / max(unsupported_token_sum, 1)
        ),
        "unsupported_top5_accuracy": (
            unsupported_top5_sum / max(unsupported_token_sum, 1)
        ),
        "unsupported_loss": (
            unsupported_loss_sum / max(unsupported_token_sum, 1)
        ),
        "unsupported_content_accuracy": (
            unsupported_content_correct_sum
            / max(unsupported_content_sum, 1)
        ),
        "unsupported_content_top5_accuracy": (
            unsupported_content_top5_sum
            / max(unsupported_content_sum, 1)
        ),
        "unsupported_content_loss": (
            unsupported_content_loss_sum
            / max(unsupported_content_sum, 1)
        ),
        "unsupported_content_perplexity": math.exp(
            min(
                unsupported_content_loss_sum
                / max(unsupported_content_sum, 1),
                20.0,
            )
        ),
        "unsupported_tokens": int(unsupported_token_sum),
        "unsupported_content_tokens": int(unsupported_content_sum),
        "whitespace_omission_rate": whitespace_omission_sum
        / max(content_token_sum, 1),
        "whitespace_mass_on_content": whitespace_mass_sum
        / max(content_token_sum, 1),
        "adjacent_whitespace_rate": adjacent_whitespace_sum
        / max(adjacent_pair_sum, 1),
        "formation_coverage": predicted_content_sum / max(content_token_sum, 1),
        "sample_conversation": shown_example.conversation_id,
        "sample_turn": shown_example.turn_index,
        "sample_source": shown_example.source,
        "context": tokenizer.decode(shown_context),
        "target": tokenizer.decode(shown_target),
        "prediction": tokenizer.decode(shown_prediction),
        "prediction_slots": shown_prediction_slots,
        "examples": len(examples),
        "decode_mode": decode_mode,
        "chunk_size": (
            chunk_size if decode_mode == "chunk_parallel" else None
        ),
        "chunk_prefix_mode": (
            "gold_completed_chunks"
            if decode_mode == "chunk_parallel"
            else None
        ),
        "context_mode": context_mode,
        "teacher_prefix_retention": teacher_prefix_retention,
        "teacher_removed_tokens": teacher_removed_tokens,
    }
    if stewardship_enabled and route_support_sum is not None:
        mean_load = route_load_sum / max(route_observations, 1)
        load_cv = mean_load.std(dim=1, unbiased=False) / mean_load.mean(
            dim=1
        ).clamp_min(1e-12)
        result["stewardship"] = {
            "support_by_relay": (
                route_support_sum / max(route_observations, 1)
            ).tolist(),
            "effective_cells_by_relay": torch.exp(
                route_entropy_sum / max(route_observations, 1)
            ).tolist(),
            "uncertainty_by_relay": (
                route_uncertainty_sum / max(route_observations, 1)
            ).tolist(),
            "mean_uncertainty_by_relay": (
                route_mean_uncertainty_sum / max(route_observations, 1)
            ).tolist(),
            "handoff_support_by_relay": (
                route_handoff_support_sum / max(route_observations, 1)
            ).tolist(),
            "stay_rate_by_relay": (
                route_stay_sum / max(route_observations, 1)
            ).tolist(),
            "widen_rate_by_relay": (
                route_widen_sum / max(route_observations, 1)
            ).tolist(),
            "narrow_rate_by_relay": (
                route_narrow_sum / max(route_observations, 1)
            ).tolist(),
            "load_cv_by_relay": load_cv.tolist(),
            "dead_cell_fraction_by_relay": mean_load.le(1e-8).to(
                torch.float32
            ).mean(1).tolist(),
            "packed_execution_by_relay": packed_execution_sum.div(
                max(len(range(0, len(examples), batch_size)), 1)
            ).tolist(),
        }
    if terminal_voting is not None:
        result["terminal_voting"] = terminal_voting
    return result


def _data_file_identity(path: str | None) -> dict | None:
    """Return a conservative file or shard-directory identity."""

    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        files = []
        for child in sorted(
            (entry for entry in resolved.rglob("*") if entry.is_file()),
            key=lambda entry: entry.relative_to(resolved).as_posix(),
        ):
            stat = child.stat()
            files.append(
                {
                    "path": child.relative_to(resolved).as_posix(),
                    "size": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                }
            )
        return {
            "path": str(resolved),
            "kind": "directory",
            "files": files,
        }
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "kind": "file",
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def _fractional_source_weight(
    other_weights: list[float],
    fraction: float,
) -> float:
    """Return the relative weight that gives one source an exact share."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    other_total = sum(other_weights)
    if other_total <= 0:
        return 1.0
    return other_total * fraction / (1.0 - fraction)


def _stable_fingerprint(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _data_resume_fingerprint_check(
    resume_state: dict,
    saved_pipeline: dict,
    *,
    current_data_fingerprint: str,
    current_validation_fingerprint: str,
    policy: str,
) -> tuple[str | None, bool]:
    """Gate portable exact resume before the provider proves corpus identity.

    The conservative data fingerprint includes absolute paths and mtimes, so
    byte-identical corpora copied to a new runtime intentionally fail strict
    ``exact`` mode. ``portable_exact`` may cross that file-identity boundary
    only when validation content is unchanged. The subsequent pipeline
    ``load_state_dict`` remains authoritative: it verifies the encoded finite
    corpus fingerprint, task geometry, queued batches, and provider RNG state.
    """

    if saved_pipeline.get("fingerprint") == current_data_fingerprint:
        return None, False
    if policy != "portable_exact":
        return "data_fingerprint_mismatch", False
    saved_data_config = resume_state.get("data_config")
    if not isinstance(saved_data_config, dict):
        return "checkpoint_has_no_data_config", False
    if (
        saved_data_config.get("validation_fingerprint")
        != current_validation_fingerprint
    ):
        return "validation_fingerprint_mismatch", False
    return None, True


def _encoded_examples_fingerprint(
    examples: list[EncodedAssistantExample],
) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(str(example.conversation_id).encode("ascii"))
        digest.update(b":")
        digest.update(str(example.turn_index).encode("ascii"))
        digest.update(b":")
        digest.update((example.source or "").encode("utf-8"))
        digest.update(b"\0")
        for token_id in example.context_ids:
            digest.update(int(token_id).to_bytes(4, "little", signed=False))
        digest.update(b"\xff")
        for token_id in example.response_ids:
            digest.update(int(token_id).to_bytes(4, "little", signed=False))
        digest.update(b"\xfe")
    return digest.hexdigest()


def checkpoint_state(
    *,
    step: int,
    model: GatorLanguageModel,
    engine: OptimizationEngine,
    tokenizer: CompactTokenizer,
    task: LanguageBatchConfig,
    data_config: dict,
    validation: dict,
    auxiliary_validation: dict | None = None,
    support_validation: dict | None = None,
    context_validation: dict | None = None,
    training_evaluation: dict | None = None,
    execution_state: dict | None = None,
    data_pipeline_state: dict | None = None,
) -> dict:
    return {
        "format": "gator-language-v2",
        "step": step,
        "model": model.state_dict(),
        "model_config": model.config.to_dict(),
        "training": engine.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "task_config": {
            "context_tokens": task.context_tokens,
            "response_tokens": task.response_tokens,
            "reference_tokens": task.target_tokens,
        },
        "data_config": data_config,
        "validation": validation,
        "auxiliary_validation": auxiliary_validation,
        "support_validation": support_validation,
        "context_validation": context_validation,
        "training_evaluation": training_evaluation,
        "execution": execution_state,
        "data_pipeline": data_pipeline_state,
    }


_ACTIVE_LANGUAGE_PIPELINES: set[LanguageBatchPipeline] = set()


def _close_active_language_pipelines() -> None:
    for pipeline in tuple(_ACTIVE_LANGUAGE_PIPELINES):
        pipeline.close()
        atexit.unregister(pipeline.close)
        _ACTIVE_LANGUAGE_PIPELINES.discard(pipeline)


def _run_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data")
    parser.add_argument("--training-recipe", choices=["standard", "smollm3-26b"], default="standard")
    parser.add_argument("--hf-tokenizer", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--hf-tokenizer-revision")
    parser.add_argument(
        "--smollm3-peak-lr",
        type=float,
        default=2e-4,
        help="peak LR after the token-equivalent SmolLM3 warmup",
    )
    parser.add_argument(
        "--smollm3-effective-batch-sequences",
        type=int,
        default=32,
        help=(
            "sequences per optimizer update (batch_size * "
            "gradient_accumulation_steps must equal this); raising it "
            "increases the effective batch size for gradient smoothing. "
            "Also rescales the token-equivalent LR/stage schedule and "
            "default --steps, same as changing --smollm3-sequence-length"
        ),
    )
    parser.add_argument(
        "--smollm3-sequence-length",
        type=int,
        default=4096,
        help="total observed-plus-output tokens per SmolLM3 formation",
    )
    parser.add_argument(
        "--smollm3-allow-context-growth",
        action="store_true",
        help=(
            "resume a SmolLM3 checkpoint at a strictly longer sequence "
            "while preserving source cursors and token-equivalent schedule "
            "progress; requires a fixed position reference long enough for "
            "the requested sequence"
        ),
    )
    parser.add_argument(
        "--smollm3-max-consecutive-nonfinite-windows",
        type=int,
        default=3,
        help=(
            "discard and retry up to this many consecutive nonfinite gradient "
            "accumulation windows; zero restores fail-fast behavior"
        ),
    )
    parser.add_argument(
        "--smollm3-tokenized-data-root",
        help="root containing Nanotron/DataTrove per-source .ds shard folders",
    )
    parser.add_argument(
        "--smollm3-stream-cache-dir",
        help="fast local-disk parent for bounded raw-HF token chunks (never use Drive)",
    )
    parser.add_argument(
        "--smollm3-token-chunk-tokens",
        type=int,
        default=65_536,
        help="tokens in each bounded on-disk/raw-stream chunk",
    )
    parser.add_argument(
        "--smollm3-prefetch-tokens",
        type=int,
        default=8_192,
        help="prepare the next disk chunk when active RAM tokens fall below this count",
    )
    parser.add_argument(
        "--smollm3-prefetch-workers",
        type=int,
        default=2,
        help="global cap on concurrent raw-HF tokenization workers",
    )
    parser.add_argument(
        "--smollm3-max-document-chars",
        type=int,
        default=262_144,
        help="direct-HF safety cap before tokenization; exact .ds mode is unaffected",
    )
    parser.add_argument(
        "--smollm3-max-open-streams",
        type=int,
        default=2,
        help="maximum simultaneously resident HF/PyArrow dataset readers",
    )
    parser.add_argument(
        "--smollm3-max-host-ram-gb",
        type=float,
        default=20.0,
        help="direct-HF process RSS guard in GiB; zero disables",
    )
    parser.add_argument(
        "--smollm3-memory-log-every-seconds",
        type=float,
        default=60.0,
        help="periodic RSS/open-stream diagnostic interval; zero disables",
    )
    parser.add_argument("--checkpoint-every-updates", type=int, default=5_000)
    parser.add_argument(
        "--smollm3-fixed-eval-every-updates",
        type=int,
        default=500,
        help=(
            "how often (in optimizer updates) to greedy-decode a fixed "
            "10-prompt set and log completions to checkpoint_dir/"
            "eval_prompts.log, for human progress monitoring"
        ),
    )
    parser.add_argument("--lmsys-data")
    parser.add_argument("--lmsys-weight", type=float, default=0.9)
    parser.add_argument(
        "--streaming-weight",
        type=float,
        help=(
            "total probability assigned to all streamed corpora; defaults to "
            "--lmsys-weight for backward compatibility"
        ),
    )
    parser.add_argument(
        "--streaming-exhaustion",
        choices=["restart", "drop"],
        default="restart",
        help=(
            "drain one complete shuffled corpus epoch before restarting, or "
            "permanently drop that source at EOF"
        ),
    )
    parser.add_argument(
        "--lmsys-source-weight",
        type=float,
        default=1.0,
        help="relative LMSYS weight inside the streamed-corpus mixture",
    )
    parser.add_argument(
        "--primary-stream-weight",
        type=float,
        default=0.0,
        help=(
            "also expose the primary SmolTalk training split as a named "
            "stream with this relative weight; use with streaming-weight 1 "
            "and weights 1:4:3:2 to reproduce the language corpus"
        ),
    )
    parser.add_argument(
        "--primary-shuffle-buffer",
        type=int,
        default=8192,
        help="shuffle buffer for the optional primary SmolTalk stream",
    )
    parser.add_argument(
        "--tinystories-data",
        help=(
            "TinyStories train text, JSONL, Parquet shard, or shard directory; "
            "the official text is split on <|endoftext|>, not newlines"
        ),
    )
    parser.add_argument(
        "--tinystories-validation-data",
        help=(
            "optional separate TinyStories validation file/directory; when "
            "omitted, the first validation stories are reserved from train"
        ),
    )
    parser.add_argument(
        "--tinystories-mix-fraction",
        type=float,
        default=0.40,
        help=(
            "TinyStories share of the named streaming mixture; all previous "
            "source weights retain their relative proportions"
        ),
    )
    parser.add_argument(
        "--tinystories-validation-stories",
        type=int,
        default=512,
        help="complete stories reserved for TinyStories validation",
    )
    parser.add_argument(
        "--tinystories-tokenizer-stories",
        type=int,
        default=50_000,
        help="training stories exposed while fitting a scratch tokenizer",
    )
    parser.add_argument(
        "--tinystories-shuffle-buffer",
        type=int,
        default=16_384,
        help="streaming shuffle buffer for TinyStories continuation examples",
    )
    parser.add_argument(
        "--chat-data",
        action="append",
        default=[],
        help=(
            "additional text/source chat JSONL; repeat once per corpus. These "
            "sources train only and retain the original validation benchmark"
        ),
    )
    parser.add_argument(
        "--chat-source-weight",
        action="append",
        type=float,
        default=[],
        help=(
            "relative weight for the corresponding --chat-data source; omit "
            "all weights to use 1.0 for every additional source"
        ),
    )
    parser.add_argument(
        "--chat-tokenizer-conversations",
        type=int,
        default=10_000,
        help="scratch-training tokenizer rows sampled from each additional corpus",
    )
    parser.add_argument(
        "--chat-max-characters",
        type=int,
        help=(
            "skip additional chat rows above this decoded length; defaults to "
            "--lmsys-max-characters"
        ),
    )
    parser.add_argument(
        "--chat-shuffle-buffer",
        type=int,
        default=16_384,
        help="per-corpus streaming shuffle buffer for additional chat data",
    )
    parser.add_argument(
        "--chat-validation-conversations",
        type=int,
        default=64,
        help=(
            "reserve this many leading conversations from each --chat-data "
            "source as held-out validation examples (excluded from "
            "training); zero disables, leaving --chat-data sources "
            "train-only as before. Without this, evaluation metrics could "
            "silently never draw from these sources at all"
        ),
    )
    parser.add_argument(
        "--prose-data",
        action="append",
        default=[],
        help=(
            "additional raw-text JSONL ({\"text\": ...} per row, e.g. "
            "Wikipedia articles); repeat once per corpus. Converted into "
            "prefix/continuation examples the same way as --tinystories-data, "
            "then interleaved by weight like the chat sources"
        ),
    )
    parser.add_argument(
        "--prose-source-weight",
        action="append",
        type=float,
        default=[],
        help=(
            "relative weight for the corresponding --prose-data source; omit "
            "all weights to use 1.0 for every prose source"
        ),
    )
    parser.add_argument(
        "--prose-tokenizer-stories",
        type=int,
        default=10_000,
        help="scratch-training tokenizer rows sampled from each prose corpus",
    )
    parser.add_argument(
        "--prose-shuffle-buffer",
        type=int,
        default=16_384,
        help="per-corpus streaming shuffle buffer for additional prose data",
    )
    parser.add_argument(
        "--prose-validation-stories",
        type=int,
        default=64,
        help=(
            "reserve this many leading stories/articles from each "
            "--prose-data source as held-out validation examples (excluded "
            "from training, same pattern as --tinystories-validation-stories); "
            "zero disables"
        ),
    )
    parser.add_argument("--lmsys-validation-conversations", type=int, default=512)
    parser.add_argument("--lmsys-tokenizer-conversations", type=int, default=50_000)
    parser.add_argument(
        "--lmsys-max-characters",
        type=int,
        help="skip LMSYS rows whose complete decoded text field exceeds this length",
    )
    parser.add_argument("--lmsys-load-mode", choices=["memory", "stream"], default="memory")
    parser.add_argument("--lmsys-shuffle-buffer", type=int, default=2048)
    parser.add_argument(
        "--lmsys-min-ascii-letter-ratio",
        type=float,
        default=0.0,
        help="reject LMSYS examples below this cheap English-oriented text ratio",
    )
    parser.add_argument(
        "--lmsys-min-response-characters",
        type=int,
        default=0,
        help="reject shorter LMSYS assistant responses before tokenization",
    )
    parser.add_argument(
        "--lmsys-require-user-prompt",
        action="store_true",
        help="require the turn immediately before an LMSYS response to be a user turn",
    )
    parser.add_argument(
        "--lmsys-skip-first-assistant",
        action="store_true",
        help="optionally skip the first LMSYS assistant reply; disabled by default",
    )
    parser.add_argument(
        "--require-complete-windows",
        action="store_true",
        help="skip examples whose full context or response+EOS would be token-truncated",
    )
    parser.add_argument(
        "--require-complete-context",
        action="store_true",
        help="skip contexts that still exceed the token window after whole-turn trimming",
    )
    parser.add_argument(
        "--require-complete-response",
        action="store_true",
        help="skip responses that do not fit with EOS instead of training their first window",
    )
    parser.add_argument(
        "--trim-context-turns",
        action="store_true",
        help="drop whole oldest turns until context fits; never token-crop a sentence",
    )
    parser.add_argument(
        "--deduplicate-examples",
        action="store_true",
        help="deduplicate normalized context/response pairs within each corpus traversal",
    )
    parser.add_argument("--checkpoint-dir", default="experimental_checkpoints/language")
    parser.add_argument(
        "--resume-checkpoint",
        help=(
            "load model and tokenizer from a language checkpoint; --steps then "
            "means additional optimizer steps"
        ),
    )
    parser.add_argument(
        "--grow-from-checkpoint",
        help=(
            "--architecture gator only: initialize a NEW, LARGER model from "
            "a smaller trained checkpoint by cloning each cell "
            "--grow-clones-per-cell times and scattering the clones across "
            "the grown ring (see gator_cell_growth.py). Every other "
            "GatorLanguageConfig field must match the checkpoint's own "
            "(vocab, d_model, embedding_dim, node_topology, relays, etc.) "
            "except --cells itself, which must equal the checkpoint's "
            "n_cells times --grow-clones-per-cell, and --neighbors, which "
            "may differ freely. Starts a fresh run at step 0 with a fresh "
            "optimizer, not a resume - mutually exclusive with "
            "--resume-checkpoint"
        ),
    )
    parser.add_argument(
        "--grow-clones-per-cell",
        type=int,
        default=4,
        help="clone multiplier applied to every cell by --grow-from-checkpoint",
    )
    parser.add_argument(
        "--grow-perturb-scale",
        type=float,
        default=0.01,
        help=(
            "fraction of each cloned parameter's own std used as Gaussian "
            "perturbation noise for clones after the first (which is an "
            "exact copy); only used by --grow-from-checkpoint"
        ),
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help=(
            "also preserve optimizer moments; the scheduler restarts and "
            "--learning-rate sets the new phase's initial LR"
        ),
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help=(
            "stop cleanly after this much training wall time; 0 disables "
            "the time limit and --steps remains the only bound"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--mixed-precision",
        choices=["none", "bf16"],
        default="none",
        help=(
            "wrap the model forward pass in torch.autocast(dtype=bfloat16); "
            "loss/backward/optimizer step stay in float32. bf16 needs no "
            "loss scaling (unlike fp16) since it keeps float32's exponent "
            "range, only trading mantissa precision"
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "backpropagate this many micro-batches (each of --batch-size) "
            "before every optimizer update, so --steps counts micro-batches "
            "processed, not optimizer updates applied (updates applied = "
            "--steps / --gradient-accumulation-steps). Lets an effective "
            "larger batch be reached without the peak-memory cost of one "
            "large batch; 1 disables accumulation and matches prior behavior"
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-examples", type=int, default=0,
                        help="validation examples per evaluation; 0 evaluates all")
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument(
        "--log-every-seconds",
        type=float,
        default=0.0,
        help=(
            "also evaluate and checkpoint on this wall-clock interval; "
            "0 disables time-based logging"
        ),
    )
    parser.add_argument(
        "--logging-slim",
        action="store_true",
        help=(
            "print only compact learning/routing metrics plus context, target, "
            "and formed samples"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--repeat-training-examples",
        type=int,
        default=0,
        help=(
            "diagnostic mode: restrict training to one deterministic subset of this "
            "many SmolTalk responses and sample that same pool repeatedly; 0 disables"
        ),
    )
    parser.add_argument("--skip-first-assistant", action="store_true",
                        help="exclude each conversation's boilerplate first assistant reply")
    parser.add_argument("--vocab-size", type=int, default=2_048)
    parser.add_argument("--context-tokens", type=int, default=64)
    parser.add_argument("--response-tokens", type=int, default=32)
    parser.add_argument(
        "--decode-mode",
        choices=[
            "parallel",
            "serial",
            "chunk_parallel",
            "teacher_forced",
            "handoff",
            "graduated_handoff",
            "token_handoff",
            "prefix_denoise",
        ],
        default="parallel",
        help=(
            "parallel forms every response slot simultaneously; serial "
            "samples one correct response prefix and predicts one next token; "
            "chunk_parallel appends completed gold chunks to the context and "
            "forms the next aligned chunk simultaneously; "
            "teacher_forced predicts every response token with shifted "
            "references and causal graph memory; handoff begins fully "
            "teacher-forced and accuracy-gates whole batches into blank "
            "parallel formation; graduated_handoff starts with alternating "
            "blank-parallel/teacher batches and increases unsupported runs "
            "from unsupported validation gains; prefix_denoise causally removes individual "
            "prefix tokens and spans while monitoring semantic context utility; "
            "token_handoff removes exactly one additional causal teacher "
            "token after sustained slot-specific mastery"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help=(
            "simultaneous target width for chunk_parallel decoding; prior "
            "completed chunks are appended to the context"
        ),
    )
    parser.add_argument(
        "--serial-positions-per-example",
        type=int,
        default=1,
        help=(
            "for decode_mode=serial: expand each training row into this "
            "many stratified prefix positions per step instead of one "
            "random position (default 1, matching original behavior). "
            "Denser per-step supervision across the full response window "
            "at a positions_per_example x compute cost."
        ),
    )
    parser.add_argument(
        "--handoff-start-content-accuracy",
        type=float,
        default=0.60,
        help=(
            "teacher-forced validation content accuracy below which handoff "
            "uses 100%% teacher-forced batches"
        ),
    )
    parser.add_argument(
        "--handoff-end-content-accuracy",
        type=float,
        default=0.90,
        help=(
            "teacher-forced validation content accuracy at which handoff "
            "uses 0%% teacher-forced batches"
        ),
    )
    parser.add_argument(
        "--handoff-accuracy-ema-decay",
        type=float,
        default=0.50,
        help=(
            "EMA decay for the teacher-forced validation content accuracy "
            "that controls handoff"
        ),
    )
    parser.add_argument(
        "--graduated-baseline-fraction",
        type=float,
        default=0.05,
        help=(
            "fraction of planned graduated_handoff steps spent alternating "
            "unsupported and teacher batches before locking the unsupported "
            "content-accuracy baseline"
        ),
    )
    parser.add_argument(
        "--graduated-accuracy-step",
        type=float,
        default=0.05,
        help=(
            "unsupported content-accuracy gain above baseline required to "
            "add one unsupported batch between teacher batches"
        ),
    )
    parser.add_argument(
        "--graduated-release-accuracy",
        type=float,
        default=0.90,
        help=(
            "unsupported content top-1 accuracy that permanently removes "
            "teacher batches"
        ),
    )
    parser.add_argument(
        "--token-handoff-start-removed",
        type=int,
        default=1,
        help="teacher-prefix tokens removed at the first token_handoff level",
    )
    parser.add_argument(
        "--token-handoff-max-removed",
        type=int,
        default=0,
        help=(
            "maximum teacher-prefix tokens removed; 0 releases every "
            "eligible teacher token"
        ),
    )
    parser.add_argument(
        "--token-handoff-advance-ratio",
        type=float,
        default=0.95,
        help=(
            "unsupported-slot accuracy divided by teacher accuracy required "
            "to advance one token"
        ),
    )
    parser.add_argument(
        "--token-handoff-retreat-ratio",
        type=float,
        default=0.75,
        help="accuracy ratio below which the curriculum restores one token",
    )
    parser.add_argument(
        "--token-handoff-min-teacher-accuracy",
        type=float,
        default=0.15,
        help="minimum teacher frontier accuracy before advancement is legal",
    )
    parser.add_argument(
        "--token-handoff-min-frontier-accuracy",
        type=float,
        default=0.0,
        help="minimum unsupported frontier accuracy before advancement is legal",
    )
    parser.add_argument(
        "--token-handoff-max-teacher-ppl-ratio",
        type=float,
        default=0.0,
        help=(
            "maximum full-teacher content-PPL ratio to the pre-handoff "
            "baseline that permits advancement; 0 disables the guard"
        ),
    )
    parser.add_argument(
        "--token-handoff-retreat-teacher-ppl-ratio",
        type=float,
        default=0.0,
        help=(
            "full-teacher content-PPL ratio that counts as regression and "
            "restores support; 0 disables the guard"
        ),
    )
    parser.add_argument(
        "--token-handoff-advance-patience",
        type=int,
        default=3,
        help="consecutive successful evaluations required to remove one token",
    )
    parser.add_argument(
        "--token-handoff-retreat-patience",
        type=int,
        default=2,
        help="consecutive regressed evaluations required to restore one token",
    )
    parser.add_argument(
        "--token-handoff-full-probe-fraction",
        type=float,
        default=0.02,
        help="training-batch fraction with every teacher token removed",
    )
    parser.add_argument(
        "--token-handoff-unforced-loss-weight",
        type=float,
        default=1.0,
        help="extra CE weight applied only to unsupported output slots",
    )
    parser.add_argument(
        "--token-handoff-loss-normalization",
        choices=["released", "active_slots"],
        default="released",
        help=(
            "normalize the unsupported-slot auxiliary by released slots "
            "(legacy) or by every active response slot"
        ),
    )
    parser.add_argument(
        "--token-handoff-allow-policy-override",
        action="store_true",
        help=(
            "on token_handoff resume, preserve dynamic controller state while "
            "accepting explicitly configured gate-policy changes"
        ),
    )
    parser.add_argument(
        "--route-temperature-schedule",
        choices=["legacy", "staged_reheat"],
        default="legacy",
        help=(
            "global-step legacy cooling or a branch-local hold and geometric "
            "reheat/cool schedule"
        ),
    )
    parser.add_argument("--route-temperature-start", type=float, default=1.25)
    parser.add_argument("--route-temperature-middle", type=float, default=0.55)
    parser.add_argument("--route-temperature-end", type=float, default=0.35)
    parser.add_argument(
        "--route-temperature-hold-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--route-temperature-middle-fraction",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--route-temperature-step-offset",
        type=int,
        default=0,
        help=(
            "completed steps in the staged route-temperature schedule before "
            "this invocation; preserves one schedule across resumable sessions"
        ),
    )
    parser.add_argument(
        "--route-temperature-total-steps",
        type=int,
        default=0,
        help=(
            "complete staged route-temperature horizon; 0 uses this "
            "invocation's --steps value"
        ),
    )
    parser.add_argument(
        "--optimizer-groups",
        choices=["monolithic", "component_scaled"],
        default="monolithic",
        help=(
            "one optimizer group or architecture-aware lexical, lane/core, "
            "private-node, and route/gate groups"
        ),
    )
    parser.add_argument(
        "--optimizer-lexical-lr-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--optimizer-lane-core-lr-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--optimizer-private-node-lr-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--optimizer-route-gate-lr-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--optimizer-train-components",
        nargs="+",
        choices=OPTIMIZER_COMPONENTS,
        default=OPTIMIZER_COMPONENTS,
        help=(
            "component-scaled optimizer groups allowed to update; omitted "
            "components are frozen but remain in the forward graph"
        ),
    )
    parser.add_argument(
        "--optimizer-retain-frozen-state",
        action="store_true",
        help=(
            "keep frozen component parameters and dormant AdamW moments in "
            "the optimizer state for later unfreezing"
        ),
    )
    parser.add_argument(
        "--optimizer-freeze-route-transport",
        action="store_true",
        help=(
            "freeze router, ingress, and address-edge transport parameters "
            "while leaving terminal-jury parameters trainable"
        ),
    )
    parser.add_argument(
        "--prefix-retention-start",
        type=float,
        default=0.95,
        help=(
            "initial fraction of eligible causal response-prefix tokens "
            "retained in prefix_denoise mode"
        ),
    )
    parser.add_argument(
        "--prefix-retention-floor",
        type=float,
        default=0.50,
        help=(
            "retention reached by the fixed prefix-denoising warmup before "
            "accuracy/context-utility feedback can remove more support"
        ),
    )
    parser.add_argument(
        "--prefix-denoise-warmup-steps",
        type=int,
        default=100_000,
        help="steps used to move from start to floor prefix retention",
    )
    parser.add_argument(
        "--prefix-zero-row-fraction",
        type=float,
        default=0.05,
        help="fraction of prefix-denoising rows with no gold response prefix",
    )
    parser.add_argument(
        "--prefix-short-row-fraction",
        type=float,
        default=0.10,
        help="fraction of rows retaining only a short leading response prefix",
    )
    parser.add_argument(
        "--prefix-short-max-tokens",
        type=int,
        default=8,
        help="maximum visible prefix tokens on a short-prefix row",
    )
    parser.add_argument(
        "--prefix-span-fraction",
        type=float,
        default=0.50,
        help=(
            "fraction of ordinary prefix-denoising rows using one contiguous "
            "missing span instead of independent token dropout"
        ),
    )
    parser.add_argument(
        "--prefix-context-utility-start",
        type=float,
        default=0.02,
        help=(
            "correct-context minus shuffled-context accuracy required before "
            "adaptive support removal begins"
        ),
    )
    parser.add_argument(
        "--prefix-context-utility-end",
        type=float,
        default=0.10,
        help=(
            "context-utility margin counted as fully ready for adaptive "
            "support removal"
        ),
    )
    parser.add_argument(
        "--reference-tokens",
        type=int,
        default=0,
        help=(
            "maximum reference pieces placed into the response window; 0 uses "
            "all response slots, while a smaller value reserves alignment slack"
        ),
    )
    parser.add_argument(
        "--formation-objective",
        choices=["slot_ce", "blank_ce", "viterbi", "monotonic"],
        default="slot_ce",
        help=(
            "historical masked slot CE, rigid all-slot BLANK CE, detached "
            "Viterbi-aligned CE, or forward-sum monotonic alignment"
        ),
    )
    parser.add_argument(
        "--alignment-warmup-fraction",
        type=float,
        default=0.05,
        help="fraction of training that blends from rigid BLANK CE into an aligned objective",
    )
    parser.add_argument(
        "--alignment-max-shift",
        type=int,
        default=4,
        help="maximum slots a target token may move right; -1 permits every monotonic path",
    )
    parser.add_argument(
        "--sequence-quality-penalty",
        type=float,
        default=0.0,
        help="weight for the whole-formation composition/length/EOS auxiliary",
    )
    parser.add_argument("--sequence-composition-weight", type=float, default=1.0)
    parser.add_argument("--sequence-length-weight", type=float, default=0.25)
    parser.add_argument("--sequence-eos-weight", type=float, default=0.10)
    parser.add_argument(
        "--whitespace-omission-penalty",
        type=float,
        default=0.0,
        help="penalize whitespace probability where the active target is non-whitespace",
    )
    parser.add_argument(
        "--adjacent-whitespace-penalty",
        type=float,
        default=0.0,
        help="penalize consecutive whitespace probability inside active target slots",
    )
    parser.add_argument(
        "--diagnose-route-conflict",
        action="store_true",
        help=(
            "at each logged step, backward two response slots' losses "
            "separately (retaining the graph) and log the cosine similarity "
            "of their gradients on shared routing/terminal parameters - "
            "diagnostic for cross-slot gradient interference on shared weights"
        ),
    )
    parser.add_argument(
        "--diagnose-route-conflict-slots",
        type=int,
        nargs=2,
        default=None,
        metavar=("SLOT_A", "SLOT_B"),
        help="which two response-slot indices to compare (default: first and last)",
    )
    parser.add_argument(
        "--diagnose-route-conflict-params",
        type=str,
        nargs="*",
        default=[
            "router_trunk_weight",
            "router_edge_key",
            "router_occupancy_weight",
            "router_bias",
            "router_weight",
            "terminal_votes",
            "terminal_vote_query",
            "shared_node_adapters",
        ],
        help="substrings selecting shared parameter names to measure conflict on",
    )
    parser.add_argument(
        "--reasoning-loops",
        type=int,
        default=1,
        choices=[1, 2],
        help=(
            "1 = normal single forward pass. 2 = decode once (no grad), feed "
            "that draft token back in one extra reserved context position, "
            "then decode again and train on the second pass only - a coarse "
            "mimic of iterative refinement (e.g. Gator's multi-relay depth), "
            "not a literal reproduction of it. Currently only supported for "
            "--architecture transformer with --decode-mode serial."
        ),
    )
    parser.add_argument(
        "--auto-lr-drop-on-conflict-cross",
        action="store_true",
        help=(
            "requires --diagnose-route-conflict; once an EMA of "
            "mean_cos_adjacent crosses from negative to >= "
            "--auto-lr-drop-threshold (after --auto-lr-drop-min-step), "
            "multiply all optimizer LRs by --auto-lr-drop-gamma once - "
            "an online, non-hindsight trigger for the same LR-freeze effect "
            "the milestone scheduler achieves with a hand-picked step"
        ),
    )
    parser.add_argument("--auto-lr-drop-threshold", type=float, default=0.0)
    parser.add_argument("--auto-lr-drop-min-step", type=int, default=500)
    parser.add_argument("--auto-lr-drop-gamma", type=float, default=0.02)
    parser.add_argument("--auto-lr-drop-ema-decay", type=float, default=0.7)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument(
        "--architecture",
        choices=["gator", "convergent", "transformer"],
        default="gator",
        help="simultaneous formation engine; both use the identical data pipeline",
    )
    parser.add_argument("--transformer-heads", type=int, default=2)
    parser.add_argument("--transformer-ff-width", type=int, default=76)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int,
                        help="language embedding width; defaults to d-model")
    parser.add_argument(
        "--lexical-embedding-dim",
        type=int,
        help=("factorized token-table width; defaults to language embedding "
              "width, while 0 explicitly disables recipe factorization"),
    )
    parser.add_argument(
        "--lexical-mode",
        choices=["private_hash", "communal"],
        default="private_hash",
        help="private modulo-16 cell prototypes or one shared vocabulary with cell identities",
    )
    parser.add_argument(
        "--cell-input-mode",
        choices=["shared_projection", "feature_gated"],
        default="shared_projection",
        help="one shared projected token view or learned per-cell feature gates",
    )
    parser.add_argument(
        "--cell-output-mode",
        choices=["shared_pool", "grouped_concat"],
        default="shared_pool",
        help="pool every cell to width d_model or concatenate grouped cell channels",
    )
    parser.add_argument("--cells", type=int, default=10)
    parser.add_argument(
        "--cell-transport-mode",
        choices=["broadcast_mix", "stewardship"],
        default="broadcast_mix",
        help="legacy broadcast coupling or semantic ownership/state handoffs",
    )
    parser.add_argument(
        "--stewardship-keep-mass",
        type=float,
        default=0.9,
        help="base adaptive nucleus mass retained for confident/familiar routes",
    )
    parser.add_argument(
        "--stewardship-handoff-keep-mass",
        type=float,
        default=0.9,
        help="base action-nucleus mass retained by each confident steward",
    )
    parser.add_argument("--stewardship-min-active", type=int, default=1)
    parser.add_argument(
        "--stewardship-max-active",
        type=int,
        default=0,
        help="maximum active cells per token; 0 permits the full population",
    )
    parser.add_argument(
        "--stewardship-exploration",
        type=float,
        default=0.01,
        help="legal-handoff probability floor and mass reserved for live scouts",
    )
    parser.add_argument(
        "--stewardship-scouts",
        type=int,
        default=1,
        help="alternate routes retained outside a familiar exploitation nucleus",
    )
    parser.add_argument(
        "--stewardship-execution",
        choices=["dense", "packed_auto"],
        default="dense",
        help="dense semantic reference or automatically packed live-route execution",
    )
    parser.add_argument(
        "--stewardship-packed-threshold",
        type=float,
        default=0.5,
        help="switch to packed execution when live branch fraction is at or below this value",
    )
    parser.add_argument(
        "--packed-stewardship-backend",
        choices=["repack", "persistent_bucketed"],
        default="persistent_bucketed",
        help=(
            "legacy pack/unpack oracle or persistent power-of-two bucketed "
            "sparse relay suffix"
        ),
    )
    parser.add_argument(
        "--stewardship-execution-check-every",
        type=int,
        default=100,
        help=(
            "completed steps between route-density observations used to latch "
            "a static packed relay plan"
        ),
    )
    parser.add_argument(
        "--receiver-memory-backend",
        choices=["attention", "convolution", "linear_scan"],
        default="convolution",
        help=(
            "quadratic attention reference, exact normalized convolution "
            "(still O(tokens^2) FLOPs despite avoiding one large tensor), "
            "or linear_scan (genuinely O(tokens), reinterprets "
            "relative_node_bias as decay-rate/mix parameters instead of a "
            "relative-position code) for receiver-specific node memory"
        ),
    )
    parser.add_argument(
        "--block-causal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "train original GTR-ARW with causal committed blocks and "
            "bidirectional refinement inside each block; supports broadcast_mix "
            "or dense stewardship transport and requires linear_scan memory plus a "
            "fixed --position-norm-length"
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        help=(
            "GTR-ARW causal/refinement block width; defaults to "
            "--response-tokens when --block-causal is enabled"
        ),
    )
    parser.add_argument(
        "--dense-router-backend",
        choices=["einsum", "grouped_bmm"],
        default="einsum",
        help=(
            "dense cell-router contraction; grouped_bmm is benchmark-gated "
            "because its cell-major transpose can outweigh GEMM gains"
        ),
    )
    parser.add_argument(
        "--dense-node-backend",
        choices=["einsum", "folded_bmm", "monarch"],
        default="folded_bmm",
        help=(
            "expanded node reference, exact wide projection with node "
            "embedding/bias folded into a relay-invariant offset, or "
            "'monarch' -- replaces the dense [cells,nodes,d,d] node_weight "
            "(~55M params at production shape) with a 2-factor Monarch-"
            "matrix structured factorization (~18x fewer params at the "
            "default --dense-node-monarch-blocks), an algorithmic FLOPs/"
            "parameter reduction rather than a kernel-efficiency change. "
            "Requires --node-compute-precision bf16 and "
            "--stewardship-execution dense; no production throughput/"
            "quality measurement yet."
        ),
    )
    parser.add_argument(
        "--dense-node-monarch-blocks",
        type=int,
        default=32,
        help=(
            "Block factor p for --dense-node-backend monarch's L:[.,.,q,p,p]/"
            "R:[.,.,p,q,q] factorization (q = d_model // p, must divide "
            "evenly). Only used when that backend is selected."
        ),
    )
    parser.add_argument(
        "--node-compute-precision",
        choices=["bf16", "mxfp8", "nvfp4", "fp8_te"],
        default="bf16",
        help=(
            "precision for the interior per-cell node compute (the "
            "expensive folded_bmm matmul), via gator/mx_precision.py. "
            "mxfp8/nvfp4 are straight-through fake-quantization (real "
            "gradients, real precision-loss signal, no hardware "
            "speedup, requires torchao). fp8_te is different: real "
            "hardware FP8 via NVIDIA Transformer Engine with CUDA-graph "
            "capture -- a real but modest speedup at production shape "
            "(measured ~1.03x) and real memory savings (~17%%), requires "
            "transformer_engine. Requires --dense-node-backend "
            "folded_bmm for any non-bf16 choice. Routing is never "
            "affected by this flag and always stays bf16."
        ),
    )
    parser.add_argument(
        "--readout-precision",
        choices=["bf16", "mxfp8", "mxfp4", "fp8_te"],
        default="bf16",
        help=(
            "precision for the vocabulary readout matmul, via "
            "gator/mx_precision.py. mxfp8/mxfp4 straight-through "
            "fake-quantization caveat same as --node-compute-precision. "
            "fp8_te is real hardware FP8 + CUDA-graph capture, but "
            "measured SLOWER than bf16 at production shape (~0.94x) "
            "once correctness is enforced (cloning the large vocab-sized "
            "output costs more than the graph saves) -- real memory "
            "savings (~23%%) hold up regardless; pick this only if "
            "memory, not speed, is the goal."
        ),
    )
    parser.add_argument(
        "--receiver-memory-cuda-graph",
        action="store_true",
        help=(
            "CUDA-graph-capture the linear_scan receiver-memory backend "
            "via gator/cuda_graph_receiver_memory.py. Requires "
            "--receiver-memory-backend linear_scan and a fixed "
            "--block-size (block-causal mode). No measured speedup at "
            "production shape (the scan is compute-bound there, not "
            "dispatch-bound) but a large real memory win (measured "
            "~83%% less peak memory in isolation) -- use this to free up "
            "VRAM for a larger batch size, not for raw throughput."
        ),
    )
    parser.add_argument(
        "--receiver-memory-hgrn-scan",
        action="store_true",
        help=(
            "Replace the block-causal forward (cache-carrying) scan in "
            "the linear_scan receiver-memory backend with "
            "flash-linear-attention's fused chunk_hgrn Triton kernel "
            "(`pip install flash-linear-attention`), via "
            "gator_linear.core.chunk_hgrn_decayed_scan. Requires "
            "--receiver-memory-backend linear_scan and a fixed "
            "--block-size; not compatible with "
            "--receiver-memory-cuda-graph yet. Measured ~5.5x faster "
            "than the existing chunked scan at production shape in "
            "isolation, verified correct to ~1e-6 forward/gradient diff."
        ),
    )
    parser.add_argument(
        "--broadcast-mix-cuda-graph",
        action="store_true",
        help=(
            "CUDA-graph-capture the broadcast_mix cell-coupling block "
            "(roll/stack/gate/project/norm) via "
            "gator/cuda_graph_broadcast_mix.py. Requires "
            "--cell-transport-mode broadcast_mix. Targets dispatch "
            "overhead from this block's several small per-relay ops, "
            "same mechanism as --receiver-memory-cuda-graph -- not yet "
            "measured at production shape."
        ),
    )
    parser.add_argument(
        "--stewardship-traffic-cuda-graph",
        action="store_true",
        help=(
            "CUDA-graph-capture the block-causal cumulative traffic "
            "bookkeeping inside _dense_stewardship_context via "
            "gator/cuda_graph_stewardship_traffic.py. Requires "
            "--cell-transport-mode broadcast_mix and a fixed "
            "--block-size (block-causal mode). Not yet measured at "
            "production shape."
        ),
    )
    parser.add_argument(
        "--dense-kernel-padding",
        type=int,
        default=1,
        help=(
            "parameter-free feature-dimension padding multiple; 1 disables "
            "manual padding and is the safe default"
        ),
    )
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help=(
            "recompute each dense relay's router/node computation during "
            "backward instead of keeping it resident; trades compute time "
            "for the peak memory that otherwise scales with cell count and "
            "context length (--architecture gator only)"
        ),
    )
    parser.add_argument(
        "--stewardship-warmup-fraction",
        type=float,
        default=0.05,
        help="fraction of training that keeps stewardship fully dense",
    )
    parser.add_argument("--stewardship-temperature-start", type=float, default=1.5)
    parser.add_argument("--stewardship-temperature-end", type=float, default=0.35)
    parser.add_argument(
        "--stewardship-familiar-entropy-penalty", type=float, default=0.001
    )
    parser.add_argument(
        "--stewardship-balance-penalty", type=float, default=0.01
    )
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument(
        "--graph-degree",
        type=int,
        default=3,
        help="outgoing connections per node for convergent Gator",
    )
    parser.add_argument(
        "--router-mode",
        choices=["private_full", "shared_low_rank"],
        default="private_full",
        help=(
            "private_full keeps one complete router per node; "
            "shared_low_rank computes one packet feature trunk and lets "
            "private node-edge keys choose local destinations"
        ),
    )
    parser.add_argument(
        "--router-rank",
        type=int,
        default=16,
        help="shared packet-feature width for shared_low_rank routing",
    )
    parser.add_argument(
        "--ingress-mode",
        choices=["fixed", "learned"],
        default="fixed",
        help=(
            "fixed seeds each lane at one static node; learned chooses "
            "packet-conditioned entry nodes from the complete graph"
        ),
    )
    parser.add_argument(
        "--active-node-top-k",
        type=int,
        default=None,
        help=(
            "optional fixed-cap number of node locations retained per "
            "packet; selected private node transforms are executed directly"
        ),
    )
    parser.add_argument(
        "--route-backend",
        choices=["dense", "packed"],
        default="dense",
        help=(
            "dense passport oracle or fixed-cap packed node frontiers; "
            "packed requires --active-node-top-k"
        ),
    )
    parser.add_argument(
        "--packed-memory-backend",
        choices=["matrix", "convolution", "fft", "fft_opaque"],
        default="matrix",
        help=(
            "packed Convergent receiver memory: dense matrix oracle, grouped "
            "convolution, eager FFT, or compile-safe opaque FFT"
        ),
    )
    parser.add_argument(
        "--packed-memory-reader-backend",
        choices=["torch", "triton"],
        default="torch",
        help=(
            "packed convolution/FFT reader reduction: portable Torch oracle "
            "or explicit fused Triton forward/backward"
        ),
    )
    parser.add_argument(
        "--packed-memory-writer-backend",
        choices=["torch", "triton"],
        default="torch",
        help=(
            "packed FFT writer scatter: portable Torch oracle or explicit "
            "fused Triton forward/backward"
        ),
    )
    parser.add_argument(
        "--packed-memory-feature-chunk",
        type=int,
        default=32,
        help="hidden features processed per packed receiver-memory pass",
    )
    parser.add_argument(
        "--packed-memory-compact-nodes",
        action="store_true",
        help=(
            "eager-only dynamic active-node union for receiver memory; leave "
            "off with --compile until fixed-cap compaction is available"
        ),
    )
    parser.add_argument(
        "--adaptive-route-min-k",
        type=int,
        default=None,
        help=(
            "minimum live routes per packet; uncertainty expands toward "
            "--active-node-top-k for unfamiliar packets"
        ),
    )
    parser.add_argument(
        "--adaptive-route-mode",
        choices=["entropy", "margin", "hybrid"],
        default="hybrid",
    )
    parser.add_argument(
        "--adaptive-route-entropy-mix",
        type=float,
        default=0.5,
        help="entropy weight in hybrid uncertainty routing",
    )
    parser.add_argument(
        "--compact-node-dispatch",
        action="store_true",
        help=(
            "eager-only removal of zero-mass adaptive routes before private "
            "node GEMMs"
        ),
    )
    parser.add_argument(
        "--node-compute-mode",
        choices=["full_private", "shared_adapter"],
        default="full_private",
        help=(
            "full private D-by-D node transforms or one shared trunk with "
            "private low-rank node adapters"
        ),
    )
    parser.add_argument(
        "--node-adapter-rank",
        type=int,
        default=8,
        help="private node adapter rank in shared_adapter mode",
    )
    parser.add_argument(
        "--ingress-groups",
        type=int,
        default=None,
        help="optional number of learned coarse groups before node ingress",
    )
    parser.add_argument(
        "--ingress-top-groups",
        type=int,
        default=2,
        help="coarse ingress groups retained per packet",
    )
    parser.add_argument(
        "--no-ingress-learned-groups",
        action="store_false",
        dest="ingress_learned_groups",
        help="derive group keys from private node keys instead of adding parameters",
    )
    parser.add_argument(
        "--active-node-mass-threshold",
        type=float,
        default=0.0,
        help=(
            "discard retained node locations below this probability mass; "
            "zero preserves dense route mass"
        ),
    )
    parser.add_argument(
        "--lanes",
        type=int,
        default=10,
        help="persistent complementary traces for convergent Gator",
    )
    parser.add_argument(
        "--terminal-vote-mode",
        choices=["global", "prompt", "slot"],
        default="prompt",
        help=(
            "one static lane jury, one prompt-conditioned jury shared across "
            "all output slots, or one prompt-and-address-aware jury per slot"
        ),
    )
    parser.add_argument(
        "--force-causal",
        action="store_true",
        help=(
            "force causal=True on every training forward pass, regardless "
            "of --decode-mode (normally causal is only auto-enabled for "
            "teacher_forced/prefix_denoise/token_handoff decode modes, and "
            "left off for serial/parallel/chunk_parallel). Requires "
            "--terminal-vote-mode global or slot, since causal execution is "
            "incompatible with prompt-mode terminal voting. Only affects "
            "the training loop's forward passes; periodic validation-set "
            "evaluation still uses its own decode-mode-driven causal default"
        ),
    )
    parser.add_argument(
        "--slot-isolation-slots",
        type=int,
        default=0,
        help=(
            "experimental balanced slot ownership for private node compute "
            "and terminal juries; zero keeps shared computation"
        ),
    )
    parser.add_argument(
        "--lane-vote-noise-std",
        type=float,
        default=0.0,
        help=(
            "gaussian noise standard deviation added to prompt/slot terminal "
            "vote logits during training only, to discourage premature "
            "winner-take-all lane lock-in; zero disables"
        ),
    )
    parser.add_argument(
        "--position-norm-length",
        type=int,
        help=(
            "fix the receiver-kernel/address position-encoding normalizer to "
            "this total sequence length instead of the current forward "
            "call's length; makes the encoding length-invariant so cached/"
            "extended-context inference doesn't rescale already-computed "
            "positions. Defaults to None (dynamic, the historical behavior). "
            "Set this to context-tokens + response-tokens to keep training "
            "numerically identical to the dynamic default at the current "
            "context length while fixing the scale for future context growth"
        ),
    )
    parser.add_argument(
        "--fp8-vocab-readout",
        action="store_true",
        help=(
            "compute the tied vocab readout matmul (readout @ token_embedding.T) "
            "in fp8 (dynamic per-tensor scaling via torchao) instead of the "
            "default dtype. Requires a Blackwell/Hopper-class GPU. Forward-only "
            "fp8 (backward stays bf16). Only wired up for --architecture "
            "convergent. 2026-07-28 benchmarks: ~1.4x speedup at vocab_size=20000 "
            "(too small a matmul to help at Gator's node/router dimensions, "
            "hence vocab-only), but ~13x worse relative numerical error than "
            "bf16 - unvalidated whether that precision cost affects learning."
        ),
    )
    parser.add_argument(
        "--lane-balance-loss-weight",
        type=float,
        default=0.0,
        help=(
            "weight on an entropy-maximizing regularizer over the terminal "
            "lane-vote distribution (prompt/slot vote modes only), penalizing "
            "collapse onto a single winning lane; zero disables"
        ),
    )
    parser.add_argument(
        "--connection-layout",
        choices=["ring", "small_world", "dense"],
        default="small_world",
        help="fixed convergent graph connection layout",
    )
    parser.add_argument("--connection-seed", type=int, default=0)
    parser.add_argument(
        "--connection-rewire",
        type=float,
        default=0.5,
        help="small-world shortcut probability for convergent Gator",
    )
    parser.add_argument(
        "--node-topology",
        choices=[
            "asymmetric",
            "symmetric",
            "hybrid_alternating",
            "hybrid_hemispheric",
        ],
        default="asymmetric",
    )
    parser.add_argument("--relays", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        help="limit PyTorch's CUDA allocator to this many decimal GB",
    )
    parser.add_argument(
        "--execution-preset",
        choices=sorted(_EXECUTION_PRESET_OVERRIDES),
        default="manual",
        help=(
            "parameter-free runtime policy: *_safe uses strict FP32 without "
            "compilation, dense_gpu is the aggressive compiled dense A/B, "
            "optimized_gpu adds automatic persistent packed execution, and "
            "manual preserves individual defaults; explicitly supplied "
            "low-level flags override a preset"
        ),
    )
    add_compute_arguments(parser)
    add_optimization_arguments(parser)
    add_data_pipeline_arguments(parser)
    raw_arguments = sys.argv[1:]
    explicit_destinations = explicit_cli_destinations(
        parser,
        raw_arguments,
    )
    args = parser.parse_args(raw_arguments)
    try:
        apply_execution_preset(
            args,
            explicit_destinations=explicit_destinations,
        )
        compute_config = compute_config_from_args(args)
        preparation_config = preparation_config_from_args(args)
    except ValueError as error:
        parser.error(str(error))

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    if args.lexical_embedding_dim is not None and args.lexical_embedding_dim < 0:
        parser.error("--lexical-embedding-dim cannot be negative")
    if args.training_recipe == "smollm3-26b":
        from gator.smollm3_recipe import (
            accumulation_steps_for_batch,
            total_updates_for_sequence_length,
        )

        if args.smollm3_effective_batch_sequences <= 0:
            parser.error("--smollm3-effective-batch-sequences must be positive")
        effective_batch = args.smollm3_effective_batch_sequences

        batch_explicit = "batch_size" in explicit_destinations
        accumulation_explicit = (
            "gradient_accumulation_steps" in explicit_destinations
        )
        if not batch_explicit and not accumulation_explicit:
            args.batch_size = 1
            args.gradient_accumulation_steps = effective_batch
        elif batch_explicit and not accumulation_explicit:
            try:
                args.gradient_accumulation_steps = accumulation_steps_for_batch(
                    args.batch_size, effective_batch
                )
            except ValueError as error:
                parser.error(str(error))
        elif accumulation_explicit and not batch_explicit:
            if (
                args.gradient_accumulation_steps <= 0
                or effective_batch
                % args.gradient_accumulation_steps
            ):
                parser.error(
                    "gradient accumulation must divide the configured "
                    "effective batch (--smollm3-effective-batch-sequences)"
                )
            args.batch_size = (
                effective_batch
                // args.gradient_accumulation_steps
            )
        recipe_defaults = {
            "mixed_precision": "bf16",
            "steps": total_updates_for_sequence_length(
                args.smollm3_sequence_length, effective_batch
            ),
            "log_every": 10,
            "checkpoint_every_updates": 5_000,
            "log_every_seconds": 900.0,
            "lexical_embedding_dim": 256,
        }
        for destination, value in recipe_defaults.items():
            if destination not in explicit_destinations:
                setattr(args, destination, value)

        from gator.smollm3_train import run as run_smollm3_recipe
        torch.manual_seed(args.seed)
        recipe_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        run_smollm3_recipe(args, parser, device=recipe_device)
        return
    if not args.data:
        parser.error(
            "--data is required unless --training-recipe smollm3-26b is selected"
        )
    if args.repeat_training_examples < 0:
        parser.error("--repeat-training-examples cannot be negative")
    if args.max_runtime_seconds < 0:
        parser.error("--max-runtime-seconds cannot be negative")
    if args.log_every_seconds < 0:
        parser.error("--log-every-seconds cannot be negative")
    if args.reference_tokens < 0 or args.reference_tokens > args.response_tokens:
        parser.error("--reference-tokens must be 0 or at most --response-tokens")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.chunk_size > args.response_tokens:
        parser.error("--chunk-size cannot exceed --response-tokens")
    if args.serial_positions_per_example < 1:
        parser.error("--serial-positions-per-example must be >= 1")
    if args.decode_mode == "serial" and args.formation_objective != "slot_ce":
        parser.error(
            "--decode-mode serial currently requires "
            "--formation-objective slot_ce"
        )
    if args.reasoning_loops == 2:
        if args.architecture not in {"transformer", "convergent"}:
            parser.error(
                "--reasoning-loops 2 currently requires --architecture "
                "transformer or convergent"
            )
        if args.decode_mode != "serial":
            parser.error(
                "--reasoning-loops 2 currently requires --decode-mode serial"
            )
    if args.decode_mode in {
        "chunk_parallel",
        "teacher_forced",
        "handoff",
        "graduated_handoff",
        "prefix_denoise",
    }:
        if args.architecture != "convergent":
            parser.error(
                f"--decode-mode {args.decode_mode} currently requires "
                "--architecture convergent"
            )
        if args.formation_objective != "slot_ce":
            parser.error(
                f"--decode-mode {args.decode_mode} currently requires "
                "--formation-objective slot_ce"
            )
        if args.terminal_vote_mode == "prompt":
            parser.error(
                f"--decode-mode {args.decode_mode} requires global or slot "
                "terminal voting"
            )
    if not (
        0.0
        <= args.handoff_start_content_accuracy
        < args.handoff_end_content_accuracy
        <= 1.0
    ):
        parser.error(
            "handoff content accuracies must satisfy "
            "0 <= start < end <= 1"
        )
    if not 0.0 <= args.handoff_accuracy_ema_decay < 1.0:
        parser.error("--handoff-accuracy-ema-decay must be in [0, 1)")
    if not 0.0 < args.graduated_baseline_fraction < 1.0:
        parser.error("--graduated-baseline-fraction must be in (0, 1)")
    if not 0.0 < args.graduated_accuracy_step <= 1.0:
        parser.error("--graduated-accuracy-step must be in (0, 1]")
    if not 0.0 < args.graduated_release_accuracy <= 1.0:
        parser.error("--graduated-release-accuracy must be in (0, 1]")
    maximum_token_removals = (
        args.response_tokens - 1
        if args.token_handoff_max_removed == 0
        else args.token_handoff_max_removed
    )
    if not (
        0
        <= args.token_handoff_start_removed
        <= maximum_token_removals
        <= args.response_tokens - 1
    ):
        parser.error(
            "token handoff removals must satisfy 0 <= start <= maximum "
            "<= response_tokens - 1"
        )
    if not (
        0.0
        < args.token_handoff_retreat_ratio
        < args.token_handoff_advance_ratio
        <= 1.0
    ):
        parser.error(
            "token handoff ratios must satisfy 0 < retreat < advance <= 1"
        )
    if not 0.0 <= args.token_handoff_min_teacher_accuracy <= 1.0:
        parser.error("--token-handoff-min-teacher-accuracy must be in [0, 1]")
    if not 0.0 <= args.token_handoff_min_frontier_accuracy <= 1.0:
        parser.error("--token-handoff-min-frontier-accuracy must be in [0, 1]")
    teacher_ppl_guard_disabled = (
        args.token_handoff_max_teacher_ppl_ratio == 0.0
        and args.token_handoff_retreat_teacher_ppl_ratio == 0.0
    )
    if not teacher_ppl_guard_disabled and not (
        math.isfinite(args.token_handoff_max_teacher_ppl_ratio)
        and math.isfinite(args.token_handoff_retreat_teacher_ppl_ratio)
        and 1.0
        <= args.token_handoff_max_teacher_ppl_ratio
        < args.token_handoff_retreat_teacher_ppl_ratio
    ):
        parser.error(
            "token handoff teacher-PPL ratios must both be zero or satisfy "
            "1 <= maximum < retreat"
        )
    if (
        args.token_handoff_advance_patience <= 0
        or args.token_handoff_retreat_patience <= 0
    ):
        parser.error("token handoff patience must be positive")
    if not 0.0 <= args.token_handoff_full_probe_fraction <= 1.0:
        parser.error("--token-handoff-full-probe-fraction must be in [0, 1]")
    if args.token_handoff_unforced_loss_weight < 0.0:
        parser.error("--token-handoff-unforced-loss-weight cannot be negative")
    optimizer_component_lr_multipliers = {
        "lexical": args.optimizer_lexical_lr_multiplier,
        "lane_core": args.optimizer_lane_core_lr_multiplier,
        "private_node": args.optimizer_private_node_lr_multiplier,
        "route_gate": args.optimizer_route_gate_lr_multiplier,
    }
    optimizer_train_components = tuple(
        dict.fromkeys(args.optimizer_train_components)
    )
    if len(optimizer_train_components) != len(args.optimizer_train_components):
        parser.error("--optimizer-train-components cannot contain duplicates")
    if any(
        value <= 0.0 or not math.isfinite(value)
        for value in optimizer_component_lr_multipliers.values()
    ):
        parser.error("optimizer LR multipliers must be finite and positive")
    if args.optimizer_groups == "component_scaled":
        if args.architecture != "convergent":
            parser.error(
                "--optimizer-groups component_scaled requires "
                "--architecture convergent"
            )
        if args.optimizer != "adamw":
            parser.error(
                "--optimizer-groups component_scaled currently requires AdamW"
            )
    elif (
        set(optimizer_train_components) != set(OPTIMIZER_COMPONENTS)
        or args.optimizer_retain_frozen_state
        or args.optimizer_freeze_route_transport
    ):
        parser.error(
            "optimizer component freezing requires "
            "--optimizer-groups component_scaled"
        )
    route_temperatures = (
        args.route_temperature_start,
        args.route_temperature_middle,
        args.route_temperature_end,
    )
    if any(
        value <= 0.0 or not math.isfinite(value)
        for value in route_temperatures
    ):
        parser.error("route temperatures must be finite and positive")
    if not (
        0.0
        <= args.route_temperature_hold_fraction
        < args.route_temperature_middle_fraction
        <= 1.0
    ):
        parser.error(
            "route temperature fractions must satisfy "
            "0 <= hold < middle <= 1"
        )
    if args.route_temperature_step_offset < 0:
        parser.error("--route-temperature-step-offset cannot be negative")
    if args.route_temperature_total_steps < 0:
        parser.error("--route-temperature-total-steps cannot be negative")
    if (
        args.route_temperature_schedule == "staged_reheat"
        and args.architecture != "convergent"
    ):
        parser.error(
            "--route-temperature-schedule staged_reheat requires "
            "--architecture convergent"
        )
    if not (
        0.0
        <= args.prefix_retention_floor
        <= args.prefix_retention_start
        <= 1.0
    ):
        parser.error(
            "prefix retention must satisfy 0 <= floor <= start <= 1"
        )
    if args.prefix_denoise_warmup_steps <= 0:
        parser.error("--prefix-denoise-warmup-steps must be positive")
    if not (
        0.0 <= args.prefix_zero_row_fraction <= 1.0
        and 0.0 <= args.prefix_short_row_fraction <= 1.0
        and args.prefix_zero_row_fraction
        + args.prefix_short_row_fraction
        <= 1.0
    ):
        parser.error(
            "prefix zero/short row fractions must each be in [0, 1] "
            "and sum to at most one"
        )
    if args.prefix_short_max_tokens <= 0:
        parser.error("--prefix-short-max-tokens must be positive")
    if not 0.0 <= args.prefix_span_fraction <= 1.0:
        parser.error("--prefix-span-fraction must be in [0, 1]")
    if not (
        0.0
        <= args.prefix_context_utility_start
        < args.prefix_context_utility_end
        <= 1.0
    ):
        parser.error(
            "prefix context utility must satisfy 0 <= start < end <= 1"
        )
    if args.whitespace_omission_penalty < 0:
        parser.error("--whitespace-omission-penalty cannot be negative")
    if args.adjacent_whitespace_penalty < 0:
        parser.error("--adjacent-whitespace-penalty cannot be negative")
    if not 0.0 <= args.alignment_warmup_fraction < 1.0:
        parser.error("--alignment-warmup-fraction must be in [0, 1)")
    if args.alignment_max_shift < -1:
        parser.error("--alignment-max-shift must be -1 or nonnegative")
    if args.sequence_quality_penalty < 0:
        parser.error("--sequence-quality-penalty cannot be negative")
    if min(
        args.sequence_composition_weight,
        args.sequence_length_weight,
        args.sequence_eos_weight,
    ) < 0:
        parser.error("sequence-quality component weights cannot be negative")
    if (
        args.formation_objective in {"viterbi", "monotonic"}
        and (
            args.whitespace_omission_penalty > 0
            or args.adjacent_whitespace_penalty > 0
        )
    ):
        parser.error(
            "rigid-slot whitespace penalties are not valid for aligned objectives; "
            "set both whitespace penalties to zero"
        )
    if not 0.0 < args.stewardship_keep_mass <= 1.0:
        parser.error("--stewardship-keep-mass must be in (0, 1]")
    if not 0.0 < args.stewardship_handoff_keep_mass <= 1.0:
        parser.error("--stewardship-handoff-keep-mass must be in (0, 1]")
    if args.stewardship_min_active <= 0:
        parser.error("--stewardship-min-active must be positive")
    if args.stewardship_max_active < 0:
        parser.error("--stewardship-max-active cannot be negative")
    if not 0.0 <= args.stewardship_exploration < 1.0:
        parser.error("--stewardship-exploration must be in [0, 1)")
    if not 0 <= args.stewardship_scouts < args.cells:
        parser.error("--stewardship-scouts must be between zero and cells - 1")
    if not 0.0 < args.stewardship_packed_threshold <= 1.0:
        parser.error("--stewardship-packed-threshold must be in (0, 1]")
    if args.stewardship_execution_check_every <= 0:
        parser.error("--stewardship-execution-check-every must be positive")
    if not 0.0 <= args.stewardship_warmup_fraction < 1.0:
        parser.error("--stewardship-warmup-fraction must be in [0, 1)")
    if args.stewardship_temperature_start <= 0 or args.stewardship_temperature_end <= 0:
        parser.error("stewardship temperatures must be positive")
    if args.stewardship_familiar_entropy_penalty < 0 or args.stewardship_balance_penalty < 0:
        parser.error("stewardship loss weights cannot be negative")
    if args.repeat_training_examples and (
        args.tinystories_data
        or args.lmsys_data
        or args.chat_data
        or args.prose_data
        or args.primary_stream_weight > 0
    ):
        parser.error(
            "--repeat-training-examples requires SmolTalk-only training; "
            "remove streamed chat sources so none enter the fixed pool"
        )
    if not 0.0 < args.tinystories_mix_fraction < 1.0:
        parser.error("--tinystories-mix-fraction must be in (0, 1)")
    if args.tinystories_validation_data and not args.tinystories_data:
        parser.error(
            "--tinystories-validation-data requires --tinystories-data"
        )
    if args.tinystories_validation_stories < 0:
        parser.error("--tinystories-validation-stories cannot be negative")
    if args.tinystories_tokenizer_stories <= 0:
        parser.error("--tinystories-tokenizer-stories must be positive")
    if args.tinystories_shuffle_buffer <= 0:
        parser.error("--tinystories-shuffle-buffer must be positive")
    streaming_weight = (
        args.lmsys_weight
        if args.streaming_weight is None
        else args.streaming_weight
    )
    if not 0.0 <= streaming_weight <= 1.0:
        parser.error("--streaming-weight/--lmsys-weight must be between 0 and 1")
    if (
        args.streaming_exhaustion == "drop"
        and args.lmsys_data
        and args.lmsys_load_mode != "stream"
    ):
        parser.error(
            "--streaming-exhaustion drop requires --lmsys-load-mode stream"
        )
    if args.lmsys_source_weight <= 0 or not math.isfinite(
        args.lmsys_source_weight
    ):
        parser.error("--lmsys-source-weight must be finite and positive")
    if args.primary_stream_weight < 0 or not math.isfinite(
        args.primary_stream_weight
    ):
        parser.error("--primary-stream-weight must be finite and nonnegative")
    if args.primary_shuffle_buffer <= 0:
        parser.error("--primary-shuffle-buffer must be positive")
    if args.chat_source_weight and (
        len(args.chat_source_weight) != len(args.chat_data)
    ):
        parser.error(
            "repeat --chat-source-weight exactly once per --chat-data source, "
            "or omit all chat source weights"
        )
    if not args.chat_source_weight:
        args.chat_source_weight = [1.0] * len(args.chat_data)
    if any(
        weight <= 0 or not math.isfinite(weight)
        for weight in args.chat_source_weight
    ):
        parser.error("--chat-source-weight values must be finite and positive")
    if args.chat_tokenizer_conversations < 0:
        parser.error("--chat-tokenizer-conversations cannot be negative")
    if args.chat_max_characters is not None and args.chat_max_characters <= 0:
        parser.error("--chat-max-characters must be positive")
    if args.chat_shuffle_buffer <= 0:
        parser.error("--chat-shuffle-buffer must be positive")
    if args.prose_source_weight and (
        len(args.prose_source_weight) != len(args.prose_data)
    ):
        parser.error(
            "repeat --prose-source-weight exactly once per --prose-data source, "
            "or omit all prose source weights"
        )
    if not args.prose_source_weight:
        args.prose_source_weight = [1.0] * len(args.prose_data)
    if any(
        weight <= 0 or not math.isfinite(weight)
        for weight in args.prose_source_weight
    ):
        parser.error("--prose-source-weight values must be finite and positive")
    if args.prose_tokenizer_stories < 0:
        parser.error("--prose-tokenizer-stories cannot be negative")
    if args.prose_validation_stories < 0:
        parser.error("--prose-validation-stories cannot be negative")
    if args.chat_validation_conversations < 0:
        parser.error("--chat-validation-conversations cannot be negative")
    if args.prose_shuffle_buffer <= 0:
        parser.error("--prose-shuffle-buffer must be positive")
    if args.resume_optimizer and not args.resume_checkpoint:
        parser.error("--resume-optimizer requires --resume-checkpoint")
    if args.grow_from_checkpoint:
        if args.resume_checkpoint:
            parser.error("--grow-from-checkpoint and --resume-checkpoint are mutually exclusive")
        if args.architecture != "gator":
            parser.error("--grow-from-checkpoint is only wired up for --architecture gator")
        if args.grow_clones_per_cell < 2:
            parser.error("--grow-clones-per-cell must be at least 2")
        if args.grow_perturb_scale < 0:
            parser.error("--grow-perturb-scale cannot be negative")
    if not 0.0 <= args.lmsys_min_ascii_letter_ratio <= 1.0:
        parser.error("--lmsys-min-ascii-letter-ratio must be between 0 and 1")
    if args.lmsys_min_response_characters < 0:
        parser.error("--lmsys-min-response-characters cannot be negative")
    if args.dense_kernel_padding <= 0:
        parser.error("--dense-kernel-padding must be positive")
    if args.graph_degree <= 0:
        parser.error("--graph-degree must be positive")
    if args.router_rank <= 0:
        parser.error("--router-rank must be positive")
    if args.packed_memory_feature_chunk <= 0:
        parser.error("--packed-memory-feature-chunk must be positive")
    if args.route_backend != "packed" and args.packed_memory_backend != "matrix":
        parser.error(
            "--packed-memory-backend convolution/fft/fft_opaque requires "
            "--route-backend packed"
        )
    if (
        args.packed_memory_reader_backend == "triton"
        and args.packed_memory_backend == "matrix"
    ):
        parser.error(
            "--packed-memory-reader-backend triton requires convolution, "
            "fft, or fft_opaque memory"
        )
    if (
        args.packed_memory_writer_backend == "triton"
        and (
            args.route_backend != "packed"
            or args.packed_memory_backend != "fft_opaque"
        )
    ):
        parser.error(
            "--packed-memory-writer-backend triton currently requires "
            "--route-backend packed and --packed-memory-backend fft_opaque"
        )
    if (
        args.packed_memory_writer_backend == "triton"
        and args.packed_memory_feature_chunk > 128
    ):
        parser.error(
            "--packed-memory-writer-backend triton supports feature chunks "
            "up to 128"
        )
    if args.compile and args.packed_memory_backend == "fft":
        parser.error(
            "--packed-memory-backend fft cannot currently be combined with "
            "--compile because TorchInductor cannot code-generate the "
            "complex64 FFT graph; use fft_opaque or convolution with "
            "--compile, or raw fft with --no-compile"
        )
    if args.packed_memory_compact_nodes and args.compile:
        parser.error(
            "--packed-memory-compact-nodes is eager-only and cannot be used "
            "with --compile"
        )
    if args.compact_node_dispatch and args.compile:
        parser.error("--compact-node-dispatch is eager-only with current kernels")
    if args.node_adapter_rank <= 0:
        parser.error("--node-adapter-rank must be positive")
    if args.node_compute_mode == "shared_adapter" and args.route_backend != "packed":
        parser.error("--node-compute-mode shared_adapter requires packed routes")
    if args.ingress_top_groups <= 0:
        parser.error("--ingress-top-groups must be positive")
    if args.ingress_groups is not None:
        if not 1 <= args.ingress_groups <= args.nodes:
            parser.error("--ingress-groups must be in [1, --nodes]")
        if args.route_backend != "packed" or args.ingress_mode != "learned":
            parser.error("hierarchical ingress requires packed learned ingress")
        if args.ingress_top_groups > args.ingress_groups:
            parser.error("--ingress-top-groups cannot exceed --ingress-groups")
    if not 0.0 <= args.adaptive_route_entropy_mix <= 1.0:
        parser.error("--adaptive-route-entropy-mix must be in [0, 1]")
    if args.adaptive_route_min_k is not None:
        if args.route_backend != "packed" or args.active_node_top_k is None:
            parser.error(
                "--adaptive-route-min-k requires --route-backend packed and "
                "--active-node-top-k"
            )
        if not 1 <= args.adaptive_route_min_k <= args.active_node_top_k:
            parser.error(
                "--adaptive-route-min-k must be in [1, --active-node-top-k]"
            )
    if args.ingress_mode == "learned" and args.router_mode != "shared_low_rank":
        parser.error(
            "--ingress-mode learned requires --router-mode shared_low_rank"
        )
    if args.active_node_top_k is not None and not (
        1 <= args.active_node_top_k <= args.nodes
    ):
        parser.error("--active-node-top-k must be in [1, --nodes]")
    if not 0.0 <= args.active_node_mass_threshold < 1.0:
        parser.error(
            "--active-node-mass-threshold must be in [0, 1)"
        )
    if args.lanes <= 0:
        parser.error("--lanes must be positive")
    if not 0.0 <= args.connection_rewire <= 1.0:
        parser.error("--connection-rewire must be between zero and one")
    if args.position_norm_length is not None:
        if args.position_norm_length <= 1:
            parser.error("--position-norm-length must be greater than 1")
        if args.architecture not in {"gator", "convergent"}:
            parser.error(
                "--position-norm-length is only wired up for "
                "--architecture gator or convergent"
            )
    if args.block_size is not None and args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.block_causal:
        if args.architecture != "gator":
            parser.error(
                "--block-causal is only wired up for --architecture gator"
            )
        if args.receiver_memory_backend != "linear_scan":
            parser.error(
                "--block-causal requires --receiver-memory-backend linear_scan"
            )
        if args.position_norm_length is None:
            parser.error("--block-causal requires --position-norm-length")
        resolved_block_size = (
            args.response_tokens if args.block_size is None else args.block_size
        )
        formation_length = (
            args.smollm3_sequence_length
            if args.training_recipe == "smollm3-26b"
            else args.context_tokens + args.response_tokens
        )
        if formation_length % resolved_block_size:
            parser.error(
                "the training formation length must be divisible by the block size"
            )
        if args.position_norm_length < formation_length:
            parser.error(
                "--position-norm-length cannot be shorter than the training formation"
            )
        if (
            args.cell_transport_mode == "stewardship"
            and args.stewardship_execution != "dense"
        ):
            parser.error(
                "--block-causal currently requires --stewardship-execution dense"
            )
    elif args.block_size is not None:
        parser.error("--block-size requires --block-causal")
    if args.fp8_vocab_readout and args.architecture != "convergent":
        parser.error("--fp8-vocab-readout is only wired up for --architecture convergent")
    if args.force_causal:
        if args.architecture != "convergent":
            parser.error("--force-causal is only wired up for --architecture convergent")
        if args.terminal_vote_mode not in {"global", "slot"}:
            parser.error(
                "--force-causal requires --terminal-vote-mode global or slot"
            )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (
        args.architecture == "convergent"
        and args.route_backend == "packed"
        and args.node_compute_mode == "full_private"
        and args.compile
    ):
        grouped_private_supported = (
            device.type == "cuda"
            and hasattr(torch, "_grouped_mm")
            and torch.cuda.get_device_capability(device) >= (8, 0)
            and (args.d_model * torch.bfloat16.itemsize) % 16 == 0
        )
        if not grouped_private_supported:
            parser.error(
                "compiled packed full-private nodes require an aligned "
                "CUDA grouped-MM device; use eager execution or "
                "--node-compute-mode shared_adapter on this device"
            )
    try:
        pipeline_config = data_pipeline_config_from_args(
            args,
            batch_size=args.batch_size,
            device=device,
        )
    except ValueError as error:
        parser.error(str(error))
    resolved_runtime = configure_float32_runtime(
        compute_config.float32,
        device,
    )
    if args.max_vram_gb is not None:
        if args.max_vram_gb <= 0:
            parser.error("--max-vram-gb must be positive")
        if device.type == "cuda":
            cuda_device = torch.cuda.current_device()
            total_vram = torch.cuda.get_device_properties(cuda_device).total_memory
            requested_vram = args.max_vram_gb * 1_000_000_000
            allocator_fraction = min(1.0, requested_vram / total_vram)
            torch.cuda.set_per_process_memory_fraction(allocator_fraction, cuda_device)
            print(
                f"gpu={torch.cuda.get_device_name(cuda_device)} "
                f"total_vram_gb={total_vram / 1e9:.1f} "
                f"allocator_cap_gb={min(args.max_vram_gb, total_vram / 1e9):.1f}",
                flush=True,
            )
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_state = None
    start_step = 0
    resume_decode_mode = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.is_file():
            parser.error(f"resume checkpoint does not exist: {resume_path}")
        resume_state = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        if resume_state.get("format") not in {
            "gator-language-v1",
            "gator-language-v2",
        }:
            parser.error("unsupported resume checkpoint format")
        start_step = int(resume_state.get("step", 0))
        if start_step < 0:
            parser.error("resume checkpoint has an invalid negative step")
        resume_decode_mode = (
            resume_state.get("data_config", {}).get("decode_mode")
        )

    task_config = LanguageBatchConfig(
        context_tokens=args.context_tokens,
        response_tokens=args.response_tokens,
        reference_tokens=(
            None if args.reference_tokens == 0 else args.reference_tokens
        ),
    )
    if resume_state is not None:
        checkpoint_task = resume_state.get("task_config", {})
        expected_task = {
            "context_tokens": task_config.context_tokens,
            "response_tokens": task_config.response_tokens,
            "reference_tokens": task_config.target_tokens,
        }
        mismatched_task = {
            key: (checkpoint_task.get(key), value)
            for key, value in expected_task.items()
            if checkpoint_task.get(key) != value
        }
        if mismatched_task:
            parser.error(
                "resume task configuration differs from checkpoint: "
                f"{mismatched_task}"
            )
        if (
            args.decode_mode == "chunk_parallel"
            and resume_decode_mode == "chunk_parallel"
        ):
            checkpoint_chunk_size = resume_state.get(
                "data_config",
                {},
            ).get("chunk_size")
            if checkpoint_chunk_size != args.chunk_size:
                parser.error(
                    "resume chunk size differs from checkpoint: "
                    f"{checkpoint_chunk_size} != {args.chunk_size}"
                )

    conversations = load_smoltalk(args.data)
    train_conversations, validation_conversations = split_conversations(
        conversations,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    chat_max_characters = (
        args.lmsys_max_characters
        if args.chat_max_characters is None
        else args.chat_max_characters
    )

    def tokenizer_conversation_stream(
        path: str,
        *,
        skip_conversations: int = 0,
        max_conversations: int,
        max_characters: int | None,
    ):
        for conversation in iter_jsonl_conversations(
            path,
            skip_conversations=skip_conversations,
            max_conversations=max_conversations,
            max_characters=max_characters,
            skip_invalid=True,
        ):
            if (
                conversation_ascii_letter_ratio(conversation)
                >= args.lmsys_min_ascii_letter_ratio
            ):
                yield conversation

    tinystories_train_skip = (
        0
        if args.tinystories_validation_data
        else args.tinystories_validation_stories
    )

    if resume_state is not None:
        tokenizer = CompactTokenizer.from_dict(resume_state["tokenizer"])
        if tokenizer.vocab_size != args.vocab_size:
            parser.error(
                "resume tokenizer vocabulary differs from --vocab-size: "
                f"{tokenizer.vocab_size} != {args.vocab_size}"
            )
    else:
        tokenizer_sources = []
        if args.tinystories_data:
            tokenizer_sources.append(
                tinystories_conversations(
                    iter_tinystories(
                        args.tinystories_data,
                        skip_stories=tinystories_train_skip,
                        max_stories=args.tinystories_tokenizer_stories,
                    ),
                    context_tokens=task_config.context_tokens,
                    response_tokens=task_config.response_tokens,
                )
            )
        tokenizer_sources.append(iter(train_conversations))
        if args.lmsys_data:
            tokenizer_sources.append(
                tokenizer_conversation_stream(
                    args.lmsys_data,
                    skip_conversations=args.lmsys_validation_conversations,
                    max_conversations=args.lmsys_tokenizer_conversations,
                    max_characters=args.lmsys_max_characters,
                )
            )
        for chat_path in args.chat_data:
            tokenizer_sources.append(
                tokenizer_conversation_stream(
                    chat_path,
                    max_conversations=args.chat_tokenizer_conversations,
                    max_characters=chat_max_characters,
                )
            )
        for prose_path in args.prose_data:
            tokenizer_sources.append(
                tinystories_conversations(
                    iter_tinystories(
                        prose_path,
                        max_stories=args.prose_tokenizer_stories,
                    ),
                    context_tokens=task_config.context_tokens,
                    response_tokens=task_config.response_tokens,
                )
            )
        tokenizer = CompactTokenizer.train(
            chain.from_iterable(tokenizer_sources),
            vocab_size=args.vocab_size,
        )
    tokenizer.save(checkpoint_dir / "tokenizer.json")
    whitespace_mask = whitespace_token_mask(tokenizer, device=device)
    legal_response_mask = legal_response_token_mask(tokenizer, device=device)
    semantic_token_mask = legal_response_mask & ~whitespace_mask
    semantic_token_mask = semantic_token_mask.clone()
    semantic_token_mask[tokenizer.EOS_ID] = False
    if tokenizer.blank_token_id is not None:
        semantic_token_mask[tokenizer.blank_token_id] = False
    forbidden_output_token_ids = tuple(
        (~legal_response_mask).nonzero(as_tuple=False).flatten().tolist()
    )
    if tokenizer.blank_token_id is None:
        parser.error("the selected formation objective requires a V2 tokenizer with BLANK")

    smoltalk_filter_config = LanguageExampleFilterConfig(
        require_complete_windows=args.require_complete_windows,
        require_complete_context=args.require_complete_context,
        require_complete_response=args.require_complete_response,
        trim_context_turns=args.trim_context_turns,
        deduplicate=args.deduplicate_examples,
    )

    def prepared_examples(examples) -> list[EncodedAssistantExample]:
        return [
            EncodedAssistantExample(
                conversation_id=example.conversation_id,
                turn_index=example.turn_index,
                context_ids=example.context_ids,
                response_ids=example.response_ids,
                source=example.source,
            )
            for example in tokenize_assistant_examples(
                examples,
                tokenizer,
                config=preparation_config,
            )
        ]

    def prepared_example_stream(examples):
        for example in tokenize_assistant_examples(
            examples,
            tokenizer,
            config=preparation_config,
        ):
            yield EncodedAssistantExample(
                conversation_id=example.conversation_id,
                turn_index=example.turn_index,
                context_ids=example.context_ids,
                response_ids=example.response_ids,
                source=example.source,
            )

    smoltalk_train_filter_stats = LanguageExampleFilterStats()
    smoltalk_validation_filter_stats = LanguageExampleFilterStats()
    train_examples = prepared_examples(
        filter_language_examples(
            assistant_examples(
                train_conversations, skip_first_assistant=args.skip_first_assistant
            ),
            tokenizer,
            task_config,
            smoltalk_filter_config,
            stats=smoltalk_train_filter_stats,
        )
    )
    available_train_examples = len(train_examples)
    if args.repeat_training_examples:
        if args.repeat_training_examples > available_train_examples:
            parser.error(
                "--repeat-training-examples exceeds the available SmolTalk "
                f"training responses ({available_train_examples})"
            )
        subset_indices = list(range(available_train_examples))
        subset_random = random.Random(args.seed + 2_000_003)
        subset_random.shuffle(subset_indices)
        selected_indices = sorted(
            subset_indices[: args.repeat_training_examples]
        )
        train_examples = [train_examples[index] for index in selected_indices]
    smoltalk_validation_examples = prepared_examples(
        filter_language_examples(
            assistant_examples(
                validation_conversations,
                skip_first_assistant=args.skip_first_assistant,
            ),
            tokenizer,
            task_config,
            smoltalk_filter_config,
            stats=smoltalk_validation_filter_stats,
        )
    )
    tinystories_validation_examples = []
    tinystories_training_filter_stats = LanguageExampleFilterStats()
    tinystories_validation_filter_stats = LanguageExampleFilterStats()
    tinystories_filter_config = LanguageExampleFilterConfig(
        require_complete_windows=args.require_complete_windows,
        require_complete_context=args.require_complete_context,
        require_complete_response=args.require_complete_response,
        trim_context_turns=args.trim_context_turns,
        minimum_ascii_letter_ratio=args.lmsys_min_ascii_letter_ratio,
        minimum_response_characters=args.lmsys_min_response_characters,
        deduplicate=args.deduplicate_examples,
    )
    if args.tinystories_data and args.tinystories_validation_stories:
        validation_story_path = (
            args.tinystories_validation_data or args.tinystories_data
        )
        tinystories_validation_examples = prepared_examples(
            filter_language_examples(
                tinystories_sentence_pairs(
                    iter_tinystories(
                        validation_story_path,
                        max_stories=args.tinystories_validation_stories,
                    ),
                    context_tokens=task_config.context_tokens,
                    response_tokens=task_config.response_tokens,
                ),
                tokenizer,
                task_config,
                tinystories_filter_config,
                stats=tinystories_validation_filter_stats,
            )
        )
    validation_examples = [
        *tinystories_validation_examples,
        *smoltalk_validation_examples,
    ]
    stream_sources = []
    stream_source_weights = []
    stream_filter_stats: dict[str, LanguageExampleFilterStats] = {}
    lmsys_validation_examples = []
    lmsys_training_filter_stats = LanguageExampleFilterStats()
    lmsys_validation_filter_stats = LanguageExampleFilterStats()
    stream_filter_config = LanguageExampleFilterConfig(
        require_complete_windows=args.require_complete_windows,
        require_complete_context=args.require_complete_context,
        require_complete_response=args.require_complete_response,
        trim_context_turns=args.trim_context_turns,
        require_user_prompt=args.lmsys_require_user_prompt,
        minimum_ascii_letter_ratio=args.lmsys_min_ascii_letter_ratio,
        minimum_response_characters=args.lmsys_min_response_characters,
        deduplicate=args.deduplicate_examples,
    )

    def training_example_factory(
        path: str,
        *,
        skip_conversations: int,
        max_characters: int | None,
        skip_first_assistant: bool,
        stats: LanguageExampleFilterStats,
    ):
        def examples():
            return filter_language_examples(
                assistant_examples(
                    iter_jsonl_conversations(
                        path,
                        skip_conversations=skip_conversations,
                        max_characters=max_characters,
                        skip_invalid=True,
                    ),
                    skip_first_assistant=skip_first_assistant,
                ),
                tokenizer,
                task_config,
                stream_filter_config,
                stats=stats,
            )

        return examples

    used_source_names: set[str] = set()

    def unique_source_name(path: str, fallback: str) -> str:
        base = Path(path).stem.strip() or fallback
        candidate = base
        suffix = 2
        while candidate in used_source_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_source_names.add(candidate)
        return candidate

    tinystories_source_index = None
    if args.tinystories_data:
        tinystories_name = unique_source_name(
            args.tinystories_data,
            "tinystories",
        )
        stream_filter_stats[tinystories_name] = (
            tinystories_training_filter_stats
        )

        def tinystories_training_examples():
            return filter_language_examples(
                tinystories_sentence_pairs(
                    iter_tinystories(
                        args.tinystories_data,
                        skip_stories=tinystories_train_skip,
                    ),
                    context_tokens=task_config.context_tokens,
                    response_tokens=task_config.response_tokens,
                ),
                tokenizer,
                task_config,
                tinystories_filter_config,
                stats=tinystories_training_filter_stats,
            )

        tinystories_source = StreamingLanguageSource(
            tinystories_training_examples,
            tokenizer,
            shuffle_buffer=args.tinystories_shuffle_buffer,
            seed=args.seed + 40_000,
            repeat=args.streaming_exhaustion == "restart",
        )
        tinystories_source_index = len(stream_sources)
        stream_sources.append((tinystories_name, tinystories_source))
        stream_source_weights.append(1.0)

    if args.primary_stream_weight > 0:
        primary_stream_stats = LanguageExampleFilterStats()
        primary_name = unique_source_name(args.data, "smoltalk_primary")
        stream_filter_stats[primary_name] = primary_stream_stats

        def primary_training_examples():
            return filter_language_examples(
                assistant_examples(
                    train_conversations,
                    skip_first_assistant=args.skip_first_assistant,
                ),
                tokenizer,
                task_config,
                smoltalk_filter_config,
                stats=primary_stream_stats,
            )

        primary_source = StreamingLanguageSource(
            primary_training_examples,
            tokenizer,
            shuffle_buffer=args.primary_shuffle_buffer,
            seed=args.seed + 30_000,
            repeat=args.streaming_exhaustion == "restart",
        )
        stream_sources.append((primary_name, primary_source))
        stream_source_weights.append(args.primary_stream_weight)

    if args.lmsys_data:
        lmsys_validation_conversations = list(
            iter_jsonl_conversations(
                args.lmsys_data,
                max_conversations=args.lmsys_validation_conversations,
                max_characters=args.lmsys_max_characters,
                skip_invalid=True,
            )
        )
        lmsys_validation_examples = prepared_examples(
            filter_language_examples(
                assistant_examples(
                    lmsys_validation_conversations,
                    skip_first_assistant=args.lmsys_skip_first_assistant,
                ),
                tokenizer,
                task_config,
                stream_filter_config,
                stats=lmsys_validation_filter_stats,
            )
        )
        validation_examples.extend(lmsys_validation_examples)

        lmsys_name = unique_source_name(args.lmsys_data, "lmsys")
        stream_filter_stats[lmsys_name] = lmsys_training_filter_stats
        lmsys_training_examples = training_example_factory(
            args.lmsys_data,
            skip_conversations=args.lmsys_validation_conversations,
            max_characters=args.lmsys_max_characters,
            skip_first_assistant=args.lmsys_skip_first_assistant,
            stats=lmsys_training_filter_stats,
        )

        if args.lmsys_load_mode == "memory":
            lmsys_source = PackedLanguageSource.build_encoded(
                prepared_example_stream(lmsys_training_examples()),
                task_config,
                seed=args.seed,
            )
            print(
                f"lmsys_loaded examples={lmsys_source.examples} "
                f"storage_gb={lmsys_source.storage_bytes / 1e9:.2f}",
                flush=True,
            )
        else:
            lmsys_source = StreamingLanguageSource(
                lmsys_training_examples,
                tokenizer,
                shuffle_buffer=args.lmsys_shuffle_buffer,
                seed=args.seed,
                repeat=args.streaming_exhaustion == "restart",
            )
        stream_sources.append((lmsys_name, lmsys_source))
        stream_source_weights.append(args.lmsys_source_weight)

    for chat_index, (chat_path, chat_weight) in enumerate(
        zip(args.chat_data, args.chat_source_weight, strict=True)
    ):
        chat_name = unique_source_name(chat_path, f"chat_{chat_index + 1}")
        chat_stats = LanguageExampleFilterStats()
        stream_filter_stats[chat_name] = chat_stats
        if args.chat_validation_conversations > 0:
            chat_validation_conversations = list(
                iter_jsonl_conversations(
                    chat_path,
                    max_conversations=args.chat_validation_conversations,
                    max_characters=chat_max_characters,
                    skip_invalid=True,
                )
            )
            chat_validation_filter_stats = LanguageExampleFilterStats()
            validation_examples.extend(
                replace(example, source=chat_name)
                for example in prepared_examples(
                    filter_language_examples(
                        assistant_examples(
                            chat_validation_conversations,
                            skip_first_assistant=False,
                        ),
                        tokenizer,
                        task_config,
                        stream_filter_config,
                        stats=chat_validation_filter_stats,
                    )
                )
            )
        chat_examples = training_example_factory(
            chat_path,
            skip_conversations=args.chat_validation_conversations,
            max_characters=chat_max_characters,
            skip_first_assistant=False,
            stats=chat_stats,
        )
        chat_source = StreamingLanguageSource(
            chat_examples,
            tokenizer,
            shuffle_buffer=args.chat_shuffle_buffer,
            seed=args.seed + 10_000 + chat_index,
            repeat=args.streaming_exhaustion == "restart",
        )
        stream_sources.append((chat_name, chat_source))
        stream_source_weights.append(chat_weight)

    for prose_index, (prose_path, prose_weight) in enumerate(
        zip(args.prose_data, args.prose_source_weight, strict=True)
    ):
        prose_name = unique_source_name(prose_path, f"prose_{prose_index + 1}")
        prose_stats = LanguageExampleFilterStats()
        stream_filter_stats[prose_name] = prose_stats

        if args.prose_validation_stories > 0:
            prose_validation_filter_stats = LanguageExampleFilterStats()
            validation_examples.extend(
                replace(example, source=prose_name)
                for example in prepared_examples(
                    filter_language_examples(
                        tinystories_sentence_pairs(
                            iter_tinystories(
                                prose_path,
                                max_stories=args.prose_validation_stories,
                            ),
                            context_tokens=task_config.context_tokens,
                            response_tokens=task_config.response_tokens,
                            source=prose_name,
                        ),
                        tokenizer,
                        task_config,
                        stream_filter_config,
                        stats=prose_validation_filter_stats,
                    )
                )
            )

        def prose_training_examples(
            prose_path=prose_path, prose_name=prose_name, prose_stats=prose_stats
        ):
            return filter_language_examples(
                tinystories_sentence_pairs(
                    iter_tinystories(
                        prose_path,
                        skip_stories=args.prose_validation_stories,
                    ),
                    context_tokens=task_config.context_tokens,
                    response_tokens=task_config.response_tokens,
                    source=prose_name,
                ),
                tokenizer,
                task_config,
                stream_filter_config,
                stats=prose_stats,
            )

        prose_source = StreamingLanguageSource(
            prose_training_examples,
            tokenizer,
            shuffle_buffer=args.prose_shuffle_buffer,
            seed=args.seed + 20_000 + prose_index,
            repeat=args.streaming_exhaustion == "restart",
        )
        stream_sources.append((prose_name, prose_source))
        stream_source_weights.append(prose_weight)

    if tinystories_source_index is not None:
        other_weights = [
            weight
            for index, weight in enumerate(stream_source_weights)
            if index != tinystories_source_index
        ]
        stream_source_weights[tinystories_source_index] = (
            _fractional_source_weight(
                other_weights,
                args.tinystories_mix_fraction,
            )
        )

    stream_weight_total = sum(stream_source_weights)
    resolved_stream_shares = {
        name: weight / stream_weight_total
        for (name, _), weight in zip(
            stream_sources,
            stream_source_weights,
            strict=True,
        )
    }

    weighted_stream_source = None
    if not stream_sources:
        train_batcher = LanguageBatcher(train_examples, task_config, seed=args.seed)
    else:
        weighted_stream_source = WeightedLanguageSource(
            stream_sources,
            stream_source_weights,
            seed=args.seed + 20_000,
        )
        train_batcher = MixtureLanguageBatcher(
            train_examples,
            weighted_stream_source,
            task_config,
            streaming_weight=streaming_weight,
            seed=args.seed,
        )
    validation_batcher = LanguageBatcher(validation_examples, task_config, seed=args.seed)
    validation_fingerprint = _encoded_examples_fingerprint(
        validation_examples
    )
    training_evaluation_batcher = (
        LanguageBatcher(train_examples, task_config, seed=args.seed)
        if args.repeat_training_examples
        else None
    )
    data_fingerprint_payload = {
        "schema": "gator-language-data-v1",
        "files": {
            "tinystories_train": _data_file_identity(
                args.tinystories_data
            ),
            "tinystories_validation": _data_file_identity(
                args.tinystories_validation_data
            ),
            "smoltalk": _data_file_identity(args.data),
            "lmsys": _data_file_identity(args.lmsys_data),
            "chat": [
                _data_file_identity(path) for path in args.chat_data
            ],
            "prose": [
                _data_file_identity(path) for path in args.prose_data
            ],
        },
        "tokenizer": _stable_fingerprint(tokenizer.to_dict()),
        "task": {
            "context_tokens": task_config.context_tokens,
            "response_tokens": task_config.response_tokens,
            "reference_tokens": task_config.target_tokens,
        },
        "finite_examples": {
            "available": available_train_examples,
            "active": len(train_examples),
            "sha256": _encoded_examples_fingerprint(train_examples),
        },
        "validation_examples": {
            "count": len(validation_examples),
            "sha256": validation_fingerprint,
        },
        "sampling": {
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "repeat_training_examples": args.repeat_training_examples,
            "streaming_weight": streaming_weight,
            "streaming_exhaustion": args.streaming_exhaustion,
            "tinystories_mix_fraction": args.tinystories_mix_fraction,
            "tinystories_validation_stories": (
                args.tinystories_validation_stories
            ),
            "tinystories_tokenizer_stories": (
                args.tinystories_tokenizer_stories
            ),
            "tinystories_shuffle_buffer": (
                args.tinystories_shuffle_buffer
            ),
            "resolved_stream_shares": resolved_stream_shares,
            "primary_stream_weight": args.primary_stream_weight,
            "primary_shuffle_buffer": args.primary_shuffle_buffer,
            "lmsys_source_weight": args.lmsys_source_weight,
            "chat_source_weights": list(args.chat_source_weight),
            "chat_validation_conversations": args.chat_validation_conversations,
            "prose_source_weights": list(args.prose_source_weight),
            "prose_tokenizer_stories": args.prose_tokenizer_stories,
            "prose_validation_stories": args.prose_validation_stories,
            "prose_shuffle_buffer": args.prose_shuffle_buffer,
            "lmsys_load_mode": args.lmsys_load_mode,
            "lmsys_validation_conversations": (
                args.lmsys_validation_conversations
            ),
            "lmsys_max_characters": args.lmsys_max_characters,
            "chat_max_characters": chat_max_characters,
            "lmsys_shuffle_buffer": args.lmsys_shuffle_buffer,
            "chat_shuffle_buffer": args.chat_shuffle_buffer,
            "skip_first_assistant": args.skip_first_assistant,
            "lmsys_skip_first_assistant": (
                args.lmsys_skip_first_assistant
            ),
            "require_complete_windows": args.require_complete_windows,
            "require_complete_context": args.require_complete_context,
            "require_complete_response": args.require_complete_response,
            "trim_context_turns": args.trim_context_turns,
            "deduplicate_examples": args.deduplicate_examples,
            "lmsys_min_ascii_letter_ratio": (
                args.lmsys_min_ascii_letter_ratio
            ),
            "lmsys_min_response_characters": (
                args.lmsys_min_response_characters
            ),
            "lmsys_require_user_prompt": args.lmsys_require_user_prompt,
        },
    }
    data_fingerprint = _stable_fingerprint(data_fingerprint_payload)

    if args.architecture == "gator":
        model_config = GatorLanguageConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            embedding_dim=args.embedding_dim,
            lexical_embedding_dim=(
                None
                if args.lexical_embedding_dim in {None, 0}
                else args.lexical_embedding_dim
            ),
            lexical_mode=args.lexical_mode,
            cell_input_mode=args.cell_input_mode,
            cell_output_mode=args.cell_output_mode,
            cell_transport_mode=args.cell_transport_mode,
            stewardship_keep_mass=args.stewardship_keep_mass,
            stewardship_handoff_keep_mass=args.stewardship_handoff_keep_mass,
            stewardship_min_active=args.stewardship_min_active,
            stewardship_max_active=(
                None
                if args.stewardship_max_active == 0
                else args.stewardship_max_active
            ),
            stewardship_exploration=args.stewardship_exploration,
            stewardship_scouts=args.stewardship_scouts,
            stewardship_execution=args.stewardship_execution,
            stewardship_packed_threshold=args.stewardship_packed_threshold,
            packed_stewardship_backend=(
                args.packed_stewardship_backend
            ),
            receiver_memory_backend=args.receiver_memory_backend,
            receiver_memory_cuda_graph=args.receiver_memory_cuda_graph,
            receiver_memory_hgrn_scan=args.receiver_memory_hgrn_scan,
            broadcast_mix_cuda_graph=args.broadcast_mix_cuda_graph,
            stewardship_traffic_cuda_graph=args.stewardship_traffic_cuda_graph,
            block_causal=args.block_causal,
            block_size=(
                args.response_tokens
                if args.block_causal and args.block_size is None
                else args.block_size
            ),
            position_reference_length=args.position_norm_length,
            dense_router_backend=args.dense_router_backend,
            dense_node_backend=args.dense_node_backend,
            monarch_block_size=args.dense_node_monarch_blocks,
            node_compute_precision=args.node_compute_precision,
            readout_precision=args.readout_precision,
            dense_kernel_padding=args.dense_kernel_padding,
            activation_checkpointing=args.activation_checkpointing,
            n_cells=args.cells,
            n_nodes=args.nodes,
            node_topology=args.node_topology,
            relays=args.relays,
            neighbors=args.neighbors,
            mask_token_id=tokenizer.MASK_ID,
            blank_token_id=tokenizer.blank_token_id,
            forbidden_output_token_ids=forbidden_output_token_ids,
        )
        model = GatorLanguageModel(model_config).to(device)
        if args.grow_from_checkpoint:
            grow_state = torch.load(
                args.grow_from_checkpoint, map_location=device, weights_only=False
            )
            grow_checkpoint_config = dict(grow_state.get("model_config", {}))
            grow_checkpoint_config.setdefault("block_causal", False)
            grow_checkpoint_config.setdefault("block_size", None)
            grow_checkpoint_config.setdefault(
                "position_reference_length", None
            )
            old_n_cells = grow_checkpoint_config.get("n_cells")
            if old_n_cells is None:
                parser.error(
                    "--grow-from-checkpoint's model_config has no n_cells; "
                    "not a --architecture gator checkpoint"
                )
            expected_new_cells = old_n_cells * args.grow_clones_per_cell
            if args.cells != expected_new_cells:
                parser.error(
                    f"--cells must equal the checkpoint's n_cells ({old_n_cells}) "
                    f"times --grow-clones-per-cell ({args.grow_clones_per_cell}) "
                    f"= {expected_new_cells}, got --cells={args.cells}"
                )
            growth_exempt_keys = {
                "n_cells",
                "neighbors",
                # A recruitment cap only needs to make sense relative to the
                # CURRENT stage's own cell count (stewardship_max_active must
                # be <= n_cells), so it's expected to differ across growth
                # stages - e.g. uncapped at 16 cells, capped at 32 once grown
                # to 64. Not an architecture change, just a runtime
                # constraint on which cells get used.
                "stewardship_max_active",
            }
            growth_mismatch = {
                key: (grow_checkpoint_config.get(key), value)
                for key, value in model_config.to_dict().items()
                if key not in growth_exempt_keys
                and grow_checkpoint_config.get(key) != value
            }
            if growth_mismatch:
                parser.error(
                    "--grow-from-checkpoint's architecture differs from the "
                    f"current config beyond n_cells/neighbors: {growth_mismatch}"
                )
            grown_state_dict, grow_scatter_order = grow_gator_state_dict(
                grow_state["model"],
                model.state_dict(),
                old_n_cells=old_n_cells,
                clones_per_cell=args.grow_clones_per_cell,
                perturb_scale=args.grow_perturb_scale,
                seed=args.seed,
            )
            model.load_state_dict(grown_state_dict, strict=True)
            print(
                f"grew model: old_n_cells={old_n_cells} "
                f"clones_per_cell={args.grow_clones_per_cell} "
                f"new_n_cells={expected_new_cells} "
                f"perturb_scale={args.grow_perturb_scale} "
                f"from={args.grow_from_checkpoint}",
                flush=True,
            )
        architecture_summary = (
            f"embedding_dim={model_config.language_width} "
            f"lexical_embedding_dim={model_config.lexical_width} cell_width={args.d_model} "
            f"cells={args.cells} lexical_mode={args.lexical_mode} "
            f"cell_input_mode={args.cell_input_mode} "
            f"cell_output_mode={args.cell_output_mode} nodes={args.nodes} "
            f"cell_transport_mode={args.cell_transport_mode} "
            f"stewardship_execution={args.stewardship_execution} "
            f"packed_stewardship_backend="
            f"{args.packed_stewardship_backend} "
            f"receiver_memory_backend={args.receiver_memory_backend} "
            f"receiver_memory_cuda_graph={args.receiver_memory_cuda_graph} "
            f"receiver_memory_hgrn_scan={args.receiver_memory_hgrn_scan} "
            f"broadcast_mix_cuda_graph={args.broadcast_mix_cuda_graph} "
            f"stewardship_traffic_cuda_graph={args.stewardship_traffic_cuda_graph} "
            f"dense_router_backend={args.dense_router_backend} "
            f"dense_node_backend={args.dense_node_backend} "
            f"dense_node_monarch_blocks={args.dense_node_monarch_blocks} "
            f"node_compute_precision={args.node_compute_precision} "
            f"readout_precision={args.readout_precision} "
            f"dense_kernel_padding={args.dense_kernel_padding} "
            f"node_topology={args.node_topology} relays={args.relays} "
            f"neighbors={args.neighbors}"
        )
    elif args.architecture == "convergent":
        model_config = ConvergentLanguageConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            embedding_dim=(
                args.d_model
                if args.embedding_dim is None
                else args.embedding_dim
            ),
            n_nodes=args.nodes,
            degree=args.graph_degree,
            router_mode=args.router_mode,
            router_rank=args.router_rank,
            ingress_mode=args.ingress_mode,
            route_backend=args.route_backend,
            packed_memory_backend=args.packed_memory_backend,
            packed_memory_reader_backend=(
                args.packed_memory_reader_backend
            ),
            packed_memory_writer_backend=(
                args.packed_memory_writer_backend
            ),
            packed_memory_feature_chunk=(
                args.packed_memory_feature_chunk
            ),
            packed_memory_compact_nodes=(
                args.packed_memory_compact_nodes
            ),
            adaptive_route_min_k=args.adaptive_route_min_k,
            adaptive_route_mode=args.adaptive_route_mode,
            adaptive_route_entropy_mix=(
                args.adaptive_route_entropy_mix
            ),
            compact_node_dispatch=args.compact_node_dispatch,
            node_compute_mode=args.node_compute_mode,
            node_adapter_rank=args.node_adapter_rank,
            ingress_groups=args.ingress_groups,
            ingress_top_groups=args.ingress_top_groups,
            ingress_learned_groups=args.ingress_learned_groups,
            active_node_top_k=args.active_node_top_k,
            active_node_mass_threshold=(
                args.active_node_mass_threshold
            ),
            relays=args.relays,
            lanes=args.lanes,
            mask_token_id=tokenizer.MASK_ID,
            blank_token_id=tokenizer.blank_token_id,
            forbidden_output_token_ids=forbidden_output_token_ids,
            connection_layout=args.connection_layout,
            connection_seed=args.connection_seed,
            connection_rewire=args.connection_rewire,
            terminal_vote_mode=args.terminal_vote_mode,
            slot_isolation_slots=(
                None
                if args.slot_isolation_slots == 0
                else args.slot_isolation_slots
            ),
            lane_vote_noise_std=args.lane_vote_noise_std,
            position_norm_length=args.position_norm_length,
            fp8_vocab_readout=args.fp8_vocab_readout,
        )
        model = ConvergentLanguageModel(model_config).to(device)
        architecture_summary = (
            f"embedding_dim={model_config.embedding_dim} "
            f"lane_width={model_config.d_model} "
            f"lanes={model_config.lanes} nodes={model_config.n_nodes} "
            f"degree={model_config.degree} relays={model_config.relays} "
            f"router_mode={model_config.router_mode} "
            f"router_rank={model_config.router_rank} "
            f"ingress_mode={model_config.ingress_mode} "
            f"route_backend={model_config.route_backend} "
            f"packed_memory_backend={model_config.packed_memory_backend} "
            f"packed_memory_reader_backend="
            f"{model_config.packed_memory_reader_backend} "
            f"packed_memory_writer_backend="
            f"{model_config.packed_memory_writer_backend} "
            f"packed_memory_feature_chunk="
            f"{model_config.packed_memory_feature_chunk} "
            f"packed_memory_compact_nodes="
            f"{model_config.packed_memory_compact_nodes} "
            f"adaptive_route_min_k={model_config.adaptive_route_min_k} "
            f"adaptive_route_mode={model_config.adaptive_route_mode} "
            f"adaptive_route_entropy_mix="
            f"{model_config.adaptive_route_entropy_mix:g} "
            f"compact_node_dispatch={model_config.compact_node_dispatch} "
            f"node_compute_mode={model_config.node_compute_mode} "
            f"node_adapter_rank={model_config.node_adapter_rank} "
            f"ingress_groups={model_config.ingress_groups} "
            f"ingress_top_groups={model_config.ingress_top_groups} "
            f"ingress_learned_groups={model_config.ingress_learned_groups} "
            f"active_node_top_k={model_config.active_node_top_k} "
            f"active_node_mass_threshold="
            f"{model_config.active_node_mass_threshold} "
            f"connection_layout={model_config.connection_layout} "
            f"connection_seed={model_config.connection_seed} "
            f"connection_rewire={model_config.connection_rewire} "
            f"terminal_vote_mode={model_config.terminal_vote_mode} "
            f"slot_isolation_slots={model_config.slot_isolation_slots} "
            f"lane_vote_noise_std={model_config.lane_vote_noise_std} "
            f"reachable_node_execution="
            f"{model_config.reachable_node_execution}"
        )
    else:
        model_config = OneShotTransformerLanguageConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_heads=args.transformer_heads,
            ff_width=args.transformer_ff_width,
            n_layers=args.transformer_layers,
            context_tokens=(
                args.context_tokens + 1
                if args.reasoning_loops == 2
                else args.context_tokens
            ),
            response_tokens=args.response_tokens,
            dropout=args.transformer_dropout,
            mask_token_id=tokenizer.MASK_ID,
            blank_token_id=tokenizer.blank_token_id,
            forbidden_output_token_ids=forbidden_output_token_ids,
        )
        model = OneShotTransformerLanguageModel(model_config).to(device)
        architecture_summary = (
            f"d_model={args.d_model} heads={args.transformer_heads} "
            f"ff_width={args.transformer_ff_width} layers={args.transformer_layers} "
            f"dropout={args.transformer_dropout} noncausal=True"
        )
    if resume_state is not None:
        checkpoint_model_config = resume_state.get("model_config", {})
        current_model_config = model.config.to_dict()

        def normalized_config_value(value):
            if isinstance(value, dict):
                return {
                    key: normalized_config_value(item)
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return tuple(normalized_config_value(item) for item in value)
            return value

        checkpoint_model_config = normalized_config_value(
            checkpoint_model_config
        )
        current_model_config = normalized_config_value(current_model_config)
        if args.architecture == "gator":
            checkpoint_model_config.setdefault("lexical_embedding_dim", None)
            checkpoint_model_config.setdefault("block_causal", False)
            checkpoint_model_config.setdefault("block_size", None)
            checkpoint_model_config.setdefault(
                "position_reference_length", None
            )
        # Checkpoints predating prompt-conditioned terminal voting are the
        # exact global-vote architecture. Only convergent checkpoints have
        # this field at all; transformer checkpoints never set it.
        if args.architecture == "convergent":
            checkpoint_model_config.setdefault(
                "terminal_vote_mode",
                "global",
            )
        if args.architecture == "convergent":
            # Pre-factorization checkpoints are the exact private-router,
            # fixed-ingress, dense-mass architecture.
            for key, value in {
                "router_mode": "private_full",
                "router_rank": 16,
                "ingress_mode": "fixed",
                "route_backend": "dense",
                "adaptive_route_min_k": None,
                "adaptive_route_mode": "hybrid",
                "adaptive_route_entropy_mix": 0.5,
                "node_compute_mode": "full_private",
                "node_adapter_rank": 8,
                "ingress_groups": None,
                "ingress_top_groups": 2,
                "ingress_learned_groups": True,
                "active_node_top_k": None,
                "active_node_mass_threshold": 0.0,
            }.items():
                checkpoint_model_config.setdefault(key, value)
        # Receiver memory and dense kernel selection are parameter-free
        # numerical backends. They may change across hardware or when
        # continuing a legacy checkpoint without changing learned tensors or
        # the represented equations.
        runtime_model_config_keys = {
            "receiver_memory_backend",
            "receiver_memory_cuda_graph",
            "receiver_memory_hgrn_scan",
            "broadcast_mix_cuda_graph",
            "stewardship_traffic_cuda_graph",
            "dense_router_backend",
            # "dense_node_backend" deliberately NOT included (removed once
            # "monarch" became a valid value): "einsum"/"folded_bmm" really
            # were parameter-free (both apply the same dense node_weight,
            # just different kernels), but "monarch" uses entirely
            # different learned parameters (monarch_left/monarch_right
            # instead of node_weight) -- silently allowing this to differ
            # across a checkpoint resume would let a folded_bmm-trained
            # checkpoint's node_weight get paired with a freshly-
            # initialized monarch_left/monarch_right (or vice versa)
            # without erroring here.
            "node_compute_precision",
            "readout_precision",
            "dense_kernel_padding",
            "packed_stewardship_backend",
            "packed_memory_backend",
            "packed_memory_reader_backend",
            "packed_memory_writer_backend",
            "packed_memory_feature_chunk",
            "packed_memory_compact_nodes",
            "compact_node_dispatch",
            # Position-encoding normalizer: switching this on for a resumed
            # checkpoint reproduces identical numerics as long as the new
            # fixed value equals the checkpoint's own context+response
            # length, so it's safe to change across a resume without
            # invalidating learned tensors.
            "position_norm_length",
            "position_reference_length",
        }
        # terminal_vote_mode prompt->global is a validated free swap ONLY at
        # lanes=1 (global's vote parameter is a softmax over one lane, i.e.
        # always weight 1.0 regardless of its value, so prompt-mode's extra
        # per-prompt conditioning was never adding real information at this
        # width). Confirmed via a frozen-weight A/B: same-quality generation,
        # state_dict compatible except one now-unused parameter
        # (terminal_vote_query). Reconcile only this specific transition;
        # any other terminal_vote_mode change, or lanes != 1, still hits the
        # hard mismatch error below.
        allow_prompt_to_global_swap = (
            args.architecture == "convergent"
            and checkpoint_model_config.get("lanes") == 1
            and current_model_config.get("lanes") == 1
            and checkpoint_model_config.get("terminal_vote_mode") == "prompt"
            and current_model_config.get("terminal_vote_mode") == "global"
        )
        if allow_prompt_to_global_swap:
            checkpoint_model_config["terminal_vote_mode"] = "global"
        mismatched_model = {
            key: (
                checkpoint_model_config.get(key),
                current_model_config.get(key),
            )
            for key in sorted(
                set(checkpoint_model_config) | set(current_model_config)
            )
            if key not in runtime_model_config_keys
            if checkpoint_model_config.get(key) != current_model_config.get(key)
        }
        if mismatched_model:
            parser.error(
                "resume model configuration differs from checkpoint: "
                f"{mismatched_model}"
            )
        if allow_prompt_to_global_swap:
            missing, unexpected = model.load_state_dict(
                resume_state["model"], strict=False
            )
            allowed_unexpected = {"core.terminal_vote_query", "terminal_vote_query"}
            if missing or set(unexpected) - allowed_unexpected:
                parser.error(
                    "resume state_dict differs from the model beyond the "
                    "expected prompt->global terminal_vote_mode swap: "
                    f"missing={missing} unexpected={unexpected}"
                )
        else:
            model.load_state_dict(resume_state["model"], strict=True)
    execution_controller = None
    if (
        args.architecture == "gator"
        and args.cell_transport_mode == "stewardship"
        and not args.block_causal
    ):
        execution_controller = StewardshipExecutionController(
            mode=args.stewardship_execution,
            threshold=args.stewardship_packed_threshold,
            relays=args.relays,
            check_every=args.stewardship_execution_check_every,
        )
        if resume_state is not None and resume_state.get("execution") is not None:
            try:
                execution_controller.load_state_dict(
                    resume_state["execution"]
                )
            except ValueError as error:
                parser.error(str(error))

    handoff_controller = None
    if args.decode_mode in {"handoff", "graduated_handoff"}:
        if args.decode_mode == "handoff":
            handoff_controller = AccuracyGatedTeacherHandoff(
                start_accuracy=args.handoff_start_content_accuracy,
                end_accuracy=args.handoff_end_content_accuracy,
                ema_decay=args.handoff_accuracy_ema_decay,
            )
        else:
            handoff_controller = GraduatedTeacherHandoff(
                planned_steps=args.steps,
                baseline_fraction=args.graduated_baseline_fraction,
                accuracy_step=args.graduated_accuracy_step,
                release_accuracy=args.graduated_release_accuracy,
                ema_decay=args.handoff_accuracy_ema_decay,
            )
        if resume_state is not None:
            saved_handoff = resume_state.get("data_config", {}).get(
                "handoff_state"
            )
            if saved_handoff is not None:
                try:
                    handoff_controller.load_state_dict(saved_handoff)
                except ValueError as error:
                    parser.error(str(error))

    prefix_denoising_controller = None
    if args.decode_mode == "prefix_denoise":
        prefix_denoising_controller = PrefixDenoisingController(
            start_retention=args.prefix_retention_start,
            floor_retention=args.prefix_retention_floor,
            warmup_steps=args.prefix_denoise_warmup_steps,
            start_accuracy=args.handoff_start_content_accuracy,
            end_accuracy=args.handoff_end_content_accuracy,
            start_context_utility=args.prefix_context_utility_start,
            end_context_utility=args.prefix_context_utility_end,
            ema_decay=args.handoff_accuracy_ema_decay,
        )
        if resume_state is not None:
            saved_prefix_denoising = resume_state.get(
                "data_config", {}
            ).get("prefix_denoising_state")
            if saved_prefix_denoising is not None:
                try:
                    prefix_denoising_controller.load_state_dict(
                        saved_prefix_denoising
                    )
                except ValueError as error:
                    parser.error(str(error))

    token_handoff_controller = None
    if args.decode_mode == "token_handoff":
        teacher_perplexity_baseline = None
        if not teacher_ppl_guard_disabled and resume_state is not None:
            for result_name in ("auxiliary_validation", "validation"):
                resumed_result = resume_state.get(result_name) or {}
                candidate = resumed_result.get("content_perplexity")
                if candidate is not None and math.isfinite(float(candidate)):
                    teacher_perplexity_baseline = float(candidate)
                    break
        token_handoff_controller = TokenwiseTeacherHandoff(
            response_slots=args.response_tokens,
            start_removed_tokens=args.token_handoff_start_removed,
            maximum_removed_tokens=maximum_token_removals,
            advance_ratio=args.token_handoff_advance_ratio,
            retreat_ratio=args.token_handoff_retreat_ratio,
            minimum_teacher_accuracy=(
                args.token_handoff_min_teacher_accuracy
            ),
            minimum_frontier_accuracy=(
                args.token_handoff_min_frontier_accuracy
            ),
            advance_patience=args.token_handoff_advance_patience,
            retreat_patience=args.token_handoff_retreat_patience,
            ema_decay=args.handoff_accuracy_ema_decay,
            full_probe_fraction=(
                args.token_handoff_full_probe_fraction
            ),
            maximum_teacher_perplexity_ratio=(
                None
                if teacher_ppl_guard_disabled
                else args.token_handoff_max_teacher_ppl_ratio
            ),
            teacher_perplexity_retreat_ratio=(
                None
                if teacher_ppl_guard_disabled
                else args.token_handoff_retreat_teacher_ppl_ratio
            ),
            teacher_perplexity_baseline=teacher_perplexity_baseline,
        )
        if resume_state is not None and resume_decode_mode == "token_handoff":
            saved_token_handoff = resume_state.get(
                "data_config", {}
            ).get("token_handoff_state")
            if saved_token_handoff is not None:
                try:
                    token_handoff_controller.load_state_dict(
                        saved_token_handoff,
                        allow_policy_override=(
                            args.token_handoff_allow_policy_override
                        ),
                    )
                except ValueError as error:
                    parser.error(str(error))

    def execution_label() -> str:
        if execution_controller is None:
            return "dense"
        packed_from = execution_controller.plan.packed_from_relay
        return (
            "dense"
            if packed_from is None
            else f"packed_from_relay_{packed_from}"
        )

    # Complete dense exploration calls are compile-friendly. Once the external
    # controller latches a packed plan, the raw model handles the complete
    # invocation (including a possible dense prefix). Both paths share one
    # module and one optimizer state.
    dense_compile_enabled = compute_config.compile.enabled
    dense_train_model = compile_module(model, compute_config.compile)
    optimization = optimization_config_from_args(args)
    optimizer_parameter_groups = (
        component_optimizer_parameter_groups(
            model,
            optimization,
            optimizer_component_lr_multipliers,
            train_components=optimizer_train_components,
            retain_frozen_state=args.optimizer_retain_frozen_state,
            freeze_route_transport=args.optimizer_freeze_route_transport,
        )
        if args.optimizer_groups == "component_scaled"
        else None
    )
    engine = OptimizationEngine(
        model,
        optimization,
        parameter_groups=optimizer_parameter_groups,
    )
    print(
        f"optimizer_groups={args.optimizer_groups} "
        f"param_groups={len(engine.optimizer.param_groups)}",
        flush=True,
    )
    for group in engine.optimizer.param_groups:
        print(
            "optimizer_group "
            f"label={group.get('group_label', 'monolithic')} "
            f"parameters={sum(parameter.numel() for parameter in group['params'])} "
            f"trainable_parameters={sum(parameter.numel() for parameter in group['params'] if parameter.requires_grad)} "
            f"lr={group['lr']:.9g} "
            f"lr_multiplier={group.get('lr_multiplier', 1.0):.9g} "
            f"weight_decay={group['weight_decay']:.9g}",
            flush=True,
        )
    if resume_state is not None:
        engine.load_state_dict(
            resume_state["training"],
            load_optimizer=args.resume_optimizer,
            load_scheduler=False,
            learning_rate=(
                args.learning_rate if args.resume_optimizer else None
            ),
            migrate_optimizer_by_name=(
                args.optimizer_groups == "component_scaled"
            ),
            preserve_group_lr_multipliers=(
                args.optimizer_groups == "component_scaled"
            ),
        )
        if engine.last_optimizer_state_migration is not None:
            migrated, available = engine.last_optimizer_state_migration
            print(
                f"optimizer_state_migration={migrated}/{available}",
                flush=True,
            )
        print(
            f"resume_checkpoint={args.resume_checkpoint} "
            f"resume_step={start_step} additional_steps={args.steps} "
            f"optimizer_moments={args.resume_optimizer} "
            f"scheduler_restarted=True initial_lr={engine.learning_rate:.6g}",
            flush=True,
        )
        resumed_baseline = checkpoint_dir / (
            "best.pt"
            if resume_decode_mode == args.decode_mode
            else "resume_baseline.pt"
        )
        has_best_history = bool(
            resume_state.get("data_config", {}).get("best_metrics")
        )
        if not resumed_baseline.exists() and not has_best_history:
            torch.save(resume_state, resumed_baseline)
            print(
                f"resume_baseline={resumed_baseline}",
                flush=True,
            )

    train_pipeline = LanguageBatchPipeline(train_batcher, pipeline_config)
    _ACTIVE_LANGUAGE_PIPELINES.add(train_pipeline)
    atexit.register(train_pipeline.close)
    data_resume_status = "fresh"
    if resume_state is not None:
        if args.resume_data_state == "reset":
            data_resume_status = "reset_requested"
        else:
            saved_pipeline = resume_state.get("data_pipeline")
            reset_reason = None
            portable_identity_mismatch = False
            if not isinstance(saved_pipeline, dict):
                reset_reason = "checkpoint_has_no_data_pipeline_state"
            else:
                if saved_pipeline.get("fingerprint") != data_fingerprint:
                    (
                        reset_reason,
                        portable_identity_mismatch,
                    ) = _data_resume_fingerprint_check(
                        resume_state,
                        saved_pipeline,
                        current_data_fingerprint=data_fingerprint,
                        current_validation_fingerprint=validation_fingerprint,
                        policy=args.resume_data_state,
                    )
                if (
                    reset_reason is None
                    and not train_pipeline.exact_resume_supported
                ):
                    reset_reason = (
                        "current_provider_is_not_exactly_restorable"
                    )
                if reset_reason is None and not isinstance(
                    saved_pipeline.get("state"), dict
                ):
                    reset_reason = "checkpoint_pipeline_state_is_invalid"
                if (
                    reset_reason is None
                    and not saved_pipeline["state"].get(
                        "exact_provider_state", False
                    )
                ):
                    reset_reason = "checkpoint_provider_state_was_not_exact"

            if reset_reason is None:
                initial_pipeline_state = train_pipeline.state_dict()
                try:
                    train_pipeline.load_state_dict(
                        saved_pipeline["state"]
                    )
                except (TypeError, ValueError, RuntimeError) as error:
                    if args.resume_data_state in {"exact", "portable_exact"}:
                        train_pipeline.close()
                        parser.error(
                            "failed to restore exact data pipeline state: "
                            f"{error}"
                        )
                    try:
                        train_pipeline.load_state_dict(
                            initial_pipeline_state
                        )
                    except (TypeError, ValueError, RuntimeError) as reset_error:
                        train_pipeline.close()
                        parser.error(
                            "failed to restore checkpoint data state "
                            f"({error}) and then failed to recover the "
                            f"seeded pipeline ({reset_error})"
                        )
                    data_resume_status = (
                        "reset_auto:pipeline_state_incompatible:"
                        f"{type(error).__name__}"
                    )
                else:
                    data_resume_status = (
                        "portable_exact_restored"
                        if portable_identity_mismatch
                        else "exact_restored"
                    )
            elif args.resume_data_state in {"exact", "portable_exact"}:
                train_pipeline.close()
                parser.error(
                    f"--resume-data-state {args.resume_data_state} cannot be "
                    "satisfied: "
                    f"{reset_reason}"
                )
            else:
                data_resume_status = f"reset_auto:{reset_reason}"

    if args.logging_slim:
        print(
            f"run device={device} architecture={args.architecture} "
            f"parameters={model.parameter_count} "
            f"preset={args.execution_preset} compile={dense_compile_enabled} "
            f"decode_mode={args.decode_mode} "
            f"chunk_size={args.chunk_size} "
            f"plan={execution_label()} train={len(train_examples)} "
            f"validation={len(validation_examples)} "
            f"validation_fp={validation_fingerprint[:12]}",
            flush=True,
        )
    else:
        print(
        f"device={device} architecture={args.architecture} "
        f"parameters={model.parameter_count} vocab={tokenizer.vocab_size} "
        f"{architecture_summary} "
        f"context={args.context_tokens} response={args.response_tokens} "
        f"decode_mode={args.decode_mode} "
        f"chunk_size={args.chunk_size} "
        f"reference={task_config.target_tokens} "
        f"tokenizer_format={tokenizer.format_version} "
        f"blank_token_id={tokenizer.blank_token_id} "
        f"formation_objective={args.formation_objective} "
        f"alignment_warmup={args.alignment_warmup_fraction} "
        f"alignment_max_shift={args.alignment_max_shift} "
        f"sequence_quality_penalty={args.sequence_quality_penalty} "
        f"sequence_quality_components={args.sequence_composition_weight}:"
        f"{args.sequence_length_weight}:{args.sequence_eos_weight} "
        f"stewardship_keep_mass={args.stewardship_keep_mass} "
        f"stewardship_handoff_keep_mass={args.stewardship_handoff_keep_mass} "
        f"stewardship_active={args.stewardship_min_active}:"
        f"{args.stewardship_max_active or args.cells} "
        f"stewardship_exploration={args.stewardship_exploration} "
        f"stewardship_scouts={args.stewardship_scouts} "
        f"stewardship_execution={args.stewardship_execution} "
        f"stewardship_packed_threshold={args.stewardship_packed_threshold} "
        f"packed_stewardship_backend={args.packed_stewardship_backend} "
        f"stewardship_execution_check_every="
        f"{args.stewardship_execution_check_every} "
        f"execution_plan={execution_label()} "
        f"stewardship_warmup={args.stewardship_warmup_fraction} "
        f"eval_stewardship_temperature={args.stewardship_temperature_end} "
        f"eval_stewardship_prune=True "
        f"whitespace_omission_penalty={args.whitespace_omission_penalty} "
        f"adjacent_whitespace_penalty={args.adjacent_whitespace_penalty} "
        f"repeat_training_examples={args.repeat_training_examples} "
        f"complete_windows={args.require_complete_windows} "
        f"complete_context={args.require_complete_context} "
        f"complete_response={args.require_complete_response} "
        f"trim_context_turns={args.trim_context_turns} "
        f"deduplicate={args.deduplicate_examples} "
        f"streaming_exhaustion={args.streaming_exhaustion} "
        f"primary_stream_weight={args.primary_stream_weight:g} "
        f"lmsys_max_characters={args.lmsys_max_characters} "
        f"chat_sources={len(args.chat_data)} "
        f"prose_sources={len(args.prose_data)} "
        f"streaming_weight={streaming_weight} "
        f"stream_source_weights={stream_source_weights} "
        f"max_vram_gb={args.max_vram_gb} "
        f"execution_preset={args.execution_preset} "
        f"pre_latch_dense_compiled={dense_compile_enabled} "
        f"compiled_pruned_routes={dense_compile_enabled} "
        f"compile_mode={compute_config.compile.mode} "
        f"compile_shape_padding={compute_config.compile.shape_padding} "
        f"batch_pipeline={pipeline_config.mode} "
        f"prefetch_batches={pipeline_config.prefetch_batches} "
        f"pin_memory={pipeline_config.pin_memory} "
        f"non_blocking_transfer={pipeline_config.non_blocking} "
        f"dedicated_copy_stream={pipeline_config.dedicated_cuda_stream} "
        f"tokenizer_workers={args.tokenizer_workers} "
        f"tokenizer_worker_backend={preparation_config.backend} "
        f"data_resume={data_resume_status} "
        f"data_fingerprint={data_fingerprint[:12]} "
        f"validation_fingerprint={validation_fingerprint[:12]} "
            f"{resolved_runtime.summary()}"
        )
    print(
        "optimization_policy "
        f"token_handoff_loss_normalization="
        f"{args.token_handoff_loss_normalization} "
        f"route_temperature_schedule={args.route_temperature_schedule} "
        f"route_temperature={args.route_temperature_start:g}:"
        f"{args.route_temperature_middle:g}:"
        f"{args.route_temperature_end:g} "
        f"route_fractions={args.route_temperature_hold_fraction:g}:"
        f"{args.route_temperature_middle_fraction:g} "
        f"route_schedule_offset={args.route_temperature_step_offset} "
        f"route_schedule_total="
        f"{args.route_temperature_total_steps or args.steps} "
        f"optimizer_groups={args.optimizer_groups} "
        f"component_lr_multipliers={optimizer_component_lr_multipliers} "
        f"train_components={optimizer_train_components} "
        f"retain_frozen_state={args.optimizer_retain_frozen_state} "
        f"freeze_route_transport={args.optimizer_freeze_route_transport} "
        f"token_handoff_teacher_ppl_guard="
        f"{args.token_handoff_max_teacher_ppl_ratio:g}:"
        f"{args.token_handoff_retreat_teacher_ppl_ratio:g}",
        flush=True,
    )
    if resolved_stream_shares:
        print(
            "corpus_mix "
            + " ".join(
                f"{name}={share:.3f}"
                for name, share in resolved_stream_shares.items()
            ),
            flush=True,
        )
    if handoff_controller is not None:
        if isinstance(handoff_controller, GraduatedTeacherHandoff):
            print(
                "graduated_handoff_config "
                f"planned_steps={handoff_controller.planned_steps} "
                f"baseline_fraction="
                f"{handoff_controller.baseline_fraction:.3f} "
                f"baseline_steps={handoff_controller.baseline_steps} "
                f"accuracy_step={handoff_controller.accuracy_step:.3f} "
                f"release_accuracy="
                f"{handoff_controller.release_accuracy:.3f} "
                f"ema_decay={handoff_controller.ema_decay:.3f} "
                f"initial_teacher_fraction="
                f"{handoff_controller.teacher_fraction:.3f}",
                flush=True,
            )
        else:
            print(
                "handoff_config "
                f"start_content_accuracy="
                f"{handoff_controller.start_accuracy:.3f} "
                f"end_content_accuracy="
                f"{handoff_controller.end_accuracy:.3f} "
                f"ema_decay={handoff_controller.ema_decay:.3f} "
                f"initial_teacher_fraction="
                f"{handoff_controller.teacher_fraction:.3f}",
                flush=True,
            )
    if prefix_denoising_controller is not None:
        print(
            "prefix_denoise_config "
            f"retention={prefix_denoising_controller.start_retention:.3f}->"
            f"{prefix_denoising_controller.floor_retention:.3f} "
            f"warmup_steps={prefix_denoising_controller.warmup_steps} "
            f"zero_rows={args.prefix_zero_row_fraction:.3f} "
            f"short_rows={args.prefix_short_row_fraction:.3f} "
            f"short_max={args.prefix_short_max_tokens} "
            f"span_rows={args.prefix_span_fraction:.3f} "
            f"accuracy_gate={prefix_denoising_controller.start_accuracy:.3f}->"
            f"{prefix_denoising_controller.end_accuracy:.3f} "
            f"context_utility_gate="
            f"{prefix_denoising_controller.start_context_utility:.3f}->"
            f"{prefix_denoising_controller.end_context_utility:.3f} "
            f"ema_decay={prefix_denoising_controller.ema_decay:.3f}",
            flush=True,
        )
    if token_handoff_controller is not None:
        print(
            "token_handoff_config "
            f"removed={token_handoff_controller.removed_tokens}->"
            f"{token_handoff_controller.maximum_removed_tokens} "
            f"accuracy_ratio="
            f"{token_handoff_controller.advance_ratio:.3f}/"
            f"{token_handoff_controller.retreat_ratio:.3f} "
            f"accuracy_floor="
            f"{token_handoff_controller.minimum_frontier_accuracy:.3f}/"
            f"{token_handoff_controller.minimum_teacher_accuracy:.3f} "
            f"patience={token_handoff_controller.advance_patience}/"
            f"{token_handoff_controller.retreat_patience} "
            f"teacher_ppl_ratio="
            f"{args.token_handoff_max_teacher_ppl_ratio:g}/"
            f"{args.token_handoff_retreat_teacher_ppl_ratio:g} "
            f"teacher_ppl_baseline="
            f"{token_handoff_controller.teacher_perplexity_baseline}",
            flush=True,
        )
    if not args.logging_slim:
        print(
            f"conversations train={len(train_conversations)} validation={len(validation_conversations)} "
            f"smoltalk_responses available_train={available_train_examples} "
            f"active_train={len(train_examples)} validation={len(smoltalk_validation_examples)} "
            f"tinystories_validation_responses="
            f"{len(tinystories_validation_examples)} "
            f"lmsys_validation_responses={len(lmsys_validation_examples)}"
        )
        print(
            f"smoltalk_filter train={smoltalk_train_filter_stats.to_dict()} "
            f"validation={smoltalk_validation_filter_stats.to_dict()}",
            flush=True,
        )
        if args.tinystories_data:
            print(
                "tinystories_filter "
                f"train={tinystories_training_filter_stats.to_dict()} "
                f"validation="
                f"{tinystories_validation_filter_stats.to_dict()}",
                flush=True,
            )
        if args.lmsys_data:
            print(
                f"lmsys_validation_filter={lmsys_validation_filter_stats.to_dict()}",
                flush=True,
            )
        if args.chat_data:
            print(
                f"stream_sources={list(stream_filter_stats)} "
                f"chat_max_characters={chat_max_characters}",
                flush=True,
            )

    saved_best_metrics = (
        resume_state.get("data_config", {}).get("best_metrics", {})
        if resume_state is not None
        else {}
    )
    saved_validation_mode = saved_best_metrics.get(
        "validation_decode_mode",
        resume_decode_mode,
    )
    best_validation_loss = (
        float(saved_best_metrics.get("validation_loss", math.inf))
        if saved_validation_mode == args.decode_mode
        else math.inf
    )
    if resume_state is not None and resume_decode_mode == args.decode_mode:
        best_validation_loss = min(
            best_validation_loss,
            float(
                resume_state.get("validation", {}).get(
                    "selection_loss",
                    math.inf,
                )
            ),
        )
    best_teacher_validation_loss = float(
        saved_best_metrics.get("teacher_validation_loss", math.inf)
    )
    if resume_state is not None:
        if resume_decode_mode == "teacher_forced":
            best_teacher_validation_loss = min(
                best_teacher_validation_loss,
                float(
                    resume_state.get("validation", {}).get(
                        "selection_loss",
                        math.inf,
                    )
                ),
            )
        elif resume_decode_mode in {
            "handoff",
            "graduated_handoff",
            "token_handoff",
            "prefix_denoise",
        }:
            best_teacher_validation_loss = min(
                best_teacher_validation_loss,
                float(
                    (resume_state.get("auxiliary_validation") or {}).get(
                        "selection_loss",
                        math.inf,
                    )
                ),
            )
    best_unsupported_validation_loss = float(
        saved_best_metrics.get("unsupported_validation_loss", math.inf)
    )
    best_free_content_loss = float(
        saved_best_metrics.get("free_content_loss", math.inf)
    )
    if resume_state is not None and resume_decode_mode == "token_handoff":
        resumed_unsupported = resume_state.get("support_validation") or {}
        best_unsupported_validation_loss = min(
            best_unsupported_validation_loss,
            float(resumed_unsupported.get("selection_loss", math.inf)),
        )
        best_free_content_loss = min(
            best_free_content_loss,
            float(resumed_unsupported.get("content_loss", math.inf)),
        )
    best_context_score = float(
        saved_best_metrics.get("context_score", -math.inf)
    )
    if resume_state is not None:
        resumed_teacher = resume_state.get("auxiliary_validation") or {}
        resumed_supported = resume_state.get("support_validation") or {}
        resumed_shuffled = resume_state.get("context_validation") or {}
        if (
            "content_accuracy" in resumed_teacher
            and "content_accuracy" in resumed_supported
            and "content_accuracy" in resumed_shuffled
        ):
            resumed_utility = (
                float(resumed_supported["content_accuracy"])
                - float(resumed_shuffled["content_accuracy"])
            )
            best_context_score = max(
                best_context_score,
                max(resumed_utility, 0.0)
                * float(resumed_teacher["content_accuracy"]),
            )
    best_training_loss = float(
        saved_best_metrics.get("training_loss", math.inf)
    )
    if resume_state is not None:
        best_training_loss = min(
            best_training_loss,
            float(
                (resume_state.get("training_evaluation") or {}).get(
                    "selection_loss", math.inf
                )
            ),
        )
    display_random = random.Random(args.seed + 1_000_003)
    if device.type == "cuda":
        torch.cuda.synchronize()
        # Reset after model/optimizer construction so this reports the real
        # training high-water mark, including the live model and optimizer
        # state plus lazy compilation and per-step intermediates.
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    last_log_time = started
    executed_steps = 0
    stewardship_enabled = (
        args.architecture == "gator"
        and args.cell_transport_mode == "stewardship"
    )
    conflict_ema = None
    conflict_lr_dropped = False
    conflict_seen_negative = False
    accumulation_steps = args.gradient_accumulation_steps
    engine.zero_grad()
    last_gradient_norm = 0.0
    for local_step in range(args.steps):
        completed = start_step + local_step + 1
        local_completed = local_step + 1
        elapsed_before_step = time.perf_counter() - started
        mean_step_seconds = (
            elapsed_before_step / local_step
            if local_step > 0
            else 0.0
        )
        runtime_log_due = (
            args.max_runtime_seconds > 0
            and (
                elapsed_before_step >= args.max_runtime_seconds
                or (
                    mean_step_seconds > 0
                    and elapsed_before_step + mean_step_seconds
                    >= args.max_runtime_seconds
                )
            )
        )
        interval_log_due = (
            args.log_every_seconds > 0
            and time.perf_counter() - last_log_time
            >= args.log_every_seconds
        )
        log_this_step = (
            completed % args.log_every == 0
            or local_completed == args.steps
            or interval_log_due
            or runtime_log_due
        )
        batch = train_pipeline.next_batch()
        active_decode_mode = args.decode_mode
        if handoff_controller is not None:
            if isinstance(
                handoff_controller,
                GraduatedTeacherHandoff,
            ):
                teacher_batch = handoff_controller.choose_teacher_batch(
                    completed_step=completed
                )
            else:
                teacher_batch = (
                    handoff_controller.choose_teacher_batch()
                )
            active_decode_mode = (
                "teacher_forced"
                if teacher_batch
                else "parallel"
            )
        if active_decode_mode == "serial":
            batch = serial_next_token_batch(
                batch,
                eos_token_id=tokenizer.EOS_ID,
                positions_per_example=args.serial_positions_per_example,
            )
        elif active_decode_mode == "chunk_parallel":
            batch = parallel_next_chunk_batch(
                batch,
                eos_token_id=tokenizer.EOS_ID,
                chunk_size=args.chunk_size,
            )
        training_output_slots = batch.target_ids.shape[1]
        teacher_forced_kwargs = {}
        prefix_corruption = None
        token_handoff_corruption = None
        token_handoff_unsupported_mask = None
        if active_decode_mode in {
            "teacher_forced",
            "prefix_denoise",
            "token_handoff",
        }:
            if tokenizer.blank_token_id is None:
                raise RuntimeError(
                    "causal response-prefix decoding requires a BLANK token"
                )
            if active_decode_mode == "token_handoff":
                assert token_handoff_controller is not None
                training_removed_tokens = (
                    token_handoff_controller.training_removed_tokens()
                )
                (
                    output_token_ids,
                    token_handoff_unsupported_mask,
                    token_handoff_corruption,
                ) = tokenwise_teacher_output_inputs(
                    batch,
                    blank_token_id=tokenizer.blank_token_id,
                    removed_tokens=training_removed_tokens,
                    seed=args.seed + 31,
                    rotation=completed,
                )
            elif active_decode_mode == "prefix_denoise":
                assert prefix_denoising_controller is not None
                output_token_ids, prefix_corruption = (
                    prefix_denoising_output_inputs(
                        batch,
                        blank_token_id=tokenizer.blank_token_id,
                        retention=prefix_denoising_controller.retention(
                            completed
                        ),
                        zero_prefix_fraction=(
                            args.prefix_zero_row_fraction
                        ),
                        short_prefix_fraction=(
                            args.prefix_short_row_fraction
                        ),
                        short_prefix_max_tokens=(
                            args.prefix_short_max_tokens
                        ),
                        span_fraction=args.prefix_span_fraction,
                    )
                )
            else:
                output_token_ids = teacher_forced_output_inputs(
                    batch,
                    blank_token_id=tokenizer.blank_token_id,
                )
            teacher_forced_kwargs = {
                "output_token_ids": output_token_ids,
                "output_mask": batch.target_mask,
                "causal": True,
            }
        if args.force_causal:
            teacher_forced_kwargs["causal"] = True
        if args.route_temperature_schedule == "staged_reheat":
            temperature_value = staged_route_temperature(
                local_step=(
                    args.route_temperature_step_offset + local_completed
                ),
                planned_steps=(
                    args.route_temperature_total_steps or args.steps
                ),
                start=args.route_temperature_start,
                middle=args.route_temperature_middle,
                end=args.route_temperature_end,
                hold_fraction=args.route_temperature_hold_fraction,
                middle_fraction=args.route_temperature_middle_fraction,
            )
        else:
            temperature_value = max(
                0.35,
                1.25 * (0.997 ** (completed - 1)),
            )
        # Python floats make Dynamo specialize the compiled Convergent graph
        # on every routing-temperature value. A scalar tensor preserves the
        # schedule while keeping one graph across training steps.
        temperature = (
            torch.tensor(
                temperature_value,
                device=device,
                dtype=torch.float32,
            )
            if args.compile and args.architecture == "convergent"
            else temperature_value
        )
        training_progress = local_step / max(args.steps - 1, 1)
        phase_progress = 1.0 if resume_state is not None else training_progress
        if args.formation_objective in {"viterbi", "monotonic"}:
            alignment_mix = (
                1.0
                if args.alignment_warmup_fraction == 0
                else min(1.0, phase_progress / args.alignment_warmup_fraction)
            )
        else:
            alignment_mix = 0.0
        stewardship_temperature = args.stewardship_temperature_start * (
            args.stewardship_temperature_end
            / args.stewardship_temperature_start
        ) ** phase_progress
        stewardship_prune = (
            stewardship_enabled
            and phase_progress >= args.stewardship_warmup_fraction
        )
        if args.stewardship_warmup_fraction < 1.0:
            pruning_progress = max(
                0.0,
                (phase_progress - args.stewardship_warmup_fraction)
                / max(1.0 - args.stewardship_warmup_fraction, 1e-9),
            )
        else:
            pruning_progress = 0.0
        mark_compiled_iteration_begin(compute_config.compile, device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(args.mixed_precision == "bf16" and device.type == "cuda"),
        ):
            if stewardship_enabled:
                execution_plan = execution_controller.plan
                active_train_model = (
                    model
                    if execution_plan.uses_packed_execution
                    else dense_train_model
                )
                logits, route_diagnostics = active_train_model(
                    batch.context_ids,
                    batch.context_mask,
                    output_slots=training_output_slots,
                    **teacher_forced_kwargs,
                    temperature=temperature,
                    stewardship_temperature=stewardship_temperature,
                    stewardship_prune=stewardship_prune,
                    stewardship_packed_from_relay=(
                        execution_plan.packed_from_relay
                    ),
                    diagnostics="route_loss",
                )
                familiar_route_entropy, route_balance_loss = (
                    stewardship_route_losses_from_statistics(
                        route_diagnostics["route_entropy"],
                        route_diagnostics["route_cell_load"],
                        route_diagnostics["route_mean_uncertainty"],
                        route_diagnostics["route_valid"],
                    )
                )
                lane_balance_loss = torch.zeros((), device=device)
            elif args.reasoning_loops == 2:
                compute_lane_balance = (
                    args.architecture == "convergent"
                    and args.lane_balance_loss_weight > 0
                    and args.terminal_vote_mode != "global"
                )
                pad_ids = torch.zeros_like(batch.context_ids[:, :1])
                pad_mask = torch.zeros_like(batch.context_mask[:, :1])
                draft_context_ids = torch.cat([batch.context_ids, pad_ids], dim=1)
                draft_context_mask = torch.cat(
                    [batch.context_mask, pad_mask], dim=1
                )
                with torch.no_grad():
                    draft_logits = dense_train_model(
                        draft_context_ids,
                        draft_context_mask,
                        output_slots=training_output_slots,
                        **teacher_forced_kwargs,
                        temperature=temperature,
                    )
                    draft_token = draft_logits.argmax(dim=-1)[:, :1]
                refine_context_ids = torch.cat(
                    [batch.context_ids, draft_token], dim=1
                )
                refine_context_mask = torch.cat(
                    [batch.context_mask, torch.ones_like(pad_mask)], dim=1
                )
                if compute_lane_balance:
                    logits, lane_trace = dense_train_model(
                        refine_context_ids,
                        refine_context_mask,
                        output_slots=training_output_slots,
                        **teacher_forced_kwargs,
                        temperature=temperature,
                        return_trace=True,
                    )
                    lane_balance_loss = lane_balance_loss_from_terminal_weights(
                        lane_trace["terminal_weights"]
                    )
                else:
                    logits = dense_train_model(
                        refine_context_ids,
                        refine_context_mask,
                        output_slots=training_output_slots,
                        **teacher_forced_kwargs,
                        temperature=temperature,
                    )
                    lane_balance_loss = torch.zeros((), device=device)
                familiar_route_entropy = torch.zeros((), device=device)
                route_balance_loss = torch.zeros((), device=device)
            else:
                compute_lane_balance = (
                    args.architecture == "convergent"
                    and args.lane_balance_loss_weight > 0
                    and args.terminal_vote_mode != "global"
                )
                if compute_lane_balance:
                    logits, lane_trace = dense_train_model(
                        batch.context_ids,
                        batch.context_mask,
                        output_slots=training_output_slots,
                        **teacher_forced_kwargs,
                        temperature=temperature,
                        return_trace=True,
                    )
                    lane_balance_loss = lane_balance_loss_from_terminal_weights(
                        lane_trace["terminal_weights"]
                    )
                else:
                    logits = dense_train_model(
                        batch.context_ids,
                        batch.context_mask,
                        output_slots=training_output_slots,
                        **teacher_forced_kwargs,
                        temperature=temperature,
                    )
                    lane_balance_loss = torch.zeros((), device=device)
                familiar_route_entropy = torch.zeros((), device=device)
                route_balance_loss = torch.zeros((), device=device)
        formation_components = formation_loss_components(
            logits,
            batch,
            tokenizer,
            valid_token_mask=legal_response_mask,
            semantic_token_mask=semantic_token_mask,
            objective=args.formation_objective,
            alignment_mix=alignment_mix,
            max_right_shift=(
                None if args.alignment_max_shift < 0 else args.alignment_max_shift
            ),
            sequence_composition_weight=args.sequence_composition_weight,
            sequence_length_weight=args.sequence_length_weight,
            sequence_eos_weight=args.sequence_eos_weight,
            compute_all=log_this_step,
            compute_sequence=args.sequence_quality_penalty > 0,
        )
        cross_entropy_loss = formation_components["slot_ce"]
        blank_cross_entropy_loss = formation_components["blank_ce"]
        formation_objective_loss = formation_components["selected"]
        if token_handoff_unsupported_mask is not None:
            token_handoff_unforced_loss = token_handoff_cross_entropy(
                formation_components["log_probs"],
                batch.target_ids,
                token_handoff_unsupported_mask,
                batch.target_mask,
                normalization=args.token_handoff_loss_normalization,
            )
        else:
            token_handoff_unforced_loss = torch.zeros((), device=device)
        active_alignment_component = (
            formation_components[args.formation_objective]
            if args.formation_objective in {"viterbi", "monotonic"}
            else None
        )
        active_alignment_loss = (
            active_alignment_component
            if active_alignment_component is not None
            else torch.zeros((), device=device)
        )
        sequence_result = formation_components["sequence"]
        sequence_auxiliary_loss = (
            sequence_result.loss
            if sequence_result is not None
            else torch.zeros((), device=device)
        )
        if (
            args.whitespace_omission_penalty > 0
            or args.adjacent_whitespace_penalty > 0
        ):
            omission_loss, adjacent_whitespace_loss = whitespace_formation_penalties(
                logits,
                batch.target_ids,
                batch.target_mask,
                whitespace_mask,
                semantic_token_mask[batch.target_ids],
                probabilities=formation_components["log_probs"].exp(),
            )
        else:
            omission_loss = adjacent_whitespace_loss = torch.zeros(
                (), device=device
            )
        loss = (
            formation_objective_loss
            + args.sequence_quality_penalty * sequence_auxiliary_loss
            + args.whitespace_omission_penalty * omission_loss
            + args.adjacent_whitespace_penalty * adjacent_whitespace_loss
            + args.token_handoff_unforced_loss_weight
            * token_handoff_unforced_loss
            + args.stewardship_familiar_entropy_penalty
            * pruning_progress
            * familiar_route_entropy
            + args.stewardship_balance_penalty
            * (1.0 - phase_progress)
            * route_balance_loss
            + args.lane_balance_loss_weight * lane_balance_loss
        )
        if args.diagnose_route_conflict and log_this_step:
            log_probs_diag = formation_components["log_probs"]
            n_slots = log_probs_diag.shape[1]
            shared_params = [
                (name, param)
                for name, param in model.named_parameters()
                if param.requires_grad
                and any(sub in name for sub in args.diagnose_route_conflict_params)
            ]
            if shared_params:
                param_list = [param for _, param in shared_params]

                def _slot_loss(slot_index: int) -> torch.Tensor:
                    active = batch.target_mask[:, slot_index].bool()
                    token_nll = -log_probs_diag[:, slot_index, :].gather(
                        1, batch.target_ids[:, slot_index, None]
                    ).squeeze(1)
                    masked = torch.where(active, token_nll, torch.zeros_like(token_nll))
                    return masked.sum() / active.sum().clamp_min(1).to(masked.dtype)

                slot_vecs = []
                for slot_index in range(n_slots):
                    slot_loss = _slot_loss(slot_index)
                    grads = torch.autograd.grad(
                        slot_loss,
                        param_list,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    flat = [g.reshape(-1) for g in grads if g is not None]
                    slot_vecs.append(torch.cat(flat) if flat else None)
                adjacent_cos = []
                distant_cos = []
                for i in range(n_slots):
                    for j in range(i + 1, n_slots):
                        vi, vj = slot_vecs[i], slot_vecs[j]
                        if vi is None or vj is None:
                            continue
                        cos = torch.nn.functional.cosine_similarity(
                            vi, vj, dim=0
                        ).item()
                        (adjacent_cos if j - i == 1 else distant_cos).append(cos)
                if adjacent_cos or distant_cos:
                    all_cos = adjacent_cos + distant_cos
                    mean_all = sum(all_cos) / len(all_cos)
                    mean_adj = (
                        sum(adjacent_cos) / len(adjacent_cos)
                        if adjacent_cos
                        else float("nan")
                    )
                    mean_dist = (
                        sum(distant_cos) / len(distant_cos)
                        if distant_cos
                        else float("nan")
                    )
                    print(
                        f"route_conflict step={completed} "
                        f"mean_cos_all={mean_all:.4f} "
                        f"mean_cos_adjacent={mean_adj:.4f} "
                        f"mean_cos_distant={mean_dist:.4f} "
                        f"n_pairs={len(all_cos)} n_adjacent={len(adjacent_cos)} "
                        f"n_slots={n_slots} shared_params={len(param_list)}",
                        flush=True,
                    )
                    if args.auto_lr_drop_on_conflict_cross and adjacent_cos:
                        conflict_ema = (
                            mean_adj
                            if conflict_ema is None
                            else args.auto_lr_drop_ema_decay * conflict_ema
                            + (1 - args.auto_lr_drop_ema_decay) * mean_adj
                        )
                        if conflict_ema < args.auto_lr_drop_threshold:
                            conflict_seen_negative = True
                        if (
                            not conflict_lr_dropped
                            and conflict_seen_negative
                            and completed >= args.auto_lr_drop_min_step
                            and conflict_ema >= args.auto_lr_drop_threshold
                        ):
                            for group in engine.optimizer.param_groups:
                                group["lr"] *= args.auto_lr_drop_gamma
                            conflict_lr_dropped = True
                            print(
                                f"auto_lr_drop step={completed} "
                                f"conflict_ema={conflict_ema:.4f} "
                                f"gamma={args.auto_lr_drop_gamma} "
                                f"new_lr={engine.optimizer.param_groups[0]['lr']:.6g}",
                                flush=True,
                            )
                else:
                    print(
                        f"route_conflict step={completed} no_shared_grads=True",
                        flush=True,
                    )
        engine.accumulate(loss, scale=1.0 / accumulation_steps)
        if local_completed % accumulation_steps == 0:
            step_result = engine.apply_accumulated_step(loss)
            engine.zero_grad()
            last_gradient_norm = step_result.gradient_norm
        else:
            # Mid-window micro-batch: no clip/step yet (clipping a partial
            # accumulation would corrupt the sum before later micro-batches
            # add to it), so report the last real update's gradient norm and
            # the unchanged learning rate for logging continuity.
            step_result = StepResult(
                step=engine.step_number,
                loss=float(loss.detach()),
                learning_rate=engine.learning_rate,
                gradient_norm=last_gradient_norm,
            )
        executed_steps = local_completed
        if (
            execution_controller is not None
            and execution_controller.should_observe(
                completed, pruning=stewardship_prune
            )
        ):
            previous_plan = execution_controller.plan
            active_fraction_by_relay = route_diagnostics[
                "route_active_fraction"
            ]
            observed_fractions = (
                active_fraction_by_relay.detach().cpu().tolist()
            )
            observed_plan = execution_controller.observe(
                observed_fractions
            )
            if observed_plan != previous_plan:
                print(
                    f"execution_transition step={completed} "
                    f"active_fraction_by_relay="
                    f"{observed_fractions} "
                    f"plan={execution_label()} "
                    f"packed_call_eager=True "
                    f"pre_latch_dense_compiled={dense_compile_enabled}",
                    flush=True,
                )
        if log_this_step:
            assert cross_entropy_loss is not None
            assert blank_cross_entropy_loss is not None
            evaluation_decode_mode = (
                "token_handoff"
                if token_handoff_controller is not None
                else (
                    "parallel"
                    if (
                        handoff_controller is not None
                        or prefix_denoising_controller is not None
                    )
                    else args.decode_mode
                )
            )
            evaluation_removed_tokens = (
                token_handoff_controller.removed_tokens
                if token_handoff_controller is not None
                else 0
            )
            evaluation_display_index = display_random.randrange(
                min(args.eval_examples, len(validation_examples))
                if args.eval_examples > 0
                else len(validation_examples)
            )
            result = evaluate(
                model,
                validation_batcher,
                tokenizer,
                device,
                batch_size=args.eval_batch_size,
                limit=args.eval_examples,
                display_index=evaluation_display_index,
                stewardship_temperature=(
                    args.stewardship_temperature_end
                    if stewardship_enabled
                    else None
                ),
                stewardship_prune=True,
                stewardship_packed_from_relay=(
                    execution_controller.plan.packed_from_relay
                    if execution_controller is not None
                    else None
                ),
                formation_objective=args.formation_objective,
                alignment_max_shift=(
                    None if args.alignment_max_shift < 0 else args.alignment_max_shift
                ),
                sequence_composition_weight=args.sequence_composition_weight,
                sequence_length_weight=args.sequence_length_weight,
                sequence_eos_weight=args.sequence_eos_weight,
                decode_mode=evaluation_decode_mode,
                chunk_size=args.chunk_size,
                teacher_removed_tokens=evaluation_removed_tokens,
                reasoning_loops=args.reasoning_loops,
                mixed_precision=args.mixed_precision,
            )
            teacher_result = None
            unsupported_result = None
            token_handoff_action = "hold"
            supported_context_result = None
            shuffled_context_result = None
            if (
                handoff_controller is not None
                or prefix_denoising_controller is not None
                or token_handoff_controller is not None
            ):
                teacher_result = evaluate(
                    model,
                    validation_batcher,
                    tokenizer,
                    device,
                    batch_size=args.eval_batch_size,
                    limit=args.eval_examples,
                    display_index=evaluation_display_index,
                    stewardship_temperature=(
                        args.stewardship_temperature_end
                        if stewardship_enabled
                        else None
                    ),
                    stewardship_prune=True,
                    stewardship_packed_from_relay=(
                        execution_controller.plan.packed_from_relay
                        if execution_controller is not None
                        else None
                    ),
                    formation_objective=args.formation_objective,
                    alignment_max_shift=(
                        None
                        if args.alignment_max_shift < 0
                        else args.alignment_max_shift
                    ),
                    sequence_composition_weight=(
                        args.sequence_composition_weight
                    ),
                    sequence_length_weight=args.sequence_length_weight,
                    sequence_eos_weight=args.sequence_eos_weight,
                    decode_mode="teacher_forced",
                    chunk_size=args.chunk_size,
                    teacher_removed_tokens=evaluation_removed_tokens,
                )
                if handoff_controller is not None:
                    if isinstance(
                        handoff_controller,
                        GraduatedTeacherHandoff,
                    ):
                        handoff_controller.observe(
                            result["content_accuracy"],
                            completed_step=completed,
                        )
                    else:
                        handoff_controller.observe(
                            teacher_result["content_accuracy"]
                        )
                if token_handoff_controller is not None:
                    unsupported_result = evaluate(
                        model,
                        validation_batcher,
                        tokenizer,
                        device,
                        batch_size=args.eval_batch_size,
                        limit=args.eval_examples,
                        display_index=evaluation_display_index,
                        stewardship_temperature=(
                            args.stewardship_temperature_end
                            if stewardship_enabled
                            else None
                        ),
                        stewardship_prune=True,
                        stewardship_packed_from_relay=(
                            execution_controller.plan.packed_from_relay
                            if execution_controller is not None
                            else None
                        ),
                        formation_objective=args.formation_objective,
                        alignment_max_shift=(
                            None
                            if args.alignment_max_shift < 0
                            else args.alignment_max_shift
                        ),
                        sequence_composition_weight=(
                            args.sequence_composition_weight
                        ),
                        sequence_length_weight=(
                            args.sequence_length_weight
                        ),
                        sequence_eos_weight=args.sequence_eos_weight,
                        decode_mode="token_handoff",
                        chunk_size=args.chunk_size,
                        teacher_removed_tokens=args.response_tokens - 1,
                    )
                    token_handoff_action = (
                        token_handoff_controller.observe(
                            unsupported_accuracy=result[
                                "unsupported_content_accuracy"
                            ],
                            teacher_accuracy=teacher_result[
                                "unsupported_content_accuracy"
                            ],
                            teacher_content_perplexity=teacher_result[
                                "content_perplexity"
                            ],
                        )
                    )
                if prefix_denoising_controller is not None:
                    evaluation_retention = (
                        prefix_denoising_controller.retention(completed)
                    )
                    supported_context_result = evaluate(
                        model,
                        validation_batcher,
                        tokenizer,
                        device,
                        batch_size=args.eval_batch_size,
                        limit=args.eval_examples,
                        display_index=evaluation_display_index,
                        stewardship_temperature=(
                            args.stewardship_temperature_end
                            if stewardship_enabled
                            else None
                        ),
                        stewardship_prune=True,
                        stewardship_packed_from_relay=(
                            execution_controller.plan.packed_from_relay
                            if execution_controller is not None
                            else None
                        ),
                        formation_objective=args.formation_objective,
                        alignment_max_shift=(
                            None
                            if args.alignment_max_shift < 0
                            else args.alignment_max_shift
                        ),
                        sequence_composition_weight=(
                            args.sequence_composition_weight
                        ),
                        sequence_length_weight=(
                            args.sequence_length_weight
                        ),
                        sequence_eos_weight=args.sequence_eos_weight,
                        decode_mode="teacher_forced",
                        chunk_size=args.chunk_size,
                        context_mode="correct",
                        teacher_prefix_retention=evaluation_retention,
                    )
                    shuffled_context_result = evaluate(
                        model,
                        validation_batcher,
                        tokenizer,
                        device,
                        batch_size=args.eval_batch_size,
                        limit=args.eval_examples,
                        display_index=evaluation_display_index,
                        stewardship_temperature=(
                            args.stewardship_temperature_end
                            if stewardship_enabled
                            else None
                        ),
                        stewardship_prune=True,
                        stewardship_packed_from_relay=(
                            execution_controller.plan.packed_from_relay
                            if execution_controller is not None
                            else None
                        ),
                        formation_objective=args.formation_objective,
                        alignment_max_shift=(
                            None
                            if args.alignment_max_shift < 0
                            else args.alignment_max_shift
                        ),
                        sequence_composition_weight=(
                            args.sequence_composition_weight
                        ),
                        sequence_length_weight=(
                            args.sequence_length_weight
                        ),
                        sequence_eos_weight=args.sequence_eos_weight,
                        decode_mode="teacher_forced",
                        chunk_size=args.chunk_size,
                        context_mode="shuffled",
                        teacher_prefix_retention=evaluation_retention,
                    )
                    prefix_denoising_controller.observe(
                        content_accuracy=teacher_result[
                            "content_accuracy"
                        ],
                        shuffled_context_accuracy=(
                            shuffled_context_result[
                                "content_accuracy"
                            ]
                        ),
                        supported_context_accuracy=(
                            supported_context_result[
                                "content_accuracy"
                            ]
                        ),
                    )
            training_result = None
            if training_evaluation_batcher is not None:
                training_result = evaluate(
                    model,
                    training_evaluation_batcher,
                    tokenizer,
                    device,
                    batch_size=args.eval_batch_size,
                    limit=0,
                    display_index=0,
                    stewardship_temperature=(
                        args.stewardship_temperature_end
                        if stewardship_enabled
                        else None
                    ),
                    stewardship_prune=True,
                    stewardship_packed_from_relay=(
                        execution_controller.plan.packed_from_relay
                        if execution_controller is not None
                        else None
                    ),
                    formation_objective=args.formation_objective,
                    alignment_max_shift=(
                        None
                        if args.alignment_max_shift < 0
                        else args.alignment_max_shift
                    ),
                    sequence_composition_weight=args.sequence_composition_weight,
                    sequence_length_weight=args.sequence_length_weight,
                    sequence_eos_weight=args.sequence_eos_weight,
                    decode_mode=evaluation_decode_mode,
                    chunk_size=args.chunk_size,
                    reasoning_loops=args.reasoning_loops,
                    mixed_precision=args.mixed_precision,
                )
            current_steps_per_second = local_completed / max(
                time.perf_counter() - started,
                1e-9,
            )
            if args.logging_slim:
                print(
                    slim_step_log_line(
                        step=completed,
                        steps_per_second=current_steps_per_second,
                        result=result,
                    ),
                    flush=True,
                )
            else:
                print(
                    f"step={completed} acc={result['accuracy']:.3f} exact={result['exact']:.3f} "
                    f"loss={result['loss']:.4f} ppl={result['perplexity']:.2f} "
                    f"selection_loss={result['selection_loss']:.4f} "
                    f"blank_ce={result['blank_ce_loss']:.4f} "
                    f"best_aligned_loss={result['best_aligned_loss']:.4f} "
                    f"best_aligned_required={result['best_aligned_required_loss']:.4f} "
                    f"aligned_loss={result['aligned_loss']:.4f} "
                    f"aligned_required={result['aligned_required_loss']:.4f} "
                    f"train_ce={float(cross_entropy_loss.detach()):.4f} "
                    f"train_blank_ce={float(blank_cross_entropy_loss.detach()):.4f} "
                    f"train_objective={float(formation_objective_loss.detach()):.4f} "
                    f"train_unforced_ce="
                    f"{float(token_handoff_unforced_loss.detach()):.4f} "
                    f"train_alignment={float(active_alignment_loss.detach()):.4f} "
                    f"alignment_mix={alignment_mix:.3f} "
                    f"sequence_loss={float(sequence_auxiliary_loss.detach()):.4f} "
                    f"ws_omit_loss={float(omission_loss.detach()):.4f} "
                    f"ws_adj_loss={float(adjacent_whitespace_loss.detach()):.4f} "
                    f"route_entropy_loss={float(familiar_route_entropy.detach()):.4f} "
                    f"route_balance_loss={float(route_balance_loss.detach()):.4f} "
                    f"stewardship_temp={stewardship_temperature:.4f} "
                    f"stewardship_prune={stewardship_prune} "
                    f"execution_plan={execution_label()} "
                    f"lr={step_result.learning_rate:.6g} grad_norm={step_result.gradient_norm:.4f}",
                    f"steps_per_second={current_steps_per_second:.2f}",
                    flush=True,
                )
            if handoff_controller is not None:
                assert teacher_result is not None
                print(
                    handoff_log_line(
                        controller=handoff_controller,
                        teacher_result=teacher_result,
                        parallel_result=result,
                    ),
                    flush=True,
                )
            if token_handoff_controller is not None:
                assert teacher_result is not None
                assert unsupported_result is not None
                print(
                    token_handoff_log_line(
                        controller=token_handoff_controller,
                        partial_result=result,
                        teacher_result=teacher_result,
                        unsupported_result=unsupported_result,
                        action=token_handoff_action,
                    ),
                    flush=True,
                )
            if prefix_denoising_controller is not None:
                assert prefix_corruption is not None
                assert teacher_result is not None
                assert supported_context_result is not None
                assert shuffled_context_result is not None
                print(
                    prefix_denoising_log_line(
                        controller=prefix_denoising_controller,
                        completed_step=completed,
                        corruption=prefix_corruption,
                        teacher_result=teacher_result,
                        supported_context_result=supported_context_result,
                        shuffled_context_result=shuffled_context_result,
                    ),
                    flush=True,
                )
            pipeline_stats = train_pipeline.stats
            if args.logging_slim:
                print(
                    f"pipeline queue={pipeline_stats.queue_depth}/"
                    f"{pipeline_stats.queue_capacity} "
                    f"sample_ms="
                    f"{1000.0 * pipeline_stats.sample_seconds / max(pipeline_stats.sampled_batches, 1):.2f} "
                    f"transfer_ms="
                    f"{1000.0 * pipeline_stats.transfer_seconds / max(pipeline_stats.device_transfers, 1):.2f} "
                    f"producer_waits={pipeline_stats.producer_waits} "
                    f"failures={pipeline_stats.producer_failures}",
                    flush=True,
                )
            else:
                print(
                    f"data_pipeline mode={pipeline_stats.mode} "
                    f"queue={pipeline_stats.queue_depth}/"
                    f"{pipeline_stats.queue_capacity} "
                    f"high_watermark={pipeline_stats.queue_high_watermark} "
                    f"sampled={pipeline_stats.sampled_batches} "
                    f"consumed={pipeline_stats.consumed_batches} "
                    f"sample_ms_per_batch="
                    f"{1000.0 * pipeline_stats.sample_seconds / max(pipeline_stats.sampled_batches, 1):.3f} "
                    f"transfer_ms_per_batch="
                    f"{1000.0 * pipeline_stats.transfer_seconds / max(pipeline_stats.device_transfers, 1):.3f} "
                    f"producer_waits={pipeline_stats.producer_waits} "
                    f"failures={pipeline_stats.producer_failures}",
                    flush=True,
                )
            if "stewardship" in result:
                stewardship_result = result["stewardship"]
                if args.logging_slim:
                    print(
                        slim_routing_log_line(stewardship_result),
                        flush=True,
                    )
                else:
                    print(
                        f"stewardship support={stewardship_result['support_by_relay']} "
                        f"effective={stewardship_result['effective_cells_by_relay']} "
                        f"uncertainty={stewardship_result['uncertainty_by_relay']} "
                        f"mean_uncertainty={stewardship_result['mean_uncertainty_by_relay']} "
                        f"handoff_beam={stewardship_result['handoff_support_by_relay']} "
                        f"widen={stewardship_result['widen_rate_by_relay']} "
                        f"narrow={stewardship_result['narrow_rate_by_relay']} "
                        f"stay={stewardship_result['stay_rate_by_relay']} "
                        f"load_cv={stewardship_result['load_cv_by_relay']} "
                        f"dead={stewardship_result['dead_cell_fraction_by_relay']}",
                        f"packed={stewardship_result['packed_execution_by_relay']}",
                        flush=True,
                    )
            if "terminal_voting" in result:
                print(
                    slim_terminal_vote_log_line(
                        result["terminal_voting"]
                    ),
                    flush=True,
                )
            if not args.logging_slim:
                print(
                    f"formation content_acc={result['content_accuracy']:.3f} "
                    f"content_ppl={result['content_perplexity']:.2f} "
                    f"content_top5={result['content_top5_accuracy']:.3f} "
                    f"ws_omission={result['whitespace_omission_rate']:.3f} "
                    f"ws_mass={result['whitespace_mass_on_content']:.3f} "
                    f"ws_adjacent={result['adjacent_whitespace_rate']:.3f} "
                    f"coverage={result['formation_coverage']:.3f}",
                    flush=True,
                )
                print(
                    f"alignment best_acc={result['best_aligned_accuracy']:.3f} "
                    f"required_acc={result['aligned_required_accuracy']:.3f} "
                    f"content_acc={result['aligned_content_accuracy']:.3f} "
                    f"content_prf={result['aligned_content_precision']:.3f}:"
                    f"{result['aligned_content_recall']:.3f}:"
                    f"{result['aligned_content_f1']:.3f} "
                    f"edit_acc={result['aligned_edit_accuracy']:.3f} "
                    f"exact={result['aligned_exact']:.3f} "
                    f"displacement={result['mean_alignment_displacement']:.3f} "
                    f"max_displacement={result['mean_max_alignment_displacement']:.3f} "
                    f"edits={result['edit_insertions_per_example']:.3f}:"
                    f"{result['edit_deletions_per_example']:.3f}:"
                    f"{result['edit_substitutions_per_example']:.3f} "
                    f"length={result['predicted_length']:.2f}/"
                    f"{result['reference_length']:.2f} "
                    f"blank={result['blank_prediction_rate']:.3f} "
                    f"complete={result['complete_fraction']:.3f} "
                    f"missing_eos={result['missing_eos_rate']:.3f} "
                    f"eos_mae={result['eos_position_mae']:.3f} "
                    f"post_eos={result['post_eos_violation_rate']:.3f} "
                    f"forbidden={result['forbidden_prediction_rate']:.3f} "
                    f"seq_parts={result['sequence_composition_loss']:.4f}:"
                    f"{result['sequence_length_loss']:.4f}:"
                    f"{result['sequence_eos_loss']:.4f}",
                    flush=True,
                )
            if training_result is not None and not args.logging_slim:
                print(
                    f"train_subset examples={training_result['examples']} "
                    f"acc={training_result['accuracy']:.3f} "
                    f"exact={training_result['exact']:.3f} "
                    f"loss={training_result['loss']:.4f} "
                    f"ppl={training_result['perplexity']:.2f} "
                    f"selection_loss={training_result['selection_loss']:.4f} "
                    f"aligned_loss={training_result['aligned_loss']:.4f} "
                    f"aligned_exact={training_result['aligned_exact']:.3f}",
                    flush=True,
                )
            if weighted_stream_source is not None and args.logging_slim:
                print(
                    f"stream samples="
                    f"{weighted_stream_source.sample_counts} "
                    f"restarts={weighted_stream_source.restarts} "
                    f"exhausted="
                    f"{sorted(weighted_stream_source.exhausted_sources)}",
                    flush=True,
                )
                print(
                    "stream_filter_acceptance="
                    f"{ {
                        name: round(stats.acceptance_rate, 3)
                        for name, stats in stream_filter_stats.items()
                    } } "
                    "stream_filter_rejections="
                    f"{ {
                        name: {
                            'seen': stats.seen,
                            'rejected_response_length': (
                                stats.rejected_response_length
                            ),
                            'rejected_language': stats.rejected_language,
                        }
                        for name, stats in stream_filter_stats.items()
                    } }",
                    flush=True,
                )
            if weighted_stream_source is not None and not args.logging_slim:
                mixture_counts = {
                    "smoltalk_everyday": train_batcher.finite_samples,
                    **weighted_stream_source.sample_counts,
                }
                print(
                    f"stream_training_filter="
                    f"{ {name: stats.to_dict() for name, stats in stream_filter_stats.items()} } "
                    f"mixture_prepared_samples={mixture_counts} "
                    f"stream_restarts={weighted_stream_source.restarts} "
                    f"stream_exhausted="
                    f"{sorted(weighted_stream_source.exhausted_sources)}",
                    flush=True,
                )
            if not args.logging_slim:
                print(
                    f"sample=source:{result['sample_source']} "
                    f"conversation:{result['sample_conversation']} "
                    f"turn:{result['sample_turn']}",
                    flush=True,
                )
            print(f"context={result['context']!r}", flush=True)
            print(f"target={result['target']!r}", flush=True)
            print(f"formed={result['prediction']!r}", flush=True)
            if unsupported_result is not None:
                print(
                    f"free_formed={unsupported_result['prediction']!r}",
                    flush=True,
                )
            if not args.logging_slim:
                print(f"slots={result['prediction_slots']!r}", flush=True)
            current_context_score = -math.inf
            if (
                teacher_result is not None
                and supported_context_result is not None
                and shuffled_context_result is not None
            ):
                context_utility = (
                    supported_context_result["content_accuracy"]
                    - shuffled_context_result["content_accuracy"]
                )
                current_context_score = max(
                    context_utility,
                    0.0,
                ) * teacher_result["content_accuracy"]
            checkpoint_best_metrics = {
                "validation_decode_mode": args.decode_mode,
                "validation_loss": min(
                    best_validation_loss,
                    float(result["selection_loss"]),
                ),
                "teacher_validation_loss": min(
                    best_teacher_validation_loss,
                    float(
                        teacher_result["selection_loss"]
                        if teacher_result is not None
                        else result["selection_loss"]
                    ),
                ),
                "unsupported_validation_loss": min(
                    best_unsupported_validation_loss,
                    float(
                        unsupported_result["selection_loss"]
                        if unsupported_result is not None
                        else math.inf
                    ),
                ),
                "free_content_loss": min(
                    best_free_content_loss,
                    float(
                        unsupported_result["content_loss"]
                        if unsupported_result is not None
                        else math.inf
                    ),
                ),
                "context_score": max(
                    best_context_score,
                    current_context_score,
                ),
                "training_loss": min(
                    best_training_loss,
                    float(
                        training_result["selection_loss"]
                        if training_result is not None
                        else math.inf
                    ),
                ),
            }
            pipeline_checkpoint = train_pipeline.state_dict()
            state = checkpoint_state(
                step=completed,
                model=model,
                engine=engine,
                tokenizer=tokenizer,
                task=task_config,
                data_config={
                    "data_fingerprint": data_fingerprint,
                    "validation_fingerprint": validation_fingerprint,
                    "tinystories_path": args.tinystories_data,
                    "tinystories_validation_path": (
                        args.tinystories_validation_data
                    ),
                    "tinystories_mix_fraction": (
                        args.tinystories_mix_fraction
                    ),
                    "tinystories_validation_stories": (
                        args.tinystories_validation_stories
                    ),
                    "tinystories_tokenizer_stories": (
                        args.tinystories_tokenizer_stories
                    ),
                    "tinystories_shuffle_buffer": (
                        args.tinystories_shuffle_buffer
                    ),
                    "resolved_stream_shares": resolved_stream_shares,
                    "smoltalk_path": args.data,
                    "lmsys_path": args.lmsys_data,
                    "lmsys_weight": args.lmsys_weight,
                    "streaming_weight": streaming_weight,
                    "streaming_exhaustion": (
                        args.streaming_exhaustion
                    ),
                    "primary_stream_weight": (
                        args.primary_stream_weight
                    ),
                    "primary_shuffle_buffer": (
                        args.primary_shuffle_buffer
                    ),
                    "lmsys_source_weight": args.lmsys_source_weight,
                    "chat_paths": list(args.chat_data),
                    "chat_source_weights": list(args.chat_source_weight),
                    "chat_max_characters": chat_max_characters,
                    "chat_shuffle_buffer": args.chat_shuffle_buffer,
                    "chat_validation_conversations": args.chat_validation_conversations,
                    "prose_paths": list(args.prose_data),
                    "prose_source_weights": list(args.prose_source_weight),
                    "prose_tokenizer_stories": args.prose_tokenizer_stories,
                    "prose_validation_stories": args.prose_validation_stories,
                    "prose_shuffle_buffer": args.prose_shuffle_buffer,
                    "lmsys_max_characters": args.lmsys_max_characters,
                    "lmsys_validation_conversations": args.lmsys_validation_conversations,
                    "skip_first_assistant": args.skip_first_assistant,
                    "repeat_training_examples": args.repeat_training_examples,
                    "require_complete_windows": args.require_complete_windows,
                    "require_complete_context": args.require_complete_context,
                    "require_complete_response": args.require_complete_response,
                    "trim_context_turns": args.trim_context_turns,
                    "deduplicate_examples": args.deduplicate_examples,
                    "lmsys_min_ascii_letter_ratio": args.lmsys_min_ascii_letter_ratio,
                    "lmsys_min_response_characters": args.lmsys_min_response_characters,
                    "lmsys_require_user_prompt": args.lmsys_require_user_prompt,
                    "lmsys_skip_first_assistant": args.lmsys_skip_first_assistant,
                    "whitespace_omission_penalty": args.whitespace_omission_penalty,
                    "adjacent_whitespace_penalty": args.adjacent_whitespace_penalty,
                    "tokenizer_format": tokenizer.format_version,
                    "blank_token_id": tokenizer.blank_token_id,
                    "reference_tokens": task_config.target_tokens,
                    "response_tokens": task_config.response_tokens,
                    "decode_mode": args.decode_mode,
                    "chunk_size": args.chunk_size,
                    "handoff_start_content_accuracy": (
                        args.handoff_start_content_accuracy
                    ),
                    "handoff_end_content_accuracy": (
                        args.handoff_end_content_accuracy
                    ),
                    "handoff_accuracy_ema_decay": (
                        args.handoff_accuracy_ema_decay
                    ),
                    "graduated_baseline_fraction": (
                        args.graduated_baseline_fraction
                    ),
                    "graduated_accuracy_step": (
                        args.graduated_accuracy_step
                    ),
                    "graduated_release_accuracy": (
                        args.graduated_release_accuracy
                    ),
                    "handoff_state": (
                        handoff_controller.state_dict()
                        if handoff_controller is not None
                        else None
                    ),
                    "token_handoff_start_removed": (
                        args.token_handoff_start_removed
                    ),
                    "token_handoff_max_removed": (
                        maximum_token_removals
                    ),
                    "token_handoff_advance_ratio": (
                        args.token_handoff_advance_ratio
                    ),
                    "token_handoff_retreat_ratio": (
                        args.token_handoff_retreat_ratio
                    ),
                    "token_handoff_min_teacher_accuracy": (
                        args.token_handoff_min_teacher_accuracy
                    ),
                    "token_handoff_min_frontier_accuracy": (
                        args.token_handoff_min_frontier_accuracy
                    ),
                    "token_handoff_max_teacher_ppl_ratio": (
                        args.token_handoff_max_teacher_ppl_ratio
                    ),
                    "token_handoff_retreat_teacher_ppl_ratio": (
                        args.token_handoff_retreat_teacher_ppl_ratio
                    ),
                    "token_handoff_advance_patience": (
                        args.token_handoff_advance_patience
                    ),
                    "token_handoff_retreat_patience": (
                        args.token_handoff_retreat_patience
                    ),
                    "token_handoff_full_probe_fraction": (
                        args.token_handoff_full_probe_fraction
                    ),
                    "token_handoff_unforced_loss_weight": (
                        args.token_handoff_unforced_loss_weight
                    ),
                    "token_handoff_loss_normalization": (
                        args.token_handoff_loss_normalization
                    ),
                    "route_temperature_schedule": (
                        args.route_temperature_schedule
                    ),
                    "route_temperature_start": args.route_temperature_start,
                    "route_temperature_middle": (
                        args.route_temperature_middle
                    ),
                    "route_temperature_end": args.route_temperature_end,
                    "route_temperature_hold_fraction": (
                        args.route_temperature_hold_fraction
                    ),
                    "route_temperature_middle_fraction": (
                        args.route_temperature_middle_fraction
                    ),
                    "route_temperature_step_offset": (
                        args.route_temperature_step_offset
                    ),
                    "route_temperature_total_steps": (
                        args.route_temperature_total_steps or args.steps
                    ),
                    "optimizer_groups": args.optimizer_groups,
                    "optimizer_component_lr_multipliers": (
                        optimizer_component_lr_multipliers
                    ),
                    "optimizer_train_components": (
                        optimizer_train_components
                    ),
                    "optimizer_retain_frozen_state": (
                        args.optimizer_retain_frozen_state
                    ),
                    "optimizer_freeze_route_transport": (
                        args.optimizer_freeze_route_transport
                    ),
                    "token_handoff_state": (
                        token_handoff_controller.state_dict()
                        if token_handoff_controller is not None
                        else None
                    ),
                    "prefix_retention_start": (
                        args.prefix_retention_start
                    ),
                    "prefix_retention_floor": (
                        args.prefix_retention_floor
                    ),
                    "prefix_denoise_warmup_steps": (
                        args.prefix_denoise_warmup_steps
                    ),
                    "prefix_zero_row_fraction": (
                        args.prefix_zero_row_fraction
                    ),
                    "prefix_short_row_fraction": (
                        args.prefix_short_row_fraction
                    ),
                    "prefix_short_max_tokens": (
                        args.prefix_short_max_tokens
                    ),
                    "prefix_span_fraction": args.prefix_span_fraction,
                    "prefix_context_utility_start": (
                        args.prefix_context_utility_start
                    ),
                    "prefix_context_utility_end": (
                        args.prefix_context_utility_end
                    ),
                    "prefix_denoising_state": (
                        prefix_denoising_controller.state_dict()
                        if prefix_denoising_controller is not None
                        else None
                    ),
                    "formation_objective": args.formation_objective,
                    "alignment_warmup_fraction": args.alignment_warmup_fraction,
                    "alignment_max_shift": args.alignment_max_shift,
                    "sequence_quality_penalty": args.sequence_quality_penalty,
                    "sequence_composition_weight": args.sequence_composition_weight,
                    "sequence_length_weight": args.sequence_length_weight,
                    "sequence_eos_weight": args.sequence_eos_weight,
                    "cell_transport_mode": args.cell_transport_mode,
                    "stewardship_keep_mass": args.stewardship_keep_mass,
                    "stewardship_handoff_keep_mass": args.stewardship_handoff_keep_mass,
                    "stewardship_min_active": args.stewardship_min_active,
                    "stewardship_max_active": args.stewardship_max_active,
                    "stewardship_exploration": args.stewardship_exploration,
                    "stewardship_scouts": args.stewardship_scouts,
                    "stewardship_execution": args.stewardship_execution,
                    "execution_preset": args.execution_preset,
                    "stewardship_packed_threshold": args.stewardship_packed_threshold,
                    "packed_stewardship_backend": (
                        args.packed_stewardship_backend
                    ),
                    "stewardship_execution_check_every": (
                        args.stewardship_execution_check_every
                    ),
                    "stewardship_warmup_fraction": args.stewardship_warmup_fraction,
                    "stewardship_temperature_start": args.stewardship_temperature_start,
                    "stewardship_temperature_end": args.stewardship_temperature_end,
                    "evaluation_stewardship_temperature": args.stewardship_temperature_end,
                    "evaluation_stewardship_prune": True,
                    "stewardship_familiar_entropy_penalty": args.stewardship_familiar_entropy_penalty,
                    "stewardship_balance_penalty": args.stewardship_balance_penalty,
                    "smoltalk_train_filter": smoltalk_train_filter_stats.to_dict(),
                    "smoltalk_validation_filter": smoltalk_validation_filter_stats.to_dict(),
                    "lmsys_training_filter": lmsys_training_filter_stats.to_dict(),
                    "lmsys_validation_filter": lmsys_validation_filter_stats.to_dict(),
                    "stream_training_filters": {
                        name: stats.to_dict()
                        for name, stats in stream_filter_stats.items()
                    },
                    "resumed_from": args.resume_checkpoint,
                    "resume_step": start_step,
                    "resume_optimizer": args.resume_optimizer,
                    "best_metrics": checkpoint_best_metrics,
                    "batch_pipeline": {
                        "mode": pipeline_config.mode,
                        "prefetch_batches": (
                            pipeline_config.prefetch_batches
                        ),
                        "pin_memory": pipeline_config.pin_memory,
                        "non_blocking": pipeline_config.non_blocking,
                        "dedicated_cuda_stream": (
                            pipeline_config.dedicated_cuda_stream
                        ),
                        "resume_policy": args.resume_data_state,
                    },
                    "preparation": {
                        "backend": preparation_config.backend,
                        "workers": preparation_config.workers,
                        "chunk_size": preparation_config.chunk_size,
                        "max_pending_chunks": (
                            preparation_config.max_pending_chunks
                        ),
                    },
                    "compute": compute_config.to_dict(),
                    "resolved_float32_runtime": (
                        resolved_runtime.to_dict()
                    ),
                },
                validation=result,
                auxiliary_validation=teacher_result,
                support_validation=(
                    unsupported_result
                    if token_handoff_controller is not None
                    else supported_context_result
                ),
                context_validation=shuffled_context_result,
                training_evaluation=training_result,
                execution_state=(
                    execution_controller.state_dict()
                    if execution_controller is not None
                    else None
                ),
                data_pipeline_state={
                    "fingerprint": data_fingerprint,
                    "state": pipeline_checkpoint,
                    "config": {
                        "batch_size": pipeline_config.batch_size,
                        "mode": pipeline_config.mode,
                        "prefetch_batches": (
                            pipeline_config.prefetch_batches
                        ),
                        "pin_memory": pipeline_config.pin_memory,
                        "non_blocking": pipeline_config.non_blocking,
                        "dedicated_cuda_stream": (
                            pipeline_config.dedicated_cuda_stream
                        ),
                    },
                },
            )
            torch.save(state, checkpoint_dir / "latest.pt")
            print(
                f"checkpoint_saved step={completed} "
                f"path={checkpoint_dir / 'latest.pt'}",
                flush=True,
            )
            if result["selection_loss"] < best_validation_loss:
                best_validation_loss = result["selection_loss"]
                torch.save(state, checkpoint_dir / "best.pt")
            if (
                teacher_result is not None
                and teacher_result["selection_loss"]
                < best_teacher_validation_loss
            ):
                best_teacher_validation_loss = teacher_result[
                    "selection_loss"
                ]
                torch.save(state, checkpoint_dir / "best_teacher.pt")
            if (
                unsupported_result is not None
                and unsupported_result["selection_loss"]
                < best_unsupported_validation_loss
            ):
                best_unsupported_validation_loss = unsupported_result[
                    "selection_loss"
                ]
                torch.save(state, checkpoint_dir / "best_unsupported.pt")
            if (
                unsupported_result is not None
                and unsupported_result["content_loss"]
                < best_free_content_loss
            ):
                best_free_content_loss = unsupported_result["content_loss"]
                torch.save(
                    state,
                    checkpoint_dir / "best_free_content.pt",
                )
            if (
                teacher_result is not None
                and supported_context_result is not None
                and shuffled_context_result is not None
            ):
                context_utility = (
                    supported_context_result["content_accuracy"]
                    - shuffled_context_result["content_accuracy"]
                )
                context_score = max(context_utility, 0.0) * teacher_result[
                    "content_accuracy"
                ]
                if context_score > best_context_score:
                    best_context_score = context_score
                    torch.save(state, checkpoint_dir / "best_context.pt")
            if (
                training_result is not None
                and training_result["selection_loss"] < best_training_loss
            ):
                best_training_loss = training_result["selection_loss"]
                torch.save(state, checkpoint_dir / "best_train.pt")
            last_log_time = time.perf_counter()
            if (
                args.max_runtime_seconds > 0
                and last_log_time - started
                >= args.max_runtime_seconds
            ):
                print(
                    f"runtime_limit_reached seconds="
                    f"{last_log_time - started:.2f} "
                    f"completed_step={completed}",
                    flush=True,
                )
                break

    train_pipeline.close()
    atexit.unregister(train_pipeline.close)
    _ACTIVE_LANGUAGE_PIPELINES.discard(train_pipeline)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    runtime_metrics = {
        "device": str(device),
        "executed_steps": executed_steps,
        "runtime_seconds": elapsed,
        "steps_per_second": executed_steps / max(elapsed, 1e-9),
        "cuda": None,
    }
    if device.type == "cuda":
        runtime_metrics["cuda"] = {
            "device_name": torch.cuda.get_device_name(device),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        }
    (checkpoint_dir / "runtime_metrics.json").write_text(
        json.dumps(runtime_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"runtime_seconds={elapsed:.2f} "
        f"steps_per_second={executed_steps / max(elapsed, 1e-9):.2f} "
        f"peak_cuda_allocated_bytes="
        f"{None if runtime_metrics['cuda'] is None else runtime_metrics['cuda']['peak_allocated_bytes']} "
        f"peak_cuda_reserved_bytes="
        f"{None if runtime_metrics['cuda'] is None else runtime_metrics['cuda']['peak_reserved_bytes']}"
    )


def main() -> None:
    """Run the CLI with exception-safe pipeline teardown."""

    try:
        _run_main()
    finally:
        _close_active_language_pipelines()


if __name__ == "__main__":
    main()
