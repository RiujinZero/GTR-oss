"""Benchmark Gator's forward-free language input pipeline.

This is deliberately a data-path microbenchmark.  It measures deterministic
synthetic batch sampling/collation and, when requested, transfer to ``device``.
It does not run a model forward, backward, optimizer step, tokenizer, or file
reader.  Consequently, the results are not Gator training throughput and are
not a model/GPU performance claim.

The compared paths are:

* ``legacy_collate``: select explicit ``EncodedAssistantExample`` objects and
  rebuild every fixed-width tensor row with ``LanguageBatcher.collate_cpu``;
* ``vectorized_batcher``: gather pre-collated rows with
  ``LanguageBatcher.sample_cpu``;
* ``packed_source``: gather compact int32 rows with
  ``PackedLanguageSource.sample_cpu`` and construct masks vectorially;
* ``serial_pipeline``: consume a packed source through
  ``LanguageBatchPipeline`` in serial mode; and
* one ``prefetch_depth_N`` result per requested bounded prefetch depth.

Representative CPU run:

    python benchmark_language_pipeline.py

Small smoke:

    python benchmark_language_pipeline.py --examples 64 --batch-size 4 \
        --iterations 3 --warmup 1 --prefetch-depths 1,2 --device cpu
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Callable, Mapping, Sequence

import torch

from gator.pipeline import (
    BatchPipelineConfig,
    BatchPipelineStats,
    LanguageBatchPipeline,
)
from gator.tasks import (
    EncodedAssistantExample,
    LanguageBatch,
    LanguageBatchConfig,
    LanguageBatcher,
    PackedLanguageSource,
)
from gator.tokenization import CompactTokenizer


MASK_64 = (1 << 64) - 1


@dataclass(frozen=True)
class BenchmarkResult:
    """One measured consumer path."""

    name: str
    batches_per_second: float
    tokens_per_second: float
    p50_consumer_ms: float
    p95_consumer_ms: float
    measured_seconds: float
    checksum: str
    pipeline_stats: Mapping[str, object] | None = None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def _parse_depths(value: str) -> tuple[int, ...]:
    try:
        depths = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        ) from error
    if not depths or any(depth <= 0 for depth in depths):
        raise argparse.ArgumentTypeError(
            "every prefetch depth must be positive"
        )
    # Preserve the requested display order while avoiding duplicate work.
    return tuple(dict.fromkeys(depths))


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    return device


def _synthetic_examples(
    count: int,
    config: LanguageBatchConfig,
) -> tuple[EncodedAssistantExample, ...]:
    """Build deterministic, fully occupied fixed-window token rows."""

    token_floor = CompactTokenizer.LEARNED_OFFSET
    token_span = 2048 - token_floor
    if token_span <= 0:
        raise RuntimeError("synthetic vocabulary does not contain text tokens")

    examples = []
    for row in range(count):
        context = tuple(
            token_floor + ((row * 131 + column * 17 + 11) % token_span)
            for column in range(config.context_tokens)
        )
        response = [
            token_floor + ((row * 193 + column * 29 + 23) % token_span)
            for column in range(config.response_tokens)
        ]
        response[-1] = CompactTokenizer.EOS_ID
        examples.append(
            EncodedAssistantExample(
                conversation_id=row,
                turn_index=row % 7,
                context_ids=context,
                response_ids=tuple(response),
                source="synthetic-fixed-window",
            )
        )
    return tuple(examples)


def _packed_source(
    packed_corpus: LanguageBatch,
    *,
    seed: int,
) -> PackedLanguageSource:
    return PackedLanguageSource(
        packed_corpus.context_ids,
        packed_corpus.target_ids,
        pad_token_id=CompactTokenizer.PAD_ID,
        eos_token_id=CompactTokenizer.EOS_ID,
        seed=seed,
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Linearly interpolated percentile for a nonempty sorted sample."""

    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class _BatchChecksum:
    """Order-sensitive deterministic digest without copying whole GPU batches."""

    def __init__(
        self,
        *,
        batch_size: int,
        context_tokens: int,
        response_tokens: int,
        device: torch.device,
    ) -> None:
        self.value = 0xCBF29CE484222325
        self.batch_index = 0
        self.row_weights = torch.arange(
            1,
            batch_size + 1,
            dtype=torch.long,
            device=device,
        )
        self.context_weights = torch.arange(
            1,
            context_tokens + 1,
            dtype=torch.long,
            device=device,
        )
        self.target_weights = torch.arange(
            response_tokens + 1,
            1,
            -1,
            dtype=torch.long,
            device=device,
        )

    def update(self, batch: LanguageBatch) -> None:
        context = (
            batch.context_ids * self.context_weights[None, :]
        ).sum(dim=1)
        target = (
            batch.target_ids * self.target_weights[None, :]
        ).sum(dim=1)
        masks = (
            batch.context_mask.sum(dim=1, dtype=torch.long) * 257
            + batch.target_mask.sum(dim=1, dtype=torch.long) * 263
            + batch.target_complete.to(dtype=torch.long) * 269
        )
        scalar = int(
            ((context * 271 + target * 277 + masks) * self.row_weights)
            .sum()
            .item()
        )
        mixed = (
            scalar
            + self.batch_index * 0x9E3779B97F4A7C15
        ) & MASK_64
        self.value ^= mixed
        self.value = (self.value * 0x100000001B3) & MASK_64
        self.batch_index += 1

    @property
    def hexadecimal(self) -> str:
        return f"{self.value:016x}"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_consumer(
    *,
    name: str,
    consume: Callable[[], LanguageBatch],
    warmup: int,
    iterations: int,
    logical_tokens_per_batch: int,
    batch_size: int,
    context_tokens: int,
    response_tokens: int,
    device: torch.device,
    stats: Callable[[], BatchPipelineStats] | None = None,
) -> BenchmarkResult:
    for _ in range(warmup):
        consume()
        _synchronize(device)

    checksum = _BatchChecksum(
        batch_size=batch_size,
        context_tokens=context_tokens,
        response_tokens=response_tokens,
        device=device,
    )
    durations = []
    measured_batches = []
    for _ in range(iterations):
        _synchronize(device)
        started = time.perf_counter()
        batch = consume()
        _synchronize(device)
        durations.append(time.perf_counter() - started)
        # Retaining the lightweight batch objects lets validation happen only
        # after every latency sample.  In particular, checksum computation
        # cannot become accidental "consumer work" hidden by the prefetch
        # producer and inflate its measured throughput.
        measured_batches.append(batch)

    for batch in measured_batches:
        checksum.update(batch)

    measured_seconds = sum(durations)
    batches_per_second = iterations / measured_seconds
    pipeline_stats = asdict(stats()) if stats is not None else None
    return BenchmarkResult(
        name=name,
        batches_per_second=batches_per_second,
        tokens_per_second=(
            batches_per_second * logical_tokens_per_batch
        ),
        p50_consumer_ms=_percentile(durations, 0.50) * 1_000,
        p95_consumer_ms=_percentile(durations, 0.95) * 1_000,
        measured_seconds=measured_seconds,
        checksum=checksum.hexadecimal,
        pipeline_stats=pipeline_stats,
    )


def _print_results(results: Sequence[BenchmarkResult]) -> None:
    legacy_rate = results[0].batches_per_second
    header = (
        f"{'method':<24} {'batches/s':>12} {'tokens/s':>14} "
        f"{'p50 ms':>10} {'p95 ms':>10} {'speedup':>9} {'checksum':>16}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<24} "
            f"{result.batches_per_second:>12.2f} "
            f"{result.tokens_per_second:>14.0f} "
            f"{result.p50_consumer_ms:>10.4f} "
            f"{result.p95_consumer_ms:>10.4f} "
            f"{result.batches_per_second / legacy_rate:>8.2f}x "
            f"{result.checksum:>16}"
        )

    print()
    for result in results:
        if result.pipeline_stats is None:
            continue
        stats = result.pipeline_stats
        print(
            f"pipeline_stats method={result.name} "
            f"sampled={stats['sampled_batches']} "
            f"consumed={stats['consumed_batches']} "
            f"queue={stats['queue_depth']}/{stats['queue_capacity']} "
            f"high_watermark={stats['queue_high_watermark']} "
            f"producer_waits={stats['producer_waits']} "
            f"sample_s={stats['sample_seconds']:.6f} "
            f"transfer_s={stats['transfer_seconds']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark deterministic Gator language batch preparation and "
            "optional device transfer without running a model."
        )
    )
    parser.add_argument("--examples", type=_positive_int, default=10_000)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--iterations", type=_positive_int, default=1_000)
    parser.add_argument("--warmup", type=_nonnegative_int, default=50)
    parser.add_argument("--context-tokens", type=_positive_int, default=64)
    parser.add_argument("--response-tokens", type=_positive_int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="'cpu', 'cuda', a concrete torch device, or 'auto'",
    )
    parser.add_argument(
        "--prefetch-depths",
        type=_parse_depths,
        default=(1, 2, 4),
        help="comma-separated bounded CPU prefetch queue depths",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pin CPU batches before transfer (default: enabled for CUDA)",
    )
    parser.add_argument(
        "--non-blocking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request non-blocking tensor transfers",
    )
    parser.add_argument(
        "--dedicated-cuda-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="let pipeline transfers use a dedicated CUDA stream",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    pin_memory = (
        device.type == "cuda"
        if args.pin_memory is None
        else args.pin_memory
    )
    config = LanguageBatchConfig(
        context_tokens=args.context_tokens,
        response_tokens=args.response_tokens,
    )
    examples = _synthetic_examples(args.examples, config)

    # Corpus construction and one-time packing are setup costs and therefore
    # intentionally happen before any measured interval.
    setup_batcher = LanguageBatcher(examples, config, seed=args.seed)
    packed_corpus = setup_batcher.batch_rows_cpu(
        torch.arange(args.examples, dtype=torch.long)
    )
    logical_tokens_per_batch = args.batch_size * (
        args.context_tokens + args.response_tokens
    )

    def transfer(batch: LanguageBatch) -> LanguageBatch:
        if pin_memory:
            batch = batch.pin_memory()
        return batch.to(
            device,
            non_blocking=args.non_blocking,
        )

    results: list[BenchmarkResult] = []

    legacy_collator = LanguageBatcher(examples, config, seed=args.seed + 1)
    legacy_random = random.Random(args.seed)

    def consume_legacy() -> LanguageBatch:
        indices = [
            legacy_random.randrange(len(examples))
            for _ in range(args.batch_size)
        ]
        selected = [examples[index] for index in indices]
        return transfer(legacy_collator.collate_cpu(selected))

    results.append(
        _benchmark_consumer(
            name="legacy_collate",
            consume=consume_legacy,
            warmup=args.warmup,
            iterations=args.iterations,
            logical_tokens_per_batch=logical_tokens_per_batch,
            batch_size=args.batch_size,
            context_tokens=args.context_tokens,
            response_tokens=args.response_tokens,
            device=device,
        )
    )

    vectorized_batcher = LanguageBatcher(
        examples, config, seed=args.seed
    )
    results.append(
        _benchmark_consumer(
            name="vectorized_batcher",
            consume=lambda: transfer(
                vectorized_batcher.sample_cpu(args.batch_size)
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            logical_tokens_per_batch=logical_tokens_per_batch,
            batch_size=args.batch_size,
            context_tokens=args.context_tokens,
            response_tokens=args.response_tokens,
            device=device,
        )
    )

    packed_source = _packed_source(packed_corpus, seed=args.seed)
    results.append(
        _benchmark_consumer(
            name="packed_source",
            consume=lambda: transfer(
                packed_source.sample_cpu(args.batch_size)
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            logical_tokens_per_batch=logical_tokens_per_batch,
            batch_size=args.batch_size,
            context_tokens=args.context_tokens,
            response_tokens=args.response_tokens,
            device=device,
        )
    )

    serial_pipeline = LanguageBatchPipeline(
        _packed_source(packed_corpus, seed=args.seed),
        BatchPipelineConfig(
            batch_size=args.batch_size,
            device=device,
            mode="serial",
            prefetch_batches=1,
            pin_memory=pin_memory,
            non_blocking=args.non_blocking,
            dedicated_cuda_stream=args.dedicated_cuda_stream,
        ),
    )
    try:
        results.append(
            _benchmark_consumer(
                name="serial_pipeline",
                consume=serial_pipeline.next_batch,
                warmup=args.warmup,
                iterations=args.iterations,
                logical_tokens_per_batch=logical_tokens_per_batch,
                batch_size=args.batch_size,
                context_tokens=args.context_tokens,
                response_tokens=args.response_tokens,
                device=device,
                stats=lambda: serial_pipeline.stats,
            )
        )
    finally:
        serial_pipeline.close()

    for depth in args.prefetch_depths:
        pipeline = LanguageBatchPipeline(
            _packed_source(packed_corpus, seed=args.seed),
            BatchPipelineConfig(
                batch_size=args.batch_size,
                device=device,
                mode="prefetch",
                prefetch_batches=depth,
                pin_memory=pin_memory,
                non_blocking=args.non_blocking,
                dedicated_cuda_stream=args.dedicated_cuda_stream,
            ),
        )
        try:
            results.append(
                _benchmark_consumer(
                    name=f"prefetch_depth_{depth}",
                    consume=pipeline.next_batch,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    logical_tokens_per_batch=logical_tokens_per_batch,
                    batch_size=args.batch_size,
                    context_tokens=args.context_tokens,
                    response_tokens=args.response_tokens,
                    device=device,
                    stats=lambda pipeline=pipeline: pipeline.stats,
                )
            )
        finally:
            pipeline.close()

    print(
        "scope=CPU fixed-window batch preparation"
        + (
            f" plus transfer to {device}"
            if device.type != "cpu"
            else ""
        )
        + "; tokenizer/file I/O/model forward/backward/optimizer excluded"
    )
    print(
        "claim_boundary=this is a forward-free input-pipeline microbenchmark, "
        "not Gator training throughput or a model/GPU performance claim"
    )
    print(
        f"examples={args.examples} batch={args.batch_size} "
        f"context={args.context_tokens} response={args.response_tokens} "
        f"iterations={args.iterations} warmup={args.warmup} "
        f"device={device} pin_memory={pin_memory} "
        f"prefetch_depths={','.join(map(str, args.prefetch_depths))}"
    )
    print()
    _print_results(results)
    checksums = {result.checksum for result in results}
    print()
    if len(checksums) != 1:
        print("correctness=FAIL deterministic checksums differ")
        raise SystemExit(1)
    print(
        f"correctness=PASS deterministic_checksum={results[0].checksum} "
        "all_methods_match=True"
    )


if __name__ == "__main__":
    main()
