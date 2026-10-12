from __future__ import annotations

import threading
import time
import unittest

import torch

from gator.pipeline import (
    BatchPipelineConfig,
    LanguageBatchPipeline,
)
from gator.tasks.language import LanguageBatch


def _batch(value: int, batch_size: int) -> LanguageBatch:
    shape = (batch_size, 2)
    ids = torch.full(shape, value, dtype=torch.long)
    mask = torch.ones(shape, dtype=torch.bool)
    return LanguageBatch(
        context_ids=ids,
        context_mask=mask,
        target_ids=ids + 100,
        target_mask=mask,
        target_complete=torch.ones(batch_size, dtype=torch.bool),
    )


def _value(batch: LanguageBatch) -> int:
    return int(batch.context_ids[0, 0])


class CounterProvider:
    exact_state_supported = True

    def __init__(self, start: int = 0) -> None:
        self.counter = start
        self.calls = 0
        self.active = 0
        self.maximum_active = 0
        self._lock = threading.Lock()

    def sample_cpu(self, batch_size: int) -> LanguageBatch:
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            value = self.counter
            self.counter += 1
            self.calls += 1
        try:
            return _batch(value, batch_size)
        finally:
            with self._lock:
                self.active -= 1

    def state_dict(self) -> dict[str, int]:
        return {"counter": self.counter}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.counter = int(state["counter"])


class FailingProvider(CounterProvider):
    def __init__(self, successful_batches: int) -> None:
        super().__init__()
        self.successful_batches = successful_batches

    def sample_cpu(self, batch_size: int) -> LanguageBatch:
        if self.counter >= self.successful_batches:
            raise ValueError("synthetic provider failure")
        return super().sample_cpu(batch_size)


class InexactPayloadProvider(CounterProvider):
    def state_dict(self) -> dict[str, int | bool]:
        return {**super().state_dict(), "exact": False}


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        time.sleep(0.005)


class BatchPipelineConfigTests(unittest.TestCase):
    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            BatchPipelineConfig(batch_size=0)
        with self.assertRaisesRegex(ValueError, "prefetch_batches"):
            BatchPipelineConfig(batch_size=1, prefetch_batches=0)
        with self.assertRaisesRegex(ValueError, "mode"):
            BatchPipelineConfig(batch_size=1, mode="unknown")  # type: ignore[arg-type]


class LanguageBatchPipelineTests(unittest.TestCase):
    def test_serial_mode_samples_and_transfers_in_order(self) -> None:
        provider = CounterProvider(start=7)
        pipeline = LanguageBatchPipeline(
            provider,
            BatchPipelineConfig(
                batch_size=3,
                device="cpu",
                mode="serial",
            ),
        )
        try:
            self.assertEqual([_value(next(pipeline)) for _ in range(3)], [7, 8, 9])
            stats = pipeline.stats
            self.assertEqual(stats.sampled_batches, 3)
            self.assertEqual(stats.consumed_batches, 3)
            self.assertEqual(stats.device_transfers, 3)
            self.assertEqual(stats.queue_depth, 0)
            self.assertFalse(stats.worker_alive)
        finally:
            pipeline.close()

    def test_prefetch_is_bounded_and_uses_one_producer(self) -> None:
        provider = CounterProvider()
        pipeline = LanguageBatchPipeline(
            provider,
            BatchPipelineConfig(
                batch_size=2,
                mode="prefetch",
                prefetch_batches=2,
            ),
        )
        try:
            _wait_until(lambda: pipeline.stats.queue_depth == 2)
            # Give a buggy unbounded producer enough time to overrun the cap.
            time.sleep(0.02)
            stats = pipeline.stats
            self.assertEqual(stats.queue_depth, 2)
            self.assertEqual(stats.queue_high_watermark, 2)
            self.assertEqual(provider.maximum_active, 1)
            self.assertEqual([_value(next(pipeline)) for _ in range(4)], [0, 1, 2, 3])
        finally:
            pipeline.close()

    def test_checkpoint_restores_queue_and_post_queue_provider_state(self) -> None:
        first_provider = CounterProvider(start=10)
        config = BatchPipelineConfig(
            batch_size=2,
            mode="prefetch",
            prefetch_batches=3,
        )
        first = LanguageBatchPipeline(first_provider, config)
        try:
            _wait_until(lambda: first.stats.queue_depth == 3)
            self.assertEqual(_value(next(first)), 10)
            _wait_until(lambda: first.stats.queue_depth == 3)

            state = first.state_dict()
            queued_values = [
                int(item["context_ids"][0, 0])
                for item in state["queued_cpu_batches"]
            ]
            self.assertEqual(len(queued_values), 3)
            self.assertEqual(
                state["provider_state"]["counter"],
                queued_values[-1] + 1,
            )
            self.assertTrue(state["exact_provider_state"])
            self.assertTrue(first.exact_resume_supported)
            expected = [_value(next(first)) for _ in range(7)]
        finally:
            first.close()

        # Construction is intentionally allowed to prefetch from the wrong
        # provider state.  load_state_dict must atomically discard those rows
        # and reset both the queue and provider.
        restored_provider = CounterProvider(start=900)
        restored = LanguageBatchPipeline(restored_provider, config)
        try:
            restored.load_state_dict(state)
            actual = [_value(next(restored)) for _ in range(7)]
            self.assertEqual(actual, expected)
            self.assertGreaterEqual(restored.stats.restored_batches, 3)
        finally:
            restored.close()

    def test_checkpoint_batches_are_independent_cpu_clones(self) -> None:
        provider = CounterProvider()
        pipeline = LanguageBatchPipeline(
            provider,
            BatchPipelineConfig(
                batch_size=1,
                mode="prefetch",
                prefetch_batches=1,
            ),
        )
        try:
            _wait_until(lambda: pipeline.stats.queue_depth == 1)
            state = pipeline.state_dict()
            state["queued_cpu_batches"][0]["context_ids"].fill_(999)
            self.assertEqual(_value(next(pipeline)), 0)
        finally:
            pipeline.close()

    def test_producer_exception_is_raised_after_earlier_batches(self) -> None:
        pipeline = LanguageBatchPipeline(
            FailingProvider(successful_batches=2),
            BatchPipelineConfig(
                batch_size=1,
                mode="prefetch",
                prefetch_batches=3,
            ),
        )
        try:
            _wait_until(lambda: pipeline.stats.producer_failures == 1)
            self.assertEqual(_value(next(pipeline)), 0)
            self.assertEqual(_value(next(pipeline)), 1)
            with self.assertRaisesRegex(ValueError, "synthetic"):
                next(pipeline)
            self.assertFalse(pipeline.stats.worker_alive)
        finally:
            pipeline.close()

    def test_context_manager_closes_and_close_is_idempotent(self) -> None:
        provider = CounterProvider()
        with LanguageBatchPipeline(
            provider,
            BatchPipelineConfig(
                batch_size=1,
                mode="prefetch",
                prefetch_batches=1,
            ),
        ) as pipeline:
            next(pipeline)
            self.assertTrue(pipeline.stats.worker_alive)

        self.assertTrue(pipeline.stats.closed)
        self.assertFalse(pipeline.stats.worker_alive)
        pipeline.close()
        with self.assertRaises(StopIteration):
            next(pipeline)

    def test_serial_checkpoint_is_exact(self) -> None:
        provider = CounterProvider(start=4)
        config = BatchPipelineConfig(batch_size=1, mode="serial")
        pipeline = LanguageBatchPipeline(provider, config)
        try:
            self.assertEqual(_value(next(pipeline)), 4)
            state = pipeline.state_dict()
            expected = [_value(next(pipeline)) for _ in range(4)]
        finally:
            pipeline.close()

        restored = LanguageBatchPipeline(CounterProvider(start=100), config)
        try:
            restored.load_state_dict(state)
            self.assertEqual(
                [_value(next(restored)) for _ in range(4)],
                expected,
            )
        finally:
            restored.close()

    def test_state_methods_alone_do_not_claim_exact_resume(self) -> None:
        provider = CounterProvider()
        provider.exact_state_supported = False
        pipeline = LanguageBatchPipeline(
            provider,
            BatchPipelineConfig(batch_size=1, mode="serial"),
        )
        try:
            self.assertFalse(pipeline.exact_resume_supported)
            self.assertFalse(pipeline.state_dict()["exact_provider_state"])
        finally:
            pipeline.close()

    def test_default_loader_rejects_inexact_state(self) -> None:
        source = LanguageBatchPipeline(
            InexactPayloadProvider(),
            BatchPipelineConfig(batch_size=1, mode="serial"),
        )
        try:
            state = source.state_dict()
            self.assertFalse(state["exact_provider_state"])
        finally:
            source.close()

        restored = LanguageBatchPipeline(
            InexactPayloadProvider(start=50),
            BatchPipelineConfig(batch_size=1, mode="serial"),
        )
        try:
            with self.assertRaisesRegex(ValueError, "not exactly"):
                restored.load_state_dict(state)
            restored.load_state_dict(state, require_exact=False)
            self.assertEqual(_value(next(restored)), 0)
        finally:
            restored.close()

    def test_current_provider_must_also_support_exact_restore(self) -> None:
        source = LanguageBatchPipeline(
            CounterProvider(),
            BatchPipelineConfig(batch_size=1, mode="serial"),
        )
        try:
            state = source.state_dict()
        finally:
            source.close()

        provider = CounterProvider(start=50)
        provider.exact_state_supported = False
        restored = LanguageBatchPipeline(
            provider,
            BatchPipelineConfig(batch_size=1, mode="serial"),
        )
        try:
            with self.assertRaisesRegex(ValueError, "current provider"):
                restored.load_state_dict(state)
        finally:
            restored.close()


if __name__ == "__main__":
    unittest.main()
