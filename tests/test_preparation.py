from __future__ import annotations

import threading
import time
import unittest

from gator.data import AssistantExample, DialogueTurn
from gator.preparation import (
    AssistantTokenizationJob,
    ParallelMapConfig,
    ordered_parallel_map,
    tokenize_assistant_example,
    tokenize_assistant_examples,
)
from gator.tokenization import CompactTokenizer


def _variable_square(value: int) -> int:
    """Module-level and therefore safe to import in a spawned worker."""

    time.sleep(0.002 * (5 - value % 5))
    return value * value


def _fail_on_three(value: int) -> int:
    if value == 3:
        raise RuntimeError("worker saw three")
    return value


class _CountingIterable:
    def __init__(self, count: int) -> None:
        self.count = count
        self.items_requested = 0

    def __iter__(self):
        for value in range(self.count):
            self.items_requested += 1
            yield value


class OrderedParallelMapTests(unittest.TestCase):
    def test_serial_and_threads_are_equivalent_and_ordered(self) -> None:
        values = list(range(12))
        serial = list(
            ordered_parallel_map(
                _variable_square,
                values,
                config=ParallelMapConfig(backend="serial", chunk_size=3),
            )
        )
        threaded = list(
            ordered_parallel_map(
                _variable_square,
                values,
                config=ParallelMapConfig(
                    backend="threads",
                    workers=4,
                    chunk_size=1,
                    max_pending_chunks=8,
                ),
            )
        )

        self.assertEqual(serial, [value * value for value in values])
        self.assertEqual(threaded, serial)

    def test_submission_is_bounded_by_chunk_queue(self) -> None:
        source = _CountingIterable(100)
        first_started = threading.Event()
        release_first = threading.Event()

        def blocked_first(value: int) -> int:
            if value == 0:
                first_started.set()
                release_first.wait(timeout=5.0)
            return value

        mapped = ordered_parallel_map(
            blocked_first,
            source,
            config=ParallelMapConfig(
                backend="threads",
                workers=3,
                chunk_size=2,
                max_pending_chunks=3,
            ),
        )
        result: list[int] = []

        def consume_one() -> None:
            result.append(next(mapped))

        consumer = threading.Thread(target=consume_one)
        consumer.start()
        try:
            self.assertTrue(first_started.wait(timeout=2.0))
            deadline = time.monotonic() + 2.0
            while (
                source.items_requested < 6
                and time.monotonic() < deadline
            ):
                time.sleep(0.001)
            # Three chunks of two are submitted before the ordered first result.
            self.assertEqual(source.items_requested, 6)
        finally:
            release_first.set()
            consumer.join(timeout=2.0)
            mapped.close()
        self.assertFalse(consumer.is_alive())
        self.assertEqual(result, [0])

    def test_worker_exception_is_propagated(self) -> None:
        for backend in ("serial", "threads"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(RuntimeError, "worker saw three"):
                    list(
                        ordered_parallel_map(
                            _fail_on_three,
                            range(8),
                            config=ParallelMapConfig(
                                backend=backend,
                                workers=2,
                                chunk_size=2,
                                max_pending_chunks=2,
                            ),
                        )
                    )

    def test_spawn_process_tokenization_smoke_preserves_order(self) -> None:
        tokenizer = CompactTokenizer(["alpha", "beta", " ", "!"])
        examples = [
            AssistantExample(
                conversation_id=index,
                turn_index=1,
                context=(DialogueTurn("user", text),),
                response=f"{text}!",
                source="spawn-smoke",
            )
            for index, text in enumerate(("alpha", "beta", "alpha beta"))
        ]
        jobs = [
            AssistantTokenizationJob(example=example, tokenizer=tokenizer)
            for example in examples
        ]

        serial = list(
            ordered_parallel_map(
                tokenize_assistant_example,
                jobs,
                config=ParallelMapConfig(backend="serial"),
            )
        )
        spawned = list(
            ordered_parallel_map(
                tokenize_assistant_example,
                jobs,
                config=ParallelMapConfig(
                    backend="processes",
                    workers=2,
                    chunk_size=2,
                    max_pending_chunks=2,
                ),
            )
        )

        self.assertEqual(spawned, serial)
        self.assertEqual(
            [result.conversation_id for result in spawned],
            [0, 1, 2],
        )

        wrapped = list(
            tokenize_assistant_examples(
                examples,
                tokenizer,
                config=ParallelMapConfig(
                    backend="threads",
                    workers=2,
                    chunk_size=2,
                ),
            )
        )
        self.assertEqual(wrapped, serial)

    def test_config_rejects_unbounded_or_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            ParallelMapConfig(backend="unknown")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ParallelMapConfig(workers=0)
        with self.assertRaises(ValueError):
            ParallelMapConfig(chunk_size=0)
        with self.assertRaises(ValueError):
            ParallelMapConfig(max_pending_chunks=0)


if __name__ == "__main__":
    unittest.main()
