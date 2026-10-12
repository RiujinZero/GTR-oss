from __future__ import annotations

import torch

from gator.tasks import LanguageBatch
from gator_language_train import (
    parallel_next_chunk_batch,
    serial_next_token_batch,
)


def test_serial_next_token_appends_prefix_and_predicts_following_token() -> None:
    batch = LanguageBatch(
        context_ids=torch.tensor(
            [
                [0, 0, 10, 11, 12],
                [0, 20, 21, 22, 23],
            ]
        ),
        context_mask=torch.tensor(
            [
                [False, False, True, True, True],
                [False, True, True, True, True],
            ]
        ),
        target_ids=torch.tensor(
            [
                [30, 31, 2, 0],
                [40, 41, 42, 2],
            ]
        ),
        target_mask=torch.tensor(
            [
                [True, True, True, False],
                [True, True, True, True],
            ]
        ),
        target_complete=torch.tensor([True, True]),
    )

    serial = serial_next_token_batch(
        batch,
        eos_token_id=2,
        prefix_lengths=torch.tensor([1, 3]),
    )

    assert serial.context_ids.tolist() == [
        [0, 10, 11, 12, 30],
        [22, 23, 40, 41, 42],
    ]
    assert serial.context_mask.tolist() == [
        [False, True, True, True, True],
        [True, True, True, True, True],
    ]
    assert serial.target_ids.tolist() == [[31], [2]]
    assert serial.target_mask.tolist() == [[True], [True]]
    assert serial.target_complete.tolist() == [False, True]


def test_serial_next_token_rejects_prefix_at_or_after_target_end() -> None:
    batch = LanguageBatch(
        context_ids=torch.tensor([[10, 11]]),
        context_mask=torch.tensor([[True, True]]),
        target_ids=torch.tensor([[30, 2]]),
        target_mask=torch.tensor([[True, True]]),
        target_complete=torch.tensor([True]),
    )

    try:
        serial_next_token_batch(
            batch,
            eos_token_id=2,
            prefix_lengths=torch.tensor([2]),
        )
    except ValueError as error:
        assert "existing next token" in str(error)
    else:
        raise AssertionError("invalid prefix length was accepted")


def test_parallel_next_chunk_appends_only_completed_chunks() -> None:
    batch = LanguageBatch(
        context_ids=torch.tensor(
            [
                [0, 0, 10, 11, 12],
                [0, 20, 21, 22, 23],
            ]
        ),
        context_mask=torch.tensor(
            [
                [False, False, True, True, True],
                [False, True, True, True, True],
            ]
        ),
        target_ids=torch.tensor(
            [
                [30, 31, 32, 33, 34, 2, 0, 0],
                [40, 41, 42, 43, 44, 45, 46, 2],
            ]
        ),
        target_mask=torch.tensor(
            [
                [True, True, True, True, True, True, False, False],
                [True, True, True, True, True, True, True, True],
            ]
        ),
        target_complete=torch.tensor([True, True]),
    )

    chunk = parallel_next_chunk_batch(
        batch,
        eos_token_id=2,
        chunk_size=3,
        chunk_indices=torch.tensor([1, 2]),
    )

    # Row zero predicts tokens 3:6 after seeing exactly the first chunk.
    # Row one predicts the final partial chunk after seeing two chunks.
    assert chunk.context_ids.tolist() == [
        [11, 12, 30, 31, 32],
        [41, 42, 43, 44, 45],
    ]
    assert chunk.context_mask.tolist() == [
        [True, True, True, True, True],
        [True, True, True, True, True],
    ]
    assert chunk.target_ids.tolist() == [
        [33, 34, 2],
        [46, 2, 0],
    ]
    assert chunk.target_mask.tolist() == [
        [True, True, True],
        [True, True, False],
    ]
    assert chunk.target_complete.tolist() == [True, True]


def test_parallel_next_chunk_rejects_chunk_after_target_end() -> None:
    batch = LanguageBatch(
        context_ids=torch.tensor([[10, 11]]),
        context_mask=torch.tensor([[True, True]]),
        target_ids=torch.tensor([[30, 31, 2, 0]]),
        target_mask=torch.tensor([[True, True, True, False]]),
        target_complete=torch.tensor([True]),
    )

    try:
        parallel_next_chunk_batch(
            batch,
            eos_token_id=2,
            chunk_size=2,
            chunk_indices=torch.tensor([2]),
        )
    except ValueError as error:
        assert "existing target chunk" in str(error)
    else:
        raise AssertionError("invalid chunk index was accepted")
