import unittest

import torch

from gator_sequence_program_benchmark import (
    active_operations_for_step,
    apply_operation,
    execute_program,
    make_program_batch,
    sample_replay_depths,
)


class SequenceProgramBenchmarkTests(unittest.TestCase):
    def test_each_operation(self) -> None:
        values = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])

        expected = (
            [0, 1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1, 0],
            [1, 2, 3, 4, 5, 6, 0],
            [6, 0, 1, 2, 3, 4, 5],
            [1, 0, 3, 2, 5, 4, 6],
            [7, 6, 5, 4, 3, 2, 1],
            [0, 1, 3, 0, 4, 1, 7],
            [0, 1, 2, 3, 4, 5, 6],
        )
        for operation, target in enumerate(expected):
            with self.subTest(operation=operation):
                actual = apply_operation(
                    values,
                    torch.tensor([operation]),
                )
                self.assertEqual(actual.tolist(), [target])

    def test_program_composes_in_instruction_order(self) -> None:
        values = torch.tensor([[0, 1, 2, 3]])
        operations = torch.tensor([[1, 5]])

        actual = execute_program(values, operations)

        self.assertEqual(actual.tolist(), [[4, 5, 6, 7]])

    def test_batch_uses_disjoint_value_and_instruction_tokens(self) -> None:
        batch = make_program_batch(
            16,
            7,
            3,
            2,
            "cpu",
            generator=torch.Generator().manual_seed(7),
        )

        self.assertEqual(batch.observed.shape, (16, 10))
        self.assertTrue(batch.observed[:, :7].lt(8).all())
        self.assertTrue(batch.observed[:, 7:].ge(8).all())
        self.assertTrue(batch.operations[:, :2].ge(1).all())
        self.assertTrue(batch.operations[:, 2].eq(0).all())
        self.assertTrue(batch.targets.lt(8).all())

    def test_curriculum_reaches_three_operations(self) -> None:
        stages = [
            active_operations_for_step(step, 10_000, 3, 0.60)
            for step in (0, 2_000, 4_000, 6_000, 9_999)
        ]

        self.assertEqual(stages, [1, 2, 3, 3, 3])

    def test_row_level_replay_interleaves_earlier_depths(self) -> None:
        depths = sample_replay_depths(
            10_000,
            3,
            replay_fraction=0.40,
            device="cpu",
            generator=torch.Generator().manual_seed(19),
        )

        fractions = [
            depths.eq(depth).float().mean().item()
            for depth in (1, 2, 3)
        ]
        self.assertAlmostEqual(fractions[0], 0.20, delta=0.02)
        self.assertAlmostEqual(fractions[1], 0.20, delta=0.02)
        self.assertAlmostEqual(fractions[2], 0.60, delta=0.02)

    def test_batch_accepts_mixed_row_depths(self) -> None:
        depths = torch.tensor([1, 2, 3, 1])
        batch = make_program_batch(
            4,
            7,
            3,
            depths,
            "cpu",
            generator=torch.Generator().manual_seed(23),
        )

        self.assertEqual(
            batch.operations.ne(0).sum(1).tolist(),
            depths.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
