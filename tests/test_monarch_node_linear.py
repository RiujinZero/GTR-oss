"""Correctness checks for gator.models.dense_kernels.monarch_node_linear
and fold_monarch_node_parameters -- the Monarch-matrix factorization
replacing GatorLanguageModel's dense per-(cell,node) node_weight (see
Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md, "Two architecture
directions" / gator_monarch branch).

The reference used to check monarch_node_linear is a naive Python-loop
implementation of the same 2-factor Monarch construction, written
independently of monarch_node_linear's batched einsum internals (same
math, deliberately different code path) -- this catches einsum
index-ordering bugs that a self-referential check couldn't.
"""

import unittest

import torch

from gator.models.dense_kernels import (
    fold_monarch_node_parameters,
    monarch_node_linear,
)


def _reference_monarch_forward(
    x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, p: int, q: int
) -> torch.Tensor:
    """Naive per-block loop reference for a SINGLE (cell, node) instance.

    x: [D]. left: [q,p,p]. right: [p,q,q]. Returns [D].
    """
    X = x.reshape(p, q)
    Y = torch.zeros(p, q, dtype=x.dtype)
    for j in range(q):
        Y[:, j] = left[j] @ X[:, j]
    Yt = Y.T  # [q,p]
    Z = torch.zeros(q, p, dtype=x.dtype)
    for m in range(p):
        Z[:, m] = right[m] @ Yt[:, m]
    return Z.T.reshape(p * q)


class MonarchNodeLinearTests(unittest.TestCase):
    P, Q = 2, 3
    D = P * Q  # 6

    def _random_factors(self, seed=0, cells=1, nodes=1):
        torch.manual_seed(seed)
        left = torch.randn(cells, nodes, self.Q, self.P, self.P, dtype=torch.float64)
        right = torch.randn(cells, nodes, self.P, self.Q, self.Q, dtype=torch.float64)
        return left, right

    def test_matches_naive_loop_reference_single_instance(self):
        left, right = self._random_factors()
        x = torch.randn(self.D, dtype=torch.float64)

        reference = _reference_monarch_forward(x, left[0, 0], right[0, 0], self.P, self.Q)

        base = x.view(1, 1, 1, self.D)  # [B=1,C=1,T=1,D]
        actual = monarch_node_linear(base, left, right)[0, 0, 0, 0]

        self.assertTrue(torch.allclose(actual, reference, rtol=1e-10, atol=1e-10))

    def test_matches_naive_loop_reference_batched_cells_nodes(self):
        cells, nodes = 3, 2
        left, right = self._random_factors(seed=1, cells=cells, nodes=nodes)
        batch, tokens = 2, 4
        torch.manual_seed(2)
        base = torch.randn(batch, cells, tokens, self.D, dtype=torch.float64)

        actual = monarch_node_linear(base, left, right)
        self.assertEqual(actual.shape, (batch, cells, tokens, nodes, self.D))

        for c in range(cells):
            for n in range(nodes):
                for b in range(batch):
                    for t in range(tokens):
                        expected = _reference_monarch_forward(
                            base[b, c, t], left[c, n], right[c, n], self.P, self.Q
                        )
                        self.assertTrue(
                            torch.allclose(actual[b, c, t, n], expected, rtol=1e-10, atol=1e-10)
                        )

    def test_linear_is_nondegenerate(self):
        # A degenerate/broken factorization (e.g. a transposition bug
        # collapsing to a low-rank or block-sparse map) would make the
        # induced D x D matrix singular or rank-deficient for generic
        # factors. Build that matrix by pushing standard basis vectors
        # through the reference (not monarch_node_linear itself) and
        # check it's full rank.
        left, right = self._random_factors(seed=3)
        columns = [
            _reference_monarch_forward(
                torch.eye(self.D, dtype=torch.float64)[i], left[0, 0], right[0, 0], self.P, self.Q
            )
            for i in range(self.D)
        ]
        dense_equivalent = torch.stack(columns, dim=1)
        self.assertEqual(torch.linalg.matrix_rank(dense_equivalent).item(), self.D)

    def test_gradient_flows_to_both_factors(self):
        cells, nodes = 2, 2
        left, right = self._random_factors(seed=4, cells=cells, nodes=nodes)
        left = left.requires_grad_(True)
        right = right.requires_grad_(True)
        base = torch.randn(1, cells, 3, self.D, dtype=torch.float64, requires_grad=True)

        out = monarch_node_linear(base, left, right)
        out.sum().backward()

        self.assertIsNotNone(left.grad)
        self.assertIsNotNone(right.grad)
        self.assertIsNotNone(base.grad)
        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertTrue(torch.isfinite(right.grad).all())
        self.assertTrue(torch.isfinite(base.grad).all())

    def test_gradcheck(self):
        cells, nodes = 1, 1
        left, right = self._random_factors(seed=5, cells=cells, nodes=nodes)
        left = left.requires_grad_(True)
        right = right.requires_grad_(True)
        base = torch.randn(1, cells, 2, self.D, dtype=torch.float64, requires_grad=True)

        self.assertTrue(
            torch.autograd.gradcheck(
                monarch_node_linear, (base, left, right), eps=1e-6, atol=1e-8
            )
        )

    def test_rejects_mismatched_factor_shapes(self):
        left, right = self._random_factors()
        bad_right = right[:, :, :, :-1, :-1]  # wrong q
        base = torch.randn(1, 1, 1, self.D, dtype=torch.float64)
        with self.assertRaises(ValueError):
            monarch_node_linear(base, left, bad_right)

    def test_rejects_wrong_input_features(self):
        left, right = self._random_factors()
        base = torch.randn(1, 1, 1, self.D + 1, dtype=torch.float64)
        with self.assertRaises(ValueError):
            monarch_node_linear(base, left, right)

    def test_production_parameter_count(self):
        # cells=8, nodes=4, d_model=1312, p=32, q=41 (the real production
        # shape this branch targets) -- q*p^2 + p*q^2 per instance,
        # ~18x smaller than the dense [cells,nodes,d,d] equivalent.
        cells, nodes, p, q = 8, 4, 32, 41
        self.assertEqual(p * q, 1312)
        per_instance = q * p * p + p * q * q
        self.assertEqual(per_instance, 95_776)
        total_monarch = cells * nodes * per_instance
        self.assertEqual(total_monarch, 3_064_832)
        total_dense = cells * nodes * 1312 * 1312
        self.assertEqual(total_dense, 55_083_008)
        self.assertGreater(total_dense / total_monarch, 17.9)


class FoldMonarchNodeParametersTests(unittest.TestCase):
    P, Q = 2, 3
    D = P * Q

    def test_matches_monarch_node_linear_on_the_embedding_itself(self):
        # fold_monarch_node_parameters's offset should equal running
        # node_embedding through the exact same Monarch transform that
        # monarch_node_linear applies to `base`, plus bias -- verify by
        # calling monarch_node_linear directly on node_embedding
        # (reshaped to look like a [B=1,C,T=1,D] "base") and comparing.
        cells, nodes = 3, 2
        torch.manual_seed(6)
        left = torch.randn(cells, nodes, self.Q, self.P, self.P, dtype=torch.float64)
        right = torch.randn(cells, nodes, self.P, self.Q, self.Q, dtype=torch.float64)
        node_embedding = torch.randn(cells, nodes, self.D, dtype=torch.float64)
        bias = torch.randn(cells, nodes, self.D, dtype=torch.float64)

        offset = fold_monarch_node_parameters(node_embedding, left, right, bias)
        self.assertEqual(offset.shape, (cells, nodes, self.D))

        for c in range(cells):
            for n in range(nodes):
                single_left = left[c : c + 1, n : n + 1]
                single_right = right[c : c + 1, n : n + 1]
                base = node_embedding[c, n].view(1, 1, 1, self.D)
                expected = monarch_node_linear(base, single_left, single_right)[0, 0, 0, 0]
                expected = expected + bias[c, n]
                self.assertTrue(torch.allclose(offset[c, n], expected, rtol=1e-10, atol=1e-10))

    def test_gradient_flows(self):
        cells, nodes = 2, 2
        torch.manual_seed(7)
        left = torch.randn(
            cells, nodes, self.Q, self.P, self.P, dtype=torch.float64, requires_grad=True
        )
        right = torch.randn(
            cells, nodes, self.P, self.Q, self.Q, dtype=torch.float64, requires_grad=True
        )
        node_embedding = torch.randn(
            cells, nodes, self.D, dtype=torch.float64, requires_grad=True
        )
        bias = torch.randn(cells, nodes, self.D, dtype=torch.float64, requires_grad=True)

        offset = fold_monarch_node_parameters(node_embedding, left, right, bias)
        offset.sum().backward()

        for tensor in (left, right, node_embedding, bias):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
