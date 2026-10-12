"""Tests for Procoim-S Phase 6: moment transfer across basis refresh."""

from __future__ import annotations

import pytest
import torch

from gator.optimizers.moment_transfer import overlap_matrix, transfer_moments


def _orthonormal_basis(k: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(dim, dim, generator=generator)
    q, _ = torch.linalg.qr(raw)
    return q[:k]


class TestOverlapMatrix:
    def test_identical_basis_has_unit_diagonal(self) -> None:
        basis = _orthonormal_basis(3, 5, seed=0)
        overlap = overlap_matrix(basis, basis)
        assert torch.allclose(overlap, torch.eye(3), atol=1e-5)

    def test_orthogonal_bases_have_zero_overlap(self) -> None:
        full = _orthonormal_basis(6, 6, seed=1)
        old_basis = full[:3]
        new_basis = full[3:]
        overlap = overlap_matrix(new_basis, old_basis)
        assert torch.allclose(overlap, torch.zeros(3, 3), atol=1e-5)

    def test_rejects_mismatched_dimension(self) -> None:
        with pytest.raises(ValueError):
            overlap_matrix(torch.randn(2, 4), torch.randn(2, 5))

    def test_rejects_non_2d_input(self) -> None:
        with pytest.raises(ValueError):
            overlap_matrix(torch.randn(4), torch.randn(2, 4))


class TestTransferMoments:
    def test_identical_basis_preserves_moments_when_floor_is_zero(self) -> None:
        basis = _orthonormal_basis(3, 5, seed=2)
        first = torch.tensor([1.0, 2.0, 3.0])
        second = torch.tensor([0.1, 0.2, 0.3])
        new_first, new_second = transfer_moments(
            basis, basis, first, second, variance_floor=0.0
        )
        assert torch.allclose(new_first, first, atol=1e-5)
        assert torch.allclose(new_second, second, atol=1e-5)

    def test_orthogonal_new_basis_gets_zero_first_moment_and_floors_second(self) -> None:
        full = _orthonormal_basis(6, 6, seed=3)
        old_basis = full[:3]
        new_basis = full[3:]
        first = torch.tensor([1.0, 2.0, 3.0])
        second = torch.tensor([0.5, 0.5, 0.5])
        new_first, new_second = transfer_moments(
            new_basis, old_basis, first, second, variance_floor=1e-4
        )
        assert torch.allclose(new_first, torch.zeros(3), atol=1e-5)
        assert torch.allclose(new_second, torch.full((3,), 1e-4), atol=1e-5)

    def test_variance_floor_is_additive(self) -> None:
        basis = _orthonormal_basis(2, 4, seed=4)
        first = torch.tensor([1.0, 1.0])
        second = torch.tensor([0.2, 0.2])
        _, without_floor = transfer_moments(basis, basis, first, second, variance_floor=0.0)
        _, with_floor = transfer_moments(basis, basis, first, second, variance_floor=0.05)
        assert torch.allclose(with_floor, without_floor + 0.05, atol=1e-5)

    def test_rejects_negative_variance_floor(self) -> None:
        basis = _orthonormal_basis(2, 4, seed=5)
        with pytest.raises(ValueError):
            transfer_moments(basis, basis, torch.zeros(2), torch.zeros(2), variance_floor=-1.0)

    def test_rejects_mismatched_moment_shape(self) -> None:
        basis = _orthonormal_basis(2, 4, seed=6)
        with pytest.raises(ValueError):
            transfer_moments(basis, basis, torch.zeros(3), torch.zeros(2), variance_floor=0.0)
