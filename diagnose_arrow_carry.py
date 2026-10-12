"""Lockstep gradient forensics for original Arrow versus Arrow-Carry."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from gator_cells_vectorized import VectorizedGatorCellRing
from parallel_prefix_benchmark import make_prefix_batch


def _tensor_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach()
    right = right.detach()
    return float((left - right).abs().max())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.reshape(-1).double()
    right_flat = right.reshape(-1).double()
    denominator = left_flat.norm() * right_flat.norm()
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(left_flat, right_flat) else 0.0
    return float(torch.dot(left_flat, right_flat) / denominator)


def _gradient_report(
    original: VectorizedGatorCellRing,
    carry: VectorizedGatorCellRing,
    *,
    limit: int,
) -> list[tuple[str, float, float]]:
    carry_parameters = dict(carry.named_parameters())
    rows: list[tuple[str, float, float]] = []
    for name, parameter in original.named_parameters():
        other = carry_parameters[name]
        if parameter.grad is None or other.grad is None:
            continue
        rows.append(
            (
                name,
                _tensor_difference(parameter.grad, other.grad),
                _cosine(parameter.grad, other.grad),
            )
        )
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows[:limit]


def _parameter_divergence(
    original: VectorizedGatorCellRing,
    carry: VectorizedGatorCellRing,
) -> tuple[str, float]:
    carry_parameters = dict(carry.named_parameters())
    rows = [
        (
            name,
            _tensor_difference(parameter, carry_parameters[name]),
        )
        for name, parameter in original.named_parameters()
    ]
    return max(rows, key=lambda row: row[1])


def _tie_gate_gradient(model: VectorizedGatorCellRing) -> None:
    gradient = model.neighbor_gates.grad
    if gradient is None:
        raise RuntimeError("neighbor_gates did not receive a gradient")
    shared = gradient.mean(dim=0, keepdim=True)
    gradient.copy_(shared.expand_as(gradient))


def _retie_gate_values(model: VectorizedGatorCellRing) -> None:
    with torch.no_grad():
        shared = model.neighbor_gates.mean(dim=0, keepdim=True)
        model.neighbor_gates.copy_(
            shared.expand_as(model.neighbor_gates)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer-epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--carry-owner",
        choices=("source", "destination"),
        default="source",
        help=(
            "source reproduces the failed outgoing-gate carry; destination "
            "preserves original Arrow's receiving-gate credit assignment"
        ),
    )
    parser.add_argument(
        "--gate-policy",
        choices=("independent", "frozen", "tied"),
        default="independent",
    )
    parser.add_argument("--report-limit", type=int, default=12)
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.batch_size <= 0
        or args.optimizer_epsilon <= 0
    ):
        parser.error("steps, batch size, and optimizer epsilon must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original = VectorizedGatorCellRing(
        carry_state=False,
        reference_dense=True,
    ).to(device)
    carry = VectorizedGatorCellRing(
        carry_state=args.carry_owner == "destination",
        source_carry=args.carry_owner == "source",
        reference_dense=True,
    ).to(device)
    carry.load_state_dict(original.state_dict())
    if args.gate_policy == "frozen":
        original.neighbor_gates.requires_grad_(False)
        carry.neighbor_gates.requires_grad_(False)

    original_optimizer = torch.optim.AdamW(
        original.parameters(),
        lr=8.0e-3,
        weight_decay=1.0e-4,
        eps=args.optimizer_epsilon,
    )
    carry_optimizer = torch.optim.AdamW(
        carry.parameters(),
        lr=8.0e-3,
        weight_decay=1.0e-4,
        eps=args.optimizer_epsilon,
    )
    print(
        f"device={device.type} steps={args.steps} "
        f"carry_owner={args.carry_owner} "
        f"gate_policy={args.gate_policy} parameters={original.parameter_count}",
        f"optimizer_epsilon={args.optimizer_epsilon:g}",
        flush=True,
    )

    checkpoints = {1, 2, 5, 10, 25, 50, 100, args.steps}
    for step in range(args.steps):
        observed, targets = make_prefix_batch(
            args.batch_size,
            7,
            device,
        )
        temperature = max(0.35, 1.25 * (0.997**step))
        original_optimizer.zero_grad(set_to_none=True)
        carry_optimizer.zero_grad(set_to_none=True)
        original_logits = original(
            observed,
            output_slots=7,
            temperature=temperature,
        )
        carry_logits = carry(
            observed,
            output_slots=7,
            temperature=temperature,
        )
        original_loss = F.cross_entropy(
            original_logits.reshape(-1, 16),
            targets.reshape(-1),
        )
        carry_loss = F.cross_entropy(
            carry_logits.reshape(-1, 16),
            targets.reshape(-1),
        )
        original_loss.backward()
        carry_loss.backward()

        if step == 0:
            print(
                f"initial_logit_max_abs="
                f"{_tensor_difference(original_logits, carry_logits):.9g} "
                f"initial_loss_abs="
                f"{abs(float((original_loss - carry_loss).detach())):.9g}",
                flush=True,
            )
            for name, maximum, cosine in _gradient_report(
                original,
                carry,
                limit=args.report_limit,
            ):
                print(
                    f"gradient name={name} max_abs={maximum:.9g} "
                    f"cosine={cosine:.9f}",
                    flush=True,
                )

        if args.gate_policy == "tied":
            _tie_gate_gradient(original)
            _tie_gate_gradient(carry)
        torch.nn.utils.clip_grad_norm_(original.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(carry.parameters(), 1.0)
        original_optimizer.step()
        carry_optimizer.step()
        if args.gate_policy == "tied":
            _retie_gate_values(original)
            _retie_gate_values(carry)

        completed = step + 1
        if completed not in checkpoints:
            continue
        parameter_name, parameter_maximum = _parameter_divergence(
            original,
            carry,
        )
        print(
            f"step={completed} "
            f"original_loss={float(original_loss.detach()):.6f} "
            f"carry_loss={float(carry_loss.detach()):.6f} "
            f"logit_max_abs="
            f"{_tensor_difference(original_logits, carry_logits):.9g} "
            f"parameter={parameter_name} "
            f"parameter_max_abs={parameter_maximum:.9g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
