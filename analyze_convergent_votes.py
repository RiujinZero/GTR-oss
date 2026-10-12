"""Report the learned terminal-voter coalition in Convergent checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import torch


@dataclass(frozen=True)
class VoteSummary:
    """Scalar diagnostics for one normalized terminal-vote vector."""

    lanes: int
    entropy: float
    effective_entropy: float
    effective_l2: float
    top1_mass: float
    top3_mass: float
    top5_mass: float
    top90_count: int
    top95_count: int
    top99_count: int


def terminal_vote_logits(
    payload: Mapping[str, object],
) -> tuple[str, torch.Tensor]:
    """Find the sole terminal-vote parameter in a checkpoint payload."""

    state = payload.get("model", payload)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a model state mapping")
    matches = [
        (str(name), value)
        for name, value in state.items()
        if str(name).endswith("terminal_votes")
        and isinstance(value, torch.Tensor)
    ]
    if len(matches) != 1:
        names = [name for name, _ in matches]
        raise ValueError(
            "expected exactly one terminal-vote tensor, "
            f"found {len(matches)}: {names}"
        )
    name, logits = matches[0]
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError(
            f"{name} must be a nonempty rank-one tensor, got {tuple(logits.shape)}"
        )
    return name, logits.detach().to(dtype=torch.float64, device="cpu")


def _mass_count(sorted_weights: torch.Tensor, threshold: float) -> int:
    cumulative = sorted_weights.cumsum(dim=0)
    return int(
        torch.searchsorted(
            cumulative,
            torch.tensor(threshold, dtype=cumulative.dtype),
        ).item()
        + 1
    )


def summarize_votes(logits: torch.Tensor) -> tuple[torch.Tensor, VoteSummary]:
    """Normalize terminal logits and return concentration diagnostics."""

    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("terminal logits must be a nonempty rank-one tensor")
    weights = logits.detach().to(dtype=torch.float64, device="cpu").softmax(0)
    sorted_weights = weights.sort(descending=True).values
    entropy = float(
        -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum()
    )
    summary = VoteSummary(
        lanes=weights.numel(),
        entropy=entropy,
        effective_entropy=math.exp(entropy),
        effective_l2=float(weights.square().sum().reciprocal()),
        top1_mass=float(sorted_weights[:1].sum()),
        top3_mass=float(sorted_weights[:3].sum()),
        top5_mass=float(sorted_weights[:5].sum()),
        top90_count=_mass_count(sorted_weights, 0.90),
        top95_count=_mass_count(sorted_weights, 0.95),
        top99_count=_mass_count(sorted_weights, 0.99),
    )
    return weights, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        type=Path,
        nargs="+",
        help="one or more prefix or language checkpoints",
    )
    args = parser.parse_args()

    for checkpoint in args.checkpoints:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, Mapping):
            raise ValueError(f"{checkpoint} is not a checkpoint mapping")
        key, logits = terminal_vote_logits(payload)
        weights, summary = summarize_votes(logits)
        ranking = weights.argsort(descending=True)
        state = payload.get("model", payload)
        query_matches = [
            value.detach().to(dtype=torch.float64, device="cpu")
            for name, value in state.items()
            if str(name).endswith("terminal_vote_query")
            and isinstance(value, torch.Tensor)
        ]
        if len(query_matches) > 1:
            raise ValueError(
                "checkpoint contains multiple terminal-vote query tensors"
            )
        checkpoint_config = payload.get("model_config", {})
        vote_mode = str(
            checkpoint_config.get(
                "terminal_vote_mode",
                "prompt" if query_matches else "global",
            )
        )
        query_rms = (
            float(query_matches[0].square().mean().sqrt())
            if query_matches
            else 0.0
        )
        print(
            f"checkpoint={checkpoint} "
            f"step={int(payload.get('step', -1))} "
            f"key={key} mode={vote_mode} lanes={summary.lanes} "
            f"effective_entropy={summary.effective_entropy:.4f} "
            f"effective_l2={summary.effective_l2:.4f} "
            f"query_rms={query_rms:.6f} "
            f"top1_mass={summary.top1_mass:.4f} "
            f"top3_mass={summary.top3_mass:.4f} "
            f"top5_mass={summary.top5_mass:.4f} "
            f"mass_counts_90_95_99="
            f"{summary.top90_count}:{summary.top95_count}:{summary.top99_count}"
        )
        print(
            "terminal_weights="
            + ",".join(
                f"{int(index)}:{float(weights[index]):.6f}"
                for index in ranking
            )
        )
        if vote_mode != "global":
            print(
                "note=terminal_weights_are_global_priors; "
                f"actual {vote_mode}-conditioned coalitions require a "
                "forward probe"
            )


if __name__ == "__main__":
    main()
