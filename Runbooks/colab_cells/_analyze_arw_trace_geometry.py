"""GTR-ARW weight-geometry trace, through a REAL near-mastery convergence.

Companion to `_analyze_trace_geometry.py` (which ran this same methodology
on GTR-L's stalled/partial breakout traces). This one points at the one
local trace where the underlying model actually worked:
`arw_10k_trace_seed0` -- the 10,000-parameter GTR-ARW cell-ring checkpoint
(`gator_cells_vectorized.py`), traced end to end from chance through a
95.6% token / 74.7% exact-match convergence (see
Runbooks/GTR_ARW_MASTERY_TRACE_LOG.md for the full accuracy curve this
geometry corresponds to). User's question motivating this
(2026-08-10): "We haven't found a cheaper GTR yet that can reproduce what
GTR-ARW does. Don't we have a local ARW weight trace to look at?" -- yes,
100 snapshots, steps 200-20000, and it had never been run through this
geometry methodology before (only the informal step-by-step read in the
mastery trace log).

Same methodology as _analyze_trace_geometry.py: global update-vector
cosine with the previous 200-step interval; weighted-average rank-4/rank-8
SVD energy capture across matrix-shaped parameters. Windows below are
keyed to the documented accuracy curve in GTR_ARW_MASTERY_TRACE_LOG.md,
not chosen blind:
  - pre_breakout: still near chance (acc 6-9%)
  - breakout_jump: the sharp 13.4%->43.2% jump, clip_rate ramping 0->1.0
  - rapid_climb: 43%->89%, clip_rate pinned at 1.0
  - mid90s_plateau: 91%->98%, the user's own predicted "slow crawl"
  - late_convergence: 96-99%, clip_rate easing back off 1.0
  - final_destabilization: the late grad_norm spike (2.7->35.5) right
    before the run's final step
"""

from pathlib import Path

import torch

TRACE_DIR = Path(
    "C:/Users/user/AppData/Local/Temp/gtrl_procoim_fix_20260805/"
    "arw_10k_trace_seed0/checkpoint/adamw_weight_trace"
)

WINDOWS = {
    "pre_breakout": [800, 1000, 1200, 1400, 1600, 1800],
    "breakout_jump": [2000, 2200, 2400, 2600, 2800, 3000],
    "rapid_climb": [3200, 3600, 4000, 4600, 5200, 5800, 6400, 7000, 7600, 8000],
    "mid90s_plateau": [
        8600, 9000, 9600, 10200, 10800, 11400, 12000, 12600, 13200, 13800,
    ],
    "late_convergence": [15000, 15600, 16200, 16800, 17400, 18000, 18600, 19000],
    "final_destabilization": [19200, 19400, 19600, 19800, 20000],
}


def load(step: int) -> dict:
    path = TRACE_DIR / f"trace_step_{step:09d}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def flat_update(payload: dict) -> torch.Tensor:
    delta = payload["interval_delta"]
    names = sorted(delta.keys())
    return torch.cat([delta[name].flatten() for name in names])


def rank_energy(tensor: torch.Tensor, rank: int) -> float | None:
    if tensor.dim() != 2:
        return None
    try:
        s = torch.linalg.svdvals(tensor.double())
    except Exception:
        return None
    total = (s**2).sum().item()
    if total <= 0:
        return None
    captured = (s[:rank] ** 2).sum().item()
    return captured / total


def weighted_rank_capture(payload: dict, rank: int) -> float:
    delta = payload["interval_delta"]
    total_energy = 0.0
    captured_energy = 0.0
    for name, tensor in delta.items():
        if tensor.dim() != 2:
            continue
        energy = tensor.double().pow(2).sum().item()
        if energy <= 0:
            continue
        frac = rank_energy(tensor, rank)
        if frac is None:
            continue
        total_energy += energy
        captured_energy += energy * frac
    if total_energy <= 0:
        return float("nan")
    return captured_energy / total_energy


def main() -> None:
    all_steps = sorted(
        int(p.stem.split("_")[-1]) for p in TRACE_DIR.glob("trace_step_*.pt")
    )
    print(f"available steps: {all_steps[0]}..{all_steps[-1]} ({len(all_steps)} total)\n")

    prev_vec = None
    for window_name, steps in WINDOWS.items():
        print(f"=== {window_name} ===")
        for step in steps:
            payload = load(step)
            vec = flat_update(payload)
            cosine = None
            if prev_vec is not None:
                denom = vec.norm().item() * prev_vec.norm().item()
                if denom > 0:
                    cosine = (vec @ prev_vec).item() / denom
            rank4 = weighted_rank_capture(payload, 4)
            rank8 = weighted_rank_capture(payload, 8)
            update_norm = vec.norm().item()
            cos_str = f"{cosine:7.4f}" if cosine is not None else "   n/a "
            print(
                f"  step={step:>7d}  update_norm={update_norm:9.4f}  "
                f"prev_cos={cos_str}  rank4={rank4*100:6.2f}%  "
                f"rank8={rank8*100:6.2f}%"
            )
            prev_vec = vec
        print()


if __name__ == "__main__":
    main()
