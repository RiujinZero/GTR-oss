"""Follow-up to Papers/PROCOIM_ADAMW_WEIGHT_GEOMETRY_REPORT.md.

That report characterized AdamW's update geometry (rotation between
intervals, low-rank structure) on a run that never left chance, and
explicitly flagged as missing evidence: "a future run through its actual
breakout, then compare plateau intervals against post-breakout
intervals." gtrl_len7_adamw_trace_seed0_gatebias3_noreadoutbias_routerocc
now has exactly that: a real breakout (pos1, then pos2), a stalled
plateau, and a full destabilization/recovery event, all traced.

Computes, per consecutive-interval pair:
  - cosine similarity between this interval's global update vector and
    the previous interval's (same "Previous cosine" metric as the report)
  - rank-4 / rank-8 SVD energy capture, per matrix-shaped parameter,
    weighted-averaged by each parameter's share of total update energy
    (same "Rank 4" / "Rank 8" columns as the report)

Reports these per-window (pre-breakout, pos1 breakout, pos2 recruitment,
stalled plateau, destabilization, post-recovery) to see whether the
rotating-low-rank pattern the original report found still holds during
genuinely productive learning, or looks different.
"""

from pathlib import Path

import torch

TRACE_DIR = Path(
    "/mnt/c/Users/user/Documents/GTR_test_deps/"
    "gtrl_len7_adamw_trace_seed0_gatebias3_noreadoutbias_routerocc/"
    "checkpoint/adamw_weight_trace"
)

WINDOWS = {
    "pre_breakout": [500, 1000, 1500, 2000],
    "pos1_breakout": [7000, 7500, 8000, 8500, 9000],
    "pos2_recruit": [20000, 21000, 22000, 23000, 24000, 24500],
    "stalled_plateau_early": [30000, 32000, 34000, 36000],
    "destabilization": [42500, 43000, 43500, 44000, 44500],
    "post_destab_dip": [48000, 48500, 49000, 49500],
    "stalled_plateau_late": [55000, 57000, 59000, 61000, 63000],
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
    prev_step = None
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
            prev_step = step
        print()


if __name__ == "__main__":
    main()
