"""One-off analysis of downloaded AdamW weight-trace artifacts for
gtrl_len7_adamw_trace_seed0_gatebias3_noreadoutbias_routerocc.

Compares per-parameter interval_delta L2 norms across four windows:
  - pre-breakout   (step 500-2000)
  - pos1 breakout  (step 7500-9000, pos1 first hits ~1.000)
  - pos2 recruit   (step 20000-24500, pos2 joins pos1 near-ceiling)
  - stalled plateau (step 55000-63000, positions 3-7 still chance)

For each window, ranks named parameters by mean interval_delta L2 norm
and reports the top movers + their semantic group, to see whether the
same parameters keep moving during the stall (mechanism still trying,
just not succeeding) or whether the stall corresponds to those
parameters going quiet (mechanism disengaged).
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
    "pos1_breakout": [7500, 8000, 8500, 9000],
    "pos2_recruit": [20000, 21000, 22000, 23000, 24000, 24500],
    "stalled_plateau": [55000, 57000, 59000, 61000, 63000],
}


def load(step: int) -> dict:
    path = TRACE_DIR / f"trace_step_{step:09d}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def per_param_norms(payload: dict) -> dict[str, float]:
    delta = payload["interval_delta"]
    return {name: tensor.norm().item() for name, tensor in delta.items()}


def main() -> None:
    groups_by_name = None
    window_top_movers: dict[str, list[tuple[str, float, str]]] = {}

    for window_name, steps in WINDOWS.items():
        accum: dict[str, list[float]] = {}
        for step in steps:
            payload = load(step)
            if groups_by_name is None:
                groups_by_name = payload["parameter_groups"]
            norms = per_param_norms(payload)
            for name, value in norms.items():
                accum.setdefault(name, []).append(value)

        means = {name: sum(v) / len(v) for name, v in accum.items()}
        ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
        top10 = [
            (name, value, groups_by_name.get(name, "?"))
            for name, value in ranked[:10]
        ]
        window_top_movers[window_name] = top10

        print(f"\n=== {window_name} (steps {steps}) ===")
        for name, value, group in top10:
            print(f"  {value:8.4f}  [{group:16s}] {name}")

    # Group-level totals per window (sum of mean per-param norms by semantic group)
    print("\n\n=== group-level totals (sum of mean per-param delta norm) ===")
    for window_name, steps in WINDOWS.items():
        accum: dict[str, list[float]] = {}
        for step in steps:
            payload = load(step)
            norms = per_param_norms(payload)
            for name, value in norms.items():
                accum.setdefault(name, []).append(value)
        means = {name: sum(v) / len(v) for name, v in accum.items()}
        group_totals: dict[str, float] = {}
        for name, value in means.items():
            group = groups_by_name.get(name, "?")
            group_totals[group] = group_totals.get(group, 0.0) + value
        ordered = sorted(group_totals.items(), key=lambda kv: kv[1], reverse=True)
        print(f"\n{window_name}:")
        for group, total in ordered:
            print(f"  {total:8.4f}  {group}")


if __name__ == "__main__":
    main()
