"""Phase-gated cell-growth training loop for Arrow (--architecture gator).

Calls gator_language_train.py IN-PROCESS as an imported module (no
subprocess) so growth orchestration stays entirely separate from the core
trainer: this script owns the phase/growth logic, gator_language_train.py
still only knows how to run --steps of training and doesn't know growth
exists. Growing between phases uses gator_cell_growth.py's clone+scatter
transform directly, also in-process.

A "phase" trains at a fixed cell count until either:
  - PPL has stayed below --ppl-threshold for --sustain-steps consecutive
    steps (the intended stopping condition), or
  - --max-steps-per-phase is hit first (a fallback so a phase that never
    sustains the threshold doesn't run forever).

Then cells are multiplied by --clones-per-cell (2 = doubling) and the next
phase starts by growing from the previous phase's final checkpoint.

Usage mirrors gator_language_train.py: growth-specific flags below, plus
any gator_language_train.py flag (--data, --d-model, --vocab-size, etc.)
passed straight through untouched. Do NOT pass --cells, --neighbors,
--checkpoint-dir, --steps, --resume-checkpoint, or --grow-from-checkpoint
directly - this script controls those across phases.

Example:
    python3 gator_growth_train.py \
        --data smoltalk_everyday.jsonl --primary-stream-weight 1 \
        --vocab-size 20000 --d-model 384 --embedding-dim 32 \
        --context-tokens 512 --response-tokens 128 --batch-size 4 \
        --node-topology hybrid_hemispheric --relays 2 \
        --cell-transport-mode stewardship --stewardship-min-active 2 \
        --stewardship-max-active 64 --decode-mode serial \
        --formation-objective slot_ce \
        --start-cells 4 --clones-per-cell 2 \
        --ppl-threshold 16 --sustain-steps 40000 \
        --segment-steps 2000 --max-steps-per-phase 200000 --max-cells 256 \
        --checkpoint-root /content/growth_run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

import gator_language_train
from gator_cell_growth import grow_gator_state_dict


def _checkpoint_step(checkpoint_path: Path) -> int:
    if not checkpoint_path.is_file():
        return 0
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return int(state.get("step", 0))


_STEP_PPL_PATTERN = re.compile(r"^step=(\d+)\b.*?\bppl=([0-9.]+)")


class _PPLTee:
    """Tee real stdout while parsing step=.../ppl=... lines as they print."""

    def __init__(self, real_stdout):
        self.real_stdout = real_stdout
        self.history: list[tuple[int, float]] = []
        self._buffer = ""

    def write(self, text: str) -> None:
        self.real_stdout.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            match = _STEP_PPL_PATTERN.match(line)
            if match:
                self.history.append((int(match.group(1)), float(match.group(2))))

    def flush(self) -> None:
        self.real_stdout.flush()


def _neighbors_for(n_cells: int) -> int:
    # Neighbors must be positive, even, and < n_cells. 2 is the only valid
    # choice at n_cells=4; 4 works for every larger cell count this loop
    # produces (doubling from 4 never revisits n_cells < 8 afterward).
    return 2 if n_cells <= 4 else 4


def _sustained_streak(history: list[tuple[int, float]], threshold: float) -> tuple[int, int] | None:
    """Return (start_step, last_step) of the current below-threshold streak."""

    start_step = None
    last_step = None
    for step, ppl in history:
        if ppl >= threshold:
            start_step = None
        elif start_step is None:
            start_step = step
        last_step = step
    if start_step is None or last_step is None:
        return None
    return start_step, last_step


def _parse_growth_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--start-cells", type=int, default=4)
    parser.add_argument("--clones-per-cell", type=int, default=2)
    parser.add_argument("--grow-perturb-scale", type=float, default=0.01)
    parser.add_argument("--ppl-threshold", type=float, default=16.0)
    parser.add_argument("--sustain-steps", type=int, default=40_000)
    parser.add_argument("--segment-steps", type=int, default=2_000)
    parser.add_argument("--max-steps-per-phase", type=int, default=400_000)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-phases", type=int, default=None)
    parser.add_argument("--checkpoint-root", required=True)
    known, remaining = parser.parse_known_args(argv)
    if known.start_cells < 4:
        parser.error(
            "--start-cells must be >= 4 (neighbors must be positive, even, "
            "and < n_cells - no valid neighbors exists below 4 cells)"
        )
    if known.clones_per_cell < 2:
        parser.error("--clones-per-cell must be at least 2")
    return known, remaining


def _clamp_stewardship_max_active(argv: list[str], cells: int) -> list[str]:
    """gator_language_train.py requires stewardship_max_active in
    [min_active, n_cells] (0 means uncapped). Early growth phases have few
    cells, so a --stewardship-max-active configured for the eventual/target
    cell count can exceed n_cells at small phases - clamp it down to cells
    for those phases rather than requiring a separate flag per phase."""

    clamped = list(argv)
    for i, token in enumerate(clamped):
        if token == "--stewardship-max-active" and i + 1 < len(clamped):
            configured = int(clamped[i + 1])
            if configured > 0:
                clamped[i + 1] = str(min(configured, cells))
    return clamped


def _run_segment(
    base_argv: list[str],
    *,
    cells: int,
    neighbors: int,
    checkpoint_dir: Path,
    steps: int,
    resume_checkpoint: Path | None,
    grow_from_checkpoint: Path | None,
    clones_per_cell: int,
    grow_perturb_scale: float,
) -> None:
    argv = [
        *_clamp_stewardship_max_active(base_argv, cells),
        "--cells", str(cells),
        "--neighbors", str(neighbors),
        "--checkpoint-dir", str(checkpoint_dir),
        "--steps", str(steps),
    ]
    if resume_checkpoint is not None:
        argv += ["--resume-checkpoint", str(resume_checkpoint)]
    elif grow_from_checkpoint is not None:
        argv += [
            "--grow-from-checkpoint", str(grow_from_checkpoint),
            "--grow-clones-per-cell", str(clones_per_cell),
            "--grow-perturb-scale", str(grow_perturb_scale),
        ]
    sys.argv = ["gator_language_train.py", *argv]
    gator_language_train.main()


def main() -> None:
    growth_args, trainer_argv = _parse_growth_args(sys.argv[1:])
    checkpoint_root = Path(growth_args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    real_stdout = sys.stdout
    tee = _PPLTee(real_stdout)
    sys.stdout = tee
    try:
        cells = growth_args.start_cells
        prev_ckpt: Path | None = None
        phase = 0
        while True:
            if growth_args.max_phases is not None and phase >= growth_args.max_phases:
                print(f"[growth] reached max_phases={growth_args.max_phases}, stopping", flush=True)
                break

            neighbors = _neighbors_for(cells)
            ckpt_dir = checkpoint_root / f"phase{phase}_cells{cells}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            latest_ckpt = ckpt_dir / "latest.pt"

            tee.history.clear()
            # If a checkpoint for this phase already exists on disk (the
            # process crashed and this phase is being retried), resume from
            # it instead of re-growing from scratch and losing that progress.
            phase_steps_run = _checkpoint_step(latest_ckpt)
            if phase_steps_run > 0:
                print(
                    f"[growth] found existing checkpoint for phase {phase} at "
                    f"step={phase_steps_run}, resuming instead of re-growing",
                    flush=True,
                )
            satisfied = False
            print(f"[growth] === phase {phase}: cells={cells} neighbors={neighbors} ===", flush=True)
            while phase_steps_run < growth_args.max_steps_per_phase:
                segment = min(growth_args.segment_steps, growth_args.max_steps_per_phase - phase_steps_run)
                _run_segment(
                    trainer_argv,
                    cells=cells,
                    neighbors=neighbors,
                    checkpoint_dir=ckpt_dir,
                    steps=segment,
                    resume_checkpoint=latest_ckpt if phase_steps_run > 0 else None,
                    grow_from_checkpoint=prev_ckpt if phase_steps_run == 0 else None,
                    clones_per_cell=growth_args.clones_per_cell,
                    grow_perturb_scale=growth_args.grow_perturb_scale,
                )
                phase_steps_run += segment

                streak = _sustained_streak(tee.history, growth_args.ppl_threshold)
                if streak is not None:
                    start_step, last_step = streak
                    if last_step - start_step >= growth_args.sustain_steps:
                        satisfied = True
                        print(
                            f"[growth] phase {phase} satisfied: ppl<{growth_args.ppl_threshold} "
                            f"sustained steps {start_step}->{last_step} "
                            f"({last_step - start_step} steps)",
                            flush=True,
                        )
                        break

            if not satisfied:
                print(
                    f"[growth] phase {phase} hit max_steps_per_phase="
                    f"{growth_args.max_steps_per_phase} without sustaining the "
                    f"PPL threshold - moving on anyway (fallback)",
                    flush=True,
                )

            if growth_args.max_cells is not None and cells >= growth_args.max_cells:
                print(f"[growth] reached max_cells={growth_args.max_cells}, stopping", flush=True)
                break

            prev_ckpt = latest_ckpt
            cells = cells * growth_args.clones_per_cell
            phase += 1
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":
    main()
