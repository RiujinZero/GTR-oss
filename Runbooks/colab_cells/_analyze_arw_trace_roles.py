"""How GTR-ARW's weights organize and move information, by structural role.

Companion to _analyze_arw_trace_geometry.py, which measured aggregate
update geometry (rank, rotation) with every parameter pooled into one
vector. This script instead groups the trace's 209 named parameters by
their architectural role and tracks each role's update magnitude over
training -- a direct answer to the user's question (2026-08-10): "Do the
traces give us a way to see how the weights organize and move
information?"

gator_cells_vectorized.py has no procoim_parameter_groups() (that's a
GTR-L/Procoim-specific method), so AdamWWeightTracer recorded everything
as "ungrouped" -- but the parameter NAMES themselves are structurally
interpretable, so the grouping below is done by name pattern instead:

  - inter_cell_coupling: left_projection/right_projection/coupling_norm/
    neighbor_gates/self_gates/output_votes -- the mechanism that actually
    moves information BETWEEN the 10 cells in the ring (this is the
    literal cross-cell wiring, the closest thing this architecture has to
    GTR-L's routing/streams).
  - routing_addressing: per-cell router.{weight,bias}, address_to_edges/
    address_to_readout/address_to_state, relative_node_bias, start_logits
    -- WITHIN-cell decisions about which node/edge/readout a token's
    address maps to.
  - node_computation: per-cell node_transforms.*, norm.* -- the actual
    per-node transform applied once information has arrived.
  - embedding: per-cell token_embedding, node_embedding, step_projection
    -- how raw tokens/positions get encoded into the cell's working
    representation in the first place.

For each role, reports delta_rms (RMS of interval_delta across every
parameter in that role, comparable across roles of different size) at
each snapshot, so the reader can see which roles move early vs late.
"""

import re
from pathlib import Path

import torch

TRACE_DIR = Path(
    "C:/Users/user/AppData/Local/Temp/gtrl_procoim_fix_20260805/"
    "arw_10k_trace_seed0/checkpoint/adamw_weight_trace"
)

ROLE_PATTERNS = {
    "inter_cell_coupling": re.compile(
        r"^(left_projection|right_projection|coupling_norm|"
        r"neighbor_gates|self_gates|output_votes)"
    ),
    "routing_addressing": re.compile(
        r"^cells\.\d+\.(router|address_to_edges|address_to_readout|"
        r"address_to_state|relative_node_bias|start_logits)"
    ),
    "node_computation": re.compile(
        r"^cells\.\d+\.(node_transforms|norm)"
    ),
    "embedding": re.compile(
        r"^cells\.\d+\.(token_embedding|node_embedding|step_projection)"
    ),
}

# Same six windows as _analyze_arw_trace_geometry.py, keyed to the
# documented accuracy curve in Runbooks/GTR_ARW_MASTERY_TRACE_LOG.md.
WINDOWS = {
    "pre_breakout (chance)": [800, 1200, 1800],
    "breakout_jump (13%->43%)": [2000, 2400, 3000],
    "rapid_climb (43%->89%)": [3600, 5200, 7000],
    "mid90s_plateau (91%->98%)": [9000, 11400, 13800],
    "late_convergence (96-99%)": [15600, 17400, 19000],
    "final_destabilization (grad spike)": [19400, 19800, 20000],
}


def role_of(name: str) -> str | None:
    for role, pattern in ROLE_PATTERNS.items():
        if pattern.match(name):
            return role
    return None


def load(step: int) -> dict:
    path = TRACE_DIR / f"trace_step_{step:09d}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def role_delta_rms(payload: dict) -> dict[str, float]:
    delta = payload["interval_delta"]
    by_role: dict[str, list[torch.Tensor]] = {role: [] for role in ROLE_PATTERNS}
    unmatched = 0
    for name, tensor in delta.items():
        role = role_of(name)
        if role is None:
            unmatched += 1
            continue
        by_role[role].append(tensor.flatten())
    if unmatched:
        raise RuntimeError(f"{unmatched} parameters matched no role pattern")
    result = {}
    for role, tensors in by_role.items():
        flat = torch.cat(tensors)
        result[role] = float(flat.square().mean().sqrt())
    return result


def main() -> None:
    role_names = list(ROLE_PATTERNS)
    header = "  ".join(f"{r:>20s}" for r in role_names)
    print(f"{'step':>7s}  {header}")

    for window_name, steps in WINDOWS.items():
        print(f"\n=== {window_name} ===")
        for step in steps:
            payload = load(step)
            rms = role_delta_rms(payload)
            row = "  ".join(f"{rms[r]:20.6e}" for r in role_names)
            print(f"{step:7d}  {row}")

    # Also report each role's share of total update energy per window,
    # since raw RMS alone doesn't show whether a role is dominant.
    print("\n\n=== share of total update energy per window (%) ===")
    for window_name, steps in WINDOWS.items():
        totals = {role: 0.0 for role in role_names}
        grand_total = 0.0
        for step in steps:
            payload = load(step)
            delta = payload["interval_delta"]
            for name, tensor in delta.items():
                role = role_of(name)
                energy = float(tensor.double().pow(2).sum())
                totals[role] += energy
                grand_total += energy
        shares = "  ".join(
            f"{role}={100 * totals[role] / grand_total:5.1f}%"
            for role in role_names
        )
        print(f"{window_name:38s} {shares}")


if __name__ == "__main__":
    main()
