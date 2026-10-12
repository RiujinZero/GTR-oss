"""Grow a trained Arrow (--architecture gator) checkpoint to more cells.

Clones each existing cell N times, scatters the clones across the grown
ring so no two siblings land within `neighbors` ring-distance of each
other (siblings get different neighbor exposure for free, since Arrow's
cell-to-cell topology is a pure function of ring index - see
gator/models/stewardship.py's ring_handoff_destinations), and applies a
small parameter perturbation per clone. Non-cell-indexed parameters
(token/output projections, the shared handoff_weight, lm_bias) are
copied unchanged.

Validated groundwork (2026-07-27 small-scale test, gator_convergent.py's
sibling analogue): scattering alone did not obviously separate siblings
under weak/synthetic gradient signal in 300 steps - real learnable data
and longer training are needed to judge whether scatter alone is enough
or whether an explicit quarantine phase (temporarily masking family
handoff edges) is also required. This module only implements the
clone+scatter+perturb step; quarantine is not yet implemented.
"""

from __future__ import annotations

import torch

PER_CELL_PARAMS = {
    "input_feature_gates",
    "cell_embedding",
    "address_to_state",
    "address_to_edges",
    "address_to_readout",
    "step_projection",
    "node_embedding",
    "node_weight",
    "node_bias",
    "router_weight",
    "router_bias",
    "relative_node_bias",
    "norm_weight",
    "norm_bias",
    "start_logits",
    "handoff_bias",
}


def scatter_order(old_n_cells: int, clones_per_cell: int) -> list[tuple[int, int]]:
    """Return [(parent_cell, clone_index), ...] in new-ring-index order.

    Interleaving by clone-generation index means every new ring position
    that is adjacent to another came from a DIFFERENT parent, and each
    parent's own clones are spread old_n_cells apart in ring index - far
    beyond any realistic `neighbors` ring-distance.
    """

    order = []
    for clone_index in range(clones_per_cell):
        for parent_index in range(old_n_cells):
            order.append((parent_index, clone_index))
    return order


def grow_gator_state_dict(
    old_state_dict: dict[str, torch.Tensor],
    new_state_dict_template: dict[str, torch.Tensor],
    *,
    old_n_cells: int,
    clones_per_cell: int,
    perturb_scale: float = 0.01,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], list[tuple[int, int]]]:
    """Return a grown state dict (matching new_state_dict_template's shapes)
    and the scatter order used, for similarity/quarantine bookkeeping.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = scatter_order(old_n_cells, clones_per_cell)
    grown = {}
    for name, template in new_state_dict_template.items():
        if name not in PER_CELL_PARAMS:
            if name in old_state_dict and old_state_dict[name].shape == template.shape:
                grown[name] = old_state_dict[name].clone()
            else:
                grown[name] = template.clone()
            continue
        parent_param = old_state_dict[name]
        # Per-cell parameters can still have non-cell dimensions that depend
        # on OTHER growth-exempt settings (e.g. handoff_bias's last axis is
        # neighbors + 1, not just a function of cell count). If a per-cell
        # slice's own shape changed between stages, there's no principled
        # way to map it - fall back to this parameter's fresh initialization
        # instead of crashing.
        if parent_param.shape[1:] != template.shape[1:]:
            grown[name] = template.clone()
            continue
        grown_param = torch.empty_like(template)
        for new_index, (parent_cell, clone_index) in enumerate(order):
            source_slice = parent_param[parent_cell]
            if clone_index == 0:
                grown_param[new_index] = source_slice
            else:
                noise = torch.randn(
                    source_slice.shape, generator=generator
                ).to(device=source_slice.device, dtype=source_slice.dtype)
                noise = noise * perturb_scale * source_slice.std().clamp_min(1e-6)
                grown_param[new_index] = source_slice + noise
        grown[name] = grown_param
    return grown, order


@torch.no_grad()
def sibling_vs_stranger_similarity(
    router_weight: torch.Tensor, order: list[tuple[int, int]]
) -> tuple[float, float]:
    """Cosine similarity of flattened per-cell router_weight, siblings vs not."""

    import statistics

    flat = router_weight.reshape(router_weight.shape[0], -1)
    flat = torch.nn.functional.normalize(flat, dim=-1)
    sim = flat @ flat.T
    sibling_sims, stranger_sims = [], []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            same_parent = order[i][0] == order[j][0]
            (sibling_sims if same_parent else stranger_sims).append(sim[i, j].item())
    return (
        statistics.mean(sibling_sims) if sibling_sims else float("nan"),
        statistics.mean(stranger_sims) if stranger_sims else float("nan"),
    )
