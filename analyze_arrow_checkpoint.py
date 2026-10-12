"""Inspect how a trained dense Arrow ensemble forms a prefix answer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from gator_cells_vectorized import VectorizedGatorCellRing
from parallel_prefix_benchmark import make_prefix_batch


def _metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float, float]:
    predictions = logits.argmax(-1)
    correct = predictions.eq(targets)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
    return (
        float(loss),
        float(correct.float().mean()),
        float(correct.all(-1).float().mean()),
    )


def _probability_summary(
    probabilities: torch.Tensor,
) -> tuple[float, float, float, float]:
    entropy = -(
        probabilities.clamp_min(1e-9)
        * probabilities.clamp_min(1e-9).log()
    ).sum(-1)
    maximum = probabilities.max(-1).values
    return (
        float(entropy.mean()),
        float(entropy.exp().mean()),
        float(maximum.mean()),
        float((maximum >= 0.9).float().mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--length", type=int, default=7)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--handoff-mode",
        choices=("reference", "destination-carry", "source-carry"),
        default="reference",
        help=(
            "inter-cell handoff semantics used while evaluating the same "
            "checkpoint"
        ),
    )
    parser.add_argument(
        "--node-top-k",
        type=int,
        default=None,
        help="post-training node support retained after every relay",
    )
    parser.add_argument(
        "--relay-limit",
        type=int,
        default=None,
        help=(
            "stop the trained four-relay computation after this many hops "
            "while preserving its original phase codes and write schedule"
        ),
    )
    parser.add_argument(
        "--active-cells",
        type=str,
        default=None,
        help="comma-separated cell IDs retained through every relay",
    )
    parser.add_argument(
        "--leave-one-cell-out",
        action="store_true",
        help="evaluate the full relay after disabling each cell in turn",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.length <= 0:
        parser.error("--batch-size and --length must be positive")
    if args.node_top_k is not None and not 1 <= args.node_top_k <= 4:
        parser.error("--node-top-k must be in [1, 4]")
    if args.relay_limit is not None and not 1 <= args.relay_limit <= 4:
        parser.error("--relay-limit must be in [1, 4]")
    active_cells = None
    if args.active_cells is not None:
        try:
            active_cells = tuple(
                int(value)
                for value in args.active_cells.split(",")
                if value.strip()
            )
        except ValueError:
            parser.error("--active-cells must contain comma-separated integers")
        if (
            not active_cells
            or len(set(active_cells)) != len(active_cells)
            or min(active_cells) < 0
            or max(active_cells) >= 10
        ):
            parser.error("--active-cells must be unique IDs in [0, 9]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    state = payload.get("model", payload)
    model = VectorizedGatorCellRing(
        carry_state=args.handoff_mode == "destination-carry",
        source_carry=args.handoff_mode == "source-carry",
        reference_dense=True,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    generator = torch.Generator(device=device).manual_seed(
        args.seed + args.length
    )
    observed, targets = make_prefix_batch(
        args.batch_size,
        args.length,
        device,
        generator=generator,
    )
    with torch.no_grad():
        logits, trace = model(
            observed,
            output_slots=args.length,
            return_trace=True,
            node_top_k=args.node_top_k,
            active_cells=active_cells,
            relay_limit=args.relay_limit,
        )

    loss, accuracy, exact = _metrics(logits, targets)
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets)
    position_loss = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        reduction="none",
    ).mean(dim=0)
    previous = torch.cat(
        (observed[:, :1], predictions[:, :-1]),
        dim=1,
    )
    predicted_deltas = (predictions - previous).remainder(12)
    step_consistency = predicted_deltas.eq(
        observed[:, 1:]
    ).float().mean(dim=0)
    print(
        f"checkpoint={args.checkpoint} step={payload.get('step', 'unknown')} "
        f"parameters={model.parameter_count} batch={args.batch_size} "
        f"handoff_mode={args.handoff_mode} "
        f"node_top_k={args.node_top_k} relay_limit={args.relay_limit} "
        f"active_cells={active_cells}",
        flush=True,
    )
    print(
        f"ensemble loss={loss:.6f} accuracy={accuracy:.6f} "
        f"exact={exact:.6f}",
        flush=True,
    )
    print(
        "position_accuracy="
        + ",".join(
            f"{float(value):.4f}"
            for value in correct.float().mean(dim=0)
        ),
        flush=True,
    )
    print(
        "position_loss="
        + ",".join(
            f"{float(value):.4f}"
            for value in position_loss
        ),
        flush=True,
    )
    print(
        "prefix_exact="
        + ",".join(
            f"{float(value):.4f}"
            for value in correct.cumprod(dim=1).float().mean(dim=0)
        ),
        flush=True,
    )
    print(
        "step_consistency="
        + ",".join(
            f"{float(value):.4f}"
            for value in step_consistency
        ),
        flush=True,
    )

    locations = trace["locations"]
    for relay in range(locations.shape[1]):
        entropy, effective, maximum, hard_fraction = _probability_summary(
            locations[:, relay]
        )
        ordered_mass = locations[:, relay].sort(
            dim=-1,
            descending=True,
        ).values
        retained_mass = ordered_mass.cumsum(-1).mean(
            dim=(0, 1, 2),
        )
        label = "initial" if relay == 0 else f"relay={relay}"
        print(
            f"nodes {label} entropy={entropy:.4f} "
            f"effective={effective:.3f}/4 max_mass={maximum:.3f} "
            f"hard90={hard_fraction:.3f} "
            f"mass_top1={float(retained_mass[0]):.4f} "
            f"mass_top2={float(retained_mass[1]):.4f} "
            f"mass_top3={float(retained_mass[2]):.4f}",
            flush=True,
        )
    for relay, entropy in enumerate(
        trace["transition_entropy"],
        start=1,
    ):
        print(
            f"transition relay={relay} expected_entropy={float(entropy):.4f} "
            f"effective_edges={math.exp(float(entropy)):.3f}",
            flush=True,
        )

    votes = trace["output_votes"]
    vote_entropy = float(
        -(votes.clamp_min(1e-9) * votes.clamp_min(1e-9).log()).sum()
    )
    print(
        f"votes entropy={vote_entropy:.4f} "
        f"effective_cells={math.exp(vote_entropy):.3f}/{model.n_cells} "
        f"min={float(votes.min()):.4f} max={float(votes.max()):.4f}",
        flush=True,
    )
    print(
        "votes values="
        + ",".join(f"{float(value):.4f}" for value in votes),
        flush=True,
    )
    print(
        "neighbor_acceptance left="
        + ",".join(
            f"{float(value):.3f}"
            for value in trace["neighbor_gates"][:, 0]
        ),
        flush=True,
    )
    print(
        "neighbor_acceptance right="
        + ",".join(
            f"{float(value):.3f}"
            for value in trace["neighbor_gates"][:, 1]
        ),
        flush=True,
    )

    cell_logits = trace["cell_logits"]
    rows: list[tuple[int, float, float, float, float]] = []
    for cell in range(model.n_cells):
        cell_loss, cell_accuracy, cell_exact = _metrics(
            cell_logits[:, cell],
            targets,
        )
        retained_weight = 1.0 - votes[cell]
        leave_one_out = (
            logits - votes[cell] * cell_logits[:, cell]
        ) / retained_weight.clamp_min(1e-9)
        leave_loss, leave_accuracy, _ = _metrics(
            leave_one_out,
            targets,
        )
        rows.append(
            (
                cell,
                cell_loss,
                cell_accuracy,
                cell_exact,
                leave_loss - loss,
            )
        )
        print(
            f"cell={cell} vote={float(votes[cell]):.4f} "
            f"loss={cell_loss:.4f} acc={cell_accuracy:.4f} "
            f"exact={cell_exact:.4f} "
            f"leave_out_loss_delta={leave_loss - loss:+.4f} "
            f"leave_out_acc={leave_accuracy:.4f}",
            flush=True,
        )

    best_cell = max(rows, key=lambda row: row[2])
    print(
        f"best_single_cell={best_cell[0]} "
        f"accuracy={best_cell[2]:.4f} exact={best_cell[3]:.4f} "
        f"ensemble_gain={accuracy - best_cell[2]:+.4f}",
        flush=True,
    )
    uniform_logits = cell_logits.mean(1)
    uniform_loss, uniform_accuracy, uniform_exact = _metrics(
        uniform_logits,
        targets,
    )
    print(
        f"uniform_ensemble loss={uniform_loss:.4f} "
        f"accuracy={uniform_accuracy:.4f} exact={uniform_exact:.4f}",
        flush=True,
    )
    vote_order = votes.argsort(descending=True)
    for retained_cells in (1, 2, 4):
        selected = vote_order[:retained_cells]
        selected_votes = votes[selected]
        selected_votes = selected_votes / selected_votes.sum()
        selected_logits = torch.einsum(
            "c,bctv->btv",
            selected_votes,
            cell_logits[:, selected],
        )
        selected_loss, selected_accuracy, selected_exact = _metrics(
            selected_logits,
            targets,
        )
        print(
            f"output_top_k={retained_cells} "
            f"cells={','.join(str(int(value)) for value in selected)} "
            f"loss={selected_loss:.4f} accuracy={selected_accuracy:.4f} "
            f"exact={selected_exact:.4f}",
            flush=True,
        )

    if args.leave_one_cell_out:
        for removed_cell in range(model.n_cells):
            retained = tuple(
                cell
                for cell in range(model.n_cells)
                if cell != removed_cell
            )
            with torch.no_grad():
                ablated_logits = model(
                    observed,
                    output_slots=args.length,
                    active_cells=retained,
                )
            ablated_loss, ablated_accuracy, ablated_exact = _metrics(
                ablated_logits,
                targets,
            )
            print(
                f"intermediate_ablation removed_cell={removed_cell} "
                f"loss={ablated_loss:.4f} accuracy={ablated_accuracy:.4f} "
                f"exact={ablated_exact:.4f} "
                f"accuracy_delta={ablated_accuracy - accuracy:+.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
