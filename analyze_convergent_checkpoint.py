"""Probe terminal-lane cooperation in a trained Convergent Gator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from gator_convergent import ConvergentGator
from parallel_prefix_benchmark import make_prefix_batch


def _metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float, float]:
    predictions = logits.argmax(dim=-1)
    accuracy = float(predictions.eq(targets).float().mean())
    exact = float(predictions.eq(targets).all(dim=-1).float().mean())
    loss = float(
        F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
    )
    return accuracy, exact, loss


def _fuse(
    lane_logits: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    weights = weights / weights.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)
    if weights.ndim == 1:
        return torch.einsum("k,btkv->btv", weights, lane_logits)
    if weights.ndim == 2:
        return torch.einsum("bk,btkv->btv", weights, lane_logits)
    if weights.ndim == 3:
        return torch.einsum("btk,btkv->btv", weights, lane_logits)
    raise ValueError("terminal weights must have one, two, or three axes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=91_337)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument(
        "--connection-layout",
        choices=("ring", "small_world", "dense"),
        default=None,
    )
    parser.add_argument("--connection-seed", type=int, default=None)
    parser.add_argument("--connection-rewire", type=float, default=None)
    parser.add_argument("--relays", type=int, default=None)
    parser.add_argument("--relay-code-span", type=int, default=None)
    parser.add_argument("--write-warmup-relays", type=int, default=None)
    parser.add_argument(
        "--bootstrap-context-mode",
        choices=("shared", "self"),
        default=None,
    )
    parser.add_argument("--lanes", type=int, default=None)
    parser.add_argument(
        "--route-top-k",
        type=int,
        default=None,
        help="renormalize every relay passport to this support for ablation",
    )
    parser.add_argument(
        "--route-top-k-start",
        type=int,
        default=0,
        help="zero-based relay where route-top-k pruning begins",
    )
    parser.add_argument(
        "--route-top-k-schedule",
        type=int,
        nargs="+",
        default=None,
        help=(
            "per-relay support; use 0 for an unpruned relay "
            "(mutually exclusive with --route-top-k)"
        ),
    )
    parser.add_argument(
        "--relay-limit",
        type=int,
        default=None,
        help=(
            "stop the trained computation after this many relays without "
            "changing its configured step codes or write schedule"
        ),
    )
    args = parser.parse_args()
    route_top_k_schedule = (
        tuple(
            None if value == 0 else value
            for value in args.route_top_k_schedule
        )
        if args.route_top_k_schedule is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    checkpoint_config = payload.get("model_config", {})
    model = ConvergentGator(
        n_nodes=(
            args.nodes
            if args.nodes is not None
            else int(checkpoint_config.get("n_nodes", 40))
        ),
        d_model=(
            args.width
            if args.width is not None
            else int(checkpoint_config.get("d_model", 8))
        ),
        degree=(
            args.degree
            if args.degree is not None
            else int(checkpoint_config.get("degree", 3))
        ),
        relays=(
            args.relays
            if args.relays is not None
            else int(checkpoint_config.get("relays", 4))
        ),
        lanes=(
            args.lanes
            if args.lanes is not None
            else int(checkpoint_config.get("lanes", 10))
        ),
        relay_code_span=(
            args.relay_code_span
            if args.relay_code_span is not None
            else checkpoint_config.get("relay_code_span")
        ),
        write_warmup_relays=(
            args.write_warmup_relays
            if args.write_warmup_relays is not None
            else checkpoint_config.get("write_warmup_relays")
        ),
        bootstrap_context_mode=(
            args.bootstrap_context_mode
            if args.bootstrap_context_mode is not None
            else str(
                checkpoint_config.get(
                    "bootstrap_context_mode",
                    "shared",
                )
            )
        ),
        connection_layout=(
            args.connection_layout
            if args.connection_layout is not None
            else str(checkpoint_config.get("connection_layout", "ring"))
        ),
        connection_seed=(
            args.connection_seed
            if args.connection_seed is not None
            else int(checkpoint_config.get("connection_seed", 0))
        ),
        connection_rewire=(
            args.connection_rewire
            if args.connection_rewire is not None
            else float(checkpoint_config.get("connection_rewire", 1.0))
        ),
        terminal_vote_mode=str(
            checkpoint_config.get("terminal_vote_mode", "global")
        ),
    ).to(device)
    model.load_state_dict(payload.get("model", payload))
    model.eval()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    observed, targets = make_prefix_batch(
        args.batch_size,
        7,
        device,
        generator=generator,
        sampling_device=torch.device("cpu"),
    )
    with torch.no_grad():
        logits, trace = model(
            observed,
            output_slots=7,
            route_top_k=args.route_top_k,
            route_top_k_start=args.route_top_k_start,
            route_top_k_schedule=route_top_k_schedule,
            relay_limit=args.relay_limit,
            return_trace=True,
        )

    lane_logits = trace["lane_logits"]
    terminal_weights = trace["terminal_weights"]
    summary_weights = terminal_weights.mean(
        dim=tuple(range(terminal_weights.ndim - 1)),
    )
    ranking = summary_weights.argsort(descending=True)
    accuracy, exact, loss = _metrics(logits, targets)
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
        f"step={int(payload.get('step', -1))} "
        f"nodes={model.n_nodes} width={model.d_model} "
        f"relays={model.relays} lanes={model.lanes} "
        f"relay_code_span={model.relay_code_span} "
        f"write_warmup_relays={model.write_warmup_relays} "
        f"bootstrap_context_mode={model.bootstrap_context_mode} "
        f"connection_layout={model.connection_layout} "
        f"connection_seed={model.connection_seed} "
        f"connection_rewire={model.connection_rewire:g} "
        f"terminal_vote_mode={model.terminal_vote_mode} "
        f"examples={args.batch_size} "
        f"route_top_k={args.route_top_k} "
        f"route_top_k_start={args.route_top_k_start} "
        f"route_top_k_schedule={route_top_k_schedule} "
        f"relay_limit={args.relay_limit} "
        f"acc={accuracy:.4f} exact={exact:.4f} loss={loss:.6f}"
    )
    print(
        "position_accuracy="
        + ",".join(
            f"{float(value):.4f}"
            for value in correct.float().mean(dim=0)
        )
    )
    print(
        "position_loss="
        + ",".join(
            f"{float(value):.4f}"
            for value in position_loss
        )
    )
    print(
        "prefix_exact="
        + ",".join(
            f"{float(value):.4f}"
            for value in correct.cumprod(dim=1).float().mean(dim=0)
        )
    )
    print(
        "step_consistency="
        + ",".join(
            f"{float(value):.4f}"
            for value in step_consistency
        )
    )
    print(
        "terminal_weights="
        + ",".join(
            f"{index}:{float(summary_weights[index]):.5f}"
            for index in ranking.tolist()
        )
    )

    for count in range(1, model.lanes + 1):
        selected = ranking[:count]
        lane_mask = torch.zeros(
            model.lanes,
            device=terminal_weights.device,
            dtype=terminal_weights.dtype,
        )
        lane_mask[selected] = 1
        weights = terminal_weights * lane_mask
        selected_metrics = _metrics(
            _fuse(lane_logits, weights),
            targets,
        )
        print(
            f"top_lanes={count} "
            f"acc={selected_metrics[0]:.4f} "
            f"exact={selected_metrics[1]:.4f} "
            f"loss={selected_metrics[2]:.6f}"
        )

    for lane in range(model.lanes):
        lane_mask = torch.ones(
            model.lanes,
            device=terminal_weights.device,
            dtype=terminal_weights.dtype,
        )
        lane_mask[lane] = 0
        weights = terminal_weights * lane_mask
        ablated_metrics = _metrics(
            _fuse(lane_logits, weights),
            targets,
        )
        single_mask = torch.zeros_like(lane_mask)
        single_mask[lane] = 1
        single_weights = terminal_weights * single_mask
        single_metrics = _metrics(
            _fuse(lane_logits, single_weights),
            targets,
        )
        print(
            f"lane={lane} weight={float(summary_weights[lane]):.5f} "
            f"alone_acc={single_metrics[0]:.4f} "
            f"alone_exact={single_metrics[1]:.4f} "
            f"without_acc={ablated_metrics[0]:.4f} "
            f"without_exact={ablated_metrics[1]:.4f} "
            f"without_loss={ablated_metrics[2]:.6f}"
        )

    locations = trace["locations"]
    for relay in range(locations.size(1)):
        q = locations[:, relay]
        entropy = -(
            q.clamp_min(1e-9) * q.clamp_min(1e-9).log()
        ).sum(dim=-1)
        node_mass = q.sum(dim=2)
        pair_overlap = 0.5 * (
            node_mass.square()
            - q.square().sum(dim=2)
        ).sum(dim=-1).mean()
        print(
            f"relay={relay} "
            f"node_effective={float(entropy.mean().exp()):.4f} "
            f"node_peak={float(q.amax(dim=-1).mean()):.4f} "
            f"lane_pair_overlap={float(pair_overlap):.6f}"
        )


if __name__ == "__main__":
    main()
