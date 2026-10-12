"""Compare fixed Gator connection graphs without training a model."""

from __future__ import annotations

import argparse
from collections import deque

import torch

from gator_convergent import (
    connection_destinations,
    graph_reachability,
)


def _shortest_paths(destinations: torch.Tensor, source: int) -> list[int]:
    distances = [-1] * destinations.shape[0]
    distances[source] = 0
    frontier = deque([source])
    while frontier:
        node = frontier.popleft()
        for destination in destinations[node].tolist():
            if distances[destination] < 0:
                distances[destination] = distances[node] + 1
                frontier.append(destination)
    return distances


def _pair_overlap(
    nodes: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    lane_sets = [
        set(nodes[lane][mask[lane]].tolist())
        for lane in range(nodes.shape[0])
    ]
    intersects = []
    jaccards = []
    for left in range(len(lane_sets)):
        for right in range(left + 1, len(lane_sets)):
            intersection = lane_sets[left] & lane_sets[right]
            union = lane_sets[left] | lane_sets[right]
            intersects.append(bool(intersection))
            jaccards.append(len(intersection) / max(len(union), 1))
    return (
        sum(intersects) / max(len(intersects), 1),
        sum(jaccards) / max(len(jaccards), 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report reachability of Gator connection layouts."
    )
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--lanes", type=int, default=10)
    parser.add_argument("--relays", type=int, default=4)
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("ring", "small_world", "dense"),
        default=("ring", "small_world"),
    )
    parser.add_argument("--connection-seeds", type=int, nargs="+", default=(0,))
    parser.add_argument(
        "--rewire-probabilities",
        type=float,
        nargs="+",
        default=(1.0,),
    )
    args = parser.parse_args()
    if args.lanes <= 0 or args.lanes > args.nodes:
        parser.error("--lanes must be in [1, nodes]")

    starts = torch.div(
        torch.arange(args.lanes) * args.nodes,
        args.lanes,
        rounding_mode="floor",
    )
    for layout in args.layouts:
        seeds = args.connection_seeds if layout == "small_world" else (0,)
        rewires = (
            args.rewire_probabilities
            if layout == "small_world"
            else (0.0,)
        )
        for seed in seeds:
            for rewire in rewires:
                degree = args.nodes if layout == "dense" else args.degree
                destinations = connection_destinations(
                    n_nodes=args.nodes,
                    degree=degree,
                    layout=layout,
                    seed=seed,
                    rewire_probability=rewire,
                )
                nodes, mask, _ = graph_reachability(
                    destinations,
                    starts,
                    args.relays,
                )
                all_distances = [
                    distance
                    for source in range(args.nodes)
                    for distance in _shortest_paths(destinations, source)
                ]
                reachable_distances = [
                    distance
                    for distance in all_distances
                    if distance >= 0
                ]
                print(
                    f"layout={layout} seed={seed} rewire={rewire:g} "
                    f"nodes={args.nodes} "
                    f"degree={degree} lanes={args.lanes} "
                    f"relays={args.relays} "
                    f"diameter={max(reachable_distances)} "
                    f"mean_distance="
                    f"{sum(reachable_distances) / len(reachable_distances):.3f} "
                    f"reachable_pairs="
                    f"{len(reachable_distances) / len(all_distances):.3f}"
                )
                for relay in range(args.relays + 1):
                    lane_counts = mask[relay].sum(dim=-1).float()
                    union = set()
                    for lane in range(args.lanes):
                        union.update(
                            nodes[relay, lane][
                                mask[relay, lane]
                            ].tolist()
                        )
                    collision_fraction, mean_jaccard = _pair_overlap(
                        nodes[relay],
                        mask[relay],
                    )
                    print(
                        f"  relay={relay} "
                        f"lane_reach_mean={float(lane_counts.mean()):.2f} "
                        f"lane_reach_min={int(lane_counts.min())} "
                        f"lane_reach_max={int(lane_counts.max())} "
                        f"population_coverage={len(union)}/{args.nodes} "
                        f"colliding_lane_pairs={collision_fraction:.3f} "
                        f"mean_lane_jaccard={mean_jaccard:.3f}"
                    )


if __name__ == "__main__":
    main()
