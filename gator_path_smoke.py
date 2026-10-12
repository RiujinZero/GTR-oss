"""Run one finite Gator-Path core forward/backward smoke check."""

from __future__ import annotations

import argparse
import math

import torch

from gator_path import (
    GatorPathCore,
    GatorPathRelay,
    PacketState,
    fixed_ring_topology,
)


def fixed_addresses(
    slots: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build a parameter-free coordinate code of shape ``[slots, width]``."""

    position = torch.linspace(
        0.0,
        1.0,
        slots,
        device=device,
    )
    channels = []
    for channel in range(width):
        frequency = 2.0 ** (channel // 2)
        angle = position * (math.pi * frequency)
        channels.append(
            torch.sin(angle)
            if channel % 2 == 0
            else torch.cos(angle)
        )
    return torch.stack(channels, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--cells", type=int, default=10)
    parser.add_argument("--payload-width", type=int, default=8)
    parser.add_argument("--address-width", type=int, default=4)
    parser.add_argument("--relays", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    if min(
        args.batch_size,
        args.slots,
        args.cells,
        args.payload_width,
        args.address_width,
        args.relays,
        args.top_k,
    ) <= 0:
        parser.error("all size and depth arguments must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")

    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    topology = fixed_ring_topology(args.cells)
    relay = GatorPathRelay(
        topology=topology,
        payload_width=args.payload_width,
        address_width=args.address_width,
        top_k=args.top_k,
        temperature=args.temperature,
        straight_through_hard=True,
    ).to(device)
    core = GatorPathCore(
        relay,
        relay_steps=args.relays,
    )

    payload = torch.randn(
        args.batch_size,
        args.slots,
        args.payload_width,
        device=device,
        requires_grad=True,
    )
    slot = torch.arange(args.slots, device=device)
    batch = torch.arange(args.batch_size, device=device).unsqueeze(-1)
    state = PacketState(
        payload=payload,
        address=fixed_addresses(
            args.slots,
            args.address_width,
            device=device,
        ),
        cell_id=(slot.unsqueeze(0) + batch).remainder(args.cells),
        node_id=(slot.unsqueeze(0) + batch).remainder(
            topology.num_nodes
        ),
        active=torch.ones(
            args.batch_size,
            args.slots,
            dtype=torch.bool,
            device=device,
        ),
        write_gate=torch.ones(
            args.batch_size,
            args.slots,
            device=device,
        ),
    )

    result = core(state)
    route_loss = torch.stack(
        [
            relay_result.route.weights.square().mean()
            for relay_result in result.relay_results
        ]
    ).mean()
    loss = result.state.payload.square().mean() + 0.01 * route_loss
    loss.backward()

    gradient_tensors = [
        parameter.grad
        for parameter in core.parameters()
        if parameter.grad is not None
    ]
    finite = (
        bool(torch.isfinite(result.state.payload).all())
        and payload.grad is not None
        and bool(torch.isfinite(payload.grad).all())
        and bool(gradient_tensors)
        and all(bool(torch.isfinite(gradient).all()) for gradient in gradient_tensors)
    )
    parameter_count = sum(
        parameter.numel() for parameter in core.parameters()
    )
    occupied = [
        relay_result.mailbox.occupied_count
        for relay_result in result.relay_results
    ]
    print(
        f"device={device.type} parameters={parameter_count} "
        f"payload_shape={tuple(result.state.payload.shape)} "
        f"relays={result.relay_steps} top_k={args.top_k} "
        f"occupied_rows={occupied} loss={float(loss.detach()):.6f} "
        f"finite={finite}"
    )
    if not finite:
        raise SystemExit("Gator-Path smoke check produced non-finite values")


if __name__ == "__main__":
    main()
