"""Developmental 4/4/2 regional training for the 10-cell Gator-Path.

Cells 0-3 first learn raw-token to canonical-number conversion. Cells 4-7
learn modular addition. Cells 8-9 remain an untrained orchestration region
until the parallel-prefix integration stage opens bridges between regions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import torch
from torch import Tensor
from torch.nn import functional as F

from gator_path import GatorPathPrefixConfig, GatorPathPrefixModel
from gator_path.topology import fixed_ring_topology
from parallel_prefix_benchmark import make_prefix_batch


NUMBER_CELLS = tuple(range(0, 4))
ARITHMETIC_CELLS = tuple(range(4, 8))
ORCHESTRATOR_CELLS = tuple(range(8, 10))


@dataclass(frozen=True)
class RegionalLayout:
    number: tuple[int, ...]
    arithmetic: tuple[int, ...]
    orchestrator: tuple[int, ...]

    @classmethod
    def from_sizes(
        cls,
        *,
        number_cells: int = 4,
        orchestrator_cells: int = 2,
    ) -> "RegionalLayout":
        if number_cells < 1:
            raise ValueError("number region needs at least one cell")
        arithmetic_cells = 10 - number_cells - orchestrator_cells
        if arithmetic_cells < 3:
            raise ValueError("arithmetic region needs at least three cells")
        number = tuple(range(number_cells))
        arithmetic = tuple(
            range(number_cells, number_cells + arithmetic_cells)
        )
        orchestrator = tuple(
            range(number_cells + arithmetic_cells, 10)
        )
        return cls(number, arithmetic, orchestrator)


DEFAULT_LAYOUT = RegionalLayout(
    NUMBER_CELLS,
    ARITHMETIC_CELLS,
    ORCHESTRATOR_CELLS,
)


class SplitRegionalOptimizer:
    """SGD route heads plus AdamW representation parameters."""

    def __init__(
        self,
        *,
        router_parameters: list[Tensor],
        transformation_parameters: list[Tensor],
        router_learning_rate: float,
        transformation_learning_rate: float,
        weight_decay: float,
    ) -> None:
        if not router_parameters:
            raise ValueError("split optimizer needs router parameters")
        if not transformation_parameters:
            raise ValueError(
                "split optimizer needs transformation parameters"
            )
        self.router_optimizer = torch.optim.SGD(
            router_parameters,
            lr=router_learning_rate,
            momentum=0.0,
            weight_decay=0.0,
        )
        self.transformation_optimizer = torch.optim.AdamW(
            transformation_parameters,
            lr=transformation_learning_rate,
            weight_decay=weight_decay,
        )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.router_optimizer.zero_grad(set_to_none=set_to_none)
        self.transformation_optimizer.zero_grad(
            set_to_none=set_to_none
        )

    def step(self) -> None:
        self.router_optimizer.step()
        self.transformation_optimizer.step()

    def clear_state(self) -> None:
        self.router_optimizer.state.clear()
        self.transformation_optimizer.state.clear()

    @property
    def router_learning_rate(self) -> float:
        return float(self.router_optimizer.param_groups[0]["lr"])

    def set_router_learning_rate(self, learning_rate: float) -> None:
        for group in self.router_optimizer.param_groups:
            group["lr"] = learning_rate

    def state_dict(self) -> dict[str, object]:
        return {
            "mode": "split_sgd_adamw",
            "router": self.router_optimizer.state_dict(),
            "transformation": self.transformation_optimizer.state_dict(),
        }


OptimizerLike = torch.optim.Optimizer | SplitRegionalOptimizer


@dataclass(frozen=True)
class OwnershipRouterSchedule:
    """Monotonically anneal route SGD as committed ownership rises."""

    maximum_learning_rate: float
    minimum_learning_rate: float
    start_ownership: float = 0.5
    target_ownership: float = 0.8

    def __post_init__(self) -> None:
        values = (
            self.maximum_learning_rate,
            self.minimum_learning_rate,
            self.start_ownership,
            self.target_ownership,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ownership router schedule values must be finite")
        if self.maximum_learning_rate <= 0:
            raise ValueError("maximum router learning rate must be positive")
        if not 0 < self.minimum_learning_rate <= self.maximum_learning_rate:
            raise ValueError(
                "minimum router learning rate must be positive and no "
                "greater than the maximum"
            )
        if not 0 <= self.start_ownership < self.target_ownership <= 1:
            raise ValueError(
                "ownership thresholds must satisfy "
                "0 <= start < target <= 1"
            )

    def learning_rate(self, ownership: float) -> float:
        """Return log-linearly interpolated LR for an ownership fraction."""

        progress = (
            (float(ownership) - self.start_ownership)
            / (self.target_ownership - self.start_ownership)
        )
        progress = min(max(progress, 0.0), 1.0)
        ratio = (
            self.minimum_learning_rate / self.maximum_learning_rate
        )
        return self.maximum_learning_rate * ratio**progress


def build_optimizer(
    model: GatorPathPrefixModel,
    *,
    learning_rate: float,
    weight_decay: float,
    mode: str,
    router_learning_rate: float | None = None,
) -> OptimizerLike:
    """Construct one global AdamW or split route/representation optimizers."""

    if mode == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if mode != "split":
        raise ValueError("optimizer mode must be 'adamw' or 'split'")

    router_parameters: list[Tensor] = []
    transformation_parameters: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if "cell_router_" in name or "node_router_" in name:
            router_parameters.append(parameter)
        else:
            transformation_parameters.append(parameter)
    return SplitRegionalOptimizer(
        router_parameters=router_parameters,
        transformation_parameters=transformation_parameters,
        router_learning_rate=(
            learning_rate
            if router_learning_rate is None
            else router_learning_rate
        ),
        transformation_learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def clear_optimizer_state(optimizer: OptimizerLike) -> None:
    if isinstance(optimizer, SplitRegionalOptimizer):
        optimizer.clear_state()
    else:
        optimizer.state.clear()


def apply_ownership_router_schedule(
    optimizer: OptimizerLike,
    schedule: OwnershipRouterSchedule | None,
    *,
    maximum_observed_ownership: float,
) -> float | None:
    if schedule is None:
        return (
            optimizer.router_learning_rate
            if isinstance(optimizer, SplitRegionalOptimizer)
            else None
        )
    if not isinstance(optimizer, SplitRegionalOptimizer):
        raise TypeError(
            "ownership router schedule requires SplitRegionalOptimizer"
        )
    learning_rate = schedule.learning_rate(maximum_observed_ownership)
    optimizer.set_router_learning_rate(learning_rate)
    return learning_rate


def regional_access_mask(
    device: torch.device,
    *,
    bridges_open: bool,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> Tensor:
    """Return isolated regions or orchestrator-mediated regional access."""

    mask = torch.zeros(10, 10, dtype=torch.bool, device=device)
    regions = (layout.number, layout.arithmetic, layout.orchestrator)
    for region in regions:
        rows = torch.tensor(region, device=device)
        mask[rows[:, None], rows[None, :]] = True
    if bridges_open:
        orchestrator = torch.tensor(layout.orchestrator, device=device)
        for skill_region in (layout.number, layout.arithmetic):
            skill = torch.tensor(skill_region, device=device)
            mask[skill[:, None], orchestrator[None, :]] = True
            mask[orchestrator[:, None], skill[None, :]] = True
    return mask


def regional_skill_access_mask(
    device: torch.device,
    *,
    skill: str,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> Tensor:
    """Open the orchestrator bridge to exactly one active skill region."""

    if skill not in {"number", "arithmetic"}:
        raise ValueError("skill must be 'number' or 'arithmetic'")
    mask = regional_access_mask(
        device, bridges_open=False, layout=layout
    )
    skill_region = (
        layout.number if skill == "number" else layout.arithmetic
    )
    skill_cells = torch.tensor(skill_region, device=device)
    orchestrator = torch.tensor(layout.orchestrator, device=device)
    mask[skill_cells[:, None], orchestrator[None, :]] = True
    mask[orchestrator[:, None], skill_cells[None, :]] = True
    return mask


def learned_dispatch_access_mask(
    device: torch.device,
    *,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> Tensor:
    """Enforce ownership law while leaving every call decision learned.

    Orchestrator cells may call either specialist. A specialist may work
    internally or return to the orchestrator, but can never transfer directly
    to the other specialist.
    """

    return regional_access_mask(
        device,
        bridges_open=True,
        layout=layout,
    )


def regional_gate_destination_table(
    device: torch.device,
    *,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> Tensor:
    """Give each specialist region one size-independent public entrance.

    Orchestrator action 3 calls the number gate and action 4 calls the
    arithmetic gate.  No other orchestrator action enters a specialist
    region, so region size cannot increase its prior routing probability.

    Specialist actions 0-3 remain local whenever their original ring
    destination stays inside the region.  Boundary-crossing actions become
    stays.  Action 4 is a direct return to an orchestrator cell from every
    specialist cell.
    """

    topology = fixed_ring_topology(10)
    if topology.num_cell_actions != 5:
        raise ValueError("gate dispatch requires five cell actions")
    if not layout.orchestrator:
        raise ValueError("gate dispatch requires an orchestrator region")

    table = torch.tensor(
        topology.cell_destination_table,
        dtype=torch.long,
        device=device,
    )
    for region in (layout.number, layout.arithmetic):
        region_set = set(region)
        for local_index, source in enumerate(region):
            for action in range(4):
                if int(table[source, action]) not in region_set:
                    table[source, action] = source
            table[source, 4] = layout.orchestrator[
                local_index % len(layout.orchestrator)
            ]

    number_gate = layout.number[-1]
    arithmetic_gate = layout.arithmetic[-1]
    for local_index, source in enumerate(layout.orchestrator):
        other = layout.orchestrator[
            (local_index + 1) % len(layout.orchestrator)
        ]
        table[source] = torch.tensor(
            (source, other, source, number_gate, arithmetic_gate),
            dtype=torch.long,
            device=device,
        )
    return table


def _identity_access(device: torch.device) -> Tensor:
    return torch.eye(10, dtype=torch.bool, device=device)


def regional_dispatch_mask(
    device: torch.device,
    *,
    skill: str,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> Tensor:
    """Force orchestrator-owned packets outward to one specialist."""

    if skill not in {"number", "arithmetic"}:
        raise ValueError("skill must be 'number' or 'arithmetic'")
    skill_region = (
        layout.number if skill == "number" else layout.arithmetic
    )
    mask = _identity_access(device)
    orchestrator = torch.tensor(layout.orchestrator, device=device)
    specialist = torch.tensor(skill_region, device=device)
    mask[orchestrator] = False
    mask[orchestrator[:, None], specialist[None, :]] = True
    return mask


def regional_compute_mask(
    device: torch.device,
    *,
    skill: str,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> Tensor:
    """Keep specialist work local while the orchestrator waits."""

    if skill not in {"number", "arithmetic"}:
        raise ValueError("skill must be 'number' or 'arithmetic'")
    skill_region = (
        layout.number if skill == "number" else layout.arithmetic
    )
    mask = _identity_access(device)
    specialist = torch.tensor(skill_region, device=device)
    orchestrator = torch.tensor(layout.orchestrator, device=device)
    mask[specialist[:, None], specialist[None, :]] = True
    mask[orchestrator[:, None], orchestrator[None, :]] = True
    return mask


def regional_return_mask(
    device: torch.device,
    *,
    skill: str,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> Tensor:
    """Permit only specialist moves that approach the orchestrator."""

    if skill not in {"number", "arithmetic"}:
        raise ValueError("skill must be 'number' or 'arithmetic'")
    skill_region = (
        layout.number if skill == "number" else layout.arithmetic
    )
    mask = _identity_access(device)
    orchestrator = tuple(layout.orchestrator)

    def distance(cell: int) -> int:
        return min(
            min((cell - target) % 10, (target - cell) % 10)
            for target in orchestrator
        )

    legal_return_destinations = set(skill_region) | set(orchestrator)
    for source in skill_region:
        mask[source] = False
        for destination in legal_return_destinations:
            if distance(destination) < distance(source):
                mask[source, destination] = True
    orchestrator_tensor = torch.tensor(orchestrator, device=device)
    mask[
        orchestrator_tensor[:, None],
        orchestrator_tensor[None, :],
    ] = True
    return mask


def orchestrator_access_mask(
    device: torch.device,
    *,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> Tensor:
    """Hold returned packets inside the orchestrator workspace."""

    mask = _identity_access(device)
    orchestrator = torch.tensor(layout.orchestrator, device=device)
    mask[orchestrator[:, None], orchestrator[None, :]] = True
    return mask


def regional_call_return_schedule(
    device: torch.device,
    *,
    skill: str,
    relay_steps: int,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> Tensor:
    """Dispatch, compute, return, and consolidate through the orchestrator."""

    if relay_steps < 4:
        raise ValueError("call/return training requires at least four relays")
    dispatch = regional_dispatch_mask(
        device, skill=skill, layout=layout
    )
    compute = regional_compute_mask(
        device, skill=skill, layout=layout
    )
    returning = regional_return_mask(
        device, skill=skill, layout=layout
    )
    consolidate = orchestrator_access_mask(device, layout=layout)
    return torch.stack(
        (dispatch, compute)
        + (returning,) * (relay_steps - 3)
        + (consolidate,)
    )


def regional_composition_schedule(
    device: torch.device,
    *,
    relay_steps: int,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> Tensor:
    """Call number, return, call arithmetic, return, then consolidate."""

    if relay_steps < 10:
        raise ValueError("regional composition requires at least ten relays")
    number_dispatch = regional_dispatch_mask(
        device, skill="number", layout=layout
    )
    number_compute = regional_compute_mask(
        device, skill="number", layout=layout
    )
    number_return = regional_return_mask(
        device, skill="number", layout=layout
    )
    orchestrate = orchestrator_access_mask(device, layout=layout)
    arithmetic_dispatch = regional_dispatch_mask(
        device, skill="arithmetic", layout=layout
    )
    arithmetic_compute = regional_compute_mask(
        device, skill="arithmetic", layout=layout
    )
    arithmetic_return = regional_return_mask(
        device, skill="arithmetic", layout=layout
    )
    return_windows = relay_steps - 7
    return torch.stack(
        (
            number_dispatch,
            number_compute,
            number_return,
            orchestrate,
            arithmetic_dispatch,
            arithmetic_compute,
        )
        + (arithmetic_return,) * return_windows
        + (orchestrate,)
    )


def number_batch(
    batch_size: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map a raw delta token ``d`` to canonical number token ``d + 4``."""

    raw = torch.randint(
        0, 12, (batch_size, 1), device=device, generator=generator
    )
    target = raw + 4
    locations = torch.tensor(
        (layout.number[0], layout.number[-1]),
        device=device,
        dtype=torch.long,
    ).expand(batch_size, -1)
    return raw, target, locations


def arithmetic_batch(
    batch_size: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> tuple[Tensor, Tensor, Tensor]:
    """Add a raw delta to a canonical number modulo twelve."""

    value = torch.randint(
        0, 12, (batch_size, 1), device=device, generator=generator
    )
    delta = torch.randint(
        0, 12, (batch_size, 1), device=device, generator=generator
    )
    observed = torch.cat((value + 4, delta), dim=1)
    target = 4 + (value + delta).remainder(12)
    locations = torch.tensor(
        (
            layout.arithmetic[-2],
            layout.arithmetic[-1],
            layout.arithmetic[-2],
        ),
        device=device,
        dtype=torch.long,
    ).expand(batch_size, -1)
    return observed, target, locations


def prefix_batch(
    batch_size: int,
    length: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
    layout: RegionalLayout = DEFAULT_LAYOUT,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create prefix data with inputs in number cells and queries in cortex."""

    observed, target = make_prefix_batch(
        batch_size, length, device, generator=generator
    )
    input_cells = torch.tensor(
        (layout.number[0],)
        + tuple(
            layout.number[index % len(layout.number)]
            for index in range(length)
        ),
        device=device,
        dtype=torch.long,
    )
    output_cells = torch.tensor(
        tuple(
            layout.orchestrator[index % len(layout.orchestrator)]
            for index in range(length)
        ),
        device=device,
        dtype=torch.long,
    )
    locations = torch.cat((input_cells, output_cells)).expand(
        batch_size, -1
    )
    return observed, target, locations


def support_width(step: int, total_steps: int, maximum: int = 6) -> int:
    """Anneal broad local exploration to the deployed top-2 support."""

    if total_steps <= 1:
        return 2
    progress = step / (total_steps - 1)
    return max(2, round(maximum - (maximum - 2) * progress))


def orchestrator_return_loss(
    diagnostics,
    *,
    layout: RegionalLayout,
    return_relays: int = 2,
) -> Tensor:
    """Encourage packets to finish near the orchestrator without scripting calls."""

    relay_results = diagnostics.relay_results
    if return_relays <= 0:
        raise ValueError("return_relays must be positive")
    selected = relay_results[-min(return_relays, len(relay_results)) :]
    device = selected[0].route.weights.device
    dtype = selected[0].route.weights.dtype
    distance = torch.empty(10, device=device, dtype=dtype)
    for cell in range(10):
        distance[cell] = min(
            min((cell - target) % 10, (target - cell) % 10)
            for target in layout.orchestrator
        )
    maximum_distance = distance.max().clamp_min(1.0)
    penalties = []
    for relay in selected:
        destination_distance = distance[
            relay.route.destination_cell
        ]
        expected_distance = (
            relay.route.weights * destination_distance
        ).sum(dim=-1)
        penalties.append(expected_distance.mean() / maximum_distance)
    return torch.stack(penalties).mean()


def retain_router_only_gradients(
    model: GatorPathPrefixModel,
    *,
    layout: RegionalLayout | None = None,
) -> None:
    """Freeze specialists while adapting routing and orchestrator transforms."""

    trainable_names = {
        "relay.cell_router_weight",
        "relay.cell_router_bias",
        "relay.node_router_weight",
        "relay.node_router_bias",
    }
    for name, parameter in model.named_parameters():
        if name in trainable_names:
            continue
        if layout is not None and name in {
            "relay.path_transform.weight",
            "relay.path_transform.bias",
            "relay.destination_norm_weight",
            "relay.destination_norm_bias",
        }:
            row_mask = torch.zeros(
                parameter.shape[0],
                device=parameter.device,
                dtype=parameter.dtype,
            )
            row_mask[list(layout.orchestrator)] = 1.0
            view_shape = (parameter.shape[0],) + (1,) * (
                parameter.ndim - 1
            )
            parameter.grad.mul_(row_mask.view(view_shape))
            continue
        if name not in trainable_names:
            parameter.grad = None


def should_train_skills(
    resume_skill_checkpoint: Path | None,
    *,
    rehearse_loaded_skills: bool,
) -> bool:
    """Return whether the developmental skill stage should execute."""

    return (
        resume_skill_checkpoint is None
        or rehearse_loaded_skills
    )


def _task_batch(
    task: str,
    batch_size: int,
    length: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    layout: RegionalLayout = DEFAULT_LAYOUT,
    orchestrator_starts: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    if task == "number":
        observed, target, locations = number_batch(
            batch_size, device, generator=generator, layout=layout
        )
    elif task == "arithmetic":
        observed, target, locations = arithmetic_batch(
            batch_size, device, generator=generator, layout=layout
        )
    elif task == "prefix":
        observed, target, locations = prefix_batch(
            batch_size,
            length,
            device,
            generator=generator,
            layout=layout,
        )
    else:
        raise ValueError(f"unknown task: {task}")
    if orchestrator_starts:
        slots = observed.shape[1] + target.shape[1]
        orchestrator = torch.tensor(
            tuple(
                layout.orchestrator[index % len(layout.orchestrator)]
                for index in range(slots)
            ),
            device=device,
            dtype=torch.long,
        )
        locations = orchestrator.expand(batch_size, -1)
    return observed, target, locations


@torch.no_grad()
def evaluate(
    model: GatorPathPrefixModel,
    *,
    task: str,
    batch_size: int,
    length: int,
    access_mask: Tensor,
    seed: int,
    layout: RegionalLayout = DEFAULT_LAYOUT,
    orchestrator_starts: bool = False,
    cell_destination_table: Tensor | None = None,
) -> dict[str, float]:
    device = model.token_embedding.weight.device
    generator = torch.Generator(device=device).manual_seed(seed)
    observed, target, initial_cells = _task_batch(
        task,
        batch_size,
        length,
        device,
        generator,
        layout,
        orchestrator_starts,
    )
    logits, diagnostics = model(
        observed,
        output_slots=target.shape[1],
        temperature=0.5,
        route_top_k=2,
        initial_cell_id=initial_cells,
        cell_access_mask=access_mask,
        cell_destination_table=cell_destination_table,
        return_diagnostics=True,
    )
    predicted = logits.argmax(dim=-1)
    correct = predicted.eq(target)
    cross_region = 0.0
    transitions = 0
    number_calls = 0.0
    arithmetic_calls = 0.0
    specialist_returns = 0.0
    specialist_shortcuts = 0.0
    for relay in diagnostics.relay_results:
        source = relay.source_cell
        destination = relay.route.destination_cell[..., 0]
        arithmetic_start = len(layout.number)
        orchestrator_start = arithmetic_start + len(layout.arithmetic)
        source_region = (
            source.ge(arithmetic_start).long()
            + source.ge(orchestrator_start).long()
        )
        destination_region = (
            destination.ge(arithmetic_start).long()
            + destination.ge(orchestrator_start).long()
        )
        cross_region += float(
            source_region.ne(destination_region).float().mean()
        )
        number_calls += float(
            ((source_region == 2) & (destination_region == 0)).sum()
        )
        arithmetic_calls += float(
            ((source_region == 2) & (destination_region == 1)).sum()
        )
        specialist_returns += float(
            (
                (source_region < 2)
                & (destination_region == 2)
            ).sum()
        )
        specialist_shortcuts += float(
            (
                (source_region < 2)
                & (destination_region < 2)
                & source_region.ne(destination_region)
            ).sum()
        )
        transitions += 1
    total_calls = number_calls + arithmetic_calls
    final_orchestrator = float(
        diagnostics.state.cell_id.ge(
            min(layout.orchestrator)
        ).float().mean()
    )
    return {
        "loss": float(
            F.cross_entropy(
                logits.reshape(-1, 16), target.reshape(-1)
            )
        ),
        "accuracy": float(correct.float().mean()),
        "exact": float(correct.all(dim=1).float().mean()),
        "correct": float(correct.sum(dim=1).float().mean()),
        "cross_region": cross_region / max(transitions, 1),
        "number_call_share": number_calls / max(total_calls, 1.0),
        "arithmetic_call_share": arithmetic_calls / max(total_calls, 1.0),
        "specialist_returns": specialist_returns,
        "specialist_shortcuts": specialist_shortcuts,
        "final_orchestrator": final_orchestrator,
    }


def train_task(
    model: GatorPathPrefixModel,
    optimizer: OptimizerLike,
    *,
    task: str,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    length: int,
    access_mask: Tensor,
    log_every: int,
    seed: int,
    global_step: int,
    layout: RegionalLayout,
    maximum_support: int,
    orchestrator_starts: bool = False,
    return_loss_weight: float = 0.0,
    return_relays: int = 2,
    cell_destination_table: Tensor | None = None,
    nonfinite_restarts: int = 0,
    ownership_router_schedule: OwnershipRouterSchedule | None = None,
) -> tuple[int, dict[str, float]]:
    device = model.token_embedding.weight.device
    generator = torch.Generator(device=device).manual_seed(seed)
    started = time.perf_counter()
    best_metrics = evaluate(
        model,
        task=task,
        batch_size=eval_batch_size,
        length=length,
        access_mask=access_mask,
        seed=seed + 100_000,
        layout=layout,
        orchestrator_starts=orchestrator_starts,
        cell_destination_table=cell_destination_table,
    )
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    maximum_observed_ownership = best_metrics["final_orchestrator"]
    current_router_learning_rate = apply_ownership_router_schedule(
        optimizer,
        ownership_router_schedule,
        maximum_observed_ownership=maximum_observed_ownership,
    )
    completed_steps = 0
    restart_count = 0
    for local_step in range(steps):
        observed, target, initial_cells = _task_batch(
            task,
            batch_size,
            length,
            device,
            generator,
            layout,
            orchestrator_starts,
        )
        route_top_k = support_width(
            local_step, steps, maximum=maximum_support
        )
        temperature = max(0.5, 1.25 * (0.998 ** local_step))
        logits, diagnostics = model(
            observed,
            output_slots=target.shape[1],
            temperature=temperature,
            route_top_k=route_top_k,
            initial_cell_id=initial_cells,
            cell_access_mask=access_mask,
            cell_destination_table=cell_destination_table,
            return_diagnostics=True,
        )
        task_loss = F.cross_entropy(
            logits.reshape(-1, 16), target.reshape(-1)
        )
        return_loss = orchestrator_return_loss(
            diagnostics,
            layout=layout,
            return_relays=return_relays,
        )
        loss = task_loss + return_loss_weight * return_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=False
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            model.load_state_dict(best_state)
            clear_optimizer_state(optimizer)
            optimizer.zero_grad(set_to_none=True)
            if restart_count < nonfinite_restarts:
                restart_count += 1
                print(
                    f"stage={task} recovered_nonfinite "
                    f"step={local_step + 1} "
                    f"restart={restart_count}/{nonfinite_restarts} "
                    "restored_best=True optimizer_reset=True",
                    flush=True,
                )
                continue
            print(
                f"stage={task} stopped_nonfinite step={local_step + 1} "
                "restoring_best=True",
                flush=True,
            )
            break
        optimizer.step()
        completed = local_step + 1
        completed_steps = completed
        if completed % log_every and completed != steps:
            continue
        metrics = evaluate(
            model,
            task=task,
            batch_size=eval_batch_size,
            length=length,
            access_mask=access_mask,
            seed=seed + 100_000,
            layout=layout,
            orchestrator_starts=orchestrator_starts,
            cell_destination_table=cell_destination_table,
        )
        maximum_observed_ownership = max(
            maximum_observed_ownership,
            metrics["final_orchestrator"],
        )
        current_router_learning_rate = apply_ownership_router_schedule(
            optimizer,
            ownership_router_schedule,
            maximum_observed_ownership=maximum_observed_ownership,
        )
        if (
            metrics["accuracy"] > best_metrics["accuracy"]
            or (
                metrics["accuracy"] == best_metrics["accuracy"]
                and metrics["loss"] < best_metrics["loss"]
            )
        ):
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        elapsed = time.perf_counter() - started
        print(
            f"stage={task} step={completed}/{steps} "
            f"global_step={global_step + completed} "
            f"steps_per_second={completed / elapsed:.2f} "
            f"top_k={route_top_k} loss={metrics['loss']:.4f} "
            f"acc={metrics['accuracy']:.3f} "
            f"exact={metrics['exact']:.3f} "
            f"cross_region={metrics['cross_region']:.3f} "
            f"final_orchestrator={metrics['final_orchestrator']:.3f} "
            f"max_ownership={maximum_observed_ownership:.3f} "
            f"router_lr={current_router_learning_rate} "
            f"return_loss={float(return_loss.detach()):.4f} "
            f"grad_norm={float(gradient_norm):.4f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    print(
        f"stage={task} restored_best "
        f"acc={best_metrics['accuracy']:.3f} "
        f"loss={best_metrics['loss']:.4f}",
        flush=True,
    )
    return global_step + completed_steps, best_metrics


def train_interleaved_skills(
    model: GatorPathPrefixModel,
    optimizer: OptimizerLike,
    *,
    number_steps: int,
    arithmetic_steps: int,
    batch_size: int,
    eval_batch_size: int,
    length: int,
    number_access_mask: Tensor,
    arithmetic_access_mask: Tensor,
    log_every: int,
    seed: int,
    global_step: int,
    layout: RegionalLayout,
    number_maximum_support: int,
    arithmetic_maximum_support: int,
    return_loss_weight: float = 0.0,
    return_relays: int = 2,
    router_only: bool = False,
    orchestrator_starts: bool = True,
    cell_destination_table: Tensor | None = None,
    nonfinite_restarts: int = 0,
    ownership_router_schedule: OwnershipRouterSchedule | None = None,
) -> tuple[int, dict[str, dict[str, float]]]:
    """Rehearse both skills so shared dispatch does not overwrite either."""

    device = model.token_embedding.weight.device
    number_generator = torch.Generator(device=device).manual_seed(seed + 11)
    arithmetic_generator = torch.Generator(device=device).manual_seed(
        seed + 29
    )
    task_order = torch.cat(
        (
            torch.zeros(number_steps, dtype=torch.long),
            torch.ones(arithmetic_steps, dtype=torch.long),
        )
    )
    shuffle_generator = torch.Generator().manual_seed(seed + 71)
    task_order = task_order[
        torch.randperm(task_order.numel(), generator=shuffle_generator)
    ].tolist()
    total_steps = len(task_order)
    started = time.perf_counter()

    def skill_metrics() -> dict[str, dict[str, float]]:
        return {
            task: evaluate(
                model,
                task=task,
                batch_size=eval_batch_size,
                length=length,
                access_mask=(
                    number_access_mask
                    if task == "number"
                    else arithmetic_access_mask
                ),
                seed=seed + eval_seed,
                layout=layout,
                orchestrator_starts=orchestrator_starts,
                cell_destination_table=cell_destination_table,
            )
            for task, eval_seed in (
                ("number", 100_011),
                ("arithmetic", 100_029),
            )
        }

    def score(metrics: dict[str, dict[str, float]]) -> float:
        number_accuracy = metrics["number"]["accuracy"]
        arithmetic_accuracy = metrics["arithmetic"]["accuracy"]
        return (
            2.0
            * number_accuracy
            * arithmetic_accuracy
            / max(number_accuracy + arithmetic_accuracy, 1.0e-12)
        )

    best_metrics = skill_metrics()
    best_score = score(best_metrics)
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    maximum_observed_ownership = min(
        best_metrics["number"]["final_orchestrator"],
        best_metrics["arithmetic"]["final_orchestrator"],
    )
    current_router_learning_rate = apply_ownership_router_schedule(
        optimizer,
        ownership_router_schedule,
        maximum_observed_ownership=maximum_observed_ownership,
    )
    completed_steps = 0
    latest_gradient_norm = torch.tensor(float("nan"), device=device)
    restart_count = 0
    for local_step, task_id in enumerate(task_order):
        task = "number" if task_id == 0 else "arithmetic"
        generator = (
            number_generator if task_id == 0 else arithmetic_generator
        )
        access_mask = (
            number_access_mask
            if task_id == 0
            else arithmetic_access_mask
        )
        observed, target, initial_cells = _task_batch(
            task,
            batch_size,
            length,
            device,
            generator,
            layout,
            orchestrator_starts,
        )
        route_top_k = support_width(
            local_step,
            total_steps,
            maximum=(
                number_maximum_support
                if task_id == 0
                else arithmetic_maximum_support
            ),
        )
        temperature = max(0.5, 1.25 * (0.998 ** local_step))
        logits, diagnostics = model(
            observed,
            output_slots=target.shape[1],
            temperature=temperature,
            route_top_k=route_top_k,
            initial_cell_id=initial_cells,
            cell_access_mask=access_mask,
            cell_destination_table=cell_destination_table,
            return_diagnostics=True,
        )
        task_loss = F.cross_entropy(
            logits.reshape(-1, 16), target.reshape(-1)
        )
        return_loss = orchestrator_return_loss(
            diagnostics,
            layout=layout,
            return_relays=return_relays,
        )
        loss = task_loss + return_loss_weight * return_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if router_only:
            retain_router_only_gradients(model, layout=layout)
        latest_gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=False
        )
        if not bool(torch.isfinite(latest_gradient_norm).item()):
            model.load_state_dict(best_state)
            clear_optimizer_state(optimizer)
            optimizer.zero_grad(set_to_none=True)
            if restart_count < nonfinite_restarts:
                restart_count += 1
                print(
                    "stage=skills_interleaved recovered_nonfinite "
                    f"step={local_step + 1} "
                    f"restart={restart_count}/{nonfinite_restarts} "
                    "restored_best=True optimizer_reset=True",
                    flush=True,
                )
                continue
            print(
                f"stage=skills_interleaved stopped_nonfinite "
                f"step={local_step + 1} restoring_best=True",
                flush=True,
            )
            break
        optimizer.step()
        completed = local_step + 1
        completed_steps = completed
        if completed % log_every and completed != total_steps:
            continue
        metrics = skill_metrics()
        current_minimum_ownership = min(
            metrics["number"]["final_orchestrator"],
            metrics["arithmetic"]["final_orchestrator"],
        )
        maximum_observed_ownership = max(
            maximum_observed_ownership,
            current_minimum_ownership,
        )
        current_router_learning_rate = apply_ownership_router_schedule(
            optimizer,
            ownership_router_schedule,
            maximum_observed_ownership=maximum_observed_ownership,
        )
        current_score = score(metrics)
        if current_score > best_score:
            best_score = current_score
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        elapsed = time.perf_counter() - started
        print(
            f"stage=skills_interleaved step={completed}/{total_steps} "
            f"global_step={global_step + completed} "
            f"steps_per_second={completed / elapsed:.2f} "
            f"top_k={route_top_k} "
            f"number_acc={metrics['number']['accuracy']:.3f} "
            f"arithmetic_acc={metrics['arithmetic']['accuracy']:.3f} "
            f"skill_hmean={current_score:.3f} "
            f"number_cross={metrics['number']['cross_region']:.3f} "
            f"arithmetic_cross={metrics['arithmetic']['cross_region']:.3f} "
            f"number_gate_traffic="
            f"{metrics['number']['number_call_share']:.3f} "
            f"arithmetic_gate_traffic="
            f"{metrics['arithmetic']['arithmetic_call_share']:.3f} "
            f"number_home={metrics['number']['final_orchestrator']:.3f} "
            f"arithmetic_home="
            f"{metrics['arithmetic']['final_orchestrator']:.3f} "
            f"max_min_ownership={maximum_observed_ownership:.3f} "
            f"router_lr={current_router_learning_rate} "
            f"return_loss={float(return_loss.detach()):.4f} "
            f"router_only={router_only} "
            f"grad_norm={float(latest_gradient_norm):.4f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    print(
        "stage=skills_interleaved restored_best "
        f"number_acc={best_metrics['number']['accuracy']:.3f} "
        f"arithmetic_acc={best_metrics['arithmetic']['accuracy']:.3f} "
        f"skill_hmean={best_score:.3f}",
        flush=True,
    )
    return global_step + completed_steps, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skill-steps", type=int, default=2_000)
    parser.add_argument("--number-steps", type=int, default=None)
    parser.add_argument("--arithmetic-steps", type=int, default=None)
    parser.add_argument("--integration-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--length", type=int, default=7)
    parser.add_argument("--relays", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.008)
    parser.add_argument("--number-learning-rate", type=float, default=None)
    parser.add_argument(
        "--arithmetic-learning-rate", type=float, default=None
    )
    parser.add_argument(
        "--integration-learning-rate", type=float, default=None
    )
    parser.add_argument(
        "--optimizer-mode",
        choices=("adamw", "split"),
        default="adamw",
        help=(
            "global AdamW or zero-momentum SGD route heads plus AdamW "
            "transformations"
        ),
    )
    parser.add_argument(
        "--router-learning-rate",
        type=float,
        default=None,
        help="SGD route-head learning rate in split optimizer mode",
    )
    parser.add_argument(
        "--ownership-router-schedule",
        action="store_true",
        help=(
            "monotonically lower split router LR as best observed final "
            "orchestrator ownership rises"
        ),
    )
    parser.add_argument(
        "--router-minimum-learning-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--router-ownership-start",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--router-ownership-target",
        type=float,
        default=0.8,
    )
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--number-cells", type=int, default=4)
    parser.add_argument("--orchestrator-cells", type=int, default=2)
    parser.add_argument(
        "--orchestrator-participates",
        action="store_true",
        help=(
            "start skill packets in the orchestrator and expose both skill "
            "bridges during regional pretraining"
        ),
    )
    parser.add_argument(
        "--interleaved-skills",
        action="store_true",
        help="shuffle number and arithmetic updates into one rehearsal stage",
    )
    parser.add_argument(
        "--call-return",
        action="store_true",
        help=(
            "use directional relay phases: orchestrator dispatch, specialist "
            "compute, mandatory return, orchestrator consolidation"
        ),
    )
    parser.add_argument(
        "--learned-dispatch",
        action="store_true",
        help=(
            "let the orchestrator learn calls to either specialist while "
            "specialists may only work locally or return"
        ),
    )
    parser.add_argument(
        "--gate-dispatch",
        action="store_true",
        help=(
            "expose exactly one number gate and one arithmetic gate to the "
            "orchestrator, with a direct specialist return action"
        ),
    )
    parser.add_argument(
        "--return-loss-weight",
        type=float,
        default=None,
        help="late-relay orchestrator-home penalty (default 0.1 in learned mode)",
    )
    parser.add_argument("--return-relays", type=int, default=2)
    parser.add_argument(
        "--maximum-support",
        type=int,
        default=None,
        help=(
            "override the broad-to-top-2 route-support maximum; use 2 for "
            "a sparse continuation after dense routes have materialized"
        ),
    )
    parser.add_argument(
        "--nonfinite-restarts",
        type=int,
        default=0,
        help=(
            "bounded best-checkpoint rewinds with optimizer-state reset "
            "instead of stopping at the first non-finite gradient"
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--resume-skill-checkpoint",
        type=Path,
        default=None,
        help="load a saved regional model and run integration only",
    )
    parser.add_argument(
        "--rehearse-loaded-skills",
        action="store_true",
        help="continue interleaved dispatch rehearsal after loading skills",
    )
    parser.add_argument(
        "--router-only-rehearsal",
        action="store_true",
        help="freeze transformations and adapt only route heads during rehearsal",
    )
    args = parser.parse_args()
    number_steps = args.number_steps or args.skill_steps
    arithmetic_steps = args.arithmetic_steps or args.skill_steps
    if min(
        number_steps,
        arithmetic_steps,
        args.integration_steps,
        args.batch_size,
        args.eval_batch_size,
        args.length,
        args.relays,
        args.log_every,
    ) <= 0:
        parser.error("step, size, length, relay, and logging values must be positive")
    if args.return_relays <= 0:
        parser.error("--return-relays must be positive")
    if (
        args.maximum_support is not None
        and not 2 <= args.maximum_support <= 6
    ):
        parser.error("--maximum-support must be between 2 and 6")
    if args.nonfinite_restarts < 0:
        parser.error("--nonfinite-restarts must be non-negative")
    if args.call_return and args.learned_dispatch:
        parser.error("--call-return and --learned-dispatch are mutually exclusive")
    if args.gate_dispatch and not args.learned_dispatch:
        parser.error("--gate-dispatch requires --learned-dispatch")
    return_loss_weight = (
        0.1
        if args.return_loss_weight is None and args.learned_dispatch
        else float(args.return_loss_weight or 0.0)
    )
    if return_loss_weight < 0:
        parser.error("--return-loss-weight must be non-negative")
    if (
        args.router_learning_rate is not None
        and args.router_learning_rate <= 0
    ):
        parser.error("--router-learning-rate must be positive")
    if args.ownership_router_schedule:
        if args.optimizer_mode != "split":
            parser.error(
                "--ownership-router-schedule requires "
                "--optimizer-mode split"
            )
        if args.router_learning_rate is None:
            parser.error(
                "--ownership-router-schedule requires an explicit "
                "--router-learning-rate maximum"
            )
        try:
            ownership_router_schedule = OwnershipRouterSchedule(
                maximum_learning_rate=args.router_learning_rate,
                minimum_learning_rate=args.router_minimum_learning_rate,
                start_ownership=args.router_ownership_start,
                target_ownership=args.router_ownership_target,
            )
        except ValueError as error:
            parser.error(str(error))
    else:
        ownership_router_schedule = None

    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    try:
        layout = RegionalLayout.from_sizes(
            number_cells=args.number_cells,
            orchestrator_cells=args.orchestrator_cells,
        )
    except ValueError as error:
        parser.error(str(error))
    config = GatorPathPrefixConfig(
        num_cells=10,
        relay_steps=args.relays,
        top_k=2,
    )
    model = GatorPathPrefixModel(config).to(device)
    cell_destination_table = (
        regional_gate_destination_table(device, layout=layout)
        if args.gate_dispatch
        else None
    )
    resumed_step = 0
    if args.resume_skill_checkpoint is not None:
        checkpoint = torch.load(
            args.resume_skill_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        resumed_step = int(checkpoint.get("step", 0))
    number_learning_rate = (
        args.number_learning_rate or args.learning_rate
    )
    arithmetic_learning_rate = (
        args.arithmetic_learning_rate or args.learning_rate
    )
    integration_learning_rate = (
        args.integration_learning_rate or args.learning_rate
    )
    isolated = regional_access_mask(
        device, bridges_open=False, layout=layout
    )
    integrated = regional_access_mask(
        device, bridges_open=True, layout=layout
    )
    if args.learned_dispatch:
        if not args.orchestrator_participates:
            parser.error(
                "--learned-dispatch requires --orchestrator-participates"
            )
        integrated = learned_dispatch_access_mask(
            device,
            layout=layout,
        )
    if args.call_return:
        if not args.orchestrator_participates:
            parser.error(
                "--call-return requires --orchestrator-participates"
            )
        try:
            integrated = regional_composition_schedule(
                device,
                relay_steps=args.relays,
                layout=layout,
            )
        except ValueError as error:
            parser.error(str(error))
    print(
        f"device={device.type} parameters={model.parameter_count} "
        f"regions=number:{len(layout.number)},"
        f"arithmetic:{len(layout.arithmetic)},"
        f"orchestrator:{len(layout.orchestrator)} "
        f"number_steps={number_steps} "
        f"arithmetic_steps={arithmetic_steps} "
        f"integration_steps={args.integration_steps} "
        f"orchestrator_participates={args.orchestrator_participates} "
        f"call_return={args.call_return} "
        f"learned_dispatch={args.learned_dispatch} "
        f"gate_dispatch={args.gate_dispatch} "
        f"optimizer_mode={args.optimizer_mode} "
        f"router_learning_rate={args.router_learning_rate} "
        f"ownership_router_schedule={args.ownership_router_schedule} "
        f"router_minimum_learning_rate="
        f"{args.router_minimum_learning_rate} "
        f"router_ownership_range="
        f"{args.router_ownership_start}:{args.router_ownership_target} "
        f"return_loss_weight={return_loss_weight}",
        flush=True,
    )

    global_step = resumed_step
    number_access = (
        regional_skill_access_mask(
            device, skill="number", layout=layout
        )
        if args.orchestrator_participates
        else isolated
    )
    arithmetic_access = (
        regional_skill_access_mask(
            device, skill="arithmetic", layout=layout
        )
        if args.orchestrator_participates
        else isolated
    )
    if args.call_return:
        number_access = regional_call_return_schedule(
            device,
            skill="number",
            relay_steps=args.relays,
            layout=layout,
        )
        arithmetic_access = regional_call_return_schedule(
            device,
            skill="arithmetic",
            relay_steps=args.relays,
            layout=layout,
        )
    elif args.learned_dispatch:
        number_access = integrated
        arithmetic_access = integrated
    maximum_support = (
        args.maximum_support
        if args.maximum_support is not None
        else (
            2
            if args.call_return
            or (
                not args.orchestrator_participates
                and len(layout.number) == 1
            )
            else 6
        )
    )
    number_maximum_support = maximum_support
    arithmetic_maximum_support = (
        maximum_support
        if args.maximum_support is not None
        else (2 if args.call_return else 6)
    )
    if not should_train_skills(
        args.resume_skill_checkpoint,
        rehearse_loaded_skills=args.rehearse_loaded_skills,
    ):
        print(
            f"stage=skills_loaded checkpoint={args.resume_skill_checkpoint} "
            f"step={global_step}",
            flush=True,
        )
    elif args.interleaved_skills:
        optimizer = build_optimizer(
            model,
            learning_rate=arithmetic_learning_rate,
            weight_decay=(
                0.0 if args.router_only_rehearsal else 1.0e-4
            ),
            mode=args.optimizer_mode,
            router_learning_rate=args.router_learning_rate,
        )
        global_step, _ = train_interleaved_skills(
            model,
            optimizer,
            number_steps=number_steps,
            arithmetic_steps=arithmetic_steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            length=args.length,
            number_access_mask=number_access,
            arithmetic_access_mask=arithmetic_access,
            log_every=args.log_every,
            seed=args.seed,
            global_step=global_step,
            layout=layout,
            number_maximum_support=number_maximum_support,
            arithmetic_maximum_support=arithmetic_maximum_support,
            return_loss_weight=return_loss_weight,
            return_relays=args.return_relays,
            router_only=args.router_only_rehearsal,
            orchestrator_starts=args.orchestrator_participates,
            cell_destination_table=cell_destination_table,
            nonfinite_restarts=args.nonfinite_restarts,
            ownership_router_schedule=ownership_router_schedule,
        )
    else:
        optimizer = build_optimizer(
            model,
            learning_rate=number_learning_rate,
            weight_decay=1.0e-4,
            mode=args.optimizer_mode,
            router_learning_rate=args.router_learning_rate,
        )
        global_step, _ = train_task(
            model,
            optimizer,
            task="number",
            steps=number_steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            length=args.length,
            access_mask=number_access,
            log_every=args.log_every,
            seed=args.seed + 11,
            global_step=global_step,
            layout=layout,
            maximum_support=maximum_support,
            orchestrator_starts=args.orchestrator_participates,
            return_loss_weight=return_loss_weight,
            return_relays=args.return_relays,
            cell_destination_table=cell_destination_table,
            nonfinite_restarts=args.nonfinite_restarts,
            ownership_router_schedule=ownership_router_schedule,
        )
        optimizer = build_optimizer(
            model,
            learning_rate=arithmetic_learning_rate,
            weight_decay=1.0e-4,
            mode=args.optimizer_mode,
            router_learning_rate=args.router_learning_rate,
        )
        global_step, _ = train_task(
            model,
            optimizer,
            task="arithmetic",
            steps=arithmetic_steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            length=args.length,
            access_mask=arithmetic_access,
            log_every=args.log_every,
            seed=args.seed + 29,
            global_step=global_step,
            layout=layout,
            maximum_support=maximum_support,
            orchestrator_starts=args.orchestrator_participates,
            return_loss_weight=return_loss_weight,
            return_relays=args.return_relays,
            cell_destination_table=cell_destination_table,
            nonfinite_restarts=args.nonfinite_restarts,
            ownership_router_schedule=ownership_router_schedule,
        )
    retained_number = evaluate(
        model,
        task="number",
        batch_size=args.eval_batch_size,
        length=args.length,
        access_mask=number_access,
        seed=args.seed + 100_011,
        layout=layout,
        orchestrator_starts=args.orchestrator_participates,
        cell_destination_table=cell_destination_table,
    )
    retained_arithmetic = evaluate(
        model,
        task="arithmetic",
        batch_size=args.eval_batch_size,
        length=args.length,
        access_mask=arithmetic_access,
        seed=args.seed + 100_029,
        layout=layout,
        orchestrator_starts=args.orchestrator_participates,
        cell_destination_table=cell_destination_table,
    )
    print(
        f"stage=skills_retained number_acc={retained_number['accuracy']:.3f} "
        f"arithmetic_acc={retained_arithmetic['accuracy']:.3f}",
        flush=True,
    )
    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": global_step,
                "architecture": asdict(config),
                "model": model.state_dict(),
                "regions": {
                    "number": layout.number,
                    "arithmetic": layout.arithmetic,
                    "orchestrator": layout.orchestrator,
                },
                "routing": {
                    "learned_dispatch": args.learned_dispatch,
                    "gate_dispatch": args.gate_dispatch,
                    "cell_destination_table": (
                        None
                        if cell_destination_table is None
                        else cell_destination_table.detach().cpu()
                    ),
                },
                "metrics": {
                    "number": retained_number,
                    "arithmetic": retained_arithmetic,
                },
            },
            args.checkpoint_dir / "gator_path_regional_skills.pt",
        )
    before = evaluate(
        model,
        task="prefix",
        batch_size=args.eval_batch_size,
        length=args.length,
        access_mask=integrated,
        seed=args.seed + 100_000,
        layout=layout,
        orchestrator_starts=args.orchestrator_participates,
        cell_destination_table=cell_destination_table,
    )
    print(
        f"stage=integration_before correct={before['correct']:.2f}/{args.length} "
        f"acc={before['accuracy']:.3f} loss={before['loss']:.4f} "
        f"cross_region={before['cross_region']:.3f}",
        flush=True,
    )
    optimizer = build_optimizer(
        model,
        learning_rate=integration_learning_rate,
        weight_decay=1.0e-4,
        mode=args.optimizer_mode,
        router_learning_rate=args.router_learning_rate,
    )
    global_step, _ = train_task(
        model,
        optimizer,
        task="prefix",
        steps=args.integration_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        length=args.length,
        access_mask=integrated,
        log_every=args.log_every,
        seed=args.seed + 47,
        global_step=global_step,
        layout=layout,
        maximum_support=maximum_support,
        orchestrator_starts=args.orchestrator_participates,
        return_loss_weight=return_loss_weight,
        return_relays=args.return_relays,
        cell_destination_table=cell_destination_table,
        nonfinite_restarts=args.nonfinite_restarts,
        ownership_router_schedule=ownership_router_schedule,
    )
    final = evaluate(
        model,
        task="prefix",
        batch_size=args.eval_batch_size,
        length=args.length,
        access_mask=integrated,
        seed=args.seed + 100_000,
        layout=layout,
        orchestrator_starts=args.orchestrator_participates,
        cell_destination_table=cell_destination_table,
    )
    print(
        f"final correct={final['correct']:.2f}/{args.length} "
        f"acc={final['accuracy']:.3f} exact={final['exact']:.3f} "
        f"loss={final['loss']:.4f} "
        f"cross_region={final['cross_region']:.3f}",
        flush=True,
    )

    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": global_step,
                "architecture": asdict(config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "regions": {
                    "number": layout.number,
                    "arithmetic": layout.arithmetic,
                    "orchestrator": layout.orchestrator,
                },
                "routing": {
                    "learned_dispatch": args.learned_dispatch,
                    "gate_dispatch": args.gate_dispatch,
                    "cell_destination_table": (
                        None
                        if cell_destination_table is None
                        else cell_destination_table.detach().cpu()
                    ),
                },
                "metrics": final,
            },
            args.checkpoint_dir / "gator_path_regional_latest.pt",
        )


if __name__ == "__main__":
    main()
