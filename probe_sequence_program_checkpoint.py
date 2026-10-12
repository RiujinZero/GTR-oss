"""Autopsy a trained compositional sequence-program Gator checkpoint.

The benchmark's aggregate accuracy cannot distinguish a true capacity ceiling
from three narrower failures:

* the terminal composer may fuse useful lane specialists badly;
* the core may know primitives but fail to carry their composition;
* the model may ignore or weakly use the instruction tokens.

This probe keeps the checkpoint frozen and measures each failure separately.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from gator_convergent import ConvergentGator
from gator_sequence_program_benchmark import (
    OP_NAMES,
    OP_TOKEN_OFFSET,
    VALUE_COUNT,
    apply_operation,
    execute_program,
)


def build_model(state: dict, device: torch.device) -> ConvergentGator:
    config = state["config"]
    model = ConvergentGator(
        n_nodes=config["nodes"],
        d_model=config["width"],
        degree=config["degree"],
        relays=config["relays"],
        lanes=config["lanes"],
        connection_layout=config["connection_layout"],
        connection_seed=config["connection_seed"],
        connection_rewire=config["connection_rewire"],
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def observed(values: torch.Tensor, operations: torch.Tensor) -> torch.Tensor:
    return torch.cat((values, operations + OP_TOKEN_OFFSET), dim=1)


@torch.no_grad()
def predict_in_batches(
    model: ConvergentGator,
    values: torch.Tensor,
    operations: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ensemble predictions, lane predictions, and target log-probs."""

    ensemble_parts: list[torch.Tensor] = []
    lane_parts: list[torch.Tensor] = []
    target_log_prob_parts: list[torch.Tensor] = []
    targets = execute_program(values, operations)
    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        logits, trace = model(
            observed(values[start:stop], operations[start:stop]),
            output_slots=values.shape[1],
            return_trace=True,
        )
        lane_logits = trace["lane_logits"]
        batch_targets = targets[start:stop]
        target_index = batch_targets[:, :, None, None].expand(
            -1,
            -1,
            lane_logits.shape[2],
            1,
        )
        target_log_probs = F.log_softmax(lane_logits, dim=-1).gather(
            -1,
            target_index,
        ).squeeze(-1)
        ensemble_parts.append(logits.argmax(-1).cpu())
        lane_parts.append(lane_logits.argmax(-1).cpu())
        target_log_prob_parts.append(target_log_probs.cpu())
    return (
        torch.cat(ensemble_parts),
        torch.cat(lane_parts),
        torch.cat(target_log_prob_parts),
    )


@torch.no_grad()
def collect_lane_logits(
    model: ConvergentGator,
    values: torch.Tensor,
    operations: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    parts: list[torch.Tensor] = []
    ensemble_parts: list[torch.Tensor] = []
    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        logits, trace = model(
            observed(values[start:stop], operations[start:stop]),
            output_slots=values.shape[1],
            return_trace=True,
        )
        ensemble_parts.append(logits.cpu())
        parts.append(trace["lane_logits"].cpu())
    return torch.cat(ensemble_parts), torch.cat(parts)


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return float(predictions.eq(targets.cpu()).float().mean())


def exact_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    return float(predictions.eq(targets.cpu()).all(1).float().mean())


def random_programs(
    rows: int,
    program_length: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randint(
        1,
        len(OP_NAMES),
        (rows, program_length),
        generator=generator,
    )


def random_values(
    rows: int,
    sequence_length: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randint(
        0,
        VALUE_COUNT,
        (rows, sequence_length),
        generator=generator,
    )


def summarize_lanes(
    ensemble: torch.Tensor,
    lane_predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, object]:
    target_cpu = targets.cpu()
    lane_correct = lane_predictions.eq(target_cpu[:, :, None])
    lane_accuracy = lane_correct.float().mean(dim=(0, 1))
    best_lane = int(lane_accuracy.argmax())
    return {
        "ensemble_accuracy": accuracy(ensemble, targets),
        "ensemble_exact": exact_accuracy(ensemble, targets),
        "best_fixed_lane": best_lane,
        "best_fixed_lane_accuracy": float(lane_accuracy[best_lane]),
        "lane_accuracies": [float(value) for value in lane_accuracy],
        "any_lane_top1_coverage": float(
            lane_correct.any(-1).float().mean()
        ),
        "all_slots_have_a_correct_lane": float(
            lane_correct.any(-1).all(1).float().mean()
        ),
    }


def all_programs(program_length: int) -> torch.Tensor:
    return torch.tensor(
        list(
            itertools.product(
                range(1, len(OP_NAMES)),
                repeat=program_length,
            )
        ),
        dtype=torch.long,
    )


class ProgramSlotComposer(nn.Module):
    """Small diagnostic jury conditioned on instructions and output address."""

    def __init__(
        self,
        *,
        feature_width: int,
        lanes: int,
        hidden_width: int,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, lanes),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.network(features), dim=-1)


def composer_features(
    programs: torch.Tensor,
    *,
    sequence_length: int,
) -> torch.Tensor:
    program_features = F.one_hot(
        programs.cpu(),
        num_classes=len(OP_NAMES),
    ).float().flatten(1)
    slot_features = torch.eye(sequence_length)
    return torch.cat(
        (
            program_features[:, None].expand(
                -1,
                sequence_length,
                -1,
            ),
            slot_features[None].expand(
                programs.shape[0],
                -1,
                -1,
            ),
        ),
        dim=-1,
    )


def fit_program_slot_composer(
    train_features: torch.Tensor,
    train_lane_logits: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_lane_logits: torch.Tensor,
    test_targets: torch.Tensor,
    *,
    hidden_width: int,
    steps: int,
    minibatch_slots: int,
    seed: int,
) -> dict[str, float | int]:
    """Fit a tiny jury while keeping every checkpoint tensor frozen."""

    feature_width = train_features.shape[-1]
    lanes = train_lane_logits.shape[-2]
    vocabulary = train_lane_logits.shape[-1]
    composer = ProgramSlotComposer(
        feature_width=feature_width,
        lanes=lanes,
        hidden_width=hidden_width,
    )
    parameter_count = sum(
        parameter.numel() for parameter in composer.parameters()
    )
    optimizer = torch.optim.AdamW(
        composer.parameters(),
        lr=0.01,
        weight_decay=1e-4,
    )
    flat_features = train_features.reshape(-1, feature_width)
    flat_logits = train_lane_logits.reshape(-1, lanes, vocabulary)
    flat_targets = train_targets.reshape(-1)
    generator = torch.Generator().manual_seed(seed)
    composer.train()
    for _ in range(steps):
        indices = torch.randint(
            0,
            flat_targets.shape[0],
            (min(minibatch_slots, flat_targets.shape[0]),),
            generator=generator,
        )
        weights = composer(flat_features[indices])
        fused = torch.einsum(
            "nk,nkv->nv",
            weights,
            flat_logits[indices],
        )
        loss = F.cross_entropy(fused, flat_targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    composer.eval()
    predictions: list[torch.Tensor] = []
    test_features_flat = test_features.reshape(-1, feature_width)
    test_logits_flat = test_lane_logits.reshape(-1, lanes, vocabulary)
    with torch.no_grad():
        for start in range(0, test_features_flat.shape[0], 8192):
            stop = min(start + 8192, test_features_flat.shape[0])
            weights = composer(test_features_flat[start:stop])
            fused = torch.einsum(
                "nk,nkv->nv",
                weights,
                test_logits_flat[start:stop],
            )
            predictions.append(fused.argmax(-1))
    prediction = torch.cat(predictions).reshape(test_targets.shape)
    return {
        "parameters": parameter_count,
        "training_steps": steps,
        "accuracy": float(prediction.eq(test_targets).float().mean()),
        "exact": float(prediction.eq(test_targets).all(-1).float().mean()),
    }


def composer_probe(
    model: ConvergentGator,
    *,
    sequence_length: int,
    program_length: int,
    train_values_per_program: int,
    test_values_per_program: int,
    batch_size: int,
    composer_steps: int,
    composer_hidden_width: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    """Learn only which frozen lane to trust for each program and slot.

    The selector is fitted on one set of values and evaluated on disjoint
    values for the same complete instruction space. A large lift therefore
    identifies a terminal-composition bottleneck rather than memorized input
    sequences.
    """

    programs = all_programs(program_length)
    program_count = programs.shape[0]
    train_generator = torch.Generator().manual_seed(seed + 101)
    test_generator = torch.Generator().manual_seed(seed + 202)

    def make_rows(
        values_per_program: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        program_rows = programs[:, None, :].expand(
            -1,
            values_per_program,
            -1,
        ).reshape(-1, program_length)
        value_rows = random_values(
            program_count * values_per_program,
            sequence_length,
            generator=generator,
        )
        return value_rows.to(device), program_rows.to(device)

    train_values, train_programs = make_rows(
        train_values_per_program,
        train_generator,
    )
    train_targets = execute_program(train_values, train_programs)
    _, train_lane_logits = collect_lane_logits(
        model,
        train_values,
        train_programs,
        batch_size=batch_size,
    )
    train_lane_predictions = train_lane_logits.argmax(-1)
    train_target_log_probs = F.log_softmax(
        train_lane_logits,
        dim=-1,
    ).gather(
        -1,
        train_targets.cpu()[:, :, None, None].expand(
            -1,
            -1,
            train_lane_logits.shape[2],
            1,
        ),
    ).squeeze(-1)
    lanes = train_lane_predictions.shape[-1]
    train_correct = train_lane_predictions.eq(
        train_targets.cpu()[:, :, None]
    ).reshape(
        program_count,
        train_values_per_program,
        sequence_length,
        lanes,
    )
    # Two frozen-core selectors: a discrete top-1 lane and a target-likelihood
    # selector. Both are learned exclusively on the probe-training values.
    lookup_lane = train_correct.float().mean(1).argmax(-1)
    lookup_logprob_lane = train_target_log_probs.reshape(
        program_count,
        train_values_per_program,
        sequence_length,
        lanes,
    ).mean(1).argmax(-1)

    test_values, test_programs = make_rows(
        test_values_per_program,
        test_generator,
    )
    test_targets = execute_program(test_values, test_programs).cpu()
    test_ensemble_logits, test_lane_logits = collect_lane_logits(
        model,
        test_values,
        test_programs,
        batch_size=batch_size,
    )
    ensemble = test_ensemble_logits.argmax(-1)
    lane_predictions = test_lane_logits.argmax(-1)
    lane_predictions = lane_predictions.reshape(
        program_count,
        test_values_per_program,
        sequence_length,
        lanes,
    )
    expanded_lane = lookup_lane[:, None, :, None].expand(
        -1,
        test_values_per_program,
        -1,
        1,
    )
    selected = lane_predictions.gather(-1, expanded_lane).squeeze(-1)
    expanded_logprob_lane = lookup_logprob_lane[
        :, None, :, None
    ].expand(
        -1,
        test_values_per_program,
        -1,
        1,
    )
    selected_logprob = lane_predictions.gather(
        -1,
        expanded_logprob_lane,
    ).squeeze(-1)
    reshaped_targets = test_targets.reshape(
        program_count,
        test_values_per_program,
        sequence_length,
    )
    ensemble = ensemble.reshape(
        program_count,
        test_values_per_program,
        sequence_length,
    )
    any_lane = lane_predictions.eq(
        reshaped_targets[:, :, :, None]
    ).any(-1)
    learned_composer = fit_program_slot_composer(
        composer_features(
            train_programs.cpu(),
            sequence_length=sequence_length,
        ),
        train_lane_logits,
        train_targets.cpu(),
        composer_features(
            test_programs.cpu(),
            sequence_length=sequence_length,
        ),
        test_lane_logits,
        test_targets,
        hidden_width=composer_hidden_width,
        steps=composer_steps,
        minibatch_slots=4096,
        seed=seed + 909,
    )
    return {
        "program_count": program_count,
        "train_values_per_program": train_values_per_program,
        "test_values_per_program": test_values_per_program,
        "ensemble_accuracy": float(
            ensemble.eq(reshaped_targets).float().mean()
        ),
        "program_slot_top1_lookup_accuracy": float(
            selected.eq(reshaped_targets).float().mean()
        ),
        "program_slot_logprob_lookup_accuracy": float(
            selected_logprob.eq(reshaped_targets).float().mean()
        ),
        "any_lane_top1_coverage": float(any_lane.float().mean()),
        "program_slot_lookup_exact": float(
            selected.eq(reshaped_targets).all(-1).float().mean()
        ),
        "learned_program_slot_composer": learned_composer,
    }


@torch.no_grad()
def suffix_oracle_probe(
    model: ConvergentGator,
    *,
    rows: int,
    sequence_length: int,
    program_length: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed + 303)
    values = random_values(
        rows,
        sequence_length,
        generator=generator,
    ).to(device)
    programs = random_programs(
        rows,
        program_length,
        generator=generator,
    ).to(device)
    final_targets = execute_program(values, programs)

    results: dict[str, object] = {}
    current_values = values
    for completed_prefix in range(program_length):
        remaining = programs[:, completed_prefix:]
        padded = F.pad(
            remaining,
            (0, completed_prefix),
            value=0,
        )
        ensemble, lane_predictions, _ = predict_in_batches(
            model,
            current_values,
            padded,
            batch_size=batch_size,
        )
        results[f"oracle_prefix_{completed_prefix}"] = summarize_lanes(
            ensemble,
            lane_predictions,
            final_targets,
        )
        current_values = apply_operation(
            current_values,
            programs[:, completed_prefix],
        )
    return results


@torch.no_grad()
def instruction_counterfactual_probe(
    model: ConvergentGator,
    *,
    rows: int,
    sequence_length: int,
    program_length: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed + 404)
    values = random_values(
        rows,
        sequence_length,
        generator=generator,
    ).to(device)
    programs = random_programs(
        rows,
        program_length,
        generator=generator,
    ).to(device)
    original_targets = execute_program(values, programs)
    original, _, _ = predict_in_batches(
        model,
        values,
        programs,
        batch_size=batch_size,
    )

    permutation = torch.randperm(rows, generator=generator).to(device)
    shuffled_programs = programs[permutation]
    shuffled_targets = execute_program(values, shuffled_programs)
    shuffled, _, _ = predict_in_batches(
        model,
        values,
        shuffled_programs,
        batch_size=batch_size,
    )
    noop_programs = torch.zeros_like(programs)
    noop, _, _ = predict_in_batches(
        model,
        values,
        noop_programs,
        batch_size=batch_size,
    )
    return {
        "original_accuracy": accuracy(original, original_targets),
        "shuffled_program_accuracy_on_shuffled_target": accuracy(
            shuffled,
            shuffled_targets,
        ),
        "shuffled_program_accuracy_on_original_target": accuracy(
            shuffled,
            original_targets,
        ),
        "prediction_change_after_program_shuffle": float(
            shuffled.ne(original).float().mean()
        ),
        "noop_copy_accuracy": accuracy(noop, values),
    }


@torch.no_grad()
def repeated_operation_probe(
    model: ConvergentGator,
    *,
    rows: int,
    sequence_length: int,
    program_length: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    generator = torch.Generator().manual_seed(seed + 505)
    values = random_values(
        rows,
        sequence_length,
        generator=generator,
    ).to(device)
    result: dict[str, dict[str, float]] = {}
    for operation in range(1, len(OP_NAMES)):
        by_depth: dict[str, float] = {}
        for depth in range(1, program_length + 1):
            programs = torch.zeros(
                rows,
                program_length,
                dtype=torch.long,
                device=device,
            )
            programs[:, :depth] = operation
            targets = execute_program(values, programs)
            predictions, _, _ = predict_in_batches(
                model,
                values,
                programs,
                batch_size=batch_size,
            )
            by_depth[f"depth_{depth}_accuracy"] = accuracy(
                predictions,
                targets,
            )
        result[OP_NAMES[operation]] = by_depth
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--train-values-per-program", type=int, default=24)
    parser.add_argument("--test-values-per-program", type=int, default=48)
    parser.add_argument("--composer-steps", type=int, default=1_500)
    parser.add_argument("--composer-hidden-width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=73_001)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model = build_model(state, device)
    config = state["config"]
    sequence_length = int(config["sequence_length"])
    program_length = int(config["program_length"])
    print(
        f"checkpoint={args.checkpoint} step={state['step']} "
        f"parameters={model.parameter_count} device={device}",
        flush=True,
    )

    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(state["step"]),
        "parameters": int(model.parameter_count),
        "saved_results": state.get("results"),
        "composer": composer_probe(
            model,
            sequence_length=sequence_length,
            program_length=program_length,
            train_values_per_program=args.train_values_per_program,
            test_values_per_program=args.test_values_per_program,
            batch_size=args.batch_size,
            composer_steps=args.composer_steps,
            composer_hidden_width=args.composer_hidden_width,
            seed=args.seed,
            device=device,
        ),
        "suffix_oracle": suffix_oracle_probe(
            model,
            rows=args.rows,
            sequence_length=sequence_length,
            program_length=program_length,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        ),
        "instruction_counterfactual": instruction_counterfactual_probe(
            model,
            rows=args.rows,
            sequence_length=sequence_length,
            program_length=program_length,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        ),
        "repeated_operations": repeated_operation_probe(
            model,
            rows=min(args.rows, 2048),
            sequence_length=sequence_length,
            program_length=program_length,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        ),
    }
    rendered = json.dumps(results, indent=2)
    print(rendered, flush=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
