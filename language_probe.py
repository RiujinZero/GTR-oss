"""Probe a simultaneous language checkpoint with fixed user prompts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import torch

from gator.data import DialogueTurn
from gator.models import (
    GatorLanguageConfig,
    GatorLanguageModel,
    OneShotTransformerLanguageConfig,
    OneShotTransformerLanguageModel,
)
from gator.tokenization import CompactTokenizer
from gator_convergent import (
    ConvergentLanguageConfig,
    ConvergentLanguageModel,
)


def model_from_checkpoint(
    state: dict,
) -> (
    GatorLanguageModel
    | OneShotTransformerLanguageModel
    | ConvergentLanguageModel
):
    """Construct the checkpoint's language architecture without guessing CLI flags."""

    config_state = dict(state["model_config"])
    architecture = config_state.get("architecture")
    if architecture == "one_shot_transformer":
        model = OneShotTransformerLanguageModel(
            OneShotTransformerLanguageConfig(**config_state)
        )
    elif {
        "lanes",
        "degree",
        "connection_layout",
    }.issubset(config_state):
        # Convergent checkpoints predate an explicit architecture field.
        # Their lane/graph configuration is disjoint from cell-based Gator.
        config_state.setdefault("terminal_vote_mode", "global")
        model = ConvergentLanguageModel(
            ConvergentLanguageConfig(**config_state)
        )
    else:
        model = GatorLanguageModel(GatorLanguageConfig(**config_state))
    model.load_state_dict(state["model"], strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("prompts", nargs="*")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    checkpoint = Path(args.checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tokenizer = CompactTokenizer.from_dict(state["tokenizer"])
    model = model_from_checkpoint(state)
    model.eval()

    prompts = args.prompts or [
        "What season is it right now?",
        "Name an animal that says meow.",
        "What is 2 + 2?",
        "I need to rent a car. What types are available?",
        "What are some popular team sports?",
        "How often should a beginner run?",
    ]
    context_tokens = int(state["task_config"]["context_tokens"])
    response_tokens = int(state["task_config"]["response_tokens"])
    rows = []
    masks = []
    for prompt in prompts:
        encoded = tokenizer.encode_turns((DialogueTurn("user", prompt),))
        encoded = encoded[-context_tokens:]
        padding = context_tokens - len(encoded)
        rows.append([tokenizer.PAD_ID] * padding + encoded)
        masks.append([False] * padding + [True] * len(encoded))
    context = torch.tensor(rows, dtype=torch.long)
    context_mask = torch.tensor(masks, dtype=torch.bool)
    with torch.no_grad():
        if (
            isinstance(model, GatorLanguageModel)
            and model.config.cell_transport_mode == "stewardship"
        ):
            data_config = state.get("data_config") or {}
            execution_state = state.get("execution") or {}
            logits = model(
                context,
                context_mask,
                output_slots=response_tokens,
                stewardship_temperature=float(
                    data_config.get("evaluation_stewardship_temperature", 0.35)
                ),
                stewardship_prune=bool(
                    data_config.get("evaluation_stewardship_prune", True)
                ),
                stewardship_packed_from_relay=execution_state.get(
                    "packed_from_relay"
                ),
            )
        elif isinstance(model, ConvergentLanguageModel):
            logits = model(
                context,
                context_mask,
                output_slots=response_tokens,
            )
        else:
            logits = model(context, context_mask, output_slots=response_tokens)
        predictions = logits.argmax(-1)

    execution_state = state.get("execution") or {}
    print(
        f"checkpoint={checkpoint} "
        f"sha256={checkpoint_sha256} "
        f"checkpoint_step={state['step']} "
        f"parameters={sum(parameter.numel() for parameter in model.parameters())} "
        f"validation_ppl={state['validation']['perplexity']:.2f} "
        f"packed_from_relay={execution_state.get('packed_from_relay')}"
    )
    for prompt, prediction in zip(prompts, predictions.tolist()):
        print(f"prompt={prompt!r}")
        print(f"formed={tokenizer.decode(prediction)!r}")
        print(f"slots={tokenizer.decode(prediction, diagnostic=True)!r}")


if __name__ == "__main__":
    main()
