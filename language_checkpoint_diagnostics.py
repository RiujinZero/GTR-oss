"""Diagnose causal language capacity inside a Gator language checkpoint.

This probe reconstructs the fixed validation sources used by the language
trainer, measures teacher-forced quality by source and token position, and
tests whether any individual lane predicts a target that the final vote loses.
It can also perform a small true autoregressive rollout from arbitrary prompts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from gator.data import (
    DialogueTurn,
    assistant_examples,
    iter_jsonl_conversations,
    iter_tinystories,
    load_smoltalk,
    split_conversations,
    tinystories_sentence_pairs,
)
from gator.preparation import tokenize_assistant_examples
from gator.tasks import (
    EncodedAssistantExample,
    LanguageBatchConfig,
    LanguageBatcher,
    LanguageExampleFilterConfig,
    filter_language_examples,
    legal_response_token_mask,
    whitespace_token_mask,
)
from gator.tokenization import CompactTokenizer
from gator_language_train import teacher_forced_output_inputs
from language_probe import model_from_checkpoint


def _prepared_examples(
    examples: Iterable,
    tokenizer: CompactTokenizer,
) -> list[EncodedAssistantExample]:
    return [
        EncodedAssistantExample(
            conversation_id=example.conversation_id,
            turn_index=example.turn_index,
            context_ids=example.context_ids,
            response_ids=example.response_ids,
            source=example.source,
        )
        for example in tokenize_assistant_examples(examples, tokenizer)
    ]


def _even_sample(
    examples: list[EncodedAssistantExample],
    limit: int,
) -> list[EncodedAssistantExample]:
    if limit <= 0 or limit >= len(examples):
        return examples
    if limit == 1:
        return [examples[len(examples) // 2]]
    return [
        examples[round(index * (len(examples) - 1) / (limit - 1))]
        for index in range(limit)
    ]


def _validation_sources(
    *,
    tokenizer: CompactTokenizer,
    task: LanguageBatchConfig,
    state: dict,
    tinystories_valid: Path | None,
    smoltalk: Path | None,
    lmsys: Path | None,
) -> dict[str, list[EncodedAssistantExample]]:
    data = state.get("data_config") or {}
    common = dict(
        require_complete_windows=bool(
            data.get("require_complete_windows", False)
        ),
        require_complete_context=bool(
            data.get("require_complete_context", True)
        ),
        require_complete_response=bool(
            data.get("require_complete_response", False)
        ),
        trim_context_turns=bool(data.get("trim_context_turns", True)),
        deduplicate=bool(data.get("deduplicate_examples", True)),
    )
    sources: dict[str, list[EncodedAssistantExample]] = {}

    if tinystories_valid is not None:
        tiny_filter = LanguageExampleFilterConfig(
            **common,
            minimum_ascii_letter_ratio=float(
                data.get("lmsys_min_ascii_letter_ratio", 0.9)
            ),
            minimum_response_characters=int(
                data.get("lmsys_min_response_characters", 8)
            ),
        )
        stories = int(data.get("tinystories_validation_stories", 512))
        sources["TinyStories"] = _prepared_examples(
            filter_language_examples(
                tinystories_sentence_pairs(
                    iter_tinystories(
                        tinystories_valid,
                        max_stories=stories,
                    ),
                    context_tokens=task.context_tokens,
                    response_tokens=task.response_tokens,
                ),
                tokenizer,
                task,
                tiny_filter,
            ),
            tokenizer,
        )

    if smoltalk is not None:
        conversations = load_smoltalk(smoltalk)
        _, validation = split_conversations(
            conversations,
            validation_fraction=0.1,
            seed=0,
        )
        sources["SmolTalk Everyday"] = _prepared_examples(
            filter_language_examples(
                assistant_examples(
                    validation,
                    skip_first_assistant=bool(
                        data.get("skip_first_assistant", True)
                    ),
                ),
                tokenizer,
                task,
                LanguageExampleFilterConfig(**common),
            ),
            tokenizer,
        )

    if lmsys is not None:
        stream_filter = LanguageExampleFilterConfig(
            **common,
            require_user_prompt=bool(
                data.get("lmsys_require_user_prompt", True)
            ),
            minimum_ascii_letter_ratio=float(
                data.get("lmsys_min_ascii_letter_ratio", 0.9)
            ),
            minimum_response_characters=int(
                data.get("lmsys_min_response_characters", 8)
            ),
        )
        conversations = list(
            iter_jsonl_conversations(
                lmsys,
                max_conversations=int(
                    data.get("lmsys_validation_conversations", 512)
                ),
                max_characters=int(
                    data.get("lmsys_max_characters", 8192)
                ),
                skip_invalid=True,
            )
        )
        sources["LMSYS"] = _prepared_examples(
            filter_language_examples(
                assistant_examples(
                    conversations,
                    skip_first_assistant=bool(
                        data.get("lmsys_skip_first_assistant", False)
                    ),
                ),
                tokenizer,
                task,
                stream_filter,
            ),
            tokenizer,
        )
    return sources


def _new_counter() -> dict[str, float]:
    return defaultdict(float)


def _record(
    counter: dict[str, float],
    *,
    nll: torch.Tensor,
    correct: torch.Tensor,
    top5: torch.Tensor,
    content: torch.Tensor,
    lane_oracle: torch.Tensor,
    vote_loss: torch.Tensor,
) -> None:
    counter["tokens"] += int(content.sum())
    counter["loss"] += float((nll * content).sum())
    counter["correct"] += int((correct & content).sum())
    counter["top5"] += int((top5 & content).sum())
    counter["lane_oracle"] += int((lane_oracle & content).sum())
    counter["vote_loss"] += int((vote_loss & content).sum())


def _finish(counter: dict[str, float]) -> dict[str, float]:
    tokens = max(counter["tokens"], 1.0)
    loss = counter["loss"] / tokens
    return {
        "tokens": int(counter["tokens"]),
        "content_loss": loss,
        "content_ppl": math.exp(min(loss, 20.0)),
        "content_accuracy": counter["correct"] / tokens,
        "content_top5": counter["top5"] / tokens,
        "lane_oracle_accuracy": counter["lane_oracle"] / tokens,
        "vote_lost_target_rate": counter["vote_loss"] / tokens,
    }


def _position_bucket(position: int) -> str:
    start = (position // 16) * 16
    return f"{start:03d}-{start + 15:03d}"


def _context_bucket(length: int) -> str:
    if length <= 128:
        return "001-128"
    if length <= 256:
        return "129-256"
    if length <= 384:
        return "257-384"
    return "385-512"


@torch.no_grad()
def causal_counterfactuals(
    model,
    tokenizer: CompactTokenizer,
    task: LanguageBatchConfig,
    examples: list[EncodedAssistantExample],
    *,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    """Measure how teacher accuracy depends on prompt and response prefix."""

    batcher = LanguageBatcher(examples, task, seed=0)
    whitespace = whitespace_token_mask(tokenizer, device="cpu")
    legal = legal_response_token_mask(tokenizer, device="cpu")
    semantic = legal & ~whitespace
    semantic = semantic.clone()
    semantic[tokenizer.EOS_ID] = False
    blank = tokenizer.blank_token_id
    assert blank is not None
    semantic[blank] = False
    counters: dict[str, dict[str, float]] = defaultdict(_new_counter)

    for offset in range(0, len(examples), batch_size):
        subset = examples[offset : offset + batch_size]
        batch = batcher.collate_cpu(subset)
        teacher_ids = teacher_forced_output_inputs(
            batch,
            blank_token_id=blank,
        )
        row_roll = torch.arange(len(subset)).roll(1)
        conditions = {
            "correct": (
                batch.context_ids,
                batch.context_mask,
                teacher_ids,
            ),
            "shuffled_context": (
                batch.context_ids.index_select(0, row_roll),
                batch.context_mask.index_select(0, row_roll),
                teacher_ids,
            ),
            "masked_context": (
                batch.context_ids,
                torch.zeros_like(batch.context_mask),
                teacher_ids,
            ),
            "shuffled_response_prefix": (
                batch.context_ids,
                batch.context_mask,
                teacher_ids.index_select(0, row_roll),
            ),
            "blank_response_prefix": (
                batch.context_ids,
                batch.context_mask,
                torch.full_like(teacher_ids, blank),
            ),
        }
        content = batch.target_mask & semantic[batch.target_ids]
        for name, (context_ids, context_mask, output_ids) in conditions.items():
            logits = model(
                context_ids,
                context_mask,
                output_slots=task.response_tokens,
                output_token_ids=output_ids,
                output_mask=batch.target_mask,
                causal=True,
            )
            log_probs = logits.log_softmax(-1)
            nll = -log_probs.gather(
                -1, batch.target_ids.unsqueeze(-1)
            ).squeeze(-1)
            prediction = log_probs.argmax(-1)
            correct = prediction.eq(batch.target_ids)
            top5 = log_probs.topk(5, dim=-1).indices.eq(
                batch.target_ids.unsqueeze(-1)
            ).any(-1)
            _record(
                counters[name],
                nll=nll,
                correct=correct,
                top5=top5,
                content=content,
                lane_oracle=torch.zeros_like(correct),
                vote_loss=torch.zeros_like(correct),
            )
    return {
        name: _finish(counter)
        for name, counter in counters.items()
    }


@torch.no_grad()
def diagnose_source(
    model,
    tokenizer: CompactTokenizer,
    task: LanguageBatchConfig,
    examples: list[EncodedAssistantExample],
    *,
    batch_size: int,
) -> dict:
    batcher = LanguageBatcher(examples, task, seed=0)
    whitespace = whitespace_token_mask(tokenizer, device="cpu")
    legal = legal_response_token_mask(tokenizer, device="cpu")
    semantic = legal & ~whitespace
    semantic = semantic.clone()
    semantic[tokenizer.EOS_ID] = False
    if tokenizer.blank_token_id is not None:
        semantic[tokenizer.blank_token_id] = False

    total = _new_counter()
    positions: dict[str, dict[str, float]] = defaultdict(_new_counter)
    contexts: dict[str, dict[str, float]] = defaultdict(_new_counter)
    terminal_effective = terminal_top1 = terminal_observations = 0.0

    for offset in range(0, len(examples), batch_size):
        subset = examples[offset : offset + batch_size]
        batch = batcher.collate_cpu(subset)
        output_ids = teacher_forced_output_inputs(
            batch,
            blank_token_id=tokenizer.blank_token_id,
        )
        logits, trace = model(
            batch.context_ids,
            batch.context_mask,
            output_slots=task.response_tokens,
            output_token_ids=output_ids,
            output_mask=batch.target_mask,
            causal=True,
            return_trace=True,
        )
        log_probs = logits.log_softmax(-1)
        nll = -log_probs.gather(
            -1, batch.target_ids.unsqueeze(-1)
        ).squeeze(-1)
        prediction = log_probs.argmax(-1)
        correct = prediction.eq(batch.target_ids)
        top5 = log_probs.topk(5, dim=-1).indices.eq(
            batch.target_ids.unsqueeze(-1)
        ).any(-1)
        content = batch.target_mask & semantic[batch.target_ids]

        lane_readout = trace["lane_readout"]
        lane_logits = (
            lane_readout @ model.core.token_embedding.weight.t()
            + model.core.lm_bias
        )
        if model.core._forbidden_output_token_ids.numel():
            lane_logits = lane_logits.index_fill(
                -1,
                model.core._forbidden_output_token_ids,
                -1e9,
            )
        lane_prediction = lane_logits.argmax(-1)
        lane_oracle = lane_prediction.eq(
            batch.target_ids.unsqueeze(-1)
        ).any(-1)
        vote_loss = lane_oracle & ~correct

        _record(
            total,
            nll=nll,
            correct=correct,
            top5=top5,
            content=content,
            lane_oracle=lane_oracle,
            vote_loss=vote_loss,
        )
        for position in range(task.response_tokens):
            mask = content[:, position]
            _record(
                positions[_position_bucket(position)],
                nll=nll[:, position],
                correct=correct[:, position],
                top5=top5[:, position],
                content=mask,
                lane_oracle=lane_oracle[:, position],
                vote_loss=vote_loss[:, position],
            )
        for row, example in enumerate(subset):
            mask = content[row]
            _record(
                contexts[_context_bucket(len(example.context_ids))],
                nll=nll[row],
                correct=correct[row],
                top5=top5[row],
                content=mask,
                lane_oracle=lane_oracle[row],
                vote_loss=vote_loss[row],
            )

        terminal = trace["terminal_weights"]
        entropy = -(
            terminal.clamp_min(1e-9)
            * terminal.clamp_min(1e-9).log()
        ).sum(-1)
        terminal_effective += float(entropy.exp().sum())
        terminal_top1 += float(terminal.max(-1).values.sum())
        terminal_observations += terminal.numel() / terminal.shape[-1]

    return {
        "examples": len(examples),
        "overall": _finish(total),
        "by_position": {
            key: _finish(value)
            for key, value in sorted(positions.items())
        },
        "by_context_tokens": {
            key: _finish(value)
            for key, value in sorted(contexts.items())
        },
        "terminal_vote": {
            "effective_lanes": terminal_effective
            / max(terminal_observations, 1),
            "top1_weight": terminal_top1
            / max(terminal_observations, 1),
        },
    }


@torch.no_grad()
def autoregressive_generate(
    model,
    tokenizer: CompactTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    context_ids = tokenizer.encode_turns((DialogueTurn("user", prompt),))
    context = torch.tensor([context_ids], dtype=torch.long)
    context_mask = torch.ones_like(context, dtype=torch.bool)
    generated: list[int] = []
    blank = tokenizer.blank_token_id
    assert blank is not None
    for _ in range(max_new_tokens):
        output_ids = torch.tensor(
            [[blank, *generated]],
            dtype=torch.long,
        )
        output_mask = torch.ones_like(output_ids, dtype=torch.bool)
        logits = model(
            context,
            context_mask,
            output_slots=output_ids.shape[1],
            output_token_ids=output_ids,
            output_mask=output_mask,
            causal=True,
        )
        token = int(logits[0, -1].argmax())
        if token == tokenizer.EOS_ID:
            break
        generated.append(token)
    return tokenizer.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tinystories-valid", type=Path)
    parser.add_argument("--smoltalk", type=Path)
    parser.add_argument("--lmsys", type=Path)
    parser.add_argument("--examples-per-source", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--autoregressive-tokens", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    state = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    tokenizer = CompactTokenizer.from_dict(state["tokenizer"])
    model = model_from_checkpoint(state)
    model.eval()
    task_state = state["task_config"]
    task = LanguageBatchConfig(
        context_tokens=int(task_state["context_tokens"]),
        response_tokens=int(task_state["response_tokens"]),
        reference_tokens=int(task_state["reference_tokens"]),
    )
    sources = _validation_sources(
        tokenizer=tokenizer,
        task=task,
        state=state,
        tinystories_valid=args.tinystories_valid,
        smoltalk=args.smoltalk,
        lmsys=args.lmsys,
    )
    sampled = {
        name: _even_sample(examples, args.examples_per_source)
        for name, examples in sources.items()
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            args.checkpoint.read_bytes()
        ).hexdigest(),
        "step": int(state["step"]),
        "parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "checkpoint_teacher_validation": state.get(
            "auxiliary_validation"
        ),
        "source_population": {
            name: len(examples) for name, examples in sources.items()
        },
        "source_sample": {
            name: diagnose_source(
                model,
                tokenizer,
                task,
                examples,
                batch_size=args.batch_size,
            )
            for name, examples in sampled.items()
        },
        "causal_counterfactuals": {
            name: causal_counterfactuals(
                model,
                tokenizer,
                task,
                examples,
                batch_size=args.batch_size,
            )
            for name, examples in sampled.items()
        },
    }
    if args.autoregressive_tokens:
        prompts = (
            "What season is it right now?",
            "Name an animal that says meow.",
            "What is 2 + 2?",
            "Tell me a short story about a dog.",
        )
        report["autoregressive"] = {
            prompt: autoregressive_generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.autoregressive_tokens,
            )
            for prompt in prompts
        }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
