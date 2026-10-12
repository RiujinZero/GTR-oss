"""SmolTalk Everyday parsing and deterministic dataset preparation.

The adapter intentionally returns text rather than token IDs. Tokenization is a
model-level decision, while conversation boundaries and leakage-free splitting
are properties of the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from statistics import mean
from typing import Iterable, Iterator, Sequence


_TURN_HEADER = re.compile(r"(?m)^(USER|ASSISTANT):[ \t]*")


@dataclass(frozen=True)
class DialogueTurn:
    """One normalized dialogue turn."""

    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError(f"unsupported dialogue role: {self.role!r}")
        if not self.text:
            raise ValueError("dialogue turn text cannot be empty")


@dataclass(frozen=True)
class Conversation:
    """One source row with its parsed turns preserved as a split unit."""

    conversation_id: int
    turns: tuple[DialogueTurn, ...]
    source: str | None = None


@dataclass(frozen=True)
class AssistantExample:
    """A response target and every dialogue turn available before it."""

    conversation_id: int
    turn_index: int
    context: tuple[DialogueTurn, ...]
    response: str
    source: str | None = None


@dataclass(frozen=True)
class SmolTalkStatistics:
    """Compact statistics used to choose context and response lengths."""

    conversations: int
    turns: int
    assistant_examples: int
    mean_turns_per_conversation: float
    response_bytes_p50: int
    response_bytes_p90: int
    response_bytes_p95: int
    response_bytes_max: int
    context_bytes_p50: int
    context_bytes_p90: int
    context_bytes_p95: int
    context_bytes_max: int


def parse_dialogue(text: str) -> tuple[DialogueTurn, ...]:
    """Parse ``USER:``/``ASSISTANT:`` dialogue without assuming blank lines."""

    matches = list(_TURN_HEADER.finditer(text))
    if not matches:
        raise ValueError("dialogue contains no USER: or ASSISTANT: turn headers")
    if text[: matches[0].start()].strip():
        raise ValueError("unexpected text before the first dialogue turn")

    turns: list[DialogueTurn] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        role = match.group(1).lower()
        turns.append(DialogueTurn(role=role, text=body))
    return tuple(turns)


def iter_jsonl_conversations(
    path: str | Path,
    *,
    start_line: int = 0,
    skip_conversations: int = 0,
    max_conversations: int | None = None,
    max_characters: int | None = None,
    skip_invalid: bool = False,
) -> Iterator[Conversation]:
    """Stream the shared SmolTalk/LMSYS ``text`` JSONL schema.

    ``max_characters`` filters on the decoded row's complete ``text`` field
    before dialogue parsing. It is intentionally a raw-data filter rather than
    silent token-window truncation.
    """

    if max_characters is not None and max_characters <= 0:
        raise ValueError("max_characters must be positive or None")
    if skip_conversations < 0:
        raise ValueError("skip_conversations must be nonnegative")

    yielded = 0
    skipped = 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            if line_number < start_line:
                continue
            if max_conversations is not None and yielded >= max_conversations:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                text = row["text"]
                if max_characters is not None and len(text) > max_characters:
                    continue
                turns = parse_dialogue(text)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                if skip_invalid:
                    continue
                raise ValueError(f"invalid conversation row at line {line_number + 1}") from error
            if skipped < skip_conversations:
                skipped += 1
                continue
            yield Conversation(
                conversation_id=line_number,
                turns=turns,
                source=row.get("source"),
            )
            yielded += 1


def load_smoltalk(path: str | Path) -> list[Conversation]:
    """Load the small SmolTalk corpus while retaining conversation split units."""

    conversations = list(iter_jsonl_conversations(path, skip_invalid=True))
    if not conversations:
        raise ValueError("SmolTalk dataset is empty")
    return conversations


def split_conversations(
    conversations: Sequence[Conversation],
    *,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[Conversation], list[Conversation]]:
    """Create a deterministic exact-size split without conversation leakage."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    indices = list(range(len(conversations)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(indices) * validation_fraction))
    validation_indices = set(indices[:validation_count])
    train = [item for index, item in enumerate(conversations) if index not in validation_indices]
    validation = [item for index, item in enumerate(conversations) if index in validation_indices]
    return train, validation


def assistant_examples(
    conversations: Iterable[Conversation], *, skip_first_assistant: bool = False
) -> Iterator[AssistantExample]:
    """Yield assistant targets, optionally excluding each dialogue's first reply."""

    for conversation in conversations:
        assistant_number = 0
        for turn_index, turn in enumerate(conversation.turns):
            if turn.role != "assistant":
                continue
            assistant_number += 1
            if skip_first_assistant and assistant_number == 1:
                continue
            yield AssistantExample(
                conversation_id=conversation.conversation_id,
                turn_index=turn_index,
                context=conversation.turns[:turn_index],
                response=turn.text,
                source=conversation.source,
            )


def format_context(turns: Sequence[DialogueTurn]) -> str:
    """Render normalized turns with explicit role labels for tokenization."""

    labels = {"user": "USER", "assistant": "ASSISTANT"}
    return "\n\n".join(f"{labels[turn.role]}: {turn.text}" for turn in turns)


def ascii_letter_ratio(text: str) -> float:
    """Return the fraction of alphabetic characters that are ASCII letters.

    This is a deliberately cheap corpus gate, not a full language detector. It
    removes strongly non-Latin rows before a small English-oriented tokenizer
    is trained while retaining ordinary punctuation, numbers, and whitespace.
    """

    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    ascii_letters = sum(
        ("a" <= character.lower() <= "z") for character in letters
    )
    return ascii_letters / len(letters)


def conversation_ascii_letter_ratio(conversation: Conversation) -> float:
    """Measure ASCII-letter ratio across every turn of one conversation."""

    return ascii_letter_ratio("\n".join(turn.text for turn in conversation.turns))


def _percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    index = int((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize_smoltalk(conversations: Sequence[Conversation]) -> SmolTalkStatistics:
    """Summarize byte lengths without imposing a tokenizer."""

    examples = list(assistant_examples(conversations))
    response_bytes = [len(example.response.encode("utf-8")) for example in examples]
    context_bytes = [len(format_context(example.context).encode("utf-8")) for example in examples]
    turn_counts = [len(conversation.turns) for conversation in conversations]
    return SmolTalkStatistics(
        conversations=len(conversations),
        turns=sum(turn_counts),
        assistant_examples=len(examples),
        mean_turns_per_conversation=mean(turn_counts),
        response_bytes_p50=_percentile(response_bytes, 0.50),
        response_bytes_p90=_percentile(response_bytes, 0.90),
        response_bytes_p95=_percentile(response_bytes, 0.95),
        response_bytes_max=max(response_bytes),
        context_bytes_p50=_percentile(context_bytes, 0.50),
        context_bytes_p90=_percentile(context_bytes, 0.90),
        context_bytes_p95=_percentile(context_bytes, 0.95),
        context_bytes_max=max(context_bytes),
    )
