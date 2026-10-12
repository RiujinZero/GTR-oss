"""Boundary-safe TinyStories parsing and prefix-continuation examples."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable, Iterator

from .smoltalk import AssistantExample, Conversation, DialogueTurn


TINYSTORIES_SOURCE = "roneneldan/TinyStories"
_END_OF_TEXT = "<|endoftext|>"
_SENTENCE_PATTERN = re.compile(
    r".+?(?:[.!?]+(?:[\"'\u201d\u2019]+)?(?=\s|$)|$)",
    flags=re.DOTALL,
)


def split_story_sentences(text: str) -> tuple[str, ...]:
    """Split one short story while retaining terminal punctuation."""

    normalized = " ".join(text.replace(_END_OF_TEXT, " ").split())
    if not normalized:
        return ()
    return tuple(
        sentence.strip()
        for sentence in _SENTENCE_PATTERN.findall(normalized)
        if sentence.strip()
    )


def _iter_delimited_text(path: Path) -> Iterator[str]:
    """Read the official multi-line text without treating lines as stories."""

    buffer = ""
    with path.open("r", encoding="utf-8") as stream:
        while chunk := stream.read(1024 * 1024):
            buffer += chunk
            while _END_OF_TEXT in buffer:
                story, buffer = buffer.split(_END_OF_TEXT, 1)
                story = story.strip()
                if story:
                    yield story
    trailing = buffer.strip()
    if trailing:
        yield trailing


def _iter_jsonl(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                text = row["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"invalid TinyStories JSONL row at line {line_number}"
                ) from error
            if not isinstance(text, str):
                raise ValueError(
                    f"TinyStories text must be a string at line {line_number}"
                )
            if text.strip():
                yield text


def _iter_parquet(path: Path) -> Iterator[str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "reading TinyStories Parquet requires pyarrow; use the official "
            "TinyStories-train.txt file or install pyarrow"
        ) from error
    parquet_file = parquet.ParquetFile(path)
    if "text" not in parquet_file.schema.names:
        raise ValueError(f"TinyStories Parquet has no 'text' column: {path}")
    for batch in parquet_file.iter_batches(
        batch_size=4096,
        columns=["text"],
    ):
        for text in batch.column(0).to_pylist():
            if isinstance(text, str) and text.strip():
                yield text


def _story_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise FileNotFoundError(f"TinyStories path does not exist: {path}")
    files = tuple(
        sorted(
            (
                file
                for file in path.iterdir()
                if file.suffix.lower()
                in {".txt", ".jsonl", ".json", ".parquet"}
            ),
            key=lambda file: file.name,
        )
    )
    if not files:
        raise ValueError(f"TinyStories directory contains no data files: {path}")
    return files


def iter_tinystories(
    path: str | Path,
    *,
    skip_stories: int = 0,
    max_stories: int | None = None,
) -> Iterator[tuple[int, str]]:
    """Yield complete stories from official text, JSONL, or Parquet files."""

    if skip_stories < 0:
        raise ValueError("skip_stories must be nonnegative")
    if max_stories is not None and max_stories < 0:
        raise ValueError("max_stories must be nonnegative or None")
    skipped = yielded = story_id = 0
    for file in _story_files(Path(path)):
        suffix = file.suffix.lower()
        if suffix == ".txt":
            stories = _iter_delimited_text(file)
        elif suffix in {".json", ".jsonl"}:
            stories = _iter_jsonl(file)
        elif suffix == ".parquet":
            stories = _iter_parquet(file)
        else:  # guarded by _story_files
            continue
        for text in stories:
            current_id = story_id
            story_id += 1
            if skipped < skip_stories:
                skipped += 1
                continue
            if max_stories is not None and yielded >= max_stories:
                return
            yield current_id, text
            yielded += 1


def tinystories_sentence_pairs(
    stories: Iterable[tuple[int, str]],
    *,
    context_tokens: int,
    response_tokens: int,
    source: str = TINYSTORIES_SOURCE,
) -> Iterator[AssistantExample]:
    """Build progressively longer prefix/continuation sentence pairs.

    A 64-token context permits the first two-sentence prefix. Every additional
    64 context slots adds one more two-sentence prefix boundary from the same
    story. Response capacity scales similarly, with at least one sentence pair.
    Final token fit remains the trainer filter's responsibility.
    """

    if context_tokens <= 0 or response_tokens <= 0:
        raise ValueError("context and response tokens must be positive")
    context_pairs = max(1, (context_tokens + 63) // 64)
    response_pairs = max(1, (response_tokens + 63) // 64)
    response_sentences = 2 * response_pairs
    for story_id, text in stories:
        sentences = split_story_sentences(text)
        for pair_index in range(1, context_pairs + 1):
            boundary = 2 * pair_index
            if boundary >= len(sentences):
                break
            response = sentences[
                boundary : min(
                    len(sentences),
                    boundary + response_sentences,
                )
            ]
            if not response:
                break
            yield AssistantExample(
                conversation_id=story_id,
                turn_index=boundary,
                context=(
                    DialogueTurn(
                        role="user",
                        text=" ".join(sentences[:boundary]),
                    ),
                ),
                response=" ".join(response),
                source=source,
            )


def tinystories_conversations(
    stories: Iterable[tuple[int, str]],
    *,
    context_tokens: int,
    response_tokens: int,
    source: str = TINYSTORIES_SOURCE,
) -> Iterator[Conversation]:
    """Expose sentence pairs to the shared tokenizer conversation API."""

    for index, example in enumerate(
        tinystories_sentence_pairs(
            stories,
            context_tokens=context_tokens,
            response_tokens=response_tokens,
            source=source,
        )
    ):
        yield Conversation(
            conversation_id=index,
            turns=(
                *example.context,
                DialogueTurn(role="assistant", text=example.response),
            ),
            source=source,
        )
