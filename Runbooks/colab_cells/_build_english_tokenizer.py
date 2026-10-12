"""Scan the English portion of the SmolLM3 training mixture and train a
fresh, smaller BPE tokenizer that reuses Llama 3.2's exact pre-tokenizer/
decoder/post-processor rules (only the learned vocab/merge table shrinks).

Usage:
    python _build_english_tokenizer.py [--target-bytes N] [--vocab-size N]
        [--corpus-path P] [--output-dir P] [--stage {1,2,3}]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = "/mnt/c/Users/user/Documents/projects/GTR"
sys.path.insert(0, REPO_ROOT)

from gator.smollm3_recipe import STAGES, HFStreamingTextSource

LLAMA_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"


def extract_text(row, spec) -> str | None:
    for key in spec.text_columns:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    parts = HFStreamingTextSource._semantic_parts(row)
    if parts:
        return "\n\n".join(parts)
    return None


def scan_corpus(*, stage_sources, target_bytes: int, corpus_path: Path) -> int:
    from datasets import load_dataset

    sources = [spec for spec in stage_sources if not spec.name.startswith("fw2-")]
    total_weight = sum(spec.weight for spec in sources)
    written = 0
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as out:
        for spec in sources:
            if written >= target_bytes:
                break
            target_for_source = int(target_bytes * spec.weight / total_weight)
            if target_for_source < 1024:
                continue
            print(
                f"[scan] source={spec.name} target_bytes={target_for_source}",
                flush=True,
            )
            try:
                dataset = load_dataset(
                    spec.repo,
                    spec.config,
                    split=spec.split,
                    revision=spec.revision,
                    streaming=True,
                    **dict(spec.load_kwargs),
                )
            except Exception as error:
                print(f"[scan] source={spec.name} FAILED to open: {error}", flush=True)
                continue
            source_written = 0
            try:
                for row in dataset:
                    text = extract_text(row, spec)
                    if not text:
                        continue
                    out.write(text)
                    out.write("\n")
                    n = len(text.encode("utf-8")) + 1
                    source_written += n
                    written += n
                    if source_written >= target_for_source:
                        break
            except Exception as error:
                print(
                    f"[scan] source={spec.name} stream error after "
                    f"{source_written} bytes: {error}",
                    flush=True,
                )
            print(
                f"[scan] source={spec.name} done bytes={source_written} "
                f"total={written}",
                flush=True,
            )
    print(f"[scan] TOTAL bytes written={written}", flush=True)
    return written


def train_tokenizer(*, corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(LLAMA_SPLIT_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[BOS_TOKEN, EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"[train] training BPE on {corpus_path} (vocab_size={vocab_size})", flush=True)
    tokenizer.train([str(corpus_path)], trainer)

    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if bos_id is None or eos_id is None:
        raise RuntimeError("trained tokenizer is missing bos/eos special tokens")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A",
        pair=f"{BOS_TOKEN} $A {BOS_TOKEN} $B",
        special_tokens=[(BOS_TOKEN, bos_id)],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": BOS_TOKEN,
                "eos_token": EOS_TOKEN,
                "add_bos_token": True,
                "clean_up_tokenization_spaces": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": BOS_TOKEN, "eos_token": EOS_TOKEN}, indent=2),
        encoding="utf-8",
    )
    print(
        f"[train] saved tokenizer to {output_dir}, "
        f"vocab_size={tokenizer.get_vocab_size()} bos_id={bos_id} eos_id={eos_id}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-bytes", type=int, default=9 * 1024**3)
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("/home/user/gtr_english_tokenizer_corpus.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(REPO_ROOT) / "Runbooks" / "tokenizers" / "gtr_arw_english_50k",
    )
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="reuse an existing corpus file instead of re-streaming",
    )
    args = parser.parse_args()

    stage_sources = STAGES[args.stage - 1]

    if not args.skip_scan:
        scan_corpus(
            stage_sources=stage_sources,
            target_bytes=args.target_bytes,
            corpus_path=args.corpus_path,
        )
    else:
        print(f"[scan] skipped, reusing {args.corpus_path}", flush=True)

    train_tokenizer(
        corpus_path=args.corpus_path,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
