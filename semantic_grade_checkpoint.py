"""Grade a Gator checkpoint's generations: whitespace-collapse diagnostic +
MiniLM semantic-similarity scoring against references.

Usage (run on the machine/session with the checkpoint + eval data available):
    python3 semantic_grade_checkpoint.py \
        --checkpoint /path/to/latest.pt \
        --data /path/to/eval.jsonl \
        --num-examples 50 \
        --response-tokens 128 \
        --chunk-size 8

Works for decode_mode in {serial, parallel, chunk_parallel} — auto-detected
from the checkpoint's own model_config.
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from gator_convergent import ConvergentGator
from gator.tokenization import CompactTokenizer

_ACCEPTED_KEYS = set(inspect.signature(ConvergentGator.__init__).parameters.keys()) - {"self"}


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = dict(ckpt["model_config"])
    filtered = {k: v for k, v in model_config.items() if k in _ACCEPTED_KEYS}
    state_dict = ckpt["model"]
    if any(k.startswith("core.") for k in state_dict):
        state_dict = {k[len("core."):]: v for k, v in state_dict.items() if k.startswith("core.")}
        filtered.setdefault("vocabulary_mode", "shared_projected")
    model = ConvergentGator(**filtered).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = CompactTokenizer.from_dict(ckpt["tokenizer"])
    data_config = ckpt.get("data_config", {})
    task_config = ckpt.get("task_config", {})
    decode_mode = data_config.get("decode_mode", "parallel")
    chunk_size = data_config.get("chunk_size", 8)
    response_tokens = task_config.get("response_tokens", 32)
    context_tokens = task_config.get("context_tokens", 64)
    return model, tokenizer, decode_mode, chunk_size, response_tokens, context_tokens


def load_examples(path, limit):
    examples = []
    with open(path) as f:
        for line in f:
            if len(examples) >= limit:
                break
            row = json.loads(line)
            text = row.get("text", "")
            if "USER:" in text and "ASSISTANT:" in text:
                prompt, target = text.split("ASSISTANT:", 1)
                prompt = prompt.strip()
                target = target.strip()
                if prompt and target:
                    examples.append((prompt, target))
    return examples


@torch.no_grad()
def free_generate(model, tokenizer, device, prompt, *, decode_mode, chunk_size,
                   response_tokens, context_tokens):
    prompt_ids = tokenizer.encode(prompt)
    context_ids = list(prompt_ids)
    generated_ids = []

    def forward_window(n_slots):
        window = context_ids[-context_tokens:]
        pad = max(0, context_tokens - len(window))
        ids_tensor = torch.tensor([[0] * pad + window], device=device, dtype=torch.long)
        mask_tensor = torch.tensor([[0] * pad + [1] * len(window)], device=device, dtype=torch.bool)
        logits = model(ids_tensor, mask_tensor, output_slots=n_slots, causal=True)
        return logits[0].argmax(dim=-1).tolist()

    if decode_mode == "parallel":
        generated_ids = forward_window(response_tokens)
    elif decode_mode == "chunk_parallel":
        remaining = response_tokens
        while remaining > 0:
            n = min(chunk_size, remaining)
            chunk_ids = forward_window(n)
            generated_ids.extend(chunk_ids)
            context_ids.extend(chunk_ids)
            remaining -= n
            if tokenizer.EOS_ID in chunk_ids:
                break
    else:  # serial
        for _ in range(response_tokens):
            next_id = forward_window(1)[0]
            generated_ids.append(next_id)
            context_ids.append(next_id)
            if next_id == tokenizer.EOS_ID:
                break

    return tokenizer.decode(generated_ids)


def whitespace_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ch.isspace()) / len(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--no-semantic", action="store_true",
                         help="skip MiniLM scoring, only report whitespace fraction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, decode_mode, chunk_size, response_tokens, context_tokens = load_checkpoint(
        args.checkpoint, device
    )
    print(f"decode_mode={decode_mode} chunk_size={chunk_size} "
          f"response_tokens={response_tokens} context_tokens={context_tokens}", flush=True)

    examples = load_examples(args.data, args.num_examples)
    print(f"loaded {len(examples)} eval examples", flush=True)

    generations = []
    for prompt, target in examples:
        generated_text = free_generate(
            model, tokenizer, device, prompt,
            decode_mode=decode_mode, chunk_size=chunk_size,
            response_tokens=response_tokens, context_tokens=context_tokens,
        )
        generations.append((prompt, target, generated_text))

    ws_fractions = [whitespace_fraction(gen) for _, _, gen in generations]
    mean_ws = sum(ws_fractions) / len(ws_fractions) if ws_fractions else 0.0
    print(f"\nmean_whitespace_fraction={mean_ws:.4f} "
          f"(examples with >90% whitespace: "
          f"{sum(1 for f in ws_fractions if f > 0.9)}/{len(ws_fractions)})", flush=True)

    print("\n--- sample generations ---", flush=True)
    for prompt, target, gen in generations[:10]:
        print(f"prompt={prompt[:80]!r}", flush=True)
        print(f"target={target[:80]!r}", flush=True)
        print(f"generated={gen[:80]!r}", flush=True)
        print(f"whitespace_fraction={whitespace_fraction(gen):.3f}\n", flush=True)

    if args.no_semantic:
        return

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\nsentence-transformers not installed; skipping semantic scoring. "
              "Install with: pip install sentence-transformers", flush=True)
        return

    print("\nloading MiniLM (all-MiniLM-L6-v2)...", flush=True)
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

    targets = [t for _, t, _ in generations]
    gens = [g for _, _, g in generations]
    target_emb = encoder.encode(targets, convert_to_tensor=True, normalize_embeddings=True)
    gen_emb = encoder.encode(
        [g if g.strip() else " " for g in gens], convert_to_tensor=True, normalize_embeddings=True
    )
    similarities = (target_emb * gen_emb).sum(dim=-1).tolist()
    mean_sim = sum(similarities) / len(similarities)

    print(f"\nmean_semantic_similarity={mean_sim:.4f}", flush=True)
    print("per-example similarity (first 10):", flush=True)
    for (prompt, target, gen), sim in list(zip(generations, similarities))[:10]:
        print(f"  sim={sim:.3f} target={target[:60]!r} generated={gen[:60]!r}", flush=True)


if __name__ == "__main__":
    main()
