"""Verify the pruned English tokenizer: vocab size, bos/eos ids, lossless
round-trip on English and on deliberately non-English/rare-character text
(via byte-fallback), and compression ratio vs. the original Llama tokenizer.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

from transformers import AutoTokenizer

NEW_TOKENIZER_PATH = (
    "/mnt/c/Users/user/Documents/projects/GTR/Runbooks/tokenizers/gtr_arw_english_50k"
)
OLD_TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"

ENGLISH_SAMPLES = [
    "The quick brown fox jumps over the lazy dog. This sentence contains "
    "every letter of the English alphabet at least once, which makes it a "
    "useful pangram for testing typefaces and tokenizers alike.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return "
    "fibonacci(n - 1) + fibonacci(n - 2)\n\nprint([fibonacci(i) for i in range(10)])",
    "In 1969, the Apollo 11 mission successfully landed the first humans "
    "on the Moon. Neil Armstrong and Buzz Aldrin spent about two and a "
    "half hours outside the lunar module, while Michael Collins orbited "
    "above in the command module.",
    "SELECT users.id, users.name, COUNT(orders.id) AS order_count FROM "
    "users LEFT JOIN orders ON orders.user_id = users.id GROUP BY "
    "users.id ORDER BY order_count DESC LIMIT 10;",
]

NON_ENGLISH_SAMPLES = [
    "héllo wörld, ceci est un texte en français avec des accents.",
    "中文测试：这是一个包含汉字的句子，用于验证字节级回退机制。",
    "Русский текст для проверки токенизатора на кириллице.",
    "emoji stress test 🎉🚀🐍💥🔥 and math symbols ∑ ∫ √ ≈ ≠ π",
    "日本語のテキストです。絵文字も含みます 😊",
]


def compression_ratio(tokenizer, samples: list[str]) -> float:
    total_chars = sum(len(s) for s in samples)
    total_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in samples)
    return total_chars / total_tokens


def main() -> None:
    new_tok = AutoTokenizer.from_pretrained(NEW_TOKENIZER_PATH)
    old_tok = AutoTokenizer.from_pretrained(OLD_TOKENIZER_NAME)

    print(f"new tokenizer vocab size: {len(new_tok)}")
    print(f"old tokenizer vocab size: {len(old_tok)}")
    print(f"new bos_token_id={new_tok.bos_token_id} ({new_tok.bos_token!r})")
    print(f"new eos_token_id={new_tok.eos_token_id} ({new_tok.eos_token!r})")

    print("\n--- English round-trip ---")
    all_ok = True
    for text in ENGLISH_SAMPLES:
        ids = new_tok.encode(text, add_special_tokens=True)
        decoded = new_tok.decode(ids, skip_special_tokens=True)
        ok = decoded == text
        all_ok &= ok
        bos_ok = ids[0] == new_tok.bos_token_id
        print(f"roundtrip_ok={ok} bos_prepended={bos_ok} n_tokens={len(ids)} text={text[:50]!r}...")

    print("\n--- Non-English / rare-character round-trip (byte-fallback expected) ---")
    for text in NON_ENGLISH_SAMPLES:
        ids = new_tok.encode(text, add_special_tokens=True)
        decoded = new_tok.decode(ids, skip_special_tokens=True)
        ok = decoded == text
        all_ok &= ok
        print(f"roundtrip_ok={ok} n_tokens={len(ids)} text={text!r}")

    print("\n--- Compression ratio (chars/token, higher = more compact) ---")
    eng_new = compression_ratio(new_tok, ENGLISH_SAMPLES)
    eng_old = compression_ratio(old_tok, ENGLISH_SAMPLES)
    non_eng_new = compression_ratio(new_tok, NON_ENGLISH_SAMPLES)
    non_eng_old = compression_ratio(old_tok, NON_ENGLISH_SAMPLES)
    print(f"English:     new={eng_new:.3f} old={eng_old:.3f} (ratio new/old={eng_new/eng_old:.3f})")
    print(f"Non-English: new={non_eng_new:.3f} old={non_eng_old:.3f} (ratio new/old={non_eng_new/non_eng_old:.3f})")

    print(f"\nALL ROUNDTRIPS OK: {all_ok}")
    if not all_ok:
        raise SystemExit("tokenizer round-trip verification FAILED")


if __name__ == "__main__":
    main()
