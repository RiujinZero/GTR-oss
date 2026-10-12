import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

from transformers import AutoTokenizer

from gator.models.language import GatorLanguageConfig, GatorLanguageModel

NEW_TOKENIZER_PATH = (
    "/mnt/c/Users/user/Documents/projects/GTR/Runbooks/tokenizers/gtr_arw_english_50k"
)
tokenizer = AutoTokenizer.from_pretrained(NEW_TOKENIZER_PATH)
VOCAB_SIZE = len(tokenizer)

FIXED = dict(
    vocab_size=VOCAB_SIZE,
    lexical_embedding_dim=256,
    n_cells=8,
    n_nodes=4,
    relays=2,
    neighbors=4,
    mask_token_id=tokenizer.eos_token_id,
    lexical_mode="private_hash",
    cell_input_mode="shared_projection",
    cell_output_mode="shared_pool",
    cell_transport_mode="broadcast_mix",
    stewardship_execution="dense",
    receiver_memory_backend="linear_scan",
    receiver_memory_hgrn_scan=True,
    dense_kernel_padding=8,
    node_topology="asymmetric",
    block_causal=True,
    block_size=16,
    position_reference_length=4096,
    dense_node_backend="folded_bmm",
    node_compute_precision="bf16",
)


def core_and_total(d_model: int):
    cfg = GatorLanguageConfig(d_model=d_model, embedding_dim=2 * d_model, **FIXED)
    model = GatorLanguageModel(cfg)
    total = sum(p.numel() for p in model.parameters())
    lexical_path = (
        model.token_embedding.weight.numel()
        + (
            model.lexical_projection.weight.numel()
            if isinstance(model.lexical_projection, __import__("torch").nn.Linear)
            else 0
        )
        + model.lm_bias.numel()
    )
    return total, total - lexical_path


TARGET_CORE = 660_000

print("d_model  embedding_dim  total  core")
candidates = [16, 32, 64, 96, 128, 160, 200]
results = {}
for d in candidates:
    d = (d // 8) * 8 or 8
    total, core = core_and_total(d)
    results[d] = core
    print(f"{d:7d}  {2*d:13d}  {total:,}  {core:,}")

below = max((d for d, c in results.items() if c <= TARGET_CORE), default=None)
above = min((d for d, c in results.items() if c >= TARGET_CORE), default=None)
print(f"\nbracket: below={below} ({results.get(below)}), above={above} ({results.get(above)})")

if below is not None and above is not None and below != above:
    lo, hi = below, above
    best_d, best_core, best_diff = None, None, None
    for _ in range(10):
        mid = ((lo + hi) // 2 // 8) * 8 or 8
        if mid == lo or mid == hi:
            break
        total, core = core_and_total(mid)
        print(f"  try d_model={mid:5d}  total={total:,}  core={core:,}  diff={core - TARGET_CORE:,}")
        diff = abs(core - TARGET_CORE)
        if best_diff is None or diff < best_diff:
            best_d, best_core, best_diff = mid, core, diff
        if core < TARGET_CORE:
            lo = mid
        else:
            hi = mid
    for d in (lo, hi):
        total, core = core_and_total(d)
        diff = abs(core - TARGET_CORE)
        if best_diff is None or diff < best_diff:
            best_d, best_core, best_diff = d, core, diff
    total, core = core_and_total(best_d)
    print(f"\nBEST d_model={best_d}  embedding_dim={2*best_d}  total_params={total:,}  core={core:,}  diff_from_target={core - TARGET_CORE:,}")
