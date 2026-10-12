import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

from gator.models.language import GatorLanguageConfig, GatorLanguageModel

# Production config decided 2026-08-17 (Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md):
# 50k-vocab English tokenizer (not the 128,256-vocab Llama tokenizer this
# sweep pattern was originally used for), dense folded_bmm node backend
# (not monarch -- toy-scale investigation concluded relays=2 is the
# production choice, and monarch was never re-validated against the
# final chat-only/lmsys findings), relays=2 (GTR-ARW default).
FIXED = dict(
    vocab_size=50000,
    lexical_embedding_dim=256,
    n_cells=8,
    n_nodes=4,
    relays=2,
    neighbors=4,
    mask_token_id=3,
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


def total_params(d_model: int) -> int:
    embedding_dim = 2 * d_model
    cfg = GatorLanguageConfig(d_model=d_model, embedding_dim=embedding_dim, **FIXED)
    model = GatorLanguageModel(cfg)
    return sum(p.numel() for p in model.parameters())


TARGET = 100_000_000

print("d_model  embedding_dim  total_params")
# known reference: d_model=1312 -> 79,678,064 at this vocab/backend (from
# the earlier english50k smoke test), so start above that and bracket up
candidates = [1312, 1472, 1632, 1792, 1952, 2112, 2272]
results = {}
for d in candidates:
    d = (d // 32) * 32
    t = total_params(d)
    results[d] = t
    print(f"{d:7d}  {2*d:13d}  {t:,}")

below = max((d for d, t in results.items() if t <= TARGET), default=None)
above = min((d for d, t in results.items() if t >= TARGET), default=None)
print(f"\nbracket: below={below} ({results.get(below)}), above={above} ({results.get(above)})")

if below is not None and above is not None and below != above:
    lo, hi = below, above
    best_d, best_t, best_diff = None, None, None
    for _ in range(12):
        mid = ((lo + hi) // 2 // 32) * 32
        if mid == lo or mid == hi:
            break
        t = total_params(mid)
        print(f"  try d_model={mid:7d}  total={t:,}  diff={t - TARGET:,}")
        diff = abs(t - TARGET)
        if best_diff is None or diff < best_diff:
            best_d, best_t, best_diff = mid, t, diff
        if t < TARGET:
            lo = mid
        else:
            hi = mid
    for d in (lo, hi):
        t = results.get(d) or total_params(d)
        diff = abs(t - TARGET)
        if best_diff is None or diff < best_diff:
            best_d, best_t, best_diff = d, t, diff
    print(f"\nBEST d_model={best_d}  embedding_dim={2*best_d}  total_params={best_t:,}  diff_from_100M={best_t - TARGET:,}")
else:
    print("TARGET not bracketed by candidates -- widen the candidate list")
