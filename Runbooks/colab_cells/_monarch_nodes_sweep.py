import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

from gator.models.language import GatorLanguageConfig, GatorLanguageModel

FIXED = dict(
    vocab_size=128256,
    d_model=1312,
    embedding_dim=2624,
    lexical_embedding_dim=256,
    n_cells=8,
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
    node_topology="symmetric",
    block_causal=True,
    block_size=16,
    position_reference_length=4096,
    dense_node_backend="monarch",
    monarch_block_size=32,
    node_compute_precision="bf16",
)


def total_params(n_nodes: int):
    cfg = GatorLanguageConfig(n_nodes=n_nodes, **FIXED)
    model = GatorLanguageModel(cfg)
    return sum(p.numel() for p in model.parameters())


TARGET = 100_000_000

print("n_nodes  total_params")
results = {}
for n in (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256):
    try:
        t = total_params(n)
        results[n] = t
        print(f"{n:7d}  {t:,}")
    except Exception as e:
        print(f"{n:7d}  FAILED: {e}")

below = max((n for n, t in results.items() if t <= TARGET), default=None)
above = min((n for n, t in results.items() if t >= TARGET), default=None)
print(f"\nbracket: below={below} ({results.get(below)}), above={above} ({results.get(above)})")

if below is not None and above is not None and below != above:
    lo, hi = below, above
    best_n, best_t, best_diff = None, None, None
    for _ in range(14):
        mid = (lo + hi) // 2
        if mid == lo or mid == hi:
            break
        t = total_params(mid)
        print(f"  try n_nodes={mid:5d}  total={t:,}  diff={t - TARGET:,}")
        diff = abs(t - TARGET)
        if best_diff is None or diff < best_diff:
            best_n, best_t, best_diff = mid, t, diff
        if t < TARGET:
            lo = mid
        else:
            hi = mid
    for n in (lo, hi):
        t = results.get(n) or total_params(n)
        diff = abs(t - TARGET)
        if best_diff is None or diff < best_diff:
            best_n, best_t, best_diff = n, t, diff
    print(f"\nBEST n_nodes={best_n}  total_params={best_t:,}  diff_from_100M={best_t - TARGET:,}")
