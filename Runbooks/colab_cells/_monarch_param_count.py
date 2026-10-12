import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

from gator.models.language import GatorLanguageConfig, GatorLanguageModel

common = dict(
    vocab_size=128256,
    d_model=1312,
    embedding_dim=2624,
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
)

dense = GatorLanguageModel(
    GatorLanguageConfig(**common, dense_node_backend="folded_bmm", node_compute_precision="bf16")
)
monarch = GatorLanguageModel(
    GatorLanguageConfig(
        **common, dense_node_backend="monarch", monarch_block_size=32,
        node_compute_precision="bf16",
    )
)

dense_total = sum(p.numel() for p in dense.parameters())
monarch_total = sum(p.numel() for p in monarch.parameters())

dense_node_weight = dense.node_weight.numel()
monarch_factors = monarch.monarch_left.numel() + monarch.monarch_right.numel()

print(f"dense total params:   {dense_total:,}")
print(f"monarch total params: {monarch_total:,}")
print(f"difference:           {dense_total - monarch_total:,}")
print()
print(f"dense node_weight alone:        {dense_node_weight:,}")
print(f"monarch_left + monarch_right:   {monarch_factors:,}")
print(f"node-compute-only difference:   {dense_node_weight - monarch_factors:,}")
