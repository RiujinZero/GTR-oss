import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

from transformers import AutoTokenizer

from gator.models.language import GatorLanguageConfig, GatorLanguageModel

NEW_TOKENIZER_PATH = (
    "/mnt/c/Users/user/Documents/projects/GTR/Runbooks/tokenizers/gtr_arw_english_50k"
)

tokenizer = AutoTokenizer.from_pretrained(NEW_TOKENIZER_PATH)
vocab_size = len(tokenizer)
print(f"new tokenizer vocab_size={vocab_size}")

common = dict(
    vocab_size=vocab_size,
    d_model=1312,
    embedding_dim=2624,
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
dense_token_embedding = dense.token_embedding.weight.numel()

print(f"dense total params:   {dense_total:,}")
print(f"monarch total params: {monarch_total:,}")
print(f"token_embedding params (both, tied to vocab_size x lexical_dim): {dense_token_embedding:,}")
print()
print("--- for comparison, at the old 128,256-vocab Llama tokenizer ---")
print("dense total params:   99,789,856")
print("monarch total params: 47,771,680")
print("old token_embedding:  32,833,536  (+ lm_bias 128,256 = 32,961,792)")
