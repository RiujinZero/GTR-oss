"""Check for actual Dynamo graph breaks in the real GatorLanguageModel
forward (not the isolated microbenchmark) with receiver_memory_hgrn_scan
enabled, using the same --compile-mode max-autotune-no-cudagraphs the
production run uses. torch._dynamo.explain reports every break and its
reason -- this tells us whether the compiled_chunk_hgrn swap actually
eliminated the graph break in situ, or whether something else in the
real call path (or the op itself, under the full autograd/compile
stack rather than the isolated test) still breaks.
"""

import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch
import torch._dynamo

from gator.models.language import GatorLanguageConfig, GatorLanguageModel


def build_config(hgrn_scan: bool) -> GatorLanguageConfig:
    return GatorLanguageConfig(
        vocab_size=64,
        d_model=32,
        embedding_dim=64,
        lexical_embedding_dim=16,
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
        receiver_memory_hgrn_scan=hgrn_scan,
        dense_node_backend="folded_bmm",
        dense_kernel_padding=8,
        node_topology="asymmetric",
        block_causal=True,
        block_size=16,
        position_reference_length=64,
        node_compute_precision="fp8_te",
        readout_precision="fp8_te",
        broadcast_mix_cuda_graph=True,
        stewardship_traffic_cuda_graph=True,
    )


def run(hgrn_scan: bool, label: str):
    print(f"\n=== {label} (receiver_memory_hgrn_scan={hgrn_scan}) ===")
    torch.manual_seed(0)
    config = build_config(hgrn_scan)
    model = GatorLanguageModel(config).cuda().train()
    context = torch.randint(4, config.vocab_size, (2, 60), device="cuda")
    mask = torch.ones_like(context, dtype=torch.bool)

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(model)(context, mask, output_slots=4)
    print("graph_break_count:", explanation.graph_break_count)
    print("graph_count:", explanation.graph_count)
    for reason in explanation.break_reasons:
        print("---")
        print("reason:", reason.reason)
        print("user_stack (last 3 frames):")
        for frame in reason.user_stack[-3:]:
            print("   ", frame)


run(False, "baseline decayed_scan")
run(True, "compiled_chunk_hgrn (hgrn_scan=True)")
