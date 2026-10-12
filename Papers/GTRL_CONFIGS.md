# GTR-L configurations

Status: **active measurement log**

This file records reproducible GTR-L architecture configurations separately
from training-quality results. Parameter counts are exact for the listed
implementation and settings. Throughput and VRAM figures are empirical and
must always include the GPU, batch sizes, context, precision, and compilation
mode.

## Configuration summary

| ID | Parameters | Width | Streams | Nodes | Slots | Training window | Train batch | Eval batch | GPU | VRAM | Throughput | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| `s4_s16_n128_d1392_100m` | 99,390,373 | 1,392 | 16 | 128 | 4 | 2,048 | 1 | 1 | NVIDIA RTX PRO 6000 | 12.7 GB | approximately 4.2 steps/s | measured |
| `s4_s16_n128_d3440_500m` | 499,037,093 | 3,440 | 16 | 128 | 4 | 2,048 | 1 | 1 | NVIDIA RTX PRO 6000 | 30.3 GB peak | approximately 1.9 steps/s | measured |

Both configurations use a 20,000-token vocabulary, tied token embeddings,
four relays, BF16 autocast, full-graph compilation, and the block-causal
cache. The observed prefix is 2,044 tokens; the four-token active output block
makes the complete training window 2,048 tokens.

The 12.7 GB and approximately 4.2 steps/s measurements were reported on
2026-08-04. The exact VRAM counter used and whether the throughput interval
excluded all compilation warmup have not yet been recorded; preserve those
details for subsequent measurements.

The 500M configuration reached an observed 30.3 GB peak and approximately
1.9 steady-state training steps/s on the same reported runtime configuration.

## 100M: `s4_s16_n128_d1392_100m`

### Architecture

| Setting | Value |
|---|---:|
| Parameters | 99,390,373 |
| Vocabulary | 20,000 |
| Tied token embeddings | yes |
| Width | 1,392 |
| Streams | 16 |
| Nodes | 128 |
| Relays | 4 |
| Graph degree | 6 |
| Buses | 4 |
| Mailbox channels | 4 |
| Mailbox width | 8 |
| Basis rank | 4 |
| Address width | 4 |
| Router rank | 8 |
| Linear-attention key width | 8 |
| Output slots | 4 |
| Observed prefix | 2,044 |
| Training window | 2,048 |
| Block causal | yes |
| Hard routing | yes |
| Controlled writing | yes |
| Private stream I/O | yes |
| Private stream rank | 4 |
| Heterogeneous streams | no |

### Runtime measurement

| Setting | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 |
| Train batch | 1 |
| Evaluation batch | 1 |
| Precision | BF16 autocast |
| Compilation | `torch.compile`, full graph |
| VRAM | 12.7 GB |
| Training throughput | approximately 4.2 steps/s |
| Measurement date | 2026-08-04 |

### Launcher arguments

```python
[
    "--experiment-tag", "gtrl_s4_s16_n128_d1392_100m",
    "--tie-token-embeddings",
    "--block-causal",
    "--vocab", "20000",
    "--streams", "16",
    "--nodes", "128",
    "--width", "1392",
    "--relays", "4",
    "--degree", "6",
    "--output-slots", "4",
    "--observed", "2044",
    "--batch-size", "1",
    "--eval-batch-size", "1",
    "--amp-dtype", "bf16",
    "--compile",
    "--compile-fullgraph",
]
```

`--context-length` is intentionally omitted. The launcher derives it as
`observed + output_slots = 2,048`.

## 500M: `s4_s16_n128_d3440_500m`

Status: **measurement pending**

This configuration changes only width relative to the 100M configuration.
All topology, vocabulary, slot, context, precision, compilation, and batch
settings remain fixed.

### Architecture

| Setting | Value |
|---|---:|
| Parameters | 499,037,093 |
| Vocabulary | 20,000 |
| Tied token embeddings | yes |
| Width | 3,440 |
| Streams | 16 |
| Nodes | 128 |
| Relays | 4 |
| Graph degree | 6 |
| Buses | 4 |
| Mailbox channels | 4 |
| Mailbox width | 8 |
| Basis rank | 4 |
| Address width | 4 |
| Router rank | 8 |
| Linear-attention key width | 8 |
| Output slots | 4 |
| Observed prefix | 2,044 |
| Training window | 2,048 |
| Block causal | yes |
| Hard routing | yes |
| Controlled writing | yes |
| Private stream I/O | yes |
| Private stream rank | 4 |
| Heterogeneous streams | no |

### Runtime measurement

Fill this table after compilation and several optimizer steps so lazy
optimizer-state allocation is included.

| Setting | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 |
| Train batch | 1 |
| Evaluation batch | 1 |
| Precision | BF16 autocast |
| Compilation | `torch.compile`, full graph |
| VRAM before final allocation | 28.6 GB |
| Observed peak VRAM | 30.3 GB |
| Peak allocated VRAM | pending |
| Peak reserved VRAM | pending |
| Training throughput | approximately 1.9 steps/s |
| Compilation time | pending |
| Measurement date | 2026-08-04 |
| Stable or OOM | reached steady state |
| Notes | steady-state throughput recorded |

The observed 30.3 GB peak leaves approximately 1.7 GB of nominal capacity on
a B70 exposing its full 32 GB. The configuration may therefore fit on one
B70, but the margin is narrow and it is unlikely to fit when ECC reduces the
memory visible to software to approximately 28 GB. Confirm with the actual
PyTorch peak-allocated and peak-reserved counters on the B70 rather than
assuming the RTX PRO 6000 allocation pattern transfers exactly.

### Launcher arguments

```python
[
    "--experiment-tag", "gtrl_s4_s16_n128_d3440_500m",
    "--tie-token-embeddings",
    "--block-causal",
    "--vocab", "20000",
    "--streams", "16",
    "--nodes", "128",
    "--width", "3440",
    "--relays", "4",
    "--degree", "6",
    "--output-slots", "4",
    "--observed", "2044",
    "--batch-size", "1",
    "--eval-batch-size", "1",
    "--amp-dtype", "bf16",
    "--compile",
    "--compile-fullgraph",
]
```

## Parameter-count basis

For these fixed settings—20,000-token tied vocabulary, 16 streams, 128
nodes, four relays, and the listed default relay components—the trainable
parameter count is:

```text
parameters(width) = 36 * width^2 + 21,188 * width + 140,773
```

Therefore:

```text
width 1,392 ->  99,390,373 parameters
width 3,440 -> 499,037,093 parameters
```

Output-slot count and context length change activation memory and work per
step but do not change this parameter count.

## Measurement procedure

For future entries:

1. Record the resolved configuration printed by the runner.
2. Allow full-graph compilation to finish.
3. Complete enough optimizer steps for AdamW moments to exist.
4. Exclude evaluation, checkpoint writes, and compilation from the steady
   training-throughput interval.
5. Record both peak allocated and peak reserved VRAM when available.
6. Record whether the displayed VRAM came from PyTorch, `nvidia-smi`, or
   another system monitor.
7. Keep train batch, evaluation batch, context, slots, and precision beside
   every throughput result.
