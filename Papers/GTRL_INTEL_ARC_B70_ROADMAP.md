# GTR-L Intel Arc Pro B70 optimization roadmap

Status: **planned**

Primary target: **Intel Arc Pro B70, one GPU**

Secondary target: **two Arc Pro B70 GPUs in one Linux workstation**

Primary implementation:

- `gator_linear/core.py`
- `gator_linear/prefix.py`
- `gator_linear/evotoken.py`
- `gtrl_evotoken_colab.py`

Reference configuration:

```text
GTR-L, tied embeddings
vocab = 20,000
nodes = 256
width = 1,504
streams = 10
relays = 4
output slots = 8
observed = 2,040
training window = 2,048
BF16 autocast
block-causal cache
```

## Objective

Make GTR-L a first-class PyTorch XPU workload on Intel Arc Pro B70 rather
than a CUDA program that happens to run after replacing a device string.

The work has four ordered goals:

1. Preserve model mathematics and checkpoint compatibility on CPU, CUDA, and
   XPU.
2. Establish correct and stable single-B70 BF16 training and cached inference.
3. Optimize the operations that prevent the B70's XMX engines and memory
   system from being used efficiently.
4. Add dual-B70 support, first for data-parallel throughput and then, only if
   needed, for models or activation sets that do not fit on one card.

Learning behavior remains outside the hardware-port acceptance boundary.
Intel and CUDA runs may not follow identical BF16 trajectories, but the XPU
port must not change the objective, routing equations, refinement schedule, or
block-cache contract.

## Target hardware

Intel lists the Arc Pro B70 with:

| Property | B70 |
|---|---:|
| Architecture | Xe2 |
| Dedicated memory | 32 GB GDDR6 |
| Memory interface | 256 bit |
| Memory bandwidth | 608 GB/s |
| Xe cores | 32 |
| XMX engines | 256 |
| PCIe | 5.0 x16 |
| Reference board power | 230 W |
| Permitted board-power range | 160--290 W |
| ECC | supported |

Sources:

- [Intel Arc Pro B70 data sheet](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf)
- [Intel Arc Pro B-series overview](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html)
- [Intel Arc Pro B-series quick reference](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/intel-arc-pro-b-series-graphics-quick-reference-guide-v1-0.pdf)

ECC may reduce the memory reported to software from 32 GB to approximately
28 GB. Keep ECC enabled for the first training validation and treat the
reported allocator capacity, not the marketing capacity, as the memory
budget. Disabling ECC is a separate reliability/performance decision and
must not be mixed into kernel comparisons.

The published B70 specification exposes PCIe 5.0 x16 but does not specify a
dedicated card-to-card fabric. The dual-GPU plan must therefore assume
communication is PCIe-bound until peer-access and collective benchmarks on
the actual workstation prove otherwise.

Resizable BAR must be enabled. Both cards should be installed in slots with
known electrical lane widths, and the measured link width must be recorded.
Two physical x16 slots do not imply that the CPU supplies x16 to both.

## Platform decision

The primary development and dual-GPU platform is Linux:

- Ubuntu 24.04 or a newer Intel-validated Ubuntu release;
- current Intel client GPU compute driver;
- stock PyTorch XPU wheel pinned to a known-good version;
- TorchInductor as the initial compiler backend;
- XCCL as the distributed XPU backend.

Intel validates a PyTorch XPU container on Ubuntu 24.04 with the B70 and
documents a B-series compute installation path:

- [Intel PyTorch XPU container support](https://dgpu-docs.intel.com/overview/release-notes/containers/pyTorch/2.11.0.html)
- [Intel client GPU installation paths](https://dgpu-docs.intel.com/driver/client/overview.html)
- [PyTorch prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-13.html)

Windows single-GPU support is a secondary validation target. PyTorch supports
XPU eager execution and `torch.compile` on current Windows releases, but
Linux is the lower-risk target for XCCL, profiling, custom kernels, and
eventual dual-GPU training.

Do not make Intel Extension for PyTorch a mandatory dependency. Stock
PyTorch has native XPU, AMP, and Inductor support. IPEX may be benchmarked as
an optional environment after the stock path is correct, but GTR-L should not
depend on model-specific IPEX rewrites.

## Current GTR-L workload

GTR-L is already composed almost entirely from portable ATen operations. It
contains no required handwritten CUDA kernel. That makes functional XPU
support relatively small, but it does not guarantee efficient XPU execution.

The work divides into four kernel families.

### Dense width transforms

Examples:

- input and output token projections;
- stream adapters;
- mailbox write/read projections;
- node-program basis transforms;
- router projections;
- vocabulary projection.

At width 1,504 and above these are the best candidates for B70 XMX
acceleration. BF16 matrix operands should remain BF16 while numerically
sensitive reductions and recurrent state remain FP32.

### Mailbox reductions

The core repeatedly maps between:

```text
[B, L, S, D]
[B, L, S, N]
[B, L, N, C, M]
```

These operations reduce across streams or nodes and are likely to be limited
by tensor traffic, layout conversion, and launch count rather than raw matrix
throughput.

### Sparse routing

Routing uses:

- small-rank edge-score matrix products;
- softmax over constant-degree edges;
- `scatter_add_` into destination nodes;
- low-rank global-bus transport;
- several normalization and gating kernels.

The fixed graph gives an opportunity to replace atomic destination scatter
with precomputed reverse adjacency and deterministic gather/reduction.

### Sequence mixing

Each relay performs:

- grouped depthwise convolution;
- forward decayed scan;
- reverse or block-local reverse scan;
- global or block-cumulative summaries;
- small channel-view transforms.

The current decayed scan constructs chunk-local decay weights and applies
repeated small `einsum` operations. It is mathematically linear in context
length, but it can materialize avoidable temporaries and submit many kernels.
This is the highest-value custom-kernel candidate after profiling.

## Phase 0: freeze evidence and build the benchmark harness

Before changing device handling, record reproducible CPU and CUDA reference
artifacts.

Required benchmark cases:

| Case | Purpose |
|---|---|
| 130k, short context | correctness and rapid iteration |
| 130k, 2,048-token window | scan and address stress |
| 100M, context 512, batch 1 and 4 | comparison with existing measurements |
| 100M, observed 2,040, slots 8, batch 1 | production training target |
| 100M cached generation | prefill, refinement, commit, and append cost |

Every performance record must contain:

- commit and dirty-tree status;
- complete resolved configuration;
- operating system, kernel, driver, PyTorch, and compiler versions;
- device name, available VRAM, ECC state, and PCIe link width;
- eager or compiled mode and compile options;
- warmup steps and measured steps;
- steps/s, observed tokens/s, target slots/s, and relay evaluations/s;
- peak allocated and reserved memory;
- compile latency and recompilation count;
- evaluation disabled/enabled state;
- checkpoint and Drive mirroring disabled/enabled state.

Benchmarks must synchronize the active backend immediately before starting
and stopping timers. A host timer without XPU synchronization is invalid.

Deliverables:

- `benchmark_gtrl_xpu.py`;
- a machine-readable JSON result;
- a short Markdown comparison table generated from the JSON;
- optional PyTorch profiler trace.

## Phase 1: backend-neutral execution

### Device selection

Add a small accelerator abstraction used by the runner:

```text
--device auto
--device cpu
--device cuda
--device cuda:0
--device xpu
--device xpu:0
```

`auto` should prefer CUDA, then XPU, then CPU unless an explicit environment
policy overrides the order. It must print the resolved accelerator before
model construction.

The abstraction should provide:

- availability detection;
- device count and device name;
- synchronization;
- BF16 capability detection;
- manual seeding;
- get/set all-device RNG state;
- allocated, reserved, and peak memory;
- reset peak-memory statistics.

Do not scatter `if device.type == "xpu"` throughout the training loop.

### RNG and checkpoint schema

The current runner directly calls `torch.cuda.manual_seed_all`,
`torch.cuda.get_rng_state_all`, and `torch.cuda.set_rng_state_all`. Replace
the checkpoint field with a versioned backend-neutral record:

```python
"accelerator_rng": {
    "backend": "xpu",
    "states": [...],
}
```

Loading must continue to accept the legacy `cuda_rng` field. A CUDA
checkpoint warm-started on XPU loads model weights and tokenizer state, but
does not pretend CUDA RNG bytes are XPU RNG state.

Exact resume on the same backend must restore:

- CPU RNG;
- accelerator RNG for every local device/rank;
- Python RNG;
- optimizer;
- position-objective EMAs;
- completed step and scheduler state.

### Mixed precision

The existing device-aware autocast call can remain:

```python
torch.autocast(device_type=device.type, dtype=torch.bfloat16)
```

Add an explicit BF16 capability check. FP16 is a fallback experiment, not an
automatic silent replacement for requested BF16.

Keep FP32 for:

- absolute-address construction;
- scan recurrence and normalizers where currently promoted;
- cumulative mailbox and occupancy cache state;
- loss reduction;
- gradient norm calculation;
- optimizer master state as selected by PyTorch.

### Phase 1 acceptance gate

On one B70:

1. XPU is detected without importing CUDA-only packages.
2. The 130k forward, backward, clipping, and AdamW step are finite in FP32.
3. BF16 autocast completes 1,000 training steps without non-finite values.
4. A checkpoint resumes and advances with the expected step and LR.
5. Cached block generation matches vectorized block-causal execution within
   the dtype-specific tolerance.
6. CPU and CUDA tests remain unchanged and passing.

## Phase 2: compiled BF16 single-GPU baseline

Use stock `torch.compile(..., backend="inductor")` first. Test:

```text
XPU FP32 eager
XPU BF16 eager
XPU BF16 compiled
```

The training shapes are static, so the desired final state is one compiled
training graph per architecture/configuration rather than repeated
specialization.

Audit the graph for:

- Python scalar values that change every step;
- data-dependent branches;
- hidden CPU tensor creation;
- unsupported in-place views;
- graph breaks around scan chunk loops;
- repeated compilations caused by evaluation or cache shapes;
- operations silently falling back to the CPU.

`fullgraph=True` remains the strict validation mode. A permissive compiled
mode may be retained for diagnosis, but it cannot be called optimized while
substantial work runs eagerly or on the host.

Provisional gate:

- compiled loss and gradients agree with XPU eager within BF16 tolerance;
- one 1,000-step compiled smoke run is stable;
- no recurring graph compilation after warmup;
- no device-to-host transfer appears in the hot training step;
- compiled throughput is no worse than 95% of eager after warmup.

The 95% threshold is only a port gate. The optimization phases should produce
a positive compiled speedup.

## Phase 3: profile before writing custom kernels

Use `torch.profiler` with CPU and XPU activity. Collect a short trace after
compilation and warmup, not during first-graph compilation.

Current PyTorch exposes `ProfilerActivity.XPU` and Intel hardware counters:

- [PyTorch profiler documentation](https://docs.pytorch.org/docs/main/profiler)
- [Intel VTune XPU offload analysis](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2026-1/xpu-offload-analysis.html)

Answer these questions for both the 130k and 100M models:

1. What percentage of step time is GEMM/oneDNN versus generated pointwise
   kernels?
2. Is the B70 compute-bound, memory-bandwidth-bound, or submission-bound?
3. How much time is spent in decayed scan?
4. How much time is spent in routing `scatter_add_`?
5. Which `einsum` shapes miss XMX acceleration?
6. How much time is spent in AdamW and gradient clipping?
7. How many layout conversions or explicit contiguous copies occur?
8. Does data preparation leave gaps in the XPU timeline?

Do not infer the 100M bottleneck from the 130k trace. Small GTR-L is expected
to be launch-bound, while the width-1,504 model may be dominated by large
matrix operations.

## Phase 4: portable kernel and layout optimizations

Each optimization must remain independently selectable until its numerical
and speed gates pass on CPU, CUDA, and XPU.

### 4A. Atomics-free route transport

The legal graph is fixed. During model construction, precompute reverse edge
lists:

```text
destination node -> [(source node, source edge index), ...]
```

Replace:

```python
zeros_like(occupancy).scatter_add_(...)
```

with bounded gather and reduction over incoming edges. Preserve exact mass
conservation and gradients.

Backends:

```text
--route-transport-backend scatter
--route-transport-backend reverse_gather
--route-transport-backend auto
```

Acceptance:

- output and gradient equivalence in FP32;
- route mass remains normalized;
- no dead destination caused by padding;
- faster or equal on B70 at N=256;
- no material CUDA regression.

### 4B. Rewrite eligible `einsum` operations

Flatten leading dimensions and express dense transforms as `linear`,
`matmul`, or `bmm` when this produces an XMX-friendly matrix:

- node-program basis;
- stream adapters;
- mailbox projections;
- edge query/key products;
- bus input/output;
- view transforms.

Pack projections that consume the same input and split their outputs
afterward. Avoid packing when it extends a large tensor lifetime enough to
increase peak VRAM.

Widths and inner ranks should be benchmarked at alignment-friendly values,
but architecture changes require an A/B learning run. A kernel optimization
may pad an internal dimension with masked zeros; it may not silently change
the configured model width or parameterization.

### 4C. Reduce layout churn

Audit every `permute(...).reshape(...)` pair. Record whether it remains a
view or creates a copy after compilation. Establish one preferred layout for:

- private stream state;
- node mailbox state;
- scan rows;
- routing occupancy.

Do not optimize a local operator by forcing two larger transposes around it.
The unit of evaluation is the whole relay.

### 4D. Optimizer and clipping

Benchmark:

- standard AdamW;
- `foreach` AdamW when supported by the pinned XPU build;
- a compiled optimizer step;
- fused or foreach norm/clipping when numerically equivalent.

Keep the portable implementation as fallback. Optimizer results must include
peak VRAM because a faster path that creates large temporary lists may reduce
the maximum trainable configuration.

## Phase 5: optimized decayed scan

Only begin custom scan work after Phase 3 establishes that scan time is
material at the 100M shape.

The required operation is:

```text
y[t] = x[t] + decay * y[t-1]
```

with independent learned decays over the compact channel dimension, plus:

- forward and reverse directions;
- optional initial recurrent state;
- optional returned final state;
- complete-block reset for the reverse block-causal scan;
- FP32 accumulation with BF16 input/output;
- gradients for values, decay, and initial state;
- static and cached one-block shapes.

The optimized implementation must not materialize the current
`[chunk, chunk, channels]` decay-weight tensor.

Development order:

1. Inductor-friendly hierarchical scan expressed in portable PyTorch.
2. Triton XPU kernel if Inductor does not generate an adequate scan.
3. SYCL/oneAPI extension only if Triton lacks required performance or
   correctness.

Backends:

```text
--scan-backend torch
--scan-backend compiled
--scan-backend triton
--scan-backend auto
```

Required tests:

- reference forward and reverse values;
- arbitrary sequence lengths and final partial chunks;
- initial-state and returned-state agreement;
- autograd and finite-difference gradient checks;
- BF16 drift against FP32 reference;
- vectorized block-causal versus cached append;
- `torch.compile(fullgraph=True)`;
- long-context numerical stability.

Promotion to `auto` requires a speed win on B70 at both context 512 and
2,048. A 130k-only speed win is insufficient.

## Phase 6: single-B70 production gate

The production candidate is the real 100M configuration:

```python
[
    "--device", "xpu:0",
    "--amp-dtype", "bf16",
    "--compile",
    "--compile-backend", "inductor",
    "--tie-token-embeddings",
    "--block-causal",
    "--nodes", "256",
    "--width", "1504",
    "--vocab", "20000",
    "--output-slots", "8",
    "--observed", "2040",
    "--batch-size", "1",
    "--eval-batch-size", "1",
]
```

Production gate:

- 10,000 consecutive steps without non-finite loss or gradients;
- checkpoint, stop, and exact same-backend resume;
- evaluation sample and collapse diagnostics preserved;
- block-cache parity tests pass;
- no hot-path CPU fallback;
- peak memory leaves a documented safety margin below actual available VRAM;
- stable throughput after compilation;
- profiler shows the largest remaining kernels and their utilization;
- CUDA behavior remains available from the same source tree.

## Phase 7: dual-B70 data parallelism

The first dual-GPU target is DistributedDataParallel, not model sharding.

Why:

- it requires the least change to model mathematics;
- each B70 can hold the 100M model independently;
- it increases training throughput;
- it provides an XCCL and checkpoint foundation needed by later sharding.

PyTorch's built-in XPU distributed backend is XCCL. `torchrun` accepts
`--nproc-per-node=xpu`:

- [PyTorch distributed backends](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch torchrun](https://docs.pytorch.org/docs/main/elastic/run.html)

Proposed launch:

```text
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m gtrl_evotoken_colab \
  --device xpu \
  --distributed-strategy ddp \
  --distributed-backend xccl \
  ...
```

### Distributed runner behavior

Each process:

- reads `LOCAL_RANK`, `RANK`, and `WORLD_SIZE`;
- selects exactly one `xpu:{local_rank}`;
- initializes XCCL;
- constructs the model on its local device;
- uses distinct deterministic training RNG derived from base seed and rank;
- receives a distinct training batch;
- participates in gradient reduction;
- participates in evaluation metric reduction.

Rank zero alone:

- prints ordinary progress;
- writes local checkpoints;
- mirrors logs and checkpoints to Drive;
- prints decoded samples.

All ranks must enter evaluation barriers and collective operations in the
same order. Rank-zero-only model evaluation is not allowed unless the DDP
wrapper has been safely bypassed and every other rank follows an explicit
barrier protocol.

### Global batch semantics

DDP changes the effective batch:

```text
global batch = per-rank batch * world size
```

Two validation modes are required:

1. **Equivalence mode:** halve the per-rank batch so the global batch and LR
   match the single-GPU run.
2. **Throughput mode:** preserve per-rank batch, double global batch, and
   treat LR selection as a separate training A/B.

Do not mix a learning-rate scaling change into the distributed correctness
gate.

### Distributed checkpoint schema

Save:

- unwrapped model state, loadable by single GPU or DDP;
- optimizer state in the format appropriate to the selected strategy;
- world size and strategy;
- one RNG/data-generator state per rank;
- global completed step;
- global and per-rank batch configuration;
- tokenizer and objective configuration.

Changing world size may preserve model weights but cannot promise
bit-identical continuation of the data stream. The loader must report this
explicitly.

### Dual-DDP acceptance gate

1. Two ranks initialize and identify different B70 devices.
2. A 130k equivalence test matches the single-GPU update within distributed
   BF16 tolerance.
3. A 1,000-step 100M run is stable.
4. Rank zero writes one coherent checkpoint and both ranks resume it.
5. No duplicate Drive mirrors or duplicate progress logs occur.
6. XCCL all-reduce time is recorded separately.
7. Throughput reaches a provisional target of at least 1.6 times the
   single-B70 rate for a compute-saturating configuration.

If scaling misses 1.6x, inspect batch size, gradient bucket sizing, overlap,
host input supply, PCIe topology, and peer access before changing model
math.

## Phase 8: using both cards for capacity

DDP stores a complete model and optimizer replica on each card. It doubles
throughput potential but does **not** turn two 32 GB cards into one automatic
64 GB allocation.

Capacity scaling is a separate project and should proceed only when a desired
configuration cannot fit on one B70.

### 8A. Sharded optimizer/model state

Evaluate FSDP or the current PyTorch composable sharding API first.

Potential benefit:

- parameters, gradients, and AdamW state can be sharded;
- checkpoint tooling is standardized;
- model equations remain unchanged.

Limitation:

- GTR-L's dominant long-context memory may be activations rather than model
  state;
- all-gather/reduce-scatter crosses PCIe;
- a 100M model is unlikely to justify this complexity solely for parameters.

Before adopting FSDP, measure the single-B70 memory breakdown:

```text
parameters
gradients
optimizer state
saved activations
compiler workspaces
temporary routing/scan tensors
```

### 8B. Stream parallelism

Streams are the most natural GTR-L-specific activation-sharding axis. Split
the ten streams five per B70 while keeping logical output slots intact.

Per relay:

1. Each rank computes its local stream messages and occupancy-weighted
   mailbox contribution.
2. All-reduce mailbox numerator and node mass.
3. Each rank reads the shared mailbox into its local streams.
4. Routing and node-program transforms advance local streams.
5. Shared-parameter gradients are reduced normally.

At output:

1. Compute the global softmax normalization of stream fusion scores using
   distributed max and sum reductions.
2. Each rank forms its local weighted hidden contribution.
3. All-reduce the fused hidden vector.
4. Apply the replicated or separately sharded vocabulary head.

Benefits:

- private stream activations are approximately halved;
- stream adapters and stream-private I/O parameters can be partitioned;
- communication uses the smaller shared mailbox and fused hidden rather than
  all private hidden states.

Costs:

- one or more collectives per relay;
- shared mixer work may remain replicated;
- streams=10 gives only a two-way partition and no useful scaling beyond
  factors of 2 and 5 without uneven shards;
- compiler and autograd support require careful static collective placement.

Stream parallelism is the preferred GTR-L-specific capacity prototype after
FSDP measurement, not before dual DDP is stable.

### 8C. Relay pipeline parallelism

With four distinct relays, two relays can reside on each card. This reduces
relay-parameter residency but transfers hidden state and occupancy between
cards.

Pipeline parallelism is unattractive for batch 1 because pipeline bubbles
erase concurrency. It becomes viable only with enough microbatches, and
EvoToken's intermediate relay losses require an explicit plan for shared
readout placement or replication.

Keep this behind stream parallelism unless memory measurement shows relay
parameters, rather than activations, are the dominant constraint.

### 8D. Width tensor parallelism

Width sharding can divide the large basis, adapter, and vocabulary matrix
work of a future width-4,096 GTR-L. It is also the most invasive option:

- residual state must be partitioned consistently;
- LayerNorm and gates require distributed reductions or replication;
- stream adapters and node-program basis need row/column parallel forms;
- tied embeddings and vocabulary loss require a distributed head;
- collectives occur inside every relay.

Do not begin width tensor parallelism for the 100M model. Revisit it for the
planned width-4,096 scale only after measuring whether stream parallelism and
state sharding are sufficient.

### Axes not recommended

**Node sharding:** every stream maintains occupancy across all nodes, local
edges can cross partitions, and the global bus is explicitly nonlocal. It
would add communication to write, read, route, and transport.

**Context sharding:** the sequence mixer contains forward/reverse recurrent
dependencies and block-cache state. Context partitioning would introduce
boundary communication in every relay and complicate exact cached inference.

## Proposed public configuration

Runner flags:

```text
--device {auto,cpu,cuda,cuda:N,xpu,xpu:N}
--distributed-strategy {none,ddp,fsdp,stream_parallel}
--distributed-backend {auto,gloo,nccl,xccl}
--route-transport-backend {auto,scatter,reverse_gather}
--scan-backend {auto,torch,compiled,triton}
--profile-device-steps N
--profile-output PATH
```

Resolved configuration and checkpoint metadata must store every selected
backend. `auto` must be resolved to the concrete implementation before the
configuration is printed and saved.

Library APIs should remain device-derived: allocations use an existing
tensor's device or an explicit caller device, never a global CUDA/XPU
default.

## Test matrix

### Portable unit tests

Run on CPU for every change:

- route transport forward/backward equivalence;
- decayed scan forward/backward equivalence;
- block-cache state and append parity;
- checkpoint legacy schema loading;
- accelerator-selection parsing;
- distributed configuration validation.

### Single-B70 tests

- FP32 eager 130k;
- BF16 eager 130k;
- BF16 compiled 130k;
- BF16 compiled 100M short-context;
- BF16 compiled 100M context 2,048;
- cached prefill/refine/append;
- checkpoint/resume;
- profiler trace with no unexpected CPU fallback.

### Dual-B70 tests

- XCCL send/all-reduce bandwidth and latency;
- peer-access capability and PCIe topology report;
- DDP update equivalence;
- DDP checkpoint/resume;
- global metric reductions;
- rank-zero-only I/O;
- DDP throughput scaling;
- selected capacity strategy only after its own correctness tests.

### CUDA regression

The Intel port is not permitted to remove:

- CUDA BF16;
- CUDA `torch.compile`;
- CUDA checkpoint resume;
- existing block cache;
- current training and evaluation logs.

Portable optimizations should be benchmarked on CUDA as well. If an
Intel-winning implementation regresses CUDA, dispatch by measured backend
rather than forcing one implementation globally.

## Performance targets

Absolute throughput targets should be set only after the first B70 baseline.
The initial relative goals are:

| Milestone | Target |
|---|---:|
| BF16 compiled versus BF16 eager | at least 1.15x |
| Optimized route + scan versus first compiled baseline | at least 1.25x |
| Dual B70 DDP versus one B70 | at least 1.60x |
| Cached append versus full reprefill | increasing advantage with context |

These are engineering goals, not reasons to hide a negative result. Report
the complete speed-memory frontier. A kernel may be retained as a low-memory
option even when it is not the fastest.

## Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| XPU operator is supported eagerly but poorly compiled | graph break or slowdown | strict full-graph test and profiler |
| Small `einsum` shapes miss XMX | low compute utilization | flatten, batch, pack, and align |
| Scan produces many temporary tensors | bandwidth and VRAM loss | hierarchical scan without weight matrix |
| Atomic routing contends | irregular kernel dominates | reverse-adjacency gather/reduce |
| FP32 promotion spreads too far | XMX underuse | limit FP32 to recurrence/reduction boundaries |
| BF16 recurrence loses long-context state | learning or cache drift | FP32 recurrent state and parity tests |
| ECC leaves only about 28 GB visible | unexpected OOM | measure real capacity and retain margin |
| Dual cards share limited CPU lanes | poor collective scaling | record topology and link widths |
| XCCL or peer path is immature on the selected stack | DDP stalls or low scaling | pin validated versions and run collective microbenchmarks |
| Evaluation/checkpointing runs on every rank | duplicate I/O or deadlock | rank-zero I/O with symmetric collective schedule |
| Two-GPU sharding is mistaken for automatic VRAM pooling | design fails at allocation | explicit DDP/FSDP/stream-parallel distinction |
| Intel-specific changes regress CUDA | lost existing capability | backend dispatch and CUDA regression gate |

## Ordered deliverables

1. Hardware/software commissioning report for one B70.
2. Backend-neutral accelerator helper and checkpoint RNG schema.
3. 130k XPU correctness suite.
4. 100M XPU eager and compiled benchmark.
5. XPU profiler report.
6. Reverse-adjacency route transport.
7. Dense projection packing/layout pass.
8. Decayed-scan backend, if justified by profile.
9. Single-B70 10k-step production gate.
10. XCCL collective benchmark on two B70s.
11. Dual-B70 DDP runner and checkpoint support.
12. Dual-B70 scaling report.
13. Memory breakdown for the first configuration that does not fit one B70.
14. FSDP versus stream-parallel capacity prototype.
15. Width tensor parallelism only if the width-4,096 design requires it.

Every numbered deliverable should be independently reviewable and should
leave a rollback point. Functional XPU support, single-GPU optimization, DDP,
and capacity sharding must not land as one inseparable change.

## Definition of done

The roadmap is complete when:

- one source tree runs GTR-L on CPU, CUDA, and Intel XPU;
- the 100M reference trains in compiled BF16 on one B70 with stable logs,
  evaluation, checkpointing, and block-cache behavior;
- the dominant B70 kernels have profiler evidence and selected backends;
- two B70s train through XCCL DDP with correct global-batch semantics and
  useful throughput scaling;
- a documented capacity-sharding path exists for configurations exceeding
  one card;
- every performance claim includes configuration, software versions,
  synchronization, warmup, peak memory, and reproducible benchmark evidence.
