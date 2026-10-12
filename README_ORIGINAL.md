# Graph Trace Relay (GTR / Gator)

Experimental code for Graph Trace Relay, a recurrent graph-routing architecture that refines and emits complete output sequences simultaneously rather than autoregressively.

The canonical technical definition is [Gator Architecture Specification](ARCHITECTURE.md).
The ordered scaling work is pinned in
[Gator Performance TODO](PERFORMANCE_TODO.md). The exact target-GPU
acceptance procedure is the
[Blackwell Opaque-FFT + Triton Release Gate](BLACKWELL_FFT_TRITON_GATE.md).

## Active architecture track

The primary scaling track is now **Convergent Gator**: one shared learned node
graph carrying several persistent complementary payload lanes. Each lane keeps
its own semantic state and probabilistic node location across recurrent
relays; local node decisions create the path, co-located traffic interacts
through graph memory, and a slot-conditioned terminal jury combines continuous
lane representations. The current language controls use N66/L10/K3/R2,
50%-rewired small-world connections, and causal prefix denoising toward
unsupported simultaneous formation. See the
[Gator Architecture Specification](ARCHITECTURE.md) for the current equations,
exact parameter counts, and evidence boundary.

The experimental routing path now supports a shared low-rank packet trunk,
private node-edge keys and biases, packet-conditioned global ingress, and
packet-local probability-mass pruning with grouped active-node execution. An
opt-in packed backend now carries fixed-cap node-ID/mass frontiers across
relays, computes exact relative receiver memory by direct convolution or FFT,
adapts live route width to packet uncertainty, and can replace full-private
node matrices with one shared trunk plus private low-rank adapters. Optional
hierarchical ingress reduces global node-key scoring while leaving every node
reachable.
Legacy full-private routing and fixed ingress remain the defaults for
checkpoint compatibility. The exact opt-in controls are documented in
[Gator Language Trainer Arguments](GATOR_LANGUAGE_TRAINER_ARGUMENTS.md).

[Gator-Path / GTR-P](Papers/GATOR_PATH_DESIGN_SPEC.md), cellular Arrow, and
Gator-S stewardship remain preserved experimental predecessors. Their path,
complementary-trace, and sparse-dispatch results informed Convergent Gator, but
they are no longer the primary scaling architecture.

The repository currently contains:

- The original 980-parameter next-token GTR toy.
- Fixed-slot simultaneous sequence formation.
- Address-anchored, controlled-writing, and receiver-relative GTR variants.
- Parallel-prefix formation benchmarks.
- Matched one-shot Transformer controls and scaling experiments.
- Sparse-node and stitched-cell Gator configurations.
- Gator-S elastic token stewardship with dense and packed relay execution.
- Gator-Path packet topology, sparse mailboxes, bounded local routing, and an
  executable synchronous shared-recurrent core plus its single-cell
  parallel-prefix trainer.
- Stable-ID BLANK formation, monotonic latent alignment, edit-aware metrics,
  and an optional whole-sequence quality objective.
- A vectorized ten-cell implementation and Colab/T4 runner.
- A reusable optimizer, data, task, tokenizer, and model API.
- A reusable compute-policy API plus exact folded dense-node kernels for
  hardware-specific execution without changing learned parameters.
- A persistent bucketed stewardship backend that keeps live token routes
  packed across the sparse relay suffix.
- A deterministic language-batch pipeline with vectorized fixed-corpus
  gathers, bounded CPU prefetch, optional pinned CUDA staging,
  ordered parallel tokenization, and exact finite-corpus sampler resume.
- Simultaneous language-formation experiments over SmolTalk, LMSYS Chat,
  Tulu3 SFT, and SmolTalk Preferences.
- A staged response-symbol and monotonic-alignment program for simultaneous
  language formation.
- Internal findings, a living configuration ledger, and a training guide in `Papers/`.
- A canonical six-item performance TODO with a rollback ledger for the next
  GPU scaling pass.
- A canonical architecture specification with equations, tensor shapes, exact
  parameter accounting, invariants, and evidence boundaries.

## Current headline result

On fresh length-7 modular parallel-prefix examples:

| Model | Parameters | Training steps | Held-out accuracy |
|---|---:|---:|---:|
| Four-layer Transformer | 39,758 | 5,000 | 8.3% (chance) |
| Original Gator-ARW | 980 | 5,000 | 20.8% |
| Sparse 40-node Gator | 10,010 | 5,000 | 24.7% |
| Ten-cell degree-2 ring | 10,000 | 5,000 | 31.9% |
| Ten-cell degree-4 ring, seed 0 | 10,000 | 20,000 | **99.8%** |

The degree-four ring also reached 91.3% for seed 1 and 41.0% for the slower,
still-improving seed 2. A 1,006,154-parameter four-layer Transformer remained
at chance (8.3%) after 20,000 steps under the same prefix protocol.

These are exploratory, primarily single-seed results. See `Papers/GTR_internal_findings.md` and `Papers/GATOR_CONFIGURATION_LOG.md` for methods, corrections, caveats, and experiment history.

## Current language program

The frozen pre-routing-optimization D20 and D400 controls use the same
Convergent graph, 2,048-piece
tied vocabulary, 512 context slots, 128 response slots, and slot-conditioned
terminal voting. They differ only in payload width and parameter count:

| Control | Architecture | Parameters |
|---|---|---:|
| D20 | N66/L10/K3/R2/SW50 | 130,442 |
| D400 | N66/L10/K3/R2/SW50 | 14,360,682 |

Training uses causal prefix denoising. Correct shifted response prefixes are
progressively corrupted with token dropout, missing spans, zero-prefix rows,
and short-prefix rows. Further support removal is gated by both
teacher-content accuracy and the accuracy advantage of the correct prompt over
a shuffled prompt. Each checkpoint report therefore separates unsupported,
teacher, supported-context, and shuffled-context behavior.

TinyStories receives 40% of training traffic. SmolTalk Everyday, LMSYS Chat,
Tulu3 SFT, and SmolTalk Preferences receive 6%, 24%, 18%, and 12%.
Epoch-safe restart keeps these shares stable without inserting duplicate
traversals into one shuffle reservoir.

The alignment-aware output stack remains available for controlled A/Bs, but
the frozen prefix-denoising runs use rigid slot cross-entropy so curriculum and
architecture changes are not confounded.

## Packed Convergent performance gate

The authoritative stage matrix, pass criteria, and evidence-artifact contract
are in [BLACKWELL_FFT_TRITON_GATE.md](BLACKWELL_FFT_TRITON_GATE.md).

The known-working resumable 100M dense fallback is frozen separately at
`agent/100m-preoptimized` commit `454338d`. FFT/Triton work stays on
`agent/packed-routing-code-only`; it must not be merged into an active fallback
training run before the target-GPU release gate passes.

The archived 140M control (`V32768/E768/D768/N154/L10/K3/R2`, top-32) is the
first target-GPU gate for the packed implementation. Its pre-optimization
language-training baseline is **3.22 steps/s** at batch 6, context 512, and
128 response slots on the RTX PRO 6000 Blackwell Server Edition.

The first full Blackwell screens found that packed direct convolution reduced
VRAM from about 41 GB to 31.5 GB but was too slow, packed matrix OOMed because
its K-loop correctness oracle retained graph-sized autograd intermediates, and
compiled FFT failed because TorchInductor could not code-generate `complex64`.
The optimized branch now exposes three orthogonal execution controls:

- `fft_opaque` keeps the complete cuFFT receiver call behind a real-input,
  real-output custom operator. Its second custom operator now applies an
  explicit chunk-local FFT/writer adjoint, so complex spectra never enter the
  Inductor graph and backward no longer replays the complete receiver;
- `--packed-memory-reader-backend triton` fuses denominator normalization,
  packed node gather, K-route weighting, and reduction in both forward and
  backward without materializing `[B,T,L,K,C]` reader tensors;
- `--packed-memory-writer-backend triton` independently accumulates packed
  writes in FFT-native layout and applies an explicit adjoint without the
  matching `[B,T,L,K,C]` route-state tensor. It is currently CUDA float32-only
  with feature chunks no wider than 128.

The writer path has passed local contracts only and is awaiting a fresh
Blackwell gate; it was not present in the qualified measurements below.

The portable Torch reader and raw eager FFT remain numerical oracles. The
native explicit backward passed its complete Blackwell gate at `56c4a55`
(bundle `fca32603...e596b`) without widening the frozen causal tolerance. At
the frozen 140M full-model training shape, dense matrix Torch reached 4.003
steps/s with 29,489 MiB peak allocated; packed opaque FFT plus Torch reached
1.196 steps/s with 19,195 MiB; and packed opaque FFT with a Triton reader and
Torch writer reached 1.458 steps/s with 19,193 MiB. The Triton reader is
therefore 21.9% faster than the matched all-Torch packed path, while the
reader-qualified packed path saves 34.9% peak allocation versus dense. Dense
remains 2.75x faster, so this is a correctness and memory result, not a
training-speed promotion. The fused-writer result is still pending. Adaptive
K, shared node adapters, and hierarchical ingress remain later architectural
A/Bs. The isolated harness reports forward, forward/backward,
optimizer-inclusive time, and CUDA memory:

```bash
python benchmark_convergent_routing.py \
  --route-backend dense --packed-memory-backend matrix \
  --vocab-size 32768 --embedding-dim 768 --d-model 768 \
  --nodes 154 --lanes 10 --degree 3 --relays 2 \
  --router-rank 96 --active-node-top-k 32 \
  --batch-size 6 --context-tokens 512 --output-slots 128 \
  --device cuda --dtype float32 --compile \
  --warmup-steps 20 --measured-steps 100

python benchmark_convergent_routing.py \
  --route-backend packed --packed-memory-backend fft_opaque \
  --packed-memory-reader-backend triton \
  --packed-memory-writer-backend triton \
  --vocab-size 32768 --embedding-dim 768 --d-model 768 \
  --nodes 154 --lanes 10 --degree 3 --relays 2 \
  --router-rank 96 --active-node-top-k 32 \
  --batch-size 6 --context-tokens 512 --output-slots 128 \
  --device cuda --dtype float32 --compile \
  --warmup-steps 20 --measured-steps 100
```

Before the full model benchmark, validate the custom-op boundary and fused
kernel on the target GPU:

```bash
python validate_fused_fft_backend.py \
  --batch-size 1 --tokens 96 --nodes 154 --lanes 10 \
  --route-width 32 --width 32 --feature-chunk 32 \
  --warmup-steps 10 --measured-steps 50 --compile
```

Run the two commands as separate processes so compiled 140M graphs and
optimizer state from the first case cannot contaminate the second case's VRAM.
The frozen corpus/trainer control can run the established dense path with
low-write Drive mirroring:

```bash
python run_gator_140m_colab.py \
  --route-backend dense --packed-memory-backend matrix \
  --batch-size 6 --steps 1000 --eval-examples 8 --log-every 100
```

The runner records the backend in `RUN_SPEC.json`, prints every trainer line
immediately, batches Drive log writes, and keeps model parameters exactly at
140,193,699.

The node-heavy 100M comparison build uses
`V32768/E256/D256/N1337/L10/K3/R2`, learned ingress, top-32 routing, slot
voting, and dense matrix execution. It contains exactly 99,979,407 parameters;
88,311,524 are private node parameters. Its dedicated low-write runner defaults
to batch 4 because its larger node axis makes receiver memory more expensive:

```bash
python run_gator_100m_nodes_colab.py \
  --batch-size 4 --steps 100000 --eval-examples 8 --log-every 1000
```

Tiny CPU smokes are correctness and launch-overhead clues only. No packed GPU
speedup is claimed until this Blackwell comparison and a matched end-to-end
language-training run both beat the 3.22 steps/s control without numerical or
capability regression.

## Quick start

```bash
python gtr_toy.py --steps 1000
python parallel_prefix_benchmark.py --variant gtrarw --steps 10000
python gator_cells_vectorized.py --neighbors 4 --steps 10000
python gator_path_prefix_train.py --steps 10000 --cells 1
python gator_language_train.py --help
python benchmark_dense_kernels.py --device cpu
python benchmark_sparse_stewardship.py --device cpu
python benchmark_language_pipeline.py
python prepare_gator_colab_bundle.py
```

The first Gator-S language launch uses
`--cell-transport-mode stewardship --execution-preset optimized_gpu`.
It begins with every token routed broadly, retains low-mass scout paths while
familiar work narrows, and lets uncertain stewards fan a token back out through
their legal neighbors. See `gator/README.md` for the complete launch controls.

The packaged notebook starts with `optimized_gpu_safe`: strict FP32 without
compilation, exact convolution receiver memory, folded dense node transforms,
bounded pinned prefetch, and automatic persistent packed execution. This is
the preserved Arrow fallback. Long compiled runs have encountered multiple
non-finite backward signatures, including a genuine localized cell failure
after almost 3,000 steps under aggressive TF32. Compile/TF32 modes remain
diagnostic controls, but further work on that graph is deferred while
Gator-Path replaces its dense replicated execution structure. Explicit
low-level flags still override preset defaults; `manual` preserves all
individual defaults.

The first Blackwell `optimized_gpu_safe --compile` smoke exposed NaN gradients
when stewardship pruning activated at its warmup boundary. The pruning event
created empty receiver memories, revealing a compiled-backward hazard in the
convolution backend: it divided by `finfo.tiny` before masking the empty result.
Empty memories now use a unit safe denominator and a finite uncertainty
sentinel, which moved the failure boundary but did not establish long-run
compiled stability. See the configuration log for the preserved failure
record.

`prepare_gator_colab_bundle.py` rebuilds `gator_colab_bundle.zip`
deterministically from the current root scripts/notebook, complete `gator`
package, tests, and research documentation. Training data and checkpoints are
never included.

PyTorch is the only runtime dependency used by the model scripts.

The complete operator reference for every language-trainer switch is
[GATOR_LANGUAGE_TRAINER_ARGUMENTS.md](GATOR_LANGUAGE_TRAINER_ARGUMENTS.md).
It separates ownership breadth from handoff breadth, documents preset
overrides, and includes practical routing recipes.

## Dense execution optimization

The default dense node backend now folds the four width-8 node transforms in
each cell into one exact width-8 to width-32 projection:

```text
W(base + node_embedding) + bias
    = W(base) + (W(node_embedding) + bias)
```

This removes the materialized `[B,C,T,N,D]` node-input tensor without changing
the model equation or parameter count. At `B=32,C=62,T=104,N=4,D=8`, that
tensor alone is 25.2 MiB in FP32 per relay. An isolated local CPU benchmark at
`B=4` measured 9.9242 ms for the expanded node kernel and 2.6216 ms for the
folded kernel (3.79x). This is a forward-kernel microbenchmark, not an
end-to-end or GPU speed claim.

The router remains on `einsum` by default because grouped BMM was only 1.13x
faster in the same CPU screen. Parameter-free dimension padding also remains
off (`--dense-kernel-padding 1`): padding to eight measured 3.0146 ms versus
1.1743 ms without padding. Use `benchmark_dense_kernels.py` on the target GPU
before changing either default.

Hardware execution is configured separately through `gator.compute`, the
trainer's `--execution-preset`, and the low-level CLI flags
`--float32-matmul-precision`, `--cudnn-tf32`,
`--compile-mode`, and `--compile-shape-padding`. The designated Blackwell
benchmark policy is FP32 matmul precision `high` with cuDNN TF32 enabled, but
no Blackwell result is claimed yet. Compute policy is checkpointed as runtime
metadata rather than architecture state.

## Sparse stewardship optimization

Packed Gator-S now defaults to
`--packed-stewardship-backend persistent_bucketed`. At the selected
dense-to-sparse relay boundary, the model packs live
`(batch, token, cell, state, passport, mass)` branches once and keeps that
representation through packed-native uncertainty, pruning, handoff transport,
and final readout. Receiver groups use size buckets, and cell-local router and
node projections gather only active cell parameter slices. Handoff transport
instantiates only retained actions rather than every possible action for every
packed row.

`--packed-stewardship-backend repack` retains the original globally padded,
pack/unpack-per-relay implementation as a forward numerical oracle. With
pruning disabled its gradients also agree; straight-through pruning
intentionally drops surrogate gradients for physically removed routes. For a
packed layout with `M` live rows and receiver-group lengths `L_g`,
power-of-two bucketing gives total row capacity below `2M` and pairwise work
below `4 * sum_g(L_g^2)`. This is a worst-case allocation/work bound, not a
measured speedup.

Packed dispatch remains eager. Dense exploration calls before the controller
latches a sparse plan can remain compiled; after latching, a packed model call
is eager, including any dense prefix before its packed suffix. Dynamic
`nonzero`/`unique` grouping may still synchronize. Initial ownership is
uniform and dense, so sparse savings cannot begin until learned pruning creates
useful support. Node passports also remain soft: every active cell still
evaluates all four node transforms.

`benchmark_sparse_stewardship.py` compares forward-only execution of the two
eager sparse backends over forced support levels. Its tiny local CPU mode is a
correctness and harness smoke, not evidence of GPU or end-to-end training
acceleration. The full local
validation currently passes 145 tests plus scratch and resume trainer smokes;
target-GPU throughput and memory measurements remain required.

## Input-pipeline optimization

Language training no longer reconstructs fixed token rows through Python on
every step. `LanguageBatcher` collates a finite corpus once and samples it with
one tensor row gather. `PackedLanguageSource` keeps large in-memory corpora in
fixed int32 tensors and also gathers rows directly. `LanguageBatchPipeline`
then supplies either an inline serial path or a bounded single-producer queue,
with optional page-locked staging and a dedicated CUDA copy stream.

The trainer defaults to `--batch-pipeline prefetch --prefetch-batches 2`.
Startup tokenization remains serial by default; ordered bounded workers are
opt-in through `--tokenizer-workers`, `--tokenizer-worker-backend`,
`--tokenizer-chunk-size`, and `--tokenizer-pending-chunks`. Worker results are
consumed in source order, so parallel preparation does not reorder the corpus.

Finite and packed sources checkpoint both sampler state and every queued CPU
batch. `--resume-data-state auto` restores that sequence only when the corpus
fingerprint and provider state are exact. Generic JSONL streams cannot yet
restore an arbitrary iterator and shuffle buffer, so the trainer labels those
resumes as a seeded data reset. `--resume-data-state exact` fails instead of
making a false continuity claim. For a byte-identical corpus copied to a new
runtime, `--resume-data-state portable_exact` allows path/mtime identity to
change only when validation content matches and the provider proves the exact
encoded training corpus, queued batches, task geometry, and RNG state.

On the local CPU, a 500-iteration forward-free screen with 4,096 examples at
batch 32, context 64, response 32 measured 559.06 batches/s for legacy
per-example collation, 4,375.49 for the vectorized finite batcher (7.83x), and
2,981.37 for direct packed-source sampling (5.33x), with identical
deterministic checksums. Prefetch was slower than direct gathering when no
model compute existed to overlap; its value must be measured inside a GPU
training step. These are input-preparation microbenchmarks, not Gator
training-throughput results. The full local suite now passes 145 tests, and a
scratch-to-exact-resume trainer smoke completed successfully.

## Colab

Upload `Gator_Colab.ipynb` or `Gator_Language_Colab.ipynb` together with
`gator_colab_bundle.zip` to `MyDrive/GTR`, select a GPU runtime, and run the
notebook cells. The language notebook uses `optimized_gpu_safe` after the first
aggressive launch exposed a non-finite backward gradient. Advance one lever at
a time: safe, safe plus `--compile`, then `optimized_gpu`. The corresponding
aggressive dense control is `dense_gpu`; its prior baseline was about
12 steps/second.

For connected-language scaling experiments, upload
`gator_convergent_colab_bundle.zip` and open
`Gator_Convergent_Scaling_Colab.ipynb`. Its editable experiment table exposes
payload width, shared-node population, persistent lane count, graph degree,
layout, rewiring fraction, relay depth, lexical width, and terminal-voter
mode. It includes ready-made fixed-budget shape, compute-width, lane-density,
global-versus-prompt-voter, connectivity, and relay-depth sweeps. The preview cell
prints exact trainable counts before training so width/node/lane comparisons
remain near 130k parameters. Every default run uses the same seed, diverse
language-bundle corpus (`1:4:3:2` SmolTalk/LMSYS/Tulu3/Preferences),
epoch-safe source restart, optimizer horizon, and validation settings.

Use `language_probe.py CHECKPOINT [PROMPT ...]` to compare saved Gator,
Convergent, and one-shot Transformer checkpoints on identical prompts. The
default season, animal, arithmetic, rental-car, team-sports, and running
probes are intentionally fixed so qualitative comparisons do not depend on
random training-log samples. Output includes the checkpoint SHA-256 and exact
parameter count for provenance.

## Current scaling rule

The validated atomic unit remains the original width-8, four-node,
four-relay, 980-parameter GTR-ARW cell. Until controlled evidence favors a
different local design, larger Gator systems should scale primarily by
stitching more of these cells together rather than widening them. The first
language-scale `private_hash` configuration uses 62 literal 980-parameter cells
behind a shared 2,048-token language shell, for 129,304 total parameters. The
communal-vocabulary A/B keeps all 62 cells but removes the modulo-16 private
tables, yielding 121,864 parameters. The current language reference combines
communal vocabulary, per-cell feature gates, grouped readout, and Gator-S
stewardship for 124,618 parameters; `broadcast_mix` is its 124,616-parameter
capacity-matched transport control.

## Status

This is active research code. Architecture names, benchmarks, and APIs may change as the experiments develop.
