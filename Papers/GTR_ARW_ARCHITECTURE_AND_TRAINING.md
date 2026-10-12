# GTR-ARW: Architecture and Training, Current Iteration (2026-08-17)

## Status

This is the consolidated reference for GTR-ARW as it is actually built and
trained today, superseding scattered notes across
`Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md`,
`Runbooks/GTR_RUN_REGISTRY.md`, and `Papers/GTR_ARW_SMOLLM3_26B_RECIPE.md`.
Those remain the detailed run-by-run and flag-by-flag references; this
document is the architecture-and-findings synthesis that ties them together.
`ARCHITECTURE.md`'s "current" sections describe the separate Convergent
variant (one shared node graph, persistent payload lanes); GTR-ARW is that
document's older "Arrow" branch, not its present focus. This paper is
GTR-ARW's equivalent up-to-date reference.

`gator_language_train.py --architecture gator` is the entry point for
everything described here. Two training recipes select entirely different
data/task machinery on top of the same architecture flags:
`--training-recipe smollm3-26b` (the SmolLM3-scaled streamed-mixture recipe,
used for every production run) and `--training-recipe standard` (a
from-scratch-tokenizer, formation-objective recipe used for the diagnostic
chat-only runs in Section 4).

## 1. Architecture

### 1.1 Cells, nodes, relays, neighbors

GTR-ARW is a recurrent, graph-routing architecture built from a population of
**cells** (`--cells`, current default 8), each containing a fixed number of
**nodes** (`--nodes`, default 4). Every sequence position (context token or
output slot) is a packet that lives inside this cell population and is
refined over a fixed number of **relays** (`--relays`) — internal
graph-processing rounds, not output-token generation steps. Relay depth and
sequence position are separate notions of time: every position updates in
parallel within one relay, and relay `r+1` consumes the graph location and
state produced by relay `r`. Each cell can hand a packet off to one of
`--neighbors` (default 4) adjacent cells per relay, plus itself, for 5
possible actions per handoff at the default neighbor count.

**Relay count is a routing/dynamics lever, not a capacity lever, for
GTR-ARW.** Confirmed directly (2026-08-16/17): toy-scale runs at `--relays 2`
and `--relays 4` produced byte-identical non-vocab parameter counts
(696,512-param core at `d_model=128`) and identical lexical-path sizes — more
relays changes how many times the existing weights are applied to route a
packet, it does not add new weight matrices. (This is GTR-ARW-specific,
 relay count *does* add parameters in GTR-L's different architecture — a real 
difference between the two Gator variants, not a bug.)

### 1.2 Cell transport: broadcast_mix vs. stewardship

`--cell-transport-mode {broadcast_mix,stewardship}` selects how packets move
between cells at each relay:

- **`broadcast_mix`** (current default, used for every run this session):
  every cell participates and neighbor-couples every relay, without
  ownership, handoff nucleus-pruning, or scout policy. This is "behaviorally
  naked" GTR-ARW — the architecture without the stewardship routing-economy
  machinery layered on top.
- **`stewardship`**: token ownership/state/node handoffs, with a nested
  ownership-nucleus and handoff-beam routing decision (`--stewardship-*`
  flags control keep-mass, handoff-keep-mass, min/max active cells,
  exploration, temperature schedule). Two execution backends exist —
  `dense` (always evaluates dense routes) and `packed_auto` (periodically
  latches a packed sparse suffix once live support drops below a threshold).
  Production and every toy/diagnostic run this session used
  `--stewardship-execution dense` even under `broadcast_mix` transport,
  since stewardship-specific routing doesn't engage under `broadcast_mix`
  regardless.

### 1.3 Node compute and precision

Each node performs a per-cell transform. `--dense-node-backend` selects the
implementation:

- **`folded_bmm`** (current default): an algebraically folded wide
  projection avoiding a large node-input tensor. Used for every production
  and toy run.
- **`einsum`**: the expanded reference implementation.
- **`monarch`**: a 2-factor Monarch-matrix structured factorization,
  ~18x fewer node-weight parameters at the default block factor. Explored in
  a separate bake-off (`gtr_arw_100m_monarch_bf16_test`,
  `Runbooks/GTR_RUN_REGISTRY.md`) that found it essentially tied on loss and
  modestly faster in throughput at matched width — not adopted for the
  current production run, which uses plain `folded_bmm`. Requires
  `--node-compute-precision bf16` strictly; incompatible with `fp8_te`.

Precision is controlled independently for node compute and vocabulary
readout: `--node-compute-precision {bf16,mxfp8,nvfp4,fp8_te}` and
`--readout-precision {bf16,mxfp8,mxfp4,fp8_te}`. Current production and every
toy run this session use `fp8_te` (real hardware FP8 via NVIDIA Transformer
Engine with CUDA-graph capture) for both — a real, if modest, throughput and
memory win at production shape, proven stable across many hours of this
session's runs with zero non-finite windows once two earlier bugs (a
graphed-instance/relay>1 gradient corruption, and a `make_graphed_callables`/
`torch.autocast` incompatibility) were fixed. `--mixed-precision bf16` wraps
the overall forward pass in `torch.autocast`; loss/backward/optimizer state
stay FP32.

Two CUDA graphs are captured for dispatch-overhead reduction, orthogonal to
the fp8_te precision win: `--broadcast-mix-cuda-graph` (cell-coupling) and
`--stewardship-traffic-cuda-graph` (context traffic bookkeeping). Both are
routine in every production/toy launch this session.

**These optimizations (fp8_te + both CUDA graphs + `--compile`) are only
validated within the `smollm3-26b` recipe's code path.** The `standard` recipe 
currently runs stably only with plain `--mixed-precision bf16`, no `--compile`, 
`--execution-preset optimized_gpu_safe`.

### 1.4 Lexical/embedding path

Token embedding is a tied, factorized path rather than a direct
`[vocab, d_model]` table:

```text
token id -> tied [vocab, lexical_width] table -> lexical projection -> Gator
```

reversed on output (the same learned projection, then the same token table
for logits — no separate output head). Parameter cost is approximately
`vocab * lexical_width + lexical_width * language_width + vocab` (tied
token table, tied factor projection, output bias). `--lexical-embedding-dim`
defaults to `--embedding-dim`; `0` disables factorization entirely.
Confirmed exact on the current production run: `vocab=50,000,
lexical_width=256, language_width=3,008` gives `50,000*256 +
256*3,008 + 50,000 = 13,620,048`, matching the run's own reported
`lexical_path` exactly.

Two lexical-input mechanisms exist, controlled by `--lexical-mode
{private_hash,communal}`: `private_hash` (current default, used by the
`smollm3-26b` recipe's HF-tokenizer path) gives cells tiny private
modulo-hash prototypes; `communal` gives one shared full-vocabulary table.
The `standard` recipe's from-scratch `CompactTokenizer` path (Section 4)
also defaults to `private_hash`.

**`--lexical-embedding-dim` is not a meaningful lever at this vocabulary
scale for GTR-ARW.** A direct A/B (256 vs. 500, the latter following a
documented `embedding_dim ≈ 0.01 × vocab_size` rule) found no measurable
advantage at 50,000-vocab toy scale (0.0085-nat loss gap at matched tokens
seen, noise-level). That rule was validated under `--architecture
convergent`, a structurally different variant — the null result here is
consistent with the architecture mismatch, not evidence the rule is wrong
in general. Current production config keeps the standing `256`.

### 1.5 Receiver memory

`--receiver-memory-backend {attention,convolution,linear_scan}` selects how
each cell/node reads its own accumulated receiver-position history:

- **`attention`**: quadratic reference implementation.
- **`convolution`** (recipe-doc default): exact normalized convolution,
  still `O(tokens²)` FLOPs despite avoiding one large tensor shape.
- **`linear_scan`** (used by every run this session): genuinely `O(tokens)`,
  reinterpreting `relative_node_bias` as decay-rate/mix parameters via a
  bidirectional `gator_linear.core` decayed scan instead of a
  relative-position code. `--receiver-memory-hgrn-scan` swaps the scan
  kernel for `flash-linear-attention`'s fused `chunk_hgrn` Triton kernel,
  measured ~5.5x faster in isolation at production shape, verified correct
  to ~1e-6 forward/gradient difference against the existing chunked scan.
  Required for `--block-causal` mode; not yet compatible with
  `--receiver-memory-cuda-graph`.

### 1.6 Block-causal formation and parallel response prediction

`--block-causal --block-size N --position-norm-length M` trains GTR-ARW with
causal committed blocks and bidirectional refinement inside each block: a
later block cannot alter an earlier one, but positions inside the active
block remain fully parallel. `--response-tokens` (aliased "slots" throughout
this session's notes) sets how many output positions are formed
*simultaneously* in one non-autoregressive forward pass — this is not
next-token autoregressive decoding, it is closer to parallel multi-token
infilling, and is a genuinely harder task per output position than standard
next-token prediction. `--position-norm-length` must cover the largest
context ever planned for a checkpoint (position codes learned at a shorter
length would need rescaling to grow later), which is why production launches
this length at `4096` even while training at `--smollm3-sequence-length
512` — extending context later reuses the existing
`--smollm3-allow-context-growth` + `--resume-checkpoint` path rather than
requiring literal RoPE/YaRN.

**Wider `--response-tokens` is measurably harder, not free.** Confirmed
twice independently at toy scale: on the broad SmolLM3 mixture
(`response_tokens=8` vs. `4`, matched by tokens-seen) and again on the
chat-only+LMSYS diagnostic corpus (`response_tokens=8` step-500 read:
`acc=0.457 exact=0.000` vs. `response_tokens=4`'s `acc=0.508 exact=0.280` at
the same step). Current production run uses `--response-tokens 8`, chosen
deliberately for the harder/richer task at real capacity rather than
avoiding it (see Section 5).

## 2. Training recipe (`smollm3-26b`)

Full flag-by-flag reference: `Papers/GTR_ARW_SMOLLM3_26B_RECIPE.md`. Summary
of what matters for understanding the current production run:

- **Data**: SmolLM3's 3-stage streamed mixture (FineWeb-Edu, DCLM, the Stack
  v2 language folders, FineWeb-2 non-English sources, math/reasoning sets,
  Cosmopedia2, etc.), scaled to a 26B-token recipe-equivalent budget
  regardless of `--smollm3-sequence-length`. Source selection is
  deterministic largest-mixture-deficit (Nanotron/Megatron-style), not
  random sampling; stage transitions and cursors persist exactly across
  resume.
- **Tokenizer**: a from-scratch English-only 50,000-vocab BPE
  (`Runbooks/tokenizers/gtr_arw_english_50k/`), reusing Llama 3.2's exact
  pre-tokenizer/byte-fallback rules, trained on a 9.66GB weighted sample of
  the mixture with the 14 non-English `fw2-*` sources excluded. A/B against
  the original 128,256-vocab Llama tokenizer showed a consistent +5.5% to
  +9.9% throughput win (smaller readout matmul) with no VRAM cost and only a
  minor, symmetric dip in chance-floor-normalized loss descent — adopted for
  every run since.
- **Effective batch**: fixed at 32 formations by the original recipe
  (`--smollm3-effective-batch-sequences`, generalized this session from a
  previously-hardcoded constant); physical batch × accumulation must equal
  it. Production uses `--batch-size 8 --gradient-accumulation-steps 4`.
- **Optimizer**: AdamW `(β1=0.9, β2=0.95, ε=1e-8)`, weight decay `0.1`
  (excluding token embedding), gradient clip norm `1.0`, linear warmup then
  scheduled decay. Production peak LR `8e-5` (`--smollm3-peak-lr`), well
  below the recipe's `2e-4` default — a deliberate choice carried from the
  original bake-off planning, not yet re-litigated against the final
  100M/context=512/relays=2 shape.
- **Checkpointing**: bounded, not unbounded — `latest.pt` every interval,
  `best.pt` only when that interval's average loss is the lowest seen so
  far (so a later destabilization can be rolled back to). Resume restores
  model weights, AdamW moments, RNG state, stage/source cursors, and
  physical-batch-geometry remapping, exactly.
- **Numerical failure handling**: fails closed. An isolated non-finite loss
  or gradient discards the whole accumulation window without advancing the
  update/schedule/token counter; `--smollm3-max-consecutive-nonfinite-
  windows` (default 3) bounds how many consecutive recoveries are tolerated
  before a hard failure.

## 3. Diagnostic instrumentation added this session

Now permanent in `gator/smollm3_train.py`, used throughout every run
described in this paper:

- **PPL** in both console and file logs (`ppl=exp(min(loss, 20.0))`).
- **Fixed 10-prompt eval** (`--smollm3-fixed-eval-every-updates`):
  greedy-decodes a fixed prompt set through the raw uncompiled model
  (deliberately bypassing the compiled training graph to avoid a
  batch-size-mismatch recompile), logged to `checkpoint_dir/
  eval_prompts.log`. The qualitative signal this produces — whether
  completions are flat single-token repeats, cycling between a few modal
  tokens, or genuinely differentiated per output slot — proved more
  diagnostic than the scalar loss for spotting degenerate collapse early.
- **Per-slot accuracy** (`slot_acc=[...]`): catches per-position fanout an
  aggregate loss would mask.
- **Diff-based steady-state throughput methodology**: `elapsed = updates /
  reported_cumulative_ups`, diffed between two checkpoints to cancel
  one-time compile/startup overhead. Necessary because the cumulative
  `updates_per_second` figure printed at any single point is contaminated
  by however much compile/autotune overhead has been diluted so far — a
  window taken too early (still inside residual autotune activity)
  systematically understates true throughput; see Section 5.3 for a
  concrete case where an early window was off by nearly 2x.

## 4. Empirical findings: the toy-scale sanity-check investigation (2026-08-14 to 08-17)

Before committing days-to-weeks of production compute, the pipeline was
sanity-checked at a drastically shrunk width (`d_model=128`, ~696,512
non-vocab core params — roughly 1/125th of the production core), keeping
tokenizer, data mixture, and precision identical to production. The
question: does the real pipeline reliably escape early single-token
collapse, and which levers matter?

### 4.1 Context length and response-token width

Context=2048 (matching the original plan) plateaued in the 7.6-7.9 loss
band for ~60,000 updates before a real break (loss 7.43, periods entering
the token mix, 2-3-token-type completions). A context=256 follow-up found
context length does not by itself explain the plateau — tok/s came in
*lower* at shorter context (data-pipeline-bound at this tiny width, not
compute-bound, likely a toy-width-only artifact) and the same slow,
ambiguous drift recurred. Narrowing `response_tokens` from 8 to 4 produced
a calmer gradient norm at matched tokens-seen (3.4 vs. 10.1) without a
decisive loss difference — consistent with Section 1.6's later confirmation
that wider response windows are a real, if modest, difficulty lever.

### 4.2 Relay count: small effect, real throughput cost

`--relays 4` (up from the default 2) held a small, consistent, non-collapsing
loss edge over `--relays 2` in both the broad-mixture toy comparison and a
later chat-only comparison — never decisive, never reversing. Throughput
cost was real and consistent: ~40% slower in the broad-mixture setting,
~1.8x slower in the chat-only setting. **Decision: `--relays 2` for
production** — the loss edge was not judged worth the compute.

### 4.3 The central finding: task/data breadth, not raw capacity, was the bottleneck

The 696,512-param toy core never produced coherent output on the full broad
SmolLM3 mixture across 164,530 updates (multiple configurations) — eval
completions cycled between single-token collapse modes ("the", commas,
periods) and never resolved into real words. The same core, pointed at a
narrow chat-only corpus instead (`--training-recipe standard`,
`smoltalk_everyday.jsonl`, 2,260 conversations), produced exact-match
completions by step 500 and kept improving cleanly for thousands of steps —
a qualitatively different regime from anything the broad-mixture runs ever
showed.

**This result was initially confounded by corpus repetition, not genuine
task-breadth learning.** The tiny 2,260-conversation corpus is the exact
corpus flagged in `Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md` §7.1 as one
where repetition drives a documented collapse (a plain transformer reaches
15x its own peak `content_ppl` by ~170 epochs on this corpus). Every
chat-only variant run this session (response_tokens=4 and 8, relays=2 and
4) reproduced this: fast early improvement, then a loss/accuracy
dissociation starting around step 7,000-9,000 (loss climbing while
discrete accuracy metrics stayed flat or improved) — most severe in the
relays=2/response_tokens=4 run, which reached `content_ppl=245.69` at
~91 epochs, nearly matching that study's own collapsed-transformer number
exactly.

**Fix and confirmation**: mixing in LMSYS-Chat-1M (~1M conversations,
`--lmsys-data --lmsys-load-mode stream`, default 90%-streamed/10%-primary
mixture) eliminated the collapse entirely — loss and accuracy both improved
monotonically-with-noise through and past the point where every
pure-smoltalk-only run had already reversed. This is the paper's central
empirical conclusion: **given enough capacity relative to the task, and a
data mixture broad/fresh enough to avoid repetition-driven collapse,
GTR-ARW learns real language** — the earlier broad-mixture toy failure was
a capacity-vs-task-breadth mismatch, not evidence the architecture itself
can't acquire language at this style of task.

### 4.4 Summary table

| Lever | Toy-scale finding | Production decision |
|---|---|---|
| Context length | Not the bottleneck by itself | 512 (staged growth planned) |
| `response_tokens` | Real, modest difficulty increase | 8 (accepted, not avoided) |
| `lexical_embedding_dim` | No measurable effect at 50k vocab | 256 (unchanged) |
| Relay count | Small loss edge for 4, real compute cost | 2 (default) |
| Task/data breadth | The dominant factor | Full SmolLM3 26B-token mixture |
| Corpus repetition | Real, documented collapse risk on tiny/narrow corpora | N/A — production mixture is large/diverse by construction |

## 5. Current production configuration (launched 2026-08-17)

`gtr_arw_100m_production_d1504_slots8_relays2`,
`Runbooks/colab_cells/_launch_production_d1504_slots8_relays2.sh`.

### 5.1 Width sizing

The bake-off era's `d_model=1312` config predates the 50k-vocab tokenizer
swap and only reaches 79,678,064 total params at that vocabulary size —
never corrected. Re-sized via bisection
(`Runbooks/colab_cells/_100m_width_sweep_dense_50k_relays2.py`, same
approach as an earlier Monarch-backend width sweep, adapted for dense
`folded_bmm` + 50k vocab + `relays=2`): **`d_model=1504, embedding_dim=3008`
→ 100,430,768 total params** (86,810,720 core + 13,620,048 lexical),
confirmed exactly by the live run's own startup log.

### 5.2 Full configuration

```bash
--training-recipe smollm3-26b --architecture gator \
--hf-tokenizer Runbooks/tokenizers/gtr_arw_english_50k \
--cells 8 --nodes 4 --d-model 1504 --embedding-dim 3008 --lexical-embedding-dim 256 \
--relays 2 --neighbors 4 --node-topology asymmetric \
--cell-input-mode shared_projection --cell-output-mode shared_pool \
--block-causal --block-size 16 --position-norm-length 4096 \
--cell-transport-mode broadcast_mix --stewardship-execution dense \
--receiver-memory-backend linear_scan --receiver-memory-hgrn-scan \
--response-tokens 8 \
--smollm3-sequence-length 512 \
--batch-size 8 --gradient-accumulation-steps 4 \
--mixed-precision bf16 --node-compute-precision fp8_te --readout-precision fp8_te \
--broadcast-mix-cuda-graph --stewardship-traffic-cuda-graph \
--smollm3-peak-lr 8e-5 \
--execution-preset optimized_gpu --compile --compile-mode max-autotune-no-cudagraphs \
--dense-kernel-padding 8 --dense-node-backend folded_bmm \
--steps 1200000 --checkpoint-every-updates 500
```

Context=512 (not the eventual larger target) is a deliberate staged choice:
cheaper to train short and extend later via
`--smollm3-allow-context-growth` than to pay a longer-context cost the
whole way through — `--position-norm-length 4096` is set now specifically
so position codes don't need rescaling when that growth happens.

VRAM is genuinely maxed at this width and batch geometry: ~15.7/16.3GB
(~96.6%) at steady state on the RTX 5080, essentially no headroom to
increase batch size without risking OOM.

### 5.3 Throughput — a methodology caution

An early diff-based read (update 100→390) gave ~0.51-0.52 updates/sec.
This was **wrong** — the window was still contaminated by residual
`torch.compile` autotune activity that continued bleeding into it. A wider
window taken once training had genuinely settled (update 1000→2260,
several overlapping sub-windows agreeing within ~0.92-0.97 u/s) gave the
real number: **~0.93-0.95 updates/sec, ~15,200-15,600 tokens/sec** — nearly
double the first estimate. The general lesson: a diff window immediately
after launch can still be inside compile overhead even after the first
several hundred updates at this model scale; confirm convergence across
multiple overlapping windows before trusting a throughput number, not just
one.

At this corrected rate, the current `--steps 1200000` phase (19.66B tokens,
~76% of the full 26B-token recipe) is on the order of **~15 days**; the
full 26B-token budget (`recipe_updates=1,586,915` at this config's 16,384
tokens/update) is **~19-20 days**.

### 5.4 Colab migration trigger point (decided 2026-08-17)

The recipe's 3-stage data mixture (`gator/smollm3_recipe.py`) states its
stage-transition update numbers (`STAGE2_UPDATE=144,992`,
`STAGE3_UPDATE=176,428`, decay-to-zero at `TOTAL_UPDATES=198,365`) at a
reference `sequence_length=4096`; `_scaled_update_count` scales them
inversely with `sequence_length × effective_batch_sequences`. At this run's
`sequence_length=512` (an exact 8x factor vs. the reference), that lands
at:

| Boundary | Reference (seq=4096) | This run (seq=512) | % of full recipe |
|---|---|---|---|
| Stage 1→2 | 144,992 | **~1,159,936** | ~73.1% |
| Stage 2→3 | 176,428 | **~1,411,424** | ~89.0% |
| Decay-to-zero | 198,365 | **~1,586,920** | 100% |

Stage 2 is ~15.9% of the full recipe (~251,488 updates), stage 3 ~11.1%
(~175,496 updates) — ~27% combined. The current Phase 1 `--steps 1200000`
budget lands ~40,064 updates *into* stage 2 before stopping on its own.

**Decision: move to Colab at the stage 1→2 transition** (update
~1,159,936), not at Phase 1's own step-budget endpoint — "clean boundary,"
moving exactly as the source mixture itself shifts rather than mid-stage.
Target hardware: an RTX Pro 6000, cited by the user as ~4x faster than the
local RTX 5080 for this workload — not yet verified against this repo's
own measured throughput; worth a real diff-based check once the migration
happens (see Section 5.3's methodology caution — don't trust an early
window on new hardware either). Whether context extension (Section 7) is
tied to this same boundary or decided independently is still open.

### 5.5 Status at time of writing (updated 2026-08-18)

Update ~75,830/1,200,000 (6.3%), loss cycling in a ~7.5-7.8 band since
~update 25,000 (down from ~10.8 at initialization), `nonfinite_windows=0`
throughout including one pause/resume cycle (GPU lent out for other work,
auto-resumed cleanly via a watcher script). Throughput settled at the
corrected ~0.95-0.96 u/s steady-state. Eval completions are still cycling
between single/few-token collapse modes (comma-heavy, period-heavy,
newline-heavy, occasional real connective words like "and" that haven't
yet persisted across checkpoints) rather than settling into one mode or
producing real content — the same shape every toy run passed through
before differentiating further, not yet informative on its own this
early. `slot_acc` shows real per-checkpoint variation (sometimes broad
across 5-6 of 8 slots, sometimes narrow) rather than any slot staying
pinned at a fixed value, which reads as continued active learning rather
than stagnation. Full per-checkpoint detail:
`Runbooks/GTR_ARW_100M_PRODUCTION_JOURNAL.md`.

## 6. CPU inference: portable edge-hardware validation (2026-08-18)

Separate from the training-side work above, real inference-side
optimization work landed on the `gator_sparse` branch (commit
`77374cc`, "Optimize GTR-ARW CPU inference prefill"), benchmarked
directly against this production run's own checkpoint
(`gtr_arw_100m_production_d1504_slots8_relays2/latest.pt` at update
71,000 — 100,430,768 params, 1,163,264,000 tokens consumed, matching
this paper's own figures exactly) on genuinely edge-class hardware: an
Intel Core i7-1185G7, FP32 eager PyTorch, batch size 1, four CPU
threads, no compilation, no quantization — the "worst-case local
inference" path, not the FP8/GPU training runtime. This is the first
real evidence bearing directly on the project's stated thesis (Section
"the whole thesis" of this session's conversation): whether GTR-ARW can
hold a conversation on constrained hardware, not just train efficiently.

**Three no-gradient CPU scheduling changes, model math unchanged:**

1. **Direct (non-vectorized) `decayed_scan` evaluation** for eager CPU
   inference specifically (training/autograd/GPU/compiled paths keep the
   existing vectorized kernel unchanged). At the production scan shape
   `[1505, 4096, 32]`: **2.405s → 0.573s median latency, 4.20x faster.**
2. **Block-causal receiver prefill in aligned 128-token slabs**, each
   reduced immediately instead of retaining the full forward/backward/
   mixed-scan tensors for the whole sequence at once. At the full
   production receiver shape, this cuts the largest live FP32 temporary
   from ~790MiB to ~25MiB and gives **19.180s → 6.125s latency, 3.13x
   faster.** Slab boundaries are aligned to the LCM of the scan chunk
   size and the block-causal block size, so every slab sees identical
   scan/block boundaries to the materialized version — a scheduling and
   tensor-lifetime change, not a numerical approximation (see below).
3. **Node-sliced folded-BMM evaluation** when the eager CPU folded-node
   result would exceed 64MiB — evaluates and reduces one node slice at a
   time instead of materializing the full `[batch, cells, tokens, nodes,
   width]` activated tensor at once. GPU and compiled execution keep the
   wide folded BMM unchanged.

**No checkpoint conversion required** — strict loading of the existing
`latest.pt` succeeds with unchanged state keys, shapes, cache schema, and
block semantics.

**Measured results** (same checkpoint, same benchmark inputs, before vs.
after):

| context | baseline prefill | optimized prefill | speedup |
|---:|---:|---:|---:|
| 128 | 182.9 tok/s | 220.9 tok/s | 1.21x |
| 512 | 163.1 tok/s | 245.9 tok/s | 1.51x |
| 1,024 | 154.6 tok/s | 243.1 tok/s | 1.57x |
| 2,048 | 98.2 tok/s | 241.0 tok/s | 2.45x |
| 4,096 | 53.4 tok/s | 236.7 tok/s | **4.43x** |

At 4,096 tokens, prefill latency fell from 76.642s to 17.303s. Per-token
latency now stays flat near 4.2ms from 512 through 4,096 tokens **instead
of degrading sharply once the old full-sequence receiver temporaries
exceeded available host memory** — this is the actual edge-relevant
result: the old path didn't just get linearly slower with context, it
hit a real memory cliff that the slab-based rewrite removes entirely.

Cached generation (measured from a 4,048-token committed prefix through
three 16-token blocks): proposal throughput 61.19 → 126.46 tok/s;
end-to-end proposal-plus-cache-commit throughput 31.32 → 70.08 tok/s
(31.93ms → 14.27ms effective per-committed-token latency).

**Correctness verified, not just assumed**: the optimized path was
checked against the legacy execution at 512 tokens — finite logits
differ by at most `4.34e-5` (mean `7.47e-6`), top-1 and top-5 selections
agree for every output slot. The focused receiver/cache/gradient/
dense-kernel/language test suites pass 100 tests and 22 parameterized
subtests. Inference-only — no claimed training-speed improvement.

**Why this matters for the project thesis**: this is real evidence that
GTR-ARW's linear-time (`linear_scan`) receiver-memory design pays off
concretely on the exact hardware class the "crappy edge device" framing
is about, not just asymptotically — a 4.43x prefill speedup and a
genuinely flat per-token-latency curve (no memory cliff) at the model's
current production scale, on a real laptop-class CPU, using the actual
in-progress production checkpoint rather than a synthetic benchmark
model. Not yet merged into this run's own branch (`gator_monarch`) —
this document incorporates the finding by reference to `gator_sparse`
commit `77374cc` rather than merging divergent branch histories, per
this session's established one-branch-per-approach convention.

### 6.1 `GatorInferenceEngine`: a reusable module for prompt/chat apps

Built on top of the above (2026-08-18, `inference-engine` branch,
based on `gator_sparse`, commit `6a79e95`): `inference_engine.py`
wraps the `prefill_cache`/`forward_cached`/`append_cache` cache API
behind a single `generate()`/`chat_generate()` call, so building a
prompt window or chat app on top of a GTR-ARW checkpoint doesn't
require managing block-size alignment or cache lifetime directly.
Checkpoint/tokenizer loading mirrors `benchmark_gtr_arw_cpu.py`'s
approach; prompt left-padding to a block-size multiple mirrors the
smollm3 fixed-prompt eval harness's own convention (`eos_token_id`
padding, so real content ends exactly at a block boundary). Supports
greedy or sampled decoding with EOS stopping.

**Smoke-tested end to end against this run's own live checkpoint**
(update=76,500 at test time) — loads, tokenizes, runs the full
prefill/generate/decode loop, and produces output correctly. Output at
this checkpoint is still degenerate (`' of of of of...'`,
`'::::::::...'`), matching exactly what the training run's own
fixed-prompt eval shows at the same update — confirms the engine is
functioning correctly, not exposing a new bug; it will simply keep
reflecting whatever this checkpoint currently produces as training
continues.

## 7. Open questions

- Whether the ~0.95-0.96 u/s steady-state throughput holds over a
  multi-day horizon, or degrades (memory fragmentation, thermal throttling,
  data-pipeline stalls at scale) — held stable through one pause/resume
  cycle so far, not yet tested over the full run length.
- Whether/when to merge the `gator_sparse` branch's CPU-inference
  optimization (Section 6) into `gator_monarch`, and whether the same
  slab-based scheduling approach has anything to offer the GPU training
  path (the commit frames it as inference-only, CPU-specific, but the
  underlying "reduce a slab immediately instead of retaining the whole
  materialized tensor" idea is not obviously CPU-only on its face).
- Whether `--smollm3-peak-lr 8e-5` (carried from earlier bake-off planning)
  is still well-tuned at `d_model=1504`/`relays=2`/`response_tokens=8` — not
  re-swept against this final shape.
- The Monarch node-compute factorization was shelved (essentially tied on
  loss, only modestly faster) but never re-evaluated against the final
  production shape/precision combination.
- Context growth from 512 to a longer target has an implemented path
  (`--smollm3-allow-context-growth`) but has not yet been exercised on a
  real long-running checkpoint.
- The corpus-repetition-collapse finding (Section 4.3) was characterized at
  toy scale on a small chat corpus; whether an analogous collapse mode
  could appear on any narrow sub-slice of the much larger production
  mixture is untested.
