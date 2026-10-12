# Gator Performance TODO

**Status:** Four-way Blackwell gate passed; packed is a negative speed result and positive memory result, while Triton writer is slightly slower than Torch writer at N154
**Rollback baseline:** `dbbd569` (`agent/packed-routing-code-only`)
**Working 100M fallback:** `454338d` (`agent/100m-preoptimized`), preserved as a separate branch
**Pinned:** July 20, 2026

The executable acceptance boundary for the opaque-FFT plus Triton stack is
documented in
[BLACKWELL_FFT_TRITON_GATE.md](BLACKWELL_FFT_TRITON_GATE.md). No item below
constitutes a GPU speed or memory result until that gate's required Blackwell
stages and fresh-process performance A/B are preserved as evidence.

This is the canonical ordered TODO for the next Gator scaling pass. The
numbered items stay separate so each optimization can be measured, reviewed, and
committed as its own rollback point.

> July 17 decision: do not execute this list mechanically before Gator-Path.
> Receiver fusion, compiled dense-prefix staging, and several tensor-lifetime
> items optimize Arrow machinery that Path removes. Preserve this document as
> the Arrow performance backlog and reuse only the transferable work:
> bounded sparse workspaces, grouped dispatch, top-k, host-barrier removal,
> optimizer fusion, and allocation discipline.

> July 19 update: Convergent Gator is now the primary reproducible path. Its
> D30/N66/L10/R2 language run sustains roughly 60--65 compiled Blackwell
> training steps/s after increasing mixed-corpus prefetch from 10 to 20,
> versus roughly 17 steps/s for historical 130k Arrow. This exposes host input
> starvation as part of the old measured cost, but does not settle model
> scaling. Apply the audit below to Convergent before extrapolating to a
> general assistant.

### Blackwell grouped-MM dtype boundary

The first 140M shared-routing smoke test reached sparse private-node execution
but failed during `torch.compile`: Blackwell's `torch._grouped_mm` meta/kernel
path required BF16 while Gator supplied FP32 activations and FP32 master
weights. Sparse grouped node GEMMs now cast only their two matrix operands to
BF16 and cast the result back to the payload dtype. Optimizer parameters,
moments, routing, memories, normalization, and readout remain FP32. Widths
whose BF16 rows are not 16-byte aligned retain the portable fallback.

Validate forward finiteness, backward finiteness, peak VRAM, numerical drift,
and step throughput on the RTX PRO 6000 before broadening mixed precision.

The corrected smoke run then exposed a separate compiler inefficiency:
the Python routing-temperature float changed every update, causing eight
specialized recompilations before Dynamo reached its cache limit. Compiled
Convergent training now passes the scheduled temperature as a scalar device
tensor, keeping its value dynamic inside one graph.

At 140M, writing full `latest.pt` and improving `best.pt` checkpoints directly
to mounted Drive at every screen report creates far more remote I/O than the
text logger. `run_gator_140m_colab.py` therefore keeps exact high-frequency
checkpoints and the live log on local Colab storage. Screen output remains
immediate. The runner appends accumulated log text to Drive after either 20
step reports or 15 minutes, mirrors only `latest.pt` every 15 minutes, and
copies all best checkpoints once on clean exit.

### July 20 packed-frontier implementation checkpoint

The 140M `D768/N154/L10/R2` control exposed the next node-scaling boundary.
`active_node_top_k=32` limited private node transforms, but the baseline route
passport was reconstructed as a dense `[B,T,L,N]` tensor after every top-k.
Learned ingress scored all nodes, occupancy and receiver memory retained the
node axis, and relay transport scattered back into dense storage. The frozen
baseline was therefore sparse in node GEMMs but not sparse end to end.

Implemented as separately switchable prototypes:

1. `PackedNodeFrontier` carries fixed-capacity `(node_ids, node_mass, valid)`
   state between relays. Dense route passports are materialized only when a
   diagnostic trace explicitly requests them.
2. Packed occupancy, exact matrix-memory, direct-convolution memory, and FFT
   memory backends preserve the dense equations while avoiding the old
   `[T,T,N]` relation tensor. Direct and FFT backends gather only requested
   receiver memories and support feature chunking.
3. Collision-merged transport and occupancy use bounded workspaces. An
   automatic dense node-only scatter path is available when `N` is no larger
   than packed capacity; larger populations retain a packed sort/search path.
   This is shape-bounded, but not yet the fused histogram/prefix-sum CUDA
   implementation.
4. Sparse node execution gathers active private weight rows before the
   Blackwell BF16 grouped GEMM cast. The packet update is scatter-added
   directly without expanding payload state to `[B,T,L,K,D]`.
5. Entropy-, margin-, and hybrid-controlled adaptive route width uses a static
   maximum-K workspace plus packet-local validity masks. This establishes the
   estimator and trace metrics; a fused bucketed dispatcher is still needed
   before smaller effective K is guaranteed to reduce wall time.
6. One shared node trunk plus private low-rank node adapters is implemented as
   an architectural A/B against fully private node matrices.
7. Hierarchical learned ingress first selects coarse groups and then scores
   nodes in the selected groups. Every node remains structurally reachable.
8. Experimental active-row AdamW support and `active_node_union_fraction`
   tracing were added to determine whether optimizer-row sparsity exists at
   real language batch sizes.
9. The shared router no longer concatenates hidden/context state or gathers a
   rank-expanded edge-key bank. At the 140M shape its all-source score GEMM
   reduces the retained router intermediate from about 1.32 GiB to 67.7 MiB.
10. Packed convolution memory reuses one FFT kernel spectrum, fuses the
    denominator with the first feature chunk, and vectorizes K-route writer
    scatter and reader gather/reduction.
11. A real-valued `fft_opaque` custom-op boundary keeps complete cuFFT forward
    and recomputation backward calls outside Inductor's complex lowering.
12. An optional `torch.library.triton_op` fuses packed-reader denominator
    normalization, node gather, route weighting, and K reduction. Its Triton
    backward accumulates numerator, denominator, and route-mass gradients.
13. A second optional Triton path builds writer features directly in
    FFT-native `[B,C,U,T]` layout and applies an explicit adjoint. It removes
    the `[B,T,L,K,C]` route-state expansion and preserves the fused mass/first
    feature transform. Non-first backward chunks also reuse their recomputed
    source spectra, and the full-node path skips identity `searchsorted`.

Still pending:

1. Sweep FFT feature chunks 32/64/128 with the winning Triton-reader/Torch-
   writer path. Width 768 currently requires 24 feature chunks at C32; C128
   reduces that to six and directly tests whether repeated cuFFT/IRFFT calls,
   backward recomputation, and launch structure are now dominant. Report the
   memory-speed frontier rather than selecting on speed alone.
2. Validate direct convolution versus FFT, chunk size, adaptive K, shared
   adapters, and hierarchical ingress on the RTX PRO 6000 Blackwell.
3. Cache context deposits and move only output packets after the bootstrap
   relay, behind a semantic A/B flag.
4. Add recomputation or a custom backward for chunked receiver memory;
   forward chunking alone does not guarantee that autograd releases every
   chunk at peak backward memory.
5. Replace eager-only active-node compaction with a static-shape or fused
   implementation before enabling it under `torch.compile`.

Target-GPU validation found that the July 2026 Colab TorchInductor build fails
raw compiled FFT receiver memory at first forward with `KeyError:
'complex64'`. Raw `fft` therefore remains eager-only. `fft_opaque` now isolates
the complete real-input/real-output FFT receiver and its backward behind
custom operators, while the packed reader can be selected independently as
`torch` or `triton`. Local schema, fake-tensor, AOT, forward, and gradient
tests pass. This is not promoted to a working GPU optimization until the
CUDA-only opcheck/equivalence tests and warmed Blackwell benchmark pass.

Execution-only stages now match dense-oracle logits, route mass, summaries,
and gradients on small graphs. The current full CPU suite passes 587 tests
with 15 expected CUDA/Triton-only skips. Adaptive
top-k, frozen context deposits, shared node trunks, and hierarchical ingress
are architectural A/Bs and must remain separately switchable. The tests prove
equation and gradient compatibility; they do not prove a GPU speedup.

Tiny CPU smoke timing illustrates why that distinction matters. Packed matrix,
direct-convolution, and FFT forward times were approximately 97.6, 57.4, and
49.9 ms respectively; forward-plus-backward was approximately 131.2, 103.5,
and 112.0 ms. The tiny dense control forwarded in approximately 13.9 ms.
Python sparse bookkeeping can dominate a small CPU case, so only the target
Blackwell benchmark can decide whether the packed implementation pays off.

## Convergent general-assistant scaling gate

- [x] Preserve a dense oracle and add exact packed-frontier, matrix-memory,
  direct-convolution, and FFT controls with forward and gradient tests.
- [x] Add adaptive route-width, shared-trunk/private-adapter, and hierarchical
  ingress controls without changing legacy defaults or the 10,010-parameter
  reference configuration.
- [x] Emit effective route K, route uncertainty, and active-node union metrics
  so sparsity claims can be checked against executed work.
- [ ] Benchmark pure forward, forward-plus-loss, backward, optimizer, batch
  preparation, validation, and checkpoint time independently with pretokenized
  device-resident batches.
- [ ] Sweep batch, context length, output slots, width, lanes, nodes, degree,
  and relays one axis at a time; report tokens/s, latency, achieved FLOP/s,
  allocated/reserved VRAM, and kernel count.
- [ ] Fit measured cost and memory curves rather than extrapolating from the
  186k model. Preserve at least lengths 96, 256, 512, 1,024, and 2,048.
- [ ] Complete fusion of the packed receiver-memory hot path before
  long-context scaling. Triton reader and writer paths now remove the two
  expanded `[B,T,L,K,C]` intermediates, but active-node initialization,
  convolution/IRFFT outputs, temporary lifetime, and surrounding sparse
  orchestration still need target-GPU profiling and fusion.
- [ ] Measure vocabulary readout separately. It scales with output slots and
  vocabulary (`B * O * E * V`) rather than full context length, but will matter
  when moving from vocab 2,048 to assistant-scale tokenizers.
- [ ] Establish a batched and batch-one inference curve from saved checkpoints.
  Training throughput cannot substitute for first-token latency, complete
  simultaneous-formation latency, or sustained request throughput.
- [ ] Do not claim assistant-scale viability until capability and performance
  move together: the model must beat the historical language floor and fixed
  semantic probes while the measured compute/memory curve remains tractable.

## Rules for every item

- Preserve model parameters, equations, routing semantics, and batch order
  unless the work is explicitly labeled as an architecture/estimator A/B.
- Keep the current implementation as a numerical oracle behind an API switch
  until the replacement passes its release gate.
- Measure N154 and N308 Convergent language shapes after compilation warmup.
- Report forward, backward, loss, gradient clipping, optimizer, end-to-end
  step time, kernel count, GPU idle gaps, allocated/reserved peak VRAM, and
  formation-quality metrics separately.
- Use forward and gradient equivalence tests where the estimator is unchanged.
- Do not promote an isolated CPU or forward-only benchmark to a GPU training
  claim.
- Add tests, benchmark controls, documentation, and one rollback commit before
  moving to the next numbered item.

## 1. Fuse or replace receiver memory

- [ ] Attribute the current grouped convolution, normalization, node
  weighting, and final reduction with PyTorch Profiler and Nsight.
- [x] Implement exact direct-convolution and FFT formulations, including
  causal/noncausal behavior, duplicate writers, epsilon handling, and gradient
  oracle tests.
- [ ] Benchmark direct convolution against FFT
  formulation at longer contexts.
- [x] Prototype a fused CUDA/Triton packed-reader epilogue that avoids
  materializing selected numerator, selected denominator, node-memory, and
  weighted-route tensors. It has an explicit Triton backward and no silent
  fallback.
- [x] Integrate the Triton reader backward into the opaque FFT VJP through the
  outer explicit adjoint, avoiding nested generated autograd.
- [x] Fuse packed writer construction and its adjoint without materializing
  route-state expansions. The result is written directly in FFT-native layout;
  numerator and denominator convolution outputs remain global because multiple
  lanes/routes reuse them.
- [ ] Replace the N-sized writer initialization/FFT table with a compile-safe
  fixed-cap active-node union. This is the remaining reason more nodes can
  improve packed memory advantage while still harming speed.
- [ ] At high N, A/B retaining the first writer/numerator chunk through the
  backward chunk loop versus recomputing it immediately before its adjoint.
  Retention saves one writer+FFT call but pins an N-sized chunk across the loop.
- [x] Preserve the dense matrix backend as the semantic oracle and verify
  forward and gradient agreement on small graphs.
- [ ] Record the context-length and cell-count break-even regions rather than
  selecting one universal backend from a single shape.

Current evidence: receiver memory is approximately 92% of estimated dense
forward arithmetic at the reference language shapes. That estimate prioritizes
measurement; it is not yet a hardware utilization result.

## 2. Preserve a compiled dense prefix before sparse execution

- [ ] Split a relay call at the latched dense-to-packed boundary.
- [ ] Run the mandatory broad/dense prefix through its compiled graph.
- [ ] Hand a stable boundary object to the eager or future fused sparse suffix
  without reconstructing dense state.
- [ ] Compile/cache by static boundary and shape rather than recompiling during
  the relay loop.
- [ ] Verify identical boundary state, logits, route summaries, and gradients
  against the current all-eager packed call.

The immediate target is the common `dense relay 0 -> packed relays 1..R-1`
case. A packed suffix must not force the expensive dense prefix back to eager
execution.

## 3. Use bounded packed workspaces and fused segmented dispatch

- [x] Carry a fixed-capacity `B*T*k_max` frontier with validity masks and
  collision-merged transport.
- [x] Add a node-only scatter occupancy path for populations that fit within
  frontier capacity, without reconstructing `[B,T,L,N]` passports.
- [ ] Replace remaining Python-visible dynamic `nonzero`, `unique`, sorting,
  bucket construction, and allocator churn with a bounded `B*T*k_max`
  workspace.
- [ ] Prototype fused segmented receiver reduction, collision merge, and
  handoff transport with CUB, Triton, or a custom CUDA extension.
- [ ] Measure padding efficiency, live-row occupancy, and overflow frequency.
- [ ] Define an explicit overflow policy that widens or falls back without
  silently dropping route mass.

The bounded representation must preserve Gator's ability to widen uncertain
routes; `k_max` is a capacity policy, not permission to discard novelty.

## 4. Replace exact capped full sorts with bounded top-k

- [x] When `max_active=k` is finite, select only the candidates that can affect
  the capped result.
- [x] Add uncertainty-controlled packet-local K with a static maximum shape
  and validity masks.
- [ ] Sort the retained candidates only where ordering is still required by
  nucleus, forced-group, and scout semantics.
- [ ] Keep the full-sort path for uncapped routing and as an oracle.
- [ ] Prove equivalence across ties, forced groups, scouts, zero mass, and
  uncertainty-driven widening.
- [ ] Profile both the 62-cell and 1,038-cell populations; the expected payoff
  grows with cell count.

This item changes execution only when bounded selection is mathematically
equivalent to the current capped result.

## 5. Remove host barriers and fuse optimizer/alignment work

- [ ] Keep finite-loss checks, loss values, and gradient norms on device during
  ordinary steps; transfer scalars only at scheduled health/log intervals.
- [ ] Measure fused/foreach AdamW against the current optimizer while
  preserving checkpoint state and update semantics.
- [ ] Separate gradient clipping time from optimizer time.
- [ ] Compile or fuse the monotonic alignment recurrence so dependent
  per-output-slot operations do not become dozens of small launches.
- [ ] Add asynchronous failure reporting that does not hide nonfinite states.

`ActiveRowAdamW` is an experimental measurement tool, not the production
optimizer. Exact AdamW cannot simply skip an inactive row: momentum and weight
decay continue to update that row even when its current gradient is zero. A
lazy active-only update therefore changes optimizer semantics, while retaining
dense Adam moments fails to save optimizer-state memory. Promote active-row
optimization only if `active_node_union_fraction` is materially below one on
real batches and a paged/fused moment implementation passes an explicit
semantic A/B. If the union of nodes touched across a batch is nearly all N,
packet-level route sparsity does not imply optimizer sparsity.

The release gate is end-to-end no-log step time plus unchanged optimizer and
alignment results, not merely a lower Python profile.

## 6. Eliminate avoidable massive single-use tensor materializations

- [ ] Run an allocation/liveness audit and list every large tensor that is
  created, consumed by one immediate contraction/reduction/scatter, and then
  discarded.
- [ ] Record tensor shape, bytes, allocation site, sole consumer, backward
  requirement, and peak-lifetime overlap at both reference model sizes.
- [ ] Fuse producer and consumer when the intermediate does not need to exist
  as an externally visible value.
- [ ] Otherwise use chunked accumulation, direct scatter/reduction, reusable
  workspaces, or selective recomputation/checkpointing instead of allocating
  the complete tensor.
- [ ] Avoid `cat`, `stack`, `expand(...).contiguous()`, repeated index grids,
  and message tensors when a direct contraction or indexed accumulation gives
  the same result.
- [ ] Do not use unsafe in-place mutation on tensors autograd has saved for
  backward. Compare both gradients and peak allocated/reserved VRAM.

Implemented first cuts:

- [x] Sparse node updates map route rows directly to packet IDs and
  scatter-add the result without expanding payload state to `[B,T,L,K,D]`.
- [x] Direct/FFT packed memory avoids the dense `[T,T,N]` relation tensor and
  supports feature-chunked accumulation.
- [x] Active private node rows are gathered before BF16 conversion.
- [x] Split shared-router hidden/context projection to remove the temporary
  semantic concatenation.
- [x] Replace rank-expanded dynamic edge-key gathering with one source-score
  GEMM plus active-source gather (about 20x smaller retained intermediate at
  the 140M target shape).
- [x] Reuse FFT kernel spectra, fuse denominator/first-chunk convolution, and
  vectorize per-route packed memory scatter/gather operations.
- [ ] Confirm with CUDA memory history which chunk outputs autograd retains,
  then add selective recomputation or a custom backward where required.

Initial audit candidates include:

- Receiver `features`, `convolved`, numerator/denominator, and node-memory
  tensors.
- Dense `[B,C,T,N,N]` route logits/transitions immediately consumed by route
  transport.
- Dense `[B,C,T,N,D]` node-affine outputs immediately reduced by destination
  probability.
- Dense stewardship `flow`, state-message, and node-message tensors created
  solely for scatter accumulation.
- Expanded scatter indices, diagnostic histories, and alignment workspaces
  whose complete materialization may not be required.

First-pass FP32 inventory at the reference training shape
`B=32,T=104,N=4,D=8` is below. Sizes are per live tensor, per relay, and
exclude allocator padding, output buffers, and any additional values retained
by autograd. `A=5` is the four-neighbor stewardship configuration plus the
local/stay action.

| Allocation site | Materialized shape | 62 cells | 1,038 cells | Immediate consumer |
|---|---:|---:|---:|---|
| `language.py:668-672` gated communal input | `[B,C,T,E=32]` | 25.19 MiB | 421.69 MiB | shared input projection |
| `memory.py:177-183` numerator input | `[B,D,C,N,T]` | 25.19 MiB | 421.69 MiB | feature concatenation |
| `memory.py:180-193` convolution features/output, each | `[B(D+1),CN,T]` | 28.34 MiB | 474.40 MiB | grouped convolution / normalization |
| `memory.py:196-203` node memory | `[B,C,T,N,D]` | 25.19 MiB | 421.69 MiB | node-weighted reduction |
| `language.py:926` router input concatenation | `[B,C,T,2D+N]` | 15.74 MiB | 263.55 MiB | router projection |
| `language.py:927-947` edge logits/transition, each | `[B,C,T,N,N]` | 12.59 MiB | 210.84 MiB | softmax / passport transport |
| `language.py:953-975` node affine/destination/weighted update, each | `[B,C,T,N,D]` | 25.19 MiB | 421.69 MiB | activation / node reduction |
| `language.py:1000-1009` handoff input concatenation | `[B,C,T,2D+N]` | 15.74 MiB | 263.55 MiB | handoff projection |
| `stewardship.py:928-937` flow | `[B,C,T,A]` | 3.94 MiB | 65.89 MiB | three transport reductions |
| `stewardship.py:935-944` state messages | `[B,T,CA,D]` | 31.48 MiB | 527.11 MiB | destination scatter-add |
| `stewardship.py:946-955` node messages | `[B,T,CA,N]` | 15.74 MiB | 263.55 MiB | destination scatter-add |

These figures are an inventory, not yet a peak-memory attribution. The first
measurement pass must establish which allocations overlap because backward
retains them, which are compiler-fused already, and which dominate actual
allocated/reserved peaks.

The first exact transformations to test are:

1. Fold the per-cell feature gate into the shared input projection instead of
   broadcasting all 32 embedding channels to every cell.
2. Split router and handoff weights by state/context/traffic input, then sum
   their projections instead of concatenating the inputs. Project
   token-invariant traffic once and broadcast its much smaller result.
3. Specialize fixed ring handoffs as sums of shifted per-action contributions,
   or accumulate actions in chunks, instead of constructing `CA` state and
   node messages for a generic scatter.
4. Fuse node affine, activation, and destination-probability reduction, with
   recomputation in backward where retaining the expanded node output would
   erase the memory saving.
5. Fuse or checkpoint the receiver correlation-normalization-reduction chain.
   This is the largest combined compute and tensor-lifetime target.
6. Tile packed receiver scores and retain only per-row normalization state,
   analogous to a segmented FlashAttention kernel, instead of preserving full
   receiver-by-writer score and probability grids.

The full-attention receiver remains a small-shape numerical oracle. It must be
guarded against production-scale execution: one `[B,C,T,T,N]` FP32 tensor is
327.44 MiB at 62 cells and 5.35 GiB at 1,038 cells before its other score and
probability copies. Full route histories similarly remain opt-in diagnostics;
ordinary training should stream summaries rather than retain dense histories.

This item is cross-cutting and may share kernels with items 1-5, but its memory
claim is separate: report avoided bytes and peak VRAM, not only wall-clock
speed.

## 7. Test shared node trunks with private node adapters

- [ ] Keep the current fully private `W_n` node transforms as the numerical
  and capability oracle.
- [x] Factor node computation into one shared trunk plus
  private node identity, bias, and low-rank/FiLM adapters.
- [ ] Match parameter count against a private-node control by spending saved
  capacity on either more nodes or wider packets.
- [ ] Measure whether private adapters preserve specialization, route
  diversity, terminal formation, and the modular-prefix phase transition.
- [ ] Report the compute and VRAM break-even separately from capability.

Promotion is intentionally deferred until the shared low-rank router, learned
ingress, and active-node execution have clean capability and hardware A/B
results. The model must not erase node identity merely to make the graph
cheaper; the hypothesis is a shared method of computation with private
node-specific corrections.

The implementation exists, but capability and target-GPU break-even remain
unmeasured. The current parameter form is approximately
`D^2 + D + N * (2D + 2DR)` for adapter rank `R`, excluding the model's separate
relative-node bias.

## 8. Hierarchical learned ingress

- [x] Implement a coarse learned group gate followed by fine node scoring
  within selected groups.
- [x] Register a deterministic node-to-group layout and preserve structural
  reachability for every node.
- [ ] Measure flat versus hierarchical ingress latency, gradient quality,
  routing diversity, and capability at both N154 and N308.
- [ ] Fuse coarse selection and fine key gathering if hierarchy survives the
  capability A/B.

At N308 with router rank 96, 28 groups, and two selected groups, the prototype
scores roughly 50 keys per packet instead of 308, about a 6.2x reduction in
ingress key scoring, while adding 2,716 group parameters. This is an operation
count, not a measured end-to-end speedup.

## Target-GPU profiling matrix

The first target is the actual 140M-class Blackwell shape rather than the old
62/1,038-cell Arrow proxy:

`B=6, context=512, output=128, T=640, D=768, E=768, V=32768, N=154,`
`lanes=10, relays=2, degree=3, router_rank=96, k_max=32`.

Keep corpus batch, checkpoint initialization, objective, seed, dtype policy,
and compile state identical within each comparison. Run this ordered matrix:

| ID | Nodes | Route backend | Receiver | Node compute | Ingress | Route width | Purpose |
|---|---:|---|---|---|---|---|---|
| A | 154 | dense | matrix oracle | full private | flat | fixed 32 | preserved numerical baseline |
| B | 154 | packed | matrix oracle | full private | flat | fixed 32 | isolate frontier bookkeeping |
| C | 154 | packed | direct convolution | full private | flat | fixed 32 | remove pairwise relation tensor |
| D | 154 | packed | FFT | full private | flat | fixed 32 | long-context receiver alternative |
| E1 | 154 | packed | winning C/D | full private | flat | fixed 8 | route-width scaling |
| E2 | 154 | packed | winning C/D | full private | flat | fixed 2 | narrow-route bound |
| E3 | 154 | packed | winning C/D | full private | flat | adaptive 2--32 | uncertainty-driven execution |
| F | 308 | packed | winning C/D | full private | flat | fixed 32 and adaptive 2--32 | node-count scaling |
| G1 | 308 | packed | winning C/D | shared adapter R16 | flat | adaptive 2--32 | shared-compute A/B |
| G2 | 308 | packed | winning C/D | shared adapter R32 | flat | adaptive 2--32 | adapter-rank A/B |
| H | 308 | packed | winning C/D | winning G/full | 28 groups, top 2 | adaptive 2--32 | hierarchical ingress A/B |

Run compiled static-shape rows first. Measure eager active-node compaction as a
separate diagnostic because the current compaction controls are deliberately
rejected with `--compile`; do not mix that result into a compiled comparison.

For every row record forward, forward-plus-backward, optimizer, complete step,
tokens/s, kernel count, achieved FLOP/s, allocated/reserved peak VRAM,
effective route K, route uncertainty, active-node union fraction, terminal
formation metrics, and numerical drift from the dense oracle. Report compile
time and steady-state time separately.

Use 100-200 warmed steps with CUDA events and one synchronization after the
timing window. Profile compilation and evaluation/checkpoint overhead
separately.

## Rollback ledger

| Item | Status | Commit | Evidence |
|---|---|---|---|
| Pre-optimization baseline | Complete | `dbbd569` | Explicit rollback point |
| Packed frontier and occupancy | Prototype implemented | `57b6a81` | Dense-equation and gradient tests |
| Direct/FFT packed receiver memory | Prototype implemented | `57b6a81` | Matrix-oracle forward/gradient tests |
| Adaptive packet-local route K | Prototype implemented | `57b6a81` | Entropy/margin/hybrid behavior tests |
| Active node gather/direct scatter | Prototype implemented | `57b6a81` | Exact dispatch and gradient tests |
| Shared trunk/private adapters | Prototype implemented | `57b6a81` | Integration and parameter-count tests |
| Hierarchical learned ingress | Prototype implemented | `57b6a81` | Reachability, cost, and integration tests |
| Active-row AdamW | Experimental only | `57b6a81` | Exact/lazy semantic and state tests; not production-wired |
| Blackwell target matrix | Complete | `80051f8` | Four-way gate passed; Triton reader/Torch writer wins packed throughput |
| Fused segmented CUDA/Triton kernels | Pending | - | Python sparse bookkeeping remains |
| 2. Compiled dense prefix | Pending | - | - |
| 3. Fully fused bounded workspace | Pending | - | Fixed-capacity representation exists |
| 4. Fused bucketed adaptive dispatch | Pending | - | Static max-K estimator exists |
| 5. Host/optimizer/alignment fusion | Pending | - | - |
| 6. Remaining tensor-lifetime elimination | In progress | `57b6a81` | Major route/node expansions removed; backward audit pending |
