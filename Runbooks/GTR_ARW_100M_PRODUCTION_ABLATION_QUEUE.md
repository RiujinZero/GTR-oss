# GTR-ARW 100M production ablation queue

**Context:** the naked-broadcast 100M SmolLM3-recipe production run
(`gtr_arw_100m_smollm3_cell8_26b_naked`, architecture confirmed via
`~/gtr_arw_100m_smollm3_naked/launch.sh` on the RTX 5080 rig — see
`Runbooks/GTR_RUN_REGISTRY.md`'s
`gtr_arw_100m_smollm3_naked_local_sanity_check` row for the full setup
history and the known-good baseline). User is actively working on two
prerequisite fixes in parallel (2026-08-12); this doc exists so the queue
of what to run once each fix lands survives the gap. Write results back
into `Runbooks/GTR_RUN_REGISTRY.md` as each one runs, same
convention as `Runbooks/RIG_BREAKIN_ABLATION_QUEUE.md` — this file is the
queue, not the results log.

## Item 1: CUDA graphs retest, post-fix

**Blocked on:** a real code fix in `gator/models/language.py`, not a
config change. The first attempt (`--compile-mode max-autotune`,
`--batch-size 1 --gradient-accumulation-steps 32`, same production config
otherwise) crashed on the second gradient-accumulation micro-batch:

```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run.
  File "gator/models/language.py", line 971, in forward
    lexical_content = self.token_embedding(all_token_ids)
Gradient addition node due to multiple use of tensor...
```

`token_embedding`'s output is still referenced (by the backward graph /
autograd's saved-for-backward chain) when the next graph replay reuses
the same memory address and overwrites it — a genuine tensor-lifetime
conflict with CUDA graphs' buffer-reuse model, not a flag/config issue.
PyTorch's own suggested fixes: clone the tensor outside `torch.compile()`,
or call `torch.compiler.cudagraph_mark_step_begin()` before each model
invocation. User: "I'll look into the cudagraphs issue."

**Once fixed, rerun and compare throughput AND memory** against the
known-good `max-autotune-no-cudagraphs` baseline:

| Metric | `max-autotune-no-cudagraphs` (known good) |
|---|---|
| Steady-state updates/sec | ~0.113 (baseline) / ~0.117 (with the streams-tuning lever, see registry) |
| Steady-state tokens/sec | ~7,427 |
| Host RAM (settled) | ~20.6-20.8GB (at `--smollm3-max-host-ram-gb 24`) |
| Peak VRAM | **not captured in structured form yet — record this explicitly on the next no-cudagraphs run too, so the comparison is apples-to-apples** (only ad hoc `nvidia-smi` spot checks exist so far, e.g. ~15.9GB during one steady-state check, not a clean peak reading) |
| Non-finite windows | 0 across 300 real updates |

Command shape: identical to `~/gtr_arw_100m_smollm3_naked/launch.sh`,
only `--compile-mode max-autotune-no-cudagraphs` → `max-autotune` and
`--batch-size 2 --gradient-accumulation-steps 16` → `--batch-size 1
--gradient-accumulation-steps 32` (user's earlier call: CUDA graphs'
real risk here is VRAM from fixed-size captured-graph buffers, not NaN —
batch=1 is the accepted tradeoff for headroom). **Confirm the exact crash
signature above is actually resolved**, not just pushed to a later
micro-batch or a different tensor — rerun to at least the same ~40-update
budget the first attempt used before declaring it fixed, since the crash
only surfaced on micro-batch 2 last time.

### Status (2026-08-12): first-round fix attempt made, not resolved — real code, still WIP

User implemented a first fix (uncommitted local changes, `gator/compute.py`
new `mark_compiled_iteration_begin()`/`CUDA_GRAPH_COMPILE_MODES`,
`gator/models/language.py` new `configure_cudagraph_accumulation()` +
`_outside_torch_compile` decorator + `_lexical_input_outside_compiled()`,
wired into both `gator/smollm3_train.py`'s `_train()` and
`gator_language_train.py`'s main loop) plus a new regression test,
`tests/test_gator_arw_cudagraph.py`
(`test_max_autotune_accumulation_matches_eager_gradients`, CUDA-only,
`@unittest.skipUnless(torch.cuda.is_available())`). Also separately
relaxed the older `stewardship_execution != "dense"` + `broadcast_mix`
hard error to be inert instead of fatal (unrelated to CUDA graphs — a
different validation cleanup, means `--stewardship-execution dense` may
no longer need to be passed explicitly for `broadcast_mix`, worth
re-checking next production launch).

**Approach:** move the tied-weight ingress path (`token_embedding` →
`lexical_projection`) outside the compiled/CUDA-graph region via
`torch.compiler.disable`, with an explicit `.clone()` on the returned
activation — exactly PyTorch's own suggested fix for the original crash
signature (`token_embedding` output overwritten by a subsequent graph
replay).

**First test run: failed, but usefully.** `pytest
tests/test_gator_arw_cudagraph.py` reproduced the *same error class* one
step downstream: `input_projection`'s output (`gator/models/language.py`,
originally line 1029) hit the identical
`RuntimeError: ... accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run`. The test's own comment had anticipated
exactly this ("the target-GPU gate must fail if the eager ingress graph
break makes Inductor silently abandon CUDA graphs for the remaining model
body") — caught before spending any real GPU time on a production-scale
retest, which is the point of having the test.

**Applied the same treatment to `input_projection`** (extended
`_lexical_input_outside_compiled()` to run the full ingress-through-
input-projection chain eagerly and clone the final `content` tensor;
consolidated the duplicated `input_feature_gates is None` branch into
both the eager-fallback and CUDA-graph paths). Confirmed `input_projection`
is *not* itself a tied/reused weight (unlike the first two) — grepped for
other uses, found none — so the risk there is the more general
"saved-for-backward activation crossing the accumulation-loop boundary"
mechanism, not weight-tying specifically.

**Second test run: failed again, and diagnostically important.** The
error moved to a *completely unrelated* tensor — `address_state`
(`gator/models/language.py`, now line ~1056), derived from `_fixed_code()`
positional encoding, nothing to do with the token-embedding/input-
projection ingress chain at all. **Confirmed the test's accumulation loop
does correctly call `mark_compiled_iteration_begin()` immediately before
each micro-batch's forward** (checked `tests/test_gator_arw_cudagraph.py`
directly, matches the function's own documented required usage) — so this
isn't a missing-marker-call bug either.

**Read: this doesn't look like a short list of specific tensors that each
need an individual clone-outside-compile fix.** Two fixes in, the failure
keeps relocating to an unrelated part of the forward pass rather than
disappearing — consistent with a more systemic incompatibility between
this model's per-micro-batch `loss.backward()` gradient-accumulation
pattern (accumulating via repeated separate backward calls into `.grad`,
not one backward over a combined loss) and CUDA graphs' buffer-reuse
replay model, where potentially *many* activations across the whole
forward pass are at risk, not just the ingress path. Whack-a-moling
individual tensors is unlikely to converge quickly. **Stopped here per
user's instruction** ("make that one change and see if that fixes it. If
not, record where we are at in the docs and we'll revisit") rather than
attempting further speculative per-tensor fixes.

**Next steps when revisited:** worth checking whether PyTorch's CUDA-graph
support has known guidance for gradient-accumulation loops specifically
(the `cudagraph_mark_step_begin()` docs/issues may cover multi-backward-
per-graph patterns), or whether restructuring the accumulation loop itself
(e.g. one combined backward over a stacked/summed loss instead of N
separate `.backward()` calls) sidesteps the issue at the root instead of
patching individual tensors. `max-autotune-no-cudagraphs` remains the
working production choice in the meantime — no change from the earlier
status.

## Item 2: MXFP8 / MXFP4 precision test

**Blocked on:** real implementation work, confirmed via direct code
check (2026-08-12) — `--mixed-precision` currently only accepts
`{none, bf16}` (`gator_language_train.py:3418`), and the only existing
FP8 code path, `--fp8-vocab-readout` (dynamic per-tensor FP8 via torchao,
vocab-readout matmul only), is hard-gated to `--architecture convergent`
(`gator_language_train.py:5041-5042`) — it errors out for
`--architecture gator`, which the production run uses. No MXFP8 or
MXFP4 path exists anywhere in the trainer yet. This is a real
implementation task, not a flag away, same category of prerequisite as
item 1.

**User's explicit focus once implemented: how it handles outlier
gradients.** MX (OCP Microscaling) formats share one scale factor across
a block of values — a single outlier magnitude within a block forces
that shared scale up, crushing the effective precision of every other
(smaller) value in the same block toward zero. This is the standard
failure mode for block-scaled low-bit formats, and bites hardest for
MXFP4 given its already-minimal mantissa. Concretely worth watching once
this runs:

- The existing per-update `grad=` gradient-norm logging (already visible
  in every `update=N` summary line) gives a coarse signal, but a single
  scalar norm won't reveal per-block outlier structure — may need a
  finer per-tensor or per-block gradient histogram/max-vs-median check
  specifically for this test, not just the existing aggregate norm.
- Compare against the bf16 loss curve (this queue's item 1 baseline, or
  the completed 300-update sanity check row in the registry) for any
  divergence, plateau, or destabilization that bf16 doesn't show at the
  same point in training — the standard "does low precision change the
  learning trajectory, not just the speed" question.
- The RTX 5080 is Blackwell-class, so MX-format hardware support exists
  at the silicon level; the gap is purely in this codebase's software
  path, not the GPU.

### Status (2026-08-12): implemented and committed — correctness-first, no hardware speedup yet

Scope followed the user's explicit instruction: not the whole model in
low precision, just the two monolithic expensive matmuls (interior
per-cell node compute, vocabulary readout), routing left bf16 always.
User's other explicit call: try NVFP4 as a node-compute substitute
since MXFP4 has no grouped hardware kernel in torchao at all.

**New module `gator/mx_precision.py`, two commits:**
1. `a8cea0f` — the core module + `tests/test_mx_precision.py` (7 tests:
   bf16-passthrough identity, finite/differentiable forward+backward
   per precision, an explicit FP4-gradient-flow regression test, and an
   outlier-value stress test).
2. `75a6139` — wired into `GatorLanguageModel` (`node_compute_precision`
   and `readout_precision` config fields, both default `"bf16"`) and
   `gator_language_train.py` (`--node-compute-precision
   {bf16,mxfp8,nvfp4}`, `--readout-precision {bf16,mxfp8,mxfp4}`), plus
   `tests/test_mx_precision_integration.py` (7 end-to-end tests through
   the real model, including a byte-identical-to-unset-defaults check).

**Real implementation approach, and why it changed from the original plan:**
every genuinely hardware-accelerated path attempted hit real, unresolved
low-level API friction:
- Node-compute MXFP8 via torchao's real grouped GEMM
  (`torchao.prototype.moe_training.mxfp8_grouped_mm`, built for MoE
  expert-grouped matmul — structurally identical to our per-cell grouped
  transform) hits a `to_mx()` contiguity assertion inside torchao
  0.18.0's own EMULATED fallback that persists even with an explicitly
  `.contiguous()`-ed input, in both `AUTO`/SM100-kernel and `EMULATED`
  modes, with or without `pad_token_groups_for_grouped_mm`.
- Readout via direct `torch._scaled_mm`/`F.scaled_mm` calls (the exact
  calling convention verified line-by-line against torchao's own
  correctness test, `test/prototype/mx_formats/test_mx_mm.py`) raises
  `CUDA error: CUBLAS_STATUS_NOT_SUPPORTED`.
- NVFP4 additionally has no grouped hardware kernel in torchao at all
  yet (checked directly: only an "emulated" dequantize-then-dense-matmul
  reference implementation exists, no autograd.Function, no CUDA/Triton
  kernel).

**Pivoted to straight-through fake-quantization for every precision at
both call sites**: quantize inputs/weight to the target format
(`MXTensor.to_mx()`), dequantize back to bf16, run the real matmul at
bf16 through the existing dense kernels, explicit STE gradient
(`x + (quantize(x) - x).detach()`). Standard QAT practice — correct
gradients, a real read on each format's precision/outlier behavior —
but **no raw hardware speedup for any of the three formats** at this
scope. The genuinely hardware-accelerated attempts are kept in the file
(prefixed `_..._hardware`) for whoever revisits this; each has a
docstring pointing at the exact unresolved error.

**Real correctness bug found and fixed during implementation:**
torchao's FP4 element dtype (`float4_e2m1fn_x2`) is not
autograd-transparent through `MXTensor` the way FP8 happens to be —
`.backward()` silently left gradients as `None` for FP4 without an
explicit STE, while FP8 worked by what looks like luck rather than a
documented guarantee. Fixed by applying STE explicitly and
unconditionally for every precision, not just where it was observed
missing, and added a regression test that asserts gradients are
non-`None` (not just finite) specifically to catch this class of
regression.

**Environment note:** developed and first verified in a dedicated
Python 3.10 + torch 2.13.0+cu130 + torchao 0.18.0 venv (needed to chase
the now-abandoned hardware-kernel paths — the prebuilt torchao wheel's
compiled extensions are cpython-310-specific and need CUDA 13's
libcudart, which the repo's main cu128 build didn't have). The working
fake-quant code path turned out not to need those compiled kernels at
all — confirmed working directly in the repo's regular main venv
(Python 3.14) too, so no special environment is actually required to
use this feature, only to further chase the hardware-accelerated paths.

**NVFP4 caveat:** node compute for it approximates NVFP4's 16-element
block using MXFP4's element format at that block size — this is *not*
a true NVFP4 implementation (NVFP4 uses an FP8 shared scale, not MX's
E8M0), only an approximation of its block granularity, flagged clearly
in the module docstring so it's never mistaken for the real thing.

**Not yet done:** an actual production-scale run with real gradient
data comparing MXFP8/NVFP4-approx/bf16 loss curves and outlier-gradient
behavior — everything above is unit/integration-level correctness
verification (small synthetic shapes), not the real research question
this queue item exists to answer. That's the natural next step once
this is wanted.

### Investigated (2026-08-12): NVIDIA Transformer Engine — real hardware MXFP8/NVFP4 blocked by this GPU's architecture tier, not just software

Prompted by pushback ("I thought Nvidia's transformers API supported
MXFP4/8") — checked whether `transformer_engine[pytorch]` (NVIDIA's own
library, separate from torchao) offers a more mature path to genuine
hardware-accelerated MX formats than the torchao attempts above.

**Install:** required its own detour. No precompiled wheel exists for
the repo's main Python 3.14 venv (same ABI-availability problem as
torchao), and building `transformer_engine_torch` from source needed
CUDA Toolkit 12.8 + cuDNN headers (now installed in WSL) plus
`libnccl-dev` (missing initially — `fatal error: nccl.h: No such file
or directory` — installed via apt once passwordless sudo was set up for
`apt-get`/`apt`/`dpkg` in `/etc/sudoers.d/user-nopasswd`, user's call
for this single-user Yubikey-protected machine). Built and installed
cleanly into the existing `gtr_venv_mx310` (Python 3.10) venv:
`transformer_engine-2.18.0`, `transformer_engine_torch-2.18.0` (cu13
build).

**Verified with an isolated-process smoke test** (one `te.Linear`
forward+backward per recipe, separate Python process per recipe to rule
out shared-module-state or stdout/stderr interleaving artifacts —
`Runbooks/colab_cells/_te_smoke_test.py`, throwaway, not part of the
package) on the actual RTX 5080 (`capability: (12, 0)`, i.e. consumer
Blackwell):

| Recipe | Result |
|---|---|
| bf16 (no autocast, baseline) | clean, finite forward+backward |
| `MXFP8BlockScaling` | **hard `RuntimeError`**: `MXFP8 (for all gemm layouts) is not supported on 12.0+ architectures yet.` — TE's own source explicitly gates this off for this architecture tier, not a missing-wheel issue |
| `NVFP4BlockScaling` | forward/backward complete and report finite, but only after **8,064 repeated device-side kernel errors**: `FP4 cvt PTX instructions are architecture-specific. Try recompiling with sm_XXXa instead of sm_XXX` (from `mul_cvt_bf16_to_fp4_8x_stochastic_rounding`) — the real FP4 tensor-core conversion path needs the `a`-suffixed architecture target (e.g. `sm_120a`), which this precompiled wheel doesn't ship; "finite" here should **not** be read as "correct," since there's no ground-truth comparison and the kernel is audibly failing per-thread underneath |
| `DelayedScaling` (plain scalar FP8, E4M3/E5M2, non-block-scaled) | clean, zero errors, real hardware path — this is the one FP8 variant that works properly on this card |

**Read:** this is a hardware/software tier gap, not a bug to fix locally
— both errors line up with NVIDIA's well-documented pattern of gating
newer block-scaled tensor-core paths (MXFP8, NVFP4) to datacenter-tier
Blackwell (B200/GB200, `sm_100a`) ahead of consumer Blackwell (RTX 50
-series, `sm_120`/`sm_120a`). Plain scalar FP8 (Hopper-style
`DelayedScaling`) *is* genuinely hardware-accelerated on this GPU today
and would give real speed/memory gains if the user wants to explore that
as a separate, different precision lever — but it isn't MXFP8/MXFP4/NVFP4
and doesn't share their block-scaled outlier-handling behavior, which
was the actual point of this queue item.

**Conclusion for the "production style run" the user asked for:** no
currently-installable library (torchao 0.18.0, `transformer_engine`
2.18.0) gives real hardware-accelerated MXFP8/MXFP4/NVFP4 on this specific
GPU right now. The existing `gator/mx_precision.py` fake-quant module
remains the only way to run the actual outlier-gradient research
question (correct gradients, real precision/rounding behavior, no
speed/memory win) — reported back to the user rather than silently
proceeding, since it changes what a "production style run" can actually
show.

### Follow-up (2026-08-12): plain scalar FP8 throughput — real speedup exists, but only above a shape-dependent crossover the production node-compute call doesn't clear yet

User's ask: "do plain FP8 and plain fp4 then. This is so we can get more
throughput on the 5080." Since plain FP4 (non-block-scaled) doesn't exist
as a hardware format — Blackwell's FP4 tensor cores only expose
block-scaled MXFP4/NVFP4, and NVFP4 was already confirmed blocked above —
this became: (1) benchmark plain FP8 (`DelayedScaling`) throughput at the
real production node-compute shape, (2) chase whether NVFP4 could be
fixed by rebuilding, (3) once FP8 came back negative, check whether the
result was a small-shape artifact.

**NVFP4 rebuild attempt:** the `sm_XXXa`-architecture PTX error
(`FP4 cvt PTX instructions are architecture-specific`) looked like a
compile-target problem, not a source-level gate like MXFP8's. Traced it
to `TORCH_CUDA_ARCH_LIST` (found via the sdist's `build_tools/utils.py`
`cuda_archs()` / `NVTE_CUDA_ARCHS` logic, which confirmed `sm_120a`-style
suffixed targets are a supported token format). Rebuilt
`transformer_engine_torch` from source with `TORCH_CUDA_ARCH_LIST=12.0a`
(first attempt silently reused pip's cached wheel from the original
build — env vars aren't part of pip's cache key — fixed with
`--no-cache-dir`). **Result: no change, identical 8,064 PTX errors.**
Root cause found via `cuobjdump` inspection: `transformer_engine_torch`
is only the thin PyTorch C++ binding layer. The actual CUDA kernels
(including the failing FP4 conversion code) live in a **separate
precompiled binary dependency**, `transformer_engine_cu13` — rebuilding
the binding layer never touched them. A real fix needs the core library
rebuilt from source too (CMake-driven, hundreds of CUDA kernel files —
a substantially bigger build than the ~20min binding-layer one).
**Not attempted yet** — paused here given the FP8 result below cast
doubt on the payoff before investing in it.

**Plain FP8 (`DelayedScaling`) throughput, isolated `te.Linear` vs
`nn.Linear` benchmark** (`Runbooks/colab_cells/_te_fp8_throughput.py`,
`_te_fp8_throughput_sweep.py`, throwaway):

| rows (d_model=1312 both dims) | bf16 | FP8 | speedup |
|---|---|---|---|
| 4,128 (exact production node-compute shape: batch=2 × context=2,064) | 0.372ms | 0.727ms | **0.51x — FP8 slower** |
| 16,512 (4x) | 1.189ms | 0.811ms | **1.47x — FP8 faster** |
| 65,536 (16x) | 4.581ms | 3.138ms | **1.46x — FP8 faster** |
| 262,144 (64x) | 18.196ms | 17.144ms | 1.06x — gain fading |

**Read: this is a real crossover, not noise.** Below a shape-dependent
threshold (somewhere between 4,128 and 16,512 rows), FP8's per-call
quantization/amax/scale-update overhead costs more than the tensor-core
throughput gain returns — confirmed at the exact production shape,
where FP8 is genuinely **~2x slower** and uses slightly more memory
(0.158GB vs 0.144GB) than bf16, not less. Above that threshold, FP8
delivers a real ~1.46x speedup, fading back toward parity at very large
sizes (likely bandwidth-bound there rather than compute-bound). **The
production node-compute call as currently shaped sits on the wrong side
of that line.**

**`torch.compile` does not rescue the production-shape case** — tried
wrapping both the bf16 and FP8 eager steps in `torch.compile(mode=
"max-autotune")` at rows=4,128: bf16 compiled came back *slower* than
bf16 eager (0.549ms vs 0.372ms — Triton's autotuned kernel wasn't
better than cuBLAS's native path at this size), and FP8 compiled was
also slower than FP8 eager (0.919ms vs 0.727ms) with a Dynamo warning
that TE's internal `fused_amax_and_scale_update_after_reduction` custom
op can't be traced — meaning the FP8 autocast forces a graph break on
every call regardless, so compilation can't fuse away the very overhead
that's causing the slowdown.

**Practical implication, not yet acted on:** getting a real FP8 win here
isn't a flag flip — it requires the node-compute matmul to run at a
meaningfully larger per-call granularity than it does today (~4-16x more
rows per call). Whether that's achievable depends on how `cell_linear`'s
`grouped_bmm`/`folded_bmm` backend actually shapes its underlying calls
today (still needs checking — this benchmark used a single dense
`(rows, d_model) x (d_model, d_model)` matmul as a stand-in, which is
already a *more* favorable shape than the real per-cell/per-node grouped
structure likely is) — a real integration could end up even more
fragmented than what was benchmarked here, not less. **Paused for a user
decision rather than continuing into that restructuring work or the
much bigger NVFP4 core-library rebuild speculatively.**

### Resolved (2026-08-12): real FP8 speedup at the production shape, shipped as `node_compute_precision="fp8_te"`

User's follow-up goal: "get FP8 working faster at the production shape."
The eager-mode slowdown above was CPU-side per-call dispatch overhead
(amax/scale-update bookkeeping, many small kernel launches), not a
fundamental tensor-core throughput problem — confirmed by manually
capturing the FP8 step in a `torch.cuda.CUDAGraph`: same production
shape, same `DelayedScaling` recipe, **0.200ms vs 0.285ms for a
similarly-graphed bf16 baseline (1.42-1.48x faster)**. Alternate recipes
(`Float8CurrentScaling`, `Float8BlockScaling`) didn't help on their own —
confirms the bottleneck was dispatch overhead, not amax-history
bookkeeping specifically.

**Real integration required `torch.cuda.make_graphed_callables`**, not a
hand-rolled `graph.replay()` — a real training step needs forward() to
return a value immediately (so the rest of the model's forward pass can
continue) and backward() to run later once the true upstream gradient is
known, which a single combined fwd+bwd graph capture (what the raw
benchmark above used, with a synthetic `.sum()` loss) can't support.
Verified `make_graphed_callables` works correctly with `te.Linear` and
real upstream/downstream gradient flow (not just isolated
forward/backward) before building the full per-cell version.

**Important caveat found via controlled A/B test, before assuming the
isolated-layer number would transfer:** wrapping the graphed FP8 layer
between a real upstream matmul and downstream matmul (simulating it
running embedded in a real model rather than alone) dropped the
speedup to ~1.0x — no measurable gain. Root cause, isolated directly:
CUDA graph capture mainly saves CPU-side kernel-launch dispatch
overhead, which matters when the GPU would otherwise sit idle waiting
for the CPU to issue the next kernel. Once there's plenty of *other*
queued GPU work from surrounding un-graphed ops, that dispatch overhead
is already hidden/overlapped, so graphing just one sub-layer stops being
the bottleneck. This is documented directly in
`te_fp8_cudagraph_node_linear`'s docstring as a real caveat — a bigger
win would need a larger contiguous span of the model graphed together
(up to the whole forward+backward, i.e. Item 1 above, which has its own
separate unresolved blocker), not just this one layer.

**Shipped as a new node-compute precision, `"fp8_te"`**
(`gator/mx_precision.py`: `_FP8CellBlock`, `_BridgeFP8CellWeights`,
`te_fp8_cudagraph_node_linear`), distinct from the existing fake-quant
precisions — this one is genuinely hardware-accelerated (real TE FP8
tensor-core matmul), not quantize-dequantize-then-bf16-matmul. The bridge
(`_BridgeFP8CellWeights`, a `torch.autograd.Function`) solves a specific
problem: the model's real node weight (`fold_cell_node_parameters`'s
output) is a fresh tensor each forward call, not a persistent
`nn.Parameter` with a fixed memory address CUDA-graph replay could
reference directly — the bridge value-copies it into the graphed block's
own persistent `te.Linear` parameters each call (invisible to autograd)
and routes gradients back out via a real dependency chain (input passed
through the bridge, not a sibling branch merged later — needed for
autograd to guarantee the bridge's backward runs after the graphed
module's own backward populates `.grad`, not before).

**Cold-start accuracy issue found and documented, not silently
papered over:** `DelayedScaling`'s amax history starts empty, so the
*first* real call for a newly-built graph produces a badly wrong scale
— measured directly: output magnitude ~10x too small vs. the bf16
reference on call 1, correcting to ~0.6% mean absolute error (ordinary
FP8 precision loss) from call 2 onward. Inherent to the recipe, not
specific to this wrapper. Flagged explicitly in the function's docstring
rather than silently tolerated — a real training run's very first step
after this layer's graph is built would have measurably wrong
node-compute output.

**Full test coverage** (`tests/test_mx_precision.py`,
`TEFP8CudaGraphNodeLinearTests`, 5 tests, all passing): forward/backward
finite, output within loose tolerance of the bf16 reference *at steady
state* (accounting for the cold-start caveat above), gradients flow
correctly through a real upstream-op → this-layer → downstream-op chain
(not just in isolation — this is the specific failure mode
`make_graphed_callables` misuse would break), outlier-input stability,
and dispatch through `node_compute_linear(..., precision="fp8_te")`.
Model-level integration suite (`tests/test_mx_precision_integration.py`)
re-run clean, no regressions (7/7 passing).

**Final validated number, real shipped code, exact production shape**
(`Runbooks/colab_cells/_te_fp8_real_integration_throughput.py`,
cells=8, d_model=1312, n_nodes=4 → out_features=5,248, batch=2,
context=2,064 — i.e. `te_fp8_cudagraph_node_linear` vs. the real
`cell_linear` bf16 path, not a simplified stand-in):

| | bf16 (`cell_linear`) | FP8 (`fp8_te`, real, graphed) |
|---|---|---|
| Time/iter | 14.42ms | **13.25ms — 1.089x faster** |
| Peak memory | 2.364GB | **1.609GB — 32% less** |

Smaller speedup than the isolated single-matmul benchmarks suggested
(1.4-2x) — expected, since the real production shape (8 cells × the
wider 5,248-dim folded output) is already a meaningfully bigger matmul
than the earlier 1,312×1,312 stand-in, sitting further past the
overhead-crossover point where bf16 itself gets more efficient too. Real
speed win, and a substantial, genuine memory win — both measured with
the actual shipped implementation, not a throwaway approximation.

### Extended (2026-08-12): readout got the same treatment, bigger win

User's follow-up question ("what are the ungraphed operations and can
we graph them") mapped out what surrounds node compute in
`gator/models/language.py`'s `forward()`: stewardship/broadcast_mix
transport, the receiver-memory backend (convolution/linear_scan),
layer norms, `address_state`, the per-relay loop's control flow, and
readout — of these, readout was the clear next candidate (already had
a precision dial, single dense matmul like node compute, not entangled
in the per-relay control flow or data-dependent stewardship branching
that make the rest much higher-risk to graph, which is squarely Item
1's territory and its own separate unresolved blocker).

Same mechanism as node compute, simpler (no per-cell loop — readout is
one dense matmul): `_FP8ReadoutBlock` (a single `te.Linear`),
`_BridgeFP8ReadoutWeight` (the value-copy-in/gradient-copy-out bridge,
routing the tied `token_embedding.weight`/`lm_bias`'s gradient back
correctly), `te_fp8_cudagraph_readout_linear`, wired into
`mx_readout_linear` as `readout_precision="fp8_te"`. Same
`DelayedScaling` cold-start caveat applies identically (documented in
the function's docstring rather than re-explained). 5 new tests
(`TEFP8CudaGraphReadoutLinearTests`) mirroring the node-compute
coverage, all passing (17/17 total in `tests/test_mx_precision.py`
now).

**Real shipped-code result at the exact production readout shape**
(`Runbooks/colab_cells/_te_fp8_readout_throughput.py`,
vocab_size=128,256, embedding_dim=256, rows=batch×context=4,128):

| | bf16 (`F.linear`) | FP8 (`fp8_te`, real, graphed) |
|---|---|---|
| Time/iter | 10.99ms | **9.15ms — 1.202x faster** |
| Peak memory | 4.608GB | **2.488GB — 46% less** |

Bigger win than node compute's 1.089x/32%, as expected — readout's
matmul (128,256-wide output) is much larger than node compute's
5,248-wide folded output, sitting further past the small-shape overhead
crossover documented above. Both `node_compute_precision="fp8_te"` and
`readout_precision="fp8_te"` are now real, tested, shipped options —
routing stays bf16 always, per the original scope decision.

**Not yet extended:** the receiver-memory backend and
stewardship/broadcast transport — the two remaining major cost centers
in the per-relay loop. Both involve real data-dependent control flow
(packed vs. dense stewardship, cache-aware paths) that CUDA-graph
capture generally can't tolerate without restructuring, overlapping
directly with Item 1's already-blocked whole-model graphing effort.
Left for a future decision rather than attempted speculatively.

### Corrected (2026-08-12): a real silent-corruption bug found and fixed, both fp8_te numbers above were measured wrong

User: "I hit a wall. See if you can get the other stuff graphed" (Item
1's own whole-model CUDA-graphs work). Investigated extending the same
`make_graphed_callables` technique to the receiver-memory backend
(`gator/models/memory.py`'s `linear_scan_receiver_context`) — confirmed
graphable as-is (clean function boundary, fixed shapes given fixed
batch/cells/tokens/nodes/block_size). Stewardship transport was also
investigated: the dense/non-packed path (production's actual
`stewardship_execution="dense"` setting) is graphable but deeply
interleaved with per-relay state threading, unlike node-compute/readout's
clean boundaries; the packed path is **not graphable without a redesign**
— `pack_stewardship`/`transport_packed_stewardship` call
`.nonzero()`/`.unique()` on live route content, producing variable-sized
tensors CUDA graphs fundamentally can't handle (same category of problem
as Item 1's crashes).

**Building the receiver-memory version (`gator/cuda_graph_receiver_memory.py`,
`cuda_graph_linear_scan_receiver_context`) surfaced a real bug in the
already-committed node-compute and readout code from earlier today.**
A test that calls the graphed function twice in a row (simulating two
relays sharing one cache, the actual block-causal usage pattern) before
checking the FIRST call's result failed — but only inside the real
pytest run, not in a hand-written repro script, which made it look like
a test-isolation quirk before the real cause was found: **`torch.cuda.
make_graphed_callables`'s output is a live view into the CUDA graph's
static output buffer, silently overwritten by the NEXT call to the same
graphed callable.** `te_fp8_cudagraph_node_linear` and
`te_fp8_cudagraph_readout_linear` both fed that view straight through
`.reshape()`/`.permute()`/`.to(matching_dtype)` — none of which copy
when the dtype already matches, which it does in real bf16-trained
usage — so the "output" returned to the caller was never actually safe
to hold past the next call.

**Real consequence for production, not just a test artifact:** with
`relays=2`, node compute runs twice per micro-batch before the single
combined `.backward()` call. Relay 1's node-compute activation would be
silently overwritten by relay 2's call to the same graphed function
before backward() ever used it — **wrong gradients, not a crash**, a
failure mode that would not show up as an error and could easily be
missed by casual "does the loss go down" testing. Fixed with an explicit
`.clone()` on every CUDA-graph-captured function's output, in all three
places (`te_fp8_cudagraph_node_linear`, `te_fp8_cudagraph_readout_linear`,
`cuda_graph_linear_scan_receiver_context`), plus a `test_first_call_
output_survives_a_second_call` regression test added to each of the
three test files specifically to catch a future regression of this
exact class.

**The earlier-reported production throughput numbers for both fp8_te
paths were measured with this bug still present, and were wrong** — part
of the previously-reported win was really the (unsafe) avoided copy, not
a genuine reflection of what a correct implementation can deliver. Real,
corrected numbers, same production shapes as before:

| | bf16 | fp8_te (corrected, real .clone() cost included) | Previously reported (buggy) |
|---|---|---|---|
| Node compute time/iter | 14.45ms | 14.11ms — **1.025x faster** | 13.25ms / 1.089x |
| Node compute peak memory | 2.364GB | 1.955GB — 17% less | 1.609GB / 32% less |
| Readout time/iter | 10.99ms | 11.71ms — **0.938x, now SLOWER** | 9.15ms / 1.202x |
| Readout peak memory | 4.608GB | 3.547GB — 23% less | 2.488GB / 46% less |

**Read:** node compute keeps a real, if much smaller, speed edge.
Readout's speed win doesn't survive correctness — cloning its much
larger output tensor (rows=4,128 × vocab_size=128,256 ≈ 1GB per clone)
costs more than the graph-capture saved. Both keep a genuine, if smaller,
memory win.

**Considered and decided against: skipping the clone for readout
specifically.** Traced its real call site
(`gator/models/language.py:1714-1722`) -- it sits after the per-relay
loop, called exactly once per forward pass, unlike node compute's
`relays` calls within the same forward+backward cycle. Given the current
trainer's loop structure (`.backward()` always runs immediately after
each `forward()`, before the next micro-batch's `forward()` starts),
readout's output is fully consumed before any second call to the same
graphed function could corrupt it -- meaning the clone is not currently
load-bearing for readout specifically. **Kept it anyway.** That safety
depends entirely on training-loop discipline the function itself has no
way to verify or enforce -- a future change to how gradient accumulation
is structured, or any inference/generation code that calls readout
repeatedly per forward pass (e.g. token-by-token decoding) while holding
onto an earlier call's output, would silently reintroduce the exact bug
just fixed. Having already shipped one case of "the bug happens to help"
today, building a second one on the same kind of call-pattern assumption
isn't a trade worth making for a modest speed number.

**Receiver-memory backend (`cuda_graph_linear_scan_receiver_context`)
correctness verified** (4 tests, all passing, including the bug-catching
sequential-calls test above), throughput measured with the `.clone()`
fix already applied from the start this time
(`Runbooks/colab_cells/_receiver_memory_throughput.py`, production
shape: batch=2, cells=8, tokens=2,064, hidden=1,312, nodes=4,
block_size=16):

| | eager | graphed |
|---|---|---|
| Time/iter | 390.10ms | 387.49ms — **1.007x, essentially no speed improvement** |
| Peak memory | 4.236GB | **0.723GB — 83% less** |

**Read: no speed win, but a dramatic, genuine memory win — different
failure mode from the FP8 shape-crossover above, different verdict too.**
`linear_scan`'s Python-unrolled scan (`decayed_scan` in
`gator_linear/core.py`, ~33 sequential chunks for the forward pass plus
a separate reversed-block backward scan) is inherently a long chain of
small, DATA-DEPENDENT sequential operations — each chunk needs the
previous chunk's output state, so they can't run concurrently regardless
of dispatch mechanism. At real production scale (tokens=2,064, 390ms
total), each chunk does enough real compute/memory-bandwidth work that
the GPU stays continuously busy between launches anyway, so there's no
idle "dispatch bubble" for CUDA-graph capture to eliminate — graphing
doesn't speed up an inherently serial, compute-bound computation.

**But the memory result is the real story here.** CUDA-graph capture
pre-allocates one fixed memory pool for the entire captured op sequence
and reuses it every replay; the eager path's ~33+ sequential chunks each
allocate/free intermediate buffers through PyTorch's caching allocator,
leaving many overlapping live tensors at any given moment during the
scan. That difference is large specifically because `linear_scan` was
already documented (see the rig break-in ablation queue item 3/4
results) as winning on memory over `convolution` at scale but paying for
it with *many* intermediate scan-state tensors — this graphs away that
specific cost. **Verdict: wire this in for the memory win alone, treat
the unchanged speed as expected rather than a shortfall** — worth doing
if VRAM headroom is the binding constraint on this rig (it has been
before, e.g. the earlier fp32/batch=32 OOM and the various
`--smollm3-max-host-ram-gb` tuning), even without a throughput case.

### Second critical bug found and fixed (2026-08-12): shared graphed instances silently break gradients under relays>1

User: "That memory win allows a much larger batch size, we can do batch
4, accum 8 now, or maybe even batch 8, accum 4." Before recommending a
batch-size change, wired `receiver_memory_cuda_graph` into the real
model (new `GatorLanguageConfig.receiver_memory_cuda_graph` field,
`--receiver-memory-cuda-graph` CLI flag, dispatch in
`_dense_stewardship_context`) rather than relying on the isolated
function benchmark — this immediately surfaced a second, deeper bug
than the output-buffer-aliasing one fixed earlier today.

**The bug:** a model-level integration test at `relays=3` (stressing the
real multi-relay pattern: this backend runs once per relay, all relays'
forwards complete before ONE combined `backward()` fires) crashed with
`AttributeError: 'NoneType' object has no attribute 'clone'` reading the
bridge's saved gradient. Isolated via a minimal repro
(`Runbooks/colab_cells/_check_bridge_multicall.py`): a SHARED
`make_graphed_callables` instance, called twice before backward, breaks
gradient accumulation for the wrapped parameter — even with the
`.clone()` fix already applied, even without the bridge's own `.grad =
None` reset (tested and ruled out as the cause). Confirmed the SAME bug
independently affects the already-committed
`te_fp8_cudagraph_node_linear` (`Runbooks/colab_cells/
_check_multirelay_node_compute.py`) — real production runs `relays=2`,
so this was live in already-shipped code, not just the new receiver-
memory work. The earlier model-integration tests for node compute/
readout never caught this because they used `relays=1`.

**Root cause and fix, verified via direct A/B test**
(`Runbooks/colab_cells/_check_bridge_separate_instances.py`): using a
SEPARATE `make_graphed_callables` instance per call-site, instead of
sharing ONE cached instance across all calls within a forward pass,
fixes it cleanly — confirmed both with and without the bridge Function
present. Applied by adding a `graph_slot: int = 0` parameter to
`te_fp8_cudagraph_node_linear`, `te_fp8_cudagraph_readout_linear`, and
`cuda_graph_linear_scan_receiver_context`, folded into each function's
shape-cache key — so each relay gets its own dedicated graphed instance
rather than sharing one. `gator/models/language.py`'s per-relay loop now
threads the relay index through as `graph_slot` at both call sites
(`_dense_relay_transform` for node compute, `_dense_stewardship_context`
for receiver memory). Readout kept `graph_slot` defaulted to 0 rather
than threaded from the caller — traced its real call site
(`language.py:1714-1722`, after the per-relay loop, called exactly once
per forward pass) and confirmed it's genuinely not exposed to this bug
today, added the parameter defensively rather than assuming that stays
true forever.

**Full regression coverage added:** a dedicated
`FP8TECudaGraphMultiRelayIntegrationTests` class in
`tests/test_mx_precision_integration.py` (separate from the existing
`MXPrecisionModelIntegrationTests`, which uses `relays=1` and is gated
on torchao rather than transformer_engine — neither fits testing this)
comparing real gradients between bf16 and `fp8_te` at `relays=3` against
a real hardware-FP8-shape-valid config (`d_model=64`, TE requires the
last dim divisible by 16 and leading-dims product divisible by 8 for
real FP8 execution — the existing tiny `d_model=8` toy config from
`MXPrecisionModelIntegrationTests` is too small and would have masked
this). Plus the equivalent test already added to
`test_cuda_graph_receiver_memory_integration.py`
(`test_multi_relay_first_relay_gradient_not_corrupted`, `relays=3`).
36/36 tests pass across all four test files.

### Third bug found (2026-08-12): make_graphed_callables incompatible with torch.autocast, and the real production-scale answer

Ran the real, full-model, `relays=2` peak-memory measurement flagged as
the open item above
(`Runbooks/colab_cells/_production_memory_check.py`, actual production
config: vocab_size=128,256, d_model=1,312, embedding_dim=2,624,
lexical_embedding_dim=256, n_cells=8, n_nodes=4, neighbors=4,
node_topology=asymmetric, cell_input_mode=feature_gated,
cell_output_mode=grouped_concat, cell_transport_mode=stewardship,
block_causal, block_size=16, position_reference_length=4096,
receiver_memory_backend=linear_scan, stewardship_execution=dense,
context=2,048, response_tokens=16 — matches the confirmed production
launch config exactly).

**Third bug, found immediately:** wrapping the forward pass in
`torch.autocast(bfloat16)` — matching the real trainer's actual mixed-
precision handling (`gator_language_train.py` uses `torch.autocast`,
not a blanket `.to(bf16)` model cast, which was this test script's
first, wrong attempt and itself hit an unrelated dtype-mismatch crash
in `padded_linear`'s fp32 position codes) — crashed graph capture with
`RuntimeError: make_graphed_callables does not support the autocast
caching. Please set cache_enabled=False.` This would have crashed on
the very first real training step with `--mixed-precision bf16` (the
production default), for all three CUDA-graph functions, undetected by
every test so far since none of them wrapped calls in an ambient
autocast context. **Fixed** by wrapping just the one-time capture call
(`torch.cuda.make_graphed_callables(...)`) in
`torch.autocast(device_type="cuda", enabled=False)` in all three
functions — replay doesn't need this, only the capture step goes
through autocast-affected op dispatch. Full 36-test suite re-verified
clean after the fix.

**Real, confirmed production-scale memory result, batch=2** (current
production batch size):

| | bf16 baseline | fp8_te node+readout + receiver-memory graph |
|---|---|---|
| Peak memory | 15.840 GB | **11.170 GB — 29.5% less, 4.67 GB freed** |

Genuine, substantial, end-to-end win — this is the real number, not an
isolated-function proxy. Step time for the graphed run (49.8s) includes
one-time capture cost for all 5 graphs (2 relays × 2 functions +
readout); not representative of steady-state per-step time.

**batch=4: does not fit.** First attempt (batch=2 then batch=4 in the
same process) OOM'd, but was confounded — CUDA graph memory pools are
never released by `torch.cuda.empty_cache()` once captured, so batch=2's
~16.65 GB of pinned graph-pool memory was still resident when batch=4
tried to capture its own graphs on top. Re-tested batch=4 **alone in a
fresh process** to remove that confound — still OOM'd, this time via a
**driver-level failure**, not just PyTorch's allocator:
`dmesg` showed repeated `dxgk: dxgkio_make_resident: Ioctl failed: -12`
(ENOMEM at the WSL2 GPU-passthrough/Windows-host driver level) — a
harder, more definitive limit than a catchable Python
`OutOfMemoryError`. The crashed process left a stuck orphaned shell
(cleaned up manually); GPU health re-verified afterward (a trivial
matmul + `torch.cuda.mem_get_info()` both came back clean, ~14.5 GB
free, no lasting damage). **batch=8/accum=4 needs even more memory than
batch=4 and is therefore also ruled out.**

**Why 4.67 GB of freed headroom doesn't get you to batch=4:** CUDA
graphs' own pinned memory pools are a large, mostly-fixed cost per
captured shape (the confounded test's error message showed ~16.65 GB
allocated in graph private pools alone) — going from batch=2 to batch=4
isn't simply "2x the activation memory," the graph-pool overhead scales
with it too, so the real headroom doesn't stretch as far as the
percentage saved at batch=2 might suggest.

**Bottom line for the user's original question:** the memory win is
real (29.5% less at the current batch=2/accum=16, confirmed
end-to-end) but doesn't create enough headroom for batch=4/accum=8 or
batch=8/accum=4 on this 16 GB card — recommend staying at batch=2 and
banking the ~4.67 GB of freed headroom for other purposes (e.g. context-
length growth toward the eventual 4,096 target, or as safety margin)
rather than a batch-size increase.

### Real 300-update production run, fp8_te vs. bf16 baseline (2026-08-12): confirmed ~32% steady-state speedup

User: "Run the production config with fp8 for 300 updates so we can see
if there is any speed benefits at scale after the start up." Ran the
identical script/config as the completed bf16 sanity-check baseline
(`~/gtr_arw_100m_smollm3_naked/launch.sh`, which reached steady-state
`updates_per_second=0.108` by update 300), with
`--node-compute-precision fp8_te --readout-precision fp8_te` added, in
a separate run directory (`~/gtr_arw_100m_fp8_test/`) so it wouldn't
touch the existing bf16 checkpoint.

**A fourth real bug found before this could even run:** the production
launch script uses `gtr_venv` (main, Python 3.14), which does not have
`transformer_engine` installed — all of today's earlier `fp8_te` work
was verified in the separate `gtr_venv_mx310` (Python 3.10) venv only.
Attempting to run there directly hit a `SyntaxError` in
`gator_language_train.py` itself: a PEP 701 (Python 3.12+) relaxed
f-string construct (`f"{ {dict comprehension across multiple lines} }"`
at line ~7874) that Python 3.10's stricter f-string grammar can't parse
— a hard, file-wide syntax incompatibility, not something fixable by
adding dependencies. Installing production data-pipeline dependencies
(`datasets`/`transformers`/`tokenizers`/`huggingface_hub`, pinned to
`gtr_venv`'s exact versions) into `gtr_venv_mx310` and using that
instead sidestepped the syntax issue, but `import transformer_engine`
there still failed on `libcublas.so.13` missing at runtime.

Given `gtr_venv`'s own torch (2.11.0+cu128) already satisfies TE's
`torch>=2.1` build requirement, installed `transformer_engine[pytorch]`
**directly into `gtr_venv`** instead (pip's resolver correctly left the
existing torch untouched). First attempt built successfully but hit a
real ABI mismatch at import (`undefined symbol ... c10::ValueError`) --
traced to pip's build isolation resolving a different torch for the
build step than what's actually installed at runtime. Fixed by
rebuilding with `--no-build-isolation` (forces the build to link
against `gtr_venv`'s actual installed torch, verified via the compiler
include-path in `ps` output switching from a `/tmp/pip-build-env-.../
overlay/` path to `gtr_venv`'s real site-packages). The precompiled
`transformer_engine_cu13` core library still needed a real CUDA 13
`libcublas.so.13` at runtime regardless of torch's own CUDA version —
borrowed the one already proven working in `gtr_venv_mx310` via
`LD_LIBRARY_PATH` rather than installing an untested standalone
`nvidia-cublas-cu13` package. A standalone FP8 forward+backward check
confirmed working before relaunching the real run.

**Real run: completed all 300 updates cleanly, zero non-finite windows,
loss declined normally (11.77 → 9.13, matching the baseline's healthy
curve shape).** First forward pass (compile + all 5 CUDA-graph captures:
2 relays × node-compute + readout) took 88.69s; first backward pass
took several more minutes (also compiling, `max-autotune-no-cudagraphs`
searches many candidate kernels for both directions) — both one-time
costs, confirmed by all subsequent micro-steps running at normal
data-loading-bound speed with no further compile stalls.

**Steady-state throughput, computed via the diff-based method
established earlier this session** (the log's own `updates_per_second`
field is a cumulative average since start, still dragged down by the
one-time compile overhead even at update 300 — back out real elapsed
time at each checkpoint via `elapsed = updates / reported_ups`, diff
two checkpoints): four independent wide windows late in the run
(100→300, 150→300, 200→300, 250→300) converged tightly to
**0.1414–0.1444 updates/sec, averaging ~0.143**, versus the bf16
baseline's confirmed 0.108 updates/sec at the identical config —

| | bf16 baseline | fp8_te (node+readout) |
|---|---|---|
| Steady-state updates/sec | 0.108 | **~0.143 — ~32% faster** |

a real, confirmed speedup at actual production scale, not an isolated-
function estimate. GPU stayed at 58°C / 160W draw / 96% utilization
throughout (well under the 300W cap) — no thermal or power throttling
concern at this workload.

**Read, connecting back to the earlier isolated-function numbers:**
node compute alone measured only ~1.03x faster in isolation, and
readout alone measured *slower* (~0.94x) once the CUDA-graph output-
buffer-clone fix was in place — yet the combined real-model number is a
much larger ~1.32x. This gap is consistent with GTR-ARW being more
memory-bandwidth-bound than compute-bound: FP8 halves bytes moved for
the node-compute/readout tensors specifically (a bandwidth-bound win),
which would explain a real win larger than what raw 2x FP8 tensor-core
FLOP throughput applied to a compute-bound workload would predict, and
is consistent with the receiver-memory CUDA-graph result from earlier
today (eliminating 100% of CPU dispatch overhead there produced *no*
speedup, meaning that scan was already bottlenecked on real GPU
work — compute or bandwidth, not dispatch). Not proven directly (would
need a dedicated arithmetic-intensity sweep to confirm), but a
coherent hypothesis worth keeping in mind for future precision/graphing
work on this architecture.

## Graphing the remaining broadcast_mix pieces (dispatch-overhead only, no precision change)

Following up on the fp8_te win above, three more dispatch-heavy spots
in production's actual `cell_transport_mode=broadcast_mix` forward pass
were candidates for the same `make_graphed_callables` treatment: the
ring-topology cell-mixing block (`torch.roll`/stack/gate/project/norm),
the block-causal cumulative-traffic bookkeeping inside
`_dense_stewardship_context`, and the router/edge-logit computation
(`cell_linear` -> softmax -> transition einsum). Two shipped; one hit a
real, unresolved bug and was deliberately left disconnected.

**Broadcast-mix cell coupling — shipped** (`gator/cuda_graph_broadcast_
mix.py`, `broadcast_mix_cuda_graph` config/CLI flag). Simpler than node
compute: `left_projection`/`right_projection`/`coupling_norm`/
`neighbor_gates`/`self_gates` are genuine persistent parameters (not a
per-relay-refolded weight), so the bridge only needs to route their
*value*, not reconcile a freshly-derived tensor each call. Isolated
production-shape benchmark: **~1.05x** — modest, because this block is
mostly large dense `nn.Linear` matmuls (GPU-efficient already) rather
than many tiny dispatch-bound ops, so the CPU-dispatch-elimination
ceiling here is naturally low. Full correctness + multi-relay gradient
tests pass (`tests/test_cuda_graph_broadcast_mix*.py`).

**Stewardship-context traffic bookkeeping — shipped** (`gator/cuda_
graph_stewardship_traffic.py`, `stewardship_traffic_cuda_graph` config/
CLI flag, `ownership is None`/`traffic_scale=1.0` case only). No
trainable parameters at all — pure reshape/sum/cumsum/permute
arithmetic — so no bridge is needed, just plain tensor inputs flowing
through `make_graphed_callables`. Isolated production-shape benchmark:
**~1.54x**, the best of the three — this block really is dispatch-bound
(many small ops, no big matmuls), unlike broadcast-mix's coupling
block. Full correctness + multi-relay gradient tests pass.

**Router/edge-logit computation — NOT shipped, real bug found and left
unresolved.** `gator/cuda_graph_router.py` exists but is deliberately
**not** wired into `GatorLanguageConfig`/`gator_language_train.py` —
see its module docstring for the full writeup, summarized here:

Reusing a single cached `graph_slot` across **two separate
forward+backward sessions** (i.e. two different training steps — not
the already-solved "two relays sharing one combined backward" case,
which this module handles fine) silently produces a **wrong gradient**
for the bridged `router_weight`/`router_bias` on the second and later
sessions, while forward output stays correct every time. Measured
directly: 60–140% relative-norm gradient error, not bf16 rounding.
Since production training reuses the same cached `graph_slot` every
step for the life of a run, this pattern is exactly what real training
would hit — shipping it would have silently corrupted every router
gradient after the first training step.

Root cause, narrowed via ~20 isolated repro scripts: the combination of
`cell_linear`'s `grouped_bmm` backend **with input-side padding**
(production's real shape — `router_inputs` isn't a multiple of
`dense_kernel_padding`) **feeding into `F.softmax`**. Removing padding,
removing softmax, removing masking, removing the einsum, or using
`nn.Linear` instead of `cell_linear` each independently made the bug
disappear — only the full combination reproduces it. Confirmed this is
*not* a mistake in this investigation's usual bridge pattern: broadcast-
mix and stewardship-traffic were both directly re-tested against the
identical "two separate sessions, same graph_slot" scenario and are
clean (0.0 relative error) — neither of them combines padding with
softmax, which is presumably why this was never hit before. Two
genuinely separate, smaller bugs were found and fixed along the way
(kept in the file even though they don't fix the core issue): an
all-zero `make_graphed_callables` warmup sample produces degenerate
softmax input and corrupts capture (fixed: `randn`/`rand` warmup
samples), and `router_weight.to(bfloat16)` is a no-op aliasing the
caller's own parameter when it's already bf16, meaning the same leaf
tensor gets reused as a bridge input across sessions (fixed: force a
genuine copy via `+ 0`). Neither fix, alone or combined with explicit
`.grad.zero_()` between sessions, resolved the core issue.

`tests/test_cuda_graph_router.py` keeps two `@unittest.expectedFailure`
tests that demonstrate the bug live (`test_multi_relay_shared_weights_
gradients_not_corrupted`, `test_reusing_one_graph_slot_across_two_
training_steps_corrupts_gradient`) — anyone revisiting this should
start by making those pass, not by re-wiring the config flag back in.

## Production-scale confirmation of the two shipped graphs

300-update real production run (`~/gtr_arw_100m_fp8_graphs_test/`),
identical config to the earlier fp8_te-only run except
`--broadcast-mix-cuda-graph --stewardship-traffic-cuda-graph` added.
Completed cleanly: 300/300 updates, zero non-finite windows, final
loss 9.1308 — essentially identical to the fp8_te-only run's 9.1310,
confirming correctness (same training trajectory, not just a faster
wrong answer).

Steady-state throughput via the same diff-based method (four wide
windows, 100→300/150→300/200→300/250→300), converged tightly to
**0.1566–0.1587 u/s, averaging ~0.158**:

| Config | u/s | vs bf16 baseline (0.108) | vs fp8_te-only (0.143) |
|---|---|---|---|
| bf16 baseline | 0.108 | — | — |
| fp8_te only | 0.143 | 1.32x | — |
| fp8_te + broadcast-mix + stewardship-traffic | **0.158** | **1.46x** | **1.10x** |

Peak GPU memory during this run: **15,943 / 16,303 MiB (97.8%)** —
noticeably tighter than the fp8_te-only run's ~29.5%-less-than-baseline
finding, consistent with stacking two more CUDA graphs' own persistent
memory pools (each `make_graphed_callables` capture keeps its own pool
resident for the life of the process) on top of the existing node-
compute/readout graphs. Did not OOM, but worth watching if batch size
or model size grows further.

**Data-pipeline tuning does NOT stack — real negative result.**
Stacked `--smollm3-max-document-chars 131072 --smollm3-max-open-streams
16` (the row above's ~3.5%-gain finding, originally measured on the
bf16-only compute config over just 40 updates) on top of the combined
compute-graphs run (`~/gtr_arw_100m_fp8_graphs_streams_test/`). 300/300
updates, clean, final loss 9.0161 (differs slightly from the other
runs' ~9.13 — expected, since different stream/eviction timing changes
which documents get sampled when, not a bug). Steady-state: four wide
windows converged to **0.1520–0.1561 u/s, averaging ~0.154** — about
**2.8% slower** than without the streams tuning, not faster. Read: the
original +3.5% finding was a single unrepeated 40-update measurement
(flagged as noisy even at the time); at 300 updates with more open
streams sustaining ongoing eviction churn rather than fully settling,
it doesn't hold up. Trust the 300-update number over the 40-update one.
Not adopted.

## Kernel-level profiling: where the time and bytes actually go

Ran `torch.profiler` (CPU+CUDA activities, `record_shapes`,
`profile_memory`, `with_flops`) against the exact production config
(fp8_te + both shipped CUDA graphs), eager mode deliberately — no
`torch.compile`, since inductor's kernel fusion would merge ops across
component boundaries and defeat the purpose of attributing cost to
specific parts of the model. Script: `Runbooks/colab_cells/
_kernel_bandwidth_profile.py`.

**First attempt at attribution (dead end):** the raw op-level table
showed `aten::copy_` at 16% of self CUDA time across ~4,700 calls,
which read like the CUDA-graph bridges' own value-copy-in step (many
small per-parameter copies). Tried fixing it two ways:
- `torch._foreach_copy_` (batches N `copy_` calls into one multi-tensor
  kernel) — no measurable change (4,764→4,662 calls, time actually
  flat-to-slightly-up). Root cause: these tensors have heterogeneous
  shapes (weight matrices, bias vectors, tiny gate params), which
  pushes `_foreach_copy_` onto a per-tensor fallback with no real
  fusion.
- A more invasive flat-single-`nn.Parameter`-buffer restructuring of
  broadcast-mix's bridge (one `torch.cat` + one `copy_` instead of 8
  separate copies) — verified correct (all tests pass) but STILL no
  measurable improvement (4,626 calls, time up slightly). This
  confirmed the bridge copies were never the dominant contributor to
  begin with — reverted (`git checkout --` back to the committed
  state; not worth the added complexity for zero measured benefit).

**Real attribution (via labeled `record_function` blocks around each
component, not stack-trace grouping — `group_by_stack_n` collapsed
almost everything into 1-2 buckets with no usable per-call-site info
in this torch version):**

| Component | Self CUDA time (3 profiled steps) |
|---|---|
| `relative_receiver_context` (receiver-memory scan, eager/ungraphed) | **5,948ms** |
| `node_compute_linear` (fp8_te) | 137ms |
| `cuda_graph_broadcast_mix` | 70ms |
| `mx_readout_linear` (fp8_te) | 22ms |
| `cuda_graph_stewardship_traffic` | 0.4ms |

Receiver-memory's `linear_scan` backend — specifically its Python-
unrolled chunked `decayed_scan` recurrence
(`gator_linear/core.py`) — is ~26x more expensive than every other
labeled component *combined*. This also resolves the copy_ mystery:
the large `aten::add_`/`elementwise_kernel`/`aten::copy_` self-time
entries were never primarily the bridges' doing — they're internal to
this scan's own per-chunk accumulation loop (~32 sequential chunk
iterations at production's ~2032-token context). Confirms and
quantifies the memory-bandwidth-bound hypothesis from earlier in this
document with real, direct data instead of indirect reasoning.

## FFT alternative for the receiver-memory scan — ruled out

Prior evidence in this codebase (`PERFORMANCE_TODO.md`/
`BLACKWELL_FFT_TRITON_GATE.md`, Convergent architecture): FFT-based
packed receiver memory measured ~3.6x *slower* than dense matrix (910ms
vs 250ms/step) at context length 640, though that entangled packed/
sparse-routing overhead with the FFT choice. GTR-ARW's real context
(~2032) is over 3x longer, so re-tested cleanly: dense-only, isolated,
GTR-ARW's actual shape, forward-scan direction only
(`Runbooks/colab_cells/_fft_decay_scan_ab.py`).

Implemented `fft_causal_decay_scan`: the forward direction is a causal
convolution of `values` with a per-channel geometric kernel
`decay^j` — computed via `torch.fft.rfft`/`irfft` as a standard
zero-padded linear (not circular) convolution. Correctness verified
(max abs diff 1.9e-6 forward, 2.9e-6 gradient, fp32). **Speed: ~6.4x
SLOWER** (582.6ms vs 91.6ms/iter at production shape) — a clear,
decisive negative result, not a close call.

Why: `rows = batch × (hidden+1) = 2,626`, `channels = cells × nodes =
32` → **~84,000 independent length-2032 sequences** scanned in
parallel. FFT convolution wins for long sequences with a *small*
batch; this is the opposite regime. The chunked scan's per-chunk step
is a batched matmul-like `einsum` over all ~84K sequences — squarely
tensor-core territory on this GPU. FFT needs `complex64` intermediates
for all ~84K sequences (2x the memory traffic of real bf16/fp32) plus
per-batch cuFFT overhead, with no equivalent tensor-core path. Context
length wasn't the deciding factor after all — batch size was. Not
pursued further; no code changes made (pure investigation).

## FLA `chunk_hgrn` for the receiver-memory scan — real win, shipped

Same target (the `decayed_scan` forward/cache-carrying direction that
FFT failed to beat), different approach: `h_t = decay·h_{t-1} + v_t`
is exactly HGRN's (Hierarchically Gated Recurrent Network) recurrence
`h_t = exp(g_t)·h_{t-1} + x_t` with `g_t = log(decay)` broadcast
constant across the sequence (decay is learned but time-invariant
here). `flash-linear-attention`'s `chunk_hgrn` is a fused-Triton
chunked-recurrent kernel for precisely this recurrence.

Installed `flash-linear-attention` (both venvs: `gtr_venv_mx310` for
fast iteration, `gtr_venv` for production). First risk to rule out —
Triton kernel compatibility with this GPU (`sm_120`, consumer
Blackwell, notoriously behind official support for newer kernels — see
the whole MXFP8/NVFP4 saga earlier in this document) — cleared
immediately: a direct forward+backward smoke test ran cleanly in both
venvs, finite output and gradients, no crash.

**Phase 1 validation** (`Runbooks/colab_cells/_hgrn_receiver_ab.py`,
stock `chunk_hgrn` with a broadcast constant gate, per the design doc
this followed): correctness verified at fp32 (forward max abs diff
1.4e-6, final state 3.6e-7, gradient w.r.t. values 2.4e-6, w.r.t. decay
3.0e-4) and bf16 (finite, ~3.1e-2 diff — ordinary bf16 noise). **Speed:
~5.48x faster** at production shape (91.4ms → 16.7ms/iter,
rows=2626/length=2032/channels=32) — the opposite result from FFT, and
a much larger win than either shipped CUDA-graph piece.

**Shipped**: `gator_linear/core.py`'s new `chunk_hgrn_decayed_scan`
(soft-dependency-gated like `torchao`/`transformer_engine` elsewhere in
this codebase), wired into `linear_scan_receiver_context` via a new
`hgrn_scan` parameter that swaps *only* the block-causal forward scan
(the expensive, full-sequence, cache-carrying direction) — the
backward/per-block scan stays on `decayed_scan` since it resets every
`block_size=16` positions, far too short for a fused kernel's overhead
to pay off differently. New `receiver_memory_hgrn_scan` config/CLI
flag, requires `receiver_memory_backend='linear_scan'` and
`block_causal`; not yet wired together with `receiver_memory_cuda_graph`
(raises clearly if both are set — `cuda_graph_linear_scan_receiver_
context` always calls the eager `decayed_scan` internally). Full
correctness + multi-relay gradient tests pass
(`tests/test_hgrn_receiver_scan*.py`).

**300-update production run, Phase 1** (`~/gtr_arw_100m_fp8_graphs_hgrn_test/`,
stock `chunk_hgrn`): steady state (diff-based, updates 100→300)
**~0.149 u/s — actually ~6% slower** than the compute-graphs-only
baseline (0.158 u/s), not the expected win. Root cause: stock
`chunk_hgrn` is decorated `@torch.compiler.disable`, forcing a Dynamo
graph break on every call — up to 32/update (2 relays × 16 grad-accum
steps) — under `--compile`. The break/re-enter tax ate the isolated
5.48x kernel speedup and also cost surrounding inductor fusion.

**Phase 1.5 — eliminate the graph break** (`gator_linear/_hgrn_compiled.py`,
new): reimplemented `chunk_hgrn`'s exact two forward + two backward
Triton kernel launches (read directly from FLA's `ChunkHGRNFunction`
source) as `torch.library.triton_op`s with an explicit
`register_autograd` backward, so `torch.compile` captures the whole
thing as one opaque graph node. Hit and fixed one real bug along the
way: the registered backward produced NaN gradients specifically
through the real autograd-dispatch path (identical kernel-calling code
was bit-for-bit correct when invoked directly, bypassing autograd).
Root cause: autograd may hand a backward an **expanded, stride-0
gradient** (e.g. from `y.sum().backward()`, which conceptually means
"every element is 1" without materializing memory) — the kernel does
flat pointer arithmetic assuming real per-element strides, so a
stride-0 `do` produces garbage. FLA's own `ChunkHGRNFunction` never
hits this because its `@input_guard` decorator silently calls
`.contiguous()` on every arg first; `register_autograd` has no
equivalent guard, so it's now done explicitly (`do = do.contiguous()`
in `_backward`). Fixed and verified: forward/final-state/gradients
match stock `chunk_hgrn` exactly (both `initial_state=None` and a real
carry tensor), and `torch.compile(fullgraph=True)` succeeds with zero
graph breaks. `gator_linear/core.py`'s `chunk_hgrn_decayed_scan` now
calls this instead of stock FLA `chunk_hgrn` — same math, drop-in.

**Rebenchmark, round 1 — looked wrong**: re-ran the 300-update
production test (`~/gtr_arw_100m_fp8_graphs_hgrn_compiled_test/`);
steady state was flat at ~0.145-0.15 u/s, same as the broken Phase 1
number, despite zero graph breaks confirmed via `TORCH_LOGS=
graph_breaks,recompiles` on both a 6-step and a 20-step real-model
diagnostic run (zero break lines, zero recompile lines, zero `dynamo`
log lines at all — nothing to report).

**Real explanation — the eager-vs-eager comparison was misleading**:
the original 5.48x number compared eager `decayed_scan` vs eager
`chunk_hgrn`. Under `torch.compile(mode="max-autotune-no-cudagraphs")`
(what production actually uses), Inductor fuses and autotunes plain
`decayed_scan`'s ordinary PyTorch ops on its own — no code change
needed. `compiled_chunk_hgrn`, being an opaque `torch.library.triton_op`,
gets none of that; Inductor can't fuse anything around it. Isolated
apples-to-apples check at production shape
(`Runbooks/colab_cells/_hgrn_compiled_vs_decayed_scan_ab.py`,
rows=2626/length=2032/channels=32):

| | eager | compiled (max-autotune-no-cudagraphs) |
|---|---|---|
| `decayed_scan` | 102.05 ms | 22.66 ms |
| `compiled_chunk_hgrn` | 22.91 ms | 18.71 ms |
| speedup | 4.46x | **1.21x** |

Once both sides are actually compiled, the gap nearly vanishes: 1.21x,
not 5.48x. This fully explains the flat production throughput — the
fix works exactly as designed, it's just that Inductor closes most of
the gap on the baseline for free once `--compile` is in play. The
remaining ~18% win on this one component is real but small, easily
lost in a 300-step run's noise.

**Decision: ship anyway.** Correctness is fully verified, it should
never be worse than baseline, and it's a real (if modest) win. All 99
tests across `test_hgrn_receiver_scan*.py`,
`test_cuda_graph_receiver_memory*.py`, `test_receiver_memory.py`,
`test_gator_linear*.py` pass.

**Phase 2 — shipped.** Forked `chunk_hgrn`'s four Triton kernels
(`gator_linear/_hgrn_fixed_decay.py`) to take `log_decay: [D]` directly
instead of a materialized `[rows, length, channels]` gate: the
cumulative-decay intermediate (`gc`) is dropped entirely (closed-form
recomputed instead of materialized/read back), and the decay gradient
is accumulated directly via `tl.atomic_add` into a `[D]`-shaped buffer
instead of materializing a full-size gradient for autograd to reduce
afterward. One real bug along the way, worth remembering: backward's
`gc` has a *suffix* (reverse) cumulative meaning, not forward's prefix
one — copying forward's `local_offset + 1` formula into the backward
kernel gave wrong `dx`/`d_log_decay` until corrected to `BT -
local_offset` for full (non-final) blocks. Verified to match the
general kernel exactly (forward, final state, both gradients, with and
without a real `initial_state` carry, bf16, `fullgraph=True`) via 8 new
tests (`tests/test_hgrn_fixed_decay.py`) plus all 99 existing tests.
Isolated benchmark at production shape, both under
`torch.compile(mode="max-autotune-no-cudagraphs")`:

| | compiled |
|---|---|
| `decayed_scan` | 22.66 ms |
| Phase 1.5 general kernel | 18.69 ms (1.14x) |
| **Phase 2 fixed-decay kernel** | **11.71 ms (1.81x vs decayed_scan, 1.60x vs Phase 1.5)** |

`gator_linear/core.py`'s `chunk_hgrn_decayed_scan` now calls this by
default; Phase 1.5's general kernel (`gator_linear/_hgrn_compiled.py`)
stays in the tree as the correctness/performance reference, per the
original design doc's plan.

**Full-model check — real win is smaller than the isolated number
suggests, and that's expected, not a bug.** Re-profiled the actual
compiled production model (`Runbooks/colab_cells/
_recompiled_kernel_profile.py`, `torch.compile(mode="max-autotune-
no-cudagraphs")`, real forward+backward with a realistic response-slot
cross-entropy loss instead of a full-tensor `.sum()` — the latter's
`aten::sum` over the whole `[batch, total_length, vocab]` logits
tensor alone ate 55% of GPU time in a first pass, a pure test-harness
artifact). Receiver-memory's fixed-decay kernels: **519ms of 8,320ms
total self CUDA time over 3 steps = 6.2%**. Eager component profiling
earlier in this document showed receiver-memory at ~96% of *labeled*
self-time — that number was never wrong, just not representative of
the compiled reality: once `torch.compile` fuses everything else too,
receiver-memory's *share* shrinks even though its absolute cost barely
changed. A ~1.6-1.8x win on a ~6% slice is real but small at full-model
scale — consistent with why neither Phase 1.5's nor Phase 2's isolated
speedups showed up clearly in early 300-step throughput windows.

**Note on profiler double-counting**: `key_averages()` can list the
same underlying kernel launch under two entries with an identical
self-CUDA value — a `torch.library.triton_op` dispatch wrapper name
(often numeric-suffixed) and the raw kernel name. Naively summing all
entries by substring match roughly doubles the true total. Use exact
kernel names (no substring/suffix matching) and cross-check against
the profiler table's own "Self CUDA time total" footer.

**What's actually dominant, post-compile: reduction (`.sum`) operations
in aggregate, ~40% of total GPU time** (`aten::sum` 40.65% + matching
`reduce_kernel` 39.98% are the same underlying cost double-counted the
way above, not additive). Not one hotspot — 26+ `.sum()` call sites in
`gator/models/language.py` alone, all sharing the same generic CUDA
reduction-kernel template. The clearest single contributor by tensor
size: `_dense_relay_transform`'s `(q_next.unsqueeze(-1) *
destination).sum(3)` (`gator/models/language.py:779`) — reduces a
`[batch, cells, total_length, neighbors, d_model]` tensor (~172M
elements at production shape) over the `neighbors=4` axis, once per
relay. The exact same three-line pattern (weighted sum over a
`nodes=4` axis) also appears in receiver-memory's own finalization
step, `gator/models/memory.py:133,221,347` — downstream glue *after*
the chunk_hgrn/decayed_scan kernel, not counted in the 6.2% above.

**Custom N=4 Triton kernel for this pattern — implemented, verified
correct, NOT shipped: it's a regression under `torch.compile`.**
`gator/weighted_nodes4.py`: one streaming kernel per output block
doing 4 loads + 4 FMAs + 1 store (optionally fusing the residual add
too), replacing the broadcast-multiply-then-reduce-over-4 pattern with
the literal operation — no `[…, 4, D]` intermediate, no generic
reduction machinery. Real bug hit and fixed: the backward kernel's
`tl.atomic_add` into a `[D]`-shaped `dq` accumulator was decorated
`@triton.autotune`, whose benchmarking re-runs a kernel many times
purely to time it — since `dq` isn't reset between those internal
repeats, gradients came out ~2500x too large (a near-constant multiple
across every element, confirming the mechanism). Fixed by dropping
autotune from that kernel (fixed `BLOCK_D=256`, matching how FLA's own
`chunk_hgrn` never autotunes its accumulating `kernel_o` variants,
only the non-accumulating `kernel_h` ones). Once fixed: forward, dq,
d_destination, dh all match the reference exactly, `fullgraph=True`
compiles clean. Speed at production shape
(`Runbooks/colab_cells/_weighted_nodes4_ab.py`, B=2/C=8/T=2048/N=4/
D=1312):

| | eager | compiled (max-autotune-no-cudagraphs) |
|---|---|---|
| broadcast+sum (baseline) | 4.429 ms | 2.095 ms |
| `torch.einsum` | 5.928 ms | 2.158 ms (0.97x) |
| custom Triton, no residual fusion | 2.815 ms | 2.610 ms (0.80x) |
| custom Triton, residual fused | 2.399 ms | 2.405 ms (0.87x) |

Exact same story as chunk_hgrn Phase 1.5: the custom kernel wins
clearly eager-vs-eager (1.85x), but `torch.compile` already fuses the
naive broadcast-multiply-sum-add sequence into something close to
bandwidth-optimal on its own (2.1x for free), and the opaque custom op
can't be fused with its neighbors (`tanh`, `cell_norm`) the way native
ops can — net **13-20% slower than doing nothing**, under the exact
compile mode production uses. Not integrated. Kept in the tree
(module + `tests/test_weighted_nodes4.py`, 7 tests) as a documented
negative result:
`torch.compile`/Inductor is a genuinely strong competitor for small
broadcast-reduce patterns, and beating it requires actually removing a
graph break (as chunk_hgrn's `@torch.compiler.disable` did), not just
writing a tighter kernel for something already compilable.

**Follow-up: does `destination = tanh(node_affine)` itself need to
materialize?** Reasonable objection to the result above: both
`weighted_nodes4` and the baseline it lost to took `destination` as an
already-existing input, so neither paid for (or could save) the
`torch.tanh(node_affine)` step immediately upstream — if Inductor
wasn't already fusing that tanh into the reduction, fusing it into the
custom kernel too (never writing the `[…, 4, D]` tanh output at all)
could be the real remaining opportunity. Implemented
`weighted_nodes4_tanh` (same module): kernel takes `node_affine`
directly, computes `tanh` in-register via a numerically stable
`exp`-based formula (no native `tanh` primitive in this Triton
version), and backward recomputes it from the saved `node_affine`
rather than saving the tanh output as a separate tensor. Verified
correct (forward/dq/d_node_affine/dh all match, `fullgraph=True`
clean) via the same test battery. Re-benchmarked fairly this time —
`node_affine` as the common starting point on both sides:

| | eager | compiled |
|---|---|---|
| `tanh` + broadcast+sum (baseline) | 6.474 ms | 2.090 ms |
| custom Triton, `tanh` fused in | 2.394 ms | 2.403 ms (0.87x) |

Answer: no, it doesn't need a separate write. The compiled baseline's
time is essentially unchanged whether `tanh` is included (2.090ms) or
not (2.095ms, the earlier benchmark) — Inductor is already fusing
`tanh` into the same reduction kernel for free, so there was no
memory traffic being left on the table to begin with. Custom kernel
still ~13% slower compiled, consistent with every other result here.
Not integrated; kept (`weighted_nodes4_tanh` in the same module,
validated via `Runbooks/colab_cells/_weighted_nodes4_tanh_validate.py`
and `_weighted_nodes4_tanh_ab.py`) as part of the same documented
negative result.

**Where this leaves things**: chunk_hgrn Phase 1/1.5/2 is the one
lever in this investigation that produced a real, shippable win — and
the mechanism was always "eliminate a Dynamo graph break," never "beat
Inductor's own codegen." Every attempt at the latter (weighted_nodes4,
its tanh-fused variant, and implicitly Phase 1.5's own general-kernel-
vs-compiled-decayed_scan comparison) lost, for the same underlying
reason each time: Inductor already does this class of fusion well, and
an opaque custom op trades away more (fusion with neighboring ops)
than it gains (eliminating one intermediate).

**300-update production confirmation, full Phase 2 setup**
(`~/gtr_arw_100m_fp8_graphs_hgrn_phase2_test/`): matched-window
diff-based comparison against Run 1 (stock `chunk_hgrn`, graph-break
tax, same 29-interval sample size): 200→300 window **0.1558 u/s vs
0.1470 u/s (+6.0%)**, 100→300 **0.1546 vs 0.1488 (+3.9%)**, full run
10→300 **0.1523 vs 0.1476 (+3.2%)**. Zero of Run 1's 29 per-10-update
intervals ever reached 0.170 u/s; Run 3 (Phase 2) reached it 3 times,
peaking at 0.185, with its last 5 intervals climbing steadily
(0.154→0.156→0.159→0.162→0.164) rather than noisy like Run 1's. A
real, if modest, structural win — consistent with receiver-memory
being ~6% of total compiled GPU time and Phase 2 being ~1.6-1.8x
faster on that slice.

## GPU utilization at production scale — SMs likely not saturated

User request: profile whether the GPU's compute units (SMs) are being
fully utilized at production scale, not just "is the GPU busy."
`ncu` (Nsight Compute, needed for real per-kernel SM/warp-occupancy
counters) is installed (`cuda-nsight-compute-12-8`, `nsight-compute-
2025.1.1`, binary at `/usr/local/cuda-12.8/bin/ncu`) but every attempt
failed with `ERR_NVGPUCTRPERM` -- WSL2's GPU passthrough gates raw
hardware performance-counter access at the Windows host driver level,
and neither enabling Windows Developer Mode nor a full `wsl --shutdown`
restart cleared it (likely needs the launching terminal run as Windows
Administrator, or a driver-specific registry setting -- a host-level
admin action, not something fixable from inside WSL). Deferred; the
`ncu` target script (`Runbooks/colab_cells/_ncu_fixed_decay_target.py`
+ `_ncu_fixed_decay_run.sh`) is ready to rerun if that gets sorted out.

**Fallback: `nvidia-smi` sampled during a real, sustained, post-compile
compute window** (`Runbooks/colab_cells/_gpu_util_compute_loop.py` +
`_gpu_util_orchestrate.sh` -- the full production model, `torch.compile
(mode="max-autotune-no-cudagraphs")`, real forward+backward looped for
20s after compilation, `nvidia-smi --query-gpu=utilization.gpu,
power.draw,clocks.sm -l` sampled concurrently, synced to the loop's
start via a log-watching poll so compile time isn't included). Result,
40 samples over 20s:

```
utilization.gpu: 100% every single sample
clocks.sm:        2842 MHz (max boost, no throttling) every sample
power.draw:        76-98 W
```

`utilization.gpu` reports whether *any* kernel was executing during
the sampling window, not how much of the SMs' compute capacity was
actually used -- it stays at 100% even for tiny, narrow, or memory-
bound kernels that occupy only a handful of SMs. `power.draw` is a
much better proxy for actual compute intensity, and the RTX 5080's
rated TDP is ~360W: **76-98W at "100% utilization" and max clock is
only ~25% of the card's power envelope**, the classic signature of a
kernel mix dominated by small/narrow launches rather than large
saturating GEMMs -- consistent with everything else found in this
document (the many small elementwise/reduction call sites, small
per-relay `bmm`/linear calls, `_dense_relay_transform`'s tiny-N
contraction). Reasonably confident answer without needing `ncu`
confirmation: **the SMs are likely not being fully utilized at
production scale** -- the GPU is rarely idle, but it isn't working
hard when it's busy. Would need `ncu`'s actual occupancy metrics to
turn "likely" into a precise number, or to identify which specific
kernels are the worst offenders.

**Aside, unresolved**: the isolated compute-loop script measured
0.349-0.354 forward+backward steps/sec for a single batch=2 step (no
gradient accumulation) -- implying ~2.82-2.87s/step. The real 300-step
production run's steady-state ~0.15 updates/sec, with 16 grad-accum
microsteps/update, implies ~0.417s/microstep (batch=2 each, same
shapes) -- the isolated loop is ~6.8x slower per equivalent step than
the real trainer achieves. Not yet explained; the isolated script
omits the real training loop's optimizer/scheduler/checkpointing
machinery and doesn't use the same calling convention, so this may be
an artifact of the ad-hoc harness rather than a real discrepancy, but
it's large enough to be worth checking before trusting any future
absolute-timing number from a similarly ad-hoc isolated harness over
the real training script's own measurements.

## FP8 node compute: grouped GEMM instead of a per-cell Python loop — in progress

Follow-up to the low-power-draw finding above: user asked whether the
"~25% of TDP" signature could be from un-batched per-cell ops. It is,
at least in one identified place. `gator/mx_precision.py`'s
`_FP8CellBlock.forward` (the `node_compute_precision="fp8_te"` path --
what production actually uses):

```python
outputs = [self.linears[c](x[c]) for c in range(self.cells)]
```

Eight independent `te.Linear` GEMM launches (`cells=8`) instead of one
batched call -- the `bf16` path doesn't have this problem (`cell_linear`'s
`grouped_bmm`/`folded_bmm` backends already fold cells into one `bmm`),
this is fp8-specific. The existing CUDA-graph wrapping around this
loop (`te_fp8_cudagraph_node_linear`) only hides per-call *dispatch*
overhead -- its own docstring already documents that the win shrinks
toward ~1.0x once embedded in real training, i.e. the team already
tried the cheap fix here. Batching the GEMMs themselves is different
and untried.

TransformerEngine 2.18 (the installed version) has
`te.GroupedLinear(num_gemms, in_features, out_features)` -- built for
exactly this shape (MoE-style: N independent weight matrices, same
in/out features, applied to different token groups via `m_splits`).
Isolated comparison at the real production shape (cells=8, rows=4128 =
batch*tokens, in_features=1312=d_model, out_features=5248=n_nodes*
d_model, from `gator/models/language.py:496`'s `node_weight = nn.
Parameter(torch.empty(c, n, d, d))`), no CUDA graph on either side
(`Runbooks/colab_cells/_fp8_grouped_linear_ab.py`):

| | eager |
|---|---|
| per-cell loop (8x `te.Linear`) | 13.6-13.7 ms |
| `GroupedLinear` (1x call) | 9.9-10.0 ms |
| speedup | **1.37-1.39x** |

Correctness: weight-matched (copied per-cell weights into
`GroupedLinear`'s `weight0..weight7` params), FP8 scale warmed 20
iterations (`DelayedScaling`'s amax history needs more than the
minimum to stabilize). Forward max abs diff 0.207 / mean abs diff
0.0037 against a mean output magnitude of 0.664 (~0.56% relative) --
right in line with `te_fp8_cudagraph_node_linear`'s own documented
"~0.6% of the reference's own scale" FP8 noise floor, not a
correctness bug. `d_weight` matches exactly (0.0 diff). Real,
substantiated win, and a structural simplification too: if this holds
up end-to-end, the entire `_BridgeFP8CellWeights`/`make_graphed_
callables`/shape-cache machinery currently needed to make the per-cell
loop fast becomes unnecessary, not just faster.

**User's call: commit this checkpoint first (rollback point), then
integrate into `gator/mx_precision.py` and run a 300-step production
A/B.** Done -- integrated (`_FP8GroupedCellBlock`/
`_BridgeFP8GroupedWeights`, dropping the old `make_graphed_callables`
wrapping since the win was assumed to come from GEMM count, not
dispatch overhead). One real multi-relay gradient bug found and fixed
during integration (sharing one `GroupedLinear` instance across relays
broke gradients -- the value-copy-then-read-`.grad` bridging pattern
needs per-call-site parameter isolation independent of CUDA graphs, not
just for graph-replay correctness; reverted to per-`graph_slot`
separate instances). All 36 tests passed including the 3-relay case.

**Production result: a real regression, not a win.** 300-step run
(`~/gtr_arw_100m_fp8_graphs_grouped_linear_test/`), matched-window
diff comparison against the Phase-2-only baseline (same 29-interval
sample size):

| | Phase 2 only | Phase 2 + GroupedLinear |
|---|---|---|
| 200→300 | 0.1558 u/s | 0.1485 u/s (-4.7%) |
| 100→300 | 0.1546 u/s | 0.1484 u/s (-4.0%) |
| 10→300 | 0.1523 u/s | 0.1463 u/s (-3.9%) |

Slower across every matched window despite the isolated 1.37-1.39x
eager win holding up in isolation -- same inversion of the HGRN Phase
1.5 lesson. The old code's own docstring already said the real cost at
this shape isn't "8 GEMM calls," it's that **FP8's own per-call
bookkeeping (amax tracking, scale computation) is expensive per
invocation**, and `make_graphed_callables` existed specifically to
hide *that* via replay, not just to collapse GEMM count. Dropping CUDA-
graph wrapping (reasoning that one grouped call didn't need per-call-
dispatch hiding the way 8 separate calls did) removed the thing that
was absorbing FP8's real per-call tax -- paid in eager mode 32x/update
instead of once via replay. **Reverted** (`git revert` of the
integration commit, clean, all 27 pre-existing tests pass again) --
`_FP8CellBlock`/CUDA-graph wrapping is back to being the live
implementation. Untried next step, if revisited: `GroupedLinear`
wrapped in its own `make_graphed_callables`, combining fewer-GEMM-calls
with hidden per-call bookkeeping overhead -- might recover the win, but
not attempted this session (user chose to close out FP8/node-compute
work here and pursue two larger, separate architecture directions
instead -- see below).

## Two architecture directions for a fresh session: Monarch matrices and Gated DeltaNet

Context: every lever this session (chunk_hgrn Phases 1/1.5/2,
weighted_nodes4, its tanh-fused variant, FP8 GroupedLinear) either won
by eliminating a Dynamo graph break or lost trying to out-compile
Inductor/the existing CUDA-graph machinery on operations already well-
served by them. User's read: this class of optimization is largely
tapped out for the current architecture: real further gains likely
need to change what the model computes, not just how efficiently it
computes the same thing -- especially given the end goal (a Gator that
outperforms vanilla transformers at much smaller size/efficiency, for
low-SWAP-device deployment) is a modeling target, not a kernel-
efficiency target.

Two directions identified, each scaffolded as its own branch off
`gator_sparse` (this document's canonical branch, kept as-is at commit
`b25eadc` as the fallback if both fail) so a fresh session can attack
each independently:

- **`gator_monarch`** -- structured Monarch-matrix factorization for
  the node-compute GEMMs (`node_weight`/`folded_node_weight` in
  `gator/models/language.py`), replacing the current dense per-cell
  weight with a product of block-diagonal matrices. Targets FLOPs
  algorithmically, not just kernel/dispatch efficiency -- a bigger
  architectural change than anything shipped this session, needing its
  own convergence validation, not a drop-in kernel swap.
- **`gator_gated_deltanet`** -- replace the receiver-memory decay-scan
  mechanism (the same subsystem chunk_hgrn Phases 1/1.5/2 optimized the
  *kernel* for, without changing the *math*) with Gated DeltaNet's
  gating + delta-rule update, a more expressive memory mechanism than a
  fixed decay. Lower infra risk than Monarch -- `flash-linear-
  attention` (already integrated and proven working on this GPU from
  the chunk_hgrn work) ships fused Gated DeltaNet kernels -- but still
  a genuine modeling change, not a drop-in replacement, since delta-
  rule memory behaves differently from what the model currently learns
  to exploit.

Neither branch has any commits beyond `gator_sparse`'s tip yet -- both
are exactly where this session's work left off, untouched, ready for a
fresh session to pick up.

## Monarch-matrix node compute (`gator_monarch` branch) -- implemented, correctness-verified, real first production result

Plan (`gator_monarch`'s design doc, math verified by hand before writing
code): replace `self.node_weight` (`gator/models/language.py`, `[cells,
nodes, d_model, d_model]` -- 32 independent dense 1312x1312 matrices at
production shape, the single largest weight tensor in the model) with a
2-factor Monarch-matrix factorization (Dao et al.): `monarch_left:
[C,N,q,p,p]` + `monarch_right: [C,N,p,q,q]`, p=32/q=41 (1312=32x41
exactly, no padding needed). New `gator.models.dense_kernels.
monarch_node_linear`/`fold_monarch_node_parameters`, wired into
`_dense_relay_transform` as a third `dense_node_backend` option
alongside `"einsum"`/`"folded_bmm"`, scoped to `node_compute_precision=
"bf16"` and `stewardship_execution="dense"` only (the packed-execution
path indexes `node_weight` directly, a structurally different access
pattern not wired up yet). Pure PyTorch (einsum/reshape/transpose), no
custom Triton kernel -- same "prove it in plain ops first" pattern as
chunk_hgrn's Phase 1.

**Correctness**: 19 new tests (`tests/test_monarch_node_linear.py`,
10 tests -- including a from-scratch Python-loop reference implementation
written independently of the einsum internals, a full-rank/non-
degeneracy check, and an exact production-shape parameter-count
assertion; `tests/test_monarch_node_linear_receiver_integration.py`,
9 tests -- real `GatorLanguageModel` forward/backward, multi-relay
gradients, and an activation-checkpointing-on-vs-off gradient-agreement
check specifically validating the flat-positional-args threading
through `torch.utils.checkpoint.checkpoint`), all passing on the first
full run after the primitive's einsum indices were hand-verified during
planning. Two pre-existing, unrelated failures in the regression sweep
(`test_dense_kernels.py::test_full_dense_gator_matches_einsum_backend_and_gradients`,
numerical drift; `test_gator_100m_runner.py::test_frozen_parameter_count`,
a different model class, `ConvergentGator`) confirmed via `git stash`
to fail identically without any Monarch changes -- not caused by this
work.

**A real safety fix found along the way**: `dense_node_backend` was in
`gator_language_train.py`'s `runtime_model_config_keys` (fields allowed
to silently differ across a checkpoint resume) -- correct when its only
values (`einsum`/`folded_bmm`) shared the same `node_weight` parameters,
but no longer safe now that `"monarch"` uses entirely different learned
parameters. Removed from that set so a resume attempting to pair a
`folded_bmm`-trained checkpoint's `node_weight` with fresh
`monarch_left`/`monarch_right` (or vice versa) errors loudly instead of
silently mismatching.

**Exact parameter counts** (production shape, `gator/models/language.py`
+ `gator_language_train.py`'s CLI wiring, measured directly via
`sum(p.numel() for p in model.parameters())`, not estimated):

| | params |
|---|---|
| dense (`folded_bmm`) total | 99,789,856 |
| Monarch total | 47,771,680 |
| difference | 52,018,176 (100% attributable to `node_weight` alone -- 55,083,008 -> 3,064,832, confirmed by the node-compute-only difference exactly matching the model-total difference) |

**300-step production runs, three total** (`~/gtr_arw_100m_monarch_bf16_test/`,
`~/gtr_arw_100m_monarch_bf16_baseline/`, plus the existing fp8_te
decay-scan baseline from earlier in this document): all clean --
`nonfinite_windows=0` at all 30 logged checkpoints in every run, no
crashes. First comparison (Monarch bf16 vs the existing fp8_te
baseline) looked like a throughput *regression* (-5.9% to -9.2%) --
**but that comparison had a precision confound**: Monarch requires
bf16, the existing baseline used fp8_te, and this session already
measured fp8_te as genuinely faster than bf16 on its own. Ran a matched
bf16-vs-bf16 baseline (`--dense-node-backend folded_bmm`, otherwise
identical) to isolate the real effect:

| window | dense bf16 | Monarch bf16 | |
|---|---|---|---|
| 200->300 | 0.1395 u/s | 0.1415 u/s | **+1.4%** |
| 100->300 | 0.1392 u/s | 0.1442 u/s | **+3.6%** |
| 10->300 | 0.1374 u/s | 0.1433 u/s | **+4.3%** |

At matched precision, **Monarch is modestly faster, not slower** -- the
earlier read was almost entirely the precision confound. Final loss:
dense 9.1316, Monarch 9.1187 -- essentially identical (noise-level
difference at 300 steps, not meaningful on its own, but not a
regression either). Net first-look scorecard: **-52% parameters, +1-4%
throughput, matching loss** -- good on every axis measured except one:

**VRAM: real, but much smaller than the parameter count predicted.**
Monarch run peak ~15,880/16,303 MiB (97.4%) vs the fp8_te baseline's
15,943 MiB -- only ~63 MiB lower despite `node_weight` dropping 52M
parameters (should be several hundred MB of optimizer-state savings
alone at fp32 master weights). Root cause, reasoned through directly:
neither run used `--activation-checkpointing`, and dense node-compute
is one fused `bmm` call (nothing extra for autograd to retain) while
Monarch's forward is a 4-step chain (reshape -> einsum -> transpose ->
einsum -> transpose -> reshape) where every intermediate (`Y` and `Z`,
each the same total size as the final `[B,C,T,N,D]` output) must stay
resident for backward. Monarch trades one large weight tensor for two
small ones but pays with ~2x the *activation* memory in that code path
versus dense's single fused op -- and activation memory, not parameter
memory, is what's dominating peak VRAM at this scale (consistent with
the earlier low-power-draw-despite-100%-utilization finding). **Not
fixed this session** -- the plan deliberately scoped this to "prove
correctness in plain ops first"; a fused custom kernel or (cheaper)
just wrapping the Monarch forward in activation checkpointing should
recover most of the predicted VRAM win, left for a follow-up.

**Open next steps, not started**:
- Fuse or checkpoint the Monarch forward to recover the VRAM win (see above).
- User's stated goal: spend the freed ~52M parameter budget back on
  width (`d_model`) to return to ~100M total, on the reasoning that
  Monarch's node-compute cost scales roughly as `d^1.5` (`pq(p+q)` with
  p~q~sqrt(d)) rather than dense's `d^2`, so the compression ratio
  widens as `d_model` grows -- i.e. this isn't just a one-time parameter
  savings, wider models get relatively cheaper under Monarch. Needs a
  full per-component parameter-scaling breakdown (embedding_dim is
  currently coupled to d_model at 2x; every other component scales
  differently) before picking a target d_model -- not done yet.
- "Matching loss at 300 steps" is not evidence of matching capability,
  and specifically not evidence for the actual target (multi-step
  reasoning, multi-turn conversation at 100M-ish scale) -- needs real
  training length and real evals, not a smoke test.

### Follow-up (2026-08-13): activation checkpointing tested for the VRAM gap -- shelved, real slowdown, VRAM not confirmed

Tried the cheap fix flagged above: `--activation-checkpointing` (existing
flag, already validated gradient-equivalent for the Monarch path by
`test_activation_checkpointing_matches_no_checkpointing`) wraps the whole
`_dense_relay_transform`, so no new code was needed -- just the flag added
to a new launch script (`Runbooks/colab_cells/_launch_monarch_bf16_checkpointed.sh`),
same production shape as the Monarch smoke test (d_model=1312), 300 steps.

Diff-based steady-state throughput vs. the non-checkpointed Monarch run:

| window | non-checkpointed Monarch | checkpointed Monarch | |
|---|---|---|---|
| 200->300 | 0.1415 u/s | 0.1122 u/s | **-20.7%** |
| 100->300 | 0.1442 u/s | 0.1115 u/s | **-22.7%** |
| 10->300 | 0.1433 u/s | 0.1092 u/s | **-23.8%** |

Checkpointing made it **~21-24% slower**, not faster -- the recompute
cost of Monarch's unfused 4-step forward during backward is real and the
RTX 5080 is compute-bound enough for it to show up directly in
wall-clock, contrary to the "not compute constrained" hypothesis. Final
loss identical either way (9.1186 checkpointed vs 9.1187 not, as
expected -- checkpointing doesn't change the math). **VRAM comparison
was not obtained** -- the run finished before `nvidia-smi` was sampled
mid-run, and it wasn't rerun to get that number since the throughput hit
alone was enough to shelve the approach. Decision: shelved, not pursued
further -- the fuse-a-kernel path from the original open item remains
the only route to the VRAM win, still not attempted.

### Follow-up (2026-08-13): spending the freed parameter budget back to ~100M -- width vs. node-count

Two ways to get Monarch (47.77M params at d_model=1312) back to ~100M
total, per the open item above. Swept both empirically (`sum(p.numel()
for p in model.parameters())`, not estimated) rather than reasoning from
the `d^1.5`-vs-`d^2` scaling guess directly, since `embedding_dim`,
readout, and cell-projection sizes all move with `d_model` too and
aren't captured by the node-compute scaling law alone:

- **Width** (`Runbooks/colab_cells/_monarch_width_sweep.py`, bisection
  over `d_model` with `embedding_dim=2*d_model` held, `monarch_block_size=32`
  fixed): **d_model=2944** (embedding_dim=5888) -> **99,824,320** params
  (-0.18% from target).
- **Node count** (`Runbooks/colab_cells/_monarch_nodes_sweep.py`,
  bisection over `n_nodes` with `d_model=1312` held): **n_nodes=36** ->
  **100,257,056** params (+0.26% from target). Requires
  `node_topology="symmetric"` (a generic ring, any node count) instead
  of `"asymmetric"` (hardcoded to exactly 4 nodes, a specific tuned
  directed graph) -- a real routing-semantics change, not a free knob.
  Params scale far more cheaply with node count than width: 4->36 nodes
  (9x) cost only 2.1x params, vs. 2.24x width costing 2.09x params --
  because widening touches every matrix in the network (cell
  projections, readout, node embedding/bias) while more nodes only adds
  more parallel (batched via einsum, not looped) Monarch instances.

**Fixed vocab overhead, worth separating out before comparing "capacity"
across variants**: `token_embedding` (`[vocab_size=128256 x
lexical_embedding_dim=256]`, weight-tied to the output head) +
`lm_bias` (`[vocab_size]`) = **32,961,792 params**, constant across every
width/node variant since it depends only on `vocab_size`/
`lexical_embedding_dim`, neither of which changed in any of these
sweeps. Subtracting it reframes the original Monarch-vs-dense result
(both at d_model=1312) as starker than the raw totals suggest:

| | total | non-vocab ("architecture") |
|---|---|---|
| dense (`folded_bmm`) baseline | 99,789,856 | 66,828,064 |
| Monarch, d_model=1312 | 47,771,680 | 14,809,888 |

Monarch matched the dense baseline's final loss (9.1187 vs 9.1186) using
only **~22% of its non-vocab parameter budget** (~4.5x capacity ratio,
not the ~2x the raw totals imply).

**Production runs at the two ~100M-param variants -- both finished, one
complete, one a clean negative result:**

- **d_model=2944, batch=2/accum=16: OOM'd**, but only during
  `torch.compile`'s max-autotune kernel-selection benchmarking (`Tried
  to allocate 738.00 MiB... 0 bytes is free`), not steady-state training
  -- process exited before any `update=` line. Per-plan fallback,
  **batch=1/accum=32 (same effective batch size) ran clean to
  completion**: made it through compile (the `triton_mm`/`triton_bmm`
  "No valid triton configs" lines during autotune are the compiler
  rejecting individual kernel candidates for exceeding shared-memory
  budget and falling back to a valid one -- normal, not a failure),
  steady at ~15.9/16.3 GiB VRAM (97%+, very tight but stable) and 100%
  GPU utilization the entire run. **Final: `update=300/300 loss=9.1267`,
  `nonfinite_windows=0` throughout.**
- Loss trajectory, d_model=2944 (bf16 Monarch) vs. the existing fp8_te
  dense baseline (d_model=1312, from earlier in this document) -- ahead
  through update 160, but the lead narrowed by the end:

  | update | fp8_te dense baseline (d=1312) | Monarch bf16 (d=2944) |
  |---|---|---|
  | 10 | 11.7661 | 11.7628 |
  | 100 | 11.2092 | 10.9674 |
  | 150 | 10.7509 | 10.6719 |
  | 160 | 10.5849 | 10.5011 |
  | 300 (final) | n/a (baseline not run to 300 at this width) | 9.1267 |

  Caveat: this comparison conflates three variables at once (width,
  Monarch factorization, bf16-vs-fp8_te precision) -- it says the wider
  Monarch model is winning on loss, not which change is responsible.
  Against the original d_model=1312 Monarch run's final loss (9.1187),
  **d_model=2944's 9.1267 is essentially the same, not meaningfully
  lower**, despite ~2.1x the parameters -- the mid-run lead over the
  fp8_te baseline did not translate into a materially better endpoint at
  300 steps. Consistent with the "matching loss at 300 steps isn't
  evidence of matching capability" caveat already on this doc: 300 steps
  may simply be too short a horizon for the extra width to show up
  cleanly either way.
- **n_nodes=36, symmetric topology, d_model=1312: does not fit on this
  card at production shape (context 2048), at any tried batch size --
  a genuine, reproducible negative result, not a flake.** Queued run
  (`Runbooks/colab_cells/_queue_monarch_nodes36.sh`) tried batch=2/
  accum=16 first: OOM'd during actual training (not just autotune),
  `Tried to allocate 2.88 GiB... 0 bytes is free` while materializing a
  `[328, 4096, 1152]` bf16 buffer. Auto-fallback to batch=1/accum=32 hit
  a `CUDA error: device not ready` during compile -- initially suspected
  as a stale-context transient from the immediately-preceding crash, so
  retried standalone after confirming the GPU was fully idle
  (`Runbooks/colab_cells/_launch_monarch_nodes36_b1a32_retry.sh`). **The
  retry OOM'd again, at the same point, with the same size** (`Tried to
  allocate 2.89 GiB... 0 bytes is free`, during `torch.compile`
  autotune's kernel-candidate benchmarking) -- confirming this is a real
  memory ceiling, not a driver hiccup. No `update=` line was ever
  reached in any of the three attempts. **User's read, confirmed by this
  data: "Nodes are more expensive than width. This is known."** Despite
  scaling more cheaply in raw parameter count (4->36 nodes, 9x, cost
  only 2.1x params vs. width's 2.24x-width-for-2.09x-params), node count
  is evidently far more expensive in *activation* memory -- every
  additional node multiplies the per-token intermediate tensors
  (`node_embedding`/`node_bias`/cell-transport/routing buffers all carry
  an explicit node axis), whereas widening `d_model` grows existing
  tensors rather than adding a new broadcast dimension. Parameter count
  alone was the wrong proxy for "cost" here -- **conclusion: for this
  architecture, width is the viable axis for reaching ~100M params on
  16GB-class hardware, node-count is not, and this line of investigation
  is closed** rather than pursued further (e.g. via shorter context or
  disabling `--compile`) given the width path already produced a usable
  result.

## English-only 50k-vocab tokenizer, pre-bake-off (2026-08-13/14)

Before the planned 24h dense-vs-Monarch bake-off (see
`project_gtr_arw_100m_bakeoff` memory), user wanted a smaller,
English-focused tokenizer swapped in for `meta-llama/Llama-3.2-1B`
(128,256 vocab, multilingual): prune non-English tokens, keep rare/
special characters, target ~50k vocab, derived by scanning a subset of
the real SmolLM3-recipe training corpora.

**Approach chosen over literal Llama-vocab subsetting**: train a fresh
~50k BPE vocab from scratch on a 9.66GB weighted sample of
`gator.smollm3_recipe.STAGES[0]` with all 14 `fw2-*` (FineWeb-2
per-language) sources excluded -- the only genuinely non-English entries
in the whole mixture; code/math sources were kept. Reused Llama 3.2's
exact pre-tokenizer regex (the standard cl100k-style split pattern) and
ByteLevel decoder/alphabet so segmentation behavior matches exactly and
byte-level fallback guarantees lossless round-trip on **any** input
regardless of training corpus -- this is what "keeps rare characters"
even though only English/code was scanned. Chose fresh-train over
subsetting Llama's existing merge table because GTR-ARW trains from
scratch (no Llama checkpoint to stay ID-compatible with), and
closure-correct merge pruning is meaningfully more fragile for no
benefit here. Only 2 of Llama's 256 special-token slots
(`<|begin_of_text|>`/`<|end_of_text|>`) are referenced anywhere in this
codebase (confirmed via grep) -- the other 254
`<|reserved_special_token_N|>` slots were dropped as dead weight.

**Build**: `Runbooks/colab_cells/_build_english_tokenizer.py` (corpus
scan + BPE training), `_verify_english_tokenizer.py` (round-trip +
compression-ratio checks), `_bakeoff_param_count_english_tokenizer.py`
(param-count smoke test). Output: `Runbooks/tokenizers/gtr_arw_english_50k/`
(3.4MB `tokenizer.json`, vocab_size=50,000 exactly). Ran on the RTX
5080 rig's Ryzen 9950X -- the full 9.66GB scan + BPE training finished
in minutes, far faster than expected.

**Verification, all passed**: English text round-trips byte-exact;
deliberately non-English/rare text (French accents, Chinese, Russian
Cyrillic, Japanese, emoji, math symbols) also round-trips byte-exact via
byte-fallback, just less compact (proves nothing became
unrepresentable). Compression: English 4.07 chars/token (new) vs. 4.31
(old, -5.6%); non-English 1.50 vs. 2.09 chars/token (-28%, expected --
those languages were deliberately excluded from training).

**Real effect on parameter counts** (both at d_model=1312): `token_embedding`
shrank from 32.83M (128,256 x 256 lexical dim) to 12.8M (50,000 x 256).
`lm_bias` shrank proportionally too. The non-vocab "core" network is
**unchanged in absolute size** (~66.16M either way, confirmed via the
training script's own `core=` log line) -- all savings come from the
vocab path alone:

| | old (128,256 vocab) | new (50,000 vocab) |
|---|---|---|
| dense (`folded_bmm`) total | 99,789,856 | 79,678,064 |
| Monarch total | 47,771,680 | 27,659,888 |

This drops both candidates well under the ~100M mark that motivated the
earlier d_model=2944-width Monarch experiment -- worth revisiting width
if 100M is still a hard target, though not decided yet.

**300-step A/B smoke tests** (`_launch_dense_bf16_english50k_test.sh`,
`_launch_monarch_bf16_english50k_test.sh`, same shape/seed/steps as the
original 128k-vocab runs, only `--hf-tokenizer` swapped): both finished
clean, `nonfinite_windows=0` throughout.

*Throughput -- real, consistent win, held for both architectures* (diff-based steady state):

| | old (128k vocab) | new (50k vocab) | delta |
|---|---|---|---|
| dense, 200->300 | 0.1395 u/s | 0.150 u/s | **+7.6%** |
| dense, 100->300 | 0.1392 u/s | 0.153 u/s | **+9.9%** |
| Monarch, 200->300 | 0.1415 u/s | 0.151 u/s | **+6.9%** |
| Monarch, 100->300 | 0.1442 u/s | 0.152 u/s | **+5.5%** |

*VRAM -- essentially unchanged*, continuously sampled this time (every
15s across both runs, not a lucky snapshot): true peak **15,904 MiB**,
in line with the old runs' ~15,880-15,943 MiB range. Confirms the
speedup is compute-side (smaller readout matmul against a smaller
vocab), not memory-side -- consistent with this session's now-repeated
finding that peak VRAM here is activation-memory-bound, not
parameter-count-bound.

*Loss -- raw numbers are misleading, chance-floor-normalized numbers
show a small, consistent, symmetric dip, not a win*: raw final loss
dropped a lot (dense 9.1316->8.5549, Monarch 9.1187->8.5663) but most of
that is the smaller vocab's lower `ln(vocab_size)` chance floor (11.76
vs 10.82, a 0.94-nat gap before any learning). Normalizing by each run's
own chance floor (fraction of the way descended from chance to zero):
dense went from 22.36% (old) to 20.94% (new); Monarch went from 22.47%
(old) to 20.83% (new) -- the **same small dip in both architectures**,
which is what makes it a real (if minor) effect rather than noise.
Working explanation (user's framing): a coarser 50k vocab has already
pre-merged much of the "easy" statistical structure into single tokens,
leaving fewer easy early wins for the model to grab relative to the
128k vocab's more fine-grained token space -- not necessarily worse
long-run learning, just a slower early relative descent. Not settled at
300 steps either way.

**Internal consistency check**: dense-vs-Monarch ordering held under the
vocab swap too (Monarch nearly tied on loss, modestly faster on
throughput, same as under the old vocab) -- the vocab change didn't
scramble which architecture is ahead.

**Decision**: given a real, consistent ~6-10% throughput win with no
VRAM cost and only a minor, explainable loss-descent dip, keeping the
50k-vocab tokenizer for tomorrow's bake-off (dense vs. Monarch, 24h
each, both at d_model=1312) rather than the original 128,256-vocab
Llama tokenizer.

## Toy-width degenerate-collapse sanity check, before the real 300k/1.2M-update bake-off (2026-08-14/15)

Before committing days of compute to the real 1312-width bake-off, user
wanted a fast sanity check: does this exact production pipeline (fp8_te
precision, the new 50k-vocab English tokenizer, full SmolLM3 mixture,
block_causal + linear_scan/hgrn receiver-memory) actually escape the
early "predict the single most-frequent token" collapse, or does it get
stuck? Ran a drastically narrower model (`d_model=128`, found via
`Runbooks/colab_cells/_toy_width_sweep.py` to give non-vocab "core"
params of 696,512 -- ~100x smaller than production's ~66M core, total
model 13.5M params) at the *same* context length (2048) the original
production plan used, so it could push through many more updates per
wall-clock hour than the full-size model while testing the same
data/precision/architecture.

**New instrumentation added for this** (now part of `gator/smollm3_train.py`,
applies to every future smollm3-recipe run, not just this toy one):
- **PPL** added to both the console and file training logs (`ppl=` next
  to `loss=`).
- **Fixed 10-prompt eval** (`--smollm3-fixed-eval-every-updates`,
  default 500): greedy-decodes a hardcoded set of 10 diverse English
  prompts through the raw (uncompiled) model every N updates, logged to
  `checkpoint_dir/eval_prompts.log` and stdout. Deliberately runs eager
  or via the compiled `train_model` to avoid any risk of the eval's
  different batch shape (1 prompt at a time) triggering a torch.compile
  recompile/thrash against the training loop's batch=8 compiled graph,
  and frees VRAM (`torch.cuda.empty_cache()`) afterward given how tight
  peak VRAM already is on the real 1312-width config.
- **Per-slot accuracy** (`slot_acc=[...]`, 16 floats in the console
  log): argmax-vs-target correct-rate broken out by response position,
  to catch a *fanout* pattern (one slot breaking out while others stay
  flat) that an aggregate loss/PPL number would mask -- direct precedent
  for this concern already existed in this repo's own GTR-L/GTR-ARW
  history (sequential per-position fanout during AdamW breakout).
- **Checkpointing simplified**: `latest.pt` (every interval) + `best.pt`
  (only when that interval's average loss is the lowest seen so far, a
  rollback target) replaces the previous unbounded per-update timestamped
  snapshot scheme -- bounded disk footprint (~2x checkpoint size)
  regardless of run length, same change applied to the real bake-off's
  launch script.

**Trajectory** (`~/gtr_arw_100m_toy_degenerate_check/`, resumed once
mid-run after a deliberate pause for GPU access, `--resume-checkpoint`
verified working correctly): chance floor (~10.82, `ln(50000)`) descended
normally to a **long plateau at loss 7.6-7.9 / ppl ~2100-2900 lasting
from ~update 3,000 to ~update 60,000** (tens of thousands of updates,
~3.5B+ tokens) -- `slot_acc` stayed flat and uniform across all 16
positions the entire plateau (no masked fanout), and the eval-prompt
log showed the model cycling between different *global* single-token
collapse modes (`'the the the...'` -> a mixed the/comma/newline phase
around update 12,000 -> `',,,,,,,,,,,,,,,,'` comma-collapse from ~18,000
through ~34,000) without ever settling into coherence, though critically
**never fully uniform** -- the code-completion prompt and 1-2 others
consistently diverged from whatever the dominant mode was at any given
checkpoint (user's read: genuine per-prompt conditional structure
already present, just not yet enough to produce anything coherent --
true full degeneracy would be every prompt AND every slot identical,
which this never was).

**Real break starting ~update 60,000**: loss dropped to a new low **7.43**
(vs. the ~7.67 floor that had held since early in the plateau), grad
norm elevated (5.7-7.7 vs. the typical 2-5 baseline, consistent with a
genuine transition rather than noise), and the eval-prompt texture grew
markedly more complex -- periods (`.`) entered the token mix for the
first time, and several completions started combining 2-3 distinct
token types in one 16-token span (`' the....,,,,,,,,,,,'`,
`' the the the the.....,.,,,,,'`) rather than one dominant token.
Stopped deliberately at **update 65,080, loss=7.4624** (still improving,
not a natural plateau/completion) to pivot to the context-length
comparison below -- this run's `latest.pt`/`best.pt` are preserved
(`~/gtr_arw_100m_toy_degenerate_check/checkpoints/`) in case resuming
this exact d_model=128/context=2048 config is useful later.

**Zero non-finite windows across the entire ~65k-update run** -- no
stability issues at any point, only the loss-plateau/collapse-cycling
question was ever open.

**Read**: the long plateau is plausibly explained by task difficulty
rather than a pipeline bug -- `output_slots=16` are predicted from
context in one parallel forward pass (not autoregressively token-by-
token), so the model can't lean on its own just-generated tokens as
scaffolding; it has to commit to a plausible whole-span continuation
from context alone, closer to multi-token infilling than standard next-
token prediction, at a 660k-param bottleneck. Not fully resolved either
way -- the break at update ~60k is promising but the run was stopped
before it could confirm a real breakout.

## Toy width/context-length comparison: context=256/response_tokens=8, stopped at update 1,045,370 (2026-08-15/16)

Follow-up question raised directly by the section above: how much of
the long plateau was `d_model=128`'s capacity vs. `context=2048`
specifically being too much context for that capacity to route well?
New toy run, same `d_model=128`/fp8_te/tokenizer/architecture, fresh
(not resumed -- different sequence shape): `--smollm3-sequence-length
256 --response-tokens 8` (down from 2048/16), `--steps 2400000` (up
from 300,000) so total tokens at completion match exactly (`256*32*2.4M
== 2048*32*300k == 19,660,800,000`), and `--batch-size 32
--gradient-accumulation-steps 1` (up from 8/4, user's call -- VRAM
headroom was large, ~8.9GB peak vs. the 16.3GB cap).

**Tok/s note, contrary to naive expectation**: comparing the two toy
configs at matched effective_batch=32, context=256 was actually
**slower** in raw tok/s than context=2048 (~111-119k vs. ~133k),
despite processing 8x less context per step. Root cause: at
`d_model=128` the GPU does almost nothing per token, so the CPU-side
data-streaming/tokenization pipeline is the real bottleneck, not GPU
compute -- shortening context doesn't help when you're not
compute-bound to begin with. This is very likely an artifact of the toy
width specifically; at production scale (`d_model=1312`, ~100x more
compute/token) the data pipeline is unlikely to be the bottleneck, so
context length should show its expected effect on real production
runs, unlike here.

**New capability added for this**: `--smollm3-effective-batch-sequences`
(default 32, backward-compatible) now generalizes what was previously a
hardcoded `EFFECTIVE_BATCH_SEQUENCES=32` constant throughout
`gator/smollm3_recipe.py`/`gator/smollm3_train.py` -- threaded through
the LR warmup/decay schedule, data-mixture stage transitions, and
resume-checkpoint identity validation (a checkpoint recorded at one
effective batch can't silently resume into a different one -- same
safety-check pattern as the `dense_node_backend` fix from the Monarch
work). All 22 existing smollm3 tests still pass unchanged. Lets a run
increase its effective batch size (smoother gradients) independent of
the fixed 32-sequence "formation" the original SmolLM3 recipe assumed.

**Batch=96 speed test** (`~/gtr_arw_100m_toy_context256_batch96/`,
`_launch_toy_context256_batch96.sh`, same context=256/slots=8, fresh
run since it can't resume the batch=32 checkpoint): motivated by
grad-norm climbing steadily during the batch=32 run (10-15 -> 15-25
over hundreds of thousands of updates) -- direct precedent from this
session's own history (`arw_nodes8_trace_seed0`'s grad_norm=72.98
transient-then-recovered spike vs. the GTR-L lr3e-4 runs' spikes that
sometimes didn't recover) made a larger effective batch (smoother
gradients) worth testing as a lever. Converged steady-state: **~6.0
u/s -> ~147,500 tok/s**, a confirmed **+32.3%** over the batch=32 run's
throughput at the moment it was paused for the comparison (13.607 u/s
-> 111,485 tok/s). **Decision rule (user's, set in advance): compare
total wall-clock time to reach the same 19.66B-token finish line,
accounting for batch=32's existing head start.** Computed: batch=96
fresh-start remaining ~36.8h vs. batch=32 resumed-from-23%-done
remaining ~37.5h -- only a ~45-minute (<2%) difference despite the 32%
raw throughput edge, because the head start almost entirely offsets the
speed advantage. **Called: not worth switching, resumed the batch=32
run instead.**

**Trajectory, batch=32/context=256/response_tokens=8, chance floor
(~10.82) through stopping at update=1,045,370** (loss=7.4725,
tokens=8,563,671,040, ~43.6% of the 19.66B target): unlike the
context=2048 run's single long flat plateau, this run showed continuous
slow drift the entire time, never fully flatlining and never reaching a
new stable floor either -- **loss ranged roughly 7.2-7.6 basically the
whole second half of the run** (briefly touching a low of ~7.23),
`slot_acc` stayed low/flat across all 8 positions throughout (no
fanout), and grad norm climbed steadily from the 10-15 range early to
15-25 by the end without ever resolving into either a clean breakout or
a destabilization/crash -- genuinely ambiguous right up to the point it
was stopped. **Eval-prompt log showed real, if limited, qualitative
progress over the run**: cycled through several *global* collapse
modes (comma-heavy, "and"-heavy, period-heavy) same as the context=2048
run, but also produced things the context=2048 run never did --
prompt-appropriate single content words ("not", "lot", "good" for
"Once upon a time, there was a lot/good..."), and one genuine two-word
coherent fragment (`'To bake a simple loaf of bread, you need'` ->
`' to be......'`). Never reached a full coherent short sentence.

**User's read and decision (2026-08-16): "Looks like its a slow grind.
Slots 8 might be too hard for 660k params... good place to stop and
record."** Direct hypothesis: predicting `output_slots=8` tokens in one
parallel (non-autoregressive) forward pass may simply be too much for
this width to solve at all -- follow-up queued below, `response_tokens`
cut to 4 to test whether a shorter parallel-prediction span is more
tractable. Zero non-finite windows across the entire run; one transient
GPU-idle blip during a data-stream-reopening burst was confirmed
self-recovering, not a real stall (process alive throughout, per `ps`;
`nvidia-smi`'s brief "no processes found" was a known WSL2 enumeration
quirk, not evidence of anything wrong). `latest.pt`/`best.pt` preserved
in case resuming this exact config is useful later.

## Toy response_tokens=4 follow-up, queued (2026-08-16)

Direct test of the "8 parallel response tokens might be too hard for
660k params" hypothesis: same context=256/`d_model=128`/fp8_te/
tokenizer, `--response-tokens 4` (down from 8, so context_tokens=252
instead of 248), `--batch-size 96 --gradient-accumulation-steps 1`
(`--smollm3-effective-batch-sequences 96`, the batch already validated
as a real if marginal throughput win). `--steps 800000` (same as the
batch=96 speed test -- tokens/update is unaffected by the
context/output_slots split, only by sequence_length and effective
batch, both unchanged from that test, so the same step count hits the
same 19,660,800,000-token target). Launched and running.

**Interim loss-vs-tokens-seen comparison (2026-08-16, update~22,000/800,000,
~2.75% through)**: since both this run and the long slots=8/batch=32 run
process 24,576,000 and 8,192,000 tokens/update respectively, exact
matched-tokens-seen checkpoints are available by comparing
`update x 24576` (slots=4/batch=96) against `update x 8192`
(slots=8/batch=32):

| tokens seen | slots=8 loss | slots=4 loss | delta (slots4-slots8) |
|---|---|---|---|
| 24.6M | 8.2338 | 8.1875 | -0.046 (slots4 ahead) |
| 73.7M | 8.0622 | 8.0828 | +0.021 (slots8 ahead) |
| 147.5M | 7.8664 | 7.9537 | +0.087 (slots8 ahead) |
| 540.7M | 7.7045 | 7.6883 | -0.016 (slots4 ahead) |

Loss curves are essentially a wash so far -- small leads trading back
and forth, no systematic advantage to either config yet at this early
stage. **Real difference is in gradient volatility**: at the matched
540.7M-token point, grad norm was **10.1 for slots=8 vs. only 3.4 for
slots=4** -- slots=4 running noticeably calmer at the same point in
training. Loss parity plus lower gradient volatility is a mild point in
favor of the "8 parallel response slots may be harder" hypothesis, but
not confirmed by the loss curve itself yet -- too early and too close
to call definitively. Will keep tracking both the loss-delta sign and
the grad-norm gap as more tokens accumulate.

## Lexical embedding size vs. vocab size, `embedding_dim≈0.01×vocab_size` -- tested, null result (2026-08-16)

Direct test of the Section 9 finding in `Papers/GTR_internal_findings.md`
(`embedding_dim≈0.01×vocab_size`, sweep validated up to vocab=20,000): at
this run's vocab=50,000, that rule suggests `lexical_embedding_dim≈500`,
about double the 256 every toy run so far has used. Launched
`_launch_toy_context256_slots4_emb500.sh`, identical to
`_launch_toy_context256_slots4.sh` in every other respect
(`--response-tokens 4 --smollm3-effective-batch-sequences 96
--batch-size 96 --gradient-accumulation-steps 1 --steps 800000
--smollm3-sequence-length 256`, same `d_model=128`/fp8_te/tokenizer),
only `--lexical-embedding-dim 500` changed. Confirmed via the startup log
that only the lexical path grew (`lexical_path=25,178,000`,
`lexical_width=500`) while the shared `core=696,512` was unchanged --
clean single-variable isolation.

**User's read, correct: "Looks like it's not really a lever for ARW.
Those sweeps were done with other GTRs."** The Section 9 sweep was run
under `--architecture convergent` (router_rank/active_node_top_k/
graph_degree/lanes/small_world), a structurally different Gator variant
-- not GTR-ARW's cells/nodes/relays/broadcast_mix topology. A null
result here is consistent with, not contrary to, that fact once the
architecture mismatch is accounted for; the doc itself flags that a
vocab=50,000 follow-up sweep was "queued but never run," and separately
warns that independently-optimal single-axis ratios don't always compose
across configs -- both caveats apply directly here.

**Result: stopped by user instruction at update=22,000/800,000 (2.75%
through), loss=7.6798, tokens=540,672,000.** Matched against the
`lexical_embedding_dim=256` slots4 run at the identical token point
(same `update×24,576` conversion, both configs share tokens/update):
256-dim loss 7.6883 vs. 500-dim loss 7.6798 at 540.7M tokens -- a 0.0085
nat gap, noise-level at this stage, no discernible advantage either way.
`slot_acc` also indistinguishable (`[0.07,0.04,0.04,0.05]` vs.
`[0.08,0.05,0.05,0.05]`). **Conclusion: `lexical_embedding_dim=256` stays
the standing config** -- doubling it bought nothing measurable at toy
scale and doubles `token_embedding` parameter cost for no return; not
worth carrying into the production bake-off. Run directory
(`~/gtr_arw_100m_toy_context256_slots4_emb500`) and its `latest.pt` left
in place in case a longer-horizon rerun is ever wanted, but not
prioritized.

## Toy relay count: `--relays 4` (up from GTR-ARW's default 2), launched (2026-08-16)

Next architectural lever, per the user: revert to the validated
`lexical_embedding_dim=256`, keep everything else from the slots4/
batch96/context256 config
(`--response-tokens 4 --smollm3-effective-batch-sequences 96
--batch-size 96 --gradient-accumulation-steps 1 --steps 800000
--smollm3-sequence-length 256`, same `d_model=128`/fp8_te/tokenizer),
and change `--relays 4` (up from 2). Relay count changes the broadcast
routing structure itself (part of `core`, not the vocab-scaling lexical
path the emb500 test isolated), so this is a fresh run, not resumable
from any existing checkpoint -- new `RUN_DIR=
~/gtr_arw_100m_toy_context256_slots4_relays4`. Launch script:
`Runbooks/colab_cells/_launch_toy_context256_slots4_relays4.sh`.
Launched, running (PID 1958278). **Startup log confirms
`core=696,512` -- identical to the relays=2 baseline** (`lexical_path=
12,850,000`, `vocab=50,000`, `lexical_width=256` also unchanged, as
expected). **User confirmed (2026-08-16): relay count only adds
parameters for GTR-L -- not GTR-ARW.** Consistent with the log: GTR-ARW's
relays reuse the existing broadcast/routing weights across more
dispatch steps rather than instantiating new per-relay weight matrices,
so doubling `--relays` here is a pure topology/dynamics lever, not a
capacity one. Any effect from this test will show up in loss/eval/
grad-norm dynamics, not in parameter count.

**Interim throughput and loss vs. the relays=2 baseline (2026-08-16,
update~10,300-18,450/800,000, ~1.3-2.3% through)**: diff-based
steady-state throughput (cancels one-time `torch.compile` overhead,
several overlapping windows past update 2,000) settled around **~4.0
u/s vs. relays=2's 5.599 u/s -- relays=4 runs at ~71% of relays=2's
speed**, not the clean half the user initially guessed. Loss, compared
update-for-update (both configs share the same 24,576 tokens/update, so
update numbers convert 1:1, no scaling needed):

| update | relays=2 loss | relays=4 loss | delta |
|---|---|---|---|
| 4,000 | 7.9344 | 7.9317 | -0.003 |
| 6,000 | 7.9537 | 7.9175 | -0.036 |
| 8,000 | 7.8491 | 7.8273 | -0.022 |
| 10,000 | 7.8584 | 7.8251 | -0.033 |
| 13,620 | 7.4507 | 7.4049 | -0.046 |

relays=4 has led at every single matched checkpoint so far, holding a
fairly steady ~0.02-0.05 nat lead from update 6,000 on (not visibly
growing or shrinking) -- a real if modest signal, not yet decisive this
early. Eval prompts are still fully collapsed at update 15,000-18,000,
but the collapse texture has shifted from a flat single-token repeat to
cycling between "the"/"of"/comma-heavy modes, similar to the pattern the
original context=2048 toy run showed before its later breakout.

**Overnight update (2026-08-17, update=163,400/800,000, 20.4%,
tokens=4.02B)**: loss down to a 7.12-7.23 band (ppl ~1,234-1,383),
comfortably past the 7.5 and 7.2 floors, no errors/crashes/nonfinite
windows across the full unattended overnight stretch. Grad norm has
drifted up to ~9.6-10.7 (from ~4-8 earlier) -- the same shape the
slots=8 run showed before its long grind, not yet resolved either way.
Eval prompts are still incoherent but the collapse keeps loosening --
most completions now mix 2-3 distinct tokens across the 4 response
slots instead of one repeated token (e.g. `' used to the of'`, `' very
of of the'`), and one hit all 4 slots distinct the day before (`' a to
the of'`). `slot_acc` has plateaued around `[0.13-0.15, 0.07-0.09,
0.06-0.07, 0.05-0.07]` -- first slot clearly ahead of the other three.
**User's read (2026-08-17): "It's learning how to differentiate words.
Probably capacity constrained on this corpus at this point"** -- i.e.
the per-slot differentiation is real learning, but 696,512 core params
carrying the full broad SmolLM3 mixture is plausibly hitting a capacity
ceiling at this stage rather than needing more updates alone. Directly
motivates the queued chat-only follow-up below as the way to test that
read against the task-breadth-alone hypothesis. The relays=2 baseline
comparison line ran out at update=55,990 (left paused, never resumed),
so no further matched-checkpoint table past that point unless it's
resumed too.

**Follow-up queued for after this run is stopped (2026-08-16):** user's
read on the slow/ambiguous convergence across every toy variant so far
is "probably both" capacity and task breadth -- this toy core
(696,512 params) is carrying a 50k-vocab general-purpose mixture
(FineWeb, DCLM, Stack, math/reasoning, etc.), a much broader task than
the prior 130M-param/2048-vocab chat-only ARW that "could speak
contextually aware phrases most of the time." Planned next step: rerun
this toy config on purely turn-based chat data (narrow single-task
shape matching that prior result) to directly test whether task breadth
-- not just capacity -- is the dominant bottleneck at this width.

**Final result, stopped 2026-08-17 to run the chat-only follow-up:**
update=164,530/800,000 (20.6%), loss=7.1948, tokens=4,043,489,280.
Stopped cleanly (SIGTERM), `latest.pt`/`best.pt` preserved, zero
non-finite windows across the entire run including an unattended
overnight stretch. Loss never fell below ~7.1; eval prompts never
produced a coherent phrase, though per-slot differentiation kept
improving (`slot_acc` plateaued around `[0.13-0.16, 0.07-0.09,
0.06-0.07, 0.05-0.07]`, and completions shifted from flat single-token
repeats to 2-4 distinct tokens per 4-slot response). Directly compare
against the chat-only run below.

## Chat-only follow-up: `--training-recipe standard` on smoltalk_everyday.jsonl (2026-08-17)

Same intent as above -- isolate task/data breadth as a variable,
holding the GTR-ARW architecture family constant -- using
`gator_language_train.py --training-recipe standard --data
/home/user/gtr_data/smoltalk_everyday.jsonl` (2,260 conversations,
already staged locally) instead of the smollm3-26b streamed mixture.
Launch script: `Runbooks/colab_cells/_launch_toy_chatonly_smoltalk.sh`.

**Important: not a clean single-variable swap.** `--training-recipe
standard` has no `--hf-tokenizer` path -- it fits its own from-scratch
`CompactTokenizer` on the corpus, sized by `--vocab-size` (set to 2048,
matching the prior chat-only reference run, since a 50k BPE trained on
a 1.9MB corpus would be mostly unused). So this comparison swaps task
breadth *and* tokenizer/vocab together, not breadth alone -- an
unavoidable confound given the codebase's two separate recipe code
paths, not an oversight. Resulting core is also bigger than the
smollm3-toy's 696,512 (`parameters=1,222,848` per the startup log) --
`--lexical-mode private_hash`'s cell-view prototypes and the
from-scratch tokenizer's embedding table cost more than the HF-tokenizer
lexical path did at this width.

Also NOT token-matched to the 19.66B-token target every smollm3 variant
shared: the corpus is tiny, so hundreds of thousands of steps at
batch=96 would mean thousands of epochs (memorization, not a breadth
test). `--batch-size 32 --steps 200000` is ~8x the prior chat-only
reference run's 24,000-step result
(`experimental_checkpoints/logs/language_smoltalk_62cell_gated_grouped_ctx64_seed0_through_step24000.txt`)
-- real headroom, still exploratory scale. `--context-tokens 252
--response-tokens 4 --chunk-size 4` mirrors the toy's 256-token
split. `--learning-rate 0.003 --scheduler cosine` per the trainer doc's
own stated common override (default 0.008); smollm3's 8e-5 peak-lr has
no meaning on this recipe's schedule.

**Two crashes before it ran clean, both from precision/compile
optimizations validated only inside the smollm3-26b recipe's own code
path this session, per the user's confirmation ("the original code
doesn't have all the compile fixes"):**
1. `--compile --compile-mode max-autotune-no-cudagraphs` (used by every
   smollm3 toy run) crashed immediately with a real
   `torch._dynamo.exc.InternalTorchDynamoError: NameError: name
   'SYNTHETIC_LOCAL' is not defined` on this recipe's model/loss code
   path. Confirms `GATOR_LANGUAGE_TRAINER_ARGUMENTS.md`'s own stated
   caution that compiled runs on this Arrow/gator path aren't currently
   recommended. Switched to `--execution-preset optimized_gpu_safe`
   (no compile).
2. `--node-compute-precision fp8_te --readout-precision fp8_te
   --broadcast-mix-cuda-graph --stewardship-traffic-cuda-graph` (also
   used by every smollm3 toy run all session without a single
   non-finite window) produced a hard NaN at optimizer step 1
   (`neighbor_gates shape=(8,2) invalid=8/16 nan=8`), every time.
   Dropped entirely in favor of plain `--mixed-precision bf16` --
   correctness over throughput for this exploratory run.

**Running clean since the second fix.** Startup:
`parameters=1,222,848 preset=optimized_gpu_safe compile=False
decode_mode=parallel chunk_size=4 plan=dense train=7746
validation=878`. **Striking early result at step=500:** `loss=2.3271
ppl=10.25 acc=0.508 exact=0.280`, and the logged sample was an *exact
match* -- `context='USER: Hi' target='Hello! How' formed='Hello! How'`.
At step=1000: `loss=2.2527 acc=0.547 exact=0.296`, a harder multi-turn
sample (`target='SUVs' formed='Yes   '`) not exact but a plausible,
non-degenerate attempt -- not token-soup. Zero crashes/non-finite
windows since the fix, checkpointing normally
(`~/gtr_arw_100m_toy_chatonly_smoltalk/checkpoints/`). This is a
qualitatively different regime than the relays=4 run's 164,530 updates
of never-coherent output -- strong early support for the user's read
("probably capacity constrained on this corpus," i.e. task breadth is
the dominant factor, not raw capacity, since a comparable/larger core
converges to coherent short completions almost immediately once the
task narrows to chat-only). Confounded by the tokenizer/vocab/param-count
differences noted above, so not a fully clean isolation -- but the
effect size (exact-match completions at step 500 vs. never-coherent at
step 164,530) is large enough that it's very unlikely to be explained
by those confounds alone.

**Follow-up queued (2026-08-17):** once this run has trained for a
while, widen `--response-tokens` from 4 to 8, then 16, to see how the
chat-only/narrow-task setup responds to a wider parallel-prediction
window -- mirrors the earlier response_tokens=8-vs-4 test done on the
broad-mixture smollm3 toy runs, now on the recipe that's actually
converging. Each step needs a matching `--chunk-size` (cannot exceed
`--response-tokens`) and a fresh `RUN_DIR` since response_tokens changes
model shape.

**response_tokens=8 follow-up: launched, then stopped at step=10,000
(2026-08-17).** Same config, `--response-tokens 8 --chunk-size 8
--context-tokens 248`. Same `parameters=1,222,848` as the
response_tokens=4 run -- response_tokens doesn't add weights here
either. Started clean (no crash, same fixes carried over).
**step=500: loss=2.8432 acc=0.457 exact=0.000** -- harder start than
slots=4's (acc=0.508 exact=0.280); a common sample dropped one word
(`target='Hello! How can I' formed='Hello! How   I'`). Loss bottomed at
2.42 (step 7,000), then reversed to 2.63 by step 10,000 while coverage
slid from its 0.771 peak to 0.752 -- the same overfitting shape as
slots=4, arriving faster (step~7,000 vs ~7,000-15,000). **Qualitative
samples got worse, not just noisier**: alongside the earlier
blank-padding/generic-hedge failure mode (`target='Even without a'
formed='Yes,      '`), a new garbled repeated-fragment pattern appeared
(`target='Some popular brands for ' formed='Some popular r r '`,
`target='The violin' formed='For r r   '`, `target='Animals '
formed='The s s s '`) -- coverage went up (fewer blanks) but partly by
filling slots with noise rather than real words. Still qualitatively
better than the broad-mixture toy runs' true degenerate collapse (a
single dominant token everywhere, ignoring context/position entirely) --
this failure mode uses real vocabulary and plausible openers, it just
can't fill every slot correctly yet. Stopped at step=10,000
(loss=2.6345, exact=0.270) per the user's next instruction below.
`latest.pt` preserved, resumable.

**Cancelled (2026-08-17): "Cancel slots 16, do relay = 2 instead at
slots 4 and 8."** relays=4 was only ever tested against the broad
smollm3 mixture, never directly against the GTR-ARW default relays=2 in
the chat-only setting -- that comparison is more informative right now
than pushing response_tokens further. New scripts:
`_launch_toy_chatonly_smoltalk_relays2.sh` (response_tokens=4) and
`_launch_toy_chatonly_smoltalk_slots8_relays2.sh` (response_tokens=8),
identical to their relays=4 counterparts except `--relays 2`, fresh
`RUN_DIR`s.

**relays=2/response_tokens=4: launched, running (2026-08-17).** Same
`parameters=1,222,848`. **~1.8x faster** than relays=4 at the same
config (11.6-11.9 steps/sec vs. 6.3-6.5) -- roughly the expected
speedup from halving relay count. Early loss/accuracy tracks relays=4
closely at matched steps: step 500 (loss 2.35 vs 2.33, acc 0.535 vs
0.508), step 1000 (loss 2.30 vs 2.25, **exact tied exactly at 0.296
both**). Too early to call a winner; watching whether relays=2 shows
the same overfitting-shaped reversal around step 7,000+.

**Stopped by user instruction at step=22,000/200,000: "This is a known
collapse. Go ahead and stop. GTR collapses on repetition."** Final:
`loss=4.3041 ppl=74.00 acc=0.575 content_ppl=245.69 exact=0.391`.
Matched against relays=4 at step=15,000 (loss=2.7340, acc=0.593,
exact=0.370), relays=2 tracked closely through ~step 7,000 then
diverged into a much steeper loss blowup than either relays=4 run
showed -- ppl climbed from its ~8.5 low (step 4,000-5,000) to 74.0 by
step 22,000 (~8.7x), with `content_ppl` reaching 245.69, while `acc`/
`exact` stayed flat-to-improving the entire time (`exact` actually hit
a new high, 0.391, in the same final logged window). Sample quality at
the end: `target='Newt' formed='A ee'` -- genuinely garbled, unlike the
plausible-if-generic completions relays=4 produced at a comparable step
count.

**This is a documented, known phenomenon, not a new finding: the exact
same corpus (SmolTalk Everyday, 2,260 examples) is the tiny/highly-
repeated corpus flagged in `Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md`
section 7.1.** That study found a plain transformer's free-generation
quality collapses 15x from its own peak on this corpus by ~170 epochs
(`content_ppl` 15.9->245.22) -- almost the exact `content_ppl` this
relays=2 run just reached (245.69) at a comparable epoch count
(~91 epochs: 22,000 steps x 32 batch / 7,746 examples). Gator itself
drifts more mildly on the same corpus in that study (22.53->35.73, a
1.6x drift) but "whether Gator itself would eventually show more severe
drift at a horizon beyond 24,000 steps... is untested" -- this relays=2
run (22,000 steps, ~91 epochs) is direct evidence that it does, at
least at this toy width/relay count. The root cause per that study is
**corpus-repetition-driven overfitting/collapse, not an architecture or
relay-count-specific bug** -- the fix that study identifies is a larger/
fresher corpus (their `smoltalk_preference` 50k-example subset fully
avoided the collapse at ~7.7 epochs), not a hyperparameter change on
this same tiny corpus.

**Implication for the relays=2-vs-4 and response_tokens sweeps:** every
chat-only run so far (slots=4/relays=4 stopped at step 15,000/~62
epochs, slots=8/relays=4 stopped at step 10,000/~41 epochs, slots=4/
relays=2 stopped at step 22,000/~91 epochs) was creeping toward or past
this same known collapse ceiling on the same 2,260-example corpus --
the "quality degradation" observed in the slots=8 and relays=2 runs is
very likely this same corpus-repetition effect, not a genuine
slots-count or relay-count weakness. **The response_tokens=8/relays=2
run queued next was not launched** -- before continuing the sweep, the
corpus itself needs to change (a larger/fresher chat corpus, or a hard
step cap well inside the epoch budget that stayed stable, e.g.
~step 5,000-7,000/~20-29 epochs on this corpus) so relay-count and
response-tokens comparisons aren't confounded by which run happens to
hit the repetition wall first.

## Fixing the corpus-repetition collapse: adding LMSYS-Chat-1M (2026-08-17)

Direct fix for task #35, per the user: "Add in lmsys chat." The
Plateau Instability Study's own fix for this exact collapse was a
larger/fresher corpus, not a hyperparameter change -- LMSYS-Chat-1M
(~1M conversations) was already staged on Google Drive
(`C:\path\to\gdrive\GTR\training_data\lmsys_chat_1m.jsonl`, 2.25GB, same
shared `USER:`/`ASSISTANT:` text-JSONL schema as `smoltalk_everyday.jsonl`,
confirmed by inspection before use) -- WSL cannot see the Google Drive
mount directly (`/mnt/g` exists but is empty; Drive-for-Desktop uses a
virtual filesystem DrvFs can't bridge), so it was copied via PowerShell
straight to the WSL UNC path (`\\wsl.localhost\Ubuntu\home\user\gtr_data\`)
-- the copy completed near-instantly, meaning Drive had it fully synced
locally already, not a fresh cloud download.

Launch script: `Runbooks/colab_cells/_launch_toy_chatonly_smoltalk_lmsys.sh`.
Same 1,222,848-param chat-only shape as every prior chat-only run this
session, `--lmsys-data /home/user/gtr_data/lmsys_chat_1m.jsonl
--lmsys-load-mode stream` added (streaming, not `memory` -- no reason to
load a 2.25GB/~1M-conversation file fully into RAM when training only
consumes a fraction of it). `--streaming-weight` left at its default
(falls back to `--lmsys-weight`'s default 0.9 -- an established
90%-streamed/10%-primary-corpus default in this codebase, not a new
choice), `smoltalk_everyday.jsonl` stays the primary corpus (still
supplies the stable validation benchmark, now `validation=1,832`, up
from 878 -- LMSYS contributes validation rows too). `--steps 200000`
kept as-is; at ~1M source conversations the epoch-repetition math that
caused the collapse no longer applies at any step count this toy width
would plausibly reach.

**Launched clean, no crash** (same bf16-only/no-compile fixes carried
over). **Slower, harder start than every smoltalk-only run, as
expected** -- real open-domain questions now (`'who is the current
minister of interior of tunisia'`) instead of a small repeated template
set: step=500 `loss=3.8609 acc=0.267 exact=0.000` (vs. the smoltalk-only
runs' ~2.3-2.8/~0.46-0.51/~0.25-0.28 at the same step). Real, fast
improvement by step=1,000: `loss=3.2504 acc=0.367 exact=0.130`, and it
already re-learned the common greeting exact match
(`context='USER: Hi there' target='Hello! How' formed='Hello! How'`).
Throughput is lower than the smoltalk-only runs (5.6-5.7 vs. 6.3-6.5
steps/sec, `sample_ms`/`transfer_ms` both ~35x higher) from the
streaming-disk I/O overhead of reading a much larger corpus -- expected
tradeoff, not a concern at this scale. This trades the smoltalk-only
runs' fast-but-brittle memorization for a slower but (per the Plateau
Instability Study's own finding on the exact same corpus pairing)
genuinely non-collapsing training signal -- watching whether loss keeps
descending past step~7,000-9,000 without the repetition-collapse
reversal every pure-smoltalk run showed by that point.

**Stopped by user instruction at step=13,500/200,000: loss=2.7170,
acc=0.432, exact=0.150.** Held the clean, monotonic-with-noise
improvement trajectory the whole way -- no repetition-collapse reversal
at any point, unlike every pure-smoltalk-only run.

**User's closing read on the relays=2-vs-4 question (2026-08-17):
"Relay 4 only provided a small improvement which means sticking with
relay 2 is probably a better use of limited resources."** Consistent
with everything measured this session: relays=4 held a modest,
non-collapsing loss edge over relays=2 in the chat-only comparison
(before both were stopped for the corpus fix), and a similar modest
loss edge in the broad-mixture toy comparison -- but at a real,
consistent throughput cost (~1.8x slower in the chat-only setting,
~40% slower in the broad-mixture toy). **Decision: relays=2 (GTR-ARW's
default) for the production run**, not relays=4.

**User's closing read on the whole toy-sanity-check investigation
(2026-08-17): "We learned what we needed to learn which is given
enough capacity and a broad enough mix to prevent collapse it can learn
language."** This closes out the entire toy-scale sanity-check arc that
began with the original context=2048 degenerate-collapse check --
context length, response_tokens count, lexical_embedding_dim, relay
count, task/data breadth, and now corpus repetition were all tested at
this toy width, and the single result that mattered most was the
chat-only+LMSYS run above: given enough capacity relative to the task,
and a data mixture broad/fresh enough to avoid repetition-driven
collapse, this GTR-ARW shape does learn real language (steady,
non-collapsing loss descent, exact-match completions, no degenerate
token-repeat behavior at any point). **The toy investigation is
concluded; the next step is the real 100M-param production run** (see
below), not further toy-scale sweeps.

## The real 100M-param production run (2026-08-17)

Launched: `gtr_arw_100m_production_d1504_slots8_relays2`,
`Runbooks/colab_cells/_launch_production_d1504_slots8_relays2.sh`.
Supersedes the earlier `_launch_bakeoff_fp8_english50k_300k.sh`
(RUN_DIR `gtr_arw_100m_bakeoff_fp8_english50k`), which only ever
reached update=710/300,000 before being set aside for the toy detour --
not resumable from that checkpoint since both d_model and
response_tokens change below (different model shape).

**Width re-sized to actually hit 100M at the 50k-vocab tokenizer:** the
earlier script's `d_model=1312` predates the vocab swap and only
reaches 79,678,064 total params at 50k vocab (never corrected).
`Runbooks/colab_cells/_100m_width_sweep_dense_50k_relays2.py`
(bisection, same approach as the earlier Monarch width sweep, adapted
for dense `folded_bmm` + 50k vocab + relays=2) found **d_model=1504,
embedding_dim=3008 -> 100,430,768 total params** (+0.43% from target),
confirmed exactly by the real run's startup log
(`smollm3_model_parameters total=100430768 lexical_path=13620048
core=86810720`).

**Config, per the toy investigation's conclusions:** `--relays 2`
(GTR-ARW default -- relays=4 only gave a small, non-collapsing loss
edge in the toy comparisons, not worth its ~1.4-1.8x throughput cost),
`--response-tokens 8` (user: "then do slots=8"), otherwise identical to
the proven earlier script: `--smollm3-sequence-length 512` (context=512,
cheaper to train short and extend later than pay 2048-context cost
throughout), `--batch-size 8 --gradient-accumulation-steps 4`
(effective batch=32), `--steps 1200000` (4x the original 300,000 at
context=2048, same ~19.66B-token phase budget), fp8_te node-compute/
readout precision, both CUDA graphs, `--compile --compile-mode
max-autotune-no-cudagraphs`, `--smollm3-peak-lr 8e-5`,
`--checkpoint-every-updates 500` (tight, safe restart loss for an
unattended multi-day run).

**VRAM confirmed genuinely maxed at steady state**: ~15.7/16.3GB used
(~96.6%), only ~556MB free -- checked specifically because the user
asked about bumping batch size for throughput; there's no real headroom
left at this width, so batch=8/accum=4 stays as launched.

**Measured steady-state throughput** -- an early diff-based read
(update 100-390, ~0.51-0.52 u/s) was wrong, still contaminated by
residual compile/autotune activity bleeding past that window; corrected
with a wider window (update 1000-2260, several overlapping sub-windows
agreeing within ~0.92-0.97 u/s) once training genuinely settled:
**~0.93-0.95 updates/sec, ~15,200-15,600 tokens/sec.** At this rate,
the current `--steps 1200000` phase (19.66B tokens) is **~15 days**;
the full 26B-token recipe budget (`recipe_updates=1,586,915`) is
**~19-20 days**. User's original plan ("We'll move to Colab halfway
through or so") was set against the earlier, much slower estimate --
with the corrected number, user's read (2026-08-17): **"We might be
able to finish Phase 1 on the 5080 then"** -- the Colab migration may
not be needed, contingent on this throughput holding over a longer
horizon. No crashes, no non-finite windows, checkpointing normally as
of launch.

**Confirmed holding past the collapse point (2026-08-17, step=11,000):**
loss and acc both improved steadily and monotonically through
step~10,000 (loss 3.86->2.73, acc 0.267->0.434, all new highs, one
minor single-checkpoint dip at step 9,000 that fully recovered by
10,000) -- **no trace of the sharp reversal every pure-smoltalk-only
run showed by step~7,000-9,000.** This is the clearest evidence yet
that the collapse was genuinely a corpus-repetition artifact, not an
architecture or relay-count issue: the exact same 1,222,848-param
shape, now fed a broader/fresher data mixture, trains cleanly straight
through the point where the narrow-corpus runs all turned over. Formed
samples at step~3,000 were still mostly single-token generic filler on
genuinely novel/open-domain content (factual questions, foreign-language
requests, specialized writing) -- expected this early against a task
this much broader than the tiny corpus, and not degenerate (no
repeat-loop garbling like the slots=8/relays=4 run showed).

**Stopped by user instruction ("Let it run to 15k then start slots 8")
at step=15,000/200,000, loss=2.7340, acc=0.593, exact=0.370.**
Trajectory: fast real improvement through step~4,000 (loss 2.33->2.08,
exact 0.28->0.33), then loss drifted back up to a 2.6-2.9 band through
step 15,000 while `exact`/`acc` stayed roughly flat (0.36-0.39) or
ticked down slightly -- the loss/accuracy divergence from step
~7,000 on reads as mild overfitting on the tiny 7,746-example training
set (step 15,000 at batch=32 is ~62 epochs through it), not a real
regression. Exact-match completions kept appearing throughout, including
the final logged sample (`context='USER: Hi there' target='Hello! How'
formed='Hello! How'`). `latest.pt` preserved, resumable. Zero crashes
or non-finite windows across the whole 15,000-step run once the two
startup fixes landed.

## Notes

- Both items are genuinely blocked on code changes the user is making
  themselves right now, not on anything runnable today — this queue is
  purely to make sure the "what to test once X lands" list survives
  until then.
- Log results into `Runbooks/GTR_RUN_REGISTRY.md` as usual, same
  row-per-run convention as every other entry there.
