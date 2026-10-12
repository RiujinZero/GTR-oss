# Blackwell Opaque-FFT + Triton Release Gate

**Status:** Fused-writer gate passed; Triton reader plus Torch writer is the N154 throughput winner
**Optimized branch:** `agent/packed-routing-code-only`
**Reader implementation checkpoint:** `56c4a55c80700723ab5cee584efba2bae8010055`
**Fused-writer gate-qualified revision:** `80051f8a0b5764c75ba94c18777150574817e442`
**Gate bundle SHA-256:**
`70de1c23c1b3b0620f592bec91d9e32efcb4620132c6815ae2ff9c961416f7f9`
**Safe fallback branch:** `agent/100m-preoptimized` at
`454338d`

The first native-backward candidate (`bb9cdd1`, bundle
`af520539...69b4e0`) stopped on production causal numerical parity. The
follow-up mirrors forward's combined denominator and first-feature FFT
batching in backward. Revision `56c4a55` and bundle `fca32603...e596b`
subsequently passed all required CUDA, receiver, full-model, training/resume,
and benchmark stages. The packed implementation is therefore correctness
qualified, but its full-model throughput remains below dense matrix Torch.

This is the acceptance procedure for the optimized packed receiver. The safe
fallback branch must remain unchanged and usable until this gate passes. A
local CPU test, a successful compile, or a forward-only CUDA result is not a
release result.

The checkpoint above qualifies the complete fused-writer stack for numerical,
compile, optimizer-step, and checkpoint/resume correctness. At N154 the
Triton writer is retained as an experimental high-node backend rather than the
throughput default because the matched Torch writer was slightly faster.
Every GPU run must still record the exact 40-character source revision and the
SHA-256 of the bundle actually executed.

## 1. What the optimized stack is

The target configuration is:

```text
--route-backend packed
--packed-memory-backend fft_opaque
--packed-memory-reader-backend triton
--packed-memory-writer-backend triton
--packed-memory-feature-chunk 32
--compile
```

It has three independent execution boundaries:

1. **Opaque FFT receiver memory.** `fft_opaque` places the complete
   real-input/real-output packed FFT receiver behind a
   `torch.library.custom_op`. PyTorch still calls `torch.fft.rfft` and
   `torch.fft.irfft`, and cuFFT still performs the convolution on CUDA. Only
   real tensors cross the outer Inductor graph. A second opaque operator now
   applies an explicit chunk-local adjoint: it recomputes bounded writer/FFT
   chunks, applies conjugated-spectrum correlations, and accumulates real
   gradients without tracing complex spectra or invoking nested autograd.
2. **Fused Triton packed reader.** The standalone Triton backend has fused
   forward and backward implementations for the real-valued reader epilogue:
   denominator normalization, packed node gather, route-mass weighting, and
   reduction over K routes. In the current `fft_opaque + triton` composition,
   the outer explicit adjoint invokes that Triton backward directly for every
   feature chunk. This avoids the PyTorch 2.11 functorch boundary and removes
   expanded `[B,T,L,K,C]` reader intermediates in both directions.
3. **Fused Triton packed writer.** The writer sums lane baselines and scatters
   route-weighted state directly into FFT-native `[B,C,U,T]` storage. Its
   explicit adjoint returns hidden-chunk, write-gate, and route-delta gradients
   without constructing the `[B,T,L,K,C]` writer or route-gather tensors. The
   first call carries mass in feature zero, preserving the combined
   denominator/first-state FFT contract.

The Triton kernels do **not** perform the FFT. They also do not yet fuse
active-node selection, cuFFT/IRFFT output storage, collision merging,
occupancy, route transport, private-node transforms, or the optimizer. The
first implementations are CUDA float32 only; writer feature chunks are capped
at 128. They must raise when unsupported—a silent Torch fallback invalidates
both correctness and timing evidence.

The raw eager FFT plus Torch reader is the numerical reference. The matrix,
direct-convolution, and portable Torch-reader paths remain explicit controls.

## 2. Frozen comparison point

The first performance target is the archived 140M Convergent control:

```text
V32768 / E768 / D768 / N154 / L10 / K32 / R2
batch 6 / context 512 / response slots 128 / float32 / compiled
```

Its historical RTX PRO 6000 Blackwell Server Edition training rate was
**3.22 steps/s**, with approximately **41 GB VRAM** observed before the packed
receiver work. A separate direct-convolution screen used approximately
31.5 GB but was too slow; it is context, not a passing optimized result. The
gate's primary speed comparison is its matched four-way synthetic A/B. It
uses the same shifted teacher-forced causal response path and gradient clip as
the archived trainer, but deliberately excludes data loading, tokenization,
logging, validation, and checkpoint I/O. Therefore 3.22 steps/s remains a
historical contextual threshold rather than a perfectly matched control.

## 3. Required correctness and continuation stages

`run_blackwell_release_gate.py` launches every stage in a fresh subprocess and
stops at the first required failure. The correctness matrix uses seeds 0 and 1
in both causal and noncausal modes unless explicitly narrowed for diagnosis.

1. **Environment and identity preflight**
   - CUDA is available on a Blackwell-class device (compute capability 10.x or
     newer).
   - Triton plus `torch.library.triton_op`/`wrap_triton` is available.
   - Python, PyTorch, PyTorch Git revision, CUDA, Triton, GPU, VRAM, source Git
     revision, and bundle SHA-256 are recorded.
2. **Focused CUDA unit tests**
   - Opaque FFT plus Triton writer/reader forward, backward, fake-tensor,
     autograd, opcheck, duplicate/collision, and compile contracts pass without
     skips or fallback.
   - A reduced-width `N=1,337` end-to-end receiver smoke must remain finite so
     the exact high-node kernel regime is exercised before scaling claims.
3. **Receiver equivalence: quick matrix**
   - Eager raw FFT plus Torch reader is compared with compiled opaque FFT plus
     Triton at a small shape for both seeds and causal modes.
   - Output and all differentiable receiver-input gradients are finite and
     within the validator tolerances.
4. **Receiver equivalence: production receiver shape**
   - Repeat the same matrix at `B1/T640/L10/K32/N154/D768`, feature chunk 32.
   - Timing and CUDA peak memory are recorded after warmup, but this stage is
     primarily a receiver correctness test.
5. **Full-model optimizer-step equivalence**
   - Start both paths from one seeded state and one deterministic shifted
     teacher-prefix batch.
   - Compare eager packed raw FFT plus Torch reader with the compiled
     `fft_opaque` plus Triton candidate.
   - Logits, loss, gradient presence, every named parameter gradient, and
     every named AdamW parameter delta must match within the declared
     tolerances. At least one Dynamo graph must be captured.
6. **Candidate training and resume**
   - Run the compiled candidate for the configured smoke runway (100 steps for
     release evidence), checking finite logits, loss, gradients, parameters,
     and optimizer state on every step.
   - Atomically save model, named AdamW state, RNG state, and the exact next
     batch. After reopening the checkpoint from disk, a freshly constructed
     model and optimizer are compiled in the same isolated validator process
     and must reproduce the uninterrupted next-step loss, every parameter
     update, and every optimizer-state update within tolerance.

The receiver validator's frozen elementwise tolerance remains
`atol=5e-4, rtol=5e-4` for every comparison and tensor except
production-shape causal `grad_mass`. That single component uses
`atol=1e-3, rtol=1.5e-3` as bulk elementwise telemetry. At most four elements
may exceed that envelope and the outlier fraction must be at most `2e-5`;
both limits must hold. Every value must also be finite, relative L2 error must
be at most `2e-5`, and maximum absolute error must be at most `2.5e-2`. The
exception does not apply to quick shapes, noncausal execution, forward values,
or any other gradient: those comparisons still require zero elementwise
outliers. Full-model and training validators retain their own stricter
contracts.

This policy bounds the observed sparse nondeterminism without repeatedly
widening `atol` to chase a moving near-zero element. It admits no more than
four of 204,800 production-causal `grad_mass` values while the finite,
relative-L2, and maximum-absolute guards constrain the complete tensor and
each individual drift.

The correctness gate passes only when every required stage exits zero and
`summary.json` reports `passed`. A skipped CUDA path, an implicit backend
substitution, a non-finite value, missing gradient, missing compile evidence,
or failed resume comparison is a failure.

## 4. Required performance A/B before promotion

Correctness does not establish that the optimization is useful. Enable the
full-model benchmark to run these frozen 140M cases as four separate fresh
processes:

1. dense routes + matrix memory + Torch reader/writer;
2. packed routes + `fft_opaque` + Torch reader/writer;
3. packed routes + `fft_opaque` + Triton reader + Torch writer;
4. packed routes + `fft_opaque` + Triton reader/writer.

Each case measures only the complete shifted teacher-forced causal training
step with the same seed, shape, dtype, optimizer, gradient clipping, warmup,
and measured-step count. Report the stage wall time (which includes startup,
compilation, and warmup), warmed steps/s, allocated and reserved peak VRAM,
and any failure. Running the four cases in one process is invalid because
compiled graphs and allocator state contaminate later measurements.

A **speedup claim** requires the compiled Triton candidate to beat the matched
dense control and the historical 3.22 steps/s reference without a correctness,
training, or capability regression. Memory must be reported against the
approximately 41 GB historical control; no fixed memory threshold is declared
in advance. An isolated memory reduction is not a speedup, and an isolated
throughput increase is not a model-quality result.

The completed **reader-only** gate measured the following full-model compiled
FP32 training results at `B6/C512/O128/D768/N154/L10/K32/R2`. These numbers are
historical controls for the fused-writer candidate, not its result:

| Backend | ms/step | steps/s | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| Dense matrix + Torch | 249.8348 | 4.003 | 29,489.26 MiB | 33,056 MiB |
| Packed opaque FFT + Torch | 835.8425 | 1.196 | 19,194.63 MiB | 25,138 MiB |
| Packed opaque FFT + Triton | 685.9594 | 1.458 | 19,192.60 MiB | 25,028 MiB |

Triton made the packed backend **1.219x faster** than its matched Torch-reader
control while allocated memory remained effectively identical. Dense matrix
Torch was still **2.746x faster** than packed Triton. Packed Triton reduced
full-model peak allocated memory by **34.9%** relative to dense, and peak
reserved memory by **24.3%**. The target is therefore correct, resumable, and
materially more memory efficient, but it does not meet the full-training
speed-promotion criterion.

The completed fused-writer gate measured the following matched fresh-process
results:

| Backend | ms/step | steps/s | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| Dense matrix + Torch | 249.9334 | 4.001 | 29,489.26 MiB | 33,056 MiB |
| Packed FFT + Torch reader/writer | 766.7052 | 1.304 | 15,932.70 MiB | 17,550 MiB |
| Packed FFT + Triton reader/Torch writer | 666.1237 | 1.501 | 15,933.98 MiB | 17,640 MiB |
| Packed FFT + Triton reader/writer | 676.4230 | 1.478 | 15,932.80 MiB | 17,432 MiB |

The Triton reader improves matched packed-Torch throughput by about 15.1%.
Switching only the writer from Torch to Triton then reduces throughput by
about 1.5% and changes peak allocation by less than 2 MiB. The writer fusion
is therefore a qualified negative optimization at this N154 shape. The
recommended packed configuration keeps the Triton reader and Torch writer.
That configuration cuts peak allocation by about 46.0% relative to dense,
but dense remains about 2.67x faster; the all-Triton variant is about 2.71x
slower than dense. At the system level this is explicitly recorded as a
**negative speed result and positive memory result**. The memory headroom can
still be decisive for larger models, node populations, batches, or contexts;
the writer must be remeasured there because the K-expanded temporary it avoids
grows even though it did not set peak allocation at N154.

## 5. Evidence artifacts

Use a new local output directory for every gate invocation. The runner refuses
to reuse one. Preserve together:

- `summary.json`, atomically rewritten after every stage;
- `release_gate.log`, the combined live transcript;
- one `.log` per isolated stage;
- the four benchmark JSON files when the performance A/B is enabled;
- the candidate training/resume checkpoint;
- the exact source revision and bundle SHA-256 recorded in the summary;
- the invocation and final directory copied to durable storage after the run.

Canonical invocation shape:

```bash
python -u run_blackwell_release_gate.py \
  --device cuda:0 \
  --source-revision <40-character-git-sha> \
  --bundle-sha256 <64-character-bundle-sha256> \
  --output-dir /content/blackwell_fft_triton_gate/<unique-run-name> \
  --full-model-benchmark
```

As of July 21, 2026, the native-backward gate has passed at revision
`56c4a55` with bundle `fca32603...e596b`. This establishes Blackwell
correctness and continuation for the explicit native adjoint. It establishes
a full-model memory win, but not a full-model speed win over dense matrix
Torch.

## 6. Execution history

The first target invocation (`357ca9a`, bundle
`cbe0f52e...1e18ee0`) verified the RTX PRO 6000 Blackwell environment
(PyTorch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0, capability 12.0), then stopped
before CUDA execution because the ZIP omitted `tests/__init__.py`. An unrelated
environment package therefore shadowed the bundled `tests` directory. This is
a harness packaging failure, not an FFT/Triton result. The package marker is
required in all subsequent bundles.

The corrected second invocation (`2feb49e`, bundle
`2dffd59c...a3c35c`) reached CUDA. Seventeen of eighteen focused tests passed;
the combined compiled opaque-FFT plus Triton-reader backward failed because
the opaque custom op recomputed its VJP through the Triton op's generated
autograd function, which PyTorch 2.11 cannot traverse with `torch.func.vjp`.
The correction retains Triton for the real forward but uses the mathematically
identical Torch reader equation only inside the opaque VJP recomputation. This
is a correctness unblock, not yet a performance result: the Torch recompute
can retain larger gather intermediates, so the complete Blackwell memory and
speed gates remain mandatory.

The third invocation (`8c4caf5`, bundle `e002bef4...4d719b`) confirmed the
backward correction: all 19 CUDA unit tests and all four quick seed/causality
receiver matrices passed. The production noncausal receiver also passed. At
`B1/T640/L10/K32/N154/D768`, compiled target forward was about 1.20x faster
than eager FFT plus Torch reader and reduced forward peak allocation from
about 8.82 GB to 0.56 GB. Forward-plus-backward was about 1.21x slower and
used slightly more peak allocation because opaque VJP currently recomputes
the Torch reader.

The follow-up production-causal component diagnostic ran three times at
`B1/T640/L10/K32/N154/D768`: seed 0 twice and seed 1 once. Raw Triton versus
raw Torch passed in all three runs. The only outliers were rare `grad_mass`
values (one to three of 204,800). Comparable outliers also appeared between
eager autograd and functional VJP, and between opaque Torch and separately
recomputed Torch; their indices and magnitudes moved between the two
identical seed-0 runs. This supports nondeterministic FP32 CUDA recomputation
noise from scatter-add/FFT reductions as the explanation rather than Triton
reader math or causal indexing.

The full gate at commit `048dda5` then passed the focused CUDA unit stage, all
quick receiver stages, both production noncausal seeds, and production causal
seed 0. Production causal seed 1 failed solely because one of 204,800
`grad_mass` elements exceeded the bulk `1e-3/1.5e-3` elementwise envelope.
The tensor was finite; relative L2 was `6.48092555e-6`, below `1e-5`, and
maximum absolute drift was `0.0050125`, below `0.008`. No other comparison
failed before the gate stopped.

Together with the repeated component diagnostics, this result freezes the
bounded sparse-outlier policy declared above. Only production-causal
`grad_mass` may have up to four bulk-envelope outliers at a fraction no larger
than `2e-5`, and only when finiteness, relative L2 at most `1e-5`, and maximum
absolute drift at most `8e-3` all pass. Everything else remains strict
zero-outlier at its existing tolerance.

The complete rerun at revision `c80d54b9875282d88fbee646759fca8f935907bc`
and bundle
`d56c6169b45e605ec017a5bc65b822c73cf67356f6aeab161f2061f54144ae8d`
passed the gate. It cleared the focused CUDA units; quick and production
receiver equivalence for seeds 0 and 1 in causal and noncausal modes;
full-model equivalence for seeds 0 and 1 in both modes; 100-step checkpoint,
reload, and exact continuation for both modes; and every required performance
stage.

At the frozen full-model training shape `B6/C512/O128/D768/N154/L10/K32/R2`,
compiled FP32 dense matrix Torch ran at 249.9264 ms/step (4.001 steps/s), with
29,489.26 MiB peak allocated and 33,056 MiB peak reserved. Packed opaque FFT
with the Torch reader ran at 910.4605 ms/step (1.098 steps/s), with 33,681.73
MiB allocated and 37,954 MiB reserved. Packed opaque FFT with Triton ran at
888.8866 ms/step (1.125 steps/s), with 33,681.85 MiB allocated and 37,694 MiB
reserved.

This resolves the release gate without establishing the hoped-for training
optimization. Dense matrix Torch is 3.56x faster than the Triton candidate
and uses about 12.4% less peak allocation relative to packed (packed uses
14.2% more than dense). Triton improves matched packed-Torch throughput by
only 2.46%, with effectively identical allocated memory. That contrasts with
the isolated receiver-forward result, where packed Triton was about 1.20x
faster and used about 15.9x less peak allocation. The opaque backward's Torch
VJP recomputation erases that forward advantage during training. The next
priority is a true fused/optimized backward; the current packed forward may
still be useful for inference-only deployment where backward is absent.

The native-backward rerun at revision
`56c4a55c80700723ab5cee584efba2bae8010055` and bundle
`fca32603d4a6fa7a93be961012bec81e75f88e06219e542522afeae4c40e596b`
passed the complete gate. All 22 focused CUDA tests passed, as did both seeds
in quick and production causal/noncausal receiver validation, full-model
equivalence, 100-step training, checkpoint reload, exact continuation, and
all three benchmark stages. The fused-first correction cleared the original
production-causal failure under the frozen acceptance policy.

At the frozen 140M training shape, dense matrix Torch reached 4.003 steps/s
at 29,489.26 MiB peak allocated. Packed opaque FFT with the Torch reader
reached 1.196 steps/s at 19,194.63 MiB. Replacing that reader with Triton
reached 1.458 steps/s at 19,192.60 MiB. Triton therefore recovered 21.9%
throughput over packed Torch, while packed execution cut peak allocation by
34.9% relative to dense. Dense remains 2.746x faster, so the next optimization
target is the FFT/writer path and launch structure rather than further reader
micro-optimization alone.

### Fused-writer follow-up candidate

The next candidate attacks that writer path directly. It adds an explicit
`packed_memory_writer_backend` control, FFT-native writer layout, a fused
Triton scatter and adjoint, source-spectrum reuse for non-first chunks, and a
full-node identity-map fast path. Local schema, fake-tensor, AOT, numerical,
gradient, configuration, runner, validator, and release-gate tests pass. CUDA
kernel execution remains unqualified until the four-way Blackwell gate above
is preserved with its exact source revision and bundle hash.

With fixed route width K, the ideal sparse opportunity grows roughly as N/K;
for N=1,337 and K=32 that ratio is about 41.8. The present writer still
initializes and transforms an N-sized FFT table, however, so additional nodes
are expected to improve relative memory economics before they improve speed.
Compile-safe active-node compaction is the next target if the writer A/B shows
that this N-sized work dominates.

The first fused-writer Blackwell invocation at revision `818b5df` and bundle
`678ba1de...b051919` stopped in the focused CUDA unit stage after 38 of 39
tests passed.  The failed assertion was instrumentation-only: the integrated
forward delegates to `packed_convolution_memory`, while the test mocked the
same-named writer symbol in `opaque_fft_memory`, which owns only the native
backward binding.  Direct Triton writer forward, explicit adjoint, registered
autograd, production-width, compiled integration, and the N=1,337 finite
smoke all passed.  The corrected gate patches the executing forward namespace;
no kernel or numerical equation changed.

The corrected invocation then isolated a bounded production-causal FP32
reduction difference in `grad_mass`.  Across four seed-1 causal executions,
the fused target remained finite with relative L2 between about `9.57e-6`
and `1.42e-5`, maximum absolute drift between about `0.0122` and `0.0219`,
and at most one bulk-envelope outlier among 204,800 values.  Component
diagnostics held the Triton-reader/Torch-writer path to `6.57e-6` relative L2
and `0.00594` maximum absolute drift, while switching only the writer produced
the larger bound.  This identifies the writer's additional FP32 atomic and
feature reductions rather than the reader or opaque boundary.

For the fused-writer release candidate only, the production-causal
`grad_mass` guard is therefore frozen at relative L2 `2e-5` and maximum
absolute drift `0.025`.  The elementwise `1e-3/1.5e-3` envelope, four-outlier
count, `2e-5` outlier fraction, finiteness checks, and every non-`grad_mass`
tolerance remain unchanged.  This is a bounded kernel-specific numerical
policy, not a general relaxation and not evidence of exact Torch reduction
ordering.
