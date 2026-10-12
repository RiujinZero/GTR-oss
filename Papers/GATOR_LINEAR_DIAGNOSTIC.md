# GTR-L bring-up diagnostic

Working notes for review, 2026-07-29, branch `gator_sparse`.
Implementation: `gator_linear/`, `gator_linear_prefix_train.py`,
`tests/test_gator_linear.py`. Design rationale: `GATOR_LINEAR_DESIGN.md`.

## Headline

Two things are settled, one is newly resolved, and one is still open.

1. **Linear scaling works, decisively.** This was the point of the branch
   and it is not in doubt.
2. **The architecture has capacity.** It fits a fixed batch to 100%.
3. **The prefix plateau was a learning rate, not a defect** -- newly
   confirmed and the most important finding here.
4. **Full prefix mastery (>=99%) is still unproven.**

## 1. Linear scaling: verified

Peak CUDA memory, batch 4, width 128, causal, four relays, measured on an
L4 against `gator_convergent`'s dense backend measured earlier the same day
on a 95 GB card:

| ctx | GTR-L | per doubling | convergent | per doubling |
|---|---|---|---|---|
| 512 | 1.44 GB | -- | 5.76 GB | -- |
| 1024 | 2.85 GB | **2.00x** | 19.64 GB | 3.41x |
| 2048 | 5.70 GB | **2.00x** | 71.96 GB | 3.66x |
| 4096 | 11.38 GB | **2.00x** | OOM at 3072 | -- |

Exactly linear against a clearly super-linear baseline. GTR-L runs **4096
context in 11.4 GB on a 22.5 GB L4**, where convergent needed 72 GB at 2048
and OOM'd at 3072 on a card four times larger. Roughly a 12x advantage at
2048, widening with length. The 620-token production ceiling is gone.

Compute scales too (CPU, fwd+bwd, prototype config):

| axis | change | cost | linear | quadratic |
|---|---|---|---|---|
| nodes 40 -> 320 | 8x | **8.39x** | 8x | 64x |
| length 16 -> 256 | 16x | **17.29x** | 16x | 256x |
| streams 5 -> 40 | 8x | **3.76x** | 8x | 64x |

Node-count memory is linear as well: 40/80/160 nodes -> 715/1330/2575 MB
(1.86x, 1.94x per doubling).

**Caveat worth stating plainly:** at *prefix* scale (L=15) GTR-L costs
715 MB training at batch 128 versus the reference's 40 MB -- about 18x
worse. Every slot carries an `[S,N]` route distribution and an `[N,C,Dm]`
mailbox that the reference simply does not have. The design is linear with
a large constant, so it loses badly on short sequences and wins hugely on
long ones. The crossover is well below production context length.

## 2. Capacity: verified

An overfit probe on 16 fixed examples reaches **loss 0.0002, accuracy
1.000, by step 300**. Evidence reaches the readout, the readout can express
the target, gradients are healthy. Whatever was wrong was never structural.

## 3. The plateau was the learning rate

This is the finding I most want reviewed, because it invalidates several
hours of my own reasoning.

**Symptom.** On the real task (fresh batches), loss pinned at exactly
`ln(12) = 2.4849`, accuracy at chance (~0.083), for 40k steps. Gradient norm
*decaying* to 0.02 -- converged, not slow. Alongside it: occupancy entropy
drifting back toward uniform (3.54 of a 3.689 max), the learnable bus gate
climbing 0.12 -> 0.58, and stream fusion weights collapsing to
0.0998..0.1004, i.e. all ten streams becoming interchangeable.

**Control.** The reference Gator-Path on the *simplest* version of the task
(length=2: predict `start+d1` and `start+d1+d2` mod 12) escapes chance at
**step 1000** and reaches **78.6%**. GTR-L on the identical task was flat at
chance through **8000** steps. That looked damning.

**Resolution.** At `--learning-rate 3e-3` instead of `8e-3`, GTR-L on
length=2 escapes chance at step 2500 and goes on to **solve the task
outright**:

```
step   500  1000  1500  2000  2500  3000  3500  4000  4500  5000  5500
acc   0.089 0.077 0.080 0.479 0.479 0.604 0.881 0.993 1.000 1.000 1.000
                        ^ escape
```

**1.000 accuracy, against the reference's 0.786 plateau on the same task.**
GTR-L is not merely viable here, it is better -- once the learning rate
matches the model.

The 8e-3 default was inherited from the reference, which has **1,106
parameters**. GTR-L has **33,901** and four LayerNorms per relay. Carrying
that hyperparameter across a 30x parameter change was an assumption I never
tested, and it cost most of the session.

The transition is abrupt -- flat for ~2000 steps, then a jump -- which is
the usual shape for modular arithmetic and is easy to mistake for a dead
run if you stop early or read a decaying gradient norm as convergence.

**Consequence for the collapse diagnostics.** The occupancy flattening, bus
opening, and stream collapse are almost certainly *downstream* of the model
sitting at the marginal, not the cause of it. With no gradient signal
distinguishing routes, there is nothing to hold occupancy structured, and
diffusion plus an openable bus does the rest. I read them as the disease;
they look more like a symptom. Worth re-measuring at the working LR before
adding machinery to fight them.

## 3b. Accumulation, not addressing: the length-7 blocker

length=2 solves at 1.000 but length=7 does not, which the learning rate
alone does not explain. Two independent probes converge on the same cause.

**Probe 1 -- copy versus sum.** Strip the task to addressing only: output
slot `j` must reproduce observed token `j`, no arithmetic.

| task | requires | result |
|---|---|---|
| COPY | addressing only | **1.000 accuracy by step 1000**, loss 0.0001, width 8 |
| prefix sum | accumulation | chance |

Routing, addressing, mailbox read/write and information flow are therefore
all sound. The defect is specific to *accumulating* values.

**Probe 2 -- what the scan actually returns.** The decayed scans were
divided by `sum_{j<=l} decay^j` to keep channels on a common scale. That
normaliser is `(1 - decay^(l+1)) / (1 - decay)`, and whenever
`1/(1 - decay)` exceeds the sequence length it grows essentially as `l + 1`
across the whole sequence. Feeding a constant input, where a true prefix sum
at slot `l` equals `l + 1`:

| slot | true sum | raw scan (decay 0.982) | normalised |
|---|---|---|---|
| 0 | 1 | 1.00 | 1.000 |
| 7 | 8 | 7.51 | **1.000** |
| 14 | 15 | 13.25 | **1.000** |

The normalised scan is **exactly flat**: it cannot distinguish one
accumulated item from fifteen. Long-range channels (`decay ~ 0.98` gives
`~55` against `L = 15`) are all in this regime; short-range channels stay
sum-like but reach only locally. So with normalisation on, **no channel can
express a long-range sum**, and the prefix task is nothing but accumulation.

This also explains the length split precisely: at length=2 the sum/mean
discrepancy is a factor of 4-5 and the model can compensate; at length=7 it
is 9-15 across a seven-deep chain.

Fix: `scan_normalise` (default `True`, behaviour unchanged) and
`--no-scan-normalise`. The downstream LayerNorm absorbs the scale when it is
off. This matters *more* for language, not less -- contexts are ~40x longer,
so the distortion is correspondingly larger.

## 4. Hypotheses tested and rejected

Recorded because the rejections were as informative as the fixes, and
because I chased two of them further than the evidence justified.

| hypothesis | test | verdict |
|---|---|---|
| Receptive field too small: `conv_kernel=3` composes to reach 4 over 4 relays and cannot express the required step at distance 7 | run length=2, which needs reach 2 | **rejected** -- length=2 also flat at chance for 8000 steps |
| Evidence pathway too weak at init | per-slot input->output sensitivity vs reference | **rejected** -- GTR-L is only 1.3-3.1x weaker, not orders of magnitude |
| Structural inability to express the target | overfit 16 fixed examples | **rejected** -- 1.000 accuracy by step 300 |
| Task is inherently hard at this scale | reference on the same length=2 task | **rejected** -- reference learns it by step 1000 |
| Learning rate mismatched to model size | LR sweep at length=2 | **CONFIRMED** -- 3e-3 escapes, 8e-3 never does |

The receptive-field theory was seductive because the arithmetic worked out
cleanly (required distance `d>=7`, available reach 4) and it predicted the
failure. It was still wrong. The length=2 control killed it in one run and
should have been the *first* experiment, not the fifth.

## 5. Real bugs found and fixed during bring-up

All five presented **identically** -- loss pinned at `ln(12)`, accuracy at
chance -- which is why they took so long to separate.

1. **Broad support must not mean uniform support.** Ingress at
   `temperature=4.0`, keys at `std=0.02` gave occupancy entropy 3.672 of a
   3.689 maximum. Flat occupancy means every node holds identical content
   and nothing can be addressed. Fixed: `temperature=1.0`, keys at
   `1/sqrt(width)`. The non-negotiable is "no node starved of gradient",
   which is *not* the same as "occupancy is flat".
2. **A shared global bus homogenises.** One shared `bus_in/bus_out` pair at
   `std=0.02` made `softmax(bus_out)` near-uniform, and the gate sat at
   0.726 -- 73% of route mass replaced by a uniform redistribution every
   relay, erasing ingress addressing by relay 1. Fixed: per-stream buses,
   selective init, gate biased closed.
3. **The mailbox carried no signal.** Measured input->output sensitivity was
   **exactly 0.0000**: the model could not see its own input. Cause: the
   mailbox carries `O(S/N)` magnitude (~0.08 of the residual stream at
   S=10, N=40), attenuating to ~1e-4 over four relays. Found by a causal
   probe, not by staring at loss curves.
4. **My fix for (3) broke selectivity.** Normalising by each node's *own*
   mass -- the `numerator/denominator` form from
   `gator_convergent._node_memory` -- fixed the scale but made an
   unaddressed node return a full-magnitude average instead of ~0,
   destroying the address signal. Fixed by normalising by the *mean* mass
   across nodes: scale corrected, relative node differences preserved.
5. **A plain cumulative scan cannot address by distance.** "Forward scan"
   implemented as a cumulative mean weights every earlier slot identically.
   Replaced with `decayed_scan`, a per-channel decayed scan evaluated in
   fixed-size chunks -- transient `[chunk, chunk, C]`, not `[L, L, C]`.
   Verified exact against the naive recurrence and **chunk-size
   independent**, which is what makes the linear blocking legitimate.

Plus one pure-efficiency fix: the four mixer views were being stacked into
`[B,L,N,C,4,Dm]` purely to write one tidy einsum, which was the single
largest allocation in the model. Summing four contractions is identical
arithmetic (verified to 3.6e-07) and cut eval memory 30%.

## 6. Diagnostics that actually worked

For reuse -- each settled in one run what training runs could not.

* **Causal sensitivity probe.** Perturb one input token, measure mean change
  in output log-probs. Reading `0.0000` proved the model was blind to its
  input; indistinguishable from bad tuning on a loss curve.
* **Overfit a tiny fixed batch.** Cannot drive 16 examples to zero loss =>
  structural. Reaching 1.000 moved the question to optimisation.
* **Run the reference on the same task.** Separates "hard task" from "broken
  model". This was the single highest-value experiment and I ran it late.
* **Shrink the task.** length=2 instead of length=7 falsified the
  receptive-field theory in one run.
* **Chunk-size equivalence.** Confirms a blocked linear kernel computes the
  intended recurrence and is not secretly quadratic.
* **Ablate the task, not the model.** Copy isolates addressing from
  arithmetic; copy succeeding at 1.000 while sum sat at chance localised the
  defect to accumulation in a single run.
* **Feed a constant and check the invariant.** A prefix sum over ones must
  return `l + 1`. Reading a flat `1.000` exposed the normaliser immediately,
  where no amount of accuracy-watching would have.

## 6b. Reading the curves: Gator training is an S-curve

Gator-family models train in long flat stretches punctuated by sharp rises,
not gradual improvement. GTR-L's own length=2 run is the clearest example:

```
step   500  1000  1500  2000  2500  3000  3500  4000  4500  5000
acc   0.089 0.077 0.080 0.479 0.479 0.604 0.881 0.993 1.000 1.000
```

Flat at chance for 2,000 steps, then effectively solved within 2,000 more.

This has a direct methodological consequence, and one I repeatedly got
wrong during this session: **a flat stretch is not evidence of failure.**
Several hypotheses here were judged on runs of a few thousand steps sitting
at chance, and at least one correct explanation was discarded that way. Use
continuous leading indicators -- stream agreement, occupancy entropy, route
entropy -- to steer while accuracy is flat, and reserve accuracy for
confirming an outcome rather than ranking configurations mid-run.

## 6c. The collapse is a frozen prediction, not stream homogenisation

Reading the per-stream dump settles which of two similar-looking failures is
actually occurring, and it is not the one I spent longest on.

At step 1500 every stream emits **one constant token regardless of input**
(`share=1.00`, `distinct=1`) -- not a weak prediction, no prediction. And
crucially the *homogeneous* control already emits six different constants
across its ten equal-width streams:

```
[uniform] widths = 32 x 10
   s0 top=11   s1 top=10   s2 top= 7   s3 top=11   s4 top= 5
   s5 top=11   s6 top= 4   s7 top=11   s8 top= 4   s9 top= 6
```

So streams differentiate on their own, without heterogeneous widths, private
adapters, or reduced contact. The `agreement -> 1.0` collapse measured at
step 18500 is a *late* consequence of already sitting at the marginal, not
its cause. Diverse voters that are each frozen still fuse to a frozen
answer.

That reframes a chunk of this session: individuality work targeted a
symptom. The disease is `distinct=1`.

Two properties make the frozen state cheap to occupy:

* the residual gate initialises at `sigmoid(0) = 0.5`, so each relay admits
  half the context-derived update and LayerNorm renormalises straight after
  -- input influence is diluted once per relay, four times over;
* the vocabulary bias can express the target marginal on its own, at no cost
  to the input pathway. `loss = ln(12)` is exactly what that buys.

Measured at initialisation, the fraction of inputs that actually change the
predicted token:

| gate bias | readout bias | sensitivity | argmax flip rate |
|---|---|---|---|
| 0.0 (default) | yes | 0.0225 | 0.0039 |
| 0.0 | no | 0.0150 | **0.0000** |
| 2.0 | yes | 0.0277 | 0.0206 |
| 3.0 | yes | 0.0275 | 0.0162 |
| **3.0** | **no** | 0.0221 | **0.0882** |

22x more argmax mobility from the pair, and they interact: removing the bias
with the gate closed is *worse* than baseline, since nothing then carries
enough input-dependent variation to move the argmax at all.

Note that mean sensitivity barely moves across the whole table (0.015-0.028)
while flip rate spans 0.0000-0.0882. **Sensitivity was the wrong metric.**
It is possible to be measurably input-sensitive and still emit a constant,
because the variation never crosses an argmax boundary. Any future
diagnosis of a frozen output should measure argmax mobility directly.

## 7. Open items

* **>=99% at length=7: running.** length=2 at 3e-3 reached **1.000**, so the
  criterion is met at short length; the criteria are defined on length=7,
  which is now running at 3e-3 and 1e-3 alongside a matched reference.
* **Escape step vs reference: answered at length=2.** Reference escapes at
  step 1000 and plateaus at 0.786; GTR-L escapes at 2500 and reaches 1.000.
  Slower to leave chance, materially better on arrival.
* **Fusion synergy and node health: re-measure.** The collapse numbers were
  all taken at the broken LR and are probably not informative about the
  working configuration.
* **LR sensitivity unmapped.** 3e-3 works, 8e-3 does not. 1e-3 and 3e-4 are
  running. Worth knowing whether the window is narrow, because that matters
  for the 130k/1M language gates.
* **Language gates not started**, blocked on the above.

## 8. What I would question if reviewing this

* Is `delta_norm` (LayerNorm on the program-basis update) right? It makes
  relay strength independent of node count, which helps scaling
  comparability, but it also normalises away the *magnitude* of the expected
  program coefficient, keeping only its direction. Route concentration may
  carry real signal that this discards.
* The bus gate learning to open toward homogenisation is a dynamic that will
  recur at language scale, where long-range transport genuinely matters. A
  closed init is a workaround, not a principle.
* GTR-L is 18x heavier than the reference at short sequence length. Fine for
  language, but it makes short-sequence tasks a poor validation vehicle --
  which is worth weighing given the acceptance criteria are defined on one.
