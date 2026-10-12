# Gator Training Guide

**Status:** Living internal guide  
**Last updated:** July 17, 2026

This guide records practical training behavior, recommended defaults, and diagnostic signals for Graph Trace Relay / Gator experiments. Findings are preliminary and should be updated as multi-seed evidence accumulates.

## Gator-Path P4 learning proof

P4 is the first training shell for the packet-path architecture. It is not an
Arrow performance run and should be judged first on whether fresh generated
prefix accuracy rises above the 8.3% chance floor.

| Setting | P4 default |
|---|---:|
| Cells | 1 |
| Internal nodes | 4 |
| Persistent packet width | 8 |
| Immutable address width | 4 |
| Relays | 4, shared weights |
| Selected paths | top-2 soft |
| Parameters | 1,218 |
| Prefix length | 7 |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 0.008, constant |
| Weight decay | 0.0001 |
| Gradient clipping | 1.0 |
| Temperature | `max(0.35, 1.25 * 0.997**step)` |
| Evaluation | deterministic fresh generated examples |

Reference command:

```bash
python -u gator_path_prefix_train.py \
  --steps 10000 \
  --batch-size 128 \
  --eval-batch-size 2048 \
  --length 7 \
  --cells 1 \
  --relays 4 \
  --top-k 2 \
  --learning-rate 0.008 \
  --log-every 500
```

The trainer reports token accuracy, exact-sequence accuracy, loss, route
entropy, actual `(cell,node)` movement, and mean occupied mailbox rows. Preserve
both best and latest checkpoints. A route may sharpen before task accuracy
moves; this is diagnostic, not proof that the algorithm has been acquired.

The local 100-step CPU sanity run reduced held-out loss from `3.3724` to
`2.5048` while remaining at chance accuracy. That is expected: it verifies
that the model learns to exclude the four non-target symbols and reaches the
rough `log(12)` baseline before any prefix algorithm is visible. It is not a
learning result. P4 needs the full 10,000-step GPU run.

## 1. Current recommended baseline

Use this configuration before changing one architectural variable at a time:

| Setting | Recommended value |
|---|---:|
| Architecture | Vectorized stitched-cell Gator-ARW |
| Cells | 10 |
| Nodes per cell | 4 |
| Hidden width per cell | 8 |
| Neighbor degree | 4 |
| Relays | 4 |
| Parameters | 10,000 |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 0.008 |
| Weight decay | 0.0001 |
| Gradient clipping | 1.0 |
| Routing temperature | `max(0.35, 1.25 * 0.997**step)` |
| Training budget | 20,000 steps |
| Compilation | `torch.compile` on GPU |

Reference command:

```bash
python -u gator_cells_vectorized.py \
  --neighbors 4 \
  --steps 20000 \
  --batch-size 128 \
  --seed 0 \
  --compile
```

## 2. Validated seed-0 reference curve

Task: simultaneous modular prefix formation, length 7.

| Step | Correct /7 | Token accuracy | Loss |
|---:|---:|---:|---:|
| 1,000 | 0.61 | 8.7% | 2.4861 |
| 2,000 | 0.86 | 12.3% | 2.3021 |
| 3,000 | 2.65 | 37.9% | 1.6600 |
| 4,000 | 3.88 | 55.4% | 1.1278 |
| 5,000 | 4.78 | 68.3% | 0.7776 |
| 6,000 | 5.36 | 76.5% | 0.5850 |
| 7,000 | 5.66 | 80.9% | 0.4407 |
| 8,000 | 6.11 | 87.3% | 0.3278 |
| 9,000 | 6.11 | 87.2% | 0.3089 |
| 10,000 | 6.25 | 89.2% | 0.2449 |
| 11,000 | 6.43 | 91.9% | 0.1975 |
| 12,000 | 6.75 | 96.5% | 0.1103 |
| 13,000 | 6.67 | 95.3% | 0.1388 |
| 14,000 | 6.72 | 96.0% | 0.1345 |
| 15,000 | 6.79 | 97.0% | 0.0851 |
| 16,000 | 6.81 | 97.3% | 0.0742 |
| 17,000 | 6.86 | 98.0% | 0.0635 |
| 18,000 | 6.93 | 99.0% | 0.0308 |
| 19,000 | 6.96 | 99.4% | 0.0221 |
| 20,000 | 6.98 | 99.7% | 0.0126 |
| Final large evaluation | 6.99 | **99.8%** | **0.0096** |

Final exact-sequence accuracy was **98.8%**.

## 3. Seed-dependent acquisition

Algorithm acquisition is reproducible across seeds, but onset time, slope, and oscillation differ.

### Seed 0

- Stayed near chance through step 2,000.
- Sharp transition between steps 2,000 and 4,000.
- Reached 76.5% by step 6,000.
- Exhibited a shelf near 87% around steps 8,000–9,000.
- Entered the 95–97% regime around steps 12,000–15,000.
- Reached 99% at step 18,000.

### Seed 1

| Step | Token accuracy |
|---:|---:|
| 2,000 | 35.0% |
| 4,000 | 46.0% |
| 6,000 | 62.6% |
| 10,000 | 77.8% |
| 12,000 | 83.9% |
| 14,000 | 62.6% |
| 16,000 | 93.3% |
| 18,000 | 92.6% |
| 20,000 | 91.4% |

Seed 1 acquired useful structure earlier but consolidated more slowly and showed strong oscillation. Final large-set accuracy was 91.3%, exact-sequence accuracy was 59.6%, and loss was 0.2515. Its best logged checkpoint was step 16,000 rather than the final checkpoint. Do not judge a run solely by which seed leaves chance first, and always preserve the best checkpoint.

### Working interpretation

- Topology appears to determine whether a high-capability solution is attainable.
- Initialization influences acquisition time and the path taken through partial solutions.
- Earlier acquisition does not guarantee faster convergence.
- Oscillation may reflect cell-role competition, routing reorganization, or finite evaluation noise.

These interpretations remain provisional until fixed-batch diagnostics and additional seeds are available.

### Seed 2

| Step | Token accuracy |
|---:|---:|
| 1,000 | 8.4% |
| 2,000 | 8.3% |
| 3,000 | 18.1% |
| 5,000 | 32.7% |
| 10,000 | 38.1% |
| 15,000 | 40.0% |
| 20,000 | 40.8% |

Final large-set accuracy was 41.0%, exact-sequence accuracy was 0%, and loss was 1.5217. Seed 2 followed a slow but persistent consolidation trajectory. It was still making incremental progress near step 20,000 and may require a substantially larger budget before another acquisition transition.

Across the first three seeds, final token accuracy spans 41.0% to 99.8%. The topology reliably outperforms chance, but optimization speed and path vary dramatically. Do not classify a slow seed as a failed basin without extending its training budget. Optimizer stabilization, resume support, and best-checkpoint retention are core training requirements rather than optional tuning.

## 4. Expected learning regimes

A healthy degree-4 run may show several regimes:

1. **Coordination phase:** accuracy remains near the 8.3% chance floor.
2. **Initial acquisition:** accuracy abruptly rises above 20%.
3. **Rapid formation learning:** accuracy climbs through 50–80%.
4. **Intermediate shelf:** progress slows or oscillates around a partial solution.
5. **Refinement transition:** accuracy moves into the 95%+ regime.
6. **Residual correction:** rare cases are resolved and exact-sequence accuracy rises sharply.

Do not terminate a run merely because it is near chance at step 1,000 or temporarily flat later. Seed 0 remained near chance at step 2,000 and ultimately reached 99.8%.

## 5. Thresholds to record

For every run, record the first step reaching:

- 10%: leaves chance reliably
- 20%: initial acquisition
- 50%: broad partial algorithm
- 75%: useful formation procedure
- 90%: strong generalization
- 95%: refinement regime
- 99%: near-solution

Also record:

- Best accuracy
- Final accuracy
- Exact-sequence accuracy
- Loss
- Correct positions out of target length
- Runtime and steps/second
- Peak VRAM

## 6. Evaluation protocol

Checkpoint minibatch accuracy is useful for monitoring but is not a final result.

For final evaluation:

1. Use a deterministic held-out generator seed.
2. Evaluate at least 4,096 fresh examples; use 100,000 when practical.
3. Report token accuracy and exact-sequence accuracy.
4. Report accuracy by output position.
5. Preserve the checkpoint SHA-256.
6. Use `torch.load(..., weights_only=True)` when loading checkpoints.

Future logging should use the same fixed evaluation set at every checkpoint. This separates real optimization oscillation from sampling noise.

## 7. Checkpoint protocol

Save every 1,000 steps during exploratory runs and more frequently around acquisition transitions.

Recommended filename:

```text
gator_{cells}cell_d{neighbors}_r{relays}_seed{seed}_step{step}.pt
```

Example:

```text
gator_10cell_d4_r4_seed1_step20000.pt
```

Never overwrite a checkpoint from another seed. Record architecture, task, optimizer, runtime, metrics, and SHA-256 in `experimental_checkpoints/README.md`.

## 8. Topology findings

At approximately 10,000 parameters and 5,000 steps:

| Configuration | Accuracy |
|---|---:|
| Four-node width-42 monolith | 8.5% |
| Sparse 40-node width-7 monolith | 24.7% |
| Ten-cell degree-2 ring | 31.9% |
| Ten-cell degree-4 ring, loop implementation | approximately 45–50% |
| Ten-cell degree-4 ring, vectorized seed 0 | 68.3% |

The vectorized and loop implementations are numerically equivalent at initialization to within `5.96e-7`; the reported training difference may include evaluation sampling and optimization divergence. Replicate before attributing it to implementation.

Current evidence favors multiple locally complete cells with inter-cell communication over naïve width scaling.

## 9. Clean A/B rules

Change one architectural variable at a time:

- Neighbor degree: 2 → 4 → 6 → 8
- Relay depth: 1 → 2 → 4 → 8
- Cell count: hold total parameters fixed when possible
- Local node topology: test only after cell-neighbor sweeps
- Coupling topology: ring, wider ring, all-to-all, hierarchical, small-world

Use the same seed and generated batches for the first A/B. Then replicate the winner across seeds.

## 10. Warning signs

### Near chance through 5,000 steps

Likely causes:

- Inadequate topology
- Excessive width with insufficient structured workspace
- Too few useful receiver-specific interactions
- Implementation or parameter-count mismatch

The width-42, four-node 10k model stayed at chance through 5,000 steps.

### Early jump followed by stagnation

The model may have learned a partial algorithm. Inspect per-position accuracy, route structure, and cell agreement before adding capacity.

### Strong oscillation

Possible causes:

- Evaluation sampling noise
- Cell-vote competition
- Neighbor-message interference
- Routing specialization changes
- Oversmoothing

Log fixed-batch accuracy, output votes, neighbor gates, gradient norm, route entropy, and cell agreement.

### Longer-length forgetting

Curriculum training from prefix length 7 to 14 transferred competence forward but catastrophically forgot shorter lengths. Mix previous lengths during training if retention is required.

## 11. Performance guidance

- Prefer `gator_cells_vectorized.py` for GPU training.
- Use `torch.compile` when supported.
- The T4 vectorized seed-0 run achieved 38.32 steps/second.
- The 20,000-step run completed in 521.91 seconds.
- The warning about insufficient SMs for `max_autotune_gemm` on a T4 is harmless.
- The 10,000 weights occupy only about 40 KB in FP32; runtime is dominated by activations, optimizer state, compilation, and kernel orchestration.
- The completed 62-cell language log ran 100,000 steps in 18,582.35 seconds
  (5.38 steps/second). Record the exact GPU name in future logs; `device=cuda`
  alone is insufficient for performance comparisons.
- Operator-observed 50k language runtimes were roughly four hours on T4 and
  forty minutes on RTX Pro 6000. This apparent approximately 6x improvement is
  provisional because the runs were not a locked hardware A/B.
- Always record compilation time separately from training runtime.

### Quality-matched architecture timing

Do not rank Gator and Transformer using steps/second alone. Use a threshold
table for held-out loss, perplexity, token accuracy, and exact-window accuracy,
then report wall-clock time and training tokens required to reach each
threshold. A model that never reaches a threshold within budget is `not
reached`, regardless of raw throughput. Preserve these three views separately:

1. Statistical efficiency: quality per training example, token, and step.
2. Compute efficiency: quality per FLOP or joule when instrumentation exists.
3. Wall-clock efficiency: elapsed time to matched held-out quality.

## 12. Historical next-experiment list

1. Complete seed 1 and seed 2 at degree 4.
2. Evaluate each final checkpoint on the same large fixed set.
3. Sweep degree 6 and degree 8 using seed 0.
4. Add fixed-batch diagnostics every 250–1,000 steps.
5. Test length extrapolation only after seed reproducibility is established.
6. Map width, cell count, node count, neighbor degree, and relay depth as separate scaling axes.

## 13. Current next experiments

The current queue is the five-stage formation-alignment program. Do not change
topology, corpus, tokenizer-piece inventory, optimizer, or stewardship schedule
inside these comparisons:

1. Preserve the passing PAD/EOS/whitespace/rendering/legal-response audit as a
   release gate.
2. Run matched `slot_ce` and `blank_ce` controls with tokenizer V2.
3. Compare their best-monotonic-aligned diagnostics without selecting on raw
   rigid accuracy alone.
4. Run detached `viterbi` and differentiable `monotonic` objectives.
5. Add the low-weight whole-sequence composition/length/EOS auxiliary only if
   aligned training improves the matched baseline.

Use the experiment matrix and claim boundary in
[`GATOR_FORMATION_ALIGNMENT_ROADMAP.md`](GATOR_FORMATION_ALIGNMENT_ROADMAP.md).

### Historical topology queue

This list records the queue before the topology sweep completed. It is retained
for provenance and is no longer the active ordering.

1. Preserve the 124,616-parameter symmetric, feature-gated, grouped-output
   SmolTalk reference run.
2. Run the asymmetric Arrow node mask with no other changes. **Completed.**
3. Repeat it with `hybrid_alternating`.
4. Repeat it with `hybrid_hemispheric`.
5. Compare best and final checkpoints on the same normal, shuffled, and blank
   validation contexts.
6. If one topology wins, replicate it across seeds before changing another
   architectural lever.
7. Compare 62 four-node cells against a matched-budget design with more nodes
   per cell only after establishing the 62-cell reference curve.
8. Evaluate each prefix checkpoint on the same large fixed set.
9. Add fixed-batch route, vote, and gate diagnostics.
10. Map cell count, node count, neighbor degree, and relay depth separately while
   retaining width 8 as the validated local-cell default.

### Locked topology-sweep controls

For symmetric, asymmetric, alternating hybrid, and hemispheric hybrid, hold all
of the following fixed:

- Filtered SmolTalk only; identical conversation split, tokenizer seed, and
  standalone-whitespace tokenizer version 1
- Seed 0 and identical deterministic batch-sampling order
- 2,048 vocabulary; context 64; 32 response slots
- 62 cells, width 8, four nodes, four relays, degree-four cell ring
- Communal lexical mode and `feature_gated` cell input
- `grouped_concat` four-channel output
- 124,616 parameters
- Batch size, optimizer, LR schedule, evaluation set, and training budget

Node adjacency has no trainable parameters, making this a clean topology A/B.
The older asymmetric and hybrid language runs are not controls for this sweep:
they used truncated/noisy I/O and shared width-8 population bottlenecks.

## 14. Scaling rule: preserve the validated cell

The current atomic unit is the 980-parameter GTR-ARW configuration:

- Width 8
- Four local nodes
- Four relays
- Immutable addresses
- Controlled writing
- Receiver-specific relative reads

Do not widen this unit merely because a larger parameter budget is available.
The width-42 monolith failed while stitched width-8 cells succeeded. Preserve
the local cell, but do **not** assume that adding cells alone proportionally
adds usable collective capacity. Cell population, communication radius, relay
depth, and readout credit form one coupled scaling law.

For the vectorized prefix ring, the exact parameter count is:

```text
P(C) = 980C + 4C + 160 = 984C + 160
```

The `4C` term contains two neighbor gates, one self gate, and one output vote
per cell. The shared left/right projections and coupling norm contribute 160.
This gives exactly 10,000 parameters at ten cells and 61,168 at 62 cells.

With `K` symmetric neighbors, ring radius is `K/2`. After `R` relays, one
cell's largest possible local receptive field is:

```text
H(C,K,R) = min(C, 1 + K R)
```

At the validated `C=10, K=4, R=4`, the theoretical 17-cell field covers the
entire ten-cell ring. At `C=62` with the same neighbor and relay settings, a
cell can integrate at most 17 of 62 cells before readout. Merely adding cells
therefore converts a globally coordinated ten-cell society into many partially
coordinated neighborhoods.

The initial output vote is also uniform. Each cell contributes `1/C` of the
logits, so the direct per-cell output gradient is diluted from `0.1` at ten
cells to approximately `0.0161` at 62 cells: a 6.2-fold reduction before any
cell specializes. This does not make larger populations useless, but it means
they need at least one of:

- relay depth or topology sufficient for the intended coordination diameter;
- hierarchical or regional aggregation;
- packet-conditioned output credit rather than a static uniform population
  vote;
- staged growth that preserves already useful specialists;
- a sparse policy learned from dense complementary branches instead of hard
  pruning at initialization.

Always report `H/C` and initial vote mass alongside parameter count. A scaling
run that increases `C` while shrinking both quantities is not a clean test of
whether more Gator cells add useful capacity.

Also report final effective output cells and the union of their relay
receptive fields. The mastered seed-0 ten-cell checkpoint and a plateaued
same-seed rerun both converged to approximately two effective output cells.
At `C=10`, two winners can still integrate the entire population because
`H=10`. At `C=62`, two fixed winners have at most two 17-cell neighborhoods
before overlap, so a large fraction of added cells can become causally
stranded after vote collapse. This is a direct reason population count alone
need not improve performance.

### Fixed-budget graph-factorization sweep

Cells are a tested parameter/communication factorization, not a vision
requirement. Hold the parameter budget and total local-node count near 10k/40
while varying the boundary:

| Decomposition | Parameters | Local width | Graph interpretation |
|---|---:|---:|---|
| `1 x 40` | 10,010 | 7 | 40 locations, one persistent payload lane |
| `10 x 4` | 10,000 | 8 | 40 locations, ten persistent cell payload lanes |
| `20 x 2` | 10,280 | 8 | 40 locations, twenty persistent cell payload lanes |

This is a systems-level factorization comparison, not a single-variable
ablation: private I/O, local router size, and graph hierarchy move with the
boundary. Its purpose is to determine whether cell modularity earns its
overhead. If the flat graph catches the modular systems by 20k, cell wrappers
are optional. If modular systems retain a material acquisition or final-
quality advantage, follow-up tests must isolate private lexical tables,
normalization boundaries, and hierarchical routing separately.

Degree six is used for `20 x 2` because `H(20,6,4)=20`, preserving full-ring
causal reach. A degree-four 20-cell ring has `H=17` and would confound cell
factorization with stranded communication range.

Record payload-lane count separately from node count. These runs determine
whether persistent semantic lanes explain the modular advantage; they do not
yet compare cell wrappers against a flat graph with the same number of
independent branch payloads.

The lane-matched flat control is:

```bash
python -u gator_convergent.py \
  --steps 20000 \
  --batch-size 128 \
  --eval-batch-size 1024 \
  --log-every 1000 \
  --seed 0 \
  --compile \
  --checkpoint experimental_checkpoints/gator_convergent_seed0_20k/checkpoint.pt
```

Compare this 10,010-parameter, 40-node, ten-lane model against 10x4 Arrow.
Record terminal effective lanes, node support, transition entropy, accuracy,
exactness, and throughput. Do not describe the one-lane `SparseGator` as the
vision-complete flat baseline.

The completed four-relay seed-0 run reached `92.1%` token accuracy, `61.0%`
exact accuracy, loss `0.2211`, and `3.56` local CPU steps/s. It narrowed from
`7.89` to `3.04` effective terminal lanes while remaining at approximately
`1.39` effective node locations per lane. Top-three terminal fusion retained
`91.2%` accuracy; the best single lane reached only `27.8%`. Treat this as
evidence that broad traces converged into complementary specialists.

This exceeds the matched local OG control by `41.1` accuracy points. Treat the
preserved historical 99%+ Arrow checkpoint as an upper-bound observation, not
the current baseline, because OG could not reproduce its trajectory in this
environment. The next clean A/B uses eight relays with every other setting
fixed. Do not scale cells, nodes, or lanes in that control.

Also run a one-relay control before changing capacity. It retains all ten
parallel traces and the same 10,010 parameters, but permits only one local
handoff and refinement. With degree three, per-lane reachable support is 3
nodes at one relay, 9 at four relays, and 17 at eight relays. Therefore the
`R=1` comparison measures the joint practical value of recurrent computation
and recurrently expanded graph reach. If that distinction matters after the
first result, follow it with a one-relay wide-neighborhood control.

Interpret R=1 correctly for the current 40-node/10-lane seeding. Starts are
four nodes apart, so degree-three frontiers remain disjoint after one handoff,
first collide after two, and cannot read shared traffic until relay three.
Thus R=1 and R=2 are independent parallel-route controls; R=3 is the minimum
depth that tests graph-mediated lane coordination. Log `lane_pair_overlap`
from the checkpoint analyzer when comparing depths.

The default R=1/R=2/R=3 sweep is operational, not perfectly causal. Its
controlled-writing schedules are respectively `[0]`, `[0,1]`, and
`[0,0.5,1]`, so R=2 also introduces a relay where formed output packets may
write back into memory. To isolate collision timing, repeat the sweep with a
shared `relay_code_span` and `write_warmup_relays` at least as large as the
largest compared depth. Then every common-prefix hop has the same code and
write mass.

Also record that R=1 still has a global within-lane read: every sequence
position in a lane begins at the same node, so receiver-relative memory can
aggregate all prefix inputs immediately. Its 10k result was `69.5%` token /
`6.2%` exact. Do not cite the current task as proof that four relays learned a
logarithmic scan until a locality-forcing ingress or memory-mask control
removes this one-hop shortcut.

At 20k the same isolated R=1 run reached `84.7%` token, `30.3%` exact,
and `0.4446` loss at `19.15` steps/s end to end. Node support had narrowed
to `1.10` effective locations while terminal support widened to `6.76`
lanes. Compare this with R=4's `92.1%` token and `61.0%` exact result:
recurrent coordination contributes much more to whole-formation coherence
than the token-accuracy gap alone suggests.

The operational R=2 run then reached `85.6%` token, `36.1%` exact, and
`0.4064` loss at 10k, averaging `9.90` steps/s. This exceeded R=1 at 20k
and R=4/R=8 at 10k. Do not assume relay-depth monotonicity during training:
additional recurrent depth increases usable compute but also lengthens the
credit-assignment chain and changes the write curriculum. Screen the entire
learning curve, not only final relay count.

Probe the R=2 checkpoint before claiming graph-mediated cooperation. In the
seed-zero control, lane overlap was `0.0` at \(q^0\) and \(q^1\) and only
appeared at terminal \(q^2\) (`0.209867`), too late for a memory read. The
best lane alone produced `25.5%`, while the top six reached `85.7%`.
Therefore R=2 is an independent recurrent ensemble; any R=3 discontinuity is
the first candidate evidence for in-graph trace cooperation.

R=2 remained the best shallow early learner but did not master by 20k. Final
performance was `91.0%` token / `50.7%` exact, and the best logged point was
`91.7%` / `55.5%` at 16k. Use R=2 when update efficiency matters, but do not
yet treat it as an asymptotic substitute for a mastered four-relay solution.

In the operational seed-zero run, R=3 did not produce a positive
discontinuity. It ended 10k at `71.3%` token / `8.5%` exact, versus R=2 at
`85.6%` / `36.1%`. Terminal support narrowed to `2.60` effective lanes at
5k before reopening to `4.27` by 10k. Treat graph collision as a learnable
coordination channel with an interference cost, not as free additional
capacity. Analyzer overlap was `0.626731` at consumed \(q^2\), so the model
did establish shared graph traffic; it simply had not learned to exploit it
as effectively as R=2's independent terminal ensemble by 10k.

R=3 remained underpowered at 20k: `82.2%` token / `25.0%` exact, with a
`4.71`-lane effective terminal coalition. Do not choose R=3 merely because it
is the first depth with cross-lane memory. For this length-seven task, it
incurs coordination interference but lacks the fourth commitment pass needed
to consolidate the interacting traces.

Use `--bootstrap-context-mode self` for the zero-parameter locality-forcing
control. Relay zero is self-context only; shared node-memory reads begin on
relay one. A one-relay model must consequently be invariant to the observed
condition tokens at its blank outputs, while two or more relays can transfer
condition information through routed meetings. This is a stronger test of
the tracing claim than merely increasing relay count under the default
`shared` bootstrap.

Use `analyze_arrow_checkpoint.py --relay-limit N` to test whether a trained
Arrow solution actually uses its recurrent depth. The option stops execution
after N hops without renormalizing relay codes or the output-write schedule.
On the mastered seed-zero checkpoint, token accuracy was `12.1%`, `20.9%`,
`58.3%`, and `99.76%` after one through four hops; exact accuracy was
`0%`, `0%`, `3.2%`, and `98.49%`. Never remove trained relays merely because
an independently trained shallow model performs well—the deep checkpoint may
have distributed an inseparable computation across all of them.

The reproducible Convergent R=4 checkpoint confirmed the pattern with
`9.5%`, `10.4%`, `35.3%`, and `91.2%` token accuracy after one through four
trained hops. Always evaluate relay truncation separately from
depth-specific retraining: they answer different questions.

The separately trained R=1 control did not reach mastery by 30k. Final
performance was `92.0%` token / `56.3%` exact, with a best logged point of
`92.3%` / `57.5%` at 29k. This is useful but not a 99% substitute: it used
50% more updates than the R=4 20k comparison and still had weaker exact
formation. Continue the shallow-depth search with R=2, not by assuming R=1
will eventually match at an economical training budget.

Always log accuracy and recurrence-step consistency by output position when
sweeping relay depth. In this run, R=1 fell from `99.95%` at the first output
to `82.67%` at output five; R=4 raised that late position to `90.04%`.
Mastered Arrow's fourth hop moved its worst late position from `19.29%` to
`99.66%`. Aggregate token accuracy hides the main relay benefit: reliable
long-range formation and a final global commitment transition.

The R=4 continuation reached `98.2%` token / `88.5%` exact at 30k, while
R=1 and R=2 remained near a 92% token ceiling. Train with enough relay depth
when mastery matters, even if R=2 wins the early-update curve. At late R=4
training, constant LR `0.008` caused large local swings despite a rising best
checkpoint. Preserve best states and test a reduced-LR consolidation phase
before adding capacity or changing topology.

Treat the default 1/4/8 sweep as the operational architecture comparison, not
as pure recurrent-depth isolation. Relay codes and controlled-writing ramps
are normalized over configured depth. The strict R=4 versus R=8 control is:

```bash
# Add these to both commands.
--relay-code-span 8 --write-warmup-relays 4
```

With matched weights and inputs, tests prove the first four routing traces are
then identical. R=8 differs only by executing four additional hops. Run this
control before attributing the entire default-depth advantage to computation
rather than the gentler eight-step write curriculum.

For local Convergent prefix runs, use the exact grouped relay path now in
`gator_convergent.py`. Against the prior committed implementation, a
single-thread batch-32 benchmark improved:

| Relay depth | Prior | Exact optimized | Gain |
|---:|---:|---:|---:|
| 4 | 7.12 steps/s | 8.82 steps/s | 24% |
| 8 | 2.70 steps/s | 3.71 steps/s | 37% |

The optimized full-model FP32 outputs agree within `4.77e-7`, with worst
observed relative parameter-gradient disagreement `5.17e-7`. Do not infer
equivalence from shapes alone: Convergent node weights are stored
`[input, output]`, while the shared grouped helper accepts
`[output, input]`. The required transpose is protected by a dedicated test.

Do not select receiver convolution for this length-nine prefix benchmark. It
was approximately `1.7x` slower than the quadratic receiver read because its
launch/setup overhead dominates at tiny sequence length. Re-evaluate it for
language-length contexts rather than generalizing this short-sequence result.

Do not set physical route width from effective-node entropy alone. The trained
four-relay checkpoint reported `1.39` effective nodes, but per-relay
renormalization gave:

| Route support | Token accuracy | Exact accuracy |
|---:|---:|---:|
| Soft | 91.45% | 58.20% |
| Top-1 | 20.54% | 0.00% |
| Top-2 | 49.71% | 0.78% |
| Top-3 | 72.64% | 14.84% |
| Top-4 | 87.71% | 46.58% |
| Top-5 | 91.16% | 57.13% |
| Top-7 | 91.45% | 58.20% |

Use top-5 as the first packed-execution candidate for this checkpoint, not as
a universal default. Re-run the ablation for every materially different
depth, seed, or task before pruning. Top-2 retained `96.73%` of final route
mass but only `49.71%` accuracy; top-4 retained `99.70%` yet lost `3.74`
accuracy points. Do not use route mass alone as a contribution score.

Degree-three reachable work grows faster than relay count:

```text
node-transform rows per lane = R(R + 2)
router-source rows per lane   = R^2
```

At eight relays this is `80/64`; a learned top-5 support cap would lower the
upper counts to approximately `38/34`. Current `route_top_k` is a capability
ablation only and still evaluates dense reachable transforms. Do not report a
speedup until selected rows are physically gathered and packed.

Do not implement packing by gathering a separate node-weight matrix for every
packet. The exact prototype reduced four-relay throughput from `7.70` to
`5.19` steps/s and eight-relay throughput from `3.12` to `2.27` on the local
single-thread screen. Bucket selected packets by node, run one contiguous
transform per node bucket, then scatter outputs.

Prefer a relay-wise support schedule over one fixed top-k. On the four-relay
checkpoint:

```text
soft support [3,5,7,9] -> 91.45% token / 58.20% exact
schedule     [3,5,6,4] -> 91.43% token / 57.42% exact
```

The schedule removes 25% of node-transform rows with essentially unchanged
token accuracy. Top-2 only at the final relay retained `90.62%`, while top-2
throughout fell to `49.71%`. Keep early exploration broad and narrow near
commitment; validate the schedule separately for each depth and seed.

After selecting relay depth, use the exact Convergent budget formula

```text
P(K,N,D) = K(2D^2 + 31D + 1) + N(D^2 + 8D + 28)
```

to separate payload-lane scaling from graph-site scaling. The first
fixed-budget screens are:

| Screen | Lanes K | Nodes N | Width D | Parameters |
|---|---:|---:|---:|---:|
| Fewer, wider traces | 6 | 40 | 9 | 9,892 |
| Current reference | 10 | 40 | 8 | 10,010 |
| More, narrower traces | 16 | 40 | 7 | 10,376 |
| Fewer, wider graph sites | 10 | 30 | 9 | 9,850 |
| More, narrower graph sites | 10 | 50 | 7 | 9,810 |

Run the lane trio first, then the node trio, using the winning relay depth,
portable batches, matched seed, optimizer, and update budget. Report both
accuracy per update and accuracy per wall-clock. These are factorization A/Bs,
not claims that width is held constant.

The flat arm uses uniform initial node mass. Its older narrow-start screen put
almost all ingress mass on one of 40 nodes and reached 24.7% at 5k; that tested
one seeded path against ten seeded cells, not the intended broad-to-specialized
vision. Preserve the narrow result as history but use broad ingress for the
factorization comparison.

For language, always record `lexical_mode`. `private_hash` adds a private
16-by-8 table per cell selected by `token_id % 16`; it matches the tiny math
cell's lexical allocation but aliases 2,048 language tokens into only 16
buckets. `communal` uses the shared full-vocabulary embedding plus a learned
cell identity and matches the first promising wide language pilot. Test these
at the same 62-cell topology before spending the communal model's saved budget
on additional cells.

Also record `cell_input_mode`:

- `shared_projection`: all cells receive the same width-8 projection.
- `feature_gated`: each cell gates the width-32 communal embedding before the
  shared projection, adding 32 parameters per cell.

At 62 cells, feature gating adds 1,984 parameters and starts exactly equivalent
to the shared model because every gate initializes to one. Feature gating
restored plausible but wrong contextual answers, confirming that the shared
input keyhole had been discarding useful language information. Formations were
still capped around four or five displayed words, showing that the input
keyhole was not the only bottleneck.

Record `cell_output_mode` as well:

- `shared_pool`: pool all cells to one width-8 vector, then project 8 to 32.
- `grouped_concat`: pool four interleaved cell groups independently, concatenate
  `4 x 8` into width 32, then apply an identity-initialized 32-to-32 mixer.

The full gated/grouped model has 124,616 parameters. Test it on the same
SmolTalk-only context-64 protocol before returning to LMSYS or larger contexts.
Record maximum coherent formed length in tokenizer pieces, not only displayed
words. The shared-pool model's longest observed phrase appeared to contain
approximately eight pieces, equal to its pooled width; treat this as a
diagnostic correlation rather than a theoretical length bound.

### Formation-span learning curve

Gator has been observed to begin with one-word formations and gradually add
words during training rather than producing a full coherent response window
immediately. Because all slots are evaluated simultaneously, record this as a
separate learning curve rather than treating short early responses only as bad
samples.

At every evaluation, log:

- Predicted tokenizer pieces before first EOS: mean, median, p10, and p90
- Fractions at or above 1, 2, 4, 8, 16, and 32 pieces
- Mean target length and prediction/target length ratio
- Accuracy for every output position
- One random decoded example, with its raw predicted piece count
- Under tokenizer V2: visible content, whitespace, BLANK, EOS, and forbidden-
  special counts separately, plus diagnostic rendering of all output slots
- After aligned metrics land: rigid accuracy/loss beside best-aligned
  accuracy/loss and aligned exactness

Do not equate hidden width with a hard token-length ceiling. Use these metrics
to distinguish progressive slot coordination from EOS bias, tokenizer spaces,
punctuation, or response-length priors.

The first grouped-readout run illustrates why best-checkpoint retention is
required. Its best logged loss/PPL occurred around step 13k (3.7461/42.36),
while step 24k had worsened to 3.7953/44.49 and step 50k reached
4.1408/62.85. Accuracy declined from a best 32.7% to 30.2%. Formation quality
and aggregate metrics were not monotonic. Preserve the best checkpoint and
evaluate it beside the final checkpoint on identical normal, shuffled, and
blank contexts. For SmolTalk-scale screens, an early-stop window around 10k–20k
is more informative than assuming a fixed 50k endpoint is best.

The matched asymmetric topology showed the same general pattern. Its best
loss/PPL arrived later, at step 18k (3.7607/42.98), and it finished at
4.0844/59.40. At 30k it was essentially tied with symmetric (47.39 versus
47.35 PPL). Do not infer stability from one middle checkpoint: report the best
region, matched-step curve, final retention, and qualitative degeneration as
separate quantities.

Hybrid alternating reinforced this rule: it matched the best displayed PPL of
42.36 at step 12k, then reached 50.43 by step 31k and 62.00 by step 50k while
qualitative response types remained recognizable. Track fixed-probe greedy
agreement, target probability, top-two margin, entropy, and content-only PPL
to distinguish behavioral collapse from confidence or calibration regression.

Hybrid hemispheric established the most stable 50k curve. It bottomed at PPL
43.12 at step 13k, stayed near 43–44 through 25k, and finished at 56.89. It was
the best topology at the matched 25k, 31k, and 50k checkpoints despite not
having the absolute best single-point minimum. For long curriculum stages,
rank configurations by both minimum PPL and retained PPL; best-checkpoint
rollback remains mandatory even with the more stable hemispheric topology.

For the next neighbor and relay screens, hybrid hemispheric is the operational
default and symmetric remains the comparison contender. Do not generalize the
current ranking to all Gator sizes. Once a neighbor or relay configuration
materially improves the curve, rerun that configuration under both masks to
test whether hemispheric retention and symmetric peak quality persist.

For 25k screens, do not silently compare against the first 25k steps of a 50k
cosine run. The current CLI sets cosine `T_max` from `--steps`; a 25k screen
therefore decays twice as quickly. Either run a matched 25k control or expose a
fixed scheduler horizon independently from the stopping budget.

The first relay-depth result is promising: six relays at four neighbors held
124,616 parameters, reached PPL 42.50 at 15k, and ended at 43.42. It processed
52.48 steps/second versus 73.17 for the four-relay 50k hemispheric run. Treat
this as evidence that recurrent depth may improve the performance basin, then
confirm with additional relay counts and a matched scheduler control.

Do not select relay count from validation PPL alone. Report steps/second,
seconds-to-best, examples-to-best, and approximate relay operations alongside
minimum and retained PPL. More recurrent computation is useful only if its
quality gain justifies the training and inference latency increase.

### Whitespace is currently an easy target class

The compact tokenizer learns standalone whitespace pieces. Logged formations
such as `Yes` followed by a distant closing quote contain many predicted spaces,
not unused slots. All-token accuracy can therefore be dominated by whitespace.

For every language evaluation, report separate accuracy and prediction rates
for whitespace-only pieces, lexical pieces, punctuation, and EOS. Also report
content accuracy with whitespace-only targets removed. Never use raw token
accuracy alone to rank language configurations using this tokenizer.

The completed four-mask topology sweep deliberately retained the standalone-
space tokenizer so node adjacency remained the only changed variable. Compare masks
using loss/PPL, qualitative formation, and eventually whitespace-excluded
accuracy; do not interpret raw accuracy semantically. After symmetric,
asymmetric, alternating hybrid, and hemispheric hybrid finished, the former
plan was to introduce a leading-whitespace tokenizer. The current priority is
instead to distinguish visible whitespace from a true non-rendering BLANK and
audit all response masks before changing token-piece segmentation.

## 15. Language-training protocol

- Remove SmolTalk's repeated first assistant greeting from scored examples.
- Split by complete conversation before tokenizer training.
- Reserve the first 512 valid LMSYS conversations for validation.
- Train the compact vocabulary only on training conversations.
- Use a 90% LMSYS / 10% filtered SmolTalk sampling mixture.
- Historical corpus screens packed fixed 64-context/32-response windows into
  CPU `int32` tensors. Alignment screens may keep a 32-piece reference cap while
  allocating 40 physical response slots for placement slack.
- Evaluate the complete validation set and print a random context every log.
- Use cosine LR decay; a printed constant 0.003 means the scheduler flag is
  missing, not that cosine decay is operating.
- Record `node_topology` and `lexical_mode` in every language checkpoint and
  run name. Run the
  all-symmetric topology first, then alternating hybrid and hemispheric hybrid.
  Retain asymmetric checkpoints rather than overwriting them.

### Output semantics and alignment gate

The release gate is now implemented. `gator-compact-v2` reserves a stable-ID
BLANK at the end of the requested vocabulary; PAD stores tensor padding, MASK
is a control marker, EOS terminates complete references, and whitespace pieces
remain visible text. Gator initializes output queries from BLANK. Response
logits forbid PAD, BOS, MASK, USER, and ASSISTANT. A clipped target remains a
right-censored prefix and receives no synthetic EOS. V2 vocabulary size is an
exact model contract; lower `--vocab-size` if a small corpus cannot fill it.

The test suite verifies:

- IDs beneath a false context mask cannot affect logits, routes, route
  gradients, loss, or metrics; PAD has zero Gator-S ownership.
- Tensor target padding contributes to no loss or metric.
- Complete targets contain exactly one scored EOS, while censored targets make
  no EOS assertion.
- The response vocabulary forbids PAD, BOS, MASK, USER, and ASSISTANT but permits
  EOS, BLANK, byte tokens, and learned text pieces.
- Whitespace masks include learned whitespace pieces and byte whitespace, but
  never structural symbols; whitespace auxiliaries never touch PAD or BLANK.
- Plain rendering hides BLANK and stops at EOS, while diagnostic rendering
  exposes every output slot.

The trainer exposes `slot_ce`, `blank_ce`, `viterbi`, and `monotonic`. The
BLANK control keeps reference piece `j` in slot `j`, scores EOS for complete
targets, and scores their remaining known slots as BLANK. A right-censored
prefix leaves its unknown continuation unscored. Viterbi chooses a detached
monotonic placement; forward-sum marginalizes every legal placement and enters
a free continuation state after a censored prefix is consumed.
Alignment consumes every reference token once in order and never collapses
repeated text tokens. `--alignment-warmup-fraction` blends from rigid BLANK CE;
`--alignment-max-shift` controls address freedom.

The implemented sequence auxiliary compares semantic token composition,
expected emitted length, and complete-response EOS count. Composition and EOS
apply only to complete references; censored prefixes receive only one-sided
under-length pressure. Start with
`--sequence-quality-penalty 0.01`; keep selected formation loss as the
checkpoint metric and remove the auxiliary if it does not improve aligned
quality. The trainer rejects old whitespace penalties for aligned objectives.
Report raw rigid metrics throughout for continuity.
The runner shares one masked log-softmax across formation losses and metrics,
does not evaluate disabled whitespace auxiliaries, and the generic optimizer
refuses to update or checkpoint through non-finite loss/gradient states.
Gradient clipping uses the normal fast PyTorch foreach path. If its FP32 norm
reduction overflows while every gradient element remains finite, the optimizer
recomputes the norm with FP64 accumulation and applies the requested clip
instead of mistaking a clip-worthy spike for a NaN. Actual NaN/Inf elements
still abort the step.
The full specification is
[`GATOR_FORMATION_ALIGNMENT_ROADMAP.md`](GATOR_FORMATION_ALIGNMENT_ROADMAP.md).

First full-stack screen:

```bash
  --response-tokens 40 --reference-tokens 32 \
  --formation-objective monotonic \
  --alignment-warmup-fraction 0.05 --alignment-max-shift 8 \
  --sequence-quality-penalty 0.01 \
  --whitespace-omission-penalty 0 --adjacent-whitespace-penalty 0
```

Before attributing any gain to alignment, rerun identical `slot_ce`,
`blank_ce`, and `viterbi` controls. Keep physical response slots and reference
cap identical across all four modes; otherwise alignment slack itself becomes
an uncontrolled change.

### Clean heterogeneous-chat streaming

The old LMSYS path could tail-crop an overlength context in the middle of a
sentence, removing the beginning of the latest user instruction and its role
marker. That explains the sentence fragments seen in logged prompts. The clean
path now drops only complete oldest turns. The surviving suffix must begin with
a user turn and fit the context window in full; otherwise the example is
rejected. The latest user message is never partially cropped.

Use a 90/10 LMSYS/SmolTalk stream with these quality controls:

- LMSYS raw row at most 512 decoded characters.
- ASCII-letter ratio at least 0.90 and response length at least 8 characters.
- Assistant response must be immediately preceded by a user turn.
- Complete 64-token context after whole-turn trimming.
- Normalized context/response deduplication per corpus traversal.
- Preserve the first LMSYS assistant response; `--skip-first-assistant` is a
  SmolTalk greeting control, while LMSYS has a separate opt-in flag.

For the historical 32-slot corpus baseline, do not require the whole assistant
response plus EOS to fit the response window. Training its first 32 tokens is an intentional fixed
formation window and retains longer, useful answers. A strict full-response
screen retained only about 12.8% of sampled LMSYS candidates. The corrected
context-only screen retained 130/210 (61.9%) in the real-data smoke test, while
SmolTalk retained 5,320/5,726 (92.9%). The trainer logs every rejection category,
mixture counts, and `stream_restarts`; checkpoints preserve the same metadata.

This cleaning pipeline is now the production baseline. Any change to context
completeness, response completeness, language ratio, deduplication, or first-
assistant handling changes the data distribution and must be labeled as an A/B.

The selected first control uses `--context-tokens 512` and
`--lmsys-max-characters 512`. With 5.4 GB observed at context 64, response 32,
and batch 32, quadratic slot scaling estimates batch 8 at approximately 43.4
GB for context 512 plus 32 response slots. Launch with `--max-vram-gb 50`,
measure actual peak allocation, and lower the batch if compile workspaces or
non-allocator CUDA overhead exceed the remaining headroom. The flag constrains
PyTorch's caching allocator; it cannot police every allocation made by CUDA or
external libraries.

**Correction:** the compiled batch-8 run actually used approximately 17 GB.
The 43.4 GB calculation was an overly conservative projection that scaled fixed
memory and assumed simultaneous residency of intermediates that the compiled
execution reuses or releases. Future batch sizing should begin from the 17 GB
empirical measurement for this exact configuration, increase one step at a
time, and continue enforcing the 50 GB allocator cap.

A later approximately one-million-parameter run with 1,038 width-8 cells,
vocabulary 2,048, and context 64 also used approximately 16 GB. This is
comfortable on the current GPU but unfavorable relative to the tiny parameter
count. Treat memory as activation-dominated. Log `memory_allocated`,
`memory_reserved`, and `max_memory_allocated` after compilation and after a
training step; do not report only the dashboard's process total. Cell count,
context length, batch size, and relay count must all accompany VRAM figures.

The legacy `broadcast_mix` grouped readout learns one static vote per cell. Pin equal voting,
static learned voting, and dynamic slot-conditioned voting as a controlled
readout sweep. For large cell populations, also log vote entropy, maximum vote,
top-k vote mass, and effective cell count `1 / sum(w**2)` so apparent scaling is
not secretly routed through only a handful of cells.

The 999,112-parameter/1,038-cell screen reached PPL 42.16 but did not materially
lengthen formations and ran at only 7.05 steps/second. Treat this as a practical
null scaling result, not proof that cell capacity is useless. The fixed
four-group readout compressed roughly 259 cells into each width-8 output
channel, and static vote gradients began highly diluted. Use the 124,616-
parameter model for the next diagnostic loop.

Diagnostic order before scaling again:

1. Verify that a 130k model can nearly memorize a small fixed training subset.
2. Log fixed training and validation PPL together to distinguish optimization
   floor from generalization floor.
3. Measure whitespace-excluded accuracy/PPL and formation span by slot.
4. Log static vote entropy, top-k mass, and effective participating cells.
5. Compare equal, static, and dynamic slot-conditioned voting.
6. Probe whether individual/frozen cell states contain target information that
   the native grouped readout fails to extract.

Use `--repeat-training-examples 500` for the first test. It selects one fixed,
seeded SmolTalk training subset, rejects LMSYS mixing, repeatedly samples only
that pool, evaluates all 500 training examples at every log interval, and saves
the lowest-training-loss state as `best_train.pt`. Validation remains the
original held-out conversation split. A successful memorization test must be
judged from the `train_subset` loss, exact accuracy, per-position/content
metrics, and formed span—not validation PPL alone.

### Fixed-subset memorization ladder

The 124,616-parameter model perfectly memorized 10 examples by step 4k, 100 by
18k, 200 by 31k, and 400 by 80k. Exact accuracy remained near zero until token
accuracy exceeded roughly 90%, then followed the expected sequence-length
relationship. The observed points approximately matched `token_accuracy^20`,
indicating an effective target length near twenty evaluated pieces rather than
a separate exact-formation defect.

The 800-example run had not fully mastered at 200k, but training reached 96.8%
token accuracy, 48.0% exact, loss 0.1675, and PPL 1.18. At batch 32 this is
8,000 sampled exposures per example. The expected independent-slot exact rate
`0.968^20 = 0.522` is close to the observed 0.480. Thus the remaining exact
gap is largely explained by residual token errors across response length.

Held-out behavior collapsed during the same run. At step 105k, held-out loss
was 19.0219 while the training subset was at loss 0.9103; the displayed sample
remained contextually plausible. By step 200k, held-out accuracy was 12.9%,
loss 42.0929, and the sample had degenerated into repetitive fragments, while
training continued improving. The printed held-out PPL of 485,165,195 is the
evaluator's `exp(20)` display cap; true `exp(42.0929)` is much larger.

Interpret this as two-stage over-specialization: confidence/calibration fails
before greedy behavior, then behavioral generalization fails under prolonged
narrow repetition. It proves that the model can form and memorize hundreds of
complete windows, but does not support deterministic repeated-subset training
as a production regime. Future training should emphasize broad streaming data,
fresh curriculum stages, rollback, and calibration controls.

### Gator-S stewardship screen

Gator-S is now available through `--cell-transport-mode stewardship`.
It adds explicit per-token cell ownership to the current Arrow population. Each
source cell processes its owned token branch, conditionally chooses self or a
ring neighbor, and sends ownership mass, hidden state, and the four-node
passport together. The final ownership trace replaces static cell votes.
End-task gradients come from whichever formation objective is selected. Rigid
CE remains the historical baseline; BLANK/aligned objectives are
implemented language-shell A/Bs and do not change stewardship handoff semantics.

Execution is independently selectable. `dense` is the compile-friendly
semantic reference and still executes all 62 cells. `packed_auto` now keeps
that compiled dense engine during broad exploration. Every configured number
of completed steps, an external controller reads one short vector containing
the input support of each relay. It latches the earliest relay whose full
remaining suffix stays below the configured fraction. Complete dense calls
before that latch remain compiled. Once a packed plan is active, the complete
model call is eager, including any dense prefix before the sparse boundary; a
staged compiled-prefix/eager-suffix call is not implemented. No route-density
tensor is copied to the host from inside the relay loop. Dormant branches
receive no Arrow transform in packed relays. The current packed dispatcher
remains eager, so route sparsity must not be equated with a speedup until
measured.

Run the first clean A/B with the existing communal, feature-gated,
grouped-output, hemispheric 62-cell language setup. Change only the population
transport and its explicitly reported routing controls:

```bash
  --cell-transport-mode stewardship \
  --stewardship-keep-mass 0.9 \
  --stewardship-handoff-keep-mass 0.9 \
  --stewardship-min-active 1 --stewardship-max-active 0 \
  --stewardship-exploration 0.01 --stewardship-scouts 1 \
  --stewardship-execution packed_auto \
  --packed-stewardship-backend persistent_bucketed \
  --stewardship-packed-threshold 0.5 \
  --stewardship-execution-check-every 100 \
  --stewardship-warmup-fraction 0.05 \
  --stewardship-temperature-start 1.5 \
  --stewardship-temperature-end 0.35 \
  --stewardship-familiar-entropy-penalty 0.001 \
  --stewardship-balance-penalty 0.01
```

At the reference dimensions, `stewardship` has 124,618 parameters versus
124,616 for `broadcast_mix`. The two-parameter difference is sufficiently close
for the first capacity A/B, but record the exact count.

The route starts completely broad: ownership is `1/C` and pruning is disabled
during the first 5% of training. Stewardship temperature then falls
geometrically from 1.5 to 0.35. Once pruning begins, the router retains an
adaptive probability nucleus rather than a fixed `top-k`. The base retained
mass is 0.90. Normalized handoff entropy raises that threshold toward 1.0, so a
confident familiar route can narrow while an uncertain route can recruit more
neighbors again. `--stewardship-max-active 0` means no CLI cap; it does not mean
zero cells. The configured scout reserves low ownership mass for an alternate
route outside the exploitation nucleus. That route stays alive even on
familiar work and can become a wider beam if a later steward is uncertain.

There are two nested elastic nuclei. The handoff nucleus is local to each
steward: an uncertain steward can retain `X` of its five legal exits. The
ownership nucleus controls the token's resulting population after arrivals
merge. Global reopening uses the maximum uncertainty among live stewards, so a
low-mass scout is allowed to fan out instead of being silenced by a
mass-weighted average. The ownership-weighted mean uncertainty is logged
separately and drives the familiarity-weighted sharpening loss.

Checkpoint selection always evaluates with pruning enabled and the fixed final
stewardship temperature, independent of the current training schedule. This is
required for comparable `best.pt` validation loss across steps. The live
training temperature/prune state remains in the step log.

Call this **elastic novelty-driven widening** only with the qualification that
the present novelty signal is route uncertainty. It has not yet been shown to
identify semantic novelty. A direct validation should compare handoff entropy
for familiar validation examples, uncommon but coherent examples, corrupted
prompts, and distribution-shifted prompts.

Log and retain all stewardship diagnostic lines:

- `support`: mean nonzero owned cells at the initial state and after each relay
- `effective`: `exp(route entropy)`, which distinguishes diffuse mass from a
  nominally large but negligible tail
- `uncertainty`: maximum normalized entropy among the token's live stewards
- `mean_uncertainty`: smoother mass-weighted uncertainty used by route loss
- `handoff_beam`: mean number of live exits retained by each token's stewards
- `widen` / `narrow`: fraction of token routes whose support changed direction
- `stay`: route mass assigned to the source cell's self action
- `load_cv`: imbalance of marginal ownership across the cell population
- `dead`: fraction of cells with effectively zero marginal load
- `packed`: whether each relay used the packed live-route kernel
- `execution_plan`: `dense` or the latched `packed_from_relay_N` suffix

Also preserve `route_entropy_loss`, `route_balance_loss`, stewardship
temperature, and pruning state. A healthy first run should demonstrate mass
conservation and finite gradients, then show nontrivial handoffs without
collapsing every token into one cell or retaining all 62 cells forever. No
numerical threshold is validated yet; compare curves against multiple seeds
before declaring either narrowing or widening optimal.

Left-padded context positions must have zero stewardship mass and are excluded
from support, load, uncertainty, and auxiliary-loss denominators. All output
queries remain valid because their target validity is not available at
inference.

Warning signs and first responses:

- All routes stay dense after warmup: lower `keep-mass` in a labeled A/B or
  inspect whether handoff logits remain uniform before increasing entropy loss.
- Immediate one-cell collapse: increase warmup/temperature or exploration;
  inspect load balance before forcing a larger minimum support.
- Same cell owns every token: verify the decaying marginal-balance term and
  check `load_cv`/dead-cell fraction.
- Support only shrinks and never reopens: compare uncertainty with widening;
  the novelty proxy or local neighbor graph may be too weak.
- High widening with no quality gain: cap active support or test whether route
  entropy is merely indecision rather than useful novelty.
- `packed` remains false after several scheduled checks: the run is still a
  compiled dense semantic experiment because support has not crossed the
  configured break-even threshold. Inspect per-relay support before changing
  the threshold.

Packed token stewardship is implemented as ragged route records containing
batch, token, cell, ownership mass, branch state, and node passport. Records
are grouped by occupied `(batch,cell)` for receiver-specific memory; only live
records receive the Arrow transform. The default
`--packed-stewardship-backend persistent_bucketed` packs once at the selected
boundary and keeps state, passport, mass, and indices packed through the
complete sparse suffix. Uncertainty, adaptive pruning, support-only handoff
transport, collision merging, and final readout are packed-native. Cell-local
router and node projections gather only active cell parameter slices. The
legacy `--packed-stewardship-backend repack` implementation remains the
globally padded, pack/unpack-per-relay forward numerical oracle.

Receiver groups use fixed power-of-two-like capacities rather than one global
maximum. If live group lengths are `L_g` and `M=sum_g L_g`, padded row capacity
is below `2M`, and pairwise receiver work is below
`4 * sum_g(L_g^2)`. Treat these as structural worst-case bounds, not measured
speed or memory results. Dynamic `nonzero`, sorting, and `unique` remain eager
and may synchronize. The node passport remains soft, so every active cell still
evaluates all four node transforms. Initial ownership is dense and uniform, so
the sparse backend cannot pay off until pruning learns useful support.

The dense and packed implementations currently agree within floating-point
tolerance in the forward equivalence tests. After straight-through pruning,
their gradient estimators need not agree: the dense graph can carry surrogate
gradient through a zero-forward route that packing removes. Compare learning
curves across the execution transition, not only forward logits. The execution
plan is checkpointed and restored on continuation. Training requests a minimal
route-loss diagnostic profile containing packed-native entropy, per-cell load,
uncertainty, and active-fraction summaries rather than a dense ownership
history. Evaluation requests route metrics without the large hidden/location
histories reserved for full trace inspection.

Use `benchmark_sparse_stewardship.py` to force support levels and compare
forward-only execution of the two eager packed backends. Its tiny CPU mode
validates the harness and
numerical path only; it is not general performance evidence. No target-GPU or
end-to-end speedup is established. The current local validation gate passes
145 tests plus scratch and exact data-pipeline resume trainer smokes.

The progress-rate logger now divides locally completed steps by elapsed time.
Older resumed logs that divided global checkpoint step by the new phase's
elapsed time showed an artificial high rate that decayed over the run; that
shape was a reporting error, not measured thermal or compute degradation.

### Dense kernel and compute policy

The milestone-3 dense backend changes execution without changing learned
parameters or Arrow equations. Use the current defaults for controlled runs:

```bash
  --dense-node-backend folded_bmm \
  --dense-router-backend einsum \
  --dense-kernel-padding 1
```

The folded node backend applies the exact identity
`W(base + node_embedding) + bias = W(base) + (W(node_embedding) + bias)`.
It joins the four width-8 node projections into one width-8-to-width-32
projection per cell and removes the materialized `[B,C,T,N,D]` node input.
That allocation is 25.2 MiB FP32 per relay at
`B=32,C=62,T=104,N=4,D=8`.

The representative local CPU forward-kernel screen used
`B=4,C=62,T=104,N=4,D=8`. Expanded node transforms measured 9.9242 ms and
folded transforms measured 2.6216 ms (3.79x). Grouped router BMM was only 1.13x
faster than `einsum` and remains opt-in. Padding the small shared projection to
eight measured 3.0146 ms versus 1.1743 ms without padding, so padding remains
one by default. These numbers are not end-to-end training timings and must not
be reported as GPU results. Run `benchmark_dense_kernels.py` on the actual
training device before changing router or padding policy.

For the first Blackwell A/B, the designated hardware policy is:

```bash
  --float32-matmul-precision high \
  --cudnn-tf32 \
  --compile --compile-mode default --compile-shape-padding
```

No Blackwell result has been recorded yet. Compare it against the exact/default
runtime policy on fixed shapes, with warmup excluded, and report forward,
backward, step time, peak allocated memory, and final learning metrics. Named
non-default compile modes cannot be combined with the custom shape-padding
option. The compute policy is checkpoint runtime metadata and may change on
resume without redefining model capacity.

Do not enable float16 AMP as a casual speed flag. The current `-1e9` route mask
overflows FP16 and the `1e-9` entropy clamp underflows it. Mixed precision
requires an explicit FP32 island for routing, softmax, entropy, and
language/alignment losses before it is a valid training option.

### Packed Convergent routing gate

The operational source of truth for target-GPU stages, pass criteria, and
artifacts is
[`../BLACKWELL_FFT_TRITON_GATE.md`](../BLACKWELL_FFT_TRITON_GATE.md). This
section explains experiment ordering; it does not replace that release gate.

Packed Convergent execution must be introduced in an order that separates
equation-preserving backend work from estimator or architecture changes:

1. dense route oracle plus matrix receiver memory;
2. packed routes plus the exact matrix receiver-memory oracle;
3. packed routes plus exact direct-convolution memory;
4. packed routes plus exact FFT memory;
5. opaque FFT plus the Torch packed reader to validate the compiler boundary;
6. opaque FFT plus the fused Triton packed reader;
7. adaptive route width as a separate estimator A/B;
8. shared-trunk/private-adapter nodes as a separate architecture A/B;
9. hierarchical learned ingress as a separate architecture A/B.

Do not change K, adapter rank, or ingress hierarchy while deciding whether the
packed representation itself is faster. `--packed-memory-compact-nodes` and
`--compact-node-dispatch` currently use data-dependent eager shapes and are
invalid with `--compile`. Adaptive K retains a static `Kmax` workspace under
compilation, so route narrowing is not a compute saving until an eager compact
run or a future K-bucketed/fused kernel demonstrates it.

Every report must record maximum and mean live route K, route uncertainty,
`node_union`, forward/backward/optimizer time, kernel count, and peak allocated
and reserved VRAM. The first target is the frozen 140M control at batch 6,
context 512, output 128, whose pre-optimization Blackwell language-training
rate was 3.22 steps/s. Tiny CPU timings are implementation clues only and must
never be promoted into a training-speed claim.

The first fused receiver launch is:

```bash
python benchmark_convergent_routing.py \
  --route-backend packed \
  --packed-memory-backend fft_opaque \
  --packed-memory-reader-backend triton \
  --packed-memory-feature-chunk 32 \
  --vocab-size 32768 --embedding-dim 768 --d-model 768 \
  --nodes 154 --lanes 10 --degree 3 --relays 2 \
  --router-rank 96 --active-node-top-k 32 \
  --batch-size 6 --context-tokens 512 --output-slots 128 \
  --device cuda --dtype float32 --compile \
  --warmup-steps 20 --measured-steps 100
```

Run the small `validate_fused_fft_backend.py` correctness/gradient gate first.
Explicit Triton selection must fail when unsupported; a silent Torch fallback
invalidates timing. Record compile time separately from warmed execution. The
opaque backward now recomputes only bounded writer/FFT chunks, invokes the
native reader adjoint, and immediately releases each chunk. Compare both
forward/backward time and peak allocation against the frozen VJP-backed gate;
the implementation change is not a win until the target GPU demonstrates it.

### Input pipeline, host overlap, and exact continuation

The production language path now has an execution-only input API:

```bash
  --batch-pipeline prefetch --prefetch-batches 2 \
  --pin-memory --non-blocking-transfer --dedicated-copy-stream
```

Finite SmolTalk examples are collated once and sampled by tensor row gather.
Large in-memory packed corpora remain int32 tensors through weighted-mixture
selection. The bounded producer prepares the next CPU batches while the model
works. A CUDA run can stage pinned transfer through a dedicated copy stream,
but the current forward waits for that copy; true device-batch lookahead is not
implemented. Per-step logs report queue depth/high-watermark, mean CPU
sampling time, host transfer-enqueue time, producer waits, and failures. The
reported transfer value is not CUDA-event H2D duration.
Source counters are labeled `mixture_prepared_samples` because they include
the bounded queue, not only batches already consumed by optimizer steps.

Startup tokenization can be parallelized independently:

```bash
  --tokenizer-workers 4 --tokenizer-worker-backend processes \
  --tokenizer-chunk-size 256 --tokenizer-pending-chunks 8
```

Results are consumed in source order. Workers perform only pure encoding;
filter statistics, deduplication, shuffle decisions, and stream restarts stay
with the coordinator. Leave workers at zero until startup preparation is a
measured bottleneck, because process creation and tokenizer serialization can
cost more than they save on a small corpus.

Checkpointed finite and packed providers restore their RNG state together with
the complete prefetched queue. The trainer fingerprints file identity,
tokenizer, task window, finite example contents, filtering, mixture weights,
and sampling seed. Use:

- `--resume-data-state auto` to restore only when the fingerprint and provider
  both support exact continuation, otherwise log a seeded reset.
- `--resume-data-state exact` for deterministic A/Bs that must fail rather than
  reset.
- `--resume-data-state reset` when intentionally beginning a new corpus phase.

Generic streamed JSONL sources currently cannot restore an arbitrary iterator
position and shuffle buffer. They explicitly report non-exact state; do not
describe an `auto` streaming resume as continuous data order.

For GPU optimization screens, begin with
`--execution-preset optimized_gpu_safe`. It keeps the convolution receiver,
folded nodes, persistent sparse execution, and bounded prefetch while using
strict FP32 and no compilation. This is the preserved long-run Arrow fallback.
Compiled Blackwell runs eventually produced multiple non-finite backward
signatures, including genuine localized cell NaNs. `--compile`,
`optimized_gpu`, and the corresponding dense controls remain available for
short diagnostic profiling, but compile investigation is deferred while
Gator-Path replaces this execution graph. Explicit low-level flags override
preset values.

The 500-iteration local forward-free benchmark with 4,096 examples at batch
32, context 64, response 32 measured 559.06 batches/s for legacy Python
collation, 4,375.49 for vectorized finite sampling, and 2,981.37 for direct
packed sampling, with identical checksums. Prefetch was slower than direct
gathering because no GPU work existed to overlap.
The required target-GPU A/B is therefore serial versus prefetch at fixed model,
checkpoint, seed, and batch order after compilation warmup. Compare
end-to-end steps/s, consumer stall time, GPU idle gaps, and H2D overlap; do not
promote the CPU microbenchmark to a training-speed claim.

Next measure sparse support, VRAM, throughput, and formation quality
independently on the target GPU. The ordered implementation and release gates
for receiver fusion, compiled-prefix staging, bounded sparse dispatch, capped
top-k, host/alignment fusion, and single-use tensor elimination are pinned in
[`PERFORMANCE_TODO.md`](../PERFORMANCE_TODO.md). Learned STOP/halting and
global `(cell,node)` dispatch remain planned.

### Matched one-shot Transformer warning

A 130,072-parameter, four-layer non-autoregressive Transformer trained on the
same clean mixture reached PPL 31.45 by step 22k, beating Gator's reported 32.8
at 250k. Its greedy responses nevertheless remained mostly one-to-three generic
content tokens followed by whitespace, and fixed semantic probes failed. Raw
PPL can therefore improve through modest probability on the reference token
while another generic token remains the argmax. Always report content-only PPL,
target rank/top-k accuracy, whitespace probability and omission rate, distinct
greedy-token concentration, formation coverage, and fixed context probes beside
aggregate PPL.

### Whitespace under-formation intervention

The first architecture-neutral output intervention added two small auxiliary
terms while retaining ordinary cross-entropy as the reported validation PPL.
Use omission weight 0.10 and adjacent-whitespace weight 0.05 for the initial
A/B. Omission penalizes whitespace probability only where the active target is
non-whitespace. The adjacency term applies only to pairs of scored target slots
and excludes legitimate target whitespace pairs. Neither term touches padding
after EOS, dictates a replacement word, or feeds predictions back into the
model. Report content-only PPL/top-1/top-5, whitespace omission/mass, adjacent
whitespace rate, and formation coverage beside aggregate PPL.
These targeted checks do not replace the broader PAD/EOS/response-vocabulary
release gate above. Reevaluate both terms after scored BLANK tails are
available; they must not be silently carried into aligned-objective runs.

The first unfiltered SmolTalk pilot reached approximately 43% token accuracy and
26.6% exact accuracy, but 26.16% of validation responses were the identical
greeting `Hello! How can I help you today?`. Treat that run as a template-collapse
diagnostic, not evidence of contextual language learning.

The corrected 62-cell private-hash asymmetric run reached 27.5% token accuracy,
loss 3.7253, and PPL 41.48 at 100k steps. The symmetric version reached 27.3%,
loss 3.7499, and PPL 42.52 at 42k. Both usually emitted one frequent opening
token followed by visually empty, usually whitespace-filled, or repetitive
slots. Symmetry alone therefore did not restore the pilot's qualitatively
plausible multi-token formations. The communal lexical and topology A/Bs are
now historical controls; the current priority is the formation-alignment
program.

## Diversified continuation protocol (2026-07-17)

The first monotonic-alignment Gator-S run changed character around steps
12k-16k. From step 1k to 16k, validation aligned loss moved 3.1950 to 2.8619,
required-token-normalized aligned loss moved 4.1869 to 3.7459, aligned content
accuracy moved 1.5% to 8.5%, content F1 moved 3.3% to 14.5%, and edit accuracy
moved 28.1% to 31.0%. These metrics improving together is evidence of weak
content learning rather than BLANK placement alone. Exact response accuracy
remained zero, outputs remained under-length, and EOS supervision was sparse,
so this is a promising partial curve rather than a solved language result.

Two new training-only corpora are available under
`MyDrive/Sonic/training_data/chat`:

- `tulu3_sft.jsonl`
- `smoltalk_preference.jsonl`

Both use the repository's existing flat `text`/`source` schema with
`USER:`/`ASSISTANT:` headers. An audit of the first 20,000 rows in each corpus
found 100% parser success. Tulu3 contained 22,660 assistant examples, 90.0% of
rows passed the 0.9 ASCII-letter gate, and 29.2% of complete transcripts were
at most 512 characters. SmolTalk Preferences contained 19,996 assistant
examples, 97.2% passed the language gate, and 11.1% of complete transcripts
were at most 512 characters. The 512-character policy therefore discards most
long rows but still leaves a large, clean short-context population.

For continuation, keep the original validation set fixed. Draw 10% from finite
SmolTalk Everyday and 90% from the streamed pool; split the streamed pool
LMSYS:Tulu3:SmolTalk-Preferences at relative weights 4:3:2. Resume from the
best checkpoint, preserve Adam moments, restart a 300k-step cosine schedule at
LR 0.001 with minimum LR 0.0001, and do not replay alignment or stewardship
warmups. At the observed 12.2 steps/s this is approximately 6.8 hours before
startup and evaluation overhead.

Do not reinterpret the diversified continuation as a clean corpus A/B. Its
purpose is an overnight capability probe: determine whether the emerging
aligned-content curve continues when repetitive exposure is reduced. The next
controlled experiment should still isolate complete-response filtering and
alignment shift separately.

## Superseded Gator-Path split-optimizer probe (2026-07-17)

The experimental split optimizer used zero-momentum SGD for cell/node route
heads and AdamW for token transformations. Its fixed-rate prefix runs
plateaued.
Route LR `0.01` stabilized final orchestrator ownership near 50%. A single
`0.1` run reached roughly 75%, but repeated attempts did not reproduce it;
treat that run as an outlier.

The graduated schedule later drove final orchestrator ownership to 99.9%
without improving prefix accuracy or loss. Removing the return loss allowed
weak task learning while ownership fell again, proving that final location was
not a capability metric. This line is superseded.

The restored original-vision protocol is:

1. Only the packet's current location/payload frontier survives each relay.
2. Learned ingress begins dense and anneals toward a deterministic first
   steward.
3. Each source cell learns only its local transformation and next handoff.
4. Broad local alternatives receive real forward packet mass early in
   training, not merely surrogate backward gradients.
5. Each relay discards its previous handoff after committing the next current
   frontier.
6. Routing and transforms use one AdamW optimizer and ordinary task loss.
7. No ownership loss and no learned/stored complete route.
8. Route histories are diagnostic-only and are not retained in ordinary
   training.

The route `A -> B -> C -> D` is the rollout of four local cell policies. It is
never a model parameter or memory object.

### Hard-only local-policy falsification

The first original-vision implementation committed a hard steward from step
zero and exposed other local choices only through a soft surrogate gradient.
Its seed-0 prefix run stayed at chance and finished step 5,000 at 8.6%
accuracy and loss 2.4858. This is a material regression against all three
historical ten-cell Arrow seeds:

| Seed | Arrow accuracy at step 5,000 | Arrow final accuracy |
|---:|---:|---:|
| 0 | 68.3% | 99.8% |
| 1 | 51.0% | 91.3% |
| 2 | 32.7% | 41.0% |

The result rules out a seed-only explanation. A dense backward surrogate is
not a dense start: unused cells cannot acquire useful downstream behavior if
they never actually receive packet state. The current `frontier_policy`
implementation therefore carries a soft current distribution `q` and
location-specific payloads through the forward pass. It still stores no route
sequence. The required validation set is seeds 0, 1, and 2 with the historical
batch, task, relay depth, optimizer, and 5,000-step checkpoints held fixed.

### Local-frontier seed-0 result

The corrected 8,596-parameter current-frontier model used one AdamW optimizer,
ordinary prefix CE, batch 128, four recurrent relays, and the historical
temperature schedule. Seed 0 produced:

| Step | Accuracy | Loss | Route entropy | Ingress entropy | Active locations / 40 |
|---:|---:|---:|---:|---:|---:|
| 500 | 8.7% | 2.4885 | 0.643 | 2.237 | 38.4 |
| 1,000 | 9.0% | 2.4864 | 0.483 | 2.181 | 37.6 |
| 1,500 | 7.7% | 2.4874 | 0.375 | 1.958 | 33.1 |
| 2,000 | 9.2% | 2.4802 | 0.128 | 1.752 | 21.4 |
| 2,500 | 9.5% | 2.4002 | 0.040 | 1.466 | 10.5 |
| 3,000 | 10.2% | 2.3843 | 0.101 | 1.411 | 8.3 |
| 3,500 | 10.0% | 2.3797 | 0.050 | 1.516 | 9.5 |
| 4,000 | 12.5% | 2.2868 | 0.032 | 1.715 | 8.8 |
| 4,500 | 15.7% | 2.2213 | 0.026 | 1.794 | 8.2 |
| 5,000 | 18.7% | 2.1477 | 0.027 | 1.813 | 8.4 |

This restores learnability but not the validated Arrow learning rate: Arrow
seed 0 reached 68.3% at step 5,000. It does decisively beat the hard-only Path
seed 0, which remained at 8.6% and loss 2.4858 at the same budget.

The path dynamics are more important than monotonic pruning. The model narrowed
from 40 locations to 8.3 by step 3,000, then widened ingress and the effective
frontier when progress slowed. Loss broke again at step 4,000 while the
downstream frontier stayed sparse. Ordinary task gradients therefore produced
the intended behavior: broad initialization, familiar local handoffs,
reopening under uncertainty, and renewed narrowing. This is effective support
inside the dense correctness engine, not a measured kernel-level speedup.

### Local-frontier three-seed result

Seeds 1 and 2 used the identical architecture, generated task, batch size,
optimizer, temperature schedule, evaluation set, and 5,000-step budget. The
locked final comparison is:

| Seed | Accuracy at 3k | Accuracy at 5k | Loss at 5k | Effective ingress cells | Ingress confidence | Route movement | Active locations / 40 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.2% | 18.7% | 2.1477 | 6.13 | 73.4% | 59.6% | 8.4 |
| 1 | 26.7% | 29.4% | 1.9478 | 5.11 | 84.8% | 56.8% | 8.8 |
| 2 | 12.5% | 13.5% | 2.1997 | 4.19 | 92.1% | 94.1% | 8.3 |
| Mean | 16.5% | 20.5% | 2.0984 | 5.14 | 83.4% | 70.2% | 8.5 |

For comparison, the historical 10,000-parameter Arrow seeds reached 68.3%,
51.0%, and 32.7% at step 5,000, a 50.7% mean. The 8,596-parameter Path
frontier therefore passes the learnability and routing-behavior gates but
fails the capability-parity gate at this budget.

All three Path seeds left the random-loss floor. This is materially different
from the hard-forward/soft-backward seed-0 control, which stayed at chance.
The three runs also establish:

1. Learned ingress probabilities become sharp without an ownership loss.
   Final per-packet top-1 confidence ranged from 73.4% to 92.1%.
2. Sharp ingress does not imply one universal first cell. The marginal
   population still used roughly four to six effective cells, consistent with
   packet-dependent specialization.
3. Soft confidence is not discrete readiness or correctness. Seed 2 had the
   highest ingress confidence and lowest effective cell count but the weakest
   final accuracy.
4. Useful local programs can have different relay rhythms. Seed 1 moved on
   56.8% of relay decisions; seed 2 moved on 94.1%.
5. Exploration can reopen. Seed 1 narrowed to 3.81 effective ingress cells at
   step 2,000, widened to 5.27 at step 2,500, and improved from 17.7% to 22.9%
   accuracy over the same interval.
6. The final effective frontier was consistently narrow: 8.3--8.8 of 40
   cell/node locations. This is the target for a later packed implementation,
   not proof that the dense reference already skips those kernels.

The immediate clean capability A/B is to restore exact parameter parity with
Arrow by adding local cell capacity without changing routing semantics. After
that, test receiver-specific relative slot reads. Temperature, novelty, or
stall-triggered reopening policies should be isolated separately; they must
not be mixed into the parameter-parity run.

### Hard-route audit

The final soft-route checkpoints were evaluated on the same deterministic
1,024-example set while hardening ingress and downstream handoffs separately:

| Seed | Routing mode | Accuracy | Loss |
|---:|---|---:|---:|
| 0 | soft ingress / soft handoffs | 18.7% | 2.1477 |
| 0 | hard ingress / soft handoffs | 9.7% | 3.4245 |
| 0 | soft ingress / hard handoffs | 18.6% | 2.1537 |
| 0 | hard ingress / hard handoffs | 10.0% | 3.4973 |
| 1 | soft ingress / soft handoffs | 29.4% | 1.9478 |
| 1 | hard ingress / soft handoffs | 9.0% | 5.1811 |
| 1 | soft ingress / hard handoffs | 20.5% | 2.5233 |
| 1 | hard ingress / hard handoffs | 9.6% | 5.4723 |
| 2 | soft ingress / soft handoffs | 13.5% | 2.1997 |
| 2 | hard ingress / soft handoffs | 11.9% | 2.4283 |
| 2 | soft ingress / hard handoffs | 13.6% | 2.2015 |
| 2 | hard ingress / hard handoffs | 11.9% | 2.6372 |

The main discretization failure is first-steward selection, not local
handoffs. Hard downstream choices preserve seeds 0 and 2 almost exactly;
seed 1 degrades but remains above chance. Hard ingress collapses seeds 0 and 1
near chance and weakens seed 2. The residual non-top-1 ingress mass is therefore
still carrying essential computation even when reported top-1 confidence is
high. This rejects deterministic first stewardship as a universal design
requirement. The retained ingress frontier must be selected by end-task
quality; top-1 is only a successful endpoint when the packet has actually
consolidated its complementary computation there.

The API and trainer expose `hard_ingress` / `--hard-ingress` independently from
`hard_handoffs` / `--hard-handoffs` so hardening curricula can be tested
without conflating the two failure modes. The first follow-up resumes the best
soft checkpoint with the same AdamW state and hardens ingress only.

That abrupt-hardening follow-up failed. Seed 1 resumed from step 5,000 with
hard ingress, soft handoffs, the exact AdamW moments, and no new objective:

| Step | Accuracy | Loss | Effective ingress cells | Ingress confidence | Active locations |
|---:|---:|---:|---:|---:|---:|
| 5,000, before adaptation | 9.0% | 5.1811 | 4.94 | 100% | 1.6 |
| 5,500 | 8.7% | 2.5346 | 2.98 | 100% | 1.6 |
| 6,000 | 8.9% | 2.5068 | 3.73 | 100% | 1.5 |

The optimizer rapidly repaired catastrophic logits but did not recover prefix
capability within 1,000 updates. Pre-clip gradient norm peaked at 15.90 during
adaptation. Sudden argmax commitment is therefore rejected for this
checkpoint. The next hardening experiment must gradually transfer cooperative
soft-ingress computation into selected stewards, for example through
ingress-only temperature annealing or a scheduled soft/hard mixture. This
should be tested before hardening downstream handoffs, which are already much
closer to discrete equivalence.

### Arrow-capacity restoration matrix

The 8,596-parameter frontier control was not an exact architectural-capacity
comparison with the validated stitched Arrow. It shared one token table across
all ten cells and used a single sequence-averaged mailbox at each cell/node.
The original Arrow gave each cell private tied token I/O and used
receiver-specific relative-position reads.

The Path trainer now exposes those two corrections independently:

| Command additions | Parameters | Controlled difference |
|---|---:|---|
| `--no-private-cell-io --no-relation-aware` | 8,596 | Historical frontier |
| `--private-cell-io --no-relation-aware` | 9,748 | Cell capacity only |
| `--no-private-cell-io --relation-aware` | 8,756 | Relative reads only |
| `--private-cell-io --relation-aware` | 9,908 | Restored near-parity model |

Everything else remains fixed: ten cells, four nodes, four recurrent relays,
five bounded local cell actions, learned soft ingress, current-frontier
handoffs, one AdamW optimizer, cross-entropy only, batch 128, LR 0.008, and the
historical temperature schedule. The trainer defaults to the 9,908-parameter
combined candidate.

Run the A/Bs one process at a time on the local laptop. A first attempt to run
multiple seeds concurrently exhausted host RAM because each process owned an
independent dense current frontier and 1,024-example diagnostic forward. Before
each launch:

1. confirm no Python trainer is already active;
2. check available physical RAM;
3. launch only one seed;
4. retain the same training batch for trajectory parity; and
5. reduce only the intermediate evaluation batch if peak diagnostic memory is
   unsafe, then perform the locked final evaluation separately.

The capability gate remains the historical step-5,000 three-seed mean:

```text
Arrow seeds 0/1/2: 68.3%, 51.0%, 32.7%
Arrow mean:        50.7%
```

Do not declare performance restored from a finite smoke, one favorable seed,
or a lower-loss chance solution. The new model must be evaluated sequentially
on seeds 0, 1, and 2 and compared at the same update budget.

### Minimal Arrow-Carry A/B

`gator_cells_vectorized.py --carry-state` is the narrowest control for the
state-carry hypothesis. It uses the validated Arrow implementation and the
same 10,000 parameters. The corrected mode preserves Arrow's destination-owned
acceptance while carrying already-projected source messages:

```text
Arrow:
destination gathers neighbors
        -> destination gate
        -> next destination state

Arrow-Carry:
source projects and offers its updated state
        -> destination gate
        -> destination commits accepted carried state
```

Projection commutes with the ring average, so ordinary Arrow and corrected
Arrow-Carry are algebraically equivalent. Their forward values, every
parameter gradient, checkpoint keys, and shapes match within floating-point
tolerance. Run the matched seed controls sequentially:

```powershell
python -u gator_cells_vectorized.py --steps 5000 --seed 0 --neighbors 4
python -u gator_cells_vectorized.py --steps 5000 --seed 0 --neighbors 4 --carry-state
```

The historical rejected source-owned variant is now named explicitly:

```powershell
python -u gator_cells_vectorized.py --steps 5000 --seed 0 --neighbors 4 --source-carry
```

Do not use `--source-carry` as the production carry baseline. It asks the
source to own an outgoing gate, changes Arrow's credit assignment, and reduces
the preserved mastered checkpoint from 99.665% to 10.659% accuracy.

Do not run those commands simultaneously on the laptop. The validated Arrow
seed-0 checkpoints already supply the reference curve, so only the carry run
is required unless implementation equivalence is being re-audited.

### AdamW epsilon and trajectory reproducibility

Arrow's acquisition curve is sensitive to numerical execution order. In a
lockstep OG-versus-destination-carry run, the initial logits differed by only
`7.75e-7` and every gradient agreed to about `6e-8` or better. Nevertheless,
default AdamW epsilon `1e-8` produced a `1.32e-5` maximum parameter difference
after the first update. Adam's normalization can turn a tiny gradient on a
previously quiet parameter into an almost sign-sized update; numerical dust
therefore becomes a real optimizer trajectory fork.

The new `--optimizer-epsilon` argument exposes this stability lever. At the
same batch and learning rate:

| AdamW epsilon | Max parameter difference after step 1 | After step 25 |
|---:|---:|---:|
| `1e-8` | `1.32e-5` | `1.31e-5` |
| `1e-6` | `1.28e-6` | `1.30e-6` |
| `1e-4` | `1.19e-7` | `2.38e-7` |

This is a numerical-equivalence diagnostic, not yet a learning-quality result.
A larger epsilon may suppress backend noise but can also delay legitimate
small-gradient specialization. Before changing the training default, run
full seed-0/1/2 curves at `1e-6` and `1e-4`, preserve the same batch stream,
and compare phase-transition timing plus final mastery. Always log optimizer
epsilon with backend, device, compilation mode, and seed.

Use `--portable-batches` for those comparisons. Without it, model
initialization consumes the CPU RNG while CUDA training batches use a separate
CUDA RNG; nominal seed 0 therefore does not mean identical examples across
devices. Portable mode samples through a dedicated CPU generator, transfers
the tiny synthetic batch to the target device, and saves that generator state
in checkpoints. Resumed runs continue at the next batch instead of silently
replaying the stream.

### Seed-0 restoration and throughput results

The private-I/O-only Path control finished first:

| Step | Shared-I/O Path | Private-I/O Path | Original Arrow |
|---:|---:|---:|---:|
| 2,000 | 9.2% | 14.0% | 12.3% |
| 3,000 | 10.2% | 18.7% | 37.9% |
| 4,000 | 12.5% | 20.8% | 55.4% |
| 5,000 | 18.7% | 21.6% | 68.3% |

Private cell I/O is a real capacity lever, but it does not restore Arrow's
learning curve. The run used 9,748 parameters, finished at loss 2.1335 and
1.51/7 correct, and required 3,107.28 seconds at 1.61 steps/second.

The parameter-identical **source-owned** Arrow-Carry diagnostic was
substantially stronger than earlier Path rewrites:

| Step | Original Arrow seed 0 | Arrow-Carry seed 0 |
|---:|---:|---:|
| 1,000 | 8.7% | 8.5% |
| 2,000 | 12.3% | **24.3%** |
| 3,000 | 37.9% | **42.4%** |
| 4,000 | **55.4%** | 47.8% |
| 5,000 | **68.3%** | 52.8% |

Source-owned carry restored the phase-transition regime, led the reference at
2k and 3k, then learned more slowly from 3k through 5k. Its final deterministic
evaluation was 52.7%, 3.69/7 correct, exact 0.1%, and loss 1.2395. This is the
strongest result among the rejected responsibility-changing Path controls,
but it does not meet the 68.3% seed-0 capability gate. The corrected
destination-owned carry is the production baseline and preserves the mastered
checkpoint at 99.665%.

Matched CPU throughput disproved a speedup from coupling direction alone:

| Cells | Parameters | Original Arrow | Dense Arrow-Carry |
|---:|---:|---:|---:|
| 10 | 10,000 | 2.88 steps/s | 2.89 steps/s |
| 62 | 61,168 | 0.57 steps/s | 0.57 steps/s |

Both dense modes still execute every cell, rebuild every receiver-specific
node memory, and run every node transform. Moving a linear projection across
the neighbor gather does not change asymptotic cost.

### Persistent node-memory milestone

`gator_path/persistent_memory.py` introduces carried receiver-specific memory
sufficient statistics:

```text
numerator[b, receiver, cell, node, width]
denominator[b, receiver, cell, node]
memory = numerator / denominator
```

A packet departure subtracts its old relative-weighted payload; its arrival
adds the transformed payload at the new cell/node. The update materializes
`[batch, receiver, changed_packet]` weights rather than
`[batch, receiver, writer, cell, node]` weights. Tests establish:

1. initial carried memory equals a fresh dense relative-memory rebuild;
2. departure/arrival updates equal rebuilding the final packet population;
3. inactive packets leave state unchanged; and
4. gradients through payloads and relative biases remain finite and nonzero.

This is the first required primitive for the actual speed-producing model.
The next integration must pair it with active-only node transforms and packed
packet ownership. Until dormant cells are skipped, carried memory is a
correctness feature rather than a measured training speedup.

## Carry-forward restoration verdict (2026-07-19)

The current reproducible restoration is the 10,010-parameter Convergent
configuration: one 40-node shared graph, ten persistent width-eight payload
lanes, four soft recurrent relays, and continuous terminal fusion. It reached
98.2% token / 88.5% exact at step 30k under constant learning rate `0.008`.
Resuming the preserved best model, AdamW state, and portable batch RNG at
learning rate `0.002` reached 99.2% / 95.2% after 1,000 cooldown updates and
99.3% / 95.6% after 2,000. A third checkpoint improved to 99.4% / 95.9%
with `0.0223` loss. Three consecutive evaluations now confirm the restored
basin rather than a single mastery spike.

Use this training sequence for the prefix control:

1. Train four relays at LR `0.008`, preserving a best checkpoint.
2. Do not interpret late high-LR oscillation as an architectural ceiling.
3. Once token accuracy enters the high-97/98% basin, resume the best complete
   state at LR `0.002`.
4. Require at least two consecutive evaluations above 99% token accuracy
   before calling mastery stable.
5. Evaluate exact-sequence accuracy and per-position error as well as token
   accuracy; the remaining errors concentrate at longer prefix distances.

The locked step-32k diagnostic scored `99.29%` token / `95.68%` exact on
4,096 examples. Three terminal lanes carried effectively all useful fusion
weight. Their best one-lane, two-lane, and three-lane accuracies were `33.25%`,
`82.65%`, and `99.27%`. Do not prune this model to the single highest-weight
lane: the apparent specialists are complementary voters. A true trained-hop
prefix ablation scored `9.61%`, `12.19%`, `34.59%`, and `99.35%` after one
through four relays, so inference must execute the complete four-hop program.

The depth controls show why the four relays should remain the baseline:

| Native depth | Tested runway | Best/final conclusion |
|---:|---:|---|
| R=1 | 30k | Approximately 92% ceiling; broad independent lane coalition |
| R=2 | 20k | Fastest early learning; approximately 92% ceiling |
| R=3 | 20k | Consumes cross-lane contact but lacks a commitment stage |
| R=4 | 32k | Reaches and holds >99% token mastery after cooldown |
| R=8 | 10k stopped | Learns, but slower credit assignment and much higher cost |

Do not return to source-owned carry, hard top-1 ingress, explicit complete
route memory, or uniform scaling of stitched cells as the default. The
evidence says the useful computation is a cooperative soft trace that
narrows naturally, communicates after sufficient recurrent depth, and then
commits. Sparse execution should be introduced only after counterfactual
top-k tests show that removing low-mass branches is safe.

### Connection-layout scaling controls

Use explicit connections rather than increasing a ring until it becomes
unreachable. The equal-parameter topology controls are:

```powershell
# Historical geometry.
python -u gator_convergent.py --connection-layout ring ...

# Preserve self and a directed backbone; replace a fraction of remaining
# local edges with deterministic shortcuts.
python -u gator_convergent.py `
  --connection-layout small_world `
  --connection-seed 0 `
  --connection-rewire 0.25 ...
```

Run the static diagnostic before spending a training budget:

```powershell
python analyze_gator_topology.py `
  --nodes 40 --degree 3 --lanes 10 --relays 4 `
  --layouts ring small_world `
  --connection-seeds 0 `
  --rewire-probabilities 0.1 0.25 0.5 1.0
```

Record both reach and collision. A layout that shortens graph diameter but
causes every lane to overlap before it has formed a distinct hypothesis may
train worse. Conversely, a layout with no cross-lane contact before readout
cannot perform recurrent reconciliation. The target is staged connectivity:
local formation first, material overlap before the penultimate relay, and a
final commitment relay afterward.

Before scaling node count, enforce this gate:

```text
some lane overlap must exist in q^(R-1)
so relay R can consume it
```

For example, an 80-node, ten-lane, degree-three ring has no lane overlap
through \(q^3\) at four relays. Its terminal \(q^4\) overlap is unread and
cannot support recurrent coordination. A 25%-rewired graph restores contact
by \(q^2\) and gives relay four meaningful cross-trace traffic to reconcile.
Reject any scaled topology that violates this timing invariant, regardless of
its nominal parameter count.

For evenly spaced lanes on a degree-three ring, check the quick bound:

\[
N\leq 2K(R-1).
\]

If it fails, do not train that ring as a scaling candidate. Either add
long-range connections, increase lanes, or increase relays first.

Treat relay depth and connection reach as one communication budget. The first
factorial probe found:

```text
R4 / 25% rewired: 79.8% token at 5k, 3.47 steps/s
R4 / 50% rewired: 38.6% token at 5k, 3.43 steps/s
R2 / 50% rewired: 62.3% token at 5k, 10.92 steps/s
```

Do not maximize both knobs independently. Higher connectivity grows the
reachable frontier; later relays then execute increasingly large router and
node-transform sets. Sweep the two-dimensional topology/depth surface and
compare both accuracy per update and accuracy per wall-clock second.

The first compute-matched result selects R2/50% as the operational speed
winner:

```text
R2 / 50%, 16k: 98.0% token, 87.1% exact, 1,494.8 seconds
R4 / 25%,  5k: 79.8% token, 21.2% exact, 1,440.6 seconds
```

Use R2/50% for the next scaling and baseline comparisons. Preserve R4/25% as
the deeper control, not the default. The R2 result is not yet a 99% mastery
checkpoint; a later low-LR cooldown can test whether its remaining two token
points are optimization noise or a true shallow-depth ceiling.

The matched 10k one-shot Transformer control establishes why raw throughput
must not be used alone:

```text
R2 / 50% Gator, 16k: 98.0% token, 87.1% exact, loss 0.0691, 10.61 steps/s
4-layer Transformer, 16k: 8.5% token, 0.0% exact, loss 2.4852, 34.68 steps/s
```

The Transformer was 3.27 times faster per update but remained at chance (8.8%
best observed accuracy). Historical screens found that even roughly 70k
Transformer parameters needed more than 20k updates, at which point scaling
had consumed the apparent speed advantage. Rank models by wall time to the
same held-out accuracy or loss threshold. If a model never reaches the
threshold, record it as `not reached`; do not extrapolate its update rate into
a capability claim.

### First 130k connected-language run

Use the tied-vocabulary R2/50% configuration:

```powershell
python -u gator_language_train.py `
  --data <smoltalk_everyday.jsonl> `
  --architecture convergent `
  --steps 10000 --batch-size 32 --eval-batch-size 64 `
  --log-every 500 --logging-slim `
  --vocab-size 2048 --context-tokens 64 --response-tokens 32 `
  --embedding-dim 32 --d-model 20 `
  --nodes 66 --graph-degree 3 --lanes 10 --relays 2 `
  --connection-layout small_world `
  --connection-seed 0 --connection-rewire 0.5 `
  --skip-first-assistant `
  --learning-rate 0.003 --scheduler cosine `
  --scheduler-total-steps 50000 --minimum-lr 0.0001 `
  --seed 0 --batch-pipeline serial
```

`--scheduler-total-steps` separates the learning-rate horizon from the update
count executed by this invocation. This pilot stops after 10k updates but uses
the same 50k cosine phase as the historical Arrow run, preventing an unfair
decay to minimum LR at the comparison point.

Compare against the historical 124,616-parameter Arrow step-10k checkpoint:
PPL `43.72`, token accuracy `32.5%`, exact `0.0%`. Also report content-only
accuracy/PPL, formation coverage, aligned metrics, displayed response, and
steps/second. Raw accuracy is not sufficient because whitespace was an easy
class in that older run.

Passing the 10k comparison is necessary but not sufficient. The asymptotic
gate is PPL below `33` on the same evaluation corpus, because the later 130k
Arrow program saturated near that level after approximately 30--40k updates.
Require the connected model to cross that floor sooner or produce materially
better content/context metrics at equal loss. A curve that wins at 10k but
converges above 33 is an optimization-speed improvement, not a capability
improvement.

Do not scale lanes independently of graph workspace. For degree three and R2,
the measured per-lane frontier is approximately six nodes. Preserve the
validated frontier density with

\[
\frac{\mathrm{lanes}\times6}{\mathrm{nodes}}\approx0.9,
\qquad
\mathrm{nodes}\approx6.6\times\mathrm{lanes}.
\]

L10/N66 satisfies this relation. L20/N66 doubles the density to 1.82, covers
64/66 nodes by R2, and approximately doubles the colliding peers seen by each
lane; use it as an overcrowding ablation. The ratio-preserving scale is
L20/N132 (192,820 parameters at width 20). If the budget must remain near
154k, use L14/N90 (153,838 parameters) as the balanced control. Compare
widths only after holding this topology density approximately fixed.

The completed L20/N66 overcrowding control confirms why this distinction
matters. On a compiled RTX PRO 6000 Blackwell run, it reached PPL 38.47 at
10k, 35.53 at 20k, 34.07 at 39k, and first crossed below 33 only at 82k. Its
best value was 32.93 at 99k; it was back to 33.32 when interrupted at 107k.
The 80k--100k mean was 33.22 and exact response accuracy remained zero.
Record this as a match to the old Arrow floor, not a meaningful break through
it. Do not spend additional long-run budget on L20/N66 before running L14/N90
or L20/N132.

This run used an older bundle and a 939-example validation split; the later
local bundle reports 898 validation examples. Treat the PPL-33 comparison as
a nominal gate until every candidate checkpoint is reevaluated with one
frozen tokenizer and validation manifest. Never infer a sub-point PPL win
across different manifests.

The active near-balanced follow-up is L16/N90 (158,600 parameters). Its static
R2 graph covers 68/90 nodes, collides 19.2% of lane pairs, and gives each lane
about 2.88 colliding peers. Compare its milestones directly with L20/N66:
PPL 42.65 at 4k, 39.99 at 7k, 38.47 at 10k, 35.53 at 20k, 34.74 at 30k,
and first below 33 at 82k.

Do not assume that the most exact node/lane ratio must win. For L16/N106, the
heuristic frontier density is almost exactly 0.91, but static seed-zero
geometry gives only about 1.62 colliding peers per lane. L16/N90 gives 2.88,
close to the validated L10/N66 value of 2.6. The more useful invariant may be
consumed peer contact plus the learned terminal coalition, rather than raw
frontier density alone.

After each candidate saves a checkpoint, run:

```powershell
python analyze_convergent_votes.py <best.pt>
```

Record entropy-effective voters, L2-effective voters, top-1/3/5 vote mass,
and the number of voters needed for 90/95/99% mass. These are cheap structural
metrics. A final claim still requires top-\(k\) and leave-one-voter-out
counterfactual accuracy on the frozen validation manifest because vote mass
does not prove functional contribution.

If L16/N90 also approaches PPL 33, do not immediately increase raw population.
Every major Gator language scale retained a width-32 lexical interface and a
single global terminal vote vector. Run the following isolation order:

1. L16/N90, width 32, global votes (control).
2. The same model with prompt-conditioned terminal voting.
3. The same model with prompt-and-slot-conditioned terminal voting.
4. The same graph and global votes with lexical width 64.
5. Slot-conditioned votes plus lexical width 64 only if either isolated
   intervention moves content likelihood or formation quality.

Use the exact same validation fingerprint. Gate on content-only PPL, top-k
target accuracy, formation coverage, formed length, and semantic probes as
well as raw PPL. The 130k one-shot Transformer reached raw PPL 31.45 while
remaining qualitatively generic, so merely crossing 33 is not mastery.

The prompt-conditioned condition is available as:

```text
--architecture convergent --terminal-vote-mode prompt
```

At L16/N90/D20 it has 158,920 parameters versus 158,600 for global voting.
The query starts at zero, so initialization is an exact forward control.

The slot-conditioned condition is available as:

```text
--architecture convergent --terminal-vote-mode slot
```

The current direct-address implementation has 158,984 parameters at
L16/N90/D20: 64 more than prompt voting. It produces weights `[B,O,K]`, one
coalition over lanes for each prompt and immutable output address. A private
four-value key per lane contributes an output-relative fixed address directly
to the vote logits. Output coordinates span the current formation rather than
the combined 512+output window. This bypasses mutable relay state and starts
slot juries distinct instead of waiting for a zero-initialized shared query to
rediscover position.
All parameters shared with the old control retain the same seeded
initialization.

The earlier indirect slot implementation used only the width-D query and did
start as exact global voting. In the D20 chunk-eight run it remained
functionally global through 50k updates: `slot_delta` stayed below 0.0005 and
one lane won every output position while the model emitted whitespace. That
result motivated the direct address path. Inspection also found that the old
combined-window sinusoid separated adjacent C512/O8 outputs by only
approximately `pi/519`; it was mathematically slot-aware but numerically close
to one address.

For D30/N66/L10, the legacy indirect slot control is 186,682 parameters and
the current direct-address slot control is 186,722, versus 186,382 for global
control. Use the existing seed-zero D30/global run as the baseline and keep
the same seed, corpus weights, validation fingerprint, scheduler, and
preprocessing. Optional address keys initialize only after every historical
shared tensor, preserving the global control's initialization sequence.

The active compute-width control is L10/N66/D30 with lexical width 32 and
global voting. It has 186,382 parameters and preserves graph geometry exactly.
If its early advantage still flattens near PPL 33, width improves acquisition
but not the terminal bottleneck. If it sustains a break below 33 with improved
content metrics, payload/node width was the limiting parameter dimension.

At 50,000 updates its raw curve was effectively tied with L20/N66/D20:
PPL `34.27` versus `34.25`. Its sparse printed samples were longer and more
context-shaped, but the examples were not matched. Continue this run through
the 300,000-step scheduler horizon before declaring a plateau. Compare saved
best checkpoints with `language_probe.py` so every model receives the same
prompts; random log samples are useful smoke evidence, not an A/B.

The intermediate D30 best at step 125,000 reported PPL `32.9427`. After a
temporary step-155,000 regression to `33.08`, step 195,000 established the
final best at PPL `32.8530`, loss `3.49204`, token accuracy `30.80%`, content
PPL `81.30`, and formation coverage `14.24%`. This was only `0.0589` PPL
behind the frozen Arrow best at step 242,000. Step 285,000 had regressed to
PPL `33.1720` even though formation coverage rose to `14.95%`; D30/global
therefore confirmed the approximately PPL-33 floor. Always compare the saved
best, not the most recent weights.

The control's original Drive directory was later reused by mistake. The
rescued authoritative checkpoint is
`ARCHIVE_global_d30_n66_l10_r2_rw50_seed0_best_step195k/`
`best_step195000_global.pt`, SHA-256
`21df0d5e7b431e52eac8b075e0b352cf835a9582ac9977318230c61582169b69`.
Before an unattended formation A/B, require all three startup checks:

```text
run_name contains formation_slot
parameters=186722
terminal_vote_mode=slot
```

Any banner reporting `186382` parameters or `terminal_vote_mode=global` is
the concluded global control. A `186682` banner identifies the legacy
indirect slot voter; `186722` identifies the direct-address voter.

For prompt/slot voting, inspect the `terminal` log line beside PPL:

```text
terminal effective=... top1=... slot_delta=... winning_lanes=... query_rms=...
```

`slot_delta=0` means every position is still using the same coalition.
Increasing delta and more than one winning lane show that the controller is
actually assigning different lane specialists to different output addresses.
Those structural changes are necessary evidence, not sufficient capability:
they must coincide with better content PPL, coverage, fixed prompts, and
formed length.

If the completed D30 run still clusters near PPL 33, run the D30 global versus
slot-conditioned formation-control A/B before the next capacity control.
The capacity control remains D50/N66/L10/K3/R2 with 50% rewiring, lexical
width 32, and the winning terminal-vote mode.
It has exactly 350,342 trainable parameters. D54 has 391,390 parameters and is
the held-back follow-up. Do not change node count, lane count, relay depth, or
corpus in the D50 run: after terminal mode is selected, the intervention is
payload/node width alone.

For this fixed topology, exact parameter count is
\(P(D)=86D^2+1318D+69442\). Subtract the 67,584-parameter tied lexical shell
when discussing computational capacity: D30 has 118,798 non-lexical
parameters, D50 has 282,758, and D54 has 323,806. The D50 run therefore gives
the graph-and-lane computation approximately 2.38x the D30 capacity while
keeping the same simultaneous 32-slot objective.

Treat prefetch depth as a training pipeline control, not an inference
optimization. Benchmark inference independently with gradients disabled and
report batch-one latency, formed tokens per second, context/output lengths,
relay count, and peak VRAM.

## Accuracy-gated teacher-to-free handoff

The first causal D400 control ended at raw PPL `11.2238`, content PPL
`22.7513`, and teacher-forced content accuracy `22.88%`. It proved that the
graph can learn conditional language distributions, but it remained far below
the proposed handoff threshold and its concatenated teacher-forced argmaxes
were not a valid free-running generation.

Use `--decode-mode handoff` to make causal support contingent on demonstrated
language mastery:

```text
teacher content accuracy <= 60%: 100% teacher batches
teacher content accuracy  = 75%:  50% teacher batches
teacher content accuracy >= 90%:   0% teacher batches
```

The controller uses a validation-accuracy EMA with default decay `0.50`.
Complete batches alternate through deterministic error diffusion. Never
partially reveal target slots during a noncausal forward: receiver memory and
traffic would let blank slots inspect future answers.

At each checkpoint, preserve three distinct artifacts:

- `best_teacher.pt`: strongest causal conditional-language state;
- `best.pt`: strongest blank-parallel formation state;
- `latest.pt`: exact controller EMA, mixture credit, and batch counts.

The schedule is reversible. If parallel batches damage teacher-content
accuracy, causal support increases again. This is intentional: handoff is a
feedback controller, not an irreversible wall-clock anneal.

For the historical language gate, use the step-242k Arrow checkpoint with
SHA-256
`6172e2c0761ecffdff3d59576559ca7b8cd7248e7659b545cb8b8b3ac74bc652`.
It has exactly 124,616 trainable parameters and saved raw PPL `32.7941`.
Probe season, cat/meow, `2 + 2`, rental cars, team sports, and beginner
running in that order. This prevents later aligned-objective checkpoints or
random printed samples from silently changing the baseline.

Do not extend the SmolTalk-only pilot after it has traversed the small finite
corpus many times. In the first Convergent run, the step-6,000 best PPL was
`42.24`; later steps oscillated upward after several dozen
dataset-equivalents. With 5,726 active examples and batch 32, step 6,000 was
already about 34 dataset-equivalents. Preserve the best checkpoint and switch
to the diverse stream rather than treating repetition-driven regression as a
model floor.

The current long-run protocol reproduces the established language-bundle
corpus as four streams: SmolTalk Everyday, LMSYS, Tulu3, and SmolTalk
Preferences weighted `1:4:3:2`. Normalized context/response pairs are
deduplicated within each traversal. Epoch-safe restart fills a shuffle buffer
from only one traversal, drains every buffered example once, and only then
starts another epoch; it therefore preserves the fixed corpus proportions
without the former duplicate-filled reservoir. The strict `drop` mode remains
a one-pass ablation, but permanently removing small sources changes the
training distribution and caused PPL to rise from `42.69` to `47.54` after
both small sources left. Resume the
step-6,000 model and optimizer moments, reset sampler state because the corpus
changed, use LR `0.001` with a 300k cosine horizon, and report every 15
minutes. Promote to the Blackwell 300k experiment only if the laptop run
meaningfully crosses PPL `33` while retaining contextual content.

### TinyStories-first language curriculum

The accuracy-gated handoff experiment now begins its corpus schedule with
TinyStories. This is a data curriculum change, not an architecture change.
TinyStories receives exactly 40% of the named training stream. The prior
SmolTalk Everyday/LMSYS/Tulu3/SmolTalk Preferences balance remains `1:4:3:2`
inside the remaining 60%, giving final shares `40:6:24:18:12`.

TinyStories supplies direct short-story prefix-to-continuation structure before
the noisier dialogue sources. It is also the first source used to fit a new
tokenizer. The official raw text must be split on `<|endoftext|>`; line breaks
are internal formatting, not independent stories.

Prefix construction scales with the formation window:

- context 64 exposes the first two-sentence prefix;
- context 128 additionally exposes the first four-sentence prefix;
- each further 64 context slots adds one more sentence-pair boundary;
- response capacity contributes one continuation sentence pair per 64 slots.

Reserve a distinct official validation file when available. Otherwise reserve
the first configured stories from training and skip them in both tokenizer and
training streams. Log the normalized corpus shares and preserve them in the
data fingerprint and checkpoints. This makes a resumed run reject silent
changes in TinyStories paths, mixture fraction, window construction controls,
or source weights.

## Compositional sequence-program control

Use `gator_sequence_program_benchmark.py` when the modular-prefix task is no
longer discriminating. It presents seven fresh values followed by three
instruction tokens and requires a one-shot seven-slot final sequence. The
operations combine permutations, value transforms, a recurrent prefix-XOR,
and sorting, so success requires instruction conditioning and ordered
composition rather than one fixed recurrence.

The first 10,010-parameter R2 screen reached 40.7% held-out token accuracy and
6.1% exact-sequence accuracy after 5,000 updates, compared with 12.5% valid
alphabet chance. Preserve that run as the no-replay curriculum baseline.

Do not train only the newest program depth in the next run. The first screen
showed classic curriculum forgetting: one-operation accuracy rose to 56.9%,
then fell to 29.8% when training switched exclusively to two operations and
to 20.0% during three-operation training. Mix earlier depths into later stages
or construct mixed-depth rows within each batch. Report all depths separately;
an improving hardest-depth score does not prove preservation of primitive
skills.

The 40% replay follow-up completed all 50k updates without reaching its 99%
mastery stop. It retained 75.7% one-op accuracy and improved two-op accuracy to
49.1%, confirming that replay addressed forgetting. Three-op accuracy,
however, flattened near 42% after 30k and ended at 42.3% token / 7.2% exact.
Do not spend another identical 50k budget. The next run must change available
computation or instruction access rather than only the replay mixture.

## Interpreting an unreachable language handoff gate

The first D20 TinyStories-first run remained at `teacher_fraction=1.0` for its
entire one-million-step budget because teacher-content accuracy plateaued near
30--31%, below the configured 60% gate. In this state:

- teacher metrics measure causal language capacity;
- blank-parallel samples test an untrained behavior and may remain letter spam;
- parallel transfer has neither succeeded nor failed because it never began;
- more updates at the same observed plateau are not the default next action.

Extract and probe `best_teacher.pt` autoregressively and with causal
counterfactuals before changing width. The resulting D20 probe, documented
below, shows that the original 60% gate rewards a teacher-prefix shortcut. A
nonzero parallel-practice floor is a valid future A/B, but token/span-level
prefix corruption is the cleaner next intervention and must not be conflated
with the completed zero-floor run.

## Diagnosing teacher-prefix shortcuts

The one-million-step D20 run demonstrates why teacher accuracy cannot be the
only handoff signal. Its best checkpoint reached 31.54% content accuracy, but
counterfactual evaluation showed that the score depended overwhelmingly on
the gold response prefix:

- shuffling the prompt changed SmolTalk and LMSYS accuracy by approximately
  zero percentage points;
- masking the prompt cost only four to seven percentage points;
- shuffling the answer prefix collapsed accuracy to 1.4--2.7%;
- blanking the answer prefix collapsed accuracy below 1%;
- true autoregressive rollout repeated common words and failed to answer.

Treat this pattern as prefix-shortcut saturation, not as prompt-conditioned
language mastery. A useful validation dashboard must report:

1. correct-prompt teacher content loss and accuracy;
2. shuffled-prompt teacher content loss and accuracy;
3. masked-prompt teacher content loss and accuracy;
4. blank-prefix and short-prefix content loss and accuracy;
5. true autoregressive rollout quality;
6. each source's realized sampling share and exhaustion count.

Define context utility as the performance margin between the correct prompt
and a shuffled prompt under the same response prefix. A handoff controller
should not reduce support merely because teacher accuracy rises; it should
also require a positive, sustained context-utility margin.

### Recommended support curriculum

Replace all-teacher versus all-blank batch selection with causal prefix
denoising:

```text
early: 90--95% of eligible response-prefix tokens visible
then:  increase random-token and contiguous-span BLANK corruption
always: include some zero-prefix and short-prefix rows
gate: continuation quality + semantic context utility
target: progressively remove support without a discontinuous objective jump
```

Use the non-rendering `BLANK` token for removed prefix tokens. Never expose a
future response token through the corruption mask. Oversample the first
content token and short-prefix cases because those are the points at which
the prompt must carry the prediction rather than the answer completing itself.

If context utility remains near zero after this curriculum, the next
architecture A/B is semantic receiver-specific context access: content-query
node reads or additional relay depth. Do not change node memory, relay depth,
and prefix corruption in one experiment.

### Corpus and checkpoint discipline

For a claimed fixed long-run mixture, use `streaming_exhaustion=restart`.
`drop` is a valid single-pass ablation, but it changes the distribution when a
source exhausts. In the D20 million-step run, the scheduled
TinyStories/SmolTalk/LMSYS/Tulu/Preferences proportions drifted increasingly
toward TinyStories as smaller streams disappeared.

The best D20 checkpoint occurred at step 330k. Content accuracy remained near
31% through step one million while content PPL worsened from 16.49 to 29.48.
When accuracy is flat and held-out content PPL regresses, rewind to the best
checkpoint and change the curriculum or data phase. Additional identical
updates are not useful runway.

Keep terminal voting and response length fixed in the first revised run.
Per-position accuracy was broadly flat over all 128 slots, longer contexts
were not worse, and the ten-lane coalition improved over individual lanes.
Those probes rule out slot erosion and terminal voting as the first levers.

### Implemented D20 retry

Use `--decode-mode prefix_denoise` for the direct retry. Its default schedule
implements the recommendation above:

```bash
--prefix-retention-start 0.95 \
--prefix-retention-floor 0.50 \
--prefix-denoise-warmup-steps 100000 \
--prefix-zero-row-fraction 0.05 \
--prefix-short-row-fraction 0.10 \
--prefix-short-max-tokens 8 \
--prefix-span-fraction 0.50 \
--prefix-context-utility-start 0.02 \
--prefix-context-utility-end 0.10 \
--handoff-start-content-accuracy 0.60 \
--handoff-end-content-accuracy 0.90 \
--handoff-accuracy-ema-decay 0.50 \
--streaming-exhaustion restart
```

Interpret the new log as follows:

- `retention` is the controller's requested ordinary-row support;
- `actual` includes zero/short rows and therefore is normally lower;
- `context_utility` is correct-context minus shuffled-context content
  accuracy under the same deterministic partial-prefix mask;
- `readiness` controls removal below the 50% scheduled floor;
- `best_context.pt` requires both positive context utility and useful correct
  teacher accuracy.

The first decision point is not whether raw teacher accuracy exceeds the old
31% record. It is whether context utility becomes decisively positive while
blank/short-prefix accuracy improves. If utility remains near zero despite
50% ordinary-prefix retention, test semantic receiver-specific context reads
or additional relays next.
