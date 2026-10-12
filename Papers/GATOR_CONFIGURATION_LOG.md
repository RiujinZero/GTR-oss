# Gator Configuration Ledger

**Benchmark:** simultaneous modular prefix formation, length 7, seed 0, batch size 128.  
**Chance accuracy:** approximately 8.3%.  
**Screen:** held-out accuracy after 5,000 training steps unless noted.

This ledger records successful, failed, and interrupted configurations. Results from the early variant-parser bug are excluded from architectural comparisons.

## Valid Gator results

| ID | Family | Parameters | Width | Nodes | Degree | Relays | Step 1k | Step 2k | Step 3k | Step 4k | Step 5k | Final held-out | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GATOR-980-ARW | Dense monolith | 980 | 8 | 4 | Dense | 4 | 8.6% | 9.0% | 8.5% | 10.8% | **20.8%** | 21.4% at 10k | 477.6 s / 10k |
| GATOR-10K-WIDE | Dense monolith | 10,160 | 42 | 4 | Dense | 4 | 8.1% | 8.1% | 8.8% | 7.8% | **8.8%** | 8.5% | 345.0 s / 5k |
| GATOR-10K-N40 | Sparse monolith | 10,010 | 7 | 40 | 3 | 4 | 8.2% | 9.9% | 19.4% | 22.5% | **24.6%** | **24.7%** | 762.8 s / 5k |
| GATOR-10CELL-RING-D2 | 10 stitched cells | 10,000 | 8/cell | 4/cell | 2 cell neighbors | 4 | 7.7% | 18.5% | 26.3% | 29.4% | **31.8%** | **31.9%** | 3,062.6 s / 5k |
| GATOR-10CELL-RING-D4-S0 | 10 stitched cells | 10,000 | 8/cell | 4/cell | 4 cell neighbors | 4 | 8.7% | 12.3% | 37.9% | 55.4% | 68.3% | **99.8% at 20k** | 521.9 s / 20k |
| GATOR-10CELL-RING-D4-S1 | 10 stitched cells | 10,000 | 8/cell | 4/cell | 4 cell neighbors | 4 | 15.0% | 35.5% | 33.6% | 45.9% | 51.0% | **91.3% at 20k** | 389.5 s / 20k |
| GATOR-10CELL-RING-D4-S2 | 10 stitched cells | 10,000 | 8/cell | 4/cell | 4 cell neighbors | 4 | 8.4% | 8.3% | 18.1% | 28.8% | 32.7% | **41.0% at 20k** | 393.0 s / 20k |
| GATOR-ARROW-CARRY-D4-S0 | 10 stitched Arrow cells, source-owned carry | 10,000 | 8/cell | 4/cell | 4 cell neighbors | 4 | 8.5% | 24.3% | 42.4% | 47.8% | 52.8% | 52.7% final eval at 5k | 1,712.3 s / 5k CPU |
| GATOR-PATH-PRIVATEIO-S0 | Current frontier, private tied cell I/O | 9,748 | 8/cell | 4/cell | local five-action ring | 4 | 8.6% | 14.0% | 18.7% | 20.8% | 21.6% | 21.6% final eval at 5k | 3,107.3 s / 5k CPU |
| GATOR-CONVERGENT-K10-R1-S0 | Flat graph, 10 persistent payload lanes | 10,010 | 8/lane | 40 shared | degree 3 | 1 | 11.2% | 31.1% | 35.6% | 38.8% | 41.8% | 92.0% token / 56.3% exact / 0.2436 loss at 30k; best 92.3% / 57.5% at 29k | 18.76 steps/s over resumed 20k--30k phase |
| GATOR-CONVERGENT-K10-R2-S0 | Flat graph, 10 persistent payload lanes | 10,010 | 8/lane | 40 shared | degree 3 | 2 | 8.3% | 27.7% | 36.2% | 50.0% | 56.4% | 91.0% token / 50.7% exact / 0.2408 loss at 20k; best 91.7% / 55.5% at 16k | 9.67 steps/s over resumed 10k--20k phase |
| GATOR-CONVERGENT-K10-R3-S0 | Flat graph, 10 persistent payload lanes | 10,010 | 8/lane | 40 shared | degree 3 | 3 | 13.0% | 34.7% | 41.8% | 44.8% | 47.1% | 82.2% token / 25.0% exact / 0.5049 loss at 20k | 5.60 steps/s over resumed 10k--20k phase |
| GATOR-CONVERGENT-K10-S0 | Flat graph, 10 persistent payload lanes | 10,010 | 8/lane | 40 shared | degree 3 | 4 | 18.3% | 35.4% | 45.1% | 48.9% | 54.0% | **98.2% token / 88.5% exact / 0.0638 loss at 30k** | 2,670.9 s for resumed 20k--30k optimized phase |
| GATOR-CONVERGENT-K10-R8-S0 | Flat graph, 10 persistent payload lanes | 10,010 | 8/lane | 40 shared | degree 3 | 8 | 8.8% | 15.1% | 17.7% | 39.5% | 54.3% | **76.2% token / 12.9% exact at 10k gate** | 1.12 steps/s final interval, pre-optimization |

## Immediate conclusions

1. Parameter count alone does not determine acquisition.
2. Expanding width from 8 to 42 while keeping four nodes destroyed the early learning event.
3. Spending the same approximately 10k budget on narrow structured computation
   outperformed naive width scaling; the degree-four stitched-cell model was the
   strongest configuration tested.
4. GATOR-10K-N40 crossed chance around step 2k–3k, earlier than the 980-parameter model's step 4k–5k transition.
5. The sparse forty-node model is computationally slower because receiver-specific memory scales with node count.
6. Ten locally complete Gator cells connected in a degree-two ring reached
   31.9% at 5k; increasing cell degree to four produced 68.3% at 5k and 99.8%
   at 20k for seed 0.
7. The first stitched-cell implementation is unvectorized and approximately four times slower than the sparse 40-node monolith on CPU.
8. Degree-four topology learned well above chance in all three tested seeds, but acquisition speed and final accuracy varied substantially.
9. Seed 2 was still progressing slowly at 20k; its 41% result should not yet be treated as a hard architectural ceiling.

The default relay-depth rows are operational comparisons, not a pure-depth
ablation. Their controlled-writing ramps and fixed relay codes are normalized
over total depth:

```text
R=4 output writing: 0, 1/3, 2/3, 1
R=8 output writing: 0, 1/7, 2/7, ..., 1
```

The clean follow-up sets `relay_code_span=8` and
`write_warmup_relays=4` for both R=4 and R=8. Tests prove that, with shared
weights and inputs, their first four location and transition traces are then
identical; only the R=8 model performs additional hops.

The one-relay topology has an additional mechanistic interpretation. Ten lane
starts are four nodes apart, while each degree-three handoff expands radius by
one. Neighboring supports remain disjoint after one handoff, first overlap
after two, and can only be read as shared node traffic on relay three because
memory is read before routing. Consequently R=1 and R=2 have no cross-lane
memory coordination; R=3 is the first depth capable of it. A topology test
locks this collision timing, and checkpoint analysis now reports pairwise lane
overlap at every relay state.

The active R=1 control nevertheless showed a clear independent-ensemble phase
transition: accuracy moved from `44.7%` at 6k to `52.3%` at 7k and `58.3%`
at 8k. During that jump, node support narrowed to `1.11` effective locations
while terminal support widened from about `3.1` to `5.4` lanes. With no
cross-lane memory available, the model recruits more independent specialists
into the final vote; recurrent models can instead coordinate a smaller
terminal coalition.

At 10k, R=1 reached `69.5%` token and `6.2%` exact accuracy, only `2.4`
token points behind R=4 but with lower sequence coherence. This control also
exposes a benchmark shortcut: every sequence position within a lane starts at
the same graph node. Receiver-relative memory therefore gives each output
slot a weighted view of all eight prefix inputs before the first handoff.
R=1 can learn a direct global aggregation without propagating information
through graph depth. The benchmark demonstrates simultaneous formation and
tiny-model acquisition, but by itself does not prove a logarithmic multi-hop
scan algorithm. A locality-forcing ingress control must distribute input
positions across graph sites or otherwise restrict the first memory read.

R=1 finished 20k at `84.7%` token, `30.3%` exact, and `0.4446` loss,
averaging `19.15` steps/s end to end. Terminal support widened to `6.76`
effective lanes while node support narrowed to `1.10`. The one-hop model is
therefore a strong independent-specialist ensemble, but it remained well
below the four-relay model's `92.1%` token and `61.0%` exact result.

Operational R=2 reached `85.6%` token, `36.1%` exact, and `0.4064` loss
at only 10k, averaging `9.90` steps/s. It outperformed R=1's complete 20k
result and every deeper model at the same 10k update count. At 7k it already
reached `69.2%`, versus R=1 `52.3%`, R=4 `61.2%`, and R=8 `66.8%`.
Relay benefit is therefore not monotonic during acquisition. Two relays are
the current early-learning sweet spot, although the default write schedules
prevent attributing that entirely to recurrent depth.

Checkpoint probing establishes that R=2 did not secretly coordinate lanes.
Measured pairwise lane overlap was exactly `0.0` at \(q^0\) and \(q^1\),
then `0.209867` only at terminal \(q^2\), after the final memory read. The
best lane alone achieved `25.5%`; the top six weighted lanes reached `85.7%`.
Its 86% result is therefore cooperative terminal voting over independently
refined traces, not shared-memory cross-lane communication.

Extending R=2 to 20k did not produce mastery. It ended at `91.0%` token /
`50.7%` exact / `0.2408` loss, with its best logged point at 16k:
`91.7%` / `55.5%`. Its early sample-efficiency advantage turned into the
same approximate 92% ceiling seen in R=1 and the initial R=4 run. R=2 is an
excellent short-budget configuration, but not yet a mastered replacement.

Operational R=3, the first depth able to consume cross-lane overlap, reached
only `71.3%` token, `8.5%` exact, and `0.8225` loss at 10k. This was
essentially tied with R=4 (`71.9%`/`8.7%`) and far below R=2
(`85.6%`/`36.1%`). Its terminal coalition collapsed to `2.60` effective
lanes at 5k before widening to `4.27` at 10k. Early cross-trace contact can
create interference and harder credit assignment; it is not automatically a
learning advantage. The contact was material, not merely topological:
pairwise overlap reached `0.626731` at \(q^2\), which relay three consumed,
and `0.786597` at terminal \(q^3\).

R=3 finished 20k at only `82.2%` token / `25.0%` exact / `0.5049` loss.
Its terminal coalition widened steadily to `4.71` effective lanes rather than
consolidating toward R=4's approximately three. It can consume cross-trace
traffic, but lacks a subsequent stage that commits those partially reconciled
traces into a coherent output.

That control is now implemented as `bootstrap_context_mode=self`, with no
additional parameters. Relay zero uses each packet's own payload instead of
shared node memory; later relays use the normal receiver-relative memory.
Tests lock the key causal property: at R=1 blank-output logits are invariant
to condition-token changes, while at R=2 they can change after routed
handoffs. Historical behavior remains `bootstrap_context_mode=shared`.

### Mastered Arrow relay-prefix ablation

The preserved 20k Arrow seed-zero checkpoint was evaluated after stopping its
trained four-relay computation early. This is a true prefix ablation:
relay-step codes and output-writing fractions remain those of the original
R=4 model.

| Executed trained relays | Token accuracy | Exact accuracy | Loss |
|---:|---:|---:|---:|
| 1/4 | 12.07% | 0.00% | 6.7087 |
| 2/4 | 20.91% | 0.00% | 7.4595 |
| 3/4 | 58.32% | 3.22% | 4.5696 |
| 4/4 | **99.76%** | **98.49%** | **0.0091** |

All four learned hops are functionally necessary. The final hop is not a
redundant LLM-style polish pass: it converts a partially formed 58% state
into near mastery. This does not prove a native shallow model cannot learn a
different solution given more updates; the active R=1 30k continuation tests
that separate question.

Immediately before mastered Arrow's final hop, positional accuracy was
`61.96,88.72,88.92,70.36,47.27,31.74,19.29%`. After hop four it became
`100.00,99.90,99.90,99.85,99.80,99.22,99.66%`. The commitment relay repairs
the complete formation, with the largest corrections at long prefix
distances; it is not merely improving confidence on already-correct tokens.

The reproducible 20k Convergent R=4 checkpoint behaved the same way:
`9.5%`, `10.4%`, `35.3%`, and `91.2%` token accuracy after one through
four trained hops. The final-hop discontinuity is therefore not unique to the
historical Arrow seed. Depth-specific training discovers different
algorithms; a good native R=2 model cannot be substituted for the first two
hops of an R=4 checkpoint.

The R=1 continuation answered that first convergence gate negatively. It
finished 30k at `92.0%` token / `56.3%` exact / `0.2436` loss; its best
logged point was `92.3%` / `57.5%` at 29k. It continued improving well
beyond 20k but did not approach 99% mastery. It roughly matched R=4's 20k
token accuracy using 50% more updates, while retaining lower sequence
coherence. R=2 is the next shallow-depth mastery candidate.

Per-position diagnostics localize the shallow ceiling to dependency distance.
Best R=1 position accuracy was
`99.95,98.34,96.58,91.11,87.30,82.67,89.89%`; R=2 showed the same late
decline. Current R=4 at 24k improved this to
`100.00,99.66,98.68,95.80,93.21,90.04,90.77%`. Step consistency decayed
from `99.95%` to `72.80%` across R=1's formation, versus `100%` to `85.21%`
through R=4's sixth output. Extra relays primarily extend reliable prefix
dependency depth rather than uniformly polishing every slot.

Continuing R=4 from 20k to 30k broke the shallow ceiling. It progressed
through `94.8%` at 22k, `97.1%` at 26k, and finished at `98.2%` token /
`88.5%` exact / `0.0638` loss. Constant LR `0.008` caused local regressions,
but the high-performance basin continued improving. Four-relay coordination
therefore changes attainable performance, not just acquisition speed. It
missed the predefined 99% mastery gate narrowly.

A best-state continuation then lowered the learning rate from `0.008` to
`0.002` while preserving the model, AdamW state, and portable batch-generator
state. Mastery appeared within 1,000 additional updates:

| Step | Token accuracy | Exact accuracy | Loss | Gradient norm |
|---:|---:|---:|---:|---:|
| 30,000, before cooldown | 98.2% | 88.5% | 0.0638 | 5.41 |
| 31,000 | **99.2%** | **95.2%** | 0.0263 | 2.14 |
| 32,000 | **99.3%** | **95.6%** | 0.0235 | 0.75 |
| 33,000 | **99.4%** | **95.9%** | 0.0223 | 1.00 |

Three consecutive checkpoints held above the 99% token-mastery threshold
while loss continued to fall. The carry-forward restoration is therefore not
merely a transient 31k evaluation spike. The current operational recipe is
four relays at the historical learning rate for basin discovery, preserved
best checkpoints, then a fourfold learning-rate cooldown for commitment.

A separate 4,096-example probe of the step-32k best checkpoint measured
`99.29%` token / `95.68%` exact / `0.023832` loss. Terminal fusion had
consolidated into three material lanes with weights `40.64%`, `29.93%`, and
`29.13%`. They were complementary rather than redundant: the best single
lane reached only `33.25%`, the best two reached `82.65%`, and the three
together restored `99.27%`. Relay-prefix ablation on the same mastered
checkpoint yielded `9.61%`, `12.19%`, `34.59%`, and `99.35%` token accuracy
after one through four trained hops. Cooldown sharpened the final commit; it
did not replace the four-stage computation with a shallow shortcut.

## Explicit-connection topology control

The first equal-parameter replacement of geometric neighbors used the
Convergent model with a degree-three small-world connection table. Self and a
directed successor were retained; 25% of remaining ring edges were replaced
by deterministic shortcuts. No trainable parameter, initialization tensor,
batch, optimizer setting, or relay schedule changed.

| Configuration | Step 3k token | Step 5k token | Step 5k exact | Step 5k loss |
|---|---:|---:|---:|---:|
| Ring | 45.1% | 54.0% | 0.4% | 1.1774 |
| 25%-rewired connections | **58.8%** | **79.8%** | **21.2%** | **0.5980** |

The connected graph began more slowly (`8.7%` versus `18.3%` at 1k), matched
the ring at 2k, then entered a much steeper acquisition phase. It ran
approximately 8--9% fewer steps per second over the first 5k because graph
closure exposed more node rows, but reached the ring's 5k capability before
3k updates. This is the first direct evidence that layout changes capability
and sample efficiency at fixed Gator parameter count.

More connectivity was antagonistic at four relays. The otherwise identical
50%-rewired R4 model reached only `38.6%` token / `0.1%` exact /
`1.4954` loss at 5k. It created substantial lane overlap before task learning
began, consistent with premature graph mixing delaying specialization.

Reducing that layout to two relays reversed much of the damage:

| Configuration | 5k token | 5k exact | Loss | Runtime | Steps/s |
|---|---:|---:|---:|---:|---:|
| R4 / 25% rewired | **79.8%** | **21.2%** | **0.5980** | 1,440.6 s | 3.47 |
| R4 / 50% rewired | 38.6% | 0.1% | 1.4954 | 1,458.8 s | 3.43 |
| R2 / 50% rewired | 62.3% | 2.8% | 0.9919 | **457.8 s** | **10.92** |

R2/50% loses capability per update to R4/25% but reaches approximately the
same 62% token level in half the wall time. The compute-matched continuation
is 16k total R2 steps, approximately the updates it can execute during one
5k R4/25% run.

The 16k compute-matched continuation finished in 1,494.8 total seconds across
its 0--5k and 5--16k phases, versus 1,440.6 seconds for R4/25% at 5k:

| Approximately 24--25 minute budget | Token | Exact | Loss |
|---|---:|---:|---:|
| R4 / 25%, 5k | 79.8% | 21.2% | 0.5980 |
| R2 / 50%, 16k | **98.0%** | **87.1%** | **0.0691** |

R2/50% first exceeded the R4/25% 5k endpoint at step 7k, reaching `81.5%`
token / `26.2%` exact in approximately 10.6 total minutes. Connectivity thus
more than compensated for removed recurrent depth on this task. At equal wall
time it improved token accuracy by 18.2 points and exact accuracy by 65.9
points.

The 4,096-example checkpoint audit reproduced `97.99%` token / `87.04%`
exact / `0.06974` loss. Terminal voting had one dominant lane at 47.8%, but
that lane alone reached only 55.6% token accuracy. Eight lanes were required
to reproduce the full result; removing the dominant lane dropped accuracy to
70.3%. The fast R2 solution therefore still uses complementary converging
traces rather than collapsing to a single effective path.

A four-layer one-shot Transformer control was parameter matched within ten
weights: 10,020 Transformer parameters versus 10,010 Gator parameters
(`d_model=14`, two heads, FF width 52). It used the same dedicated portable
batch generator, 16k updates, batch 128, seed zero, and LR `0.008`.

| Matched 16k control | Parameters | Steps/s | Token | Exact | Loss |
|---|---:|---:|---:|---:|---:|
| R2 / 50% Gator | 10,010 | 10.61 continuation | **98.0%** | **87.1%** | **0.0691** |
| Four-layer one-shot Transformer | 10,020 | **34.68** | 8.5% | 0.0% | 2.4852 |

The Transformer completed in 461.4 seconds and was 3.27 times faster per
update, but it stayed at chance throughout the run (best logged token accuracy
8.8%). The speed difference is therefore not a capability-speed advantage:
the Transformer did not reach even the beginning of Gator's acquisition
curve. Earlier screens also showed that roughly 70k Transformer parameters
were still insufficient at 20k updates; by the time Transformer size and
training budget were increased enough to search for learning, the nominal
per-step speed advantage had largely evaporated. Comparisons must report wall
time to a common held-out capability threshold, with an unreached threshold
recorded explicitly rather than treating faster chance predictions as a win.

### 130k Convergent language scale

The first language transfer scales the R2/50% connected architecture to a tied
2,048-piece vocabulary:

| Component | Configuration | Parameters |
|---|---|---:|
| Tied vocabulary and bias | `2048 x 32 + 2048` | 67,584 |
| Ten lane input/output views | `2 x 10 x 20 x 32` | 12,800 |
| Remaining lane computation | 10 lanes, width 20 | 11,010 |
| Shared graph | 66 nodes, width 20, degree 3 | 38,808 |
| **Total** | R2, small-world, 50% rewired | **130,202** |

This increases both payload width (`8 -> 20`) and node population (`40 -> 66`)
without replicating the large lexical table per lane. The shared embedding is
weight-tied at readout; learned lane-specific projections give each persistent
trace a different lexical view. Terminal voting is fused in width-32 embedding
space before the 2,048-way projection, avoiding a
`[batch,slots,lanes,vocabulary]` activation.

Static seed-zero geometry retains pre-read interaction at the larger node
count: 6.7% of lane pairs overlap after the first handoff, so relay two can
consume cross-trace contact. The terminal frontier covers 42/66 nodes with
28.9% lane-pair overlap. An 84-node/width-18 alternative fit the same budget
but weakened the consumed-contact geometry and was rejected for this first
transfer.

The local 10k run uses SmolTalk Everyday only, context 64, response 32, batch
32, seed zero, LR `0.003`, and the historical 50k cosine horizon so its step
10k learning rate matches the earlier Arrow trajectory. The full-batch smoke
test ran at 2.64 steps/second on the local five-year-old i7 with approximately
0.87 GB resident memory.

The gate was stopped at step 9,000 after the validation curve exposed the
known finite-corpus repetition failure. The active filters supplied only 5,726
training responses; 9,000 batches of 32 correspond to roughly 50
dataset-equivalents. Convergent nevertheless reached its best PPL `42.24` at
step 6,000, beating the older Arrow SmolTalk-only best of `42.36` at step
13,000. It crossed Arrow's step-10,000 PPL `43.72` by step 3,500. This is a
roughly 2--3x update-efficiency improvement, but the subsequent oscillation
and regression cannot be interpreted as an asymptotic floor because the
training stream was already heavily repeated.

The preserved step-6,000 checkpoint was therefore promoted directly into a
ten-hour local continuation. The continuation reproduces the established
language-bundle corpus proportions as four exhaustible streams: SmolTalk
Everyday, LMSYS, Tulu3, and SmolTalk Preferences at `1:4:3:2` weights
(10:40:30:20 percent while all remain active). It uses per-traversal
deduplication, complete-context/whole-turn filtering, and epoch-safe restart.
Each source must drain one complete shuffled traversal before it can restart,
so no duplicate traversal is inserted merely to fill a reservoir. SmolTalk
Everyday continues to supply the stable validation split and also enters the
training mixture as its own independently tracked stream.
The continuation preserves model and AdamW moments, resets sampler state,
restarts at LR `0.001`, and uses a 300k cosine horizon. Logs record per-source
samples and traversal restarts every 15 minutes. The promotion gate for a
Blackwell run remains a meaningful break below PPL `33` with contextual
formed responses, not merely a lower whitespace-weighted aggregate loss.

The first attempted mixed continuation exposed why restart telemetry is
mandatory. After only 1,549 new updates, SmolTalk Preferences had emitted
11,222 samples and restarted five times; LMSYS and Tulu3 had not restarted.
The filtered preference corpus therefore contained only roughly 2,200 usable
examples under this configuration. That run was stopped at global step 7,549
(PPL `52.98`) and rejected as a no-repeat experiment. The trainer now supports
`--streaming-exhaustion drop`: a shuffle-buffered source drains every accepted
example exactly once, leaves the mixture at EOF, and never silently cycles.
The primary SmolTalk split can also enter as an exhaustible named stream, which
preserves the language bundle's `1:4:3:2` corpus weights without sampling its
small finite pool with replacement.

The corrected four-stream run validated the intervention at its first
wall-clock report. After 1,424 new updates (global step 7,424), validation PPL
was `42.69`, loss `3.7540`, and token accuracy `30.5%`. The immediately prior
restart-contaminated attempt had PPL `52.98` after 1,549 new updates. Thus the
matched `1:4:3:2` corpus plus true no-repeat sampling recovered approximately
10 PPL at essentially the same continuation depth. Preferences exhausted
cleanly after 4,661 emitted examples; all four restart counters remained zero.
The primary SmolTalk stream had emitted 5,055 examples and was expected to
leave the mixture shortly afterward. This is the expected early regime:
PPL in the 40s near 2,000 mixed updates. The next gate is entry into the 30s
near 12,000 mixed updates, followed by the historical PPL-33 crossover.

The strict one-pass run then exposed a second confound. At 2,873 new updates,
both Preferences (4,661 examples) and primary SmolTalk (5,565 examples) had
left the mixture. PPL rose from `42.69` to `47.54` as training abruptly became
LMSYS/Tulu-only. This does not show that source repetition is beneficial; it
shows that permanent source removal changes the objective. The original
restart implementation also had a concrete bug: when a corpus contained fewer
examples than its shuffle-buffer target, it restarted immediately while
filling the buffer, placing multiple copies of the same traversal into one
reservoir before any examples were emitted.

The corrected default is epoch-safe restart. A source fills from one
deduplicated traversal, emits/drains every buffered example once, and only
then starts the next traversal and increments its restart counter. This
preserves the `1:4:3:2` training proportions across a long run while forbidding
duplicate-filled shuffle buffers. Strict `drop` remains a useful distribution-
shift ablation, not the default language training policy.

The first Blackwell scaling launch used width 20, 66 shared nodes, 20 lanes,
degree three, R=2, and 50% rewiring. The exact trainable count is `154,012`.
On an NVIDIA RTX PRO 6000 Blackwell Server Edition with compilation and a
four-batch prefetch pipeline, the run warmed from `27.84` steps/second at its
first compiled report to a median `63.72` steps/second from step 10k onward.
At batch 32 with 64 context plus 32 response slots, the steady rate is about
196,000 processed slot states/second and 65,000 simultaneous response-slot
predictions/second. It is roughly 3.75x the prior 17-step/second 130k Arrow
language measurement, although this is not yet a matched capability result:
the Convergent run is slightly larger and must still report its PPL/content
curve. At sustained throughput, 250k updates require about 65 minutes before
evaluation and checkpoint overhead.

The first curve from this launch was materially delayed relative to the
ten-lane model, although it was still descending when inspected. Static
topology analysis gives a concrete scaling hypothesis. With ten lanes and 66
nodes, the R2 frontier covers 42/66 nodes, 28.9% of lane pairs collide, and
each lane collides with about 2.6 peers. With twenty lanes but the same 66
nodes, the R2 frontier covers 64/66 nodes, 28.4% of lane pairs collide, and
each lane now collides with about 5.4 peers. The pairwise percentage is nearly
unchanged, but the absolute number of colliding lane pairs rises about fourfold
and the shared graph is almost saturated.

This suggests the provisional frontier-density invariant

\[
\rho_{\mathrm{frontier}} \approx \frac{L\,F(K,R)}{N},
\]

where \(L\) is lane count and \(F(K,R)\) is the mean reachable frontier of one
lane at degree \(K\) and relay depth \(R\). For degree three and R2,
\(F\approx6\). Both L10/N66 and L20/N132 therefore have
\(\rho_{\mathrm{frontier}}\approx0.91\). The clean recovery test is L20/N132:
if its acquisition curve advances relative to L20/N66, node count must scale
approximately linearly with lanes at fixed degree and depth. Width remains a
separate lever governing transformation capacity at each occupied node.

The authoritative L20/N66 run was interrupted after step 107,000. Its curve
was:

| Step | Validation PPL | Accuracy |
|---:|---:|---:|
| 1,000 | 54.72 | 27.7% |
| 4,000 | 42.65 | 29.1% |
| 7,000 | 39.99 | 29.7% |
| 10,000 | 38.47 | 29.9% |
| 20,000 | 35.53 | 30.4% |
| 30,000 | 34.74 | 30.5% |
| 39,000 | 34.07 | 30.6% |
| 43,000 | 33.75 | 30.7% |
| 65,000 | 33.06 | 30.8% |
| 82,000 | 32.97 | 30.7% |
| **99,000 best** | **32.93** | **30.8%** |
| 107,000 | 33.32 | 30.8% |

The live estimate that PPL 33 had been reached at step 39k was incorrect; that
report was PPL 34.07. The first logged value below 33 occurred at step 82k.
Mean PPL over steps 80k--100k was 33.22, and over the final 101k--107k window
it was 33.32. The 32.93 minimum is only 0.07 below the historical PPL-33 gate
and is inside the surrounding validation oscillation. Exact response accuracy
remained zero.

This launch used bundle SHA-256
`0156434dcd612a1dd6c9dc7f9e3dda19503ea2e6d9e1629254cccd6fe5e7d6da`
and reported 5,251 primary training examples plus 939 validation examples.
The later local bundle reports 5,565/898. The 107-point L20/N66 curve is
internally authoritative, but its comparison with the historical Arrow
PPL-33 floor is nominal rather than a bit-for-bit matched evaluation. A final
architecture claim requires evaluating saved checkpoints through one frozen
corpus, tokenizer, and validation manifest.

Qualitatively, the model learned contextual openings and short formations:
examples included `Hello! How can I help you today? Is there`, `The capital
of`, `Yes, there are several`, and `Why did`. It still frequently produced
one-to-three-word answers followed by BLANK/space slots, repeated local words,
or generic refusals. This is evidence of contextual learning, but not a
material language improvement over Arrow.

The result is therefore a matched-floor outcome, not a win: L20/N66 required
roughly 80k updates to touch the old Arrow floor and showed no sustained break
below it by 107k. Doubling lanes while leaving node workspace fixed added
coordination cost without measurable asymptotic gain. L20/N66 should remain
the overcrowding control; L14/N90 is the matched-parameter balanced test and
L20/N132 is the ratio-preserving capacity test.

The next launched A/B uses L16/N90 at width 20, degree three, R2, and 50%
rewiring. It has exactly 158,600 parameters, only 3.0% more than L20/N66.
Static seed-zero analysis measures R2 population coverage of 68/90 (75.6%),
19.2% colliding lane pairs, mean lane Jaccard 0.029, and approximately 2.88
colliding peers per lane. The latter is close to L10/N66's 2.6 and far below
L20/N66's 5.4. Its heuristic frontier density is 1.08: slightly busier than
the 0.91 target, but a strong near-balanced recovery test at nearly matched
parameter count.

An operator-observed early improvement in L16/N90 motivated a second clean
axis: L10/N66 with payload/node width increased from 20 to 30 while lexical
width remains 32. The graph, lane count, contacts, relays, rewiring, and global
voter are unchanged. Exact parameters rise from 130,202 to 186,382:

| Component | D20 | D30 |
|---|---:|---:|
| Tied lexical shell | 67,584 | 67,584 |
| Lane computation | 23,810 | 41,710 |
| Shared graph | 38,808 | 77,088 |

This is the authoritative compute-width A/B. Its first 50,000 updates were
quantitatively tied with the completed L20/N66/D20 control:

| Step | D30/L10 PPL | D20/L20 PPL |
|---:|---:|---:|
| 5,000 | 41.78 | 41.51 |
| 10,000 | 38.76 | 38.47 |
| 20,000 | 35.84 | 35.53 |
| 30,000 | 34.73 | 34.74 |
| 40,000 | 34.37 | 34.26 |
| 50,000 | 34.27 | 34.25 |

The displayed D30 samples nevertheless used a modestly broader formation:
across the ten 5,000-step displays they averaged about 2.4 word-like pieces
and 8.9 non-space characters, versus 1.8 and 5.1 for D20 at the corresponding
milestones. These are unmatched random examples and a tiny sample, so this is
a qualitative signal rather than proof. The D30 run continues unchanged to
300,000 updates, matching the historical Arrow horizon and the cosine
scheduler horizon. A win requires a sustained break below PPL 33 and/or
reproducibly better fixed-prompt/content diagnostics on the same validation
fingerprint.

The D30 run reached approximately 60--65 steps/s on an RTX PRO 6000 Blackwell,
versus about 17 steps/s for historical 130k Arrow language training. This is a
roughly 3.5--3.8x throughput advantage despite D30 having 49.6% more
parameters than the exact Arrow checkpoint.
Increasing the prefetch queue from 10 to 20 unlocked the upper end of that
range, identifying host-side mixed-corpus preparation as part of the measured
training bottleneck. The scaling notebook now defaults to 20 prefetched
batches for this target. That speed result is already material; capability
remains provisional.

At step 125,000, an intermediate D30 best reached raw PPL `32.9427`, loss
`3.49477`, and token accuracy `30.69%` on the 939-example validation manifest
with fingerprint
`d1c64124105659d2cf53d84a530ee2e6e436157f0847c153ee437195212e98e0`.
This was only `0.1486` PPL behind the frozen Arrow best (`32.7941`) at roughly
half as many updates. The step-155,000 latest checkpoint briefly regressed to
PPL `33.08`, demonstrating why best-checkpoint selection is mandatory.

The final best arrived at step 195,000: PPL `32.8530`, loss `3.49204`, token
accuracy `30.80%`, content PPL `81.30`, content top-5 accuracy `30.81%`, and
formation coverage `14.24%`. It was only `0.0589` PPL behind Arrow while
using 80.6% as many updates. By step 285,000 the latest weights had regressed
to PPL `33.1720`, loss `3.50171`, token accuracy `30.75%`, content PPL
`82.94`, and content top-5 accuracy `30.60%`. Formation coverage continued
creeping upward to `14.95%`, but that additional placement signal did not
improve the selected language objective. The D30/global control therefore
confirmed the approximately PPL-33 plateau rather than breaking it.

The original Drive run directory was subsequently reused accidentally by a
fresh global-vote launch. The true step-195,000 best had already been frozen
locally and was rescued to
`ARCHIVE_global_d30_n66_l10_r2_rw50_seed0_best_step195k/`
as `best_step195000_global.pt`, SHA-256
`21df0d5e7b431e52eac8b075e0b352cf835a9582ac9977318230c61582169b69`.
Do not treat the overwritten directory's later `best.pt` as this control.
Future launches must use a mode-bearing run name and verify both parameter
count and `terminal_vote_mode` in the startup banner before being left
unattended.

The fixed six-prompt probe at D30 step 195,000 formed:

| Prompt | Arrow step 242k | D30 step 195k |
|---|---|---|
| Season | `I is currently` | `I m sorry` |
| Cat/meow | `Here` | `The` |
| 2 + 2 | `The + 2 is equal` | `2 + 2 2 2` |
| Rental cars | `You can started to` | `There are sorry to` |
| Team sports | `Some popular sports` | `Some popular sports include include` |
| Beginner running | `It is a` | `This m g g` |

This is not general language mastery, but it is a stronger acquisition-speed
result than raw PPL alone. The random validation sample formed
`Hello! How can I I you today`, showing a seven-word contextual scaffold with
one local repetition, while fixed probes remained inconsistent. Wait for the
final best checkpoint before calling the architecture an asymptotic winner.

### 350k under-compute hypothesis

The 32-slot language task may be under-computed at D30. With vocab 2,048,
lexical width 32, N66/L10/K3/R2, global voting, and address width four, exact
parameter count as a function of payload width \(D\) is:

\[
P(D)=86D^2+1318D+69442.
\]

Therefore:

| Width | Total parameters | Non-lexical/graph-and-lane parameters |
|---:|---:|---:|
| D30 | 186,382 | 118,798 |
| D50 | 350,342 | 282,758 |
| D54 | 391,390 | 323,806 |

D50 is the next clean capacity test. It preserves N66, L10, degree three,
R2, 50% rewiring, lexical width 32, corpus, objective, and global terminal
vote. Only payload/node width changes. D54 is held as a follow-up rather than
being launched simultaneously. This isolates the hypothesis that 32
simultaneous output positions need more transformation capacity without
confounding it with more lanes, nodes, contacts, or recurrent depth.

Prefetch depth is training-only. Its improvement demonstrates that the
Blackwell was partly input-starved, but it must not be counted as an inference
speedup. Inference benchmarks must separately report batch-one latency,
formed tokens per second, context length, output slots, relay count, and peak
allocated VRAM with autograd and optimizer state absent.

Prompt-conditioned terminal voting is now implemented as a separate flag.
`--terminal-vote-mode prompt` adds one width-\(D\) confidence query per lane
and chooses one jury per prompt after the final relay. That jury is shared
across all output slots. At L16/N90/D20 it adds 320 parameters, producing
158,920 total. Zero query initialization exactly reproduces global-vote logits;
a one-step real-language smoke produced finite loss/gradients, saved the mode
and validation fingerprint, and learned a nonzero query. The global mode
remains the default and checkpoint-compatible control.

Slot-conditioned terminal voting is the next formation-control A/B:

```text
--terminal-vote-mode slot
```

It reuses the same \(K D\)-parameter confidence query as prompt voting but
does not average the 32 output states. It produces terminal weights
`[batch, output_slot, lane]`, so every immutable output address can select a
different prompt-aware coalition. The zero query makes its initial logits
exactly equal to global voting, and unit coverage verifies finite nonzero
query gradients plus exact shared-vocabulary fusion. At L16/N90/D20 it is
158,920 parameters, identical to prompt voting.

At the active D30/N66/L10 control it adds only \(10\times30=300\)
parameters, increasing the exact count from 186,382 to 186,682. A real
one-step SmolTalk trainer smoke completed training, evaluation, and checkpoint
save/load. The saved configuration retained `terminal_vote_mode=slot`; the
query RMS moved from zero to `0.00099992` and remained finite. The scaling
notebook's default `formation` experiment launches only this D30/slot
condition because the matching seed-zero D30/global baseline is complete.
The intended run name is
`formation_slot_d30_n66_l10_r2_seed0_300000steps`; its startup banner must
report `186682` parameters and `terminal_vote_mode=slot`.

Language evaluation now logs formation-controller structure directly:
entropy-effective voters, mean top-lane mass, adjacent-slot vote delta, query
RMS in the checkpoint, and the mean number of distinct winning lanes across
one response. A real one-step smoke printed `effective=3.00 top1=0.334
slot_delta=0.0002 winning_lanes=1.00 query_rms=0.001000`; the nonzero slot
delta confirms the controller began differentiating positions immediately
without disrupting the initially broad jury.

This intervention was promoted ahead of D50 after the D30 run showed broader
formed fragments without a material PPL improvement. Width bought roughly
one additional coherent word in favorable probes, but the model still
frequently stopped, repeated, or blanked the remainder. Nodes transform
content and lanes carry complementary proposals; the shared terminal jury is
the component that actually assembles those proposals into 32 fixed output
positions. Raw lane or node count is therefore not the next isolated lever.

The first epoch-safe 130,202-parameter laptop report arrived after 1,348 new
updates (global step 7,348): PPL `43.36`, loss `3.7696`, accuracy `30.1%`, and
1.48 steps/second. Preferences had emitted 8,804 examples with exactly one
restart: its first 4,661-example epoch had been fully drained before epoch two
began. Primary SmolTalk had emitted 4,220 examples with zero restarts. No
source was exhausted, so the `1:4:3:2` distribution remained stable. This
directly distinguishes the fix from the old reservoir, which had already
restarted Preferences five times at comparable depth merely to fill its
buffer.

The next two wall-clock reports remained monotonic: global step 8,673 reached
PPL `42.26`, and step 9,849 reached PPL `41.47`. The terminal-vote probe shows
that language currently uses a broad global jury. At the SmolTalk step-6,000
checkpoint, entropy-effective/L2-effective voters were `6.45/5.05`, the top
three held 74.47% of vote mass, and seven voters were needed for 90%. At mixed
step 9,849 these widened to `6.85/5.45`; top-three mass fell to 70.09%, top-five
mass was 83.33%, and seven were still needed for 90%. This differs from the
roughly three-material-voter coalition seen in the mastered prefix model and
supports treating final coalition size as a scaling variable.

Global step 10,904 reached the local continuation's next best raw PPL,
`41.44`, at `1.35` steps/s. This completes the requested local
approximately-10k scale gate: the 130,202-parameter R2/50 architecture trains
and improves, but it is far too early to compare asymptotic language capability
with a 242k-step Arrow checkpoint.

Interpret this 10k run as an early-efficiency gate. The prior 130k Arrow
language program ultimately approached a PPL floor near 33 after roughly
30--40k steps. The connected model does not establish an asymptotic language
win merely by beating Arrow's 10k PPL; it must break PPL 33 on a matched
evaluation corpus while maintaining content coverage and contextual
formations. If the 10k curve is still descending materially, continue it on
the Blackwell to at least the old floor window.

### Frozen historical Arrow language control

The authoritative raw-PPL Arrow control is:

```text
MyDrive/GTR/experimental_checkpoints/
  language_lmsys_clean_62cell_hemi_ctx64_seed0_v3/best.pt
```

Its SHA-256 is
`6172e2c0761ecffdff3d59576559ca7b8cd7248e7659b545cb8b8b3ac74bc652`.
It is step 242,000 with exactly 124,616 trainable parameters, vocab 2,048,
width-32 communal lexical input/output, 62 width-8 cells, four nodes per cell,
four relays, four neighbors, feature-gated input, grouped-concat output, and
hemispheric topology. Its saved 950-example validation reports raw PPL
`32.7941`, loss `3.49025`, and token accuracy `31.21%`.

The historical “130k Arrow” label was rounded. Relative to the exact
checkpoint, D20 Convergent at 130,202 parameters is 4.5% larger and D30 at
186,382 is 49.6% larger. Preserve both exact counts in scaling claims.

Do not substitute the later `language_gators_align_*` checkpoint as the raw
PPL control. That model was selected on monotonic aligned loss; its rigid-slot
raw PPL is intentionally not comparable to the earlier objective.

The fixed checkpoint probe produced:

| Prompt | Arrow step 242k | Local Convergent step 10.9k |
|---|---|---|
| Season | `I is currently` | `I'm` |
| Cat/meow | `Here` | `I` |
| 2 + 2 | `The + 2 is equal` | `The` |
| Rental cars | `You can started to` | `You can s` |
| Team sports | `Some popular sports` | `Some` |
| Beginner running | `It is a` | `It` |

Neither checkpoint answers these probes correctly. Arrow has greater
formation length and some contextual lexical selection, while early
Convergent often selects the same appropriate opening but has not yet formed
the continuation. Repeat this exact probe list on the D30 300k best
checkpoint. The checkpoint tokenizers are self-contained, so prompts are
identical even though historical validation manifests predate fingerprint
logging.

## Parameter formulas

For dense Gator-ARW with hidden width \(D\) and \(N\) nodes:

\[
P_{dense}=ND^2+2DN^2+2ND+30D+N^2(N+5)+5N.
\]

For sparse degree-\(K\) Gator-ARW:

\[
P_{sparse}=ND^2+2ND+30D+NK(2D+N+5)+5N.
\]

For \(M\) literal 980-parameter cells:

\[
P_{stitched}=980M+P_{coupling}.
\]

Ten cells leave approximately 200 parameters for coupling under a 10k budget.

For language, the vocabulary shell is separated from the validated cell. Each
stitched local cell retains the original parameter allocation:

\[
P_{cell}=128_{lexical}+128_{address}+32_{node}+32_{step}
+288_{node\ transforms}+336_{router}+16_{norm}+16_{relative}+4_{start}=980.
\]

The 130k language configuration contains a 68,136-parameter shared language
shell, 160 shared coupling parameters, 62 literal 980-parameter cells, and 248
cell-coupling gates, totaling **129,304 parameters**.

## Planned stitched-cell family

| Topology | Description | Primary hypothesis |
|---|---|---|
| Serial | Each four-node cell feeds the next | Greater compositional depth |
| Parallel | Cells process the same formation independently, then combine | Competing hypotheses / ensemble specialization |
| Ring | Sparse cell-to-cell neighbor traffic | Distributed recurrent workspace |
| Hierarchical | Local cells feed one or more coordinator cells | Local-to-global formation |
| Small-world | Local rings plus a few long-range bridges | Efficient propagation without global mixing |

## Transformer controls

Four-layer one-shot Transformers trained for 5,000 steps:

| Parameters | Held-out accuracy |
|---:|---:|
| 806 | 8.21% |
| 2,204 | 8.32% |
| 3,698 | 8.47% |
| 10,484 | 8.3% |
| 20,726 | 8.5% |
| 39,758 | 8.3% |
| 1,006,154 (20k steps) | 8.3% final; 8.8% best |

The 20,726-parameter Transformer memorized 100 fixed examples perfectly by step 600 but achieved only 8.6% on fresh held-out examples.

Four-million- and sixteen-million-parameter curiosity screens did not cross the
20%-by-10k candidate threshold in observed training, but complete final logs
were not preserved and no precise result is assigned to them here.

## Language configurations and runs

| ID | Parameters | Embedding | Cells | Width | Lexical mode | Node mask | Budget / result |
|---|---:|---:|---:|---:|---|---|---|
| GATOR-LANG-WIDE-PILOT | 130,000 | 32 | 10 | 32 | Communal | Symmetric | 20k; approximately 43% token, PPL approximately 24; greeting-confounded |
| GATOR-LANG-62-HASH-ASYM-S0 | 129,304 | 32 | 62 | 8 | `private_hash` | Asymmetric | 100k; 27.5%, loss 3.7253, PPL 41.48 |
| GATOR-LANG-62-HASH-SYM-S0 | 129,304 | 32 | 62 | 8 | `private_hash` | Symmetric | 42k; 27.3%, loss 3.7499, PPL 42.52; interrupted |
| GATOR-LANG-62-COMMUNAL-SYM-S0 | 121,864 | 32 | 62 | 8 | `communal` | Symmetric | Early run not obviously improved; long-row crop confounded |
| GATOR-LANG-62-COMMUNAL-SYM-CTX512-S0 | 121,864 | 32 | 62 | 8 | `communal`, shared input | Symmetric | Exploratory; only approximately 1–3 formed words |
| GATOR-LANG-62-COMMUNAL-SYM-CTX64-SMOL-S0 | 121,864 | 32 | 62 | 8 | `communal`, shared input | Symmetric | SmolTalk-only exploratory run; same short-formation collapse |
| GATOR-LANG-62-COMMUNAL-GATED-SMOL-S0 | 123,848 | 32 | 62 | 8 | `communal` + per-cell feature gates | Symmetric | Plausible wrong answers returned; formations capped around 4–5 displayed words |
| GATOR-LANG-62-COMMUNAL-GATED-GROUPED-SMOL-S0 | 124,616 | 32 | 62 | 8 | Feature-gated input + four-channel grouped readout | Symmetric | 50k complete: best 32.7% / PPL 42.36 at 13k; final 30.2% / PPL 62.85 |
| GATOR-LANG-62-COMMUNAL-GATED-GROUPED-SMOL-A-S0 | 124,616 | 32 | 62 | 8 | Feature-gated input + four-channel grouped readout | Asymmetric | 50k complete: best loss/PPL 3.7607/42.98 at 18k; final 30.7% / PPL 59.40 |
| GATOR-LANG-62-COMMUNAL-GATED-GROUPED-SMOL-ALT-S0 | 124,616 | 32 | 62 | 8 | Feature-gated input + four-channel grouped readout | Hybrid alternating | 50k complete: best 32.7% / PPL 42.36 at 12k; final 30.1% / PPL 62.00 |
| GATOR-LANG-62-COMMUNAL-GATED-GROUPED-SMOL-HEMI-S0 | 124,616 | 32 | 62 | 8 | Feature-gated input + four-channel grouped readout | Hybrid hemispheric | 50k complete: best PPL 43.12 at 13k; final 30.4% / PPL 56.89 |
| GATOR-LANG-62-COMMUNAL-GATED-GROUPED-SMOL-HEMI-N8-S0 | 124,616 | 32 | 62 | 8 | Same; 8 neighbors, 4 relays | Hybrid hemispheric | 25k: best/final PPL 43.12/44.18; faster cosine decay |
| GATOR-LANG-62-COMMUNAL-GATED-GROUPED-SMOL-HEMI-R6-S0 | 124,616 | 32 | 62 | 8 | Same; 4 neighbors, 6 relays | Hybrid hemispheric | 25k: best/final PPL 42.50/43.42; 52.48 steps/s |
| GATOR-LANG-1038-COMMUNAL-GATED-GROUPED-SMOL-HEMI-S0 | 999,112 | 32 | 1,038 | 8 | Feature-gated input + four-channel grouped readout | Hybrid hemispheric | 25k: best/final PPL 42.16/43.02; 7.05 steps/s; no material span gain |

The wide pilot was useful for pipeline validation but is not the preferred
scaling direction. The 62-cell models preserve width 8 and increase the
population to 248 graph-node instances. The completed asymmetric run and
interrupted symmetric private-hash screen did not reproduce the wide pilot's
plausible multi-token formations. Because
the symmetric private-hash screen closely tracked the asymmetric one, lexical
organization is now being tested before hybrid node masks.

The clean lexical A/B deliberately leaves the communal model under budget. It
removes 62 private `16 x 8` tables and adds 62 cell-identity vectors, reducing
the system by 7,440 parameters while holding the 62-cell graph fixed. If
communal wins, a later budget-matched run can spend the released parameters on
additional cells.

The feature-gated run isolates a different bottleneck. It adds one learned
32-feature gate per cell before the shared `32 -> 8` projection. Gates start at
one, so initialization is exactly identical to the shared-projection model;
the 1,984 new parameters only allow cells to develop different views of the
communal embedding during training.

Feature gating restored plausible contextual language but did not restore long
formations. This separates semantic input bandwidth from formation output
bandwidth. Inspection then exposed the matching output bottleneck: all 62
width-8 states were globally pooled to one width-8 vector before the 8-to-32
readout projection. The grouped-output run pools four interleaved cell groups
separately and concatenates their width-8 states, restoring a full width-32
population output.

That readout is still a severe compression rather than 62 independent output
heads. At 62 cells, the four groups contain about 15--16 cells each and reduce
each group to one width-8 vector. At 1,038 cells, each group contains about
259--260 cells but still reduces to one width-8 vector. Uniform initialization
therefore lowers each cell's direct group contribution from roughly `1/16` to
`1/260`, another approximately 16-fold credit dilution, while the
degree-four/four-relay communication field remains only 17 cells. This gives a
mechanical explanation for why scaling from 62 to 1,038 cells barely moved the
PPL floor: parameter count grew, but output bandwidth, communication diameter,
and per-cell terminal credit did not.

Across these language screens, formed span expanded progressively with
training: runs typically began with one word and added words only after further
optimization. Record formation length as a time series in tokenizer pieces and
displayed words. This is an observed simultaneous-decoding behavior, not yet a
claim that Gator has an intrinsic length-growth law.

The grouped-output run improved held-out PPL from 54.04 at step 1k to a best
logged 42.36 at step 13k. Random samples included context-appropriate
openings such as `You're welcome! I hope`, `It's recommended`, and `No,`, while
other answers remained generic or wrong. This is encouraging at 124,616
parameters but must be validated with shuffled- and blank-context controls.

Training well past the best region harmed held-out performance: final accuracy
was 30.2%, loss 4.1408, and PPL 62.85 at 50k. The run completed in 687.26
seconds at 72.75 steps/second. Compare topologies using each run's best
checkpoint and threshold-crossing time, not final 50k values alone.

The matched asymmetric run completed in 686.61 seconds at 72.82 steps/second.
It did not improve peak quality: its best PPL was 42.98 at 18k versus 42.36 at
13k for symmetric. The two curves were nearly identical at 30k (47.39 versus
47.35 PPL). Asymmetric finished with a lower PPL than symmetric (59.40 versus
62.85), so the completed logs do not support a blanket claim that asymmetric
was less stable. They do show that it learned the best solution later and did
not match the symmetric peak.

Hybrid alternating matched the symmetric PPL floor at displayed precision
(42.36 at 12k) but regressed faster through the middle, reaching PPL 50.43 at
31k. It finished at 62.00 PPL and 30.1% accuracy. This supports treating node
topology primarily as a convergence and stability lever under the current
124,616-parameter setup, rather than a lever that changes peak capability.

Hybrid hemispheric had the slightly higher best PPL of 43.12 at 13k, but it
held the best matched retention thereafter: 44.48 at 25k, 47.52 at 31k, and
56.89 at 50k. It crossed two PPL above its minimum at step 26k, later than the
other clean topology curves. Its complete runtime was 683.38 seconds at 73.17
steps/second. This is the strongest evidence in the sweep that topology changes
training stability more than the reachable language-performance basin.

Operational decision after the topology sweep: use hybrid hemispheric as the
default for subsequent configuration screens because it retained validation
quality best. Keep symmetric as an active contender because it achieved the
best single-point loss. After neighbor count and relay count produce a
meaningfully different configuration, benchmark both masks again; topology
effects may interact with communication breadth and recurrent depth.

The first breadth/depth screens favor computation depth. Eight neighbors
reproduced the PPL 43.12 floor without lowering it. Six relays at four neighbors
lowered the hemispheric minimum to 42.50 and finished the 25k run at 43.42,
with no added parameters. Neighbor-16 was deferred because neighbor-8 showed no
capability gain. Continue the relay sweep before revisiting wider connectivity.

Pinned architecture investigations after the first one-million-parameter
launch: (1) dynamic slot-conditioned cell voting versus equal/current-static
voting, and (2) activation-memory scaling. The 1,038-cell model used roughly
16 GB at vocabulary 2,048 and context 64, similar to the observed 16–17 GB of a
much smaller model at context 512. The absolute fit is comfortable on the
current Blackwell GPU, but the memory-to-parameter ratio is not suitable for
large-scale projections without checkpointing, fusion, chunking, or sparse
cell activation.

The one-million-parameter population screen did not justify further dense size
scaling. It reached a technically new best PPL of 42.16, but remained in the
same short-formation basin and ran at 7.05 steps/second. Return to 62 cells for
diagnostic iterations. Population scaling resumes only after readout routing,
formation span, training-floor behavior, and sparse stewardship are better
understood.

Historical design clarification, now realized by the later Gator-S checkpoint:
the dense language implementation represented roughly
three quarters of the intended architecture. The final direction gives each
token sparse persistent stewardship coordinates `(cell, node)` and executes
only occupied cells/nodes. Tokens sharing or approaching nodes influence one
another through local memory and traffic; tokens then migrate synchronously.
The current all-cell slot replication and population vote are prototype
mechanisms, not intended large-scale execution semantics.

Design motivation: Gator is intended to approximate sparse, shifting
brain-inspired activity at an abstract computational level. A thought recruits
relevant areas rather than activating the full population uniformly. Treat this
as motivation for conditional computation and specialist coalitions, not as a
literal neuroscience-equivalence claim.

Fixed-subset diagnostic result: the 124,616-parameter hemispheric model reached
100% token/exact training accuracy on pools of 10, 100, 200, and 400 examples
at steps 4k, 18k, 31k, and 80k respectively. An 800-example pool reached 96.8%
token and 48.0% exact accuracy at 200k, with training loss/PPL 0.1675/1.18.
Held-out loss simultaneously rose to 42.0929 and qualitative output eventually
degenerated. This rules out a hard simultaneous-slot or native-readout failure
at these pool sizes, while demonstrating catastrophic specialization under a
narrow deterministic training lane.

Clean LMSYS pipeline result: a strict policy requiring both the complete prompt
and complete response+EOS to fit retained only about 12.8% of sampled LMSYS
candidates and was rejected as unnecessarily destructive. The production policy
requires a complete prompt after whole-turn trimming but treats the first 32
response tokens as the simultaneous target window. With vocabulary 2,048,
context 64, a 512-character row cap, ASCII-letter ratio 0.90, minimum response
length 8, user-prompt requirement, and per-traversal deduplication, the local
real-data smoke test retained 130/210 LMSYS candidates (61.9%), 44/76 LMSYS
validation candidates (57.9%), and 5,320/5,726 SmolTalk training candidates
(92.9%). Streaming completed its first tested batch without restarting. This is
a data-pipeline correction, not an architecture result.

Matched language baseline launched: `TRANSFORMER-LANG-130K-CLEAN-S0` uses four
fully noncausal encoder layers, width 38, two heads, FF width 67, no dropout,
and 130,072 parameters. Its 64 context states and 32 learned empty output
queries share one encoder pass; no correct response token is supplied and no
prediction is fed back. All cleaned-corpus, tokenizer, mixture, batch, optimizer,
loss, and evaluation settings match the 62-cell run. A discarded 250-step
launch pilot reached PPL 54.34 and 28.2% token accuracy at 2.54 steps/second on
the local CPU. The corrected 100k run logs every 250 steps and selects
`best.pt` by minimum validation loss rather than maximum raw token accuracy.

Final matched Transformer result: stopped by operator at step 22,000 after the
best and latest evaluation both reached loss 3.4485 / PPL 31.45, token accuracy
31.4%, and exact accuracy 0.0%. Runtime throughput reached 6.03 steps/second on
the local CPU. The Transformer therefore surpassed the 130k Gator's reported
best PPL 32.8 with far fewer steps. This is a valid PPL result, but greedy probes
remained dominated by short generic openings and whitespace. At step 15k / PPL
31.92, fixed prompts produced `I m sorry` for the season question, repeated `2`
tokens for `2 + 2`, `Here` for the animal question, and a complete memorized
greeting for `Say hello`. Record the baseline as winning raw cross-entropy while
failing the current qualitative context-formation bar; content-only likelihood,
formation coverage, and whitespace-omission diagnostics are required before a
stronger architectural comparison.

Whitespace intervention implemented: the initial clean Gator A/B adds a 0.10
penalty on whitespace probability at active non-whitespace target slots and a
0.05 penalty on adjacent whitespace probability inside scored target pairs.
Padding after EOS is excluded, isolated normal spaces are not penalized, and
aggregate validation PPL remains ordinary cross-entropy. New logs/checkpoints
include content-only PPL/top-1/top-5, whitespace omission and probability mass,
adjacent-whitespace rate, formation coverage, raw training CE, and both
auxiliary terms. Synthetic high-whitespace tests produced omission/adjacency
penalties 0.99997/0.99994 with finite gradients; whitespace confined to padded
slots produced approximately 1.1e-6/1.3e-12. A one-step real-corpus smoke test
completed through training, evaluation, diagnostics, and checkpoint writing.

Status: increased relay depth is a promising candidate, not a settled default.
Before adoption, measure whether its PPL improvement survives matched scheduler
controls and seeds, and compare best PPL per wall-clock second, per training
example, and per relay operation. Preserve four relays as the baseline until
that speed/compute trade-off is quantified.

Inspection of the logged string representations showed that the model filled
many remaining slots with standalone whitespace tokens. The tokenizer makes
spaces an easy, frequent class, so the 32.7% raw token accuracy is
whitespace-confounded. Preserve the run as evidence that the architecture can
produce plausible content, but do not use its raw accuracy as the corrected
topology baseline.

### Historical clean topology sweep (completed)

Under the deliberately unchanged tokenizer version 1, change only
`node_topology` in this order:

1. `asymmetric`
2. `hybrid_alternating`
3. `hybrid_hemispheric`

Every run remains at 124,616 parameters with the same SmolTalk split, seed,
batches, context 64, output 32, 62 width-8 cells, feature-gated input, grouped
width-32 output, optimizer, and step budget. Earlier topology results used the
noisy/truncated data path and shared I/O bottlenecks and are not valid members
of this corrected A/B.

The asymmetric run began after the symmetric reference with no other declared
change. Standalone whitespace remains a shared measurement confound across the
sweep; it does not invalidate relative mask comparison, but raw accuracy must
not be treated as semantic accuracy. At the time, tokenizer version 2 was
deferred until all four masks finished. The current version-2 requirement is
the stable-ID BLANK contract in the formation-alignment roadmap, not an
unlabeled tokenizer substitution.

### Language hardware notes

- Completed 100k private-hash run: 18,582.35 seconds, 5.38 steps/second.
- Operator estimate for a 50k screen on T4: approximately four hours.
- Operator estimate for a 50k screen on RTX Pro 6000: approximately forty minutes.
- Treat the apparent approximately 6x improvement as provisional until the
  exact same checkpoint, batch, precision, compilation, and software stack are
  benchmarked on both devices.
- Compare architectures by time-to-matched held-out quality. High throughput
  at chance accuracy is not useful algorithmic performance.

## Invalid or incomplete runs

- Early GTR variant and relay-sweep results that attempted to distinguish AW from ARW are invalid because the parser enabled relative reads for all `gtr*` names.
- The approximately 79,826-parameter four-layer Transformer screen was interrupted and has no valid result.

## Gator-S stewardship implementation checkpoint (2026-07-16)

The first token-stewardship architecture is implemented behind
`cell_transport_mode=stewardship`. It replaces the static population
mix/vote shell with a learned five-action handoff law (`stay` plus four ring
neighbors) while retaining the validated Arrow cell internals. At the reference
62-cell communal, feature-gated, grouped-output configuration it has 124,618
parameters, two more than the 124,616-parameter broadcast control.

Each token begins with uniform ownership across cells. A source steward applies
its Arrow transform first, then distributes ownership, branch state, and node
passport together. Different destinations retain distinct attempted
computations; arrivals at the same token/cell merge by route-mass-weighted
averaging. Absolute population traffic, rather than per-cell-normalized
occupancy, is now exposed to the handoff router so a nearly idle cell and a
crowded cell are distinguishable.

Routing is elastic. A handoff-entropy signal raises adaptive nucleus mass for
uncertain work, so a live route can fan out to several candidate stewards and
those candidates can route again on later relays. Confident routes narrow to an
exploitation nucleus, but one configurable low-mass scout remains outside it;
pruning is therefore not an absorbing decision. This is a route-uncertainty
novelty proxy, not yet proof of semantic out-of-distribution detection.

The final implementation applies this rule twice. Every source steward has a
local elastic action beam, so an uncertain low-mass scout can independently try
`X` legal exits. Maximum entropy among a token's live stewards controls global
reopening; ownership-weighted mean entropy remains the auxiliary familiarity
signal. Left-padded context positions carry zero ownership and are excluded
from packed support, route diagnostics, and auxiliary losses.

Two execution kernels share the same model parameters:

- `dense` is the semantic oracle and computes every candidate cell.
- `packed_auto` switches below a live-support threshold, groups only occupied
  `(batch,cell,token)` records, and applies memory/node transforms only to those
  records. The current packed dispatcher is eager, not a fused compiled kernel.

Validation at this checkpoint:

- 21 unit/integration tests pass.
- Ownership, state, and node passport conservation pass collision cases.
- Dense and packed full-model logits/traces agree within `2e-6`.
- Packed forward/backward produces finite, nonzero handoff gradients.
- Legacy broadcast checkpoint round-trip is bitwise exact.
- A two-step final-code real-Smoltalk CPU smoke deliberately crossed a 50%
  warmup boundary: step 1 trained at temperature 1.5 with pruning off; step 2
  trained at 0.35 with pruning on. Both checkpoint evaluations used the fixed
  final 0.35/pruned policy. Training, diagnostics, best/latest checkpointing,
  and packed forward/backward completed. At step 2, evaluation support moved
  from 6.0 to 5.70 to 5.61, local handoff beams averaged 2.54 and 2.60 exits,
  and 10% of second-relay token routes widened. This demonstrates that support
  changes are not monotonic by construction; it is a mechanism smoke result,
  not language-quality evidence.

## Language formation-alignment implementation checkpoint (2026-07-16)

The next language lever is the output objective, not another simultaneous
topology change. The implementation now includes stable-ID tokenizer V2
BLANK, forbidden-special response masking, exact padding isolation, explicit
complete/right-censored metadata, four selectable formation objectives,
aligned/edit diagnostics, and an optional sequence-quality auxiliary.

The controlled matrix is:

| ID | Change from previous stage | Status at this checkpoint |
|---|---|---|
| `FORM-0` | Existing masked rigid slot CE | Implemented historical baseline |
| `FORM-AUDIT` | Verify PAD/EOS/whitespace/rendering/legal-response masks | Implemented; unit suite passes |
| `FORM-B` | Reserve a stable-ID non-rendering BLANK in the requested vocabulary; rigid all-slot BLANK-tail CE | Implemented as `blank_ce` |
| `FORM-METRIC` | Add best-monotonic-aligned accuracy and Viterbi/forward-sum loss as diagnostics only | Implemented in every evaluation |
| `FORM-V` | Train detached Viterbi aligned CE | Implemented as `viterbi` |
| `FORM-M` | Train differentiable monotonic forward-sum loss | Implemented as `monotonic` |
| `FORM-MQ` | Add low-weight composition/length/EOS sequence auxiliary | Implemented behind an independent weight; evidence pending |

The tokenizer update does not renumber existing compact-tokenizer IDs. The
version-2 format reserves an instance-specific final `blank_token_id`; old
checkpoints remain loadable but cannot silently opt into aligned objectives.
Monotonic alignment inserts BLANK while consuming every reference piece exactly
once and in order, so adjacent identical text pieces remain distinct.

Lock model, seed, corpus and batch order, output slots, reference-token cap,
optimizer, stewardship schedule, and checkpoint policy across this matrix.
Record rigid metrics at every stage for continuity. Aligned NLL, aligned
content F1/exactness, EOS correctness, post-EOS violations, BLANK/whitespace
rates, wall time, and peak VRAM are separate outputs. The unit suite and
end-to-end trainer/checkpoint-reload smokes are finite. No controlled
language-quality result exists yet, and implementation is not evidence that
the new objective will improve language. See
[`GATOR_FORMATION_ALIGNMENT_ROADMAP.md`](GATOR_FORMATION_ALIGNMENT_ROADMAP.md)
for equations, safeguards, and claim boundaries.

Right-censored references now leave the unknown continuation unscored in rigid
BLANK CE and use a free absorbing continuation state in Viterbi/forward-sum
alignment. Complete references still require BLANK after EOS. Both output-slot
and required-token-normalized aligned NLL are logged, along with content
precision/recall/F1 and optimizer fail-fast protection against non-finite loss
or gradients.

Final local validation passed 57 tests, including BLANK-transparent UTF-8 byte
fallback, true right-censored continuation, documented edit/content metrics,
and optimizer non-finite fail-fast. The end-to-end smoke trained
`formation_objective=monotonic` with 32 reference pieces across 40 physical
slots, sequence weight 0.01, finite gradients/losses, forbidden-special rate
0.000, v2 checkpoint save/reload, and diagnostic slot rendering. The smoke
used a tiny random CPU model and is an integration check only.

A final Gator-S smoke also exercised packed stewardship on every relay, shared
one response log-softmax across all formation losses/metrics, left an 8-piece
censored prefix free after consumption inside 12 physical slots, saved and
reloaded the V2 checkpoint, and reproduced the saved stewardship evaluation
policy in `language_probe.py`. It completed with finite gradients, zero
forbidden-token predictions, and distinct output-slot versus required-token
aligned losses.

## 2026-07-17: first aligned-language turn and overnight extension

The 124,618-parameter Gator-S monotonic run showed its first coordinated
validation improvement after step 12k. At step 16k it reached aligned loss
2.8619, required-token aligned loss 3.7459, required accuracy 30.0%, aligned
content accuracy 8.5%, content F1 14.5%, and edit accuracy 31.0%. Each was at
or effectively tied with the run's best value. A refusal sample also became
contextually appropriate while remaining grammatically damaged. This updates
the interpretation from "alignment operational but mostly exploited" to
"weak semantic learning is now visible." It does not remove the under-length,
repetition, sparse-EOS, or zero-exactness caveats.

Tulu3 SFT and SmolTalk Preferences were audited as compatible flat transcript
JSONL sources. The first 20,000 rows of each parsed without error. The trainer
now supports named weighted chat streams, per-source counters/restarts, fixed
legacy validation, and checkpoint continuation that preserves model/tokenizer
and optionally optimizer moments while restarting the LR schedule.

The overnight configuration resumes `best.pt`, trains 300,000 additional
steps at LR 0.001 cosine-decayed to 0.0001, and samples 10% SmolTalk Everyday
plus a 90% streamed share split LMSYS:Tulu3:SmolTalk-Preferences at 4:3:2.
No complete-response or alignment-shift lever changes in this capability run.
The local integration gate now passes 59 tests plus a scratch-to-resume
three-corpus smoke.

## 2026-07-17: performance milestone 1-2 implementation checkpoints

The first scaling pass changed execution only; it did not alter learned
parameters or the Arrow equations.

Milestone 1 replaced the dense receiver/writer attention materialization with
an exact normalized grouped one-dimensional convolution. The quadratic
attention path remains the semantic oracle. Forward values and gradients match
in the equivalence suite, and the representative local CPU microbenchmark was
1.91x faster. The primary GPU hypothesis is reduced allocation: at
`B=32,C=62,L=104,N=4`, the former score tensor alone contains 85,835,776
values per relay. This checkpoint is commit `a9d3a48`.

Milestone 2 moved dense-to-packed selection outside the relay loop. The
trainer now compiles the static dense path even when policy is `packed_auto`.
A checkpointed controller periodically observes a four-value support vector,
chooses the earliest relay whose complete remaining suffix stays below the
configured break-even fraction, and latches a static dense-prefix/packed-suffix
plan. Packed calls remain eager.
This removes the previous per-relay GPU-to-CPU boolean checks and prevents
`packed_auto` from disabling compilation for the entire broad-exploration
phase.

Training diagnostics were split into profiles. Ordinary optimizer steps retain
only ownership, mean route uncertainty, and valid-route masks; evaluation
retains route metrics but omits location and hidden histories; full trace mode
remains available for inspection. Relay-step projections and dense address-edge
projections are now computed once per forward rather than once per relay.

Forward equivalence does not imply sparse backward equivalence after
straight-through pruning. Dense execution may preserve surrogate gradient
through a zero-forward route that packing omits, so the transition is an
explicit estimator A/B and must be evaluated by learning curve as well as
logits. The resumed-run throughput display was also corrected to use local
completed steps. The previous apparent decline from an inflated initial rate
was a logging artifact.

## 2026-07-17: performance milestone 3 implementation checkpoint

Milestone 3 added an exact folded dense-node backend and a reusable hardware
compute-policy API. It changes execution only. Parameter count, learned node
functions, routing equations, and checkpoint model state are unchanged.

For every cell/node affine transform,

```text
W(base + node_embedding) + bias
    = W(base) + (W(node_embedding) + bias).
```

The `folded_bmm` backend combines four width-8-to-width-8 node transforms into
one width-8-to-width-32 projection per cell, reshapes the result back to four
nodes, and adds a live folded offset computed once per forward. This removes
the materialized `[B,C,T,N,D]` node-input tensor. At
`B=32,C=62,T=104,N=4,D=8`, the avoided tensor contains 6,602,752 FP32 values,
or 25.2 MiB, per relay before backward storage.

`benchmark_dense_kernels.py` verifies numerical equivalence and times the
isolated kernels. On the local CPU at `B=4,C=62,T=104,N=4,D=8`, the expanded
node kernel measured 9.9242 ms and the folded kernel 2.6216 ms, a 3.79x
forward-kernel speedup. Grouped router BMM was only 1.13x faster and remains
opt-in; `einsum` is the default. Padding narrow operations to a multiple of
eight measured 3.0146 ms versus 1.1743 ms with padding one, so
`dense_kernel_padding=1` remains the default. These are isolated CPU
microbenchmarks, not end-to-end or GPU performance results.

The public `gator.compute` API and shared CLI now expose:

- `--dense-router-backend`
- `--dense-node-backend`
- `--dense-kernel-padding`
- `--float32-matmul-precision`
- `--cudnn-tf32` / `--no-cudnn-tf32`
- `--compile-mode`
- `--compile-shape-padding` / `--no-compile-shape-padding`

The requested policy and resolved device capability are stored as runtime
metadata in checkpoints. The designated Blackwell benchmark policy is FP32
matmul precision `high` plus cuDNN TF32, but no Blackwell measurement is
claimed at this checkpoint. AMP remains deferred: the current `-1e9` route
mask overflows FP16 and the `1e-9` entropy clamp underflows it. A safe future
AMP implementation needs FP32 routing, softmax, entropy, and language/alignment
loss islands.

## 2026-07-17: performance milestone 4 implementation checkpoint

**Commit:** milestone 4 rollback point (see repository history)

Milestone 4 replaces relay-by-relay sparse repacking and global receiver-group
padding with a persistent bucketed stewardship backend. The new default is:

```text
packed_stewardship_backend=persistent_bucketed
```

The CLI equivalent is
`--packed-stewardship-backend persistent_bucketed`. The prior implementation
remains selectable as `repack` and is retained as the sparse forward numerical
oracle. With pruning disabled its gradients also agree. With
straight-through pruning, physically dropped sparse rows intentionally lose
the zero-forward surrogate gradients retained by the dense representation.

At the selected dense-to-sparse boundary, the persistent backend packs live
`(batch, token, cell, state, node passport, ownership mass)` branches once.
Uncertainty measurement, ownership reconstruction when required, adaptive
nucleus pruning, weighted traffic summaries, support-only handoff transport,
collision merging, and final shared/grouped readout operate directly on the
packed representation. Ordinary no-diagnostics execution does not reconstruct
dense state or passport tensors between sparse relays. Full diagnostics may
still unpack explicit snapshots by request.

Receiver groups are assigned fixed power-of-two-like capacities instead of
padding every group to the busiest group. For live group lengths `L_g` and
`M=sum_g L_g`, each assigned capacity is below `2 L_g`; total padded row
capacity is therefore below `2M`, and pairwise receiver work is below
`4 * sum_g(L_g^2)`. Cell-local router and node projections similarly
bucket rows by active cell and gather only those cells' parameter slices.
Handoff transport receives the retained support mask and instantiates only live
actions instead of materializing every legal action for every packed row.
These are structural execution/allocation bounds, not measured training
speedups.

The controller semantics remain static and checkpointable. Complete dense
exploration calls before a sparse plan is latched remain eligible for
`torch.compile`. Once the plan is latched, the packed model invocation is eager
in full, including any dense prefix before the packed boundary. The current
trainer does **not** stage a compiled dense prefix and eager packed suffix
inside one call.

Known limits are explicit:

- Initial ownership is uniform and dense, so sparse savings begin only after
  pruning learns useful support.
- Dynamic `nonzero`, sorting, and `unique` remain eager and may synchronize.
- The node passport remains soft; every active cell still evaluates all four
  node transforms.
- `benchmark_sparse_stewardship.py` compares forward-only eager sparse
  implementations.
  Its tiny local CPU configuration is a harness/correctness check and cannot
  establish a GPU or end-to-end advantage.
- No target-GPU throughput or memory speedup is claimed at this checkpoint.

Ordinary training diagnostics now reduce entropy, per-cell load, uncertainty,
and active support directly from packed rows instead of stacking a dense
`[B,R+1,C,T]` ownership history.

Local validation passes the full 106-test suite. A scratch language trainer
smoke latched a packed suffix and completed finite forward/backward optimizer
steps, and a checkpoint-resume smoke restored that execution plan and
continued training. These are implementation checks, not language-quality or
performance results.

## 2026-07-17: performance milestone 5 implementation checkpoint

**Commit:** milestone 5 rollback point (see repository history)

Milestone 5 moves fixed-corpus collation, background preparation, device
transfer, and data-state continuity behind reusable APIs. It changes execution
only; token windows, sampling probabilities, model equations, parameters, and
losses are unchanged.

`LanguageBatcher` now pre-collates each finite corpus and selects complete
examples with one tensor row gather. `PackedLanguageSource` keeps its fixed
int32 storage tensorized through mixture selection rather than converting rows
back to Python token tuples and collating them again. Historical Python
`random.Random` draw order is preserved for clean A/Bs.

`LanguageBatchPipeline` supplies serial or bounded single-producer prefetch.
Queued batches remain on the CPU, may be page locked, and can transfer through
a dedicated CUDA stream with event-based source lifetime tracking. It exposes
queue, sampling, transfer, and failure statistics to the trainer. Ordered
thread and spawned-process preparation is separately available through
`gator.preparation`; its pending work is bounded and results are yielded in
source order.

The current compute stream waits for each staged copy before forward, so this
milestone overlaps CPU preparation but does not yet implement device-batch
lookahead. Reported transfer timing is host enqueue latency; CUDA-event H2D
timing belongs in the target-GPU profile.

Checkpoint semantics are deliberately conservative. Finite and packed
providers serialize RNG state, nested mixture state, and prefetched batches in
future consumption order. The trainer fingerprints input file identity,
tokenizer, task window, finite examples, filtering, mixture weights, and seed.
Generic streaming iterators and shuffle buffers are explicitly non-exact.
`resume-data-state=auto` therefore resets their seeded data phase and labels
that choice, while `exact` fails rather than claiming continuity.

The forward-free local CPU benchmark at batch 32, context 64, response 32
reported:

| Input path | Batches/s | Relative to legacy |
|---|---:|---:|
| Legacy per-example collation | 559.06 | 1.00x |
| Vectorized finite batcher | 4,375.49 | 7.83x |
| Direct packed source | 2,981.37 | 5.33x |
| Serial pipeline wrapper | 2,334.25 | 4.18x |
| Prefetch depth 2 | 2,239.21 | 4.01x |

All paths produced the same deterministic checksum in a 500-iteration,
4,096-example run after 50 warmups. Prefetch was expectedly slower than direct
gathering without any model work to overlap, so this is evidence for removing
Python collation overhead, not an end-to-end GPU training-speed result.
`benchmark_language_pipeline.py` makes that claim boundary explicit.

The local validation gate passes 139 tests. A two-step scratch language run
used the prefetched provider, logged queue/transfer statistics, saved its
pending queue and sampler state, and a one-step continuation reported
`data_resume=exact_restored`. Target-GPU serial/prefetch profiling remains
required before changing the queue defaults or claiming Gator throughput.

## 2026-07-17: next performance program pinned

The post-milestone-5 scaling work is now canonical in
[`PERFORMANCE_TODO.md`](../PERFORMANCE_TODO.md), using commit `742f004` as its
baseline. The ordered tracks are:

1. Fuse or replace receiver memory and its reductions.
2. Preserve a compiled dense prefix before a sparse suffix.
3. Use bounded packed workspaces and fused segmented dispatch.
4. Replace exact capped full-population sorts with bounded top-k.
5. Remove per-step host barriers and fuse optimizer/alignment work.
6. Eliminate avoidable massive tensors that are instantiated for one immediate
   consumer and then discarded.

Each item requires its own oracle, profiler evidence, 62-cell and 1,038-cell
measurements, tests, documentation, and rollback commit. This entry records the
plan only; none of the six items is claimed complete.

## 2026-07-17: current trainer optimization integration gate

The implemented milestone-1-through-5 execution paths were exercised through
the real `gator_language_train.py` optimizer loop on the local CPU. This was an
integration and crossover screen, not a Blackwell or production-shape
benchmark. All runs used SmolTalk Everyday, seed 123, slot CE, two relays, FP32,
and the same deterministic batch order. Compilation was disabled because the
available PyTorch runtime was CPU-only.

The first matrix used `B=2,C=6,T=12,N=4,D=8`, vocabulary 320, and 60 optimizer
steps:

| Execution | Steps/s | Validation loss | Accuracy | Result |
|---|---:|---:|---:|---|
| Attention + expanded nodes + dense + serial | 18.45 | 3.3217 | 0.156 | Numerical/trainer oracle |
| Convolution + folded nodes + dense + serial | 20.09 | 3.3217 | 0.156 | 1.089x; matched reported metrics |
| Optimized dense + CPU prefetch | 18.23 | 3.3217 | 0.156 | Deterministic, but overhead-dominated |
| Packed repack + serial | 19.16 | 3.3936 | 0.156 | Sparse oracle; different estimator trajectory |
| Persistent bucketed + serial | 16.73 | 3.2807 | 0.203 | Too little population for sparse payoff |
| Persistent bucketed + CPU prefetch | 15.29 | 3.2807 | 0.203 | Complete current stack; smallest-shape overhead |

The exact dense convolution/folded-node combination improved end-to-end local
trainer throughput by 8.9% while preserving loss, accuracy, and sampled output
at the reported precision. CPU prefetch reduced throughput by 9.3% relative to
optimized serial execution because sampling required only about 0.25 ms while
the bounded producer and transfer bookkeeping remained active. This does not
predict the result for a GPU-sized model where preparation can overlap a much
longer optimizer step.

A second screen increased the population to 62 cells while retaining a small
local window: `B=1,C=62,T=24,N=4,D=8`, embedding width 32, support capped at
four of 62 cells, and 20 steps:

| Execution | Steps/s | Active support | Result |
|---|---:|---:|---|
| Optimized dense + serial | 9.55 | 62 then 4/4 | Reference |
| Persistent bucketed + serial | 9.31 | 62 then 4/4 | Only 2.5% behind dense |
| Persistent bucketed + CPU prefetch | 7.96 | 62 then 4/4 | Host overhead still dominates |

The narrowing gap from 16.7% at six cells to 2.5% at 62 cells is preliminary
evidence of the intended sparse crossover, not proof of a speedup. Persistent
dispatch remained eager and dynamic. The next target-GPU gate must repeat this
at the real 62-cell and 1,038-cell batch/window shapes with CUDA-event timing,
compilation warmup, kernel counts, live support, and peak VRAM.

At step two, repack and persistent training reported identical task CE
(`5.7592`) but different gradient norms and then different learning
trajectories. This is the already documented estimator boundary, not a forward
equivalence failure: after straight-through pruning, persistent execution
physically removes dormant rows and therefore cannot retain the dense/repack
graph's zero-forward surrogate gradients. Packed forward values match their
oracle within the focused tests, and gradients match when pruning is disabled.
The receiver, dense-kernel, stewardship, execution-controller, and pipeline
gate passed 65 tests in 6.728 seconds.

## 2026-07-17: optimized Colab trainer bundle

The language trainer now exposes explicit parameter-free GPU execution
presets:

- `dense_gpu_safe` / `optimized_gpu_safe`: strict-FP32 dense and
  persistent-sparse policies without compilation.
- `dense_gpu` / `optimized_gpu`: the corresponding aggressive policies with
  compilation and FP32 Tensor Core/TF32 settings.

`manual` preserves the individual low-level CLI defaults. Explicit low-level
flags override preset values, the selected preset is printed in the startup
record, and it is stored in checkpoint data metadata. The previously observed
aggressive dense baseline is approximately 12 steps/second.

`prepare_gator_colab_bundle.py` now builds the archive from the complete current
root scripts/notebook, `gator` package, test suite, and paper set. This replaces
the stale hand-built archive, which omitted newer compute, execution, sparse
kernel, pipeline, and preparation modules. Training corpora and checkpoints are
excluded. An isolated extraction of the new manifest passed all 143 tests.

## 2026-07-17: aggressive GPU preset non-finite backward

The first Colab launch of `optimized_gpu` produced a finite forward loss but
the optimizer rejected a non-finite aggregate gradient norm before updating
parameters. The traceback did not identify the step or source parameter, so it
cannot yet distinguish a compiled-kernel issue, TF32 sensitivity, or an
existing backward instability exposed by different rounding.

The corrected bundle changes the notebook to `optimized_gpu_safe`, which keeps
the structural execution optimizations but disables compilation and uses
strict FP32 with cuDNN TF32 disabled. The staged isolation order is:

1. `optimized_gpu_safe`
2. `optimized_gpu_safe --compile`
3. `optimized_gpu`

Dense controls are `dense_gpu_safe` and `dense_gpu`. Explicit low-level flags
now override preset defaults, so individual runtime levers can be isolated.
Non-finite failures now report optimizer step, learning rate, parameter name,
shape, and NaN/+Inf/-Inf counts while still refusing the update and clearing
gradients. A local end-to-end safe-preset smoke completed through packed-plan
latching without a non-finite value. The corrected source suite passes 145
tests.

### Follow-up: strict FP32 plus compilation isolates the warmup boundary

The `optimized_gpu_safe --compile` Colab run also failed, despite resolving to
`float32_matmul_precision=highest`, `cudnn_tf32_effective=False`, and finite
startup values. The improved diagnostic located the failure at optimizer step
103 and reported widespread NaNs through the cell stack, including every
element of `input_feature_gates`, `cell_embedding`, `address_to_state`,
`step_projection`, `node_embedding`, `node_weight`, and `node_bias`.

This timing coincides with the 5% stewardship warmup boundary in the 2,000-step
smoke run: broad dense exploration is compiled successfully, then route pruning
turns on at approximately step 101. The evidence therefore points to the
compiled pruned/straight-through backward path, not TF32 and not the optimizer.

An initial safety fence compiled only unpruned exploration, but that would have
saved only the approximately 103 warmup steps and left nearly the entire run
eager. Inspection instead found two pruning-only non-finite hazards. First, the
convolution receiver divided empty memories by `torch.finfo(dtype).tiny` and
masked the value afterward. Although eager autograd cancels that dead branch,
compiler fusion can still encounter reciprocal-tiny and
denominator-squared-underflow intermediates. Second, dense route uncertainty
used `-inf` as the inactive-cell sentinel before a reduction.

The receiver now selects a unit denominator for empty memories *before*
division, then masks the finite zero result. Route uncertainty uses `-1`, which
is an exact finite exclusion sentinel because normalized entropy lies in
`[0, 1]`. Regression tests require finite zero forward and backward values for
an entirely empty receiver and finite gradients for an empty invalid route.
The temporary eager fence was removed: dense pruning remains compiled, and
only a latched ragged packed plan changes to eager execution. The next
Blackwell smoke must cross step 103 to validate the fix in the target compiler.

### Follow-up: step 1,904 exposes FP32 norm-reduction overflow

The next Blackwell run crossed the original boundary and trained normally
through step 1,000 at 13.56 steps/second with finite loss and a `0.7608`
gradient norm. It later stopped at optimizer step 1,904. Unlike the earlier
failure, the diagnostic found no NaN or infinite value in any parameter
gradient: PyTorch's aggregate FP32 L2 norm alone was non-finite. No packed-plan
transition had been logged.

This is consistent with reduction overflow: finite gradient elements can
overflow when squared and accumulated in FP32 before clipping is applied. The
optimizer now retains PyTorch's fast foreach clipping on ordinary steps. Only
when that reduction reports non-finite does it recompute per-tensor norms with
FP64 accumulation. If the stable norm is finite, it clips the still-finite
gradients and performs the update; genuine NaN/Inf gradient elements still
fail closed and clear the pending gradients. A regression injects `1e30`
finite gradients, verifies that their ordinary FP32 norm overflows, and checks
that clipping leaves model parameters and Adam state finite.

### Compile status frozen before Gator-Path

A later Blackwell run used the fully aggressive `optimized_gpu` preset with
both matmul and cuDNN TF32 enabled, compilation enabled, and broader
stewardship nuclei:

```text
stewardship_keep_mass=0.97
stewardship_handoff_keep_mass=0.97
```

It trained normally through the 2,000-step report, then stopped at optimizer
step 2,973 with genuine NaN gradient elements. The invalid counts map almost
exactly to one width-8 cell slice: 32 feature-gate values, 8 cell-embedding
values, 32 address/state values, and 256 node-weight values, with the same
localized pattern across other cell parameters. This is not the already-fixed
aggregate-norm overflow.

The available evidence establishes three facts but does not isolate one final
compiler defect:

1. Compiled pruning originally exposed dead-branch receiver normalization
   hazards near step 103; finite safe branches moved the failure substantially.
2. A later non-finite aggregate norm at step 1,904 contained only finite
   gradient elements and is handled by overflow-safe fallback clipping.
3. Aggressive compiled TF32 still produced a genuine localized cell failure at
   step 2,973.

Because Gator-Path removes Arrow's replicated receiver-memory and dense
all-cell execution graph, further tuning of this compiled graph is deferred.
The preserved Arrow fallback is:

```bash
--execution-preset optimized_gpu_safe
```

That policy uses strict FP32, no compilation, exact convolution receiver
memory, folded node transforms, bounded prefetch, and automatic packed
execution. `--compile`, `dense_gpu`, and `optimized_gpu` remain experimental
diagnostic controls rather than supported long-run training policies. The
optimizer must continue to fail closed on genuine NaN/Inf gradients; silently
skipping corrupt steps is not an accepted workaround.

## Gator-Path P4 single-cell prefix trainer

P4 adds the first task shell around the executable packet-path core.

```text
parameters=1218
cells=1
nodes=4
payload_width=8
address_width=4
relays=4 (shared parameters)
top_k=2 soft
task=simultaneous modular parallel prefix
length=7
```

All input and output positions are persistent `[B,L,8]` packets. Context
packets write immediately; output packets use the controlled schedule
`0, 1/3, 2/3, 1`. The relay now consumes both mailbox payload means and mailbox
address means, allowing a receiver to distinguish the sequence region
represented by local traffic. Output logits are decoded once after the fourth
relay through tied token embeddings plus a fresh immutable address projection.

The model retains the fixed five cell actions even with one cell, so offsets
alias to the same physical cell but remain distinct learned edge types. This
preserves the P5 action contract and explains why the proof model is 1,218
parameters rather than exactly 980.

A 100-step CPU sanity run completed at 8.97 steps/second:

| Step | Correct /7 | Accuracy | Held-out loss | Route entropy |
|---:|---:|---:|---:|---:|
| 0 | 0.62 | 8.9% | 3.3724 | 2.139 |
| 50 | 0.62 | 8.9% | 2.5050 | 1.836 |
| 100 | 0.52 | 7.4% | 2.5048 | 1.654 |

This confirms finite optimization and early target-alphabet learning only.
Algorithm acquisition must be evaluated from the 10,000-step T4 run on fresh
generated examples.

## 2026-07-18: mastered Arrow circuit inspection

The original saved 20,000-step vectorized checkpoints were loaded into the
locked dense reference and evaluated on the same fresh 512- or 1,024-example
length-seven batch. The new `analyze_arrow_checkpoint.py` trace reports node
location entropy, expected transition entropy, output votes, individual-cell
accuracy, and leave-one-out utility.

| Seed | Ensemble acc. | Effective output cells | Best single output cell | Ensemble gain | Final effective nodes |
|---:|---:|---:|---:|---:|---:|
| 0 | 99.67% | 2.065 | 69.00% | +30.66 points | 1.230 / 4 |
| 1 | 90.15% | 1.000 | 90.15% | +0.00 points | 1.453 / 4 |
| 2 | 41.10% | 1.957 | 26.56% | +14.54 points | 1.130 / 4 |

Seed 0's learned output weights were approximately `0.658` for cell 0 and
`0.320` for cell 4, with the remaining eight cells receiving approximately
2.2% combined. Neither major output cell solved the task alone; their mixture
did. This is direct evidence that dense Arrow discovered complementary
computation rather than ten mutually exclusive candidate routes.

Seed 1 found a different circuit: its output vote collapsed almost completely
onto cell 7, which reached the ensemble's 90.15% accuracy. This does not prove
the other cells were inactive because cell 7 had already integrated neighbor
messages across four relays. Final readout support and intermediate compute
support are different measurements.

All three checkpoints learned sharp node transitions. Seed 0's expected
effective outgoing edge count was only `1.02--1.06` across relays. Seed 2 was
also sharply routed but remained at 41.1%, proving that route sharpness alone
does not identify a good circuit. Sparse support must be learned from task
utility; forcing top-1 ownership at initialization can freeze a coherent but
weak circuit or remove complementary branches before credit differentiates
them.

Post-training pruning shows that entropy also understates the functional
frontier. Re-normalizing the mastered seed-0 location distribution after every
relay produced:

| Retained nodes per cell/slot | Token accuracy | Exact sequence |
|---:|---:|---:|
| 4 | 99.67% | 97.85% |
| 3 | 98.73% | 91.60% |
| 2 | 77.48% | 20.31% |
| 1 | 41.27% | 0.39% |

Although the unpruned distribution averaged only 1.23 effective nodes at the
final relay, low-mass branches carried important corrections. A safe frontier
cannot be selected from entropy or maximum probability alone. It requires
measured loss/quality sensitivity, learned distillation into retained paths,
or a contribution-aware pruning objective.

The mass audit makes the point sharper. At the final relay, top two locations
held 99.23% of location mass but pruning to them reduced accuracy to 77.48%.
Top three held 99.95% and retained 98.73%. Meanwhile, each occupied source's
outgoing transition averaged only `1.02--1.06` effective edges. The learned
program is therefore best described as a few complementary node strands whose
individual next hops are nearly deterministic—not one uncertain path and not
a dense wave over equivalent alternatives.

Final output sparsity was safer: retaining only seed 0's top two output cells
preserved 98.69% token accuracy in the top-three-node run. This is readout
sparsity, not proof that the other eight cells were computationally dormant
during the relays.

Intermediate cell ablation proves the distinction. Disabling cells 0 through
9 one at a time reduced the mastered checkpoint from 99.8% on the audit batch
to, respectively:

```text
23.6%, 25.8%, 78.9%, 43.9%, 69.0%,
48.1%, 83.0%, 27.6%, 29.0%, 24.4%
```

Even the least damaging removal cost 16.8 accuracy points. Several cells with
nearly zero final vote were among the most important intermediate relays.
Thus, the mastered ten-cell Arrow circuit is population-dense during
computation, approximately three-strand sparse inside each cell, and two-cell
sparse only at terminal readout. Static output votes cannot be used as an
active-cell mask.

A same-seed local eager-CPU reference rerun finished all 20,000 steps at 51.0%
token accuracy, 0% exact, and loss 1.3163. It plateaued near 51--52% through
most of the run, then experienced a step-20k gradient-norm spike to 31.35. The
run learned the same broad circuit shape: its two dominant output cells held
68.3% and 31.2% of vote mass. They reached 38.9% and 15.2% individually and
51.6% together. Removing any intermediate cell reduced accuracy; the smallest
drop was 11.6 points and the largest was 36.5 points. The mastered
checkpoint's corresponding two-cell mixture reached 99.7%. The phase
transition is therefore not simply "discover two cells"; it is continued
specialization and cooperation within a population-dense intermediate circuit
and sparse terminal readout.

The preserved mastered checkpoint was also evaluated without retraining under
all three inter-cell interpretations on the same 1,024 examples:

| Handoff | Accuracy | Exact | Loss |
|---|---:|---:|---:|
| OG destination gather | 99.665% | 97.852% | 0.010192 |
| Destination-owned carried arrivals | 99.665% | 97.852% | 0.010192 |
| Source-owned gated broadcasts | 10.659% | 0.000% | 14.808550 |

The corrected carried-arrival form therefore retains the mastered Arrow
function. The failed source-owned form does not merely learn more slowly; it
assigns learned gates to the wrong side of each handoff and destroys the
existing circuit.

The full corrected destination-carry eager-CPU run completed at 51.8% token
accuracy, 0.1% exact, and loss 1.0529, versus 51.0% / 0% / 1.3163 for the
locked local OG reference. On the common 1,024-example forensic batch it
reached 52.4%. Its circuit structure matched the OG plateau control:

- cells 5 and 9 held 78.4% and 21.4% of terminal mass;
- their continuous two-cell fusion reached 52.3%;
- every cell remained intermediate-ablation important;
- removing one intermediate cell cost between 5.1 and 35.5 accuracy points;
- node transitions used only about `1.04--1.07` effective outgoing edges.

Corrected carry therefore reproduces both performance and the characteristic
population-dense-intermediate/two-terminal circuit organization. It is no
longer an underperforming architecture; the remaining gap is the stochastic
51% basin shared with the local OG control.

This explains a major cell-scaling failure mode. In the ten-cell, degree-four,
four-relay ring, a terminal cell's receptive field covers the entire
population. In a 62-cell ring it covers at most 17 cells. If static output
votes again collapse to roughly two terminal cells, much of the enlarged
population sits outside the winners' causal neighborhoods and receives weak or
no useful terminal credit. More cells add parameters without adding reachable
computation. Future population scaling must change communication diameter,
readout routing, or hierarchy rather than only `C`.

The resulting execution rule is:

```text
train broad -> measure contribution -> narrow learned support -> execute sparse
```

not:

```text
initialize sparse -> hope the missing complementary circuit is rediscovered
```

The first convergent-lane implementation initially materialized
`[B, receiver, writer, lane, node]` memory scores and evaluated all 40 router
sources and node transforms for every lane. Two exact factorizations now keep
the forward and every tested gradient within tolerance:

1. aggregate lane mass/state per writer node before applying the
   receiver-relative kernel;
2. evaluate router rows and node transforms only for graph nodes reachable
   from each lane's fixed broad start after the current relay.

On local batch-128 smokes while the machine was under background training
load, interval throughput rose from approximately 0.47 steps/s to
1.41--1.56 steps/s after the exact sparse execution changes, then to about
2.1 steps/s after reallocating the model to private width-8 lane views and a
smaller local router. This is an implementation gain, not yet a target-GPU
benchmark or a capability result. The model still preserves ten distinct
payload lanes and full location tensors for its diagnostic oracle.

The one-cell/40-node configuration was previously screened for 5,000 updates
and reached 24.7% at 10,010 parameters. A matched seed-0 20,000-step extension
is now recorded separately to determine whether one large internal graph can
eventually reproduce the stitched-cell phase transition.

That historical flat model initialized almost all location mass on node 0.
Because its degree-three ring can move only four hops in four relays, the
screen did not expose most of the 40-node graph to the initial packet
frontier. The corrected factorization arm initializes uniformly across all 40
nodes, retaining the same 10,010 parameters and learning objective. It is the
proper broad-to-specialized flat comparison; the narrow-start 24.7% remains a
separate historical result.

Broad ingress still does not make the present one-cell model fully equivalent
to ten-cell Arrow. `SparseGator` keeps one hidden vector per token and mixes
all selected node outputs back into it after every relay.
`VectorizedGatorCellRing` keeps one hidden vector per token per cell, giving
ten persistent semantic lanes. The 1x40 run tests whether node breadth alone
can replace lane breadth. A future vision-complete flat control must keep
separate payload state for every retained trace.

### 2026-07-18: Convergent ten-trace carried-state result

The lane-matched flat control completed 20,000 portable-batch seed-0 updates:

| Metric | Convergent, 4 relays | Locked local OG Arrow |
|---|---:|---:|
| Parameters | 10,010 | 10,000 |
| Final token accuracy | 92.1% | 51.0% |
| Final exact accuracy | 61.0% | 0.0% |
| Final loss | 0.2211 | 1.3163 |
| Local CPU throughput | 3.56 steps/s | 1.86 steps/s |

This is the first carried-state design to exceed the matched local OG basin
while retaining the original approximately 10k parameter budget. It is the
new reproducible leader: OG Arrow could not reproduce its historical
compiled-CUDA 99%+ trajectory in the current environment. That preserved
checkpoint remains a useful upper-bound observation, while relay-depth and
seed controls test whether Convergent can also reach that older basin.

The terminal distribution narrowed from `7.89` effective lanes at step 1,000
to `3.04` at step 20,000. Final 4,096-example forensics found:

| Terminal fusion | Token accuracy | Exact accuracy |
|---|---:|---:|
| Best lane alone | 24.8% | 0.0% |
| Best two lanes | 44.3% | 0.4% |
| Best three lanes | 91.2% | 56.7% |
| All ten lanes | 91.5% | 58.1% |

The three dominant weights were `39.5%`, `31.5%`, and `28.6%`. This directly
confirms complementary trace convergence: the answer is not stored in one
winning route, and graph-node count alone is not a substitute for multiple
payloads.

#### Exact relay-kernel optimization audit

Relay profiling showed receiver mixing and the repeated small grouped
projections as the dominant eager-CPU work. The following changes preserve
the Convergent equation:

1. hoist receiver kernels, relay encodings, write schedules, and reshaped
   address-edge views out of the recurrent loop;
2. factor router semantics from occupancy so the expanded
   `[B,T,K,source,input]` tensor is never instantiated;
3. fold node embeddings through their affine transforms instead of repeating
   `base + node_embedding` for every reachable node;
4. combine receiver numerator and denominator into one batched matrix
   multiply.

A full committed-versus-optimized model comparison found a maximum forward
difference of `4.77e-7`; the largest observed relative gradient difference
was `5.17e-7`. The controlled single-thread batch-32 benchmark improved from
`7.12` to `8.82` steps/s at four relays and from `2.70` to `3.71` steps/s at
eight relays, approximately `24%` and `37%`.

This audit also caught a dangerous shape-compatible error: original
Convergent node weights use `[input, output]`, while the shared grouped helper
expects `[output, input]`. Because both axes have width eight, the incorrect
transpose produced valid shapes but substantially different behavior.
Explicit orientation and full-equation regression checks are now required
for all future kernel substitutions.

A normalized convolutional receiver implementation was also evaluated. It is
useful for longer language contexts, but at the prefix task's total length of
nine it was about `1.7x` slower than the quadratic attention kernel. It was
therefore rejected from the Convergent prefix hot path rather than retained
as unused complexity.

#### Learned route-support ablation

The final four-relay checkpoint reports only `1.39` effective node locations,
but entropy is not a safe physical-pruning threshold. Applying top-k
renormalization after every relay on one matched 1,024-example batch produced:

| Per-trace node support | Token accuracy | Exact accuracy |
|---:|---:|---:|
| Soft oracle | 91.45% | 58.20% |
| 1 | 20.54% | 0.00% |
| 2 | 49.71% | 0.78% |
| 3 | 72.64% | 14.84% |
| 4 | 87.71% | 46.58% |
| 5 | 91.16% | 57.13% |
| 7 | 91.45% | 58.20% |

Thus low-mass branches carry complementary corrections. Top-1 or top-2 would
repeat the original Path failure; top-5 is the first nearly lossless support
for this checkpoint. At the final relay, top-2 retained `96.73%` of passport
mass but only `49.71%` accuracy, while top-4 retained `99.70%` of mass but
still lost `3.74` accuracy points. Mass is therefore not a reliable proxy for
computational contribution.

With degree three, exact reachable node-transform rows sum to
`R(R+2)` per lane and router source rows sum to `R^2` before graph saturation.
That is `24/16` rows at four relays and `80/64` at eight. A learned top-5 cap
would reduce the eight-relay upper counts to `38/34`, but the current ablation
only zeros location mass; it does not yet provide a physically packed speedup.

A direct gathered-weight prototype was value/gradient equivalent but slower:
at batch 32 on one CPU thread, top-5 execution fell from `7.70` to `5.19`
steps/s at four relays and from `3.12` to `2.27` at eight. Materializing a
different node-weight block for every batch/token/lane selection costs more
than the small transforms it removes. The production sparse kernel must group
packets by selected node and reuse each node's weight, not gather replicated
weights per packet.

Relay-localized pruning shows why a fixed support is suboptimal. Top-2 applied
only at the final relay retained `90.62%` token accuracy, versus `49.71%` when
applied throughout. The near-lossless schedule was:

```text
soft reachable support:  [3, 5, 7, 9] -> 91.45% token / 58.20% exact
scheduled support:       [3, 5, 6, 4] -> 91.43% token / 57.42% exact
```

It reduces node-transform rows from 24 to 18 per lane while preserving token
accuracy within `0.02` points. Early branches are future computational options;
late branches can narrow near commitment. This is direct evidence for
broad-to-narrow relay schedules rather than one global top-k.

### 2026-07-19: Teacher-forced language control

The blank-parallel D400/N66/L10/K3/R2/C512/O128 run remained near raw PPL 58.
The exact same language architecture under causal teacher forcing reached
raw PPL `11.2238` and content-only PPL `22.7513` at step 30,000. All 128 response positions
are predicted in one training forward pass, but every position receives the
correct earlier response tokens through a causal graph mask. Novel inference
therefore remains autoregressive.

This is decisive diagnostic evidence that the D400 graph, language shell,
optimizer, and mixed corpus can learn language. It isolates blank simultaneous
formation as the dominant difficulty in the PPL-58 run. It is not evidence
that teacher-forced decoding itself performs simultaneous novel generation.

The frozen D400 configuration contains 14,360,642 parameters. The next capacity
control changes only payload width from D400 to D20. With slot-conditioned
terminal voting the D20 model contains exactly 130,402 parameters; the older
130,202 count refers to global terminal voting. Both controls use the
non-repeating 1:4:3:2 SmolTalk/LMSYS/Tulu3/Preferences stream, context 512,
128 response targets, batch 8, seed 0, LR 0.003, and the same 300k cosine
horizon.

The final teacher-forced metrics were loss `2.4180`, token accuracy `40.51%`,
top-5 accuracy `68.53%`, content token accuracy `22.88%`, content top-5
accuracy `58.02%`, and exact accuracy `0.15%`. Runtime was 2,496.31 seconds at
12.02 steps/s on an RTX PRO 6000 Blackwell Server Edition. The best checkpoint
was the final step, so the 30k window did not reveal a regression.

The terminal jury narrowed from `9.04` entropy-effective lanes and `19.9%`
top-lane mass at step 1,000 to `2.32` effective lanes and `78.8%` top-lane
mass at step 30,000. Reported distinct winning lanes reached one by step
7,000. This is meaningful specialization but does not authorize pruning:
prefix experiments already showed that low-mass lanes can carry essential
corrections.

Do not quote the concatenated `formed` line as a model generation. Each
position is an argmax conditioned on its own correct earlier target tokens;
the displayed string is not rolled out through the model and can be
internally inconsistent.

The next D20 run uses an accuracy-gated handoff. Below `60%` smoothed
teacher-forced validation content accuracy it remains exactly 100% causal
teacher forcing. Between `60%` and `90%`, the whole-batch teacher fraction
falls linearly; at `90%` it reaches zero. Complete batches are selected by
deterministic error diffusion rather than per-token target corruption or a
new stochastic router. Every log evaluates both modes. Blank-parallel
validation selects `best.pt`, causal validation selects `best_teacher.pt`,
and controller state is included in `latest.pt`.

Because D400 ended its 30k control at `22.88%` content accuracy, this schedule
would not have begun handoff in that window. That is the intended guardrail:
the model must first acquire substantially stronger causal language paths
before those paths are asked to operate without gold prefixes.

Full frozen configuration, archival requirements, result fields, and decision
rules are in `Papers/GATOR_TEACHER_FORCED_LANGUAGE_CONTROL.md`.

### 2026-07-19: TinyStories-first handoff corpus

The next D20 accuracy-gated run adds `roneneldan/TinyStories` as the first
tokenizer and training source. It receives 40% of the named stream. The
established `1:4:3:2` SmolTalk Everyday/LMSYS/Tulu3/SmolTalk Preferences
balance is preserved inside the remaining 60%, resolving to
`40%/6%/24%/18%/12%`.

At context 64, a story begins with a two-sentence input prefix and the following
sentence pair as its target. Each additional 64 context slots adds another
two-sentence prefix boundary from the same complete story. Response targets
grow by one sentence pair per 64 response slots. The parser reads the official
raw text using `<|endoftext|>` boundaries rather than the historically
problematic line-based interpretation.

TinyStories has its own restartable shuffle stream, validation reservation,
filter counters, file/directory fingerprint, and checkpointed mix settings.
The trainer prints normalized source shares before training so a corpus-order
or weighting mistake is visible before the first step.

### 2026-07-19: 10k compositional sequence-program benchmark

`gator_sequence_program_benchmark.py` replaces the single modular-prefix rule
with a fresh seven-symbol sequence plus a three-instruction program. The
instruction vocabulary contains no-op, reverse, left/right rotation, adjacent
pair swap, value complement, prefix XOR, and sort. Gator must execute the
instructions in order and form all seven final symbols simultaneously.

The input space contains \(8^7=2,097,152\) value sequences. Fully active
three-instruction programs draw from seven non-no-op instructions, giving 343
ordered programs and roughly 719 million value/program combinations. The
first run saw only 640,000 freshly generated training examples, and evaluation
used a disjoint deterministic generator.

The initial screen used the exact 10,010-parameter Convergent model:

```text
nodes=40 width=8 lanes=10 degree=3 relays=2
layout=small_world rewire=0.5
batch=128 seed=0 LR=0.008
curriculum: one operation through step 1k,
            two operations through step 2k,
            three operations thereafter
```

| Step | Trained depth | 1-op token | 2-op token | 3-op token | 3-op exact | 3-op loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1 | 56.9% | 29.4% | 19.6% | 0.0% | 2.8772 |
| 2,000 | 2 | 29.8% | 42.4% | 27.6% | 0.2% | 2.0816 |
| 3,000 | 3 | 20.9% | 28.4% | 37.1% | 4.0% | 1.5318 |
| 4,000 | 3 | 20.9% | 28.3% | 39.3% | 5.1% | 1.4853 |
| 5,000 | 3 | 20.0% | 27.4% | **40.7%** | **6.1%** | **1.4428** |

Valid-output chance is 12.5%. The model therefore learned a conditional
instruction interpreter and achieved exact fresh-program solves, not merely a
copy or marginal-frequency shortcut. At step 1k it also showed zero-shot lift
on untrained two- and three-instruction compositions.

The curriculum exposed a clear failure: moving exclusively to a deeper program
caused earlier primitive accuracy to regress. This run must remain the clean
no-replay baseline. The next A/B should replay shallower program depths while
training the current depth, and should report operation-restricted suites so
sorting cannot hide weaknesses in permutation and prefix-XOR execution.

The replay A/B then ran from scratch for the complete 50,000-step budget. It
used a 40% row-level replay fraction: at depth three, 60% of rows contained
three active instructions and the remaining 40% was divided uniformly between
one- and two-instruction programs. It stopped only for 99% held-out three-op
token accuracy or the 50k limit.

Replay prevented the catastrophic primitive collapse but did not produce
mastery:

```text
step 50,000
one-op token accuracy:   75.7%
two-op token accuracy:   49.1%
three-op token accuracy: 42.3%
three-op exact:           7.2%
best three-op exact:      8.5%
three-op loss:            1.4005
runtime:                  5,904.34 seconds
```

Three-op accuracy reached 40.3% at 30k and remained in the 40.4--42.4% range
thereafter. Replay improved retained primitives and modestly raised the
hardest-depth ceiling over the 5k no-replay screen, but additional updates at
the same R2/width/topology ratio produced a clear plateau. The next
intervention should change computation—relay depth, operation-conditioned
regional specialization, or program-token ingress—not merely extend this
optimizer trajectory.

#### Frozen-checkpoint composer and composition autopsy

The best replay checkpoint at step 44k was probed without changing its 10,010
core parameters. The full machine-readable result is
`experimental_checkpoints/sequence_program/composer_capacity_probe_full.json`.

The existing global terminal composer achieved 42.02% token accuracy across
all 343 ordered three-operation programs on fresh values. Although at least
one lane's top-1 choice was correct in 79.39% of output slots, a new
1,354-parameter MLP conditioned on the complete program and output address
raised frozen-core accuracy only to 42.76% and exact accuracy from roughly
8.5% to 9.0%. A per-program/per-slot lane lookup was worse at 26.59%.
Therefore the 42% plateau is not primarily a terminal-voter selection
failure. The lanes contain broad candidate coverage, but their correct
answers are not organized into a stable specialist mapping that a small
composer can recover.

An oracle-suffix intervention localized the larger failure:

| Frozen-core input | Remaining learned operations | Token accuracy | Exact |
|---|---:|---:|---:|
| Original values and full program | 3 | 42.17% | 8.94% |
| Operation 1 executed externally | 2 | 49.31% | 12.08% |
| Operations 1–2 executed externally | 1 | **77.08%** | **42.14%** |

The checkpoint reacts strongly to its instruction tokens: shuffling programs
changed 74.95% of predictions, preserved 42.93% accuracy on the newly implied
targets, and reduced accuracy against the old targets to 17.89%. It is not
merely ignoring the program.

Primitive quality is highly uneven. At depth one, complement, sort, left
rotation, and right rotation reached 98.47%, 95.67%, 95.42%, and 89.46%;
reverse and pair-swap reached only 62.59% and 62.20%; prefix XOR reached just
24.11%. Repeating operations exposes additional composition failure:
three repeated rotations fall to 23.96–26.81%, while three repeated sorts
remain at 98.33%.

Diagnosis: this is a core execution/credit-assignment ceiling for the current
R2 configuration, not evidence of a simple terminal composer ceiling. It is
not yet possible to distinguish parameter capacity from insufficient
recurrent compute, but extending the same optimizer trajectory is already
ruled out. The clean next A/B is the same approximately-10k budget with at
least three relays, followed by primitive-balanced sampling so prefix XOR
cannot remain the hidden floor.

### 2026-07-20: 130k TinyStories handoff terminal result

The D20/N66/L10/R2/C512/O128/B8 accuracy-gated language run completed one
million updates without crossing the 60% teacher-content gate. Teacher content
accuracy rose from the mid-20% range to approximately 30--31%, then failed to
break out. Consequently `teacher_fraction` remained 1.0 for the entire run:
no blank-parallel training batch was ever selected.

This was initially classified as a causal-capacity plateau for the
130,402-parameter configuration, not a negative result for
teacher-to-parallel transfer. The transfer mechanism was never exercised.
Blank-parallel letter spam is therefore expected and cannot measure whether a
successfully trained causal representation would survive handoff. The
checkpoint autopsy below supersedes the capacity interpretation: it found a
teacher-prefix shortcut that makes raw teacher accuracy an invalid capacity
gate. The run still rules out insufficient update count as the simple
explanation because it received roughly 39k fixed token-slots per parameter.

Preserve `best_teacher.pt` for the autoregressive and counterfactual probes.
Future handoff experiments should first repair the causal-support curriculum.
Do not retroactively lower the threshold and call the original run a completed
handoff.

### 2026-07-20: D20 checkpoint autopsy revises the diagnosis

The preserved best teacher checkpoint occurred at step 330,000, not at the
one-million-step endpoint:

```text
checkpoint: best_teacher.pt
SHA-256: 1973dc29b033a9f8a0860e6da57fc477826d273b7a11418402e9e0fd06cfce73
parameters: 130,402
teacher content accuracy: 31.535%
teacher content top-5: 65.267%
teacher content PPL: 16.494
final-step content accuracy: 31.052%
final-step content PPL: 29.481
```

Per-source probing of 64 held-out examples found 31.74%/16.80 PPL on
TinyStories, 26.70%/19.79 PPL on SmolTalk Everyday, and 31.80%/14.29 PPL on
LMSYS. Accuracy stayed broadly flat over the 128 response positions and did
not decline with context length. The terminal ensemble used about six
effective lanes and lost a target preferred by an individual lane only
3.6--6.5% of the time. Neither response-slot address decay nor terminal voting
explains the main ceiling.

Causal counterfactuals identified the shortcut. Shuffling the input prompt
while preserving the gold response prefix changed content accuracy from
30.25% to 27.63% on TinyStories, 25.36% to 25.46% on SmolTalk, and 31.47% to
31.47% on LMSYS. Shuffling the response prefix reduced those accuracies to
1.36%, 2.70%, and 2.02%; blanking it reduced them to 0.97%, 0.39%, and 0.73%.
True autoregressive generations collapsed into repeated high-frequency words.

Conclusion: the unreachable gate measured a gold-prefix continuation
shortcut, not a clean capacity threshold for prompt-conditioned language. The
next D20 run should use token/span-level prefix corruption, including
zero-prefix and short-prefix samples, and measure a correct-context versus
shuffled-context utility margin. Do not lower the whole-batch 60% threshold
and call that curriculum progress.

The run also failed to preserve its nominal `40:6:24:18:12` source mixture
because `streaming_exhaustion=drop` permanently removed sources. SmolTalk
exhausted near 20k, Preferences near 140k, Tulu near 320k, and the remaining
non-TinyStories stream later exhausted. Use restartable, epoch-safe streams
for fixed-mixture long runs. The step-330k best and worsening PPL thereafter
also make best-checkpoint rewind mandatory.

### 2026-07-20: prefix-denoising curriculum implemented

The next D20 run uses `decode_mode=prefix_denoise`; the historical handoff
implementation remains available unchanged. Prefix denoising preserves the
causal mask while replacing selected shifted response-prefix tokens with the
non-rendering `BLANK` token.

Default training policy:

```text
retention:              0.95 -> 0.50 over 100,000 completed steps
zero-prefix rows:       5%
short-prefix rows:      10%, retaining 1--8 tokens
ordinary corruption:   50% independent token dropout
                       50% one contiguous missing span
adaptive accuracy:     0.60 -> 0.90
adaptive context use:  0.02 -> 0.10 correct-minus-shuffled accuracy
feedback EMA:           0.50
stream exhaustion:     restart
```

The fixed retention ramp guarantees partial-support practice even if teacher
accuracy remains below 60%. Removal below 50% requires both accuracy and
semantic context utility; the weaker readiness signal wins. Regressing
accuracy or context utility restores the adaptive portion of support.

Each report now evaluates blank formation, full teacher forcing, and matched
partial-prefix correct/shuffled contexts. The paired context probe uses the
exact same deterministic missing-token mask on both sides. The compact
`prefix_denoise` line records requested/actual retention, corruption row
counts, all PPL/accuracy pairs, instantaneous and smoothed context utility,
and readiness.
`best_context.pt` maximizes positive context utility multiplied by correct
teacher accuracy. The Colab runner uses epoch-safe `restart`, preserving the
named `40:6:24:18:12` corpus shares instead of silently converging to
TinyStories-only training.

### 2026-07-20: D20 eight-token next-chunk control

The first graduated full-128 run showed that the supported lane learned
rapidly while unsupported formation remained dominated by whitespace. At
step 50,000, checkpoint inspection found 24.54% teacher content accuracy and
28.46 content PPL, but only 0.35% unsupported content accuracy, 18.87% top-5,
335.75 content PPL, and 2.87% formation coverage. All unsupported metrics were
improving, so full 128-slot formation is not rejected; it is likely
under-capacity at 130,402 parameters.

The next controlled experiment changes only the decoding horizon:

```text
architecture:          Convergent D20/N66/L10/K3/R2, slot voting
parameters:            130,402
reference window:      128 tokens
context window:        512 tokens
decode mode:           chunk_parallel
chunk width:           8 simultaneous tokens
training prefix:       prompt + complete prior gold chunks
current chunk support: none
formation loss:        slot-specific CE
corpus:                restartable 40:6:24:18:12 mixture
```

Training selects an aligned chunk uniformly per row. Validation rotates
deterministically through chunk positions and therefore measures
oracle-prefix next-chunk learning. It must not be reported as free-running
chat quality until predicted chunks are committed and rolled forward without
gold prefixes. Chunk widths 8, 16, 32, 64, and 128 are now the proposed
formation-horizon scaling sweep.

#### Indirect slot-voter result and direct-address intervention

The first chunk-eight run used the legacy 130,402-parameter indirect slot
voter. At step 50,000 it reported:

```text
loss/PPL:             3.9417 / 51.51
all-token top-1:      39.1%
all-token top-5:      50.4%
content top-5:        18.8%
terminal top-1 mass:  30.0%
effective voters:     7.58 / 10
adjacent slot delta:  0.0005
winning lanes:        1.00
formed sample:        whitespace only
speed:                76.50 steps/s
```

The 39.1% top-1 value was whitespace accuracy, not semantic mastery.
Terminal concentration increased from 15.5% at 5k to 30.0% at 50k, but all
eight positions retained nearly identical juries. Content top-5 fluctuated
near 19--21% without a sustained rise. This is sufficient to reject waiting
longer for the indirect voter to differentiate in this control.

The replacement adds a direct immutable-address term to each slot's lane
logits:

\[
v_{i,k} =
b_k
+ \langle h_{i,k},q_k\rangle/\sqrt D
+ \langle a_i,r_k\rangle/\sqrt 4.
\]

Here `r_k` is a private four-value key for lane `k`. The jury uses
`fixed_code(output_slots, 4)` rather than slicing the code normalized across
context plus output. In the failed C512/O8 run, adjacent output addresses were
only about `pi/519` apart, explaining why a nominally slot-aware judge
remained numerically global.

The D20/L10 model gains only 40 parameters, from 130,402 to **130,442**. The
keys initialize after all historical tensors so shared seeded initialization
remains exact. New slot juries differ immediately; both the mutable query and
immutable-address keys receive ordinary task gradients. Legacy slot
checkpoints remain loadable: a missing address key is inserted as zero,
preserving their saved forward behavior and their original relay addresses.

The compact step log now includes semantic top-1, formation coverage, and
whitespace omission so a modal-whitespace solution cannot masquerade as
token accuracy:

```text
content_acc=... content_top5=... coverage=... ws_omission=...
```

### 2026-07-20: 140M shared-routing language sanity configuration

The first normal-scale Gator control freezes:

```text
V32768 / E768 / D768
N154 / L10 / K3 / R2 / SW50
shared low-rank router rank 96
learned packet-conditioned ingress
top-32 active nodes per packet
direct-address slot voting
C512 / O128
parameters: 140,193,699
```

Its first teacher-forced batch failed before optimization in compiled sparse
node execution. This was not an OOM or learning failure. PyTorch's Blackwell
`torch._grouped_mm` path required BF16 operands but received FP32 activation
and node-weight matrices. The corrected kernel boundary casts only grouped
private-node GEMM operands to BF16, restores FP32 output immediately, and
keeps FP32 master parameters and optimizer state. The initial failed run is a
kernel-compatibility observation only and contains no capability result.

The corrected run used only approximately 11.2 GB at batch one. From step 100
to step 600, validation PPL moved `57.93 -> 36.43` with a low of `34.14`,
content top-1 moved `2.6% -> 4.5%`, formation coverage remained
`79.4%--88.1%`, and whitespace omission fell as low as `11.9%`. Terminal
juries were already distinct: adjacent slot delta grew `0.0038 -> 0.0079`
and six to nine lanes won some output position. These are early
teacher-forced acquisition signals, not free-generation results.

Throughput warmed from 2.62 to 5.76 steps/s, but Dynamo reported that the
changing Python routing-temperature float exhausted its eight-entry
recompile cache. The trainer now supplies that schedule as a scalar device
tensor so subsequent compiled runs can retain one graph.

The run used approximately 11.2 GB at batch one, leaving room for a batch
scaling probe. Remote-write behavior was also corrected before a long run:
full 140M optimizer checkpoints are staged locally rather than rewritten
through mounted Drive at each visible report. Drive receives batched log
appends every 20 reports or 15 minutes, one resumable `latest.pt` mirror every
15 minutes, and all locally preserved best checkpoints at normal completion.

### 2026-07-20: teacher-forced D20 and 140M checkpoints before routing optimization

The direct-address D20 control and the first 140M shared-routing control are
the rollback baselines for the packed-routing optimization pass. Both use
teacher forcing, so their concatenated `formed` strings are diagnostic
per-position mosaics rather than free-running responses.

The 130,442-parameter `D20/N66/L10/K3/R2` run reached its best saved
validation point at step 130,000:

```text
loss/PPL:             1.8734 / 6.5104
all-token accuracy:   57.93%
all-token top-5:      78.05%
content accuracy:     34.35%
content top-5:        63.64%
content PPL:          19.5753
formation coverage:   97.98%
whitespace omission:   2.02%
evaluation examples:  2
```

Its latest inspected checkpoint was step 145,000 and had regressed to PPL
7.1723 and 27.31% content accuracy. The two-example evaluation makes the
individual reports noisy; retain the step-130k best rather than treating the
step-145k decline as a settled curve.

The 140,193,699-parameter `V32768/E768/D768/N154/L10/K3/R2` control produced
its archived best at step 58,000:

```text
loss/PPL:             1.8368104869 / 6.2764873633
all-token accuracy:   60.6657%
all-token top-5:      77.9123%
content accuracy:     33.2454%
content top-5:        61.4776%
content PPL:          22.3858
formation coverage:   94.1953%
whitespace omission:   5.8047%
evaluation examples:  8
```

The checkpoint is archived at
`experimental_checkpoints/gator140m_teacherforced_shared_r96_seed0_step58000/best.pt`
with SHA-256
`5c5071a45a7437b64b4aa1247a0977444b5fad2b63f8479cfcfdb6012d4f69b2`.

Qualitatively, 140M learned stronger lexical fragments and stable narrative
scaffolding, but word ordering, substitutions, repetition, and sentence
assembly were not remotely proportional to its parameter increase. Equal raw
PPL is not an apples-to-apples comparison because the controls use vocabulary
sizes 2,048 and 32,768, batch sizes 32 and 6, evaluation sets of two and eight
examples, and radically different token exposure per parameter. The result is
nonetheless a clear formation-efficiency warning.

The geometry changed width far more than graph capacity: `D20 -> D768` is a
38.4x width increase, while `N66 -> N154` is only 2.33x; lanes, degree, and
relays remained fixed. Node density therefore fell from 3.30 to 0.20 nodes per
width unit, a 16.5x reduction. The working hypothesis for the next scaling
pass is that very wide nodes become generalists and make Gator more like a
dense recurrent network, while capable narrow nodes make local handoffs and
specialized traces consequential. This is a hypothesis, not yet a scaling
law. The next clean capacity A/B is `N154 -> N308` with width, lanes, degree,
relays, router rank, and active top-k held fixed after node/routing overhead is
reduced enough to make the run practical.

### 2026-07-20: packed-frontier and scalable-node implementation checkpoint

Commit `dbbd569` freezes the D20 and archived 140M teacher-forced routing
baseline immediately before this optimization pass. The archived 140M weights
and legacy defaults remain unchanged. Commit `57b6a81` implements the packed
routing/node prototype and its 453-test gate. Every new backend is opt-in:

| Lever | Compatibility default | New option | Status |
|---|---|---|---|
| Route state | `dense` | fixed-cap `packed` node IDs/mass/mask | implemented |
| Packed receiver memory | `matrix` oracle | direct `convolution`, raw `fft`, or `fft_opaque` | implemented |
| Packed reader epilogue | expanded Torch gather | fused Triton forward/backward used by the opaque receiver | native explicit bridge implemented; Blackwell rerun pending |
| Feature workspace | complete width | configurable feature chunks | implemented |
| Route width | fixed top-K | entropy/margin/hybrid Kmin..Kmax | implemented estimator A/B |
| Node compute | full-private `D x D` matrices | shared trunk + private low-rank adapters | implemented architecture A/B |
| Learned ingress | flat all-node score | balanced hierarchical group gate | implemented architecture A/B |
| Node-row optimizer | ordinary AdamW | experimental active-row AdamW | library probe only |

Packed transport collision-merges legal local destinations and carries the
frontier directly into the next relay. Dense passports are reconstructed only
for requested traces. Direct and FFT memory use the same learned
relative-position equation as the matrix oracle and gather only the memories
requested by packed readers. The FFT kernel spectrum is reused across the
denominator and all hidden-feature chunks; the denominator shares its feature
transform with the first hidden chunk. K-route writer scatter and reader
gather/reduction are vectorized rather than launched once per route.
Integrated frontiers also use a
trusted unique-route memory path rather than rebuilding a packet-local K-by-K
duplicate matrix.

Two target-shape materializations found during review were also removed. The
dynamic shared router now scores all `N * degree` source edges with one FP32
GEMM and gathers K logits, replacing a rank-expanded indexed key bank. At the
140M shape this reduces that intermediate from about 1.32 GiB to 67.7 MiB
(approximately 20x), at the cost of scoring N rather than K sources. The
shared-adapter CUDA path groups rows by node and applies private down/up
matrices without gathering `[packet,K,D,adapter_rank]` parameter banks. Its
Blackwell grouped-MM kernel remains a required on-device release test.

The correctness gate covers dense-versus-packed logits, route traces, and all
parameter gradients; matrix-versus-direct-versus-FFT memory values and
gradients; adaptive route mass conservation; shared-adapter packed/dense
oracles; hierarchical ingress reachability; and exact packet scatter. The full
local suite passed **453 tests** with one CUDA-only grouped-MM test skipped.
That skipped test remains part of the Blackwell release gate rather than being
inferred from module-only CPU smokes.

One serialized tiny CPU screen (`B1`, 12 total positions, `D8/N8/L2/R2/K2`)
measured approximately:

| Packed memory | Forward | Forward + backward |
|---|---:|---:|
| matrix | 28.3 ms | 213.6 ms |
| direct convolution | 16.2 ms | 69.5 ms |
| FFT | 250.3 ms | 347.9 ms |

This is deliberately recorded as non-target evidence. At tiny sequence length
FFT launch overhead dominates, and local CPU timings have high variance. They
do not establish the ordering on a Blackwell at 640 total positions.

The first target-GPU gate is the frozen
`V32768/E768/D768/N154/L10/K3/R2/Kmax32` control. Its pre-optimization language
training rate was **3.22 steps/s** at batch 6, context 512, and 128 response
slots on the RTX PRO 6000 Blackwell Server Edition. Run dense and packed
matrix in separate processes first, followed by direct convolution and FFT.
Only after an execution-only winner is established should Kmin, shared node
adapters, or hierarchical ingress change. The N308 capacity A/B remains after
that gate.

Important caveats are frozen with the code:

- Static compiled adaptive K changes valid route semantics but still executes
  a Kmax-shaped workspace; compute savings require eager compaction or future
  K-bucketed/fused kernels.
- Hierarchical ingress uses a hard top-group choice. Every node is structurally
  reachable, but training needs exploration/warmup and group-coverage logging
  before unused groups can be ruled out.
- Exact AdamW cannot skip inactive rows because momentum and weight decay move
  them. Lazy active-row updates change optimizer semantics and still allocate
  dense moments; real state savings require paged or fused state.
- Python sorting, collision merge, segmented dispatch, and scatter remain
  potential GPU launch bottlenecks. No Blackwell speedup is claimed yet.

### 2026-07-21: first 140M packed-memory Blackwell screen

The first full-shape packed run used the frozen 140,193,699-parameter control
at batch 6, context 512, 128 response slots, N154/L10/R2/K32, and compiled
execution. `packed + FFT + compile` reached model and corpus construction but
failed during first-forward TorchInductor code generation with
`KeyError: 'complex64'`; the Colab Inductor/Triton build cannot emit the FFT
path's complex signature. The trainer and 140M runner now reject that
combination before launch. Eager FFT remains valid.

The replacement compiled direct-convolution run reduced observed VRAM from
approximately **41 GB to 31.5 GB**, a **9.5 GB / 23%** reduction. It did not
reach its first 1,000-step report within approximately 11 minutes, versus
about 5.2 minutes per 1,000 steps at the historical 3.22 steps/s baseline.
Because the first report includes compilation time, this is not a precise
steady-state measurement, but it was already too slow for the intended long
run. Classify direct convolution as a meaningful memory win and a provisional
throughput loss at T640.

The next full run keeps packed routing and private-node dispatch but restores
the compiled matrix receiver. FFT compilation or a custom real-valued/fused
receiver remains the desired long-context path.

That packed-matrix follow-up OOMed at the same full B6/C512/O128 shape on the
97,887 MiB Blackwell. The matrix backend is an exact correctness oracle, but
it reconstructs node memory with shape `[B,T,N,D]` (about 1.8 GB for one FP32
tensor at B6/T640/N154/D768); retained forward/backward and relay
intermediates multiply that footprint. It is therefore not a viable full-size
packed training backend without recomputation or fusion. Revert production
training to the established dense-matrix execution while compiled FFT or a
custom fused receiver is developed.

### 2026-07-21: opaque FFT and fused Triton reader milestone

The optimized branch now separates the two mechanisms previously described
loosely as the “FFT/Triton kernel”:

1. `fft_opaque` is the exact existing FFT receiver behind a real-input,
   real-output `torch.library.custom_op`. A second opaque backward operator
   recomputes the receiver with `torch.func.vjp`. Complex cuFFT spectra no
   longer appear in the outer Inductor graph.
2. `packed_memory_reader_backend=triton` is a real-valued fused packed-reader
   epilogue. It combines denominator normalization, node gather, route-mass
   weighting, and K reduction. Its registered Triton backward atomically
   accumulates numerator, denominator, and route-mass gradients.

The standalone Triton path uses that registered backward. In the combined
`fft_opaque + triton` path, PyTorch 2.11 cannot traverse the generated Triton
autograd wrapper from the outer `torch.func.vjp`, so only the real forward is
Triton-fused for now; opaque VJP recomputation uses the mathematically
equivalent Torch reader. A manual VJP bridge is a separate performance
milestone.

These switches are orthogonal and parameter-free. The intended compiled A/B
is:

```text
--route-backend packed
--packed-memory-backend fft_opaque
--packed-memory-reader-backend triton
--packed-memory-feature-chunk 32
--compile
```

The portable Torch reader, raw eager FFT, direct convolution, and matrix path
remain explicit controls. Triton selection is deliberately strict: the first
implementation requires CUDA float32 plus `torch.library.triton_op` and raises
instead of silently falling back.

Local release evidence covers raw-versus-opaque FFT forward values and all
four differentiable input gradients, matrix/direct/FFT full-model
compatibility, custom-op schema/fake/autograd/AOT contracts, and the portable
reader's duplicate/collision gradients. CUDA-only tests for Triton forward,
backward, opcheck, and compiled opaque-FFT-plus-Triton training are present but
skip on the Windows CPU host. Therefore this milestone establishes the code
and numerical boundary, not a speed or memory win. The next required record is
a warmed RTX PRO 6000 benchmark against the historical 3.22 steps/s / roughly
41 GB control and the 31.5 GB direct-convolution screen.

The optimized trainer also carries the same PAD-safe token-handoff auxiliary
loss as the frozen fallback: masked forbidden PAD labels are removed with
`torch.where` before reduction, avoiding the original `0 * inf` step-one NaN.

The target-GPU proof is specified in
[`../BLACKWELL_FFT_TRITON_GATE.md`](../BLACKWELL_FFT_TRITON_GATE.md). Until its
correctness, continuation, and fresh-process performance stages pass on the
Blackwell, this entry remains an implementation milestone rather than a GPU
result. The later complete-gate record below supersedes this pending status.

The first Blackwell gate launch on July 21 reached and recorded the intended
RTX PRO 6000 environment, but its CUDA unit stage could not import the bundled
test modules because `tests/__init__.py` was absent. No kernel executed. The
bundle contract now includes that package marker; the attempt is recorded as a
harness failure rather than model or kernel evidence.

The second launch reached the CUDA kernels: 17/18 focused tests passed. The
remaining combined training test exposed a PyTorch 2.11 transform boundary,
not an FFT or Triton forward mismatch: `torch.func.vjp` could not traverse the
generated autograd function for the nested Triton reader. Opaque backward now
recomputes that reader with the equivalent Torch equation while public forward
remains fused Triton. This may increase backward memory, so no optimization
claim is made until the full Blackwell equivalence, memory, speed, and training
continuation matrix passes.

The third Blackwell launch cleared all 19 CUDA unit tests, all four quick
seed/causality matrices, and the production noncausal receiver. Its isolated
production-shape forward screen was promising: approximately 7.33 ms for the
compiled target versus 8.77 ms for eager FFT/Torch, with peak allocated memory
about 0.56 GB versus 8.82 GB. The current Torch-reader VJP erased that gain in
training: forward-plus-backward was about 41.34 ms versus 34.05 ms and peak
allocation was slightly higher.

The production-causal component diagnostic then ran three times at
`B1/T640/L10/K32/N154/D768`: seed 0 twice and seed 1 once. Raw Triton versus
raw Torch passed every run. Only `grad_mass` produced rare outliers (one to
three of 204,800), and similar outliers appeared in eager-autograd versus
functional-VJP and opaque-Torch versus separately recomputed-Torch controls.
The exact indices and magnitudes moved between identical seed-0 repetitions.
This supports nondeterministic FP32 CUDA scatter-add/FFT reduction
recomputation as the explanation rather than Triton reader math or causal
indexing.

The full gate at commit `048dda5` subsequently passed the focused CUDA unit
stage, every quick receiver stage, both production noncausal seeds, and
production causal seed 0. Production causal seed 1 stopped the gate on exactly
one of 204,800 `grad_mass` elements under the bulk `atol=1e-3,
rtol=1.5e-3` envelope. The tensor remained finite, relative L2 was
`6.48092555e-6` against a `1e-5` limit, and maximum absolute drift was
`0.0050125` against a `0.008` limit. No broader numerical failure was
observed before the fail-fast stop.

The resulting policy remains deliberately narrow. For production-shape
causal `grad_mass` only, the bulk elementwise envelope is telemetry and may
report at most four outliers, with an outlier fraction no greater than
`2e-5`; both caps must pass. Finiteness, relative L2 at most `1e-5`, and
maximum absolute error at most `8e-3` remain mandatory. All quick shapes,
noncausal runs, forward tensors, and other gradients retain their existing
tolerances and require zero outliers. This bounds the demonstrated sparse
CUDA recomputation noise instead of broadening `atol` again to follow a
moving near-zero mismatch.

The complete rerun at commit `c80d54b9875282d88fbee646759fca8f935907bc`
with bundle SHA-256
`d56c6169b45e605ec017a5bc65b822c73cf67356f6aeab161f2061f54144ae8d`
passed. It cleared the focused CUDA units; quick and production receiver
equivalence for seeds 0 and 1 in causal and noncausal modes; full-model
equivalence for seeds 0 and 1 in both modes; 100-step checkpoint/reload/resume
for both modes; and all performance stages.

The frozen compiled FP32 full-model training benchmark used
`B6/C512/O128/D768/N154/L10/K32/R2`:

| Backend | ms/step | steps/s | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| Dense matrix + Torch | 249.9264 | 4.001 | 29,489.26 MiB | 33,056 MiB |
| Packed opaque FFT + Torch | 910.4605 | 1.098 | 33,681.73 MiB | 37,954 MiB |
| Packed opaque FFT + Triton | 888.8866 | 1.125 | 33,681.85 MiB | 37,694 MiB |

Dense matrix Torch was 3.56x faster than packed Triton and used about 12.4%
less peak allocated memory relative to packed; expressed in the other
direction, packed Triton allocated 14.2% more than dense. Triton was only
2.46% faster than matched packed Torch, while their allocated memory was
effectively identical. This is a correctness and continuation release result,
not a blanket performance win.

The training result contrasts sharply with the isolated receiver-forward
screen, where packed Triton was about 1.20x faster and allocated about 15.9x
less memory than eager FFT/Torch. Backward currently recomputes the portable
Torch reader through the opaque VJP, erasing the forward benefit during full
training. The next optimization priority is a true fused/optimized backward.
Until then, the current packed forward is most promising for inference-only
deployment, where no backward recomputation is required.

### 2026-07-21: native chunk-local opaque FFT backward

The follow-up removes the whole-receiver `torch.func.vjp` replay. The opaque
backward now reconstructs the discrete active-node plan once, recomputes the
shared denominator once, and consumes hidden features in bounded chunks. For
each chunk it:

1. recomputes the packed writer state and FFT numerator;
2. calls the selected reader adjoint directly (`triton` on the target path,
   the explicit Torch oracle elsewhere);
3. applies the exact real FFT adjoint by embedding the cropped cotangent and
   multiplying by conjugated spectra;
4. immediately folds the writer cotangent into hidden state, route delta, and
   write-gate gradients; and
5. retains only denominator, route-mass, write, and kernel-spectrum sums.

After the feature loop, one denominator adjoint and one inverse transform
finish the shared kernel gradient. The relative-kernel derivative preserves
the existing detached maximum used for numerical stabilization. The mass
adjoint preserves both reader mass and writer mass paths, including invalid
routes, epsilon floors, duplicate collision merging, and the trusted
unique-route fast path. The outer denominator clamp is differentiated
separately from the reader's safety clamp.

No nested autograd graph, `torch.func.vjp`, full receiver replay, or
full-width `[B,T,L,K,D]` route tensor remains in this backward. The remaining
route-feature workspace is bounded by `feature_chunk_size`. This is the full
mathematical adjoint for the existing receiver equation; a later fused writer
adjoint could reduce launch and materialization cost without changing it.

Local CPU float64 tests compare all four gradients against eager FFT autograd
across causal/noncausal, compact/full active tables, duplicate/unique routes,
and single/multiple/tail chunks. A regression test makes both whole-receiver
recompute and `torch.func.vjp` fatal and verifies one explicit reader-adjoint
call per chunk. The all-invalid compact branch returns exact zero values and
gradients. CUDA tests require full receiver and gradient parity while proving
that the Triton reader backward is actually invoked. CUDA correctness,
continuation, and performance remain pending until the complete Blackwell
gate is rerun; the VJP-backed `c80d54b` measurements remain the frozen
comparison rather than being attributed to this candidate.

The first native-backward Blackwell gate used commit
`bb9cdd147dd6208ad702897ef6f5f0e0d9dd4394` and bundle SHA-256
`af520539f6b99ae44c5570eecd3bef983a465d61460a4bc3c50e27698e69b4e0`.
All 22 CUDA unit tests and all four quick seed/causality validators passed.
The production noncausal receiver also passed every output and gradient
comparison. At `B1/T640/L10/K32/N154/D768`, feature chunk 32:

| Receiver path | Forward | Forward + backward | Peak forward allocation | Peak F+B allocation |
|---|---:|---:|---:|---:|
| Eager FFT + Torch reader | 8.5663 ms | 33.6811 ms | 8.815 GB | 4.595 GB |
| Opaque FFT + Triton, native backward | 7.1495 ms | 31.8027 ms | 0.556 GB | 0.671 GB |

The native path was 1.198x faster in forward and 1.059x faster for
forward-plus-backward. Peak allocated memory was 15.86x lower in forward and
6.85x lower for forward-plus-backward; measured allocation delta was 9.41x
lower for forward-plus-backward. Relative to the prior VJP-backed packed
receiver, native backward improved receiver forward-plus-backward throughput
by about 1.34x and cut peak allocation by about 7.07x. This recovers a real
production-shape receiver crossover, but it is not yet a full-model or causal
training result.

The fail-fast gate stopped at production causal seed 0. `grad_mass` had zero
elementwise outliers under the declared `1e-3/1.5e-3` envelope, remained
finite, and stayed below the `0.008` maximum-absolute guard at `0.0075035`.
Its relative L2 was `1.5727e-5`, above the frozen `1e-5` ceiling. Other
outputs and gradients passed elementwise checks, although several causal
relative-L2 values also moved into the `1e-5` range. No tolerance was widened.

Review found no causal-mask or adjoint-equation error. One avoidable numerical
ordering mismatch did remain: forward fuses writer mass and the first state
chunk into one FFT batch, while native backward recomputed and differentiated
them as separate FFT batches. CUDA may select different cuFFT batching plans,
and early causal denominators amplify their rounding difference. The follow-up
now mirrors forward exactly by combining those features in both recomputation
and adjoint. Besides improving numerical parity, this removes one FFT and one
adjoint FFT per receiver call. A new Blackwell component diagnostic and gate
rerun are required before changing the claim boundary.

The complete native-backward rerun used commit
`56c4a55c80700723ab5cee584efba2bae8010055` and bundle SHA-256
`fca32603d4a6fa7a93be961012bec81e75f88e06219e542522afeae4c40e596b`.
It passed all 22 focused CUDA tests, quick and production receiver validation
for seeds 0 and 1 in causal and noncausal modes, full-model equivalence in all
four seed/mode combinations, 100-step training and exact checkpoint
continuation in both modes, and all three full-model benchmarks. The
fused-first correction cleared the original production-causal failure under
the unchanged acceptance policy.

| 140M compiled FP32 training backend | ms/step | steps/s | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| Dense matrix + Torch | 249.8348 | 4.003 | 29,489.26 MiB | 33,056 MiB |
| Packed opaque FFT + Torch | 835.8425 | 1.196 | 19,194.63 MiB | 25,138 MiB |
| Packed opaque FFT + Triton | 685.9594 | 1.458 | 19,192.60 MiB | 25,028 MiB |

Triton improved packed full-model throughput by 21.9% over the Torch reader,
showing that the reader kernel matters. It did not change whole-model memory
because the FFT/writer and remaining model state dominate that total. Packed
Triton used 34.9% less peak allocated memory than dense matrix Torch, but
dense remained 2.746x faster. This establishes a deployable correctness and
memory milestone, not a training-speed crossover. Future performance work
should target FFT/writer fusion, launch count, and the remaining graph work;
reader-only optimization cannot close the measured gap by itself.

### 2026-07-21: frozen 100M node-heavy language control

The next language comparison targets the widely used approximately 100M model
class while returning to the proven dense-matrix backend. “Moderate width” is
interpreted as tensor-core-aligned D256 rather than literal D25; D25 would
require roughly ninety thousand private nodes and make ingress impractical.

| Component | Configuration / parameters |
|---|---:|
| Vocabulary | V32768, tied E256: 8,421,376 |
| Lane computation | L10, D256: 2,662,450 |
| Private node computation | N1337: 88,311,524 |
| Shared routing and addresses | rank 96, degree 3: 584,057 |
| **Total** | **99,979,407** |

The frozen graph is `D256/N1337/L10/K3/R2`, learned ingress, active top-32,
small-world rewire 0.5, slot voting, C512/O128, and the same mixed
TinyStories/SmolTalk/LMSYS/Tulu/Preferences teacher-forced corpus as the 140M
control. Batch 4 is the initial Blackwell setting because `[B,T,N,D]` receiver
memory is about twice the 140M control's node-memory volume at B4 despite the
smaller width. The dedicated runner is `run_gator_100m_nodes_colab.py`.

### 2026-07-23: plateau/regression root-cause study (CPU, small-scale)

A dedicated investigation into why free-generation quality repeatedly plateaus
and regresses during training, prompted by the observation that the original
unmoderated Gator (single jury, monolithic optimizer, shared nodes) produces
qualitatively better free generation than the multi-jury/staged-optimizer/
isolated-slot combination despite worse logged PPL. Full write-up, exact
configs, and every result table are in
[`GATOR_PLATEAU_INSTABILITY_STUDY.md`](GATOR_PLATEAU_INSTABILITY_STUDY.md);
raw logs are preserved at
`experimental_checkpoints/plateau_instability_study_20260723/`.

Headline finding: the plateau is a universal, architecture-independent
calibration-drift instability under constant-LR training on a small dataset —
present even in the Arrow architecture, which has no shared-node concept at
all. Multi-jury terminal voting, per-slot node isolation, and the staged/
node-priority optimizer are independent, non-additive severity amplifiers on
top of that same instability, not separate causes; stacked together at real
scale they turned a gradual drift into a runaway explosion (content PPL
85 → 39,879 by step 33,000) versus the plain architecture's much gentler
climb (97 → 592 by step 36,500) over a comparable horizon. Isolation's harm
was confirmed not to be a node-starvation artifact via a capacity-matched
control. Investigation continues in WSL/Colab per
[`HANDOFF_PLATEAU_INVESTIGATION_WSL_COLAB.md`](../HANDOFF_PLATEAU_INVESTIGATION_WSL_COLAB.md);
the highest-priority untested variable is a learning-rate decay schedule, since
every run in this study and its predecessors used constant LR.
