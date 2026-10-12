# AdamW Weight-Geometry Trace for GTR-L

Status: original plateau-only trace (2026-08-05) now followed up with a
real breakout/plateau/destabilization trace (2026-08-09) — see "Follow-up:
geometry through an actual breakout" below.  
Date: 2026-08-05, follow-up 2026-08-09  
Purpose: characterize the parameter-space motion produced by AdamW so Procoim can approximate its useful geometry without storing full Adam moments.

## Executive result

AdamW did not leave chance during this local run, so this is not a successful-learning benchmark. It nevertheless produced a clear and consistent geometric result:

- AdamW's net update over each 500-step interval was strongly low-rank.
- The orientation of that low-rank update rotated substantially between intervals.
- One fixed Procoim K=8 search basis captured almost none of the AdamW movement.

Across all 20 measured intervals:

| Measurement | Mean | Median | Range |
|---|---:|---:|---:|
| Cosine with previous interval update | 0.2413 | 0.2001 | 0.0109–0.4582 |
| Energy captured by fixed group-shared K=8 basis | 0.3862% | 0.3478% | 0.0671–1.0457% |
| Energy captured by fixed per-tensor K=8 basis | 3.6952% | 3.6242% | 2.1719–6.0039% |
| Energy captured by best rank-4 matrix approximation | 91.5951% | 92.6003% | 80.0261–96.4934% |
| Energy captured by best rank-8 matrix approximation | 98.5099% | 99.0212% | 94.0365–99.5748% |

This supports a rotating-factorized Procoim search: use low-rank parameter directions, but periodically regenerate their orientation rather than accumulating coefficients forever in one fixed random subspace.

## Run configuration

The run used the small local GTR-L prefix model:

| Setting | Value |
|---|---|
| Parameters | 46,701 |
| Output slots | 4 |
| Streams / nodes / width | 10 / 40 / 8 |
| Relays / degree | 4 / 6 |
| Batch / evaluation batch | 128 / 512 |
| Optimizer | AdamW |
| Learning rate | 0.003, constant |
| Weight decay | 0.0001 |
| Gradient clipping | 1.0 |
| Routing | Hard |
| Scan normalization | Disabled |
| Temperature | Fixed at 1.0 |
| Prefix loss vocabulary | Full |
| Seed / evaluation seed | 0 / 12345 |
| Device / precision | CPU FP32 |
| Compile | Disabled |
| Trace interval | 500 optimizer steps |

Hard routing was active throughout. `node_entropy=0.000` confirms discrete node selection; the separately logged `route_entropy` describes aggregate route utilization and is not evidence of soft routing.

## Learning and throughput

The run was manually stopped at step 10,000 rather than spending roughly another 12.6 laptop-hours reaching 100,000.

- Chance token accuracy for the 12-token task is 8.333%.
- Steps 500–10,000 ranged from 7.5% to 9.3% accuracy, averaging 8.42%.
- Final accuracy was 8.6%; exact formation accuracy remained 0.
- Final loss was 2.4847, effectively equal to `ln(12) = 2.4849`.
- Mean logged throughput was 1.69 steps/s; steady-state throughput reached about 1.98 steps/s.
- No gradient clipping occurred.

The network changed internally despite flat task performance:

- route entropy fell from 1.636 at step 500 to 1.108 at step 10,000;
- bus-gate activation rose from 0.057 to 0.467;
- logged gradient norm fell from 0.3075 to 0.0373.

Thus AdamW was reorganizing routing and communication, but this local run ended before any known delayed accuracy breakout.

## Interval geometry

`fixed_shared` is projection into the original group-shared random K=8 Procoim basis. `fixed_tensor` allows a separate fixed K=8 span per tensor. `rank4` and `rank8` are optimal SVD energy captures for matrix-shaped parameter updates.

| End step | Update norm | Previous cosine | Fixed shared | Fixed tensor | Rank 4 | Rank 8 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 7.9316 | — | 0.3789% | 4.1800% | 84.9780% | 96.1989% |
| 1,000 | 2.4901 | 0.2679 | 0.2533% | 3.8498% | 91.8072% | 99.2068% |
| 1,500 | 1.7097 | 0.4300 | 0.5466% | 4.7277% | 89.9352% | 98.9116% |
| 2,000 | 2.9984 | 0.2001 | 0.2521% | 2.1719% | 96.2435% | 99.5748% |
| 2,500 | 2.3179 | 0.3253 | 0.2472% | 2.5864% | 95.6420% | 99.5576% |
| 3,000 | 2.0743 | 0.3713 | 0.4927% | 3.8871% | 92.3432% | 99.1898% |
| 3,500 | 12.3893 | 0.1065 | 0.3089% | 3.7014% | 85.8253% | 97.8487% |
| 4,000 | 1.7676 | 0.0109 | 0.4960% | 3.4162% | 92.7332% | 98.9224% |
| 4,500 | 1.6469 | 0.1420 | 0.5213% | 3.8069% | 94.6241% | 99.1087% |
| 5,000 | 1.3827 | 0.3862 | 0.7485% | 5.1322% | 92.7623% | 98.2363% |
| 5,500 | 1.2225 | 0.4208 | 1.0457% | 6.0039% | 96.4934% | 99.4010% |
| 6,000 | 1.7856 | 0.4582 | 0.4989% | 3.4016% | 95.5448% | 99.0180% |
| 6,500 | 6.2247 | 0.1852 | 0.1057% | 2.6986% | 87.8133% | 96.9276% |
| 7,000 | 15.9182 | 0.1481 | 0.0671% | 2.7995% | 80.0261% | 94.0365% |
| 7,500 | 3.2410 | 0.0780 | 0.4282% | 3.5469% | 91.2246% | 98.9342% |
| 8,000 | 3.2650 | 0.1500 | 0.3167% | 3.0631% | 93.4024% | 99.0481% |
| 8,500 | 2.4641 | 0.3404 | 0.5086% | 4.9061% | 92.4674% | 99.0244% |
| 9,000 | 3.4626 | 0.2953 | 0.3145% | 4.5926% | 94.0940% | 99.2233% |
| 9,500 | 7.0594 | 0.1542 | 0.0896% | 2.8505% | 94.1569% | 99.3017% |
| 10,000 | 8.9086 | 0.1142 | 0.1042% | 2.5821% | 89.7847% | 98.5273% |

Large movement bursts at steps 3,500, 6,500–7,000, and 9,500–10,000 did not produce an accuracy breakout. They did, however, retain the same rotating low-rank structure.

## Interpretation for Procoim

The result rules out a fixed low-dimensional random search space as a good AdamW analogue. The old fixed K=8 basis could trap nearly all accepted Procoim displacement inside a tiny subspace, while AdamW repeatedly moved into new orientations.

The useful approximation is:

1. factorized/low-rank directions for broad matrix coverage;
2. multiple inexpensive proposals per bank;
3. short-lived coefficient memory inside the current basis;
4. periodic basis rotation;
5. coefficient-moment reset when the basis rotates.

The current implementation follows this result with rotating factorized directions, K=32 by default, rank-1 factors, and basis rotation every eight group-local candidate banks. It retains a fixed-Rademacher mode for controlled A/B tests.

The missing evidence is geometry during productive learning. A future Colab run should use the same trace format on a known-good configuration through its actual breakout, then compare plateau intervals against post-breakout intervals. The local 10,000-step trace establishes the search-space shape but cannot establish which movements caused learning.

## Follow-up: geometry through an actual breakout (2026-08-09)

This closes the gap flagged above ("the missing evidence is geometry
during productive learning... cannot establish which movements caused
learning"). `gtrl_len7_adamw_trace_seed0_gatebias3_noreadoutbias_routerocc`
(length=7, 48,125 params, `--update-gate-bias 3.0 --no-readout-bias
--router-occupancy --graph-layout ring`, `--learning-rate 3e-3`) produced
a real trace through two successful position-recruitment events (pos1 at
step ~7500-9000, pos2 at step ~20000-24500), a long stalled plateau, and
a full destabilization-and-recovery event (all positions crashed to
chance at step 44,000, recovered by step 69,000 back to the same
2-position configuration). See `Runbooks/GTRL_ADAMW_BREAKOUT_LOG.md` for
the accuracy trajectory this geometry corresponds to.

Same methodology as above (global update-vector cosine with the previous
500-step interval; weighted-average rank-4/rank-8 SVD energy capture
across matrix-shaped parameters), computed via
`Runbooks/colab_cells/_analyze_trace_geometry.py`:

| Window | Steps | `prev_cos` range | rank4 range | rank8 range |
|---|---|---|---|---|
| pre-breakout | 500-2000 | 0.03-0.08 | 88-97% | 99.9-100% |
| pos1 breakout | 7000-9000 | 0.01-**0.16** | 95-98% | 99.8-100% |
| pos2 recruitment | 20000-24500 | -0.01-**0.21** | 93-96% | 99.8-99.9% |
| stalled plateau (early) | 30000-36000 | -0.02-0.05 | 96-99% | 99.8-100% |
| destabilization (crash) | 42500-44500 | -0.03-0.19 | 94-96% | 99.8-99.9% |
| post-destab dip | 48000-49500 | 0.00-0.13 | 94-95% | 99.8-99.9% |
| stalled plateau (late) | 55000-63000 | -0.02-0.01 | 94-97% | 99.8-100% |

**Low-rank structure is confirmed even more strongly than the original
run, and holds identically across every regime** — rank-4/rank-8 capture
percentages are statistically indistinguishable whether the model is
learning, stalled, or actively crashing. Low-rank-ness is a constant
structural property of AdamW's update on this architecture, not a
learning-specific signature.

**Rotation is the differentiating signal, and it's sharper than the
original finding.** Median `prev_cos` across nearly this entire trace
sits near 0 (occasionally slightly negative) — well below the original
run's own 0.20 median. Update direction is essentially uncorrelated
between consecutive 500-step intervals almost everywhere, *including*
during the destabilization crash itself (cosines -0.03 to 0.19, no more
directed than the quiet plateau). But **the two highest cosine values in
the entire 127-snapshot dataset — 0.146 at step 9000 and 0.206 at step
24500 — land exactly at the tail end of the pos1 and pos2 recruitment
windows**, the moment each position locks in. This did not happen by
coincidence at both independent recruitment events.

**Revised interpretation for Procoim:** the rotating-factorized design
(low-rank directions, periodic basis rotation) already matches the
dominant regime correctly — most of the time, AdamW's own movement has no
persistent direction to chase, so a search that also keeps rotating is
the right default. What this follow-up adds: **the rare, brief windows
of directional consistency are not noise to rotate through, they're the
signal that a position is actively consolidating.** A mechanism that
detects rising `prev_cos` (or an analogous forward-only proxy — e.g.
sustained multi-step agreement between accepted candidate directions)
and temporarily *suppresses* basis rotation during that window, instead
of rotating on the usual schedule, would target exactly the moments this
trace shows matter — not yet implemented in the current
`--procoim-directional-basis-rotation-*` machinery, which rotates on a
fixed forwards/banks schedule regardless of whether consolidation is
underway.

## Follow-up: GTR-ARW's genuine convergence trace (2026-08-10)

Both prior sections' geometry came from GTR-L, which has never reached
mastery on this task family this session (best result: ~6% exact-match at
length=4, vs. GTR-ARW's 74.7-98.8% exact-match at the harder length=7).
User's question motivating this section: "We haven't found a cheaper GTR
yet that can reproduce what GTR-ARW does. Don't we have a local ARW weight
trace to look at?" -- yes: `arw_10k_trace_seed0` (10,000-param GTR-ARW
cell-ring, `gator_cells_vectorized.py`, AdamW, 100 snapshots every 200
steps from step 200-20,000, full breakout-to-convergence per
`Runbooks/GTR_ARW_MASTERY_TRACE_LOG.md`: 95.6% token / 74.7% exact-match
final). This is the first time this geometry methodology has been run on
a trace from a model that actually works, rather than a stalled or
never-escaped-chance one. Analysis script:
`Runbooks/colab_cells/_analyze_arw_trace_geometry.py`.

**Low-rank structure holds here too, even more tightly than GTR-L's own
traces** -- `rank8` capture is 99.96-100% at literally every measured
point across all six windows (chance through final destabilization),
`rank4` is 94-99.6%. Confirms low-rank-ness is a property of AdamW's
update on this task family generally, not something specific to GTR-L's
architecture or its stalled dynamics.

**The rotation signature is the opposite of what GTR-L's stalled trace
showed, and it directly complicates the consolidation-gate premise from
the section above.** Windowed to the documented accuracy curve:

| Window | Steps | acc range | `prev_cos` range |
|---|---|---|---|
| pre_breakout | 800-1800 | ~6-9% (chance) | 0.34-0.71 (declining) |
| breakout_jump | 2000-3000 | 13%->43% | 0.28-0.41 |
| rapid_climb | 3200-8000 | 43%->89% | 0.17-0.47 (noisy) |
| mid90s_plateau | 8600-13800 | 91%->98% | 0.11-0.43 (declining) |
| late_convergence | 15000-19000 | 96-99% | 0.02-0.19 (near zero) |
| final_destabilization | 19200-20000 | final grad_norm spike 2.7->35.5 | **-0.03 to -0.15 (negative)** |

Update-direction consistency **decays monotonically** from moderately
directed early (0.6-0.7, before the model has learned anything) down
through near-zero and finally negative right at the documented late
`grad_norm` spike -- the textbook signature of settling into a narrow
minimum, where consecutive steps increasingly overshoot and correct
rather than keep pushing one direction. There is no spike-during-success
moment to find here, in contrast to GTR-L's stalled trace (where the two
highest cosines in the whole dataset landed exactly at the tail of both
successful position-recruitment events against an otherwise near-zero
baseline).

One real update-magnitude event stands out instead: `update_norm` jumps
roughly 4x right at the breakout (1.4 pre-breakout -> 14-16 at steps
2000-2400), then decays and plateaus around 3-4.5 for the entire
remainder of the run, never returning near zero -- consistent with
`clip_rate` staying pinned near 1.0 through almost the whole climb per
the mastery-trace log, so gradient clipping keeps imposing a floor on
step size even as the model settles.

**Honest interpretation:** the "rare aligned window marks consolidation"
finding that motivated Procoim's consolidation-cosine gate does not
generalize to the one architecture in this project that actually
converges -- it was most likely a signature specific to GTR-L's partial,
stuck dynamics (a model recruiting individual positions one at a time
while stalled overall), not a universal principle about what "learning
happening" looks like in AdamW's update geometry. A real converging run
instead shows steady decay toward chance-level (then negative)
directional consistency as the loss surface narrows near the optimum --
the opposite shape. This doesn't invalidate the consolidation gate's
mechanism (freezing a rotating basis during a detected spike is still a
coherent idea on its own terms), but the evidence that motivated it
should now be read as "true of GTR-L's stall dynamics specifically," not
"true of successful convergence in general."

**Caveat on the comparison above (user, 2026-08-10): "The architectures
differ though. They're more like in the same family than the same
architecture."** Correct, and worth stating explicitly rather than
implied: GTR-ARW's flat 10-cell vectorized ring (`gator_cells_vectorized.py`)
and GTR-L's multi-stream/routing/mailbox linear-scaling design
(`gator_linear/core.py`) are structurally different enough that the
rotation-signature contrast above should be read as a family-level
observation ("what does AdamW's geometry look like near a real
convergence, on this general task family"), not an architecture-controlled
one. The two were never meant to isolate the same variable the way the
in-project A/Bs (rotation cadence, LR sweeps) do.

### Follow-up: how weights organize and move information (2026-08-10)

User's question: "Do the traces give us a way to see how the weights
organize and move information?" The trace payload saves every parameter
by its actual name (not just anonymous tensors), and
`gator_cells_vectorized.py`'s names are structurally meaningful --
`router.weight`, `address_to_edges/readout/state`, `left_projection`/
`right_projection`/`coupling_norm` (the literal cross-cell wiring),
`neighbor_gates`/`self_gates`/`output_votes`, `node_transforms`,
`token_embedding`. Grouping the 209 traced parameters into four
interpretable roles and tracking each role's share of total update
energy across the same six windows
(`Runbooks/colab_cells/_analyze_arw_trace_roles.py`):

| Window | inter_cell_coupling | routing_addressing | node_computation | embedding |
|---|---:|---:|---:|---:|
| pre_breakout (chance) | 1.2% | 46.5% | 13.5% | 38.8% |
| breakout_jump (13%->43%) | 1.5% | 44.9% | 27.0% | 26.6% |
| rapid_climb (43%->89%) | 0.9% | 46.8% | 36.7% | 15.5% |
| mid90s_plateau (91%->98%) | 0.5% | 55.9% | 31.4% | 12.3% |
| late_convergence (96-99%) | 0.8% | 44.9% | 41.0% | 13.3% |
| final_destabilization | 0.7% | 49.6% | 36.6% | 13.1% |

**Cross-cell coupling (`left_projection`/`right_projection`/
`coupling_norm`/`neighbor_gates`/`self_gates`/`output_votes` -- the
literal mechanism that moves information between the 10 cells in the
ring) never exceeds 1.5% of total update energy, at any point from
chance through full convergence.** This role has only ~200 of the
model's parameters (two shared 8x8 projections plus per-cell scalar
gates), and per-parameter (`delta_rms`, not shown in the table) it moves
at a comparable rate to every other role -- so this is not "barely
learned," it's that GTR-ARW allocates almost no parameter budget to
inter-cell communication in the first place. A real structural contrast
with GTR-L, which has dedicated routing/stream/mailbox subsystems that
are large components in their own right, not a couple of small shared
matrices bolted onto a ring of otherwise-independent cells.

**A real temporal shift in where update energy concentrates:**
`embedding`'s share falls from 38.8% (chance) to 12-15% (plateau onward);
`node_computation`'s share rises from 13.5% (chance) to 36-41% (late
convergence). Loosely: relatively more of what moves early is about how
tokens get represented in the first place; relatively more of what moves
late is the actual per-node computation applied to that representation.
`routing_addressing` (the role with the most parameters, spread across
all 10 cells' `router`/`address_to_*` matrices) stays the dominant share
throughout (45-56%) with no strong trend -- consistently active at every
stage rather than settling early or late.

**Follow-up: per-cell breakdown of node_computation (2026-08-10).** User's
observation, connecting this finding to an architectural decision: "We've
tried making the cells fatter to get around the scaling issue but the
results were mixed. But the nodes live inside each cell and the trace
analysis you did shows the energy moving into the nodes not the cells
towards mastery." Correct, and breaking `node_computation`'s energy down
by individual cell index (`cells.N.node_transforms.*`/`cells.N.norm.*`)
sharpens it further -- the shift isn't just role-level, it's about *how
many cells participate at all*:

| Window | Distribution across the 10 cells |
|---|---|
| pre_breakout (chance) | cells 3+4 alone hold 93% (40.6% + 52.3%); the other 8 are nearly frozen |
| mid90s_plateau | spreads across nearly all 10 -- highest: cell 5 (17.4%), cell 9 (16.6%), cell 2 (15.8%); lowest: cell 8 (2.7%) |
| late_convergence | same spread, same leaders: cell 9 (18.7%), cell 2 (17.1%), cell 5 (12.9%); cell 8 still lowest (2.2%) |

At chance, the optimizer is effectively only touching two of the ten
cells' internal computation. Reaching mastery requires that activity
spread to engage almost the whole ring -- and once it does, which cells
lead/lag stabilizes (cells 2/5/9 consistently most active, cell 8
consistently least) from the plateau through full convergence, rather
than continuing to shift. This directly supports widening node capacity
(`d_model`, see `--d-model` added to `gator_cells_vectorized.py`)
over adding cells: the mechanism needs most of the *existing* cells doing
real work, not more cells that would plausibly start out frozen the same
way most of these ten did early on. It also gives a concrete, non-obvious
reason "fatter cells" got mixed results previously (per the user, exact
mechanism of that earlier attempt not re-derived here) -- if whatever
that change did wasn't specifically about growing what each *node* can
represent, it wouldn't target the thing that this trace shows moves
during the transition from chance to mastery.

## Preserved artifacts

The stopped run remains under:

```text
C:\tmp\gtrl_adamw_boundary_slots4_seed0_100k
```

Contents:

- `train.stdout.log` — complete console metrics through step 10,000;
- `train.stderr.log` — warnings only;
- `checkpoint\gator_linear_prefix_latest.pt` — step-10,000 checkpoint;
- `trace\trace_step_*.pt` — 21 snapshots: step 0 and every 500 steps through step 10,000.

