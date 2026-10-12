# Cell-growth ablation: growing an Arrow Gator to 64 cells vs. starting there

2026-07-27. First test of a standing research question: can Arrow reach a
target cell count by growing from a small, densely-trained "nursery"
instead of instantiating and densely training the full target cell count
from step one? The motivating cost concern isn't just training speed - it's
whether growth lets us avoid ever running a dense (all-cells-active) pass
at the full target size, which is where the memory/speed blowup documented
elsewhere in this session happens (see `Runbooks/ARROW_RUNBOOK.md`: cell
count is the dominant activation-memory driver, since every activation
tensor is shaped `[batch, cells, tokens, ...]`).

## Mechanism

`gator_cell_growth.py` + `gator_language_train.py --grow-from-checkpoint`:

- Clone every cell in a trained checkpoint `N` times (`--grow-clones-per-cell`).
- Scatter the clones around the grown ring so no two siblings land within
  `--neighbors` ring-distance of each other. Arrow's cell-to-cell topology
  is a pure function of ring index (`gator/models/stewardship.py`'s
  `ring_handoff_destinations` maps cell `i` to `i ± 1, i ± 2, ...` mod cell
  count) - there's no arbitrary adjacency list to edit, so "scattering
  descendants to different neighborhoods" reduces to choosing which ring
  index each clone occupies. Interleaving by clone-generation index
  (`0a 1a 2a 3a 0b 1b 2b 3b 0c 1c 2c 3c` for a 4-cell nursery grown 3x)
  guarantees every ring-adjacent pair is from a different parent, and each
  parent's own clones sit a full nursery-width apart - far beyond any
  realistic `neighbors` value.
- Clones after the first get a small Gaussian perturbation (fraction of the
  parent parameter's own std).
- Non-cell-indexed parameters (token/output projections, the shared
  `handoff_weight`, `lm_bias`) are copied unchanged.
- Starts a fresh run at step 0 with a fresh optimizer - this is a
  from-a-warm-start initialization, not a resume.

## Setup

- Data: `smoltalk_everyday.jsonl` (real, 2260 conversations).
- `--decode-mode serial` throughout, deliberately, to avoid confounding
  with the separate (and still only partially understood) parallel-decode
  instability documented in `Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md`.
- Toy scale: vocab=4000, d_model=16, embedding_dim=8, context=128,
  response=32, batch=4, `cell_transport_mode=broadcast_mix`,
  `node_topology=hybrid_hemispheric`, `formation_objective=slot_ce`,
  constant LR 3e-4. Absolute numbers (memory in MB, not GB) reflect this
  toy scale - the interesting result is the *scaling pattern*, not the
  absolute figures.
- Growth schedule: nursery (4 cells, 5,000 steps) -> grow x4 -> stage 2
  (16 cells, 10,000 steps) -> grow x4 -> stage 3 (64 cells, 15,000 steps).
  30,000 steps total, matching the naive baseline's step budget exactly.
- Naive baseline: 64 cells, random init, 30,000 steps, otherwise identical
  config.
- Both were then extended by another 15,000 steps (naive via
  `--resume-checkpoint`, growth via a normal resume of the already-grown
  64-cell stage-3 checkpoint) to see whether a "behind" model catches up
  given more steps once fully grown - though as it turned out, growth
  wasn't behind at all (see below).

## Results at the matched 30,000-step point

| Metric | Growth (4->16->64) | Naive (64 from scratch) |
|---|---|---|
| loss | 3.8919 | 4.1107 |
| **ppl** | **49.00** | **60.99** |
| acc / exact | 0.312 | 0.250 |
| content_acc | 0.091 | 0.000 |
| content_ppl | 187.89 | 265.85 |
| content_top5 | 0.273 | 0.091 |
| ws_omission | 0.182 | 0.273 |
| coverage | 0.818 | 0.727 |

Growth wins on every metric, not just the headline PPL. Naive's
`content_acc=0.000` means it got zero real content tokens right at this
point; growth's small but nonzero 0.091 and its 3x-better `content_top5`
(0.273 vs 0.091) suggest a real, if early, capability gap, not just a
lucky loss-curve coincidence.

**Peak memory during training**, the actual motivating question:

| Stage | Cells | Steps | Peak CUDA memory |
|---|---|---|---|
| Nursery | 4 | 5,000 | 37 MB |
| Stage 2 | 16 | 10,000 | 93 MB |
| Stage 3 | 64 | 15,000 | 310 MB |
| Naive | 64 | 30,000 (all of it) | 308 MB |

For **half of its total 30,000-step budget** (nursery + stage 2, steps
1-15,000), the growth path never touched more than 93 MB - 3.3x to 8.4x
less than the naive run needed for its *entire* run. Growth also finished
its full 30,000 steps in 261.13s total wall-clock vs. naive's 284.40s for
the same step count - about 9% faster in absolute terms, on top of the
quality lead, at this scale.

A 20% relative PPL gap (49.00 vs 60.99) at only 30k steps on a toy model
is a large effect for a run this short, not a marginal one - if this were
initialization noise you'd expect a few percent, not a gap that holds up
consistently across content_ppl, content_top5, and coverage at the same
time.

## Results at 45,000 steps (both extended by +15,000)

| Metric | Growth | Naive |
|---|---|---|
| loss | 3.7850 | 3.8163 |
| **ppl** | **44.03** | **45.44** |
| content_acc | 0.091 | 0.091 |
| content_ppl | 178.36 | 193.83 |
| content_top5 | 0.273 | 0.182 |
| ws_omission | 0.182 | 0.091 |
| coverage | 0.818 | 0.909 |

Growth's lead narrows considerably (PPL gap 20% -> 3%) but doesn't
reverse - it stays modestly ahead on ppl/content_ppl/content_top5, while
naive catches all the way up on content_acc and actually pulls ahead on
whitespace-omission and coverage specifically. This is a genuinely mixed
picture at 45k, unlike the clean sweep at 30k: naive's longer exposure at
the full target size seems to buy back most (not all) of growth's head
start, at least on some axes.

Note: growth's 45,000-step total includes only 30,000 steps actually run
*at* 64 cells (15,000 in stage 3 + 15,000 in the extension) - the other
15,000 were spent at smaller sizes. So growth reached a comparable/better
result than naive's 45,000 steps at full size while spending only 30,000
of its own steps at that size, plus a cheaper 15,000-step warm-up.

## Interpretation

The best current explanation: the growth path spent its first half at a
much smaller, easier-to-optimize search space (4 then 16 cells), where
useful routing structure could form quickly and cheaply, then transferred
that head start into the 64-cell stage - rather than the naive run
spending its entire budget searching a much larger space from a cold
random start. The clean 30k-step sweep and the narrowing (but persisting)
45k-step gap together suggest this is mostly a genuine head-start effect
that a sufficiently long naive run can partially, but perhaps not fully,
buy back - though 45k steps is still short and this needs a longer/larger
follow-up before treating that as settled.

## Caveats

- Toy scale only (64 cells, d_model=16, ~4k vocab). The memory-scaling
  *pattern* (peak memory tracks current stage size, not final target size)
  should hold at real scale by the same architectural argument that
  produced it here, but the absolute numbers obviously don't transfer.
- Single seed, single run per condition - no variance estimate. Given how
  large and consistent the 30k-step gap was, it's unlikely to be pure
  noise, but this hasn't been confirmed with repeated seeds.
- `cell_transport_mode=broadcast_mix` was used, not `stewardship` - the
  dense/packed execution-controller dynamics (and the associated memory
  behavior once a model has grown enough to prune) haven't been tested
  under growth yet.
- No quarantine phase (temporarily blocking parent/sibling communication)
  was implemented - scattering alone was tested. An earlier, much smaller
  and weaker test (synthetic random-token targets, 300 steps, see session
  log) suggested scattering alone might not differentiate siblings much in
  weight-space, but wasn't repeated here with real data/longer training to
  confirm either way.
- Growth schedule (4->16->64, clones_per_cell=4 throughout, step split
  5k/10k/15k) was a single reasonable choice, not swept. Different splits,
  clone multipliers, or intermediate stage counts weren't compared.

## Follow-up: sparse recruitment (`cell_transport_mode=stewardship`, capped at 32/64 cells)

The first ablation used `broadcast_mix` (every cell always active every
pass). This follow-up switches to `cell_transport_mode=stewardship` with
`--stewardship-max-active 32` once cell count exceeds 32 - i.e. only half
the grown population can be recruited per token per forward pass,
simulating the sparse-recruitment regime this whole growth idea is
ultimately meant to support. The cap only applies once it's meaningful
(nursery at 4 cells and stage 2 at 16 cells stay uncapped -
`stewardship_max_active` must be <= the stage's own cell count).

Same schedule and step budget as the first ablation (4->16->64,
5k/10k/15k = 30k, then both extended +15k to 45k). One implementation
bug found and fixed along the way: `handoff_bias`'s non-cell dimension
depends on `neighbors + 1` (not just cell count), which legitimately
differs across stages (`neighbors` is forced to 2 at 4 cells, 4 at
larger stages) - `grow_gator_state_dict` now falls back to a per-cell
parameter's fresh initialization if its non-cell dimensions don't match
between stages, rather than crashing on a shape mismatch.

### Results at the matched 30,000-step point

| Metric | Growth (4->16->64, capped) | Naive (64 from scratch, capped) |
|---|---|---|
| loss | 4.0633 | 3.9667 |
| **ppl** | 58.17 | **52.81** |
| acc / exact | **0.312** | 0.188 |
| content_acc | **0.091** | 0.000 |
| content_ppl | 256.90 | **199.29** |
| content_top5 | 0.273 | **0.364** |
| coverage | 1.000 | 1.000 |

**This reverses the headline finding from the uncapped experiment**: under
sparse recruitment, naive is now ahead on raw PPL/content_ppl/content_top5,
while growth stays ahead on accuracy and content_acc (naive got zero exact
content tokens right despite its better perplexity). Not a clean win for
either side at this step count - a genuinely different, more mixed
picture than the uncapped case.

### Results at 45,000 steps (both extended by +15,000)

| Metric | Growth | Naive |
|---|---|---|
| loss | 3.8002 | 3.8577 |
| **ppl** | **44.71** | 47.35 |
| content_acc | 0.182 (tied) | 0.182 (tied) |
| content_ppl | 206.65 (tied) | 206.83 (tied) |
| **content_top5** | **0.364** | 0.182 |
| coverage | 0.909 (tied) | 0.909 (tied) |
| ws_omission | 0.091 (tied) | 0.091 (tied) |

By 45k, growth has caught back up and re-taken the PPL lead (44.71 vs
47.35), and the two runs have converged almost exactly on content_acc,
content_ppl, coverage, and whitespace-omission - genuinely tied, not
close-but-different. The one metric that stays clearly separated is
**content_top5: growth is 2x better** (0.364 vs 0.182), meaning growth's
top-5 candidate quality is meaningfully better even where top-1 accuracy
has converged.

### Interpretation

Under sparse recruitment, growth's advantage is smaller and noisier than
under `broadcast_mix` - it briefly trails at 30k on the aggregate loss
metrics before re-taking the lead by 45k, rather than leading cleanly
throughout. A plausible read: capping recruitment to 32-of-64 cells
reintroduces a real optimization problem (which 32 get used, and does
that choice stay good) that growth's inherited structure doesn't
fully insulate against, at least not immediately - both runs have to
work through real routing-selection dynamics under the cap, and growth's
head start shows up more as steadier long-run quality (content_top5,
final PPL) than as an immediate advantage at every checkpoint.

One point the earlier "cell starvation" question raised: naive's routing
had to select a working subset of 64 randomly-initialized, otherwise-
identical cells from a cold start, while growth's cells already carried
some inherited differentiation before ever facing the cap - this is the
most likely reason growth ends up ahead again by 45k even after a rockier
start, though this session didn't measure per-cell recruitment frequency
directly to confirm which cells specifically got starved in either run.

### Caveats specific to this follow-up

- Peak memory for the nursery/stage-2 stages wasn't re-captured before the
  Colab sessions were released (only the final 64-cell-stage figures were
  pulled: ~318MB for both growth and naive, identical since both ran the
  same cap at the same cell count). The memory-scaling pattern from the
  first ablation should still hold architecturally (it's driven by
  activation tensor shape, not transport mode), but this wasn't directly
  re-verified under `stewardship`.
- `stewardship_execution=dense` was used throughout (not `packed_auto`) -
  the dense/packed execution-controller transition itself (the original
  hardware/software co-design motivation) still hasn't been tested under
  growth.
- Single seed, single run per condition, same as the first ablation.

## Follow-up: schedule sweep — more steps at the cheap early stages

Same sparse-recruitment setup (`stewardship`, capped at 32/64), same
45,000-step total budget, but reallocated: 10,000/12,000/23,000
(nursery/stage2/stage3) instead of the original 5,000/10,000/15,000+15,000
extension. More of the budget spent while the model is still small and
cheap, less spent at the capped 64-cell size.

| Metric | New schedule (10k/12k/23k) | Old schedule (5k/10k/15k+15k) | Naive (45k) |
|---|---|---|---|
| **ppl** | **36.44** | 44.71 | 47.35 |
| content_ppl | **149.76** | 206.65 | 206.83 |
| content_top5 | 0.364 (tied) | 0.364 | 0.182 |
| content_acc | 0.091 | 0.182 | 0.182 |
| acc / exact | 0.312 | 0.375 | 0.375 |
| coverage | 0.909 (tied) | 0.909 | 0.909 |
| ws_omission | 0.091 (tied) | 0.091 | 0.091 |

A substantially better PPL/content_ppl (36.44 vs 44.71/47.35 - a large
jump, not a marginal one) and the same 2x content_top5 edge over naive,
at the cost of somewhat worse content_acc and raw accuracy than either
the old schedule or naive. Not a strict win on every axis, but the
size of the PPL/content_ppl gain makes this the most promising schedule
tested so far - front-loading steps into the cheap small-population
stages looks like a real, exploitable lever, not just noise. This trades
some exact-match precision for better calibration; which of the two
matters more depends on the eventual use case.

## Follow-up: how far can growth push cell count before OOM? (single G4)

A different question from the quality ablations above: not "is growth
better," but "how large a cell count can growth reach on one G4 before
running out of memory," at a more realistic context length (1024, not
128) and with recruitment capped at 64 active cells once cell count
exceeds that. Grew by 4x each stage (4 -> 16 -> 64 -> 256 -> 1024 -> 4096),
100 steps per stage (just enough for a stable memory/throughput reading,
not full convergence), same toy d_model/vocab/embedding_dim as the
quality ablations, `context_tokens=1024, response_tokens=32, batch_size=4`.

| n_cells | max_active | peak memory | steps/sec |
|---|---|---|---|
| 4 | uncapped | 0.105 GB | 16.95 |
| 16 | uncapped | 0.395 GB | 15.19 |
| 64 | uncapped | 1.468 GB | 6.99 |
| 256 | 64 | 5.727 GB | 2.13 |
| 1024 | 64 | 22.854 GB | 0.55 |
| 4096 | 64 | **OOM** | — |

Coarse boundary: between 1024 and 4096 cells. A follow-up doubling sweep
(4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048 -> 4096,
same config, finer steps to pin down where the ceiling actually sits)
filled in the gaps and pushed one step further:

| n_cells | max_active | peak memory | steps/sec |
|---|---|---|---|
| 4 | uncapped | 0.105 GB | 17.13 |
| 8 | uncapped | 0.199 GB | 18.54 |
| 16 | uncapped | 0.395 GB | 15.34 |
| 32 | uncapped | 0.740 GB | 11.17 |
| 64 | uncapped | 1.468 GB | 7.22 |
| 128 | 64 | 2.874 GB | 4.02 |
| 256 | 64 | 5.728 GB | 2.14 |
| 512 | 64 | 11.433 GB | 1.09 |
| 1024 | 64 | 22.861 GB | 0.55 |
| 2048 | 64 | 45.667 GB | 0.27 |
| 4096 | 64 | **OOM** | — |

**Precise boundary: 2048 cells fit (45.67 GB), 4096 did not.** Memory
scaling stayed almost perfectly linear the whole way (45.667/22.861 =
1.998x for exactly 2x cells) - no sign of the linear trend breaking down
even at this larger scale. Note the GPU tier used here clearly has well
over 24 GB available (2048 cells alone used 45.67 GB), more headroom than
the "G4 ~ 24 GB" assumption in the original coarse sweep - the useful
takeaway is the *scaling relationship*, not an assumed absolute ceiling
for any particular card. At a 97.7M-param real config (per the production
convergent run elsewhere in this project), the same cell count would cost
proportionally far more memory than this toy d_model=16 config, so these
absolute numbers don't transfer directly to a larger model - only the
linear-in-cell-count scaling pattern should.

A real, important caveat this sweep surfaces: **capping recruitment to 64
active cells does NOT cap memory scaling with total cell count.** Memory
kept growing roughly 4x per 4x cell increase even past the point where
recruitment was capped (256->1024 cells: ~4x more cells, ~4x more memory,
despite both being capped at the same 64 active cells) - because the
per-token ownership/mass tensors still need one slot per cell regardless
of how many of those cells actually get used on a given forward pass. The
`stewardship_max_active` cap limits *compute* (which cells' node updates
actually run) but not the *memory* footprint of tracking ownership across
the full population. This means the earlier hope that a recruitment cap
would let growth scale cell count independently of memory doesn't hold in
the current implementation - the ownership-tracking tensors themselves
are the uncapped cost, a different bottleneck than the one this whole
growth investigation started from (activation tensors during dense
computation). Fixing this would need ownership/mass tracked in a
genuinely sparse structure (only for recruited cells), not the current
dense `[batch, cells, tokens]`-shaped tensor - a real, separate
engineering problem from cell growth itself.

Throughput also degrades sharply with cell count even within the capped
regime (2.13 -> 0.55 steps/sec from 256 to 1024 cells, worse than the
~4x memory scaling) - likely the fixed per-cell bookkeeping overhead
(constructing/updating the full ownership tensor every step) rather than
the capped compute itself.

For scale, this toy config's parameter count barely moves with cell
count - `n_cells=1024` at `d_model=16, embedding_dim=8, vocab=4000` is
only **~2.17M params** (4->44.7k, 16->69.8k, 64->169.6k, 256->569.2k,
1024->2.17M, 2048->4.30M, 4096->8.56M). The memory cost is almost
entirely activation/ownership-tensor overhead from tracking per-cell
state per token, not the weights themselves - a genuinely lopsided
params-to-memory ratio compared to a dense model of similar size.

**Practical takeaway (user's call, given current resources): treat ~512
cells as the realistic working ceiling** (11.4 GB, ~1 step/sec at this
toy config - already a meaningful compute/memory cost for iteration
speed) rather than chasing the raw 2048-cell technical limit, and scale
further capability by widening cells (`d_model`) instead of adding more
of them. Widening costs parameters/FLOPs, not this activation-tensor
explosion, so it's a structurally different (and cheaper) scaling axis
than cell count - multi-thousand-cell Gators are viewed as a "high-end,
better-funded-hardware" scale for later, not a near-term target.

## A practical recipe this makes possible: size selection by growing until it hurts

A useful methodological consequence of the two follow-ups above: fixing
the architecture size upfront (how many cells, before ever training)
stops being necessary. Since growth is cheap relative to training a
target size from scratch (the quality ablations), and the OOM/throughput
sweep shows memory and speed degrade predictably and near-linearly with
cell count, sizing a Gator becomes a search procedure instead of a guess:

1. Fix the cheap dimensions (vocab size, `d_model`, context length) for
   the task at hand.
2. Start a small nursery and grow cell count (clone + scatter), same
   mechanism as the ablations above.
3. Keep growing until hitting whichever practical ceiling comes first:
   OOM, or steps/sec too slow to iterate on (a judgment call, not a hard
   number - this session's own call was ~512 cells at a toy config,
   trading further cell-count growth for widening `d_model` instead once
   that felt too slow).
4. That last-successful stage's cell count is the working size for this
   task/hardware combination - not a value chosen in advance.

This doesn't remove the need to separately tune `d_model` (a different,
cheaper axis - see the "practical takeaway" above) or to validate that
quality is actually still improving at the chosen size (the quality
ablations, not the OOM sweep, answer that) - it specifically replaces
"guess a cell count and hope it's right" with "grow until it stops being
worth it," using tooling this session already built and validated.

Stopping at a practical ceiling is a pause, not a dead end: the
checkpoint at whatever cell count training stopped at is a fully valid
model in its own right, and also a valid `--grow-from-checkpoint` source
later. Get access to more VRAM/compute down the line, and growth resumes
from exactly that checkpoint rather than starting over from a fresh
nursery - the current hardware ceiling only bounds *when* you stop
growing, not what you can eventually reach with the same lineage.

## Open next steps

- Sweep the growth schedule itself (more/fewer stages, different step
  splits) to see if the head-start effect is sensitive to how gradually
  the population grows.
- Test with `stewardship_execution=packed_auto` (dense->packed transition
  itself) — this session tested `stewardship`+capped recruitment but with
  `dense` execution throughout; the actual dense->packed transition remains
  untested under growth.
- Implement genuinely sparse ownership/mass tracking (only for recruited
  cells) so `stewardship_max_active` caps memory as well as compute — the
  OOM sweep found the current dense `[batch, cells, tokens]` ownership
  tensor is the real uncapped cost at large cell counts, a separate
  problem from the activation-tensor cost this investigation started from.
- Track per-cell recruitment frequency directly (not just the aggregate
  `support`/`dead` diagnostics) to confirm whether naive's cold-start
  routing actually starves specific cells more than growth's does, per the
  "cell starvation" hypothesis this follow-up's results are consistent
  with but didn't directly measure.
- Re-test sibling differentiation (weight-space and/or hidden-state
  similarity) under real data and full-length training, now that the
  mechanism is wired into the real trainer instead of a standalone script.
- Repeat at a larger scale (more cells, wider d_model) to see whether the
  effect size holds, grows, or shrinks - this session only tested one
  scale point.
- Multiple seeds per condition to get a real variance estimate before
  treating the 30k-step gap as a settled result rather than a strong first
  signal.
