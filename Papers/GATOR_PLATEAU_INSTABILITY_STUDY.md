# Gator Plateau/Regression Root-Cause Study

**Status:** Root cause isolated to decode mode (section 6). Capability validation phase
(section 7) found that Gator's dramatic tiny-corpus advantage over a transformer was mostly
a corpus-repetition artifact — on a fairer, less-repeated corpus a real but much smaller
Gator edge survives (21.16 vs. 29.50, ~28% better). See section 7.1.
**Primary question:** Why does Gator's free-generation quality repeatedly improve, plateau,
and then regress during training, and why did the "modern" combination of multi-jury
terminal voting, per-slot node isolation, and a staged/node-priority optimizer look worse
in practice than the original unmoderated architecture despite better logged PPL?
**Leading theory at start:** catastrophic rewriting of shared graph nodes by the optimizer.

## 1. Prior evidence (existing research, re-verified)

`Papers/GATOR_OPTIMIZER_ALIGNMENT_CPU_STUDY.md` and the 1M-scale block-coordinate/alternating
experiments in `experimental_checkpoints/` already contained most of the raw signal; this
study re-verified the key numbers directly from source logs rather than trusting summaries.

- Private-node update-to-parameter ratios under monolithic AdamW were consistently the
  largest of any parameter group despite the smallest raw gradients (Adam's per-parameter
  normalization). A 1.30x private-node LR multiplier "radically accelerated" autonomous
  release but caused excessive teacher-forgetting; every static multiplier tested traded
  retention for autonomy. The bracket search settled on 1.25x as the best free-first static
  rate, but 10,000-step continuations from that checkpoint (`09a`–`09d` in
  `experimental_checkpoints/optimizer_alignment_colab/`) **all drifted upward** in free PPL;
  none sustained the improvement.
- Direct drift quantification (`09_route_semantic_drift.json`) confirmed higher relative
  private-node LR causes measurably more parameter/functional drift (parameter cosine
  similarity to reference: 0.68 at 1.25x vs. 0.84 at 0.8x), but drift severity did not
  cleanly predict which arm had the better logged free PPL — an early instance of the
  PPL/quality mismatch documented in section 3.
- Re-extracted directly from `experimental_checkpoints/gator1m_alternating/` (previously
  unexplored): at matched step 20,000, four block-coordinate freeze variants split cleanly.
  Private-node-only training held content PPL 271; private-node+jury held 283; route/jury-
  frozen collapsed to 806; transport-frozen collapsed to 797. Whichever variant kept private
  nodes as the actively-training component landed **~3x better**.
- Re-extracted directly from `experimental_checkpoints/gator1m_n109_d64_l32_all_open_seed0_50k`
  vs. `..._slot_isolated_seed0_50k` (RUN_SPEC-confirmed matched configs, the only changed flag
  is `--slot-isolation-slots 32`): the isolated run started **comparably** (content PPL 151.7
  vs. 165.2 at step 5,000, isolated slightly ahead) but diverged **catastrophically faster**
  (2,354 vs. 345 by step 15,000). Node starvation was suspected (109 nodes / 32 slots ≈ 3.4
  nodes/slot with no LR recalibration) but not yet confirmed as the cause — see section 4.

### 1.1 A code-level bug found during this study

`run_gator_100m_staged_colab.py` (the runner behind the currently-bundled Gator-100M Colab
package) hardcodes `state["fixed_route_temperature"] = 1.25` at its Phase A→B→C
transitions, i.e. route temperature stays permanently hot/undifferentiated through the
entire private-node-only autonomous handoff phase. Its own validated 1M-scale ancestor,
`run_gator_1m_block_coordinate_colab.py`, instead computes this value via
`staged_temperature_at_step(retained_step)` — the same 1.25→0.55→0.35 cooling schedule that
`GATOR_TWO_PHASE_TEACHER_HANDOFF_TEMPLATE.md` and the CPU alignment study both validate as
necessary for retention. This divergence has not yet been confirmed as a contributor to the
plateau (the small-scale experiments below did not vary route temperature), but it is a
concrete, fixable discrepancy between the documented policy and the shipped runner and should
be corrected or deliberately justified before the next production 100M run.

## 2. User correction (critical, overrides prior framing)

Hands-on training on this laptop established two facts that the PPL-driven prior research
had gotten backwards or missed:

1. **Slot isolation does not fix the collapse.** It only raises the floor / delays the point
   of collapse into gibberish; it does not prevent it.
2. **The unmoderated original Gator (single jury, monolithic optimizer) produces better
   *qualitative* free generation** — contextually plausible "broken English" — **than the
   multi-jury + staged-optimizer + isolated-slot combination, despite worse quantitative PPL.**

This means PPL-based checkpoint selection, as used throughout the prior optimizer-alignment
and scaling studies, is not a reliable proxy for actual generation quality and may have been
selecting against it. Every finding below was designed and read with this in mind — actual
`formed=`/`free_formed=` text samples were inspected alongside PPL numbers wherever
practical, not PPL alone.

## 3. New small-scale ablation campaign (this study)

All new runs used SmolTalk Everyday only (`smoltalk_everyday.jsonl`, 2,260 conversations,
identical corpus to the historical CPU study), CPU execution (no CUDA available on the test
laptop), batch size 16, seed 0, constant learning rate 0.001, AdamW epsilon 1e-8, weight
decay 1e-4, gradient clip 1.0. **Learning rate was never varied — every single run in this
entire investigation, historical and new, used a constant LR.** This is flagged as the single
most important untested variable; see section 6.

### 3.1 Five-cell factorial (35,324–35,856 params, 8,000 steps each, `decode_mode=parallel`)

Base config: V1024/E12/D24/N20/degree4/relays2/lanes4, router `shared_low_rank` rank 8,
ingress learned, active top-k 4, `small_world` graph seed 0 rewire 0.5, context 24/response
12 tokens.

| Cell | Change from cell 1 | Final content PPL (step 8,000) | vs. cell 1 |
|---|---|---:|---:|
| 1 | baseline: global jury, monolithic optimizer, shared nodes | 99.3 | 1.0x |
| 2 | + `terminal-vote-mode slot` (multi-jury) | 255.2 | 2.6x |
| 3 | + `slot-isolation-slots 12` (isolation, ~1.67 nodes/slot, starved) | 510.4 | 5.1x |
| 4 | + `optimizer-groups component_scaled --optimizer-private-node-lr-multiplier 1.25` (full combo) | 617.9 | 6.2x |
| 5 | component-scaled optimizer alone (no jury/isolation change) | 170.0 | 1.7x |

Every cell shows the same qualitative shape regardless of configuration: content PPL bottoms
between steps 2,000–3,600, then climbs steadily for the remainder of the run while content
top-1/top-5 accuracy and formation coverage keep *improving*. Manual inspection of `formed=`
samples in cell 1 confirmed the numeric signature is real: samples in the post-peak window
(steps 4,800–6,800) show visibly degenerate repeated-character text (`'A tttts'`, `'A can rr
r'`, `'Yes, thesssr'`) that are absent from the earlier, better-PPL window.

Cells 4 vs. 3 (+21%) and cell 5 vs. cell 1 (+71%) show the staged optimizer's contribution is
real but **sub-additive** when stacked on top of jury+isolation — the three factors do not
combine additively; once jury+isolation have already driven severity up 5x, an additional
destabilizing factor has less marginal room to worsen things further.

### 3.2 Capacity-matched isolation pair (45,098–45,210 params, 8,000 steps, monolithic optimizer)

Designed to rule out node starvation as a confound for the historical 1M all_open/isolated
result. Base config: V512/E8/D24/N48/degree4/relays2/lanes4, router `shared_low_rank` rank 6,
context 24/response 12.

| Config | Final content PPL (step 8,000) |
|---|---:|
| Shared, global jury (48 nodes, no split) | 64.4 |
| Isolated, 4 nodes/slot (`slot-isolation-slots 12`, generously provisioned) | 412.6 (6.4x) |

Even with 4 nodes/slot instead of the historical run's ~3.4 or this study's own severely
starved 1.67/slot factorial cell, isolation still produced a **larger** relative degradation
(6.4x) than the starved factorial comparison (5.1x). **Isolation's harm is not a node-
starvation artifact** — it degrades the base instability even when reasonably provisioned.

### 3.3 Teacher-forced control (35,856 params, 8,000 steps, otherwise identical to cell 1)

| Decode mode | Peak content PPL (step) | Final content PPL (step 8,000) | Ratio |
|---|---:|---:|---:|
| `parallel` (cell 1) | 55.4 (3,200) | 99.3 | 1.8x |
| `teacher_forced` | 27.8 (2,000) | 101.0 | 3.6x |

Teacher-forcing delays onset and roughly doubles the achievable peak quality (content PPL
27.8 vs. 55.4), but by step 8,000 it converges to **essentially the same absolute endpoint**
as the parallel-decode baseline (101.0 vs. 99.3). This is a difference in *timing and peak*,
not a difference in *kind* — teacher-forcing does not appear to prevent the underlying
instability, only postpone and soften it, at least within this step budget.

### 3.4 Real-scale confirmation (in progress at time of writing; see handoff doc for live status)

Two long runs at 26,538 (`architecture=gator`/Arrow, single jury inherent to the
architecture, monolithic optimizer, `decode_mode=parallel`, cells=10/nodes=4/relays=2) and
61,770 params (`architecture=convergent`, multi-jury + isolation (`slot-isolation-slots 16`)
+ staged optimizer (private-node 1.25x), otherwise the CPU-study frozen-proxy scale) were
launched for up to 50,000 steps each to check whether the small-scale pattern holds at
longer horizons and whether it reveals a difference in *kind*, not just degree, between the
plain and "modern" configurations.

**Arrow completed its full 50,000-step run** (18,859 seconds ≈ 5.24 hours on CPU). Final
content PPL: 679.73, oscillating noisily in a 600–900 range over the last several thousand
steps rather than continuing to climb — roughly 7x its early optimum (~97–99) but a
**contained plateau**, not a runaway. Sample generation at step 50,000:
`formed='Yes, ceseeisse    '` — a coherent opening word followed by garbled continuation,
consistent with degraded-but-bounded output.

**Convergent also completed its full 50,000-step run** (19,597 seconds ≈ 5.44 hours). Final
content PPL: **702,730.51** — roughly 8,300x its early optimum (~85), and still climbing with
no sign of leveling off at the final step (690,546 → 702,731 over the last 500 steps). Its
full trajectory (85 → 39,879 @33,000 → 178,092 @42,000 → 242,983 @45,000 → 432,903 @45,500 →
702,731 @50,000) is the same shape as the historical all-open 1M diagnosis (165 → 5,886 by
step 50,000), reproduced here at 16x smaller scale (61,770 vs. ~1M params) and reaching two
orders of magnitude further before hitting the same step budget. Terminal-vote diagnostics at
the final step (`effective=1.62, top1=0.758, winning_lanes=1.00`) show the multi-jury routing
has collapsed onto essentially one permanent lane.

**Final matched-step comparison, both runs complete at step 50,000, same corpus, same
seed:** Arrow 679.73 vs. Convergent 702,730.51 — a **~1,034x gap**. This is the single
cleanest, most decisive result in the whole study. Both architectures degrade from their
early optimum, but the plain/unmoderated one plateaus into a noisy, bounded, still-partially-
readable state, while the "modern" combination undergoes total unbounded collapse. This
directly and quantitatively explains the user's original qualitative observation.

### 3.5 Open question: does Arrow's "cells" structure matter, or is it irrelevant?

Motivated by the observation that Arrow (`architecture=gator`) uses many small cells (each
with a handful of private nodes and restricted neighbor-only connectivity) while Convergent
uses one large node pool with degree/small-world connectivity, we asked whether discarding
"cells" in favor of "a single graph with more nodes" threw away a structural protection.

At the one point with matched step counts (step 8,000), Arrow (cells) showed content PPL
133.96 — **slightly worse**, not better, than the plainest cells-free Convergent baseline
(cell 1, 99.3). This is a real data point against cells being independently protective, but
it is confounded (Arrow and Convergent differ in transport mechanism, lexical handling, and
more, not just cells vs. single graph).

A cleaner, single-variable test was run: cell 1's exact config, changing only graph
connectivity from `small_world`/50% rewire (long-range shortcuts across the whole node pool)
to `ring`/0% rewire/degree 2 (strictly local, neighbor-only connectivity — the closest
existing analog to what cells structurally enforce), holding node count, jury mode,
optimizer, and everything else fixed (35,324 vs. 35,856 params).

**Result: the local/cell-like topology was worse, not better** — final content PPL 122.0 at
step 8,000, versus cell 1's 99.3 (a 23% regression). The two trajectories tracked each other
closely through most of training (within ~10% at every logged step through 7,200) and both
show the identical bottom-then-climb shape; the ring/local variant simply ended slightly
worse. Combined with the Arrow-vs-Convergent comparison above, **both available pieces of
evidence argue against cells/local connectivity being a protective mechanism** — restricting
node connectivity to neighbors only does not defend against the underlying calibration-drift
instability, and mildly hurts if anything. This question is considered resolved for this
architecture family and parameter scale; it does not need to be repeated on Colab unless a
much larger node count changes the picture.

## 4. Synthesis

1. **Root cause, universal:** training simultaneous/parallel-formation output with a
   constant learning rate on a small, finite dataset produces an inherent calibration-drift
   instability. It appears in every configuration tested — both architectures, every jury
   mode, with or without isolation, with or without the staged optimizer. Content PPL bottoms
   early (step ~2,000–3,600 in every experiment regardless of scale or config) then climbs
   steadily while accuracy and coverage keep improving: the model is guessing at more of the
   response with progressively less-calibrated confidence, not simply forgetting language.
2. **Decode-mode autonomy governs onset speed and achievable peak, not whether the
   instability occurs.** Full autonomy (`parallel` from step 1) triggers it almost
   immediately at a mediocre peak; teacher-forcing delays it and roughly doubles peak
   quality, but converges toward a similar collapsed endpoint given enough constant-LR steps.
3. **Multi-jury, node-isolation, and the staged/node-priority optimizer are independent,
   non-additive severity amplifiers** on top of the same base instability, not separate
   causes. They compound sub-additively (roughly 2–5x individually, saturating when
   stacked) at small scale, and — per the real-scale runs in progress — may convert a
   contained, gradual drift into a genuine runaway explosion at production scale.
4. **Isolation's harm is not a node-starvation artifact**, confirmed via a matched-capacity
   test with 4 nodes/slot instead of ~1.7–3.4.
5. **Arrow's "cells" are not independently protective.** Both the confounded (Arrow vs.
   plain Convergent) and confound-free (small-world/rewired vs. ring/local, single-variable)
   comparisons show local/cell-like connectivity performing the same or slightly worse than
   full small-world connectivity. Discarding cells for a single larger graph did not throw
   away a protective mechanism, at this scale.

## 5. Recommendations for the next phase (GPU/Colab)

1. **Test a learning-rate schedule as the primary untested variable.** No experiment in this
   study or the prior CPU-alignment work ever varied LR away from constant. Given the
   universal "improves then regresses" signature under constant LR, a cosine or step decay
   schedule is the cheapest, least exotic candidate fix and should be tried before further
   architectural changes.
2. ~~Finish or reproduce the ring/no-rewire connectivity-locality test~~ — done, see section
   3.5; cells/local connectivity are not protective at this scale and this does not need to
   be repeated unless a much larger node count is suspected to change the picture.
3. Re-run the 5-cell factorial and the capacity-matched pair at 10–100x the parameter count
   and step budget on GPU to confirm the small-scale severity ordering holds, and specifically
   to check whether the gradual-vs-explosive distinction (section 3.4) reproduces reliably.
4. Fix or deliberately justify the hardcoded `fixed_route_temperature = 1.25` in
   `run_gator_100m_staged_colab.py` (section 1.1) before the next production 100M run.
5. Stop selecting or reporting "best" checkpoints by PPL/CE alone. Given the confirmed
   PPL/quality mismatch (section 2), any future ablation should inspect actual generated text
   samples before calling a treatment an improvement.

## 6. WSL/Colab GPU phase (2026-07-23) — decode mode isolated as the driver

All runs below use the cell-1 config unless noted: Convergent, 35,856 params, V1024/E12/D24,
N20/degree4/relays2/lanes4, `shared_low_rank` router rank 8, learned ingress, `small_world`
rewire 0.5, global jury, monolithic optimizer, constant LR 0.001, seed 0, SmolTalk Everyday,
8,000 steps, on a Colab L4 GPU. Logs for every run in this section are under
`experimental_checkpoints/plateau_instability_study_20260723/`.

### 6.1 Connectivity, closed

Ring/no-rewire (locality-only, degree 2) scored content PPL 153.23 @ step 8,000 vs.
small_world's 99.3 — worse, not better. Locality is not protective; section 3.5's open
question is resolved.

### 6.2 LR schedule and the auto-trigger — a real mitigation, not a fix, and scale-dependent

Constant LR peaked 52.56 @ 2,800 then exploded to 201.67 @ 8,000 (3.8x). Cosine capped the
explosion (83.66 final) without raising the achievable peak — worse than simply stopping
constant-LR training at its own best checkpoint (52.56). A hindsight milestone LR drop
(gamma 0.02 at step 3,000, picked by looking at the PPL curve) froze the model flat at ~58
PPL with no further learning after the drop.

An online, non-hindsight trigger was built and tested: back-prop each response slot's loss
separately (retaining the graph), average pairwise gradient cosine similarity across all slot
pairs on the shared router/terminal parameters, and drop the LR the first time an EMA of the
adjacent-slot cosine crosses from negative back to zero (see section 6.3 for why this signal
exists). With a moderate gamma (0.3, chosen so the model keeps learning after the drop rather
than freezing), it fired at step 3,200 (seed 0) and step 3,600 (seed 1) — close to the
hindsight-picked step both times, entirely from training-internal state. Final content PPL:
85.89 (seed 0) and 84.83 (seed 1), vs. 201.67 / 100.67 for unmitigated constant LR at those
seeds. Severity of the unmitigated regression is itself seed-dependent (3.8x vs 2.2x at seeds
0/1), so the trigger's *relative* benefit varies by seed even though its firing step does not.

**This is checkpoint-rollback with better timing, not a fix** — flagged explicitly during the
investigation, because it lowers the floor of the damage without addressing why the damage
happens. Confirmed by a scale test: at ~495k params (14x the cell-1 scale, same recipe,
`d_model=61/nodes=104/embedding_dim=30/router_rank=20`, following a "~1.7 nodes per unit of
width" sizing heuristic), constant LR peaked 50.41 @ 1,800 then exploded to **2,821.72 @ 8,000
— a 56x regression**, far worse than the 35k-param run's 3.8x. The gamma-0.3 trigger only dented
this (391.99 final, still 7.8x from its own peak); gamma had to be dropped to 0.02 (full
freeze) to hold the 495k model flat, exactly mirroring the small-scale milestone result. **This
directly contradicts this study's working assumption that Gator's instability is the same
regardless of parameter count** — severity scales up sharply with size, and any accepted
LR-based mitigation needs per-scale retuning, not a universal setting.

### 6.3 Cross-slot gradient conflict — the leading theory going into this phase, tested directly and refuted

The working theory motivating this phase (user hypothesis) was that the optimizer makes
conflicting writes to shared routing/terminal weights because different response slots want
different tokens through shared capacity (e.g. slot 0 wanting "The", slot 1 wanting "cat").
Tested directly: back-prop each slot's loss separately and measure gradient cosine similarity
on the shared parameters.

A single slot pair (0 vs. 11) gave pure noise — mean cosine ~0 across every training phase,
stdev (0.096) larger than any window mean. Averaging over all 66 slot pairs revealed a real,
much lower-variance signal, but the opposite of the theory: adjacent-slot cosine is strongly
**negative** early in training (-0.31 to -0.72, steps 200–1,800, while content PPL is still
improving — legitimate task differentiation as untrained slots pull apart) and crosses back to
near-zero/weakly positive exactly as the plateau begins (~step 2,000) and stays weakly positive
through the entire regression phase (steps 4,000–8,000). **Cross-slot conflict is highest during
healthy improvement and has already resolved before the regression starts** — this rules out the
theory as originally stated. (The crossing point of this same signal is what the section 6.2
online LR trigger uses — a useful marker of "differentiation is done," not evidence of ongoing
conflict driving the regression.)

### 6.4 Architecture control — not Gator-specific

A plain non-causal transformer (`--architecture transformer`, `OneShotTransformerLanguageModel`
— a BERT-style encoder that fills all response slots in one non-causal pass, i.e. the same
one-shot parallel-formation training paradigm as Gator, not a conventional causal/autoregressive
transformer) was trained on the identical recipe and data at matched param counts (~52k and
~478k, 16,000 steps since transformers converge slower here). Both **degraded almost
monotonically from step ~400**, far worse than Gator at any tested scale: content PPL reached
684,848 (52k params) and 7,905–15,551 (478k params, noisier) by step 16,000. **The instability
is not Gator-specific — a completely different, denser architecture under the same training
paradigm is equally or more unstable.** This rules out Gator's routing/shared-node machinery as
the aggravating factor; whatever is happening is a property of the training objective, not the
architecture executing it.

### 6.5 Decode mode, isolated variable — the actual driver

Holding architecture, data, and every other setting fixed, only `--decode-mode` was varied:

| Decode mode | What it does | Peak content PPL | Final content PPL @ 8,000 |
|---|---|---:|---:|
| `slot_ce` (default) | All 12 response slots predicted independently and simultaneously, no conditioning on each other's true values | 52.56 @ 2,800 | 201.67 |
| `viterbi` | Same simultaneous independent prediction, but with a learned/relaxed alignment between response length and slot count | ~56 @ 2,600 | 161.32 |
| `monotonic` | As `viterbi`, forward-sum monotonic alignment instead of Viterbi | ~54 @ 3,000 | 299.02 (worse than default) |
| `teacher_forced` | Still one simultaneous forward pass, but each slot's prediction is causally conditioned on the true previous response tokens (shifted references) — the standard way transformers train | 26.61 @ 2,800 | 102.29 (closely reproduces the historical CPU-study number: peak 27.8 @ 2,000, final 101.0 @ 8,000) |
| `serial` | True one-token-at-a-time autoregressive: one randomly chosen response position trained per step, conditioned on the true prefix | **never regresses** — monotonic improvement the entire run | **25.55**, still falling |

`viterbi`/`monotonic` exist to solve *where* a token maps to a slot when response length and
slot count don't line up cleanly — not the mechanism at play here, since Gator's slot
assignment is already fixed by architecture (each of the 12 response positions has a dedicated
terminal vote). Confirmed: neither fixes the regression, and `monotonic` is slightly worse than
doing nothing. **This rules out positional/slot-alignment ambiguity as the mechanism.**

`teacher_forced` isolates the remaining candidate: conditioning each slot's *content* (not
position) prediction on the true values of the other slots, computed in one simultaneous
forward pass. This gives a much better floor (2x lower peak than default) but still regresses
by step 8,000, closely matching the original CPU-study number — conditioning is a major lever,
not a complete fix, when computed within one simultaneous pass.

`serial` removes both the simultaneity and (implicitly) supplies the cleanest possible
conditioning, one true-prefix-conditioned token at a time, and shows **zero regression**,
the best result of any configuration tested in this entire study.

**Working conclusion:** the root cause is training all response positions' content
predictions independently and simultaneously in one pass, without conditioning each on the
others' true values. It is not: Gator's architecture (6.4 rules this out), parameter count
(6.2 shows severity scales *up*, not flat, with size — the opposite of this study's original
working assumption), cross-slot gradient conflict on shared weights (6.3 rules this out), or
positional/alignment ambiguity (6.5's viterbi/monotonic result rules this out). Conditioning
helps substantially (`teacher_forced` vs. `slot_ce`) but a residual regression remains even
with full causal conditioning, so long as it is computed as one simultaneous pass rather than
genuinely sequential per-token updates (`serial`) — why simultaneous-but-conditioned computation
still differs from truly sequential computation is not yet understood and is an open question.
Gator's relay/routing structure is *already* intended to let packets become mutually
context-aware as they traverse the graph (this is where inter-packet dependency is supposed to
enter architecturally); reconciling why that existing mechanism does not substitute for
explicit causal conditioning the way `teacher_forced`/`serial` do is unresolved and worth its
own investigation.

## 7. Next phase: capability, not just stability

Every mitigation found in section 6 (LR schedules, the auto-trigger, three-phase
route-freeze training) only lowers the *floor* of the damage; none addresses why the floor is
disproportionately mediocre for a model that has seen the same small, deduplicated corpus
thousands of times. `serial`/`teacher_forced` decode give a much better floor with no
architecture change, which reframes the priority: Gator's parallel-decode *training* speed is
not a hard product requirement (Gator's value proposition is being a smaller/smarter
architecture at a given parameter count, not fast parallel training), so the open question is
no longer only "how do we stabilize parallel decode" but "does Gator actually produce a
smarter model than a same-sized transformer at all, once both are trained on a decode mode
that doesn't regress."

Planned validation (not yet started as of this writing): train a Gator and a plain transformer
at matched parameter counts, both with `serial` decode, for the same number of steps, and
compare response coherence directly (qualitative generation quality, not just content PPL) —
before deciding whether `serial`/`teacher_forced` should replace `slot_ce` as Gator's standard
training recipe.

### 7.1 Gator vs. transformer, `serial` decode — a corrected finding: the "crossover" was mostly a corpus-repetition artifact, but a real, smaller Gator edge survives it

Motivation: Gator's architectural advantage over a same-sized transformer is already
established on structured/algorithmic tasks (modular math), where a well-documented
early-transformer-lead-then-Gator-crossover pattern exists. Whether this generalizes to
natural language was untested. Both models used `--decode-mode serial` (the only decode mode
shown in section 6 to avoid regression), same SmolTalk Everyday corpus, same seed 0, batch 16,
constant LR 0.001. Gator: 35,856 params (cell-1 config). Transformer: 51,872 params
(`d_model=24/heads=2/ff_width=76/layers=4`) — roughly 1.45x Gator's size, not an exact match;
a tighter param-matched rerun is a natural follow-up if a cleaner comparison is wanted.

**8,000 steps:** transformer finished at content PPL 22.88, Gator at 24.80 — transformer
slightly ahead, consistent with the modular-math precedent of an early transformer lead.
Gator's content PPL was still falling fast at the 8,000-step cutoff (42.68 → 26.59 → 27.09 →
24.80 over the last four logged points), suggesting 8,000 steps was too short a horizon to
see Gator's full trajectory — motivating the longer run below.

**24,000 steps, tiny corpus (SmolTalk Everyday, 2,260 conversations, ~170 epochs):**
transformer bottomed early (16.34 @ step 3,600) then began climbing around step 6,000–10,000
and kept climbing for the rest of the run, noisily but unmistakably, reaching 245.22 by step
24,000 — a **15x regression from its own best checkpoint**, under the exact same `serial`
decode mode that showed zero regression for Gator at 8,000 steps in section 6.5. Gator bottomed
later and lower (22.53 @ step 13,000) and drifted only mildly afterward, ending at 35.73 @
step 24,000 — a **1.6x drift from its own best**. Final: Gator 35.73 vs. transformer 245.22,
a dramatic ~6.9x gap.

**This result does not hold on a fairer, less-repeated corpus — the transformer's collapse
was mostly a corpus-repetition artifact, not a fundamental weakness.** The identical
architectures, decode mode, seed, and step count (24,000) were rerun on a 50,000-example subset
of `smoltalk_preference` (~7.7 epochs instead of ~170; validated against the training script's
own `parse_dialogue` to drop malformed/empty-turn rows, length-filtered to ≤1,300 chars to fit
the same 24/12-token context/response budget — see caveats below). Result: **transformer stayed
completely stable through all 24,000 steps, finishing at content PPL 29.50, with no sign of the
tiny-corpus collapse at all.** Gator finished at **21.16**. Side by side:

| Corpus | Epochs | Gator final | Transformer final | Gap |
|---|---:|---:|---:|---:|
| SmolTalk Everyday (2,260 examples) | ~170 | 35.73 | 245.22 (collapsed, 15x from its own peak) | ~6.9x |
| SmolTalk Preference subset (50,000 examples) | ~7.7 | 21.16 | 29.50 (fully stable) | ~1.4x |

**Corrected conclusion:** the dramatic tiny-corpus "crossover" was substantially a story about
which architecture handles ~170-epoch repetition/overfitting better, not a general claim about
which architecture is smarter at language. On a corpus with realistic (much lower) repetition,
the transformer does not collapse at all under `serial` decode. A genuine Gator edge survives
even here (21.16 vs. 29.50, Gator ~28% better) but it is far smaller than the original result
suggested, and this is still only ~7.7 epochs on 50k examples — not a large/fresh-data regime
by modern standards. Whether Gator's edge holds, grows, or shrinks further at even lower
repetition (a full single-epoch pass over the full 230k-example/617MB `smoltalk_preference`
file, or a genuinely large corpus like Wikipedia) is untested and the natural next step.

Two things worth being precise about, updated for both results:

1. **`serial` decode's stability is architecture- and corpus-dependent, not universal.**
   Section 6.5's "serial decode shows zero regression" held for Gator at 8,000 steps on the
   tiny corpus. The transformer's apparent instability under the same decode mode at 24,000
   steps on that same tiny corpus turned out to be a repetition artifact, not a property of
   the decode mode or the architecture — it fully disappeared with more/fresher data. A
   decode-mode fix (or an apparent architecture weakness) validated at one corpus repetition
   level should not be assumed to generalize to a different one without checking.
2. **Gator drifts mildly on both corpora** (22.53 → 35.73 on the tiny corpus; a similar mild
   climb-then-settle pattern visible in the 50k-corpus run) but never comes close to the
   transformer's tiny-corpus collapse magnitude. Whether Gator itself would eventually show
   more severe drift at a horizon beyond 24,000 steps, or at even higher repetition, is
   untested.

**Caveats on the 50k-example rerun, for anyone reproducing this:**
- `--skip-first-assistant` (used in the tiny-corpus run) was dropped for this corpus, because
  `smoltalk_preference` is mostly single-turn Q&A and that flag requires ≥2 assistant turns per
  conversation — with it enabled, the tiny 10%-validation split ended up with zero surviving
  examples even though the training split had enough. This is a real recipe difference, not a
  perfectly matched rerun.
- Only a 50,000-row subset of the ~230,470-row (~617MB) `smoltalk_preference` file was used,
  additionally filtered to rows that (a) parse cleanly under the training script's own
  `parse_dialogue` (a nontrivial fraction of raw rows have malformed/empty turns) and (b) are
  ≤1,300 characters (matching SmolTalk Everyday's own max length) so they fit the same 24/12
  context/response token budget without changing the model architecture. This is not the full
  corpus and not a random sample — it is biased toward shorter conversations.

### 7.2 Does scaling the transformer close the gap? No — and the relationship isn't even monotonic

Follow-up to 7.1's ~28% Gator edge (21.16 vs. 29.50) on the 50k-example corpus: does simply
giving the transformer more parameters close it? Three larger transformer sizes were trained
with the identical recipe (`--decode-mode serial`, 24,000 steps, same 50k-example corpus,
seed 0): `d_model=32` (78,016 params), `d_model=40` (110,064 params), `d_model=48`
(147,232 params) — all keeping `heads`/`ff_width` scaled proportionally, `layers=4` fixed.

| Transformer size | Final content PPL @ 24,000 steps |
|---|---:|
| 51,872 params (7.1 baseline) | 29.50 |
| 78,016 params | 49.55 |
| 110,064 params | 52.04 |
| 147,232 params | 39.78 |
| **Gator, 35,856 params** | **21.16** |

**None of the larger transformers beat the smallest one, let alone approached Gator, despite
up to 2.8x more parameters — and the ordering among the three larger sizes is not monotonic**
(110k params was the worst of the three, worse than both the smaller 78k and the larger
147k). This does not look like a clean capacity-vs-quality relationship; it looks
noise-dominated at this step count/corpus size, consistent with this study's repeated finding
that transformer content PPL keeps drifting upward over long `serial`-decode horizons (section
7.1) while Gator stabilizes. Simply scaling transformer parameter count is not, on this
evidence, a way to close the gap to Gator — whatever Gator is doing differently is not
reproduced by giving the transformer more width. A cleaner test (multiple seeds per size, or a
longer horizon to see whether the larger transformers' drift stabilizes at a lower floor
given more steps) is the natural follow-up if this scaling question is revisited.

### 7.3 Two reasoning loops per step essentially closes the gap — the likely real source of Gator's edge

Motivated by the question "what happens if the equal-size transformer gets more compute per
step, mimicking (not literally reproducing) Gator's 2-relay iterative depth?" Two different
interpretations of "give it 2 loops" were tested on the 51,872-param transformer, same
50,000-example corpus, `--decode-mode serial`, seed 0:

**Interpretation A — 2x total training steps (48,000 instead of 24,000):** a clear failure.
Content PPL kept climbing well past the 24,000-step endpoint, reaching **215.79 by step
48,000** — 7.3x worse than the 24,000-step baseline's 29.50. More constant-LR steps simply
means more time to drift; this confirms the section 7.1/7.2 pattern rather than contradicting
it. Longer training is not the lever.

**Interpretation B — 2 reasoning loops per training step:** implemented as a new
`--reasoning-loops {1,2}` flag in `gator_language_train.py` (currently supported for
`--architecture transformer` with `--decode-mode serial` only). When set to 2: the model
decodes once with `torch.no_grad()` to produce a draft token, that draft is fed back into one
extra reserved context position (the model is built with `context_tokens + 1`, adding only 24
parameters — 51,896 vs. 51,872, effectively the same size), then the model decodes again with
the draft visible, and the loss is computed only on this second, refined pass. This applies at
both training and evaluation time (`evaluate()` takes the same `reasoning_loops` parameter) so
reported metrics reflect the two-pass mechanism consistently.

Result, same 24,000 steps, same corpus, same near-identical param count:

| Config | Final content PPL @ 24,000 steps | Steps/sec |
|---|---:|---:|
| Gator (35,856 params) | 21.16 | ~28–30 |
| Transformer, 1 pass (51,872 params) | 29.50 | ~48–49 |
| Transformer, 2x total steps (48,000, still 1 pass) | 215.79 (catastrophic) | ~57 |
| **Transformer, 2 reasoning loops/step (51,896 params)** | **22.34** | **~44** |

**The 2-reasoning-loop transformer essentially matches Gator (22.34 vs. 21.16, within ~6%)**
— a dramatic improvement over the single-pass transformer's 29.50, using almost exactly the
same parameter count, and it does this while running *faster* than Gator (~44 vs. ~28–30
steps/sec) despite two forward passes per step, meaning Gator's per-pass routing computation
costs more than two transformer passes combined at this scale.

**This is the strongest evidence yet on what Gator's edge actually is.** Sections 7.1/7.2
showed the edge is real (survives corpus-repetition correction) but couldn't be bought back
with transformer parameter count. This result suggests it can be bought back with a cheap
architectural change unrelated to Gator's specific node/routing machinery: **giving the model
one extra round of self-conditioned refinement before committing to an answer.** That is
functionally closer to what Gator's multi-relay graph traversal does (the packet's
representation is refined across relay rounds before the terminal vote) than anything a
single-pass transformer does — consistent with the user's framing that this mimics, without
literally reproducing, Gator's 2-relay depth.

Caveats: single seed, one corpus, one param scale, and the "draft" here is argmax-decoded
under `no_grad` (a cheap approximation of iterative refinement, not a learned latent
reasoning step) — whether this holds at other scales, generalizes to `parallel`/`teacher_forced`
decode modes, or whether Gator retains an edge specifically in cases requiring more than one
extra refinement round, are all open.

**Update — question (b) answered, and it's a striking asymmetry.** `--reasoning-loops 2` was
extended to `--architecture convergent` (Gator/Convergent has no fixed `context_tokens` model
config the way the transformer does, so no model-size adjustment was needed — the two-pass
training-loop logic works unchanged). Because the first transformer comparison wasn't
param-matched (51,872 vs. Gator's 35,856, ~1.45x), a Gator config was built at 51,062 params
(`d_model=29/nodes=24/embedding_dim=14/router_rank=10`, within 1.6% of the transformer) to
make the comparison fair. Same corpus, same 24,000 steps, same seed 0:

| Config (~50k–52k params) | Final content PPL @ 24,000 steps |
|---|---:|
| Original Gator (35,856 params, 1-pass, ratio 0.83) | 21.16 |
| Transformer, 1-pass (51,872 params) | 29.50 |
| Transformer, 2 reasoning loops (51,896 params) | 22.34 (≈ matches original Gator) |
| Gator, matched size, wrong ratio 0.83 (51,062 params, 1-pass) | 27.38 |
| Gator, matched size, wrong ratio 0.83 (51,062 params, 2 reasoning loops) | 2,278.29 — catastrophic (~83x worse) |
| Gator, matched size, correct ratio 1.7 (50,325 params, 1-pass) | **17.01** |
| Gator, matched size, correct ratio 1.7 (50,325 params, 2 reasoning loops) | 1,042.10 — still catastrophic |

**Correction to an earlier reading of this table.** The first matched-size Gator config used
`d_model=29/nodes=24` — a nodes-to-width ratio of 0.83, not the project's validated ~1.7
nodes-per-unit-width sizing heuristic (confirmed as the actual ratio used in the real 100M
production config: `d_model=368/nodes=626` ≈ 1.70). Re-run at the correct ratio
(`d_model=24/nodes=41/embedding_dim=12/router_rank=8`, 50,325 params): **the larger Gator now
clearly beats the smaller original Gator, 17.01 vs. 21.16.** The earlier conclusion drawn from
the wrong-ratio result — "bigger Gator did worse, consistent with section 6.2's finding that
Gator doesn't scale monotonically" — does not hold once the sizing heuristic is respected.
Node-to-width ratio, not raw parameter count, is what determines whether scaling Gator up
helps; get the ratio wrong and more parameters can look actively harmful. Section 6.2's own
35k-vs-495k comparison should be treated with this caveat too — its ratio was not verified at
the time and is worth re-checking before continuing to cite that comparison as evidence against
size-monotonicity.

Two findings survive both ratio conditions:

1. **Reasoning loops help the transformer and hurt Gator in both ratio conditions — the
   opposite effect on the two architectures given the identical mechanism** (2,278.29 at the
   wrong ratio, 1,042.10 at the correct ratio — both catastrophically worse than either 1-pass
   baseline). The most plausible explanation: feeding a self-generated draft back through
   Gator's own routing/relay machinery means the draft re-enters the exact computation that
   produced it, likely compounding rather than correcting whatever calibration drift is
   already present (a feedback loop through a decision-making substrate). A transformer's
   extra context slot is just one more token for ordinary self-attention to weigh in on — no
   such feedback loop through a routing mechanism exists. This answers the earlier open
   question directly: **Gator is not already doing this via its relays in a way that makes the
   explicit loop redundant — the explicit loop is actively harmful for Gator, while being the
   single most effective intervention found in this entire study for the transformer.**
2. **Ratio does modulate severity even though it doesn't eliminate the incompatibility** —
   1,042.10 (correct ratio) vs. 2,278.29 (wrong ratio) is still more than 2x better, so node/width
   configuration is a real lever on how badly the loop mechanism hurts Gator, just not enough
   to make it net-positive at either ratio tested.

Open follow-ups: does a gentler feedback mechanism (e.g. feeding a soft/probability-weighted
draft rather than a hard argmax token, or routing the draft through a dedicated side-channel
instead of re-entering the primary graph traversal) avoid destabilizing Gator while still
capturing whatever benefit refinement provides transformers? Does 3+ reasoning loops keep
improving the transformer, plateau, or start hurting it too (diminishing/negative returns from
too much drafting)? Does this asymmetry hold at other param scales? Does re-checking section
6.2's 35k-vs-495k Gator comparison at a verified-consistent node ratio change that section's
conclusion the way it changed this one?

**One more data point — the scaling benefit is not unlimited, even at the correct ratio.** A
third Gator size was tested at the same 1.7 ratio: `d_model=61/nodes=104/embedding_dim=30/
router_rank=20`, 494,964 params (matching section 6.2's original 495k config). Same corpus,
same 24,000 steps, same seed 0: final content PPL **25.19** — worse than the 50,325-param
version's 17.01, though still close to the original 35,856-param Gator's 21.16. So across the
three sizes tested at the correct ratio (35,856 → 50,325 → 494,964), quality improves from the
smallest to the middle size, then degrades again at the largest — a peak, not a monotonic
"bigger is better" relationship even once the node/width ratio is fixed. Whether this reflects
a true capacity/data mismatch at ~495k params for a corpus this size (50,000 examples), or is
itself a milder version of the same underlying constant-LR/long-horizon drift this whole study
has been chasing, is unresolved and the natural next question if this line of investigation
continues.

## Section 8: 100M-scale follow-up — Gator is unusually batch-size-sensitive at width

After this study closed, an attempt to run the full ~100M-param production Gator config
(`d_model=368/nodes=626`, ratio 1.7, same architecture as sections 6-7 but scaled up, on the
real multi-corpus production mixture at 512-token context/128-token response, `--batch-size 1`)
showed content collapse: `content_acc` and `coverage` pinned at exactly 0.000, `formed=' '`
(blank) on every logged sample, PPL climbing rather than converging. A sequence of sanity
checks isolated the cause:

- **Target/padding/masking mechanics were verified correct** (real content, not padding, is
  what gets trained on and measured) and **individual corpora were verified clean** (TinyStories:
  0% empty responses across 13.6M constructed pairs; `smoltalk_preference`: 92%+ acceptance under
  the same quality filters when context-window rejection is accounted for) — ruling out both a
  data-pipeline bug and corpus-mixture-quality as explanations.
- **Swapping the full 5-way corpus mixture for the single curated 50k-row corpus validated in
  section 7 did not meaningfully change the outcome** at 100M scale (content PPL still in the
  several-hundreds range) — ruling out corpus heterogeneity/mixture as the primary driver.
- **Reducing vocab size (32,768→1,024) and context/response window (512/128→64/32) to match
  section 7's validated small-scale regime produced the first sign of real content formation**
  at 100M scale (`coverage` and `content_top5` became nonzero, `ws_omission` dropped below 1.0)
  — but final PPL (~382-751 depending on run) was still nowhere close to the ~17-21 PPL the
  50,325-param Gator reaches in section 7.3, so vocab/context size alone does not explain the
  gap either.
- **The dominant lever turned out to be `--batch-size`.** Section 7's validated 50,325-param
  result used `--batch-size 16`; every 100M-scale attempt (production run and all sanity checks
  above) used `--batch-size 1`, matching the real production spec (a 100M Gator does not fit
  larger batches at full 512-token context/32,768 vocab on a single GPU without the packed
  execution backend, which trades ~40% memory savings for a training-time backward-pass
  slowdown). A reduced-vocab/context 100M Gator rerun with `--batch-size 16` (LR unchanged at
  3e-4) reached `content_acc=0.038` with an exact correct-token prediction (`formed='a'`
  matching `target='a'`) by step 7,000, and PPL 72.69 by step 10,000-11,000 — a result no
  batch-1 100M run (Gator or transformer control, matched params/vocab/context) came anywhere
  near. A parallel real-world run also found that manually dropping the learning rate
  (3e-4→1e-4→3e-5) measurably improved early PPL at batch=4, consistent with batch size and
  learning rate acting on the same underlying lever (gradient-noise magnitude relative to the
  loss landscape) rather than being independent fixes.
- A matched-params (~99M) transformer control, batch=1, same reduced vocab/context, same
  curated corpus, finished at PPL 265.36/content PPL 443.40 — comparable to Gator's batch=1
  result (382.38/463.76), no large architecture gap at batch=1. This suggests the batch-size
  sensitivity observed here may be specific to Gator (or at least much more severe for Gator)
  rather than a universal small-batch problem — worth testing the transformer control at
  larger batch too before treating this as Gator-specific.

**Working conclusion:** Gator's node/routing decisions appear far more sensitive to per-example
gradient noise than a transformer's smoother attention mechanism, becoming unusable at
`batch-size=1` at 100M-parameter width even though the same architecture (scaled down) trains
fine at batch=1 in sections 6-7. This is a new, previously-unmeasured cost of the routing
mechanism: Gator's node/routing state also makes it substantially more memory-hungry per batch
element than a transformer at matched param count, so the batch sizes needed to train it stably
at width are harder to afford. Unresolved and worth follow-up: does the packed execution
backend (cheaper memory, slower backward pass) let a larger batch fit profitably despite the
per-step slowdown; does the transformer control show similar sensitivity at large batch (i.e.
is this really Gator-specific or just under-tested for transformers too); is there a minimum
batch size below which Gator specifically becomes unstable, and does that threshold scale with
node count/width the same way the 1.7 node/width ratio does.

## Section 9: 10M-scale dedicated batch-size study — confirms the threshold, rules out several alternative explanations

Section 8 found that batch size (not vocab, context, or corpus mixture) was the dominant lever
at 100M scale, but only tested a handful of configs under real production time pressure. This
section builds a purpose-sized ~10M-param intermediate model (`d_model=156/nodes=265`, ratio
1.699, `embedding_dim=122/router_rank=41/active_node_top_k=14`, same 1.7 node/width ratio,
full production `context_tokens=512/response_tokens=128`, `vocab_size=20,000`) specifically to
make this kind of sweep affordable — 10-30 steps/sec instead of 100M's ~1-4 steps/sec — and
runs a systematic effective-batch sweep plus several alternative-hypothesis ablations, all at
8,000 real optimizer updates for comparability, using the curated 50k-row corpus, `serial`
decode, and `terminal_vote_mode=prompt` (justified below).

**Two new pieces of infrastructure landed as part of this section, both now real CLI flags:**
- `--mixed-precision bf16`: wraps the training/eval forward pass in
  `torch.autocast(dtype=torch.bfloat16)`. Profiling at full production scale found the
  actual bottleneck is the receiver-kernel matmul in `_node_memory`/packed-memory node context
  (`O(n_nodes × context_length² × d_model)`, ~65% of total step time in `aten::bmm` + its
  backward) — not vocab size, and **identical between `route_backend=dense` and
  `route_backend=packed`**, which is why packed execution showed no speed advantage in section
  8 (it doesn't touch the actual bottleneck). `--mixed-precision bf16` gave a measured
  ~1.5-1.6x real speedup on this exact workload (2.5-2.7 → 4.32 steps/sec at batch=6,
  vocab=32,768, context=512), no numerical issues, no loss scaling needed (bf16 keeps
  float32's exponent range).
- `--gradient-accumulation-steps N`: backpropagate N micro-batches (each `--batch-size`) before
  one clipped optimizer step, so `--steps` counts micro-batches, not updates
  (updates = `--steps / N`). Validated correct against a direct same-effective-batch control
  (batch=1×accum=4 vs. batch=4 direct: nearly identical trajectories, `loss=4.11/ppl=60.9` vs.
  `loss=4.03/ppl=56.2` at the same 1,000 real updates). Accumulation trades **speed for memory
  only** — it does not parallelize, so it is strictly slower in real-updates/second than a
  true larger physical batch at the same effective size; it exists purely to reach an effective
  batch that would otherwise not fit.

**Architecture correction made mid-section: `terminal_vote_mode=slot` doesn't do what it's for
under `serial` decode.** `serial_next_token_batch()` always builds exactly one target slot
(`training_output_slots == 1`), so slot mode's per-position fixed-address jury — designed to
stop many *simultaneously-formed* output positions from collapsing onto an identical decision —
has no simultaneity to differentiate under serial decode; only the corpus position varies
step-to-step, not multiple positions within one step. A head-to-head at full 100M scale
(effective batch 18, `slot` vs `prompt`, otherwise identical) confirmed this: `prompt` was
equal-or-better at every matched real-update count and never showed the large loss spikes
`slot` did (771 and 1,090 at steps where `prompt` stayed under 320). All 10M-scale tests in
this section use `terminal_vote_mode=prompt` accordingly.

**Effective-batch sweep (physical batch × accumulation, 8,000 real updates each):**

| Effective batch | Recipe | Final content PPL | Behavior |
|---|---|---:|---|
| 1 | batch=1 | 630.95 | Very noisy; one spike to 26,031 |
| 4 | batch=4 | 330.11 | Plateaus ~300-330 |
| 8 | batch=8 | 260.91 | Plateaus ~240-260 |
| 16 | batch=8, accum=2 | 180.94 | **Still descending at step 8,000** — never plateaued |
| 32 | batch=4, accum=8 | 192.21 | Also still-descending, but **ties 16, does not beat it** |

Effective batch 16 is the first (and, per the 16-vs-32 tie, apparently the last) point where
training escapes the plateau-and-flatten pattern within this step budget — 4 and 8 both clearly
plateau, 16 breaks free, and 32 costs 2x the compute for a statistically indistinguishable
result. **Effective batch ~16 looks like the practical ceiling, not a floor on an
open-ended "more is better" curve.** This matches the qualitative pattern already established
for node/width ratio (peaks, doesn't scale monotonically) rather than the naive
bigger-is-strictly-better assumption.

Caveat carried over from section 8, still unresolved: this whole sweep only covers 8,000 real
updates. The real 100M production run plateaus around content PPL 140-180 even at 50,000 real
steps — 6x further than tested here. So "effective batch 16 breaks the plateau" is confirmed
*within this window*, but whether it keeps descending past the 140-180 band at a longer horizon
(matching the production experience) or eventually flattens there too remains untested.

**Three alternative-hypothesis ablations, all negative or only marginally positive results —
batch size continues to look like the real, not-yet-circumvented lever:**

1. **Cross-lane gradient conflict (PCGrad/agreement-weighting motivation).** Proposed
   hypothesis: Gator's parallel lanes might be pulling shared parameters in conflicting
   directions, and batch size only works by statistically averaging out that conflict rather
   than fixing anything. Checked feasibility first: lanes are **not** cleanly separable the way
   PCGrad assumes (`_node_memory`'s writer-mass computation sums across the lane axis at every
   relay, before any lane reads the shared memory back — lanes mix at every hop, not just at
   final fusion, so there's no clean way to isolate "lane r's independent gradient" without first
   untangling coupling baked into the forward pass). A cheaper, directly testable proxy — reduce
   to `lanes=1` (removing all cross-lane interaction entirely) at batch=4 — was **not** more
   stable than `lanes=10` at the same batch (final PPL 343.00 vs. 330.11, comparably noisy, if
   anything a mid-run spike to 805 that lanes=10 didn't show). **Null result: argues against
   cross-lane conflict as the dominant driver.**
2. **Per-component learning rate damping** (`--optimizer-groups component_scaled`,
   `route_gate` multiplier 0.1x, `lane_core` 0.3x, testing whether the higher-variance
   routing/gating decisions — not the whole model — are the actual noise source). Real but
   modest improvement, stacking at any batch size tested: batch=4 improved 330.11→288.60 (~13%),
   batch=8 improved 260.91→252.28 (~3%). **Does not replace needing the bigger effective
   batch** — neither combination came close to effective-batch-16's still-descending
   130-180 trajectory. Worth keeping as a free stacked win, not a substitute.
3. **Two-phase route/jury freeze**, testing whether the original staged Phase A/B/C curriculum's
   known stability benefit (route-freezing) transfers to `serial` decode. Phase 1 (routes +
   terminal jury co-adapting, 5,000 real updates): ended PPL 362.41, noisy. Phase 2 (same run,
   `route_gate` component frozen via `--optimizer-train-components` excluding it — note this
   freezes the terminal-jury parameters too, since `terminal_vote`/`terminal_address` markers are
   grouped into the same `route_gate` optimizer bucket as router/ingress in this codebase — 3,000
   more real updates): ended PPL 292.57. Better than continuing to co-adapt, but still **worse
   than the plain unfrozen batch=8 baseline (260.91)** at the same total step budget. **Route/jury
   freezing after a warmup is not the missing piece either** — it's a mild improvement over doing
   nothing, but plain unfrozen training with balanced per-component LR still wins.

**Section 9 conclusion:** three independent alternative explanations for the small-batch
instability (cross-lane conflict, routing-specific gradient noise needing only LR damping,
insufficient route/jury adaptation time) were tested and each gave a null or only marginal
result. Batch size remains the dominant, uncircumvented lever, with effective batch ≈16 as the
apparent practical target — beyond which more batch costs compute without buying quality at this
step horizon. Two free, stackable, real infrastructure wins came out of the investigation
regardless: `--mixed-precision bf16` (~1.5-1.6x speedup, no cost) and per-component LR damping
(~3-13% quality improvement at a given batch, no added compute cost).

**Open, not yet tested — untested architectural ratios beyond node/width:** every scaled-down
test config in this whole study (including all of section 9) has set `embedding_dim` by
preserving the production config's ratio to `d_model` (288/368 ≈ 0.7826×) without ever testing
whether that ratio itself is good, too wide, or too narrow — it was inherited, not validated,
the same way the node/width ratio was before section 6.2 actually tested it. Similarly, the
graph-connectivity parameters (`--graph-degree`, `--connection-rewire`, `--relays`) have never
been swept independently of node/width — there may be a "golden ratio" (or a stable-basin
range) for connectivity the same way there is for node/width, and getting it wrong (too sparse
or too dense) could plausibly explain some of the residual instability at larger scale that
batch size alone hasn't fully resolved. This is the natural next phase: a dedicated config
scaling study, using a small vocab and `--batch-size 4` (deliberately below the effective-16
threshold, to keep each config's baseline behavior visible and iteration cheap) to sweep
`embedding_dim/d_model` and connectivity parameters independently, holding the validated 1.7
node/width ratio fixed.

### Section 9 addendum: embedding_dim is a function of vocab_size, not d_model — corrected finding

The first pass of the config scaling study swept `embedding_dim` at four ratios relative to
`d_model` while holding node/width fixed at the validated 1.699 (`d_model=156, nodes=265`),
using `vocab=2048`, `batch_size=4`, `gradient_accumulation_steps=1`, `route_backend=dense`,
`terminal_vote_mode=prompt`, `decode_mode=serial`, `mixed_precision=bf16`, monolithic
optimizer, 8,000 steps:

| embedding_dim/d_model ratio | embedding_dim | final PPL (vocab=2048) |
| --- | --- | --- |
| 0.25× | 39 | 95.88 (91.12 on rerun) |
| 0.5× | 78 | 101.16 |
| 0.7826× (production ratio) | 122 | 92.83 (83.13 on rerun) |
| 1.0× | 156 | 93.34 |

At this small vocab the range is a shallow ~9-13 point band with no sharp optimum — this was
initially (and incorrectly) written up as "embedding_dim is not a sensitive lever." **That
conclusion was an artifact of testing at vocab=2048.** The correct framing, raised directly:
embedding_dim's job is to give each vocabulary entry room to be distinguishable before
projection into `d_model` — so it should be sized against `vocab_size`, not `d_model`. Holding
embedding_dim fixed and raising vocab_size exposes this cleanly: at vocab=20,000 (same
`d_model=156, nodes=265`, otherwise identical harness), sweeping `embedding_dim` produces a
clear, monotonic, non-flat result:

| embedding_dim | total model parameters | final PPL (vocab=20,000) |
| --- | --- | --- |
| 39 | 8.04M | 217.07 |
| 76 | 8.89M | 196.48 |
| 122 | 9.96M | 174.18 |
| 200 | 11.76M | 154.37 |
| 800 | 25.63M | 150.21 |
| 1200 | 34.88M | 148.09 |

The 39→200 range shows a real, substantial improvement (217→154, ~29% relative) for a modest
parameter cost (8.04M→11.76M, +46%). Past 200 the curve visibly knees over: 800 and 1200
(4-6x wider, +118-198% params over 200) only buy another 4-6 points combined (154.37→150.21→
148.09) — a still-improving but rapidly-diminishing tail, not a reversal. No embedding_dim
tested here was ever too wide to help; the practical ceiling is compute/parameter budget, not
quality. **Conclusion: embedding_dim must
scale with `vocab_size`, not `d_model`** — the earlier "flat" finding held only because
vocab=2048 is small enough that even a narrow embedding has room to separate every token.
Practical guidance until a dedicated vocab-sweep pins down the actual functional form: size
embedding_dim relative to vocab_size (roughly `embedding_dim ≈ 0.01×vocab_size` fits the knee
observed here, i.e. ~200 at vocab=20,000), and treat "ratio to `d_model`" as the wrong
mental model entirely. Unlike `active_node_top_k` (which trades quality for real per-step
compute/memory, see below), embedding_dim's cost is a one-time linear parameter cost
(`vocab_size × embedding_dim` for the table, plus `2 × lanes × d_model × embedding_dim` for
the two projections) with no extra compute-per-step penalty until it grows large enough to
dominate total model size, which is what happens past the ~200-800 range at this small model
scale. A follow-up sweep (fixed `embedding_dim=39`, vocab rising 2,048→50,000) is queued to
directly measure the point where a fixed embedding_dim starts costing quality, which will
pin down the actual `embedding_dim(vocab_size)` relationship more precisely than crossing
both variables.

**Emerging Gator sizing procedure**, given the ratios validated so far: (1) pick a total
parameter budget and a `vocab_size`; (2) size `embedding_dim` against `vocab_size` (not
`d_model`) using the knee-point relationship above, and compute its exact parameter cost
(embedding table + both projection matrices); (3) subtract that from the total budget to get
the remaining "core" budget for the graph/routing machinery; (4) allocate the core budget
using the validated ratios — node/width ≈1.7, `graph_degree` at the baseline (≈4, tested wider
and narrower both worse-or-invalid), `relays=2` (fixed, tested and settled), `lanes` as small
as tolerable (a free lever — 5 ties 10 on quality at real speed/memory savings), and
`active_node_top_k` sized to whatever compute budget is actually available (a real
quality/compute tradeoff, not free — see below). Embedding_dim is a vocab-driven cost paid
first, not one of the ratios applied to the rest of the model.

### Section 9 addendum: lanes / active_node_top_k / graph_degree sweep

With node/width fixed at 1.699 and embedding_dim held at the original 0.7826× ratio (122, at
vocab=2048, same 10M harness as above), each of `lanes`, `active_node_top_k`, and
`graph_degree` was swept independently around the harness baseline (`lanes=10, degree=4,
top_k=14`), 8,000 steps each:

| axis | value | final PPL | steps/sec |
| --- | --- | --- | --- |
| lanes | 5 | 96.39 | 22.46 |
| lanes | 10 (baseline) | 92.83 / 96.47 (repeat) | 15.8–16.9 |
| lanes | 20 | 98.44 | 10.26 |
| active_node_top_k | 7 | 101.81 | 21.5 |
| active_node_top_k | 14 (baseline) | 92.83 | 16.9 |
| active_node_top_k | 28 | 86.98 (best) | 12.0 |
| graph_degree | 2 | invalid — `small_world` layout requires degree ≥ 3 | — |
| graph_degree | 4 (baseline) | 92.83 | 16.9 |
| graph_degree | 8 | 101.01 | ~15–16 |

**lanes is flat-to-slightly-worse as it increases** — 5 and 10 tie within noise, 20 is a clear
loser on both quality and speed (36% slower). This is a genuinely free lever: shrinking lanes
from 10 toward 5 costs nothing on quality and buys real speed/memory. **active_node_top_k
shows a real, monotonic, compute-costly tradeoff** — wider consistently helps (7→14→28 keeps
improving) but at real per-step cost (top_k=28 ran ~29% slower than baseline). Because higher
top_k also raises peak activation memory, at larger model scale it can force a smaller
physical batch (see below), which compounds the slowdown through gradient-accumulation
overhead on top of top_k's own cost — so top_k's apparent "win" needs to be judged against the
compute budget actually available, not in isolation. **graph_degree looks close to optimal at
the baseline (4)** — going wider (8) modestly hurt, and going narrower (2) isn't even a valid
configuration under the `small_world` connection layout, which requires degree ≥ 3.

A direct attempt to combine the "best" ratios (node/width 1.697, embedding_dim ratio 0.25×,
active_node_top_k ratio 0.105 matching the topk28 winner) into a fresh ~50M-parameter config
(`d_model=302, nodes=513, embedding_dim=76, active_node_top_k=54, lanes=8, degree=4,
vocab=20,000, context=512`) underperformed a simpler baseline config at the same scale
(`d_model=296, nodes=500, embedding_dim=148, active_node_top_k=8, graph_degree=16, lanes=8` —
PPL ~51–56 by step ~38,500 real-update-equivalent) by a wide margin — the combined-ratio
config was still at PPL ~137–150 at a matched step count before its session died. Retesting
the same combined config with `active_node_top_k` dialed back to 14 (isolating top_k as the
suspect) tracked nearly identically to the top_k=28 variant (~168 vs ~150 at matched steps),
both clearly worse than the simpler baseline — indicating **the individual-axis "winners"
from small-scale, single-variable sweeps do not compose cleanly**; there is likely
synergistic/parasitic interaction between axes (embedding_dim, node/width, degree, top_k)
that single-variable sweeps cannot see, and any combined config needs to be validated as a
whole rather than assembled from independently-optimal parts. Also worth noting: the
combined-ratio config's higher `active_node_top_k` (54) forced an OOM at the original
batch=4/accum=4 split at production-like context (512) — the same 15.96 GiB allocation
failure recurred even after halving batch size to 2, indicating the OOM was driven by
`active_node_top_k` itself (which scales per-step node-update tensor size), not primarily by
batch size, and was only resolved by lowering top_k to 28.

(Relay depth is already settled at 2: even relay counts work, odd counts are bad, and returns
diminish past 2, so relays was held fixed and excluded from this sweep.)

### Section 9 addendum: `--compile` is required at 100M scale — not just a speed optimization

Applying the new sizing formula (embedding_dim from vocab_size, golden ratios for the rest) at
a real ~100M-parameter target (`d_model=379, nodes=643, embedding_dim=200, active_node_top_k=32,
lanes=5, graph_degree=4, relays=2, vocab=20,000, context=512, response=128`,
`route_backend=dense`) reliably **OOM'd in eager mode** — `torch.OutOfMemoryError` inside
`_node_memory`'s `torch.matmul` (`gator_convergent.py`), trying to allocate ~14.94 GiB with
~86-90 GiB already resident on a 94.97 GiB GPU. This reproduced identically across
`active_node_top_k=32` and `=14` and across `batch_size=1..4` (same allocation size every
time) — confirming the failure is **not driven by `active_node_top_k` or batch size**, but by
the dense backend's node-update matmul, which touches the full node set
(`nodes × context² × d_model`-scale) regardless of how many nodes are "active." Adding
`--compile` (`torch.compile`/inductor) resolved the OOM entirely at the exact same
architecture and batch size — kernel fusion evidently avoids materializing the large
intermediate tensor that eager mode allocates in one shot. **Practical implication:
`--compile` is not optional at this scale; without it, this class of config cannot run at
all**, independent of the speed benefit `--compile` also provides.

Separately, re-confirmed at 100M scale (previously only validated at smaller scale):
`--mixed-precision bf16` gives both a real speed win (fp32/compile: ~3.1→3.8 steps/sec and
still ramping; bf16/compile: ~6.8→7.3 steps/sec — roughly 1.8-2x here) **and** a real memory
win (peak GPU memory dropped from ~75 GiB to ~45 GiB on the same architecture/batch size,
per a parallel local run at the same config). This makes bf16 a second (independent)
mitigation for the same class of OOM `--compile` fixes, not merely a throughput optimization
— worth enabling by default at this scale rather than treating it as an optional speedup.

## Training optimizations

### Batch size vs. gradient accumulation — a real but model/GPU-dependent throughput effect, not a universal rule

Across three different model scales, shrinking the physical batch size while raising
`--gradient-accumulation-steps` to hold (or exceed) the same effective batch produced wildly
different throughput outcomes — the effect is real where it appears, but it is not a general
"always prefer small batch + accumulation" rule. It depends on both model width and GPU.

First, a methodology note: the naive read of raw `steps_per_second` is a lifetime running
average from step 0, so it climbs for the first few hundred steps purely as one-time
`torch.compile` overhead gets diluted (observed directly on a 50M/G4 run: 2.07→7.06 steps/sec
over the first 300 steps with no config change at all) — a single early reading is not
trustworthy on its own. All figures below are steady-state, confirmed to hold over thousands
of steps (in the 50M case, tens of thousands), ruling out warmup dilution as an explanation for
any of them.

| scale | GPU | batch=4, accum=4 | batch=4, accum=8 (or next step tested) | result |
| --- | --- | --- | --- | --- |
| 130K (`d_model=30, nodes=51`) | L4 | — (tested batch=32/accum=1: 2.73 steps/sec) | 7.63 steps/sec | raw steps/sec favors small batch, but at matched effective batch (32) the large-batch config actually completes one optimizer update faster (0.366s vs 1.048s) — **big batch wins here** |
| 50M (`d_model=300, nodes=510`) | G4 | 8 steps/sec | 15 steps/sec | **small batch + more accumulation nearly doubles throughput** |
| 100M (`d_model=379, nodes=643`) | Pro 6000 | 9.3 steps/sec | 9.6 steps/sec | negligible change either way |

The 50M numbers were also checked at intermediate accumulation values (`batch=8/accum=2`: 4
steps/sec; `batch=4/accum=4`: 12 steps/sec; `batch=4/accum=6`: 15 steps/sec, where it
saturates — accum past 8 gave no further gain), and converting to real throughput
(`examples/sec = batch_size × steps_per_second`) confirms the win is real data processed per
second (32→48→60), not just more (cheaper) micro-batches counted per second.

**The effect is not monotonic with scale, and reverses direction between the smallest and
largest models tested**: at 130K the large batch wins, at 50M the small batch wins by nearly
2x, and at 100M the difference nearly vanishes. Two different GPUs are also involved (L4/G4 for
the smaller scales, a Pro 6000 workstation card for the 100M), which is itself a live
confound — the 100M's near-null result could reflect the model being wide enough to have
already left whatever cache/bandwidth-sensitive regime the 50M sits in, **or** it could reflect
the Pro 6000's different cache/bandwidth profile compared to the L4/G4 cloud instances, and
these two explanations have not yet been disentangled (that would require running the same
50M config on the Pro 6000, or the 100M config on an L4/G4, neither of which has been done).

**Leading mechanism (best current guess, not confirmed):** the dense `route_backend`'s
node-update matmul scales with `nodes × context² × d_model`, and batch size multiplies that
working set directly. A larger physical batch can push the per-step working set past what fits
efficiently in a given GPU's cache hierarchy, making large-batch steps genuinely
memory-bandwidth-bound rather than compute-bound, while a smaller physical batch stays in a
cheaper, cache-friendly regime — but where that crossover point sits depends on both the
model's per-example working set (width/context) and the specific GPU's cache size and
bandwidth, which is exactly why the same lever helps enormously at one scale, does nothing at
another, and reverses at a third.

**Practical guidance: benchmark this per model/GPU combination before committing to a
setting — do not assume either direction transfers.** This is distinct from (and additional
to) the previously-documented effective-batch≈16 **stability/quality** threshold earlier in
Section 9; this finding is purely about raw throughput at a given effective batch, and no
quality difference has been observed between physical-batch choices at matched effective
batch in any of the three scales tested.

**Practical guidance:** prefer the smallest physical batch size that keeps per-step overhead
reasonable, and reach whatever effective batch you need via `--gradient-accumulation-steps`
rather than raising `--batch-size` directly. Throughput gains from more accumulation saturate
once the physical batch is already small enough to be cache-friendly (observed at accum=8 for
the 50M config — accum=6→8 kept improving, but pushing further gave no additional speed).

## Closing summary of the capability-validation phase (section 7)

Parallel-decode instability is a property of the training objective (independent,
simultaneous per-slot prediction under constant LR), not of Gator's architecture or its
parameter count — `serial` decode eliminates it entirely at the scales tested here. Once
that confound is removed, Gator does show a real capability edge over a matched-size plain
transformer, but only when Gator's own node-to-width ratio (≈1.7) is respected — the same
config at the wrong ratio can look worse than a much smaller Gator. A cheap architectural
change (2 reasoning loops: draft, feed back, refine) closes almost all of the transformer's
gap to Gator, but the identical mechanism catastrophically destabilizes Gator itself,
regardless of ratio — a genuine, asymmetric incompatibility, not a bug in the comparison.
Finally, even at the correct ratio, Gator's quality does not scale monotonically with size in
this setup — it peaks at an intermediate size and degrades at both ends of the range tested.
Node/width ratio, not raw parameter count, is the dominant lever for Gator's scaling behavior

## Section 10: a candidate mechanism for the core PPL/quality decoupling — bidirectional teacher-forcing leakage (2026-07-27, hypothesis, not yet tested)

Everything above documents *that* PPL and real generation quality decouple under constant
LR; this section proposes a specific, testable *why*, surfaced during unrelated work on the
100M convergent production run (see `Runbooks/CONVERGENT_RUNBOOK.md`'s causal-execution
section for the full context).

**Background:** convergent Gator's receiver-memory kernel defaults to non-causal
(bidirectional) attention over the full observed+output window — every position, including
teacher-forced output/response slots, can attend to every other position, past or future,
unless `causal=True` is explicitly requested. The production run had trained non-causally for
1.3M steps. A resume was attempted with causal execution forced on
(`terminal_vote_mode=global` + `--force-causal`); PPL immediately jumped from ~32 to ~220 and
the run needed to substantially relearn. However, free-generation samples produced partway
through that relearning subjectively looked *better* than the pre-switch non-causal
checkpoint's output, despite the much worse PPL.

**Hypothesis:** non-causal training during teacher forcing doesn't just give the model *more
context* — it may let response/output positions partially observe the ground-truth target
tokens they are being scored against, via bidirectional attention back from later output
slots to earlier ones within the same forced sequence. This would function as a training-time
shortcut ("leakage"): the model can partially infer what it's supposed to predict from
information that would never be available during true autoregressive generation, driving PPL
down without building the genuinely autoregressive skill that real free generation requires.
Switching to causal execution removes this shortcut — PPL gets *worse* because the easy
signal is gone, but real generative quality can improve because the model is now optimizing
the task it will actually be evaluated on at inference time.

**Why this would explain the study's central finding:** if constant-LR training gradually
learns to exploit more of this bidirectional leakage over the course of a run — a plausible
failure mode, since there is no explicit pressure against it — that alone would produce
exactly the decoupling this whole study has repeatedly observed: PPL/token-accuracy improving
while real content quality (content_ppl, content_acc, actual free-generation coherence)
degrades. It would also explain why `content_acc` has repeatedly been the metric that
diverges from raw PPL/accuracy across this entire investigation (Sections 1, 4, 6): content
accuracy is scored against genuinely generated tokens, which leakage cannot help with, while
PPL is computed under the same teacher-forced, leakage-prone conditions that produced it.

**Not yet tested.** This is a hypothesis with circumstantial support (one training-time
regression, one inference-only A/B showing comparable-or-better quality under causal
execution — see the runbook), not a confirmed mechanism. The direct test: compare
`content_ppl`/`content_acc` (not raw PPL) between matched-step causal and non-causal runs,
and separately, directly measure whether non-causal teacher-forced loss is suspiciously lower
than what causal execution can achieve at the same step count, which would be the leakage
signature. If confirmed, this would be a strong candidate for the single root cause this
entire study has been chasing, and would argue for causal execution being adopted for
training generally, not just as an inference-time convenience — in tension with the
training-time regression observed so far, which needs to be understood (e.g. as a
recalibration cost rather than evidence against causal training) before that recommendation
could be made with confidence.
observed in this study.
