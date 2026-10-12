# GTR-L EvoToken tail-position ablation sweep (nodes=40, width=128, output-slots=6)

Status: **experimental**, early-training leading indicators only (50k of 10M planned steps per
arm, single seed, no repeated trials).

Related: `Papers/GATOR_LINEAR_EVOTOKEN.md` (method), `Papers/GATOR_EVOTOKEN_SLOT_WIDTH_SWEEP_NODES80.md`
(the capacity-scaling sweep this work pivots away from), `Runbooks/GTRL_EVOTOKEN_COLAB_MODULE.md`
(the ablation module and full flag reference).

## Motivation

The nodes=80 capacity sweep (see the nodes80 report) was stopped early because results across
slot counts were inconsistent and uninformative. But that sweep's own raw logs surfaced a much
more specific, reproducible pattern worth chasing directly: for output-slots=6, the front
(logical position 0) output climbs to ~0.35-0.40 accuracy over training while positions 4-5 (the
tail) stay flat near 0.06-0.09, regardless of which capacity axis (width or nodes) was scaled up.

Two competing hypotheses were proposed:

1. **Data/distance-limited**: later output positions are inherently harder to predict — useful
   signal decays with distance from context — and this is not fixable by scheduling or loss
   changes, only (maybe) by more capacity or better architecture.
2. **Promotion-schedule / credit-assignment starvation**: the relay's promotion schedule
   structurally gives tail positions fewer effective refinement passes, or training/inference
   promotion order and loss weighting under-serve them — a fixable scheduling artifact.

A purpose-built module (`gtrl_evotoken_colab.py`) was written to test between these directly via
six controlled interventions plus a fine-tuning probe, all starting from the same
nodes=40/width=128/output-slots=6 baseline that the earlier capacity sweeps already established
as the best config on record.

## Method

Same core config as the nodes=40/width=128 reference point (`GatorLinearConfig(vocab_size=VOCAB,
streams=10, width=128, nodes=40, relays=4, degree=6, buses=4, mailbox_channels=4,
mailbox_width=8, basis_rank=4, address_width=4, router_rank=8, conv_kernel=9, hard_routing=True,
linear_attention=True, controlled_writing=True, causal=False, block_causal=False)`), masked-only
CTS v5 objective, batch 16, lr 1e-3, output_slots=6, promotions=(2,1,2,1). `--resume-source none`
on every arm to prevent checkpoint-tag collisions. `--position-gradient-every-evals 0` (the
gradient-norm diagnostic was disabled after it was found to cause crashes; the by-position
accuracy and per-position EMA diagnostics used throughout this report do not depend on it).
Trained live on Colab (single L4), 2026-08-02 to 2026-08-03. Metrics pulled from the log line
closest to step 50,000 for arms 44-49 (all from-scratch runs); arm 50 is a fine-tune probe judged
by plateau, not a step count (see below).

Seven arms:

| Arm | Task | Flag(s) added to baseline | Tests |
|---|---|---|---|
| 44 | baseline | — | reference point |
| 45 | slot-permutation | `--slot-permutation-mode rotate` | physical-slot vs logical-position specialization |
| 46 | training promotion order | `--promotion-schedule-mode rotating` | random vs forced-order training promotion |
| 47 | inference promotion order | `--inference-promotion-mode rotating` | confidence-based vs forced-order inference promotion |
| 48 | balanced loss | `--position-balance-gamma 0.5` | reweighting loss toward weak positions |
| 49 | worst-position loss | `--worst-position-weight 0.25` | smooth-max objective term on the worst position |
| 50 | tail-only probe | `--tail-loss-slots 2`, fine-tuned from arm 44's checkpoint | removes ALL front-position competition for signal |

## Results (arms 44-49, closest-to-50k-step datapoint)

| Arm | refined_acc | Δ vs baseline | modal_lift | Δ vs baseline | by_pos (front→tail) | tail (pos 4, 5) |
|---|---:|---:|---:|---:|---|---|
| 44 baseline | 0.1836 | — | +0.133 | — | [0.39 0.27 0.17 0.13 0.06 0.08] | 0.06 / 0.08 |
| 45 permutation | 0.1576 | **-14.2%** | +0.107 | **-19.5%** | [0.33 0.16 0.14 0.13 0.08 0.09] | 0.08 / 0.09 |
| 46 train promotion | 0.1699 | **-7.5%** | +0.119 | **-10.5%** | [0.34 0.25 0.16 0.12 0.07 0.08] | 0.07 / 0.08 |
| 47 infer promotion | 0.1654 | **-9.9%** | +0.115 | **-13.5%** | [0.35 0.23 0.14 0.12 0.07 0.08] | 0.07 / 0.08 |
| 48 balanced loss | 0.1725 | **-6.0%** | +0.122 | **-8.3%** | [0.36 0.21 0.16 0.13 0.08 0.09] | 0.08 / 0.09 |
| 49 worst-position | 0.1745 | **-5.0%** | +0.124 | **-6.8%** | [0.36 0.23 0.17 0.12 0.08 0.08] | 0.08 / 0.08 |

**Every intervention tested made overall accuracy worse. None beat baseline.** Ranked by
regression size, smallest to largest: 49 < 48 < 46 < 47 < 45. The two loss-reweighting arms (48,
49) were the mildest regressions and modestly raised the tail (0.08-0.09 vs baseline's 0.06-0.08)
without severely damaging the front — the closest either any arm came to "raising the tail rather
than dragging the front down." The scheduling arms (45, 46, 47) were worse, and permutation (45)
was the worst of all, cutting front-position accuracy roughly in half (0.39→0.33→0.16 at position
1) while only modestly moving the tail.

**Arm 45's `ema_physical` vs `ema_logical` split is the key diagnostic result.** With physical
slots permuted every step, `ema_physical` came out nearly flat and uniform across all 6 physical
slots (0.152-0.174, no front/tail split) — confirming permutation successfully broke any
fixed-physical-slot specialization. But `ema_logical` still showed the same strong front-to-tail
decline as every other arm (0.323 → 0.090). **The weakness travels with the logical future-offset,
not with a hardware/execution-path location.** This rules out "some physical slots just get better
because of a hard-wired execution quirk" and points at the objective/promotion structure or the
underlying task difficulty as the real driver.

**A consistent "drags everything down" pattern across every scheduling/reweighting intervention.**
Every arm that narrowed the front/tail gap did so mostly by making the front worse, not by making
the tail meaningfully better — a net-negative reallocation of a shared, apparently fixed pool of
learning signal, not a genuine fix. This matches the user's own independent finding from a
separate worst-position-loss test outside this sweep ("drags everything down"), though this
sweep's own matched-config arm 49 showed a considerably milder version of that collapse
(front position 0 only dropped 0.39→0.36, not a dramatic crash) — the exact magnitude appears
sensitive to the precise config/weight used, though the direction is consistent.

## Arm 50: the decisive test

Arm 50 removes the ambiguity above entirely by eliminating front-position competition altogether
(`--tail-loss-slots 2` zeroes the loss weight for positions 0-3, `weights_logical=[0,0,0,0,1,1]`),
fine-tuned from a baseline checkpoint with a fresh optimizer and step counter reset to 0. If the
tail's weakness were caused by competition for a fixed signal budget, giving it the *entire*
budget with zero competition should show clear improvement. It did not.

| Step (own counter) | ema_logical[4] | ema_logical[5] |
|---:|---:|---:|
| 300 (warm-start) | 0.102 | 0.084 |
| 12,000 | 0.108 | 0.088 |
| 24,300 | 0.109 | 0.092 |
| 24,600 | 0.110 | 0.098 |
| 24,900 | 0.107 | 0.082 |
| 37,200 | 0.102 | 0.084 |
| 37,500 | 0.119 | 0.096 |
| 37,800 (stopped) | 0.105 | 0.088 |

Both tail positions plateaued almost immediately (by step ~12k) in a tight, noisy band —
0.10-0.12 for position 4, 0.08-0.10 for position 5 — and **37,800 steps of exclusive, uncontested
training signal did not move them meaningfully beyond where baseline's own shared/competitive
training already had them** (baseline's final ema_logical for positions 4,5 was 0.110/0.096 —
statistically indistinguishable from arm 50's plateau). Meanwhile the front position, no longer
trained at all, visibly decayed (ema_logical[0]: 0.35 → 0.23), confirming the fine-tune was
actually reallocating signal as intended and the tail's flatness isn't an artifact of the probe
not working.

**Caveat on arm 50's starting point:** the warm-start loaded from `source_step=75,432`, not the
documented arm 44 datapoint at step=50,100 — arm 44's checkpoint on Drive evidently continued
being updated past the step this report's baseline row uses, at some point before the session that
ran arms 47-50 picked back up. This was verified as a legitimate, architecture-matched checkpoint
(not corruption), just a later point in baseline training than intended. Low risk to the
conclusion below — 75k vs 50k steps are both early relative to the 10M-step budget, and the
plateau observed is far too flat and immediate to be explained by this discrepancy.

## Conclusion

The evidence across all seven arms favors **hypothesis 1 (data/distance-limited)** over
hypothesis 2 (fixable credit-assignment/scheduling artifact):

- Breaking physical-slot specialization (arm 45) didn't break the logical front/tail pattern —
  it travels with logical position, not hardware placement.
- Every scheduling change (arms 46, 47) and loss-reweighting change (arms 48, 49) that tried to
  redistribute the model's effort toward the tail did so at a net accuracy cost, mostly by
  degrading the front rather than genuinely lifting the tail.
- The most direct test (arm 50) — remove all competition, give the tail the entire signal budget
  for 37k+ steps — produced no improvement beyond what baseline's normal shared training already
  achieved.

This does not rule out that a fundamentally different architecture change (not just a
loss/schedule tweak within the current relay structure) could still help the tail. But within the
scope of interventions tested here, **the tail-position weakness looks like an inherent property
of predicting further into the future / deeper into the relay-promotion chain**, not a scheduling
bug. The loss-reweighting arms (48, 49) are the least-bad option if some tail improvement is
wanted despite the net cost — they gave the mildest regressions and the only real (if modest)
tail gains — but neither should be adopted as a new default without accepting the accuracy loss
that comes with it.

## Caveats

- **Early snapshot only**: 50k of 10M planned steps for arms 44-49, single seed, no repeated
  trials. Arm 50 stopped at 37.8k steps of its own fine-tune counter upon observing a plateau, not
  a fixed budget.
- **Arm 50's warm-start point** was step=75,432 of baseline training, not the documented
  step=50,100 baseline datapoint used for the rest of this report's comparisons (see caveat above).
- **`refined_acc`/`modal_lift` for arm 50 are not comparable** to the other arms' numbers — with
  positions 0-3 receiving zero loss weight, overall accuracy necessarily craters as those
  positions drift untrained. Only the tail `by_pos`/`ema_logical` values are informative for arm 50.
- **Position-gradient-norm diagnostics were unavailable** (`--position-gradient-every-evals 0`,
  disabled after crashes) — this report relies entirely on accuracy-based evidence
  (`by_pos`, `ema_logical`/`ema_physical`), not gradient-norm evidence, to support its conclusion.
- Not an exhaustive test of every possible scheduling or architecture change — only the six
  interventions the ablation module implements were tried.

## Next

No further arms are planned in this sweep. Per the user's decision (2026-08-03), the next step is
an open-ended baseline run (no ablation flags, no 50k stop) left running overnight to see whether
simply training longer changes the tail-position pattern — independent of any of the interventions
tested here. See TaskList task 52.

## Addendum (2026-08-03): the "plateau" conclusion holds, but needed a much longer horizon than
## this report's original 50k/37.8k-step snapshots to confirm

The open-ended baseline run mentioned above (task 52) was carried to step=525,900 — roughly
10x the step budget used for every ablation arm above. Two things came out of it:

**Headline accuracy did not plateau early.** `best_refined` kept climbing well past this report's
50k-step reference point and past the 250-290k mark (the user's independent comparison point,
where their prior GTR Convergent architecture permanently plateaued for good): 0.2279 (flat,
~343k-404k) → 0.2298 (~428k) → 0.2318 (~477k) → 0.2363, a new best essentially at the final
checkpoint (~525k). This means **arm 50's tail-only fine-tune plateau claim, judged over only
37.8k of its own steps, was too short a horizon to distinguish "plateaued" from "still improving
very slowly"** — a real qualification the user flagged directly during monitoring.

**But the tail-position ceiling itself held up under the longer horizon.** Unlike the headline
number, `ema_logical` for positions 4-5 plateaued around step~340k (roughly 0.10-0.13) and did
not move further despite ~185k additional steps of normal, fully-competitive shared training —
final values (step=525,900): position 4 = 0.120, position 5 = 0.096, statistically indistinguishable
from where they sat at step 343k. **This lands at essentially the same tail ceiling arm 50 reached
via exclusive tail-only training (no front-position competition at all)** — two independent
training paths (tail-only fine-tune vs. ~10x-longer shared training) converging on the same
~0.10-0.13 tail accuracy band is stronger evidence for the report's core conclusion than either
result alone: the tail-position weakness looks like a genuine ceiling regardless of how much
signal or how many steps it's given, not an artifact of competing with the front for a fixed
budget within a short horizon.

**Revised takeaway:** trust the report's core conclusion (data/distance-limited ceiling, not a
fixable scheduling artifact) more than before, but read "plateau" as specifically about
tail-position accuracy, not overall model accuracy — the model as a whole keeps improving on a
much longer timescale than this report's 50k-step arms could show, even while the front/tail gap
itself stays roughly fixed.
