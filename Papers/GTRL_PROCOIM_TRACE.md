# GTR-L under Procoim: architecture trace, learning mechanism, and lever catalog

Status: **research notes, actively evolving** — written 2026-08-07 as a
snapshot of the overnight Procoim investigation. Treat every number in the
Results section as a checkpoint, not a final verdict; the underlying
"continuous learning" question (§6) is explicitly still open.

Related documents:

- `Papers/GTRL_ARCHITECTURE.md` — the full GTR-L architecture reference this
  document's §1 summarizes. Read that document for anything not covered here.
- `Papers/GTRL_CONFIGS.md` — production-size (100M/500M) GTR-L configurations
  and measured throughput/VRAM, trained with plain AdamW.
- `Papers/PROCOIM_OPTIMIZER.md`, `Papers/PROCOIM_OPTIMIZER_SPEC_V0.2.md` —
  the original Procoim design docs (partially stale relative to tonight's
  additions; this document supersedes them for the mechanisms it covers).
- `Papers/PROCOIM_RETENTION_100K_REPORT.md`,
  `Papers/PROCOIM_ADAMW_WEIGHT_GEOMETRY_REPORT.md` — two earlier diagnostic
  reports whose findings directly motivated features described in §3-4.
- `gator/optimizers/procoim.py` — the implementation. Every mechanism
  described below is a direct read of this file as of commit `ac94996` plus
  same-night uncommitted changes (adaptive reduction-fraction, adaptive K,
  forwards-based and stall-based basis rotation).

---

## 1. What GTR-L is, briefly

GTR-L predicts a block of output tokens simultaneously instead of one token
at a time. Every sequence position holds several private **streams** (parallel
interpretations), which communicate only by writing into and reading from
shared **node** mailboxes — never by averaging hidden states together. A
**relay** is one full write/mix/read/route/transform/update pass; GTR-L
normally stacks four. **EvoToken** is a refinement policy layered on top:
output slots start as MASK embeddings and evolve into decoded tokens over
one phase per relay, with a scheduled subset "promoted" to a hard commitment
at each phase.

The two central state tensors are the private hidden state
`H : [B, L, S, D]` and the node occupancy `Q : [B, L, S, N]`. Width (`D`)
drives quadratic parameter growth (`36*D^2 + 21,188*D + 140,773` parameters
for the current 16-stream/128-node/4-relay production topology); nodes and
streams are comparatively cheap to scale because no node or stream owns a
full `D x D` transform of its own — relays share one small program basis
(`basis_rank`) and nodes only carry coefficient vectors into it.

Every run in this document uses a **small proxy task**, not full language
modeling: a modular-prefix-formation task with `--length 2` (2 output slots)
and a tiny GTR-L instance (streams=10, nodes=40, width=8, relays=4,
46,701 parameters). The proxy exists to let Procoim iterate fast; see §6 for
why the prefix task's single-correct-answer structure makes it an imperfect
stand-in for language.

## 2. Why Procoim exists

Procoim ("Progressive Continuous Improvement") is a forward-only optimizer:
it never calls `.backward()`. The motivation is explicitly biological, not
just an engineering shortcut around backprop memory costs — quoting the
design rationale behind this work: biological systems don't run exact
gradient credit assignment, they make an educated guess and refine from
there. Procoim's antithetic-probe-then-accept-or-reject loop is a direct
structural analogue of perturbation/node-noise learning rules from
computational neuroscience.

Two concrete problems with AdamW motivate it:

1. **VRAM.** Backprop needs the backward graph plus AdamW's own two moment
   buffers per parameter — all memory that a forward-only method avoids.
2. **No train/inference split.** Because Procoim never builds a backward
   graph, there is no structural reason the same accept/reject loop that
   improves the model during "training" couldn't keep running while the
   model is deployed — i.e. genuine continuous/online learning during
   inference, not just cheaper offline training. This is the deeper,
   still-unproven goal (§6), not merely "train faster/cheaper."

The accepted success framing (settled across several restatements this
session): **calculated guesses will never beat a calculated plan, so Procoim
will never out-learn AdamW per step** — but because each Procoim step is so
much cheaper (no backward graph, no full Adam moment buffers), it can still
win on wall-clock time-to-accuracy as long as its per-step learning rate
stays above some "livable fraction" of AdamW's. Empirically, across tonight's
tests, that livable fraction landed around **15-20%**.

## 3. The core learning mechanism

### 3.1 Episodes and candidate banks

Procoim organizes training into **episodes**. Each episode:

1. Runs a deterministic baseline forward pass on the current example.
2. Repeatedly samples a **candidate bank** — a batch of antithetic ±ε
   perturbations applied to one parameter group at a time (`routing`,
   `receiver_memory`, `stream_mix`, `node_compute`, `readout`, optionally
   `embedding`).
3. Scores each candidate against the current baseline. A candidate is
   **eligible** only if it reduces loss on unsolved ("active") output slots
   by at least the required fraction (`reduction_fraction`, see §4.1)
   without regressing any "protected" slot beyond tolerance.
4. If an eligible candidate also survives whichever reference/verification
   gate is enabled (§3.3), it is **accepted**: applied to the live weights,
   and its slot(s) move toward the PROTECTED state.
5. The episode ends when every slot is solved (`completed=all_slots`) or the
   candidate-bank budget (`--procoim-max-candidate-banks`) is exhausted
   (`completed=candidate_bank_budget`, i.e. a stall).

Per-slot state is a small lifecycle: **ACTIVE** (still being optimized) →
**SOFT_FROZEN** → **PROTECTED** (regressions here are vetoed). This is what
makes "reduction_fraction" a per-slot bar rather than a whole-batch one, and
why difficulty scales with slot count: more slots means more simultaneous
per-slot bars to clear before an episode can complete cleanly.

### 3.2 Directional Adam ("hybrid Procoim")

`--procoim-directional-adam` is the informal "hybrid Procoim" — the actual
fusion of perturbation search with Adam-style refinement implied by the
optimizer's name (distinct from an earlier, never-implemented spec idea of
splitting parameter groups between Procoim and AdamW). Mechanism:

1. Estimate a gradient direction via `K` (`candidate_directions`) antithetic
   forward probes in a **seeded low-rank rotating basis** (not a fixed random
   subspace — see §3.4 for why).
2. Run real Adam (momentum + adaptive step, with Dykstra projection to
   respect protected-slot constraints) **in that low-rank coefficient
   space**, not in full parameter space.
3. Verify the resulting update with an actual forward pass before committing
   Adam's moments — i.e. Adam's internal bookkeeping only ever advances on
   empirically-confirmed improvements, never on an unverified extrapolated
   step the way ordinary Adam would.

This maps directly onto the "guess, then refine" philosophy: the antithetic
probing is the biologically-plausible educated guess (forward-only, no exact
gradient); Adam-in-coefficient-space is the refinement, biasing future
guesses toward directions that have paid off before.

**Why a rotating basis, not a static one.** `Papers/PROCOIM_ADAMW_WEIGHT_GEOMETRY_REPORT.md`
traced real AdamW's own weight updates every 500 steps and found them
strongly low-rank (rank-8 captures ~98.5% of update energy) but **rotating**
substantially between intervals (mean cosine similarity with the previous
interval's dominant subspace was only 0.24). A fixed K=8 random search basis
captured under 1% of that movement. This directly motivated
`directional_basis_mode="rotating_factorized"` (now the default): low-rank
directions that periodically re-randomize orientation instead of sitting in
one static subspace forever. See §3.4 for the three ways "periodically" can
be defined.

### 3.3 Three reference/verification mechanisms

Beyond the per-episode active/protected-slot bookkeeping in §3.1, Procoim has
three **independent, stackable** guards against a locally-eligible candidate
that only helps the current example at the expense of everything else:

| Mechanism | Flag | Behavior |
|---|---|---|
| **Anchor bank / consolidation** | `--procoim-anchor-buffer-size`, `--procoim-anchor-sentinel-count`, `--procoim-consolidation-interval-examples` | Periodically audits accepted updates against a bank of retained past examples; can hard-veto a candidate that regresses the bank's net loss. |
| **Past-reference partner** | `--procoim-past-reference-buffer-size`, `--procoim-past-reference-policy` (`improve`/`nonregress`), `--procoim-past-reference-rotation-forwards` | Maintains a reservoir of prior episode batches; a uniformly-selected past batch must ALSO pass the policy's bar before a candidate may commit. The recalled partner rotates every N forwards (0 = one partner per episode). |
| **Reference guard** | `--procoim-reference-guard-batch-size` | A FIXED, independently-generated batch (same size every check, never rotates) that every candidate must also improve. Zero disables it. |

These are not equivalent, and tonight's evidence separates them clearly:

- **Anchor/consolidation was found to be an active bottleneck**, not a
  helpful guard, on the small proxy task — every run with it enabled stalled
  at ~4 accepted updates total regardless of how far other thresholds were
  loosened (§5.2). It has been **disabled** (`0 0 0`) in every "good" run
  since that discovery.
- **Past-reference (rotating) vs reference-guard (fixed)** was run as a
  direct A/B tonight (§5.4). The mechanistic reason a fixed reference is
  harder: every accept must land in the intersection of "helps the current
  example" AND "helps this one unchanging batch" — that intersection stays
  narrow and **constant** for the whole run. A rotating partner keeps that
  same narrow-intersection constraint, but it *moves*, so across many
  episodes the model gets many different narrow windows to pass through
  instead of one persistently narrow one it must keep re-satisfying. This
  produced a measurable accept-rate gap (29 vs 97 accepts at a similar step
  count in one comparison). The rotating mechanism is expected to suit
  language better for the same reason — a static reference-guard bank risks
  the search overfitting specifically to that bank.

### 3.4 Three directional-basis rotation modes

Separate from *which reference gates a candidate*, this controls *when the
low-rank probe subspace itself gets reshuffled* for directional Adam. All
three are mutually exclusive (validated); priority when multiple are
configured is stall-banks > forwards > bank-count.

| Mode | Flag | Rotates when... |
|---|---|---|
| **Bank-count** (default) | `--procoim-directional-basis-rotation-banks` (default 8) | a parameter group has attempted N candidate banks, regardless of outcome. |
| **Forwards-based** | `--procoim-directional-basis-rotation-forwards` | a parameter group has accumulated N total search forwards. |
| **Stall-based** (newest) | `--procoim-directional-basis-rotation-stall-banks` | a parameter group has failed N *consecutive* banks; the streak resets to 0 on any success. |

The forwards-based mode has a real interaction with adaptive K (§4.2) worth
understanding: at the K=256 adaptive ceiling, one directional-Adam bank costs
`2*256=512` forwards — already past a `rotation_forwards=500` threshold in a
*single* bank attempt. So a struggling group (wide K, by design) also gets
its basis reshuffled almost every attempt, while a group succeeding at
narrow K needs ~15-16 banks to accumulate one rotation. Confirmed empirically
(§5.4): groups pinned at K=256 under forwards-based rotation showed
`group_gain_per_forward` roughly **15-20x lower** than a group that stayed
narrow. Reverting to bank-count-based rotation evened this gap out. Stall-
based rotation was built specifically to decouple "how hard a group is
struggling" from "how often its subspace gets yanked out from under it" — a
group that's succeeding with its current basis is never rotated on a
schedule; a group that's stuck gets exactly one fresh subspace attempt within
its remaining episode budget.

## 4. Adaptive mechanisms (built this session)

Both were added specifically because a fixed reduction-fraction bar and a
fixed K get harder to satisfy as a run progresses (more of the "easy"
accepts have already been taken), causing episodes to slow down and
eventually stall near the end of a long budget.

### 4.1 Adaptive `reduction_fraction`

`--procoim-adaptive-reduction-fraction` (with `--procoim-reduction-fraction-decay-rate`,
default 0.1, and `--procoim-minimum-reduction-fraction`, default 1e-6) makes
the per-slot CE-reduction bar decay **logarithmically with completed episode
count**:

```text
effective = max(minimum_reduction_fraction,
                 reduction_fraction / (1 + decay_rate * log1p(episodes)))
```

The underlying principle, stated directly: **difficulty scales exponentially
with slot count, so the CE-reduction bar has to come down as slot count (or
training progress) increases** — late-stage gains are intrinsically harder
to find than early ones, so the bar should relax rather than stay fixed. A
longer run accumulates more episodes, so `log1p` growth flattens the decay
earlier in relative terms — a 30k-step run (few tens of episodes) is still
actively decaying for its whole duration, while a 100k+-step run spends most
of its length against a near-stabilized bar.

### 4.2 Adaptive `candidate_directions` (K)

`--procoim-adaptive-candidate-directions` (with `--procoim-minimum-candidate-directions`,
default 16, and `--procoim-candidate-directions-growth`, default 2.0) starts
each parameter group's search at a **narrow** K and widens it on failure:

```text
on a failed bank:  K = min(declared_max, max(minimum_K, ceil(K * growth)))
on a successful bank: K resets to minimum_K
```

Note `--procoim-candidate-directions` itself keeps its original meaning — the
declared/maximum width, which is what actually sizes directional Adam's
persistent moment buffers. Narrower probes are zero-padded up to that
declared width before touching the moment state, so widening K adaptively
never requires resizing anything persistent. The user's framing for why this
is worth doing: **"K is the cheapest lever" and also "K has the largest
effect"** — search a small K first (cheap, covers easy cases fast, enables
early exit), only pay for a wide search when a group is actually stuck.
Confirmed live in the 30k/250k runs: groups that kept failing climbed to the
K=256 ceiling; groups that kept succeeding stayed at K=16-32.

## 5. Results log

All runs below use the small proxy task (`--length 2`, 46,701-parameter
GTR-L, streams=10/nodes=40/width=8/relays=4, hard routing, `--balanced-prefix-batches`,
`--prefix-loss-vocabulary valid`, batch-size=144, seed=0) unless noted.
**Chance accuracy on this task is ≈8.33%** (1/12 valid symbols); "escaped
chance" is defined as accuracy sustaining above 2x chance (≈16.7%).

### 5.1 Two real config bugs found early

Before any of the mechanism work below, two plain misconfigurations were
caught live and fixed:

- `--prefix-loss-vocabulary` was `full` (16 symbols) instead of `valid` (the
  12 symbols the task can actually emit). Under `full`, the search could
  score fake "improvement" by suppressing the 4 dead symbols' logit mass
  without ever moving real argmax accuracy — a plausible explanation for
  loss-inches-down-while-accuracy-stays-flat patterns seen broadly.
- `--batch-size` was `1` instead of `144`, making every candidate probe's
  loss estimate maximally noisy.

### 5.2 The anchor/consolidation bottleneck

Every run with anchor/consolidation enabled (the initial default) hit a hard
ceiling of **~4 accepted updates total**, regardless of how far
`reduction_fraction`, K, past-reference-rotation-forwards, or even routing
mode (hard vs soft) were tuned. A user-supplied reference script revealed the
fix: **disable anchor/consolidation entirely**
(`--procoim-anchor-buffer-size 0 --procoim-anchor-sentinel-count 0 --procoim-consolidation-interval-examples 0`).
This, combined with `--balanced-prefix-batches` and `--hard-routing`, is the
first config that actually learned past a handful of updates.

### 5.3 The reference config and the historical benchmark

The reference script's **original run** (30k steps, fixed K=128,
`reduction_fraction=0.0001`, past-reference buffer=64, no anchor) produced
the numbers that became the standing benchmark for the rest of the night:

| Metric | Value |
|---|---:|
| valid accuracy | 9.38% |
| valid exact-match | 0.98% |
| final CE | 2.4916 |
| accepted updates | 34 |
| partner vetoes | 87 |
| throughput | 6.19 forwards/s |

Reproducing this config locally (native Windows Python) landed very close —
`valid_acc=9.3%`, `valid_exact=1.0%` at step 36,705/100,000 before being
superseded by the adaptive run below — but ran at only ~2.9-4.7 steps/s
(vs the reference's 6.19), a throughput gap eventually traced to environment
(WSL/`uv`-installed CPU torch wheel vs native-Windows system Python), not
config.

### 5.4 Adaptive-mechanism runs (this session's new features)

**30k adaptive test** (`slot2_adaptive_rf_and_k_30k_seed0`) — adds adaptive
reduction-fraction (start 0.0001, decay-rate 0.1) and adaptive K
(min 16 → ceiling 256, growth 2.0) on top of the reference config, same 30k
budget:

| Metric | Reference (30k) | Adaptive (30k) |
|---|---:|---:|
| valid accuracy | 9.38% | 7.03% (best 8.89%) |
| valid exact-match | 0.98% | **1.17%** |
| accepted updates | 34 | **46** |
| CE | 2.4916 | 2.7543* |

*not directly comparable — the reference's 2.4916 was `train_loss`; this
run's eval-`loss` field reads 2.7543. Exact-match, the metric the user cares
about most, beat the reference. Verdict at the time: "mixed but genuinely
promising" — cleared the bar to proceed to a full-length run.

**250k past-reference run** (`slot2_adaptive_full_250k_seed0`, launched via
WSL/`uv` to match the reference environment's throughput, adds
`--procoim-directional-basis-rotation-forwards 500`) — final result:

| Metric | Value |
|---|---:|
| valid accuracy | 8.40% (best 9.18%) |
| valid exact-match | **1.37%** (best of the whole night) |
| CE (eval loss) | 2.7719 |
| accepted updates | 264 |
| runtime | 36,398s (~10.1 hours) |
| throughput | 6.86 steps/s |
| escaped chance | **No** — `escaped_chance_at=None` |

Beat the 30k adaptive test on every axis at the full budget (exact
1.17%→1.37%, valid_acc 7.03%→8.40%, accepts 46→264) and beat the historical
reference on exact-match (0.98%→1.37%) and accept count (34→264), while
landing a bit under the reference's raw valid_acc (9.38% vs 8.40%,
though `best_acc=9.18%` came close).

**Reference-guard A/B, bank-based rotation** (`slot2_adaptive_refguard_250k_seed0`,
first launch) — isolates reference mechanism only (past-reference → fixed
`--procoim-reference-guard-batch-size 144`), everything else identical to the
250k past-reference run including forwards-based rotation initially, then
reverted to bank-based rotation per the analysis in §3.4. Result: **genuinely
stalled** by step 95,788 — `directional_adam_accepts_total` frozen at 24 for
~39,500 consecutive steps, all six parameter groups simultaneously pinned at
the K=256 ceiling, every episode exhausting the full 16-bank budget with zero
accepts. Confirms the reference-guard mechanism is measurably harder to
satisfy than the rotating past-reference partner (fewer accepts at a matched
step count even before the stall: 29 vs the past-reference run's 97 at a
similar point).

**Reference-guard A/B, stall-based rotation** (same run, relaunched with
`--procoim-directional-basis-rotation-stall-banks 8` after the bank-based
stall) — **currently in progress**. Latest checkpoint read (non-invasively,
without disturbing the run): step=43,113/250,000 (17.2%), `valid_acc=8.7%`,
`valid_exact=1.2%`, `accepted_total=20`. Too early to compare against the
bank-based stall point (which took until step 95,788 to fully lock up); this
run has not yet reached that horizon.

### 5.5 The AdamW comparison point

A matched-config AdamW baseline (same architecture, batch-size=64,
`--length 7`, not `--length 2`) was run for ~90 minutes (pid=9070, ~3.01
steps/s, 15,000 steps): loss plateaued flat at ~2.485, accuracy oscillated
0.080-0.085 — essentially **at chance** (1/12 = 0.0833) at that step count.
This is a materially different task configuration (7 slots, not 2) from the
Procoim runs above, so it is not a clean apples-to-apples number, but it is
the only AdamW datapoint on record on this same small architecture and
should not be read as a strong bar — AdamW itself hadn't clearly escaped
chance within this budget either.

## 6. Cross-run observations and open questions

**A soft ~10% valid_acc ceiling appears across every run tonight**,
regardless of mechanism: the 30k adaptive test touched exactly 10.0% once and
reverted; the historical reference topped at 9.38%; the completed 250k
past-reference run's best was 9.18%; the in-progress stall-based refguard run
has only reached 8.7-9.4% so far. Five structurally different configurations
(fixed vs adaptive reduction-fraction, fixed vs adaptive K, three basis-
rotation modes, past-reference vs reference-guard) all show the same shape:
brief touches at or just under 10%, then reversion — never a sustained hold
above it, and **`escaped_chance_at=None` on every single run without
exception**. This reads as a real ceiling in the current config, not noise
around a climbing trend.

**CE declines smoothly; accuracy moves in flat-then-jump spikes, not a
smooth climb.** `train_loss` drops continuously whenever accepts are
happening, but eval `loss`/`valid_acc`/`valid_exact` sit flat for long
stretches then jump. Working explanation: most accepted updates narrow or
widen the probability margin between the top token and the rest without
flipping which token wins the argmax — CE moves continuously with every
accept, but discrete accuracy only jumps when an update (or an accumulated
sequence of them) actually crosses a slot's decision boundary.

**The honest summary of tonight's work, stated directly and worth repeating
verbatim in spirit: everything built this session improved the *search
mechanism's* health** — more accepted updates, CE that genuinely and
continuously declines when accepts are happening, no more hard stalls once
anchor/consolidation was removed and the adaptive mechanisms landed. **None
of it has yet demonstrated the actual goal: sustained, compounding progress
toward real task competence.** "The search works better now" and "we've made
progress toward continuous learning" are different claims, and only the
first one has real evidence behind it so far.

**Why the prefix task may not predict language behavior.** The proxy task is
algorithmic with exactly one correct circuit to converge to (a fixed
modular-arithmetic rule) — even a noisy search eventually stumbles onto it
and gets reinforced by the accept-only-if-verified gate. Language has no
single correct circuit, only a broad statistical target. It is genuinely
unclear whether that helps Procoim (more forgiving, nothing sharp to miss)
or hurts it (no clean signal to eventually lock onto). This is not
answerable from prefix-task data alone — the real validation gate is testing
Procoim directly on an actual, even toy, language task and checking whether
the same continuous-noisy-improvement shape shows up there, or whether it
stalls differently.

**Forward-looking scale expectation.** K is expected to need to reach into
the thousands once Procoim is applied to language on a large model — this
46,701-parameter proxy model is already at K=256 (adaptive ceiling) and may
not have needed to go that far only because of its own tiny size. Treat K as
a quantity that should scale with model/task complexity, not a fixed small
constant, when this work moves to language.

**Sequencing idea, not yet tried.** One standing proposal: train with AdamW
first at small context (fast, well-understood, gets a model off the ground),
then switch to Procoim once context/model size grows large enough that
AdamW's backward-pass memory footprint becomes the real constraint — since
Procoim's actual structural advantage (no backward graph) only matters once
that constraint bites. Whether Procoim can pick up and keep improving an
already-AdamW-converged checkpoint as reliably as it learns from scratch is
completely untested.
