# GTR-L EvoToken output-slot-width sweep (core width=128)

Status: **experimental**, early-training leading indicators only

Related: `Papers/GATOR_LINEAR_EVOTOKEN.md` (method), `gator_linear/evotoken.py` (implementation),
`Papers/GATOR_EVOTOKEN_SLOT_WIDTH_SWEEP_WIDTH64.md` (width=64 sweep, which this report extends).

## Motivation

The width=16/32/64 sweeps established that quality degrades monotonically with output-slot count,
that width-doubling helps unevenly across slot counts, and that width=64 was the first transition
where the hardest configuration (slots=10) showed no clear gain (−0.7% refined_acc, −1.7% modal_lift,
within noise). Per the user's explicit decision rule: if width=128 shows strong renewed gains at the
"end slots" (8 and 10), continue doubling to 256 and 512; if gains have flattened there, stop the
width axis and pivot to a nodes-scaling sweep (`nodes=40` held fixed, `width` fixed, `nodes` varied)
instead.

## Method

Same as the width=16/32/64 sweeps (`GatorLinearConfig(vocab_size=VOCAB, streams=10, nodes=40,
relays=4, degree=6, buses=4, mailbox_channels=4, mailbox_width=8, basis_rank=4, address_width=4,
router_rank=8, conv_kernel=9, hard_routing=True, linear_attention=True, controlled_writing=True,
causal=False)`), with `width=128` (doubled from 64), same masked-only CTS v5 objective, batch 16,
lr 1e-3, seed 0, TinyStories no-whitespace tokenizer. `OUTPUT_SLOTS_TO_RUN` ∈ {2, 4, 6, 8, 10}, each
trained to ~50,000 of the planned 10,000,000-step schedule and stopped manually once past that mark.
Trained live on Colab (single L4), 2026-08-01/02, run tag `gtrl-130k-evotoken-relay-cts-masked-s{N}-v5`.
Total model parameters: 1,172,813 at width=128 (core-only, constant across slot counts), vs 484,301 at
width=64, 232,205 at width=32, and 129,197 at width=16.

Metrics below are pulled from the log line closest to step 50,000 in each run's full log
(`gtrl_evotoken_relay_relay_cts_masked_s{N}_v5.log` in the `GTR-L/gtrl_evotoken_relay/run_logs/`
Drive folder).

## Results

| slots | step | refined_acc | onepass_acc | best | exact | modal | modal_lift | weak_pos | retained | collapse_streak | promotions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2  | 50,100 | 0.2617 | 0.2676 | 0.2910 | 0.1172 | 0.117 | **+0.209** | 0/2  | 0.461 | 0 | (1,0,1,0) |
| 4  | 50,100 | 0.2490 | 0.2520 | 0.2568 | 0.0391 | 0.165 | **+0.188** | 0/4  | 0.366 | 0 | (1,1,1,1) |
| 6  | 50,100 | 0.1823 | 0.1823 | 0.1823 | 0.0156 | 0.260 | **+0.132** | 0/6  | 0.316 | 0 | (2,1,2,1) |
| 8  | 50,100 | 0.1387 | 0.1465 | 0.1450 | 0.0117 | 0.369 | **+0.082** | 1/8  | 0.255 | 0 | (2,2,2,2) |
| 10 | 50,100 | 0.1336 | 0.1316 | 0.1402 | 0.0117 | 0.479 | **+0.071** | 1/10 | 0.228 | 0 | (3,2,3,2) |

`collapse_streak` never left 0 in any of the five runs. Slots=8's step=50,100 snapshot was checked
against neighboring steps (49,800: refined_acc=0.1309, 50,400: 0.1411) and is representative, not a
noisy outlier — the regression reported below is a real read, not a snapshot artifact.

## Width=16 vs 32 vs 64 vs 128, matched slot count

| slots | refined_acc (16→32→64→128) | 16→32 | 32→64 | 64→128 | modal_lift (16→32→64→128) | 16→32 | 32→64 | 64→128 |
|---:|---|---:|---:|---:|---|---:|---:|---:|
| 2  | 0.2422→0.2559→0.2578→0.2617 | +5.7%  | +0.7%  | +1.5%  | +0.189→+0.203→+0.205→+0.209 | +7.4%  | +1.0%  | +2.0%  |
| 4  | 0.1846→0.1914→0.2266→0.2490 | +3.7%  | +18.4% | +9.9%  | +0.123→+0.130→+0.165→+0.188 | +5.7%  | +26.9% | +13.9% |
| 6  | 0.1283→0.1491→0.1569→0.1823 | +16.2% | +5.2%  | **+16.2%** | +0.077→+0.098→+0.106→+0.132 | +27.3% | +8.2%  | **+24.5%** |
| 8  | 0.1211→0.1279→0.1421→0.1387 | +5.6%  | +11.1% | **−2.4%** | +0.064→+0.071→+0.085→+0.082 | +10.9% | +19.7% | **−3.5%** |
| 10 | 0.1133→0.1207→0.1199→0.1336 | +6.5%  | −0.7%  | **+11.4%** | +0.050→+0.058→+0.057→+0.071 | +16.0% | −1.7%  | **+24.6%** |

## Reading the curve: the "end slots" split apart from each other

Width=128 breaks the pattern in a genuinely new way. Every prior transition had a single standout
outlier slot count; this one has an outlier (slots=6, back to being the largest gainer after ceding
that spot to slots=4 at 32→64) **and** a reversal (slots=8, the first slot count in the entire sweep
to regress on a stable, noise-checked snapshot). Slots=10 — which flatlined at 32→64 — rebounded to
one of the two largest gains in this transition (+11.4%/+24.6%, essentially tied with slots=6).

This directly complicates the decision rule set up after width=64: the criterion was "strong renewed
gains at the end slots (8 and 10)." Slots=10 clearly satisfies that. Slots=8 does the opposite — a
real, checked regression, not noise. The two configurations the rule was built to move together on
have split in opposite directions for the first time.

A plausible reading: slots=8's `(2,2,2,2)` promotion schedule is the most evenly-loaded schedule in
the sweep (each relay handles exactly 2 promotions, same absolute count as slots=4's `(1,1,1,1)` times
two), while slots=10's `(3,2,3,2)` is the most unevenly-loaded. Combined with slots=6's `(2,1,2,1)`
schedule also gaining strongly, this is at least consistent with the working hypothesis from the
width=32 report — that added width helps most where the promotion schedule is uneven — moving from
"undercut" (at 32→64, where the most regular schedule was the biggest gainer) back to "supported."
But this hypothesis has now flipped support twice across four transitions, so it should be treated as
a loose correlation to keep watching, not a settled explanation.

## Recommendation on width=256 vs pivoting to nodes-scaling

This is a genuine split decision, not a clean read either way, and is flagged here rather than forced:

- **For continuing to width=256:** slots=10 — the single hardest configuration in the sweep, and the
  one that most directly represents "does capacity keep paying off at the far end" — posted its
  strongest transition yet. If the promotion-schedule-unevenness correlation above holds, slots=10
  should keep benefiting from further width at least as much as it just did.
- **For pivoting to nodes-scaling:** slots=8 regressing on a checked, non-noisy snapshot is the first
  negative result anywhere in this sweep. If it is an early signal that width alone is no longer a
  uniformly reliable lever past this point, throwing more width at 256 risks diminishing or mixed
  returns on some slot counts while others keep gaining — exactly the kind of unevenness that a
  nodes-scaling sweep (a different capacity axis) could help disentangle from a width-specific effect.

Given the split, this report does not pick a side — it's the one decision point in this sweep that
depends on which failure mode (a regressing mid/high slot count vs a plateaued highest slot count)
the user weighs as the more important signal to chase. Both the width=256 sweep and the nodes-scaling
sweep are cheap to spec once a direction is chosen.

## Caveats

- **Early snapshot only**, same as prior sweeps — 50k of 10M planned steps, single seed, no repeated
  trials.
- **Not a controlled parameter-count ablation.** Width=128 has ~2.42x the parameters of width=64
  (1,172,813 vs 484,301), ~5.05x width=32's, and ~9.08x width=16's — sub-quadratic growth throughout,
  since only some parameter blocks (relay/attention projections) scale as width² while others
  (embeddings) scale linearly or not at all.
- Slots=2 and slots=4 for this width were launched by the user directly (not via the automated
  stop/edit/relaunch cycle used for slots 6/8/10), on a freshly reset Colab session for a clean
  24-hour execution window; this does not affect comparability of the reported datapoints.

## Next

Pending user direction: either the width=256 sweep (slots 2, 4, 6, 8, 10) or a nodes-scaling sweep
(`nodes=40` varied, width held at its current value) — see Recommendation above. The 30-run LR sweep
(slots ∈ {2, 8} × width ∈ {16, 32, 64, 128, (256 if run)} × lr ∈ {3e-3, 3e-4, 3e-5}, vs the existing
lr=1e-3 baselines) remains fully specified but not yet started, per standing instruction to wait for
this report's outcome and a confirmed width list.
