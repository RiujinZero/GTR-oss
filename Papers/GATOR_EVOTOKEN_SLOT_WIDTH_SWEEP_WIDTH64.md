# GTR-L EvoToken output-slot-width sweep (core width=64)

Status: **experimental**, early-training leading indicators only

Related: `Papers/GATOR_LINEAR_EVOTOKEN.md` (method), `gator_linear/evotoken.py` (implementation),
`Papers/GATOR_EVOTOKEN_SLOT_WIDTH_SWEEP.md` (width=16 sweep),
`Papers/GATOR_EVOTOKEN_SLOT_WIDTH_SWEEP_WIDTH32.md` (width=32 sweep, which this report extends).

## Motivation

The width=16 and width=32 sweeps both found that quality degrades monotonically with output-slot
count, and that doubling core width helps unevenly across slot counts rather than uniformly. This
sweep repeats the same five-point slot sweep at width=64 to extend the width=16/32/64 comparison
and test whether the pattern of uneven, slot-dependent gains continues, and whether the slot counts
that have "materially underperformed" throughout the sweep (slots=8, slots=10) are still gaining
from added width or have started to plateau.

## Method

Same as the width=16/32 sweeps (`GatorLinearConfig(vocab_size=VOCAB, streams=10, nodes=40, relays=4,
degree=6, buses=4, mailbox_channels=4, mailbox_width=8, basis_rank=4, address_width=4, router_rank=8,
conv_kernel=9, hard_routing=True, linear_attention=True, controlled_writing=True, causal=False)`),
with `width=64` (doubled from 32), same masked-only CTS v5 objective, batch 16, lr 1e-3, seed 0,
TinyStories no-whitespace tokenizer. `OUTPUT_SLOTS_TO_RUN` ∈ {2, 4, 6, 8, 10}, each trained to ~50,000
of the planned 10,000,000-step schedule and stopped manually once past that mark. Trained live on
Colab (single L4), 2026-08-01/02, run tag `gtrl-130k-evotoken-relay-cts-masked-s{N}-v5`. Total model
parameters: 484,301 at width=64 (core-only, constant across slot counts), vs 232,205 at width=32 and
129,197 at width=16.

Metrics below are pulled from the log line closest to step 50,000 in each run's full log
(`gtrl_evotoken_relay_relay_cts_masked_s{N}_v5.log` in the `GTR-L/gtrl_evotoken_relay/run_logs/`
Drive folder).

## Results

| slots | step | refined_acc | onepass_acc | best | exact | modal | modal_lift | weak_pos | retained | collapse_streak | promotions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2  | 50,100 | 0.2578 | 0.2539 | 0.2812 | 0.1133 | 0.123 | **+0.205** | 0/2  | 0.437 | 0 | (1,0,1,0) |
| 4  | 50,100 | 0.2266 | 0.2344 | 0.2393 | 0.0312 | 0.161 | **+0.165** | 0/4  | 0.353 | 0 | (1,1,1,1) |
| 6  | 50,100 | 0.1569 | 0.1647 | 0.1634 | 0.0156 | 0.272 | **+0.106** | 0/6  | 0.291 | 0 | (2,1,2,1) |
| 8  | 50,100 | 0.1421 | 0.1401 | 0.1426 | 0.0078 | 0.342 | **+0.085** | 2/8  | 0.252 | 0 | (2,2,2,2) |
| 10 | 50,100 | 0.1199 | 0.1203 | 0.1320 | 0.0078 | 0.434 | **+0.057** | 1/10 | 0.216 | 0 | (3,2,3,2) |

`collapse_streak` never left 0 in any of the five runs. Slots=10's step=50,100 snapshot landed on a
locally noisy point (neighboring steps 49,800/50,400 showed refined_acc 0.1266-0.1285, ~5-9% above
this exact line) — flagged in the caveats below rather than cherry-picking a friendlier step.

## Width=16 vs 32 vs 64, matched slot count

| slots | refined_acc (16→32→64) | 16→32 | 32→64 | modal_lift (16→32→64) | 16→32 | 32→64 |
|---:|---|---:|---:|---|---:|---:|
| 2  | 0.2422→0.2559→0.2578 | +5.7%  | +0.7%  | +0.189→+0.203→+0.205 | +7.4%  | +1.0%  |
| 4  | 0.1846→0.1914→0.2266 | +3.7%  | **+18.4%** | +0.123→+0.130→+0.165 | +5.7%  | **+26.9%** |
| 6  | 0.1283→0.1491→0.1569 | **+16.2%** | +5.2%  | +0.077→+0.098→+0.106 | **+27.3%** | +8.2%  |
| 8  | 0.1211→0.1279→0.1421 | +5.6%  | +11.1% | +0.064→+0.071→+0.085 | +10.9% | +19.7% |
| 10 | 0.1133→0.1207→0.1199 | +6.5%  | **−0.7%** | +0.050→+0.058→+0.057 | +16.0% | **−1.7%** |

## Reading the curve: the "big outlier gain" keeps moving, and slots=10 has stalled

The pattern first noticed comparing width=16→32 (where slots=6 stood out with the largest relative
gain) does **not** repeat at width=32→64 — instead slots=4 is the standout (+18.4%/+26.9%), while
slots=6 reverts to an ordinary moderate gain (+5.2%/+8.2%). Notably, slots=4's promotion schedule
`(1,1,1,1)` is the *most regular* one on the list, which undercuts rather than supports the earlier
working hypothesis that width specifically helps runs with uneven promotion schedules — the "special"
slot count is not fixed to a schedule property, and appears to genuinely move between width-doubling
transitions rather than settle on one structural explanation.

More importantly for the sweep's motivating question: **slots=10 shows no clear gain at all from
32→64** (−0.7% refined_acc, −1.7% modal_lift, within noise of flat), and slots=8's +11.1%/+19.7% gain,
while real, is smaller than slots=4's. This is the first width transition in the sweep where the
hardest configuration did not clearly benefit — a concrete, if early, signal that the "hard/easy gap
closes with more width" hypothesis (see width=32 report and prior discussion) may be running out of
room by width=64, at least for the very highest slot counts. Whether this is a genuine plateau or an
artifact of this specific noisy snapshot (see slots=10's neighboring-step variance above) is exactly
what width=128 will help settle.

## Caveats

- **Early snapshot only**, same as prior sweeps — 50k of 10M planned steps, single seed, no repeated
  trials. Slots=10's flat/negative 32→64 read in particular should be treated cautiously given the
  visible step-to-step noise at this snapshot (neighboring steps showed +5-9% instead).
- **Not a controlled parameter-count ablation.** Width=64 has ~2.09x the parameters of width=32
  (484,301 vs 232,205), and ~3.75x width=16's (129,197) — sub-quadratic growth in both steps, since
  only some parameter blocks (relay projections, attention) scale as width² while others (token/stream
  embeddings) scale linearly or not at all with width.
- **Checkpoint-collision bug recurred and was root-caused during this sweep** (see project memory):
  the experiment tag doesn't encode core width, so a width transition can find and try to resume a
  differently-shaped checkpoint. Now fixed with an unconditional cleanup step at the top of the launch
  cell (runs on every "Run all", not just after a natural process exit).

## Next

Width=128 sweep (slots 2, 4, 6, 8, 10) is next, per the user's explicit decision rule: if slots=8/10
show strong renewed gains at width=128, continue doubling to 256 and 512; if the plateau seen here at
slots=10 (and to a lesser extent slots=8) persists or deepens, that's the signal to stop the width
axis and pivot to a parallel nodes-scaling sweep (holding width fixed, varying `nodes=40` instead) as
the next capacity-scaling experiment. See `Papers/GATOR_EVOTOKEN_SLOT_WIDTH_SWEEP_WIDTH128.md`
(pending) for that result.
