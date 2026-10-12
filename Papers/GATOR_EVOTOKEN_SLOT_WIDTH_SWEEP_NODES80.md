# GTR-L EvoToken nodes-scaling sweep (nodes=80, width=128 fixed)

Status: **experimental**, early-training leading indicators only. **Stopped early** — see
"Why this sweep stops here" below.

Related: `Papers/GATOR_LINEAR_EVOTOKEN.md` (method), `Papers/GATOR_EVOTOKEN_SLOT_WIDTH_SWEEP_WIDTH128.md`
(the nodes=40/width=128 baseline this report compares against).

## Motivation

The width-doubling sweep (16→32→64→128) ended on a split result: slots=8 regressed at
width=64→128 while slots=10 rebounded strongly, undercutting a clean width-only story. Per the
user's decision, the next capacity axis tested was `nodes` (40→80, width held at 128) — a
different way of adding capacity than width, with a very different parameter-count profile
(nodes=40→80 adds only +2.4% params, vs any width-doubling step adding 80–142%).

## Method

Same as the width sweeps (`GatorLinearConfig(vocab_size=VOCAB, streams=10, width=128, relays=4,
degree=6, buses=4, mailbox_channels=4, mailbox_width=8, basis_rank=4, address_width=4,
router_rank=8, conv_kernel=9, hard_routing=True, linear_attention=True, controlled_writing=True,
causal=False)`), with `nodes=80` (doubled from 40), same masked-only CTS v5 objective, batch 16,
lr 1e-3, seed 0. `OUTPUT_SLOTS_TO_RUN` ∈ {2, 4, 6, 8} — **slots=10 was not run**; the sweep was
stopped after slots=8 per the decision below. Trained live on Colab (single L4), 2026-08-02, run
tag `gtrl-130k-evotoken-relay-cts-masked-s{N}-v5`. Total model parameters: 1,200,453 at nodes=80
vs 1,172,813 at nodes=40 (+2.4%).

Metrics are pulled from the log line closest to step 50,000 in each run's full log.

## Results

| slots | step | refined_acc | onepass_acc | best | exact | modal | modal_lift | weak_pos | retained | promotions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 50,100 | 0.2793 | 0.2812 | 0.2969 | 0.1406 | 0.104 | **+0.227** | 0/2 | 0.458 | (1,0,1,0) |
| 4 | 50,100 | 0.2246 | 0.2217 | 0.2451 | 0.0352 | 0.188 | **+0.163** | 0/4 | 0.363 | (1,1,1,1) |
| 6 | 50,100 | 0.1829 | 0.1732 | 0.1829 | 0.0117 | 0.235 | **+0.132** | 0/6 | 0.308 | (2,1,2,1) |
| 8 | 50,100 | 0.1401 | 0.1431 | 0.1440 | 0.0156 | 0.347 | **+0.083** | 1/8 | 0.263 | (2,2,2,2) |

`collapse_streak` stayed at 0 throughout all four runs.

## nodes=40 vs nodes=80, matched slot count

| slots | refined_acc (n40→n80) | Δ relative | modal_lift (n40→n80) | Δ relative |
|---:|---|---:|---|---:|
| 2 | 0.2617→0.2793 | **+6.7%** | +0.209→+0.227 | **+8.6%** |
| 4 | 0.2490→0.2246 | **−9.8%** | +0.188→+0.163 | **−13.3%** |
| 6 | 0.1823→0.1829 | +0.3% | +0.132→+0.132 | +0.0% |
| 8 | 0.1387→0.1401 | +1.0% | +0.082→+0.083 | +1.2% |

Four slot counts, three different outcomes: a real gain (slots=2), a real regression (slots=4,
checked stable against neighboring steps — not noise), and two essentially-flat results (slots=6,
8). No slot count shows the same pattern as any other, and nothing here recovers the clean
"harder configs benefit more" story that the width sweep's early transitions suggested.

## Throughput: the real cost of nodes-scaling

Measured directly from raw-log timestamp pairs (not estimated): nodes=40/width=128 trains at a
consistent **857 steps/min**, confirmed independently across slots=6, 8, and 10's raw logs (all
three give exactly 857/min from 600-step timestamp windows). nodes=80/width=128 trains at
**~530–600 steps/min**, consistent across all four completed slot counts (slots=2: ~551–600/min,
slots=4: ~543/min, slots=6: ~533/min, slots=8: ~529/min).

**That's a ~35% throughput reduction for a +2.4% parameter increase.** `nodes` behaves structurally
more like a sequence-length axis for the relay/router/attention computation than a weight-count
axis — the routing/attention compute must process 2x the nodes every step even though the actual
weight matrices are mostly shared/reused across nodes and barely grow. The result: nodes-scaling
gives width-sweep-beating parameter efficiency but width-doubling-level compute cost. Any ROI
comparison between the width and nodes axes needs to pick one of these framings explicitly —
**ROI per parameter favors nodes heavily; ROI per GPU-second likely favors width** — and this
report does not resolve which framing matters more for the project's actual constraints (that is
a separate decision, not a data question).

## Why this sweep stops here

Per the user's explicit call (2026-08-02): the nodes=80 results across slot counts were judged
"getting noisy and not informative" — a real gain, a real regression, and two flat results with
no discernible pattern relating slot count to outcome. Continuing to nodes=160 (the originally
planned next step) was cancelled rather than deferred. Instead, the investigation pivots to a
more targeted question: **why do the tail (highest-index) output positions stay weak regardless
of which capacity axis is scaled?** This was observed directly in this sweep's own raw logs — for
slots=6 at nodes=80, position 0's accuracy climbed from 0.21 to ~0.35 over the run while positions
4–5 stayed flat near 0.06–0.09 the entire time, unaffected by the extra nodes. A new ablation
module (`gtrl_evotoken_colab.py`, see `Runbooks/GTRL_EVOTOKEN_COLAB_MODULE.md`) was built
specifically to separate the candidate explanations (data/distance-limited vs. promotion-schedule
credit-assignment starvation) via targeted interventions rather than further blind capacity
scaling. See `Papers/GATOR_EVOTOKEN_TAIL_POSITION_ABLATIONS.md` (pending) for that result.

## Caveats

- **Early snapshot only**, same as every prior sweep stage — 50k of 10M planned steps, single
  seed, no repeated trials.
- **slots=10 was never run at nodes=80** — the sweep was cut short at slots=8. The nodes=40-vs-80
  comparison table above is missing that row entirely; do not assume a slots=10 result exists.
- **Not a controlled parameter-count ablation** in the traditional sense — nodes=40→80 adds far
  fewer parameters than any width step did, which is itself the interesting finding here (see
  throughput section), not a flaw to correct for.
- slots=4's regression was checked against neighboring steps (49,800/50,400) and confirmed stable,
  not a single noisy snapshot.

## Next

Nodes-scaling (this axis) is not planned to continue to nodes=160 unless the tail-position
ablations reopen the question. See `Papers/GATOR_EVOTOKEN_TAIL_POSITION_ABLATIONS.md` for the
active follow-up work.
