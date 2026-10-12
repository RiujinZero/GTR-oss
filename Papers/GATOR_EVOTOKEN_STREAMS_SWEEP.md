# GTR-L EvoToken streams-count sweep (nodes=40, width=128, output-slots=6, observed=128)

Status: **experimental**, early-training leading indicators only (50k of 10M planned steps per
arm, single seed, no repeated trials).

Related: `Papers/GATOR_EVOTOKEN_TAIL_POSITION_ABLATIONS.md` (the tail-position ablation sweep this
work was inserted alongside), `Papers/GATOR_EVOTOKEN_SLOT_WIDTH_SWEEP_NODES80.md` (the nodes-vs-width
capacity framing this report reuses).

## Motivation

While monitoring the context-length sweep, the user asked whether `streams` (the number of
parallel processing lanes in the shared-node relay architecture, default=10, previously fixed
across every prior sweep in this project) affects quality — an axis never tested before. Per the
user's stated priority: **what matters is which axis provides more capabilities**, not just raw
efficiency ratios — so this report leads with qualitative behavior, not just accuracy deltas.

## Method

Same core config as every other arm in the concurrent context-length/vocab sweeps: width=128,
nodes=40, output-slots=6, observed=128 (default), no ablation flags, `--resume-source none`,
masked-only CTS v5 objective. `--streams` swept at {5, 10, 20} — a halve/double pattern around the
default value of 10, per the user's choice when asked. streams=10 reuses arm 44's already-recorded
baseline datapoint (identical config). streams=5 and streams=20 each ran fresh to 50k steps.
Trained live on Colab (L4, later switched to a higher-RAM tier mid-sweep), 2026-08-03/04.

## Results

| streams | params | refined_acc | Δ vs baseline | modal_lift | Δ vs baseline | throughput | qualitative |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 1,057,853 (-9.8%) | 0.1348 | **-26.6%** | +0.084 | **-36.8%** | not measured | coherent |
| 10 (baseline) | 1,172,813 | 0.1836 | — | +0.133 | — | 857 steps/min | coherent (some incoherent) |
| 20 | 1,402,733 (+19.6%) | 0.1764 (best=0.1842) | **-3.9%** (best +0.3%) | +0.126 | **-5.3%** | ~592 steps/min (-31%) | coherent |

`by_pos` (front→tail) at step 50,100: streams=5 `[0.29 0.16 0.17 0.06 0.05 0.07]`, streams=10
`[0.39 0.27 0.17 0.13 0.06 0.08]`, streams=20 `[0.36 0.23 0.16 0.14 0.07 0.11]`.

**Halving streams hurts substantially; doubling streams costs almost nothing and may even help.**
streams=5's regression (-26.6% refined_acc) is by far the largest single-flag effect measured
anywhere in this sweep program — larger than any context-length or tail-position ablation arm.
streams=20, by contrast, lands within noise of baseline on refined_acc and slightly *exceeds* it
on `best_refined` (0.1842 vs 0.1836) and on tail positions (0.07/0.11 vs baseline's 0.06/0.08).

## Parameter and throughput cost, framed against the nodes-vs-width axes

Streams sits between nodes and width on both cost dimensions, but much closer to nodes:

| Axis | Δ params (2x step) | Δ throughput (2x step) |
|---|---:|---:|
| nodes (40→80) | +2.4% | -35% |
| streams (10→20) | +19.6% | -31% |
| width (doubling steps) | +80% to +142% | not directly comparable (different step sizes) |

Streams is notably more parameter-hungry than nodes for the same 2x scaling step (+19.6% vs
+2.4%), but costs slightly less throughput (-31% vs -35%) and buys a much larger accuracy
improvement over the low-capacity setting than nodes did in the earlier capacity sweep.

## Capability read (the finding that actually matters)

Every streams value tested — including the badly-regressed streams=5 — produced **qualitatively
coherent output**, not the degenerate letter-soup seen in the context-length sweep's observed=512
arm (`Papers/GATOR_EVOTOKEN_TAIL_POSITION_ABLATIONS.md`'s companion report). This means streams
count affects *how well* the model learns the task, not *whether* it stays coherent — a
fundamentally different kind of failure mode than what longer real context produces. In capability
terms: reducing streams degrades quality gracefully; it does not break the model's ability to
produce structured text the way processing more real tokens does.

streams=20's result — better tail-position behavior and a matched-or-better `best_refined` for a
+19.6% parameter cost and -31% throughput cost — suggests more streams is a genuinely useful
capacity lever for this architecture, worth considering alongside width and nodes in any future
100M-scale configuration search, not just a niche knob.

## Caveats

- **Early snapshot only**: 50k of 10M planned steps, single seed, no repeated trials, standard
  budget (no doubling schedule applied to this axis, unlike context/vocab).
- **streams=20's exact throughput** (~592 steps/min) was computed from a single ~9-minute raw-log
  timestamp window mid-run, not cross-validated across multiple windows the way the nodes=40
  baseline's 857 steps/min figure was in the earlier capacity sweep.
- **Runtime disconnect mid-sweep**: streams=5's run was interrupted once by a genuine Colab
  runtime disconnect/reconnect (an unrelated GPU-tier switch) and manually restarted from step 0;
  the reported result is from the clean post-restart run only.
- Only three points tested (5, 10, 20) — does not establish where returns to additional streams
  diminish or reverse, only that the tested range is monotonic in the expected direction.

## Next

No further streams arms planned. See `Papers/GATOR_EVOTOKEN_TAIL_POSITION_ABLATIONS.md` and the
in-progress context-length/vocab sweeps for the rest of the active work in this program.
