# Gator Config How-To: Sizing, Ratios, and Training Flags

This is a practical reference for building a new Gator config, distilled from the
plateau-instability study (`GATOR_PLATEAU_INSTABILITY_STUDY.md`). It does not re-derive the
evidence — see that doc for the actual experiments and numbers. This doc is just the recipe.

## 1. Sizing procedure (in order)

1. **Pick a total parameter budget and a `vocab_size`.**
2. **Size `embedding_dim` against `vocab_size`, not `d_model`.** This is the single most
   important correction from the study: an earlier sweep at `vocab=2048` found embedding_dim
   "flat" against quality, which was an artifact of the vocab being too small to expose the
   effect. At `vocab=20,000`, embedding_dim shows a real, monotonic improvement from 39→200
   (PPL 217→154, a knee — the range past 200 up to 1200 keeps improving but very slowly:
   154→150→148). Rule of thumb until a dedicated vocab-sweep pins the exact curve down:
   `embedding_dim ≈ 0.01 × vocab_size`, e.g. ~200 at vocab=20,000. Compute its exact parameter
   cost: `vocab_size × embedding_dim` (embedding table) + `2 × lanes × d_model × embedding_dim`
   (input + output projections).
3. **Subtract that cost from the total budget** — what's left is the "core" budget for the
   graph/routing machinery (`d_model`, `nodes`, everything else).
4. **Allocate the core budget using the validated ratios:**
   - `nodes / d_model ≈ 1.7` (the one ratio with a sharp, validated optimum — don't stray far
     from this; both directions are worse).
   - `graph_degree = 4` — tested wider (8, modestly worse) and narrower (2, invalid — the
     `small_world` connection layout requires `degree ≥ 3`). Stay at the baseline.
   - `relays = 2` — even relay counts work, odd counts are bad, returns diminish past 2. Fixed,
     don't sweep this.
   - `lanes`: a genuinely **free** lever. 5 and 10 tie on quality; 20 is a clear loser (worse
     quality *and* ~36% slower, ~2x memory). Shrink lanes as far as tolerable — it costs
     nothing to go small.
   - `active_node_top_k`: the one axis with a **real quality/compute tradeoff** (not free).
     Wider consistently helps quality (monotonic 7→14→28 in one sweep), but costs real
     per-step compute and — critically — real memory, and can force a smaller physical batch
     at scale (which then eats more from gradient-accumulation overhead). Size this to
     whatever compute/memory budget you actually have, not to whatever the small-scale sweep
     said was "best." At production model sizes (~100M), top_k in the 30s+ range has already
     been observed to OOM even at `batch_size=1`; something in the mid-teens (`~14`) is a
     safer starting point until you've confirmed a specific budget can go wider.

**Composition warning:** independently-optimal single-axis values do **not** compose cleanly.
A 50M-scale test that combined several "winning" ratios (node/width 1.697, embedding_dim ratio
0.25×, top_k ratio 0.105) performed dramatically worse than a simpler baseline config at the
same scale — likely synergistic/parasitic interaction between axes that single-variable
sweeps can't see. Always validate a combined config as a whole; don't assume ratios stack.

## 2. Config flags — what to set and why

- `--route-backend dense` — the packed backend was tested and shows no speed advantage at
  production scale (the `bmm`-equivalent cost in `_node_memory` dominates identically in both
  backends), so it adds complexity for no benefit. Use `dense`.
- `--terminal-vote-mode prompt` — now the default in the trainer and bundle scripts (was
  `global`/`slot` previously). `slot` mode's design rationale (differentiating simultaneously
  formed output positions) does not apply under `--decode-mode serial`, which only ever forms
  one output slot per training step. A head-to-head at 100M scale confirmed `prompt` is
  equal-or-better at every matched step count and never showed `slot`'s large loss spikes.
- `--decode-mode serial` — eliminates the parallel-decode plateau/regression instability
  entirely (root cause: independent, simultaneous per-slot prediction under constant LR is a
  property of the *objective*, not the architecture). Don't use parallel decode-mode.
- `--compile` — **required at ~100M scale, not optional.** A 100M config that OOM'd reliably
  in eager mode (`_node_memory`'s matmul allocating ~15 GiB with ~86-90 GiB already resident)
  trained cleanly once `--compile` was added, at the identical architecture and batch size.
  Compile's kernel fusion evidently avoids materializing the large intermediate tensor eager
  mode allocates in one shot. Also gives a real speed win independent of the memory fix.
  Quantified at the 1.5M-param shape (`d_model=80 nodes=95 relays=2 router_rank=96
  top_k=32`, context=620, L4, 2026-07-29): peak allocated **18.55 GB → 14.88 GB (~20%
  less)** and steady-state throughput **0.99 → 1.43 steps/sec (~44% faster)**.
  **Measure steady-state, not a short run**: compile's one-time TorchInductor warmup made
  the same config look 25% *slower* over 60 steps (0.65 vs 0.87 steps/sec cumulative); the
  per-log cumulative rate only crossed eager's around step ~160 and was still climbing at
  step 400. Use a few hundred steps and the incremental rate between the last two logs.
  **Compile is config-shape-sensitive — smoke-test it before trusting it.** The same flag
  NaN'd out every gradient at optimizer step 14 on a ~10M ablation shape (`d_model=128
  nodes=60 relays=1 router_rank=16 top_k=8`) at context=512, i.e. at the *smallest* context
  tried, so it was the architecture shape and not sequence length. Unconfirmed suspicion:
  `relays=1` skips `_receiver_kernel`/`_node_memory` entirely on the sole relay
  (`bootstrap_context_mode="self"`), so a compiled `relays=1` graph is a genuinely different
  fusion problem than `relays>=2`; the very low `router_rank=16` may also underflow once
  Inductor reorders the routing softmax. Run ~20 steps eager-vs-compiled on any new
  architecture shape and check for non-finite gradients before committing to a long run.
- `--mixed-precision bf16` — gives both a real speed win (~1.5-2x, confirmed at both 50M and
  100M scale) **and** a real memory win (~75 GiB → ~45 GiB peak at 100M scale, same config).
  Treat this as a second, independent OOM mitigation alongside `--compile`, not merely a
  throughput optimization — enable it by default at production scale.
- `--gradient-accumulation-steps N` (paired with `--batch-size`): use this to hit a target
  effective batch size when memory won't allow the physical batch to get there directly.
  Important caveat: **accumulation is strictly slower than an equivalent true larger physical
  batch** — it trades speed for memory, it does not parallelize. Only reach for it when memory
  actually forces a smaller physical batch; don't use it by default.

## 3. Batch size / effective batch size

A dedicated sweep at 10M/vocab=20,000/context=512 scale found a clear threshold:
- Effective batch 4 and 8 both plateau (never break out of a ~140-330 PPL band within the
  tested window).
- Effective batch 16 breaks into continuous descent and never plateaus in the same window.
- Effective batch 32 ties effective batch 16 (192.21 vs 180.94) at 2x the compute cost — not
  worth it.

**Target effective batch ≥ 16.** This is the practical minimum for getting real learning
signal through past the small-batch plateau; below it, more steps alone will not help. Above
it (32+), you're paying for compute without a proportional quality return.

## 4. Quick reference table

| Parameter | Guidance |
| --- | --- |
| `embedding_dim` | `≈ 0.01 × vocab_size` (a first-pass estimate; refine with a dedicated vocab sweep before locking in) |
| `nodes / d_model` | `≈ 1.7` (sharp optimum, validated) |
| `graph_degree` | `4` (baseline; don't go below 3, don't go much above 4) |
| `relays` | `2` (fixed) |
| `lanes` | as small as tolerable (5-10; free lever, no quality cost seen down to 5) |
| `active_node_top_k` | real quality/compute tradeoff; size to actual memory budget, not to a small-scale "winner" |
| `route_backend` | `dense` |
| `terminal_vote_mode` | `prompt` (now the trainer default) |
| `decode_mode` | `serial` |
| effective batch size | `≥ 16` |
| `--compile` | required at ~100M+ scale (memory, not just speed) |
| `--mixed-precision bf16` | enable by default at production scale (memory + speed) |
