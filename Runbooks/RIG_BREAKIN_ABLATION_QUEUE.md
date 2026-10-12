# New-rig break-in ablation queue

**Context:** new local rig (RTX 5080, Ryzen 9 9950X, 32GB DDR5, Gen 5 SSD)
coming online 2026-08-11, intended for fast local iteration with Colab as
heavy-VRAM fallback. Before committing it to the long 100M SmolLM3-recipe
migration (`gtr_arw_100m_smollm3_cell8_26b`, see
`Runbooks/GTR_RUN_REGISTRY.md`), the plan is a handful of speed
tests / ablations to break the machine in. This doc exists specifically so
that list survives a session/machine migration — write results back into
`Runbooks/GTR_RUN_REGISTRY.md` as each one runs, this file is the
queue, not the results log.

## Priority 1 — directly feeds the active SmolLM3-reproduction question

### 1. Routing-capacity bisection
`--routing-dim 8` (10k reference config, `gator_cells_vectorized.py`) landed
at `acc=0.622`/`exact=0.030` vs. the unbottlenecked reference's
`0.956`/`0.747` — a severe regression, but never bisected toward the floor.
Router's natural output is `n_nodes*n_nodes=16` edges; its unbottlenecked
input is `2*d_model+n_nodes=20`. Try `--routing-dim 12` and
`--routing-dim 16` next (below and at the output-space size) to see where
the floor actually sits between 8 (fails) and 20/unbottlenecked (works).

```bash
python gator_cells_vectorized.py --steps 20000 --seed 0 --routing-dim 12 \
  --adamw-weight-trace-every 200 \
  --adamw-weight-trace-dir <checkpoint>/adamw_weight_trace
python gator_cells_vectorized.py --steps 20000 --seed 0 --routing-dim 16 \
  --adamw-weight-trace-every 200 \
  --adamw-weight-trace-dir <checkpoint>/adamw_weight_trace
```

### 2. Seed variance at `d_model=26` / `d_model=32`
One seed each showed a huge swing — `d_model=26` seed=0 never broke out
(`acc=0.653`/`exact=0.009`, still climbing at step 20k) while seed=1 nearly
mastered it (`acc=0.976`/`exact=0.851`). `d_model=32` seed=0 landed in
between (`acc=0.907`/`exact=0.566`, noisier convergence). Never ran enough
seeds to know if this is typical variance or a fluke — run seeds 2-4 at
each width to actually characterize it.

```bash
for seed in 2 3 4; do
  python gator_cells_vectorized.py --steps 20000 --seed $seed --d-model 26 \
    --adamw-weight-trace-every 200 \
    --adamw-weight-trace-dir <checkpoint>/adamw_weight_trace
  python gator_cells_vectorized.py --steps 20000 --seed $seed --d-model 32 \
    --adamw-weight-trace-every 200 \
    --adamw-weight-trace-dir <checkpoint>/adamw_weight_trace
done
```

### 3. `linear_scan` vs `convolution`, context-length isolation
The VRAM win for `linear_scan` is currently one anecdote (6.4GB at
context=4096/batch=1 on the 100M SmolLM3 config) compared against a
different config's 11GB/8GB numbers (1.27M params, context=512,
`convolution` backend) — never actually controlled. With a local GPU, sweep
context length at fixed batch/params/backend, then compare backends at
matched context, using `gator_language_train.py --architecture gator`.
Suggested: small-ish model (~5-20M params for fast iteration), context in
{512, 1024, 2048, 4096}, `--receiver-memory-backend` in
{convolution, linear_scan}, batch fixed. Record peak VRAM and steps/sec at
each point — this is the test that actually answers "is it context² that's
different, and by how much."

### 4. `linear_scan` at small-scale language modeling
Only tested at two extremes: the 10k-param synthetic prefix task (failed
outright, flat at chance) and the 100M SmolLM3 run (looking okay so far, but
early). Nothing in between. A small local `GatorLanguageModel` run (e.g.
`n_cells=4`, `d_model` small, quick to iterate) comparing
`receiver_memory_backend={convolution, linear_scan}` at matched params would
tell you faster than waiting out the 100M run whether the "free-text
formation doesn't need `linear_scan`'s missing oscillating relative-position
structure" hypothesis actually holds, or whether the 100M run's early loss
descent is a red herring that catches up with it later (tie this to the
already-flagged destabilization watch window if useful — same question,
cheaper venue).

## Priority 2 — older pending items, never executed, still real gaps

### 5. GTR-L context-length sweep completion
`observed={512, 1024, 1536}` already done; `observed=2048` (100k steps)
never run. Compile the report across all four once done.

### 6. GTR-L vocab sweep
`vocab=2048` baseline exists; `vocab={4096, 8192, 16384}` (50k steps each)
never run. Compile the report across all four once done.

### 7. Stalled Procoim threads
- Local hybrid run: directional-Adam + rotating past-references, slots=2,
  100k steps.
- Diagnostic B: `b128` (basic Procoim, no extra mechanisms) at length=2,
  100k steps.
- Consolidation-gate on/off A/B at length=7 (the gate itself and its unit
  test are done — `gator/optimizers/procoim.py` — the actual A/B run was
  never launched).

## Notes

- Priority 1 items are ordered roughly by cheapness/speed, not necessarily
  importance — 1 and 2 reuse `gator_cells_vectorized.py`'s existing 10k/48k
  reference configs (fast, CPU-provable already), 3 and 4 need a real GPU
  and are the actual point of "break in the new rig."
- Log every result into `Runbooks/GTR_RUN_REGISTRY.md` as usual,
  same table format as the existing GTR-ARW rows.
- None of this blocks or competes with the 100M SmolLM3 migration — the plan
  discussed was to break the rig in with these first, then let the 100M run
  take over once it's confirmed stable.
