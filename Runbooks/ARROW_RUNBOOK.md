# Arrow (cell-ring Gator) runbook

`--architecture gator`: the cell-ring architecture (`gator/models/language.py`'s
`GatorLanguageModel`). Historically shown to resist the degenerate whitespace/
constant-token collapse that `--architecture convergent` exhibits under the
same objectives — see `Papers/GATOR_LANGUAGE_EXPERIMENT.md`'s synthesis
section for the direct A/B.

This doc tracks the config validated in the 2026-07-26 scaling/optimization
session (vectorization confirmation, bf16 fix, cell-count sweep at 100M
scale). It has **not** run anywhere near as long as the convergent 100M
production run — treat step counts beyond ~2000 as unvalidated territory.

## Known findings that shape this config

- The forward pass is **already fully vectorized** across cells (batched
  `bmm`/`einsum` in `gator/models/dense_kernels.py`, no Python loop) —
  `gator_cells.py`/`gator_cells_vectorized.py` are disconnected legacy
  prototypes, not what `--architecture gator` actually runs.
- **bf16 requires the fix landed 2026-07-26** (commit `29de4fa` +
  `5f1808c`): `relative_node_bias` dtype casts at 4 call sites, and the
  active-fraction ratio computed in float32 instead of the autocast dtype
  (bf16's 8-bit mantissa let count-ratio rounding drift outside `[0,1]` and
  crash `StewardshipExecutionController.observe()` partway through a
  longer run). Without both fixes, `--mixed-precision bf16` either crashes
  immediately or crashes after enough steps for the ratio-rounding bug to
  trip.
- `--decode-mode chunk_parallel` is **not wired up** for this architecture —
  the trainer rejects it outright. Use `parallel` or `serial`.
- `StewardshipExecutionController`'s `packed_auto` mode is a mid-training
  optimization only — it cannot rescue a config that OOMs on its very first
  dense forward pass. If you OOM at step 0, packed execution won't help;
  reduce cell count or width instead.
- **Cell count is the dominant memory-cost driver, not raw parameter
  count** — every activation tensor scales as `[batch, cells, tokens, ...]`,
  so cell count multiplies memory directly, while `d_model` mostly costs
  parameters/FLOPs, not activation memory. A controlled 4-point sweep at
  ~100M params (400/40/10/4 cells, widening `d_model` to compensate) showed
  BOTH memory and steps/sec improving monotonically as cell count decreased,
  all the way down to 4 cells — the practical minimum under
  `hybrid_hemispheric` topology (requires an even cell count). This
  corrected an earlier premature conclusion (drawn from only 2 data points)
  that fewer/bigger cells trades memory for speed.
- **15-step smoke tests are compile-time-dominated and understate steady-
  state throughput by ~10x.** The 4-cell/100M config measured 0.14 steps/sec
  at 15 steps but ~1.3-1.4 steps/sec by step 1000-1400 of a longer run —
  `torch.compile`'s one-time cost (60-100+ seconds at this scale) dominates
  a lifetime average over only 15 steps. Always run at least several hundred
  steps before trusting a throughput number.
- `--activation-checkpointing` (new flag, opt-in) checkpoints the dense
  relay's router/node computation — genuinely reduces memory where applied,
  but a second expensive segment (`_dense_stewardship_context`'s receiver
  memory) is not yet checkpointed, so it isn't sufficient alone to fit very
  high cell counts (e.g. 400 cells at 100M params still OOMs even with it on).

## Architecture: 4-cell / 100M config (best of the tested cell-count sweep)

| Field | Value |
|---|---|
| `vocab_size` | 20000 |
| `context_tokens` / `response_tokens` | 512 / 128 |
| `d_model` | 2432 |
| `embedding_dim` | 200 |
| `n_cells` | 4 (practical minimum under `hybrid_hemispheric`) |
| `relays` | 4 |
| `neighbors` | 2 |
| `node_topology` | `hybrid_hemispheric` |
| `lexical_mode` | `communal` |
| `cell_input_mode` | `feature_gated` |
| `cell_output_mode` | `shared_pool` |
| `cell_transport_mode` | `stewardship` |
| `formation_objective` | `slot_ce` |
| `decode_mode` | `parallel` |
| Params | ~100.2M |

Stewardship settings: `stewardship-min-active 2`, `stewardship-max-active 0`
(unbounded), `stewardship-keep-mass 0.9`, `stewardship-handoff-keep-mass 0.9`,
`stewardship-exploration 0.01`, `stewardship-scouts 1`,
`stewardship-execution packed_auto`.

Measured at this config (batch 4, bf16, `--compile`, single G4): ~1.3-1.4
steps/sec steady-state, ~18.9GiB/34.2GiB memory footprint at the 15-step
smoke-test point (steady-state footprint not separately re-measured after
the longer run's crash-and-fix).

**Caution on learning rate / schedule:** `--learning-rate 3e-3 --scheduler
constant` ran cleanly to step 1400 of a 2000-step test but that was the same
run that hit the bf16 active-fraction crash (now fixed at the numerics
level — the crash was a precision bug, not proof the LR/schedule itself is
safe long-term). Nothing in this architecture has tested an LR schedule
either; treat 3e-3/constant as unvalidated beyond ~1400-2000 steps.

## Historical production config (62-cell, smaller scale — for reference)

Recovered from checkpoint archaeology this session: `n_cells=62,
neighbors=4`, `formation_objective=monotonic`, `response_tokens=40`,
`reference_tokens=32`, `node_topology=hybrid_hemispheric`,
`lexical_mode=communal`, `cell_input_mode=feature_gated`,
`cell_output_mode=grouped_concat`, tokenizer `gator-compact-v2`, blank token
ID 2047, `stewardship_keep_mass=0.9`, `stewardship_exploration=0.01`,
`stewardship_execution=packed_auto`, `alignment_max_shift=8`,
`alignment_warmup=0.05`, `sequence_quality_penalty=0.01`. This is the
"cells work but nobody characterized the shape/count scaling" baseline that
motivated the 2026-07-26 sweep — don't copy it for new 100M-scale runs
without re-checking against the sweep's findings first.

## Colab cell

See `Runbooks/colab_cells/arrow_100m_launch_cell.py`. Same bundle
(`gator_colab_bundle.zip`) as the convergent runbook — both architectures
live in the same trainer (`gator_language_train.py --architecture gator`
vs `convergent`). This is a short smoke/validation cell (a few thousand
steps), not a long unattended production loop like the convergent one —
Arrow hasn't earned that level of trust yet at this scale.

## Open items

- Extend `--activation-checkpointing` to `_dense_stewardship_context`'s
  receiver-memory computation (the second bottleneck).
- Find where the "fewer cells" trend reverses below 4 cells (try non-hybrid
  topologies to go lower) or at larger context/batch.
- Wire up `--decode-mode chunk_parallel` for this architecture.
- Try an LR schedule instead of constant (same open item as convergent).
- Implement the `router_weight`-sharing-across-cells proposal
  (`Papers/GATOR_LANGUAGE_EXPERIMENT.md`'s design doc) and A/B it.
- Run this config long enough (thousands of steps, not 2000) to get a real
  read on whether it collapses, plateaus, or keeps improving — nothing
  Arrow-specific has run anywhere near as long as the convergent production
  run.
