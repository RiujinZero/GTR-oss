# Gator-100M serial-direct Colab bundle

Upload `gator100m_serial_direct.zip` to:

```text
/content/drive/MyDrive/GTR/colab_bundles/gator100m_serial_direct.zip
```

Paste the complete contents of `GATOR100M_SERIAL_DIRECT_LAUNCH_CELL.py` into one
browser-started Colab cell. Edit the `PATHS` block at the top if the bundle,
checkpoint root, or any dataset is elsewhere.

## Why this exists

`run_gator_100m_staged_colab.py` (used by `gator100m_staged_colab.zip` /
`GATOR100M_COLAB_LAUNCH_CELL.py`) hardcodes
`state["fixed_route_temperature"] = 1.25` at its Phase A→B→C transitions,
keeping route temperature permanently hot instead of cooling it — a known,
still-unfixed bug (see `Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md` section
1.1). It also uses the multi-jury/staged-optimizer/teacher-handoff curriculum
that the 2026-07-23 plateau-instability study found to be a severity
amplifier on top of the base training instability.

This bundle sidesteps both by calling `gator_language_train.py` **directly**,
with no staged-phase wrapper:

- No hardcoded temperature — `gator_language_train.py`'s own
  `route_temperature_schedule=legacy` cooling (1.25 → 0.55 → 0.35) runs
  automatically every invocation, the same schedule visible in every log in
  this study.
- `--decode-mode serial` — the study's clean fix. At small scale (35k-52k
  params, 24,000 steps) this eliminated the improve-then-regress pattern
  entirely; see sections 6.5 and 7 of the study doc.
- `--optimizer-groups monolithic` instead of `component_scaled` — the
  per-component LR multipliers exist to support the staged Phase A/B/C
  freeze/rewind curriculum, which this script does not implement.

Same 100M architecture as `run_gator_100m_curriculum_colab.py`
(`D_MODEL=368, N_NODES=626` — ≈1.7 nodes/width, `EMBEDDING_DIM=288,
ROUTER_RANK=96, LANES=10, DEGREE=4, RELAYS=2, ACTIVE_NODE_TOP_K=32,
VOCAB_SIZE=32768`, `~99.98M` params) and the same full multi-corpus
interleaved data mixture (SmolTalk Everyday + TinyStories + LMSYS Chat 1M +
Tulu3 SFT + SmolTalk Preference, same weights/limits as the staged bundle's
`_common_command`), targeting 800,000 steps.

## Caveat — this is untested at 100M scale

Everything the "serial decode fixes it" finding rests on was run at
35k-52k params, 24,000 steps, on a 50k-example subset of one corpus. It has
**not** been validated at 100M params, 800,000 steps, on the full
multi-corpus mixture. `gator_language_train.py`'s own `serial_next_token_batch`
mechanics and route-temperature cooling are architecture/scale-independent
code paths, so there's no known reason this shouldn't work the same way, but
treat the first several hundred thousand steps as a live test of that
extrapolation, not an assumed result — watch `formed=`/`target=` samples and
`content_ppl`, not just raw PPL, per the study's own repeated finding that
the two can diverge.

## Runtime model

`gator_language_train.py --steps N` runs **N steps this invocation**, added
on top of whatever step the resumed checkpoint recorded — it is not an
absolute target. The launch cell loops: read the current step from
`latest.pt` (0 if none exists), run one `SEGMENT_STEPS`-step (default 10,000)
foreground invocation with `--resume-checkpoint` once a checkpoint exists,
sync checkpoints/logs to Drive periodically, and repeat until the cumulative
step count reaches `TARGET_STEPS` (800,000) or the Colab runtime disconnects
(in which case, rerun the same cell — it picks up from the last synced
checkpoint on Drive; copy it back into `LOCAL_CHECKPOINT_DIR` first if the
local Colab work directory was lost).
