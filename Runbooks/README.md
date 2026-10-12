# Runbooks

Known-good sample configs and ready-to-paste Colab launch cells for the two
architectures `gator_language_train.py` supports. Check here before
reconstructing a launch config from scratch or from an old bundle README.

- **`CONVERGENT_RUNBOOK.md`** — `--architecture convergent`, the currently
  running 100M-param production config (3.5M-step target).
- **`ARROW_RUNBOOK.md`** — `--architecture gator`, the cell-ring architecture,
  4-cell/100M config validated in the 2026-07-26 scaling session.
- **`colab_cells/`** — matching paste-into-Colab launch scripts for each.
  Both use the same `gator_colab_bundle.zip` (built by
  `prepare_gator_colab_bundle.py`) — rebuild and re-upload it after any
  trainer/model code change before relying on these cells.

Both runbooks link back to the underlying research docs
(`Papers/GATOR_LANGUAGE_EXPERIMENT.md`, `Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md`)
for the full reasoning — these files are meant to be the fast/practical
reference, not a replacement for those.
