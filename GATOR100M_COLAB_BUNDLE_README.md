# Gator-100M staged Colab bundle

Upload `gator100m_staged_colab.zip` to:

```text
/content/drive/MyDrive/GTR/colab_bundles/gator100m_staged_colab.zip
```

Paste the complete contents of `GATOR100M_COLAB_LAUNCH_CELL.py` into one
browser-started Colab cell. Edit the path block at the top if the bundle,
checkpoint root, or any dataset is elsewhere. `BATCH_SIZE` and
`LOG_EVERY_STEPS` are exposed in the same block. The default step-based report
interval is 10,000 steps, and every child-process log line is streamed directly
to the active Colab cell as well as the local and Drive logs. The launch loop
passes both settings explicitly as `--batch-size` and
`--log-every-steps` on every segment invocation.

`SLOT_ISOLATION_SLOTS` is also exposed and passed explicitly. Its default is
`0`, meaning all 626 nodes are shared by all output slots. Setting it to `128`
creates balanced private-node ownership across the 128 response slots. Treat
that as a new architecture treatment: choose a new `run_name` and start a fresh
run rather than resuming a shared-node checkpoint.

The launch block also exposes the handoff gates:

```text
HANDOFF_MIN_TEACHER_ACCURACY = 0.80
HANDOFF_MIN_FRONTIER_ACCURACY = 0.80
HANDOFF_ADVANCE_RATIO = 0.95
```

Decode behavior is also exposed in the launch cell. Leave `DECODE_MODE` unset
for the staged defaults, set it to `"serial"` for a serial baseline, or set
the `PHASE_A_DECODE_MODE`, `PHASE_B_DECODE_MODE`, and
`PHASE_C_DECODE_MODE` values independently. Serial mode requires the
`slot_ce` formation objective.

The launch cell also exposes retreat/advance patience, retreat ratio, probe
fraction, unforced-loss weight, loss normalization, and policy-override
controls for the token handoff.

These may be changed before restarting a stalled Colab invocation. The trainer
preserves the checkpoint's released-token count, accuracy EMA, patience
streaks, optimizer moments, and exact data cursor while explicitly replacing
the gate policy with these command-line values.

The runner is resumable and repeatedly executes 10,000-step foreground
segments:

1. Co-adaptation: all component groups train with full teacher forcing. Every
   configured corpus is sampled through the same weighted interleaved loader.
2. Route formation: rewind to the best co-adaptation checkpoint, freeze
   lexical/lane/private-node parameters, and train route transport plus jury
   under full teacher forcing with the same interleaved material.
3. Staged handoff: rewind to the best route checkpoint, freeze every component
   except private nodes, hold route temperature fixed, and start with zero
   additional teacher tokens removed. The first removal requires at least 80%
   teacher frontier accuracy, 80% unsupported frontier accuracy, and the 0.95
   unsupported/teacher ratio for three evaluations.

The plateau rule is less than 1% teacher-content-PPL improvement after at least
two segments. Safety caps are configurable. A discarded plateau probe is kept
separately and does not replace the retained best checkpoint.

Drive behavior:

- pending logs append every 900 seconds;
- `latest.pt` atomically replaces the Drive copy every 3,600 seconds when a
  newer trainer checkpoint exists;
- the phase state, best rewind checkpoints, last run spec, optimizer state,
  data state, and handoff-controller state persist in the run directory;
- the same launch cell resumes automatically after a Colab reconnect.

The initial bundle version marked generic streaming sources as non-exact. When
resuming one of those checkpoints, `resume_data_state=auto` preserves model and
optimizer state but deterministically resets the mixed-corpus cursor once.
This bundle checkpoints each stream's encoded shuffle buffer, RNG, epoch
position, and raw cursor. Checkpoints written after upgrading are therefore
exactly restorable; `auto` restores them exactly and only falls back when
reading a legacy non-exact checkpoint.

The fixed architecture is:

```text
V32768 / E288 / D368 / N626 / L10 / K4 / R2
shared-low-rank router rank 96
learned ingress, active top-k 32
99,980,100 parameters
```
