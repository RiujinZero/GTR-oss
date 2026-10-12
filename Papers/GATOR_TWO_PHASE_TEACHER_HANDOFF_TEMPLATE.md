# Gator Two-Phase Teacher-to-Formation Training Template

**Status:** Reusable experimental template, updated from the optimizer and scaling ablations  
**Primary example:** Gator-100M, 800,000 total optimizer steps  
**Training objective:** Establish a causal language baseline under full teacher forcing, then progressively remove shifted response tokens until the model can form responses without answer-prefix support.

## 1. Purpose

Use this template when a new Gator configuration must learn language structure and unsupported simultaneous formation in the same training program.

The program deliberately separates two problems:

1. **Causal language acquisition:** learn to predict every response position with the correct shifted response prefix available.
2. **Teacher removal:** retain that capability while answer-prefix tokens are removed one at a time.

This prevents a fresh model from being asked to learn basic language statistics and missing-token recovery from the first optimizer step. It also creates an interpretable teacher-forced checkpoint before the formation curriculum begins.

Slot zero remains intrinsically unsupported in teacher-forced mode because it has no previous response token.

## 2. Template variables

Record these values before launch.

| Variable | Meaning | Gator-100M example |
|---|---|---:|
| `TOTAL_STEPS` | Complete experimental horizon | 800,000 |
| `TF_FRACTION` | Minimum teacher-forced share | 0.20 |
| `TF_STEPS` | `TOTAL_STEPS * TF_FRACTION` | 160,000 |
| `HANDOFF_STEPS` | Remaining curriculum horizon | 640,000 |
| `RESPONSE_SLOTS` | Fixed response length | 128 |
| `MAX_REMOVED` | Eligible shifted teachers | 127 |
| `LOG_EVERY` | Validation/controller interval | 1,000 |
| `ADVANCE_RATIO` | Frontier accuracy / teacher accuracy | 0.95 |
| `RETREAT_RATIO` | Ratio that triggers support restoration | 0.75 |
| `MIN_TEACHER_ACC` | Minimum meaningful teacher frontier | 0.15 |
| `ADVANCE_PATIENCE` | Successful evaluations per release | 3 |
| `RETREAT_PATIENCE` | Failed evaluations per retreat | 2 |
| `FULL_PROBE_FRACTION` | Fully unsupported training probes | 0.02 |
| `UNFORCED_LOSS_WEIGHT` | Additional loss on unsupported slots | 2.0 |
| `HANDOFF_LOSS_NORMALIZATION` | Scale free loss by active unsupported slots | `active_slots` |
| `TRANSFER_NODE_LR_MULT` | Private-node multiplier during autonomy transfer | 1.25 |
| `STABILIZE_NODE_LR_MULT` | Private-node multiplier after transfer plateaus | 0.80 |
| `OTHER_GROUP_LR_MULT` | Lexical, lane/core, and route/gate multiplier | 1.00 |
| `ROUTE_TEMPERATURES` | Staged exploration / cooling schedule | 1.25 / 0.55 / 0.35 |

The phase boundary is a minimum step, not an unconditional release command. Phase 1 should continue if its exit gates are not met.

## 3. Phase 1: full teacher-forced baseline

### Objective

Train a competent causal Gator before removing any eligible shifted response token.

### Required settings

```text
decode_mode=teacher_forced
steps=TF_STEPS
log_every=LOG_EVERY
```

Keep architecture, tokenizer, corpus mixture, sampler seed, batch shape, optimizer family, and validation manifest frozen for the full experiment.

### Minimum exit gates

Do not enter token handoff until all gates hold:

1. At least `TF_STEPS` updates have completed.
2. Teacher-forced content accuracy is at least `MIN_TEACHER_ACC`.
3. Validation PPL is finite and no longer falling rapidly.
4. Gradients and optimizer updates remain finite.
5. The tokenizer and validation fingerprint match the frozen run specification.
6. A restorable checkpoint exists with model, optimizer, sampler, and queued-data state.

Recommended plateau test: compare the mean PPL of the latest five evaluations with the preceding five. Treat an improvement smaller than 1% as a plateau signal, not as a mandatory statistical test.

### Required checkpoint

Preserve the transition checkpoint separately:

```text
phase1_teacher_step_<TF_STEPS>.pt
```

Never overwrite this checkpoint with a later handoff checkpoint. It is the reusable causal baseline and rollback point.

## 4. Phase 2: tokenwise teacher handoff

### Objective

Resume the trained causal model and remove exactly one additional shifted teacher token only after the current unsupported frontier performs nearly as well as the teacher-forced control at the same addresses.

### Required settings

```text
decode_mode=token_handoff
steps=HANDOFF_STEPS
token_handoff_start_removed=1
token_handoff_max_removed=0
token_handoff_advance_ratio=0.95
token_handoff_retreat_ratio=0.75
token_handoff_min_teacher_accuracy=0.15
token_handoff_advance_patience=3
token_handoff_retreat_patience=2
token_handoff_full_probe_fraction=0.02
token_handoff_unforced_loss_weight=2.0
token_handoff_loss_normalization=active_slots
route_temperature_schedule=staged_reheat
route_temperature_start=1.25
route_temperature_middle=0.55
route_temperature_end=0.35
optimizer_groups=component_scaled
optimizer_private_node_lr_multiplier=1.25
optimizer_lexical_lr_multiplier=1.0
optimizer_lane_core_lr_multiplier=1.0
optimizer_route_gate_lr_multiplier=1.0
```

`token_handoff_max_removed=0` means all `RESPONSE_SLOTS - 1` eligible teachers may eventually be removed.

Do not start at zero removals. With no newly unsupported frontier addresses, the controller lacks a meaningful frontier/teacher comparison from which to advance. Phase 1 supplies the zero-removal baseline; Phase 2 begins at one.

### Controller rule

At each evaluation:

```text
ratio = unsupported accuracy at removed addresses
        / teacher-forced accuracy at the same addresses
```

- Advance one level after `ADVANCE_PATIENCE` consecutive evaluations with teacher accuracy at least `MIN_TEACHER_ACC` and the smoothed ratio at least `ADVANCE_RATIO`.
- Restore one teacher token after `RETREAT_PATIENCE` consecutive evaluations below `RETREAT_RATIO`.
- Hold otherwise.

At 1,000-step evaluation intervals, 126 additional releases with patience three require at least 378,000 steps. The 640,000-step example leaves 262,000 steps for holds, retreats, and final unsupported consolidation.

### Route-led stabilization inside Phase 2

The `1.25x` private-node multiplier is an autonomy-transfer rate, not a
permanent optimizer law. After route temperature has reached `0.35` and the
matched fully unsupported curve has plateaued, resume from the strongest free
checkpoint with private nodes at `0.8x` and lexical, lane/core, and route/gate
groups at `1.0x`. Preserve all AdamW moments, exact data position, and the
handoff-controller state.

Trigger this transition from observed state, not a fixed width formula or a
blind calendar step. The 130k proxy placed the useful switch around 90--100%
of its initial 20k screening horizon; wider 500k nodes showed a materially
different release timescale. Stop if stabilization does not improve the
matched free curve over its patience window. Continuing merely to release
more slots is not an objective.

## 5. Resume contract

Phase 2 resumes from the preserved Phase-1 checkpoint.

```text
--resume-checkpoint <phase1 checkpoint>
--resume-optimizer
--resume-data-state exact
```

Important trainer semantics:

- `--steps` is the number of **additional** updates in the new invocation.
- `--resume-optimizer` restores AdamW moments.
- The learning-rate scheduler starts a new phase; `--learning-rate` is the initial rate for Phase 2.
- `--resume-data-state exact` must succeed or the run is not an exact continuation.
- The token-handoff controller begins fresh when changing from `teacher_forced` to `token_handoff`.

Choose the Phase-2 learning rate explicitly. Do not accidentally restart the original peak learning rate after 160,000 updates. Record both phase schedules in the run specification.

## 6. Command skeleton

Use the same `<COMMON_ARGS>` in both invocations.

### Phase 1

```powershell
python -u gator_language_train.py `
  <COMMON_ARGS> `
  --decode-mode teacher_forced `
  --steps 160000 `
  --log-every 1000 `
  --checkpoint-dir <RUN_DIR>/phase1_teacher
```

### Phase 2

```powershell
python -u gator_language_train.py `
  <COMMON_ARGS> `
  --decode-mode token_handoff `
  --steps 640000 `
  --log-every 1000 `
  --checkpoint-dir <RUN_DIR>/phase2_handoff `
  --resume-checkpoint <RUN_DIR>/phase1_teacher/latest.pt `
  --resume-optimizer `
  --resume-data-state exact `
  --token-handoff-start-removed 1 `
  --token-handoff-max-removed 0 `
  --token-handoff-advance-ratio 0.95 `
  --token-handoff-retreat-ratio 0.75 `
  --token-handoff-min-teacher-accuracy 0.15 `
  --token-handoff-advance-patience 3 `
  --token-handoff-retreat-patience 2 `
  --token-handoff-full-probe-fraction 0.02 `
  --token-handoff-unforced-loss-weight 2.0 `
  --token-handoff-loss-normalization active_slots `
  --route-temperature-schedule staged_reheat `
  --route-temperature-start 1.25 `
  --route-temperature-middle 0.55 `
  --route-temperature-end 0.35 `
  --route-temperature-hold-fraction 0.10 `
  --route-temperature-middle-fraction 0.80 `
  --optimizer-groups component_scaled `
  --optimizer-lexical-lr-multiplier 1.0 `
  --optimizer-lane-core-lr-multiplier 1.0 `
  --optimizer-private-node-lr-multiplier 1.25 `
  --optimizer-route-gate-lr-multiplier 1.0
```

The dedicated runner may wrap these commands, but its generated `RUN_SPEC.json` must retain the effective arguments for both phases.

## 7. Metrics and curves

Plot all metrics against **cumulative optimizer step**, not phase-local step.

### Teacher baseline

- validation loss and PPL;
- overall token top-1/top-5 accuracy;
- content-only top-1/top-5 accuracy;
- exact response accuracy;
- formation coverage and whitespace omission;
- gradient norm and steps/second.

### Handoff frontier

- visible and removed teacher counts;
- unsupported frontier accuracy;
- teacher accuracy at the same addresses;
- smoothed frontier/teacher ratio;
- controller action and streaks;
- fully unsupported content accuracy and PPL;
- partial versus fully unsupported batch counts;
- best teacher and best unsupported checkpoints;
- a separate best fully unsupported **content-CE** checkpoint.

Always distinguish overall accuracy from content accuracy. Whitespace and other easy structural tokens can keep overall accuracy high while semantic formation remains weak.

`best_unsupported.pt` is formation-aware: the current trainer selects it by
unsupported slot CE, including blank placement. That is useful but is not the
same objective as free content PPL. Until the trainer natively writes both,
the run controller must preserve the checkpoint corresponding to the minimum
logged fully unsupported content CE as a separate artifact. Never replace one
checkpoint with the other; report both.

## 8. Stop, extend, and rollback rules

### Extend Phase 1

Extend full teacher forcing when any exit gate fails at `TF_STEPS`. Record the reason and the new boundary before continuing.

### Hold Phase 2

Let the controller hold when frontier quality is below threshold. Do not release teachers solely to meet a calendar schedule.

### Roll back

Return to `phase1_teacher_step_<TF_STEPS>.pt` when:

- loss or gradients become non-finite;
- teacher-forced validation regresses materially after the phase transition;
- frontier releases repeatedly advance and retreat without net progress;
- data-state continuity or validation fingerprints fail;
- unsupported training damages the causal baseline without improving unsupported metrics.

Do not silently change architecture, tokenizer, topology, or corpus mixture during rollback. Such a change starts a new experiment.

## 9. Required controls

Run or preserve these comparisons:

| Control | Purpose |
|---|---|
| Immediate token handoff from step 1 | Tests whether the teacher baseline phase is necessary |
| Full teacher forcing for the entire budget | Measures maximum supported causal capability |
| TF20% then token handoff | Primary two-phase treatment |
| Matched Transformer | Provides a conventional parameter/data/step baseline |

The existing Gator-100M preoptimization run is the immediate-handoff control. It began at one removed teacher token and remained at that level through the observed 56,000-step log while content accuracy stayed low.

## 10. Reusable run record

Copy and complete this block for every launch:

```text
experiment_id:
git_commit:
bundle_hash:
architecture:
parameter_count:
tokenizer_fingerprint:
validation_fingerprint:
corpus_mix:
seed:
total_steps:
teacher_forced_steps:
handoff_steps:
phase1_learning_rate_schedule:
phase2_learning_rate_schedule:
phase1_exit_step:
phase1_exit_metrics:
phase1_checkpoint:
resume_optimizer:
resume_data_state:
handoff_start_removed:
handoff_max_removed:
advance_ratio:
retreat_ratio:
minimum_teacher_accuracy:
advance_patience:
retreat_patience:
full_probe_fraction:
unforced_loss_weight:
final_removed_tokens:
best_teacher_checkpoint:
best_unsupported_checkpoint:
best_free_content_checkpoint:
best_free_content_ppl:
transfer_node_lr_multiplier:
stabilization_trigger:
stabilization_node_lr_multiplier:
route_temperature_schedule:
handoff_loss_normalization:
stop_reason:
```

## 11. Interpretation boundary

Success requires more than lower teacher-forced PPL. The two-phase program succeeds only if it:

1. establishes a stronger causal/content baseline than immediate handoff;
2. progresses beyond the one-token frontier;
3. preserves teacher-forced capability while support is removed; and
4. materially improves fully unsupported content accuracy, PPL, or qualitative formation.

If Phase 1 improves PPL but Phase 2 cannot release support, record the result as a stronger causal Gator with an unresolved formation transition—not as successful simultaneous formation.
