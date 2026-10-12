# Gator-100M N/D-Balanced Training Regimen

**Status:** production run contract  
**Base:** `agent/packed-routing-code-only`  
**Runner:** `run_gator_100m_curriculum_colab.py`  
**Global target:** 800,000 optimizer steps  
**Primary objective:** minimum fully unsupported content cross-entropy, with
free top-1/top-5 accuracy and formation coverage as joint diagnostics.

## 1. Frozen architecture

The run uses the aligned approximately 100M configuration selected after the
node-count/width and topology screens:

```text
V32768 / E288 / D368 / N626 / L10 / K4 / R2
shared-low-rank router rank 96
learned ingress, active top-k 32
small-world graph, seed 0, rewire 0.50
slot-conditioned terminal voting
C512 / O128
```

Exact trainable parameters: **99,980,100**. `N/D = 1.7011`. E288 and D368 are
both grouped-MM aligned, avoiding the physical padding tax of the literal
D370/N629 arithmetic solution. Degree four retains the high-ROI R2 fanout
improvement; relay depth remains two because R3 was harmful and R4 bought only
a marginal gain over R2 at substantially greater sequential cost.

This is a larger-budget validation of the provisional N/D region, not proof
of a universal scaling law. The original ratio evidence came from smaller
coverage-normalized fixed-ingress models; this run uses learned ingress.

## 2. Frozen data program

The tokenizer, filtering, corpus paths, source weights, and validation
construction remain identical across all phases and resumes. The effective
training mixture is:

| Source | Share |
|---|---:|
| TinyStories | 40% |
| SmolTalk Everyday | 6% |
| LMSYS Chat | 24% |
| Tulu3 SFT | 18% |
| SmolTalk Preferences | 12% |

The streamed non-TinyStories sources retain relative weights `1:4:3:2` and
restart only after a complete shuffled epoch. Training requires complete
context windows, trims whole context turns, deduplicates examples, and skips
the boilerplate first assistant response. Validation and tokenizer
fingerprints are checkpointed and exact data-state resume is mandatory.

## 3. Optimizer contract

All phases use AdamW with:

```text
base LR             3e-4, constant
weight decay        1e-4
epsilon             1e-8
global grad clip    1.0
```

Parameters are placed into named lexical, lane/core, private-node, and
route/gate groups from step zero. AdamW moments migrate by parameter name at a
phase boundary. The scheduler is constant, so a resumed process does not
silently restart a warmup or peak. Multipliers are:

| Phase | Lexical | Lane/core | Private nodes | Route/gate |
|---|---:|---:|---:|---:|
| Teacher foundation | 1.00 | 1.00 | 1.00 | 1.00 |
| Autonomy transfer | 1.00 | 1.00 | **1.25** | 1.00 |
| Route-led stabilization | 1.00 | 1.00 | **0.80** | 1.00 |

The 1.25 multiplier is a transfer cap, not a permanent width-derived rule.
The 0.80 phase slows semantic drift inside nodes after routes have adapted.
Teacher PPL is telemetry; it cannot veto a free-quality improvement.

## 4. Curriculum state machine

### Phase A — causal teacher foundation

Minimum range: global steps `0–160,000` (`20%` of the planned horizon).

```text
decode_mode=teacher_forced
component multipliers=1/1/1/1
route temperature=legacy initial cool to 0.35
```

At 160k, handoff begins only when:

1. content accuracy is at least 15%;
2. content PPL is finite;
3. the geometric-mean content-PPL improvement between the preceding and most
   recent ten validation points is below 1%;
4. the checkpoint contains model, AdamW, tokenizer, controller-compatible task
   metadata, and an exactly restorable data pipeline.

If the gate fails, teacher forcing extends in 20k increments. At 240k, failure
to pass is a stop-and-diagnose condition: the runner refuses to consume the
formation budget with an unready causal substrate. The exact exit checkpoint
is preserved as `phase1_teacher_step_<STEP>.pt`.

### Phase B — autonomy transfer

Handoff starts with one of the 127 shifted response teachers removed. The
controller removes or restores one token at a time:

```text
advance ratio                 0.95
retreat ratio                 0.75
minimum teacher accuracy      0.15
advance / retreat patience    3 / 2 evaluations
fully unsupported batches     2%
unsupported loss weight       2.0
loss normalization            all active response slots
teacher-PPL retention guard   disabled
```

Private nodes run at 1.25× while all other components remain at 1.00×. Route
temperature follows one resumable 480k-update program:

```text
1.25 hold through 10%
geometric cooling to 0.55 at 80%
geometric cooling to 0.35 at 100%
```

`--route-temperature-step-offset` and
`--route-temperature-total-steps` make this one global schedule even when it
crosses several Colab sessions. It never reheats merely because a process was
restarted.

After the route schedule reaches 0.35, the runner compares adjacent ten-point
geometric-mean windows of fully unsupported content PPL. If improvement is
still at least 1%, transfer continues in 20k global-step increments. If the
curve plateaus or worsens, stabilization begins. Architecture-specific
release speed is therefore observed rather than imposed by calendar.

### Phase C — route-led node stabilization

The transfer `latest.pt` and strongest free-content checkpoint are preserved
separately. Stabilization branches from the strongest
`best_free_content.pt` available at the switch, preserving that checkpoint's
AdamW moments, exact corpus position, prefetch queue, and token-handoff
controller. Private-node LR falls to 0.80×; other groups stay at 1.00× and
route temperature remains fixed at 0.35.

The branch trains to global step 800,000. Branching from an earlier best-free
checkpoint can make physical work exceed 800k because discarded transfer
updates are not relabeled; every retained checkpoint still has an honest
global optimizer step. Final selection is by the lowest fully unsupported
content loss, not by `latest.pt`.

If no post-cooling plateau occurs before 800k, the conditional stabilization
phase is not fabricated. Transfer continues to the target and the best-free
checkpoint remains the selected artifact.

## 5. Native checkpoints and selection

The trainer writes distinct objectives:

| Artifact | Selection objective |
|---|---|
| `latest.pt` | most recent resumable state |
| `best.pt` | active curriculum selection loss |
| `best_teacher.pt` | fully teacher-supported selection loss |
| `best_unsupported.pt` | fully unsupported formation-aware slot loss |
| `best_free_content.pt` | **fully unsupported content cross-entropy** |

Historical minima are embedded in `data_config.best_metrics`, preventing a
later session from replacing a genuinely better checkpoint with a merely
better-than-resume checkpoint. `best_unsupported.pt` and
`best_free_content.pt` remain separate because blank placement and semantic
content are both useful but are not the same objective.

## 6. Multi-day resume and Drive policy

Stable Drive directory:

```text
/content/drive/MyDrive/GTR/experimental_checkpoints/
gator100m_nd1p7_d368_n626_k4_r2_optimizer3phase_seed0_800k
```

Every invocation uses fresh local scratch and automatically discovers the
stable Drive `latest.pt`. Resume restores:

- exact model weights;
- every AdamW moment, with named-group migration at policy changes;
- global optimizer step;
- token-handoff level, EMA, streaks, probe credit, and move counts;
- streaming source RNG/cursors/restarts;
- finite-corpus state and validation identity;
- queued prefetched batches and provider RNG state.

Drive writes are deliberately separated:

- append pending logs every **15 minutes**;
- atomically replace `latest.pt` every **2 hours**;
- force a final log, latest, metadata, and changed-best sync when an invocation
  exits normally or through the runner's `finally` path.

The default invocation has a 22-hour trainer runtime ceiling so a browser-held
24-hour Colab runtime has time to serialize and mirror its final state. At
worst, an abrupt machine loss discards approximately two hours of optimizer
work; the checkpoint's exact data state remains authoritative on replay.

`CURRICULUM_STATE.json` records phase decisions independently of a checkpoint.
This is required when stabilization intentionally rewinds from transfer latest
to transfer best-free.

## 7. Launch cell

Run this from a browser-started Colab cell so the browser owns the active
execution. Re-run the same cell after a runtime reconnect; do not edit a PID or
manually select a phase.

```python
%cd /content/GTR
!git switch agent/packed-routing-code-only
!python -u run_gator_100m_curriculum_colab.py \
  --session-steps 100000 \
  --session-runtime-seconds 79200 \
  --batch-size 1 \
  --eval-examples 16 \
  --log-every 1000 \
  --log-sync-seconds 900 \
  --checkpoint-sync-seconds 7200 \
  --route-backend dense \
  --packed-memory-backend matrix \
  --compile
```

Dense/matrix is the speed default. If the target GPU cannot hold batch one,
use the validated packed backend as a memory fallback and record that backend
as a separate performance treatment; packed execution is not represented as a
speed optimization.

## 8. Monitoring and intervention rules

Plot against cumulative global step:

- teacher, partial, and fully unsupported content PPL;
- free content top-1 and top-5 accuracy;
- formation coverage and whitespace omission;
- removed/visible teacher tokens, advance/retreat actions, EMA, and streaks;
- route entropy/effective nodes and winning-lane diversity;
- per-component LR and update-to-parameter ratios;
- gradient norm, throughput, peak VRAM, and phase transitions.

Stop and repair rather than silently reset when:

- a loss, gradient, or optimizer update becomes non-finite;
- exact data resume or tokenizer/validation fingerprint validation fails;
- model configuration or parameter count differs;
- checkpoint sync cannot atomically replace the stable Drive artifact;
- the 240k teacher safety bound is reached without passing its gates.

Do not reset AdamW, the sampler, the token-handoff controller, or route
temperature to make a resume convenient. Do not promote teacher PPL over the
free deployment objective. Preserve `latest`, transfer-latest,
formation-aware best, and free-content best so a noisy endpoint cannot erase
the useful trajectory.

## 9. Completion criteria

The run is complete only when:

1. a restorable checkpoint reaches global step 800,000;
2. the stable run specification still reports 99,980,100 parameters and the
   frozen corpus fingerprints;
3. the best fully unsupported content checkpoint exists;
4. free PPL, accuracy, top-5, coverage, release trajectory, elapsed time, and
   backend are summarized from the complete curve;
5. qualitative generations are produced from the selected best-free checkpoint
   with every shifted response teacher removed.

The result may validate or reject the proposed ratio and optimizer transfer.
It must not be called successful simultaneous formation solely because teacher
PPL improved.
