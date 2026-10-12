# Gator Optimizer-Alignment Study

**Status:** Active, sequential local/Colab experiment  
**Machine policy:** One foreground training cell at a time; no detached jobs or
PID keepalives. The frozen proxy was calibrated locally, then moved to the
authenticated Colab T4 session with a 6 GB CUDA allocator ceiling.  
**Goal:** Probe how loss scaling, routing temperature, optimizer grouping,
optimizer-state continuity, and sparse-node activity interact with Gator's
intended routing and specialization.

## Frozen proxy

| Component | Setting |
|---|---:|
| Vocabulary | 1,024 |
| Lexical embedding width | 12 |
| Payload width | 32 |
| Private nodes | 30 |
| Persistent lanes | 4 |
| Degree / relays | 4 / 2 |
| Router | shared low rank, rank 8 |
| Ingress | learned, global |
| Active nodes per packet | top 4 |
| Terminal vote | slot conditioned |
| Graph | small world, seed 0, rewire 0.5 |
| Context / response slots | 32 / 16 |
| Trainable parameters | **61,770** |

Parameter composition is 13,312 tied lexical parameters, 13,332 lane
parameters, 32,760 private-node parameters, 1,886 router/ingress parameters,
and 480 address-edge parameters. Private nodes therefore comprise 53.0% of
the proxy.

The corpus is the local SmolTalk Everyday JSONL only. Large LMSYS, Tulu, and
preference corpora are excluded so I/O, source mixing, and RAM do not confound
the optimizer comparisons. Every treatment will fork from the same preserved
teacher-forced checkpoint and use the same tokenizer, validation manifest,
model weights, and data state.

## Decision order

1. Calibrate CPU throughput and memory on the frozen proxy.
2. Establish a shared teacher-forced checkpoint.
3. Compare legacy additive versus normalized unsupported-token loss.
4. Compare the historical fast route cooling against staged cooling/reheat.
5. Verify neutral parameter grouping, then test specialized group settings.
6. Compare full optimizer-state preservation against route/gate-only reset.
7. Test epsilon, accumulation, fused AdamW, and lazy row updates only when
   their diagnostic gates are met.
8. Confirm the selected treatment and write the final recommendation.

## Result ledger

| Test | Changed factor | Steps | Best teacher content PPL | Best unsupported content PPL | Frontier | Steps/s | Peak RSS | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 00 | Unmodified teacher-forced calibration | 200 | 35.37 overall PPL | n/a | n/a | 6.40 | not captured | **pass** |
| 01 | Shared teacher-forced acquisition | 8,000 | **27.41** at step 4,500 | n/a | n/a | 9.83 | 1,723 MiB host / 80 MiB CUDA | **pass; branch from best step 4,500** |
| 02a | Legacy additive released-slot loss | 3,000 | 30.02 at step 5,000 | **117.35** at step 7,250 | 4/15 released | 8.06 | 1,919 MiB host / 75 MiB CUDA | **fail as control; release outran learning** |
| 02b | Slot-normalized released-slot loss | 3,000 | **29.55** at step 6,000 | 162.67 at step 7,250 | 1/15 released | 8.35 | 1,921 MiB host / 75 MiB CUDA | **safer, but underpowered; tune coefficient** |
| 02c | Slot-normalized loss, coefficient 4 | 3,000 | **29.37** at step 5,000 | **116.67** at step 7,500 | 4/15 released | 21.48 L4 | 2,197 MiB host / 114 MiB CUDA | **upper bound; still releases too fast** |
| 02d | Slot-normalized loss, coefficient 2 | 3,000 | **28.93** at step 5,000 | 130.87 at step 7,500 | 4/15 released | 25.86 L4 | 2,103 MiB host / 73 MiB CUDA | **provisional loss winner; controller still needs a retention guard** |
| 03a | Full staged route reheat, coefficient 2 | 3,000 | 32.10 at step 7,500 | **130.04** at step 7,500 | 1/15 released | 27.09 L4 | 2,150 MiB host / 73 MiB CUDA | **route win; test milder reheat to reduce initial shock** |
| 03b | Mild staged route reheat, coefficient 2 | 3,000 | 34.57 at step 5,000 | **108.42** at step 7,500 | 4/15 released | 26.86 L4 | 2,153 MiB host / 73 MiB CUDA | **free-output win, retention fail; keep full reheat for optimizer tests** |
| 04a | Neutral component parameter groups | 3,000 | 34.25 at step 7,250 | **118.57** at step 7,500 | 2/15 released | 24.93 L4 | 2,068 MiB host / 73 MiB CUDA | **state migration pass; grouping is not trajectory-neutral under hard routing** |
| 04b | Private-node LR 0.70x | 3,000 | 34.60 at step 7,250 | 130.06 at step 7,500 | 2/15 released | 25.47 L4 | 2,068 MiB host / 73 MiB CUDA | **reject; balanced update ratios learned worse** |
| 04c | Private-node LR 1.30x | 3,000 | 35.66 at step 7,500 | **107.67** at step 7,500 | 3/15 released after final eval | 25.42 L4 | 2,069 MiB host / 73 MiB CUDA | **provisional grouping winner; node priority improves the joint tradeoff** |
| 05a | Reset route/gate AdamW state | 3,000 | 37.87 at step 7,500 | 127.33 at step 7,500 | 2/15 released | 25.78 L4 | 2,066 MiB host / 73 MiB CUDA | **reject; preserved route moments stabilize reheat** |
| 05b | Reset all AdamW state | 3,000 | 39.65 at step 6,250 | **106.08** at step 7,500 | 4/15 released | 25.06 L4 | 2,066 MiB host / 73 MiB CUDA | **reject; free-token acceleration destroys retained competence** |
| 06a | Late multiplicative LR drop, 0.35x at branch step 2,250 | 3,000 | **31.10** at step 7,500 | 160.22 at step 7,500 | 1/15 released | 25.35 L4 | 2,079 MiB host / 73 MiB CUDA | **retention win, handoff underpowered; test a gentler or event-gated decay** |
| 06b | Late multiplicative LR drop, 0.70x at branch step 2,250 | 3,000 | 36.42 at step 6,250 | 143.21 at step 7,500 | 3/15 released | 24.78 L4 | 2,066 MiB host / 73 MiB CUDA | **reject; dominated by constant LR on retained and free PPL** |
| 07a | AdamW epsilon 1e-6 | 3,000 | **34.73** at step 7,500 | 180.60 at step 7,500 | 1/15 released | 25.35 L4 | 2,080 MiB host / 73 MiB CUDA | **reject; damps useful low-amplitude handoff adaptation** |

## Test 00 -- unmodified calibration

### Hypothesis

The 61,770-parameter proxy should run safely within the RAM ceiling and fast
enough to support sequential 3,000-step treatment screens. This test also
verifies that the current trainer accepts the node-heavy configuration before
experimental optimizer controls are added.

### Frozen controls

- Decode mode: `teacher_forced`
- Objective: masked slot cross-entropy
- Optimizer: monolithic AdamW
- Learning rate: 0.001, constant
- Weight decay: 0.0001
- Gradient clip: 1.0
- Batch size: 16
- Validation: fixed first 128 examples every 100 steps
- Data pipeline: serial
- Seed: 0

### Result

The trainer instantiated the exact 61,770-parameter model and retained 1,605
training plus 187 held-out responses after filtering. It completed 200 CPU
updates in 31.27 seconds at 6.40 steps/second with finite loss and gradients.
Loss/PPL improved from 3.8635/47.63 at step 100 to 3.5660/35.37 at step 200.

Content top-1 accuracy and formation coverage remained zero; the model emitted
only BLANK slots. Content top-5 held near 19.2%. This is an expected early
acquisition state and establishes that later tests must not mistake structural
token learning or falling overall PPL for useful content formation. Terminal
voting narrowed from 3.00 to 2.36 effective lanes while top-1 terminal mass
rose from 0.506 to 0.634.

The latest checkpoint is 795,887 bytes with SHA-256
`61136C6BD8A8A47C2FE9EE87E6C29C347EDBE5968838372DBD357079470FFD02`.
Peak RSS was not captured in this first direct launch; the monitored runner
must capture it for every subsequent test. The tiny checkpoint and completed
run are already far below the 6--7 GB safety ceiling.

**Decision:** pass. At the measured rate, an 8,000-step teacher acquisition
run has approximately 21 minutes of raw update time before evaluation
overhead.

## Test 01 -- shared teacher-forced acquisition

### Hypothesis

Eight thousand updates should move the proxy beyond BLANK-only structural
learning and create one reusable causal checkpoint. If teacher content
accuracy remains below 15%, continue the same exact trajectory in 2,000-step
increments, capped at 16,000 steps.

### Changed factor

Only optimizer-step horizon changes relative to Test 00. The model, corpus,
tokenizer policy, optimizer, loss, route cooling, batch, seed, and validation
policy remain frozen. Evaluation runs every 250 steps over the same first 128
held-out examples. This checkpoint is acquisition evidence, not unsupported
simultaneous-generation evidence.

### Result

The foreground T4 cell completed all 8,000 updates in 814.03 trainer seconds
(821.31 seconds including setup and artifact hashing), sustaining 9.83
steps/second. Peak allocated/reserved CUDA memory was only 80/96 MiB and peak
host RSS was 1,723 MiB. The source bundle and corpus SHA-256 values exactly
matched the laptop copies:

- source: `70de1c23c1b3b0620f592bec91d9e32efcb4620132c6815ae2ff9c961416f7f9`
- corpus: `01118075022fc535601ce0bad46ddadb57fbfc9b5b06236c4a259b856b9834d9`

Acquisition crossed the 15% teacher-content-accuracy gate at step 1,000 and
therefore required no extension. The best selection checkpoint occurred at
step 4,500 with overall PPL 12.19, content PPL 27.41, content top-1 accuracy
24.73%, content top-5 accuracy 53.30%, and formation coverage 93.19%.

Continuing the identical constant-LR trajectory to step 8,000 worsened overall
PPL to 14.02 and content PPL to 32.25. Content accuracy rose modestly to
26.35%, but the loss reversal beginning after roughly step 4,500 is evidence
that a fixed 0.001 learning rate secretly overtrains this small proxy. The
latest checkpoint is therefore not the neutral handoff state.

The best checkpoint is 795--797 KB and the latest checkpoint SHA-256 is
`5bc44181cc6da7fdd2a1a5c7963f5621b30abe207b3e7769b207d4371a6444b0`.

**Decision:** pass acquisition, freeze `best.pt` at step 4,500 as the common
causal checkpoint, and treat the post-peak degradation as a scheduler/LR
mismatch to revisit after the primary loss, route, grouping, and state tests.
All treatment branches use the current compiled dense T4 path. Packed routing
is excluded from this tiny proxy because it trades speed for memory and would
confound optimizer comparisons.

## Test 02a -- legacy additive released-slot objective

### Hypothesis

The historical objective may let token handoff advance because the supported
slots dominate the mean cross-entropy while the newly unsupported suffix is
added as a second term whose scale changes with the number of released slots.
This arm is the control for the slot-normalized objective; it is not a proposed
final treatment.

### Changed factor

The frozen step-4,500 teacher checkpoint was resumed with its complete AdamW
state and exact data position, then switched from `teacher_forced` to
`token_handoff` for 3,000 additional updates. All other optimizer and routing
settings remained historical: monolithic AdamW, constant 0.001 learning rate,
legacy route temperature, and additive unsupported-token loss. Execution used
the current `dense_gpu` compilation preset; packed routing was not used.

### Result

The foreground T4 cell completed in 372.30 trainer seconds (378.87 seconds
including setup and hashing), ending at 8.06 steps/second after compile warmup.
Peak host RSS was 1,919 MiB; peak allocated/reserved CUDA memory was 75/94 MiB.
The latest checkpoint SHA-256 is
`591cbdff99fcc4943ce261d9a71127d74692ac31fa35551351b3d84e566f4859`.

The best retained primary checkpoint arrived after only 500 branch updates,
at global step 5,000. Its fully teacher-forced auxiliary evaluation had content
PPL 30.02, content top-1 accuracy 22.84%, top-5 accuracy 52.49%, and formation
coverage 92.65%. The partially released evaluation had content PPL 30.63,
accuracy 23.65%, and coverage 89.62%. Both were already worse than the shared
step-4,500 teacher checkpoint's 27.41 content PPL and 24.73% accuracy.

Despite that regression, the handoff controller advanced from one released
slot to four, at branch steps 750, 1,500, and 2,250, with no retreat. At the
best unsupported checkpoint (global step 7,250), fully unsupported content PPL
was 117.35, accuracy 8.69%, top-5 accuracy 28.64%, and formation coverage only
21.83%. The corresponding teacher-forced content PPL was 41.32. At the final
step, teacher content PPL had deteriorated further to 42.29 while free content
PPL remained 120.56. The controller therefore rewarded local frontier parity
even while both retained teacher knowledge and global unsupported formation
were degrading.

The optimizer diagnostics expose a second architectural mismatch to test
later. Private-node gradients were much smaller than lane-core gradients, yet
their update-to-parameter ratio was commonly the largest group ratio
(approximately 0.08%--0.16% per sampled update). A monolithic AdamW learning
rate therefore does not preserve the intuitive importance implied by raw
gradient magnitude; Adam's normalization lets many private node weights move
more aggressively than the shared lane and lexical substrate.

The compiled run also reported a graph break at the Python scalar branch around
`selected.any()` in the sparse node update. That is a later kernel/compilation
optimization opportunity, not a reason to substitute packed routing in these
optimizer ablations.

**Decision:** fail the historical objective as a release policy. Keep this arm
as the matched control and run the slot-normalized treatment from the exact
same step-4,500 weights, optimizer moments, data position, and compiled-dense
execution path.

## Test 02b -- slot-normalized released-slot objective

### Hypothesis

Dividing the unsupported-token term by every active response slot, rather than
only the released slots, should stop a single frontier token from receiving the
same aggregate weight as the complete supported response. This should preserve
teacher-acquired formation while making the controller's advance signal more
honest.

### Changed factor

Only unsupported-loss normalization changed relative to Test 02a. This arm
restored the exact original step-4,500 checkpoint from Drive and verified its
SHA-256 as
`f7b1c6815f457ec4782cd9fd2d8b19da08cfd2f90a668cd64c1aed779dff4d34`.
It resumed the same AdamW moments and data cursor, used the same constant 0.001
learning rate and legacy route schedule, and ran on the same compiled-dense T4
path for 3,000 updates.

### Result

The foreground cell completed in 359.26 trainer seconds (366.56 seconds with
setup and archive creation), ending at 8.35 steps/second. Peak host RSS was
1,921 MiB; peak allocated/reserved CUDA memory was 75/94 MiB. The downloaded
ZIP is 2,915,132 bytes with SHA-256
`c64b7689ff65c353fc358702a116118eb0834c44a380d2cb30a8a5da6b75b64f`.

The best primary checkpoint occurred at global step 6,000. Its teacher-forced
content PPL was 29.55 with 24.53% top-1 accuracy, 52.76% top-5 accuracy, and
91.98% formation coverage. The partially released evaluation had overall PPL
13.39, content PPL 30.49, top-1 accuracy 23.58%, and 90.30% coverage. Relative
to 02a's best retained checkpoint, normalized loss improved teacher content PPL
by 1.6%, restored 1.69 percentage points of teacher top-1 accuracy, and slightly
improved overall partial PPL.

More importantly, the controller made no false advances. It held at one of 15
released slots for all 3,000 updates because frontier/teacher parity usually
remained below the 0.95 advance threshold. At the final checkpoint, teacher
content PPL was 32.44 versus 42.29 for 02a, and partial overall PPL was 14.16
versus 17.57. Normalization therefore substantially reduced catastrophic drift.

That stability came with an opposite failure. Because one released slot now
contributes only about 1/15 of a response-scale loss term, the frontier signal
was too weak to master and release. The best fully unsupported checkpoint had
content PPL 162.67, 6.20% top-1 accuracy, and 18.80% formation coverage, worse
than 02a's 117.35 PPL despite much better teacher retention. The two arms bound
the useful scale: legacy additive is too strong and slot-normalized weight 1 is
too weak.

Private-node update ratios again tended to exceed shared-lane, lexical, and
route/gate ratios despite much smaller raw private-node gradient norms. This
replication strengthens the case that monolithic AdamW is independently
misaligned with Gator's distributed private-node substrate.

**Decision:** retain slot normalization as the structurally correct denominator,
but do not use coefficient 1 as the final objective. Add a bounded coefficient
screen between the two extremes before routing ablations. The target is the
lowest fixed coefficient that earns real controller advances without materially
worsening teacher PPL or formation coverage.

## Test 02c -- slot-normalized objective, coefficient 4

### Hypothesis

A fixed coefficient of 4 should make each unsupported token receive five times
the ordinary per-token weight after including base slot cross-entropy. This is
far below legacy's roughly 16-times weight when only one of 15 content slots is
released, but strong enough to avoid the coefficient-1 stall.

### Changed factor

Only `--token-handoff-unforced-loss-weight` changed from 1 to 4 relative to
02b. The exact baseline checkpoint, optimizer moments, data state, controller,
legacy route schedule, constant learning rate, and compiled-dense execution
remained fixed. This was the first arm on the user-created persistent L4
runtime, so throughput is not directly comparable with the T4 rows.

### Result

The L4 completed 3,000 updates in 139.68 trainer seconds (149.99 seconds with
compile/setup/archive overhead), ending at 21.48 steps/second. Peak host RSS
was 2,197 MiB; peak allocated/reserved CUDA memory was 114/128 MiB. The local
ZIP SHA-256 is
`2e301d8442e3e7dc0decc62637b4a237f42f44ca80e8a6c2d819fb6eb895283d`.

The best primary checkpoint appeared at global step 5,000. Teacher content PPL
was 29.37 with 24.12% top-1 accuracy, 53.17% top-5 accuracy, and 93.94%
formation coverage. Partial content PPL was 29.85 with 24.39% accuracy and
90.90% coverage. This is the best early checkpoint among the three loss arms.

The controller then advanced at branch steps 1,000, 1,750, and 2,500, ending at
four released slots. Final fully unsupported content PPL improved to 116.67,
slightly better than legacy's best 117.35, but teacher content PPL degraded to
40.20 and partial formation coverage fell to 75.61%. Coefficient 4 therefore
improves the tradeoff but does not prevent release from outrunning retained
formation.

The convergence with legacy at four released slots is mathematically expected.
Legacy's extra per-token weight is approximately `15 / released_slots`; at four
slots that is 3.75, close to the fixed normalized coefficient of 4. The
normalized formulation fixes the exploding first-frontier scale, but coefficient
4 recreates legacy pressure later in the release sequence.

**Decision:** retain coefficient 4 as the upper bound, not the selected value.
Test coefficient 2 from the same baseline. It gives unsupported tokens three
times ordinary weight and should distinguish controlled progress from a merely
slower version of legacy collapse.

## Test 02d -- slot-normalized objective, coefficient 2

### Hypothesis

Coefficient 2 gives each unsupported token three times ordinary total weight.
It should advance more deliberately than coefficient 4 while still avoiding
the coefficient-1 stall.

### Changed factor

Only the fixed unsupported-loss coefficient changed from 4 to 2. The arm used
the same exact baseline, full AdamW state, data cursor, controller, legacy route
schedule, constant 0.001 learning rate, and compiled-dense L4 execution.

### Result

The L4 completed 3,000 updates in 116.01 trainer seconds (120.16 seconds
including harness overhead), ending at 25.86 steps/second. Peak host RSS was
2,103 MiB; peak allocated/reserved CUDA memory was 73/92 MiB. The downloaded
ZIP SHA-256 is
`fe9f72a05690aeb0212700cf6aa006cf53bb5b99bdac7614d58484306b18cf0b`.

The best primary checkpoint again occurred at global step 5,000. Teacher
content PPL was 28.93 with 24.93% top-1 accuracy, 52.36% top-5 accuracy, and
93.13% coverage. Partial overall/content PPL was 13.35/30.24 with 23.58%
content accuracy and 89.82% coverage. This is the strongest retained-teacher
checkpoint among the loss treatments that made real handoff progress.

The controller advanced at branch steps 1,500, 2,250, and 3,000, ending at four
released slots. At the final checkpoint, teacher content PPL was 37.88, partial
content PPL was 38.22, and fully unsupported content PPL was 130.87. This is a
clear midpoint: substantially better teacher retention than coefficient 4
(40.20) or legacy (42.29), but weaker fully unsupported PPL than coefficient 4
(116.67) or legacy (117.35).

The final advance also demonstrates that loss scale is not the whole failure.
Once local frontier accuracy approaches teacher-frontier accuracy, the ratio
gate repeatedly saturates even if whole-response teacher PPL is worsening. A
fixed coefficient controls how quickly this happens; it cannot tell the
controller whether the substrate is globally retaining formation.

**Decision:** select slot normalization with coefficient 2 as the provisional
loss treatment for route and optimizer ablations. Add an absolute global
teacher-retention/coverage guard to the final training template; do not rely on
frontier parity alone. Compare staged route reheat against this legacy-route
arm before changing optimizer groups.

## Test 03a -- full staged route reheat

### Hypothesis

The shared teacher checkpoint reaches handoff with the historical route
temperature already pinned at 0.35. Reheating should let routing and ingress
rediscover useful paths under the new unsupported-token distribution rather
than forcing private nodes to adapt behind effectively frozen routes.

### Changed factor

Relative to the coefficient-2 legacy-route arm, only training route temperature
changed. It held 1.25 for the first 10% of the 3,000-step branch, cooled
geometrically to 0.55 by 80%, then to 0.35 at the end. Evaluation remained at
the standard cold setting. The final implementation keeps the step counter and
piecewise schedule as tensors inside the compiled graph. An earlier aborted
attempt used Python schedule arithmetic, hit Dynamo's eight-recompile limit,
and is excluded from all results.

### Result

The valid L4 arm completed in 110.77 trainer seconds (114.86 seconds with
harness overhead), ending at 27.09 steps/second with no recompile warning. Peak
host RSS was 2,150 MiB; peak allocated/reserved CUDA memory was 73/92 MiB. The
downloaded ZIP SHA-256 is
`02fabe9c594892026f729e67d18f9d098ba48dade436e0bd18525cd3e96fba22`.

Reheating caused a deliberate but severe early perturbation: partial overall
PPL rose to 21.34 after 250 updates, versus 13.59 for the cold route. Quality
then recovered throughout cooling, and the best primary, teacher, and fully
unsupported selections all occurred at the final step rather than early in the
branch.

At global step 7,500, teacher content PPL was 32.10 with 25.27% top-1 accuracy
and 94.34% formation coverage. Partial content PPL was 33.16 with 25.27%
accuracy and 93.46% coverage. Fully unsupported content PPL was 130.04 with
7.61% accuracy and 31.13% coverage. The controller held at one released slot.

Against coefficient-2 cold routing at the same horizon, reheat improved teacher
content PPL from 37.88 to 32.10, partial formation coverage from 83.09% to
93.46%, free coverage from 19.54% to 31.13%, and slightly improved free PPL
from 130.87 to 130.04 despite exposing three fewer unsupported training slots.
Route/gate update ratio also rose to 0.182% on the first update, confirming that
the schedule reactivated a routing subsystem that was cold at handoff.

**Decision:** staged reheat beats leaving the mature teacher routes frozen at
0.35. Test one milder 0.75 -> 0.45 -> 0.35 schedule to determine whether the
same benefit can be retained without the initial quality shock; then freeze the
route policy for optimizer grouping.

## Test 03b -- mild staged route reheat

### Hypothesis

Reducing the initial reheat from 1.25 to 0.75 and the middle target from 0.55
to 0.45 should retain enough routing plasticity for unsupported-token learning
while avoiding Test 03a's early quality shock.

### Changed factor

Only the route-temperature schedule changed from Test 03a. It held 0.75 for
the first 10% of the 3,000-step branch, cooled geometrically to 0.45 by 80%,
and reached 0.35 at the end. Slot-normalized coefficient 2, monolithic AdamW,
the complete resumed optimizer/data state, controller settings, and compiled
dense execution were unchanged.

### Result

The L4 completed in 111.68 trainer seconds (115.74 seconds including harness
overhead), ending at 26.86 steps/second. Peak host RSS was 2,153 MiB; peak
allocated/reserved CUDA memory was 73/92 MiB. The downloaded ZIP SHA-256 is
`645a6de827fc60dadb01cec9bca64e2a75f828772b9ea950a2e0666e45b7c1ca`.

The milder start did remove the transient shock: partial overall PPL was 15.83
after 250 updates rather than 21.34 under full reheat. It also restored rapid
frontier release. The controller advanced at branch steps 750, 1,500, and
2,250, ending at four of 15 released slots.

That speed reproduced the retention failure. The best primary checkpoint was
again early, at global step 5,000. Its teacher content PPL was 34.57 with
22.71% top-1 accuracy, 49.93% top-5 accuracy, and 91.37% formation coverage;
partial content PPL was 36.11 with 23.18% accuracy and 89.15% coverage. By the
final step, teacher content PPL had worsened to 38.22 and partial formation
coverage had fallen to 80.12%.

Free-running quality improved materially: final unsupported content PPL was
108.42 with 9.30% top-1 accuracy, 28.98% top-5 accuracy, and 29.04% coverage.
The checkpoint selector retained global step 7,000 as `best_unsupported`
because its overall unsupported PPL was fractionally better, although its
content PPL was 114.63; the final checkpoint is therefore the relevant
content-PPL minimum. The generated samples remained repetitive, so this is a
useful optimization signal rather than evidence of good language quality.

Compared with full reheat, the mild schedule trades retained structure for
faster autonomous-token fitting. Full reheat finished with teacher content PPL
32.10 and partial coverage 93.46% while mild reheat finished at 38.22 and
80.12%, respectively. The mild arm's 108.42 free content PPL is better than
full reheat's 130.04, but it achieves that result after releasing four slots
rather than one. Those are different exposure levels, and the local frontier
ratio still cannot detect the global regression.

**Decision:** reject mild reheat as the optimizer-ablation baseline. Freeze the
full 1.25 -> 0.55 -> 0.35 schedule because it tests optimizer behavior without
letting the current controller silently exchange teacher competence for rapid
release. Preserve mild reheat as a later acceleration option only after adding
an absolute teacher-retention and formation-coverage guard.

## Test 04a -- neutral component parameter groups

### Hypothesis

Splitting AdamW into lexical, lane/core, private-node, and route/gate parameter
groups while keeping every group at learning rate 0.001 and weight decay
0.0001 should verify name-based state migration before unequal component rates
are tested. In smooth models this would be expected to reproduce the
monolithic full-reheat arm to numerical tolerance.

### Changed factor

Only optimizer parameter-group topology changed from Test 03a. All 27 saved
AdamW parameter states were migrated by parameter name. Every group retained
the same learning rate, decay, betas, epsilon, full route-reheat schedule,
loss, data cursor, and controller state. The grouped counts were 16,384 lexical,
10,112 lane/core, 32,760 private-node, and 2,514 route/gate parameters.

### Result

The migration gate passed `27/27`. On the first update, component update norms
matched the monolithic arm within about `4e-9`: for example, the private-node
norm was 0.065142998 versus 0.065143002. The grouping code therefore preserved
the intended optimizer state and scalar hyperparameters.

The run nevertheless diverged after those tiny kernel-level rounding changes.
By branch step 250, frontier accuracy was 0.280 versus 0.237 in the monolithic
arm. At branch step 2,750 the grouped trajectory crossed the controller gate
and released a second slot; the monolithic trajectory remained at one. This is
not evidence that equal groups are a useful treatment. It is evidence that
Gator's hard top-k routing and thresholded handoff controller amplify nearly
invisible numerical differences into different data paths.

The best primary checkpoint was global step 7,250. Teacher content PPL was
34.25 with 22.37% top-1 accuracy, 52.36% top-5 accuracy, and 90.70% coverage;
partial content PPL was 35.70 with 22.98% accuracy and 88.61% coverage. At the
final step, teacher content PPL was 35.43 with 23.32% accuracy and 93.60%
coverage. Fully unsupported content PPL was 118.57 with 7.35% accuracy and
23.52% coverage. Compared with monolithic Test 03a, this path learned free
tokens faster but retained less teacher quality and released one extra slot.

The L4 completed in 120.35 trainer seconds (124.44 seconds including harness
overhead), ending at 24.93 steps/second. Peak host RSS was 2,068 MiB; peak
allocated/reserved CUDA memory was 73/92 MiB. The downloaded ZIP SHA-256 is
`f7ed8c46f0070aa820df9b54fbaa4fa0bec088889c929586b616478a80c46c1d`.

**Decision:** pass optimizer-state migration, but reject the assumption that
parameter regrouping is a behaviorally neutral control. Compare every unequal
group treatment against this grouped trajectory, not only against monolithic
Test 03a. For production-scale claims, repeat the selected grouped treatment
across seeds because hard routing turns floating-point perturbations into a
real trajectory fork.

## Test 04b -- reduced private-node learning rate

### Hypothesis

Private-node update-to-parameter ratios averaged roughly 1.34e-3 in the
neutral grouped arm, versus 0.87e-3 lexical, 0.95e-3 lane/core, and 0.92e-3
route/gate. Reducing only the private-node learning rate to 0.70x should bring
its effective update ratio into the shared-component range and test whether
monolithic AdamW is moving Gator's distributed transforms too aggressively.

### Changed factor

Only the 32,760 private-node parameters changed learning rate, from 0.001 to
0.0007. The other three component groups remained at 0.001, all groups retained
weight decay 0.0001, and all 27 AdamW states were migrated from the same teacher
checkpoint. Full reheat, loss coefficient 2, data order, and controller were
identical to Test 04a.

### Result

The mechanical target was reached. Mean sampled private-node update ratio fell
from 1.34e-3 to 0.94e-3, while the other groups remained approximately stable.
The first private-node update fell from 0.06514 to 0.04560 in norm. The
controller still released its second slot at branch step 2,750, so the lower
node rate did not make the handoff gate more conservative.

Quality moved in the wrong direction. At the best primary checkpoint, global
step 7,250, teacher content PPL was 34.60 with 22.64% accuracy and 90.03%
coverage; partial content PPL was 35.92. At the final step, teacher content PPL
was 35.61, partial content PPL was 37.53, and fully unsupported content PPL was
130.06 with 7.14% accuracy and 22.64% coverage. The equal-group control ended
at 35.43, 35.45, and 118.57, respectively. Equalizing update ratios therefore
slowed useful private-node adaptation without improving retention.

The L4 completed in 117.77 trainer seconds (122.01 seconds including harness
overhead), ending at 25.47 steps/second. Peak host RSS was 2,068 MiB; peak
allocated/reserved CUDA memory was 73/92 MiB. The downloaded ZIP SHA-256 is
`a8f01189d41535a6b34b5bcc11e5d27cdd9e4611f014cf94bb59b29f0c36196e`.

**Decision:** reject private-node LR 0.70x. The larger native AdamW update ratio
is not sufficient evidence that the optimizer is fighting the architecture;
on this proxy it appears to compensate for node-local learning. Test a clean
private-node-priority arm next, changing only the node multiplier, before
considering reductions to the shared lane or route rates.

## Test 04c -- increased private-node learning rate

### Hypothesis

If the naturally larger private-node updates are useful compensation for
distributed node-local learning, increasing only the private-node learning
rate to 1.30x should improve autonomous-token learning. Keeping every shared
and routing group at 1.0x avoids conflating node priority with route freezing
or reduced shared-substrate learning.

### Changed factor

Only the private-node multiplier changed from the neutral grouped control,
from 1.0 to 1.30, producing learning rate 0.0013 for 32,760 parameters. All
other groups stayed at 0.001 with the same weight decay. The exact baseline,
27/27 migrated AdamW states, data path, full reheat, loss, and controller were
unchanged.

### Result

The mean sampled private-node update ratio rose from 1.34e-3 to 1.53e-3 and
the first update ratio reached 2.21e-3. Despite that larger movement, the arm
improved the joint outcome. Its best primary and unsupported selections both
occurred at global step 7,500. Partial content PPL was 34.79 with 24.53% top-1
accuracy, 50.81% top-5 accuracy, and 91.17% coverage. Teacher content PPL was
35.66 with 24.12% accuracy, 51.48% top-5 accuracy, and 94.14% coverage.

Fully unsupported content PPL fell to 107.67 with 8.83% top-1 accuracy,
25.13% top-5 accuracy, and 27.22% coverage. Against the equal-group control,
partial PPL improved 1.8%, free PPL improved 9.2%, free accuracy gained 1.48
percentage points, and both partial and free coverage improved. Teacher PPL
was 0.6% worse, but teacher accuracy and coverage were better. More
importantly, this arm roughly matched mild reheat's 108.42 free PPL while
retaining far better teacher PPL and partial formation coverage.

The controller advanced at branch step 2,250 and again immediately after the
final evaluation at branch step 3,000. The logged final metrics therefore
evaluate two released slots, while the saved controller state ends with three
of 15 released. This again argues for an absolute retention guard before the
next branch is allowed to consume the new exposure level.

The L4 completed in 118.01 trainer seconds (122.24 seconds including harness
overhead), ending at 25.42 steps/second. Peak host RSS was 2,069 MiB; peak
allocated/reserved CUDA memory was 73/92 MiB. The downloaded ZIP SHA-256 is
`8895bc641354454b8203250d08cdadacb5546ac854cb91b2815bd3b4dafef14d`.

**Decision:** provisionally select private-node LR 1.30x. The architecture does
benefit from explicit node priority; equal relative update ratios were the
wrong target. Keep route/gate and shared rates at 1.0x for the state-continuity
ablation so node priority is not confounded with a colder router.

## Test 05a -- route/gate optimizer-state reset

### Hypothesis

Routing, ingress, and terminal-vote moments learned under complete teacher
forcing may encode the old causal packet distribution. Clearing only those
moments at handoff could let the reheated router adapt faster while preserving
lexical, lane/core, and private-node history.

### Changed factor

Twelve AdamW states were removed from the otherwise identical teacher
checkpoint: router trunk/key/occupancy/bias, learned ingress key/bias,
address-to-edge routing, relative-node bias, and terminal vote/query/address
parameters. The remaining 15 of 27 states migrated normally. Model weights,
the 1.30x private-node rate, all scalar optimizer settings, full reheat, loss,
data, and controller remained fixed.

### Result

Fresh route state made the transition substantially more abrupt. The first
route/gate update norm rose from 0.02210 in the preserved-state arm to 0.04858,
and its update ratio rose from 1.82e-3 to 4.00e-3. That burst did not buy better
adaptation. The controller released its second slot at branch step 2,500 and
finished there.

At global step 7,500, teacher content PPL was 37.87 with 23.92% accuracy and
93.60% coverage. Partial content PPL was 36.54 with 25.00% accuracy and 89.62%
coverage. Fully unsupported content PPL was 127.33 with 7.88% accuracy and
30.73% coverage. Against preserved-state Test 04c, teacher PPL worsened 6.2%,
partial PPL worsened 5.0%, and free PPL worsened 18.3%. The higher free
formation coverage was not enough to offset the poorer token distribution.

The L4 completed in 116.37 trainer seconds (120.61 seconds including harness
overhead), ending at 25.78 steps/second. Peak host RSS was 2,066 MiB; peak
allocated/reserved CUDA memory was 73/92 MiB. The downloaded ZIP SHA-256 is
`03f5e0dfb4a01a687c07f81f2ba2ac7079fc2236f48e8fc9f9f8b2bc8fd9f31f`.

**Decision:** reject route/gate reset. Teacher-phase moments are useful damping
for the deliberately reheated route distribution. Preserve complete optimizer
state at handoff; test a complete reset only as the remaining boundary case.

## Test 05b -- complete optimizer-state reset

### Hypothesis

A completely fresh AdamW state is the opposite boundary from full continuity.
It may adapt rapidly to unsupported inputs, but it removes all damping learned
during teacher acquisition and should reveal whether fast free-token progress
is merely destructive phase-transition shock.

### Changed factor

The model and exact data state resumed from the shared step-4,500 checkpoint,
but `--resume-optimizer` was removed. All 27 AdamW states were therefore fresh.
The same grouped rates, including private-node LR 0.0013, full route reheat,
loss coefficient 2, controller, and 3,000-step horizon were retained.

### Result

The first update confirmed the expected shock. Relative update ratios were
5.13e-3 lexical, 4.58e-3 lane/core, 6.13e-3 private-node, and 4.00e-3
route/gate--roughly three to six times the corresponding preserved-state
updates. Lexical parameter norm then expanded from 24.55 to 40.38 over the
branch, far more than in any continuity arm.

The controller advanced at branch steps 1,000, 1,750, and 2,500, ending at
four released slots. Its best retained primary checkpoint was global step
6,250, where teacher content PPL was already 39.65 and partial content PPL was
40.60. At the final step, teacher content PPL reached 44.06 with 21.70%
accuracy, while partial content PPL was 41.89 and coverage had fallen to
83.76%.

Fresh state did fit autonomous tokens quickly: final fully unsupported content
PPL was 106.08 with 8.89% accuracy and 34.43% coverage. This narrowly beats the
preserved arm's 107.67 free PPL, but it does so after twice the evaluated slot
exposure and with catastrophic teacher regression. The unsupported checkpoint
selector retained step 7,250 because its overall PPL was fractionally better;
the final checkpoint is the content-PPL minimum.

The L4 completed in 119.71 trainer seconds (123.96 seconds including harness
overhead), ending at 25.06 steps/second. Peak host RSS was 2,066 MiB; peak
allocated/reserved CUDA memory was 73/92 MiB. The downloaded ZIP SHA-256 is
`22416ea53b6dee8fdbf7796141b5d1549043124f4a983864006db0d9fb1c4fdf`.

**Decision:** reject complete reset and select full AdamW-state continuity.
State reset can manufacture better free PPL by accelerating release while
destroying the causal substrate; it is misaligned with Gator's staged handoff
intent.

## Test 06a -- aggressive late learning-rate drop

### Hypothesis

The teacher-acquisition run reached its best content PPL well before its fixed
learning-rate endpoint, while the selected handoff treatment continued moving
the node-local weights aggressively. A late multiplicative reduction should
limit post-peak damage without suppressing the early routing and unsupported-
token adaptation needed at handoff.

### Changed factor

This arm retained the selected slot-normalized coefficient 2, full route
reheat, 1.30x private-node learning rate, and all 27 migrated AdamW states. A
milestone scheduler reduced every parameter group's rate by 0.35 at branch
step 2,250 of 3,000. Shared groups therefore moved from 0.001 to 0.00035 and
private nodes from 0.0013 to 0.000455, preserving the component-rate ratio.
All model, data, controller, compilation, and evaluation controls matched Test
04c.

### Result

The L4 completed in 118.32 trainer seconds (122.60 seconds including harness
work), sustaining 25.35 steps/second. Peak host RSS was 2,079 MiB; peak
allocated/reserved CUDA memory was 73/92 MiB. The downloaded ZIP SHA-256 is
`ac71d92e3c1f78dac03b98271123153a53116f06b2a4bb74fe54892ce40b8678`;
the latest checkpoint SHA-256 is
`13162ba8a836237b52cdfdc2e1eb14e39441b9ee66ba89eaa44d629fbd8f39c8`.

Retention improved decisively. At global step 7,500, teacher content PPL was
31.10 with 25.34% top-1 accuracy, 52.16% top-5 accuracy, and 93.06% formation
coverage. Partial content PPL was 33.26 with 24.80% accuracy, 51.42% top-5
accuracy, and 91.24% coverage. Relative to constant-LR Test 04c, teacher PPL
improved 12.8% and partial PPL improved 4.4%.

The same treatment underfit autonomous formation. Fully unsupported content
PPL was 160.22 with 7.88% accuracy, 23.38% top-5 accuracy, and 24.80%
coverage, versus 107.67, 8.83%, and 27.22% in Test 04c. The controller held at
one released slot for the entire branch. The drop therefore protected the
teacher substrate partly by preventing the model from reaching the next
unsupported exposure level, not by improving both objectives simultaneously.

The result also clarifies a scheduler implementation constraint. A single
multiplicative milestone preserves the intended 1.30:1 private-node/shared
rate ratio. A stock cosine scheduler with one absolute minimum learning rate
would collapse those group ratios near its floor and silently undo the
component policy. Production decay should use multiplicative, per-group
floors or equivalent ratio-preserving schedules.

**Decision:** keep 0.35x as the retention-side bound, not as the optimizer
winner. Test a 0.70x milestone at the same branch step. If the midpoint still
blocks autonomous learning, prefer the constant node-priority rate until the
drop can be triggered by an absolute teacher-retention/controller event rather
than elapsed steps alone.

## Test 06b -- gentler late learning-rate drop

### Hypothesis

A 0.70x milestone at the same branch step should preserve more late node and
router adaptation than Test 06a while still damping the constant-rate arm's
retention drift. It is the midpoint test between the aggressive 0.35x bound
and no decay.

### Changed factor

Only the milestone multiplier changed from 0.35 to 0.70. Shared group rates
fell from 0.001 to 0.0007 at branch step 2,250, while the private-node rate
fell from 0.0013 to 0.00091. Full optimizer-state continuity, component rate
ratios, route reheat, loss coefficient 2, and every model/data control remained
fixed.

### Result

The L4 completed in 121.09 trainer seconds (125.37 seconds including harness
work), sustaining 24.78 steps/second. Peak host RSS was 2,066 MiB; peak
allocated/reserved CUDA memory was 73/92 MiB. The downloaded ZIP SHA-256 is
`01749b5211237ad1490c6b40c9e5cfd137e5d7b02108e654bb8c704d01c588f0`;
the latest checkpoint SHA-256 is
`2f3904d0884b35b9ad9dbd4ab1e006109b060cda4c4316c1435ec0cfda39fcbf`.

The controller advanced to two released slots at branch step 1,750, before the
milestone, and to three at branch step 2,500, after it. The best primary
checkpoint was global step 6,250: teacher content PPL 36.42, 22.71% accuracy,
49.73% top-5 accuracy, and 91.71% coverage; partial content PPL 38.03, 23.05%
accuracy, and 89.96% coverage. Its fully unsupported content PPL was 163.21.

By global step 7,500, teacher content PPL was 38.37 with 22.37% accuracy and
94.41% coverage. Partial content PPL was 38.44 with 22.71% accuracy and 88.14%
coverage. Fully unsupported content PPL improved to 143.21, but accuracy was
only 7.35% and formation coverage 21.36%.

Constant-rate Test 04c remains better on both objectives: teacher content PPL
35.66 versus 38.37 and free content PPL 107.67 versus 143.21 at the matched
endpoint. The gentler decay neither retained the 0.35x arm's protection nor
matched constant-rate autonomous learning. The hard handoff transitions make
elapsed-time milestones especially brittle because a drop can land before or
after a discrete exposure change.

**Decision:** reject fixed 0.70x decay and select constant LR for the current
controller. For the reusable training template, decay must be event-gated:
hold the acquisition/early-handoff rate until unsupported competence advances,
then reduce it only when an absolute teacher-PPL or formation-retention guard
detects drift. Preserve each component group's multiplicative LR ratio during
that reduction.

## Test 07a -- larger AdamW epsilon

### Hypothesis

Hard top-k routing amplified tiny kernel-level differences in Test 04a, so a
larger AdamW epsilon could reduce denominator sensitivity and stabilize the
handoff trajectory. The treatment is useful only if it preserves autonomous
node adaptation; numerical damping by itself is not the objective.

### Changed factor

AdamW epsilon changed from `1e-8` to `1e-6` for all four component groups.
The selected constant learning rates, including private-node LR 0.0013, all
27 migrated moments, full reheat, normalized loss coefficient 2, controller,
and model/data controls remained fixed. The saved optimizer confirms epsilon
`1e-6` in every group after state migration.

### Result

The L4 completed in 118.33 trainer seconds (122.57 seconds including harness
work), sustaining 25.35 steps/second. Peak host RSS was 2,080 MiB; peak
allocated/reserved CUDA memory was 73/92 MiB. The downloaded ZIP SHA-256 is
`e420184b3b7b67c432fdbfd0754d02582b86b3be2d92daca4e584425b691f773`;
the latest checkpoint SHA-256 is
`43f033cda07355706b22425f2dca41ec395d3d05e1616d081b19d31e746ae5e5`.

The larger epsilon stabilized the supported path but suppressed transition
learning. At global step 7,500, teacher content PPL was 34.73 with 24.06%
accuracy, 53.37% top-5 accuracy, and 91.85% formation coverage. Partial
content PPL was 37.68 with 23.32% accuracy and 89.29% coverage. Fully
unsupported content PPL was 180.60 with 6.33% accuracy, 22.57% top-5 accuracy,
and 25.67% coverage. The controller never advanced beyond one released slot.

Relative to default-epsilon Test 04c, teacher PPL improved 2.6%, but partial
PPL worsened 8.3%, free PPL worsened 67.7%, and the controller made no
progress. Update ratios did not collapse globally, yet the resulting hard
routing path behaved as if the low-amplitude adaptations required for the
frontier had been damped. This is architectural evidence against choosing
epsilon solely for numerical smoothness.

**Decision:** reject `1e-6` and retain AdamW epsilon `1e-8`. No intermediate
epsilon sweep is warranted: the larger value strongly worsened the joint
objective, while the default is numerically finite and stable. Skip fused
AdamW, accumulation, and lazy-row treatments in this proxy because optimizer
updates are not the throughput bottleneck, gradients are stable at batch 16,
and top-k activity across batches touches enough nodes that lazy updates would
change semantics without a demonstrated compute gate.

## Node-only 130k transfer validation

The transfer test precedes architecture scaling so width, population, and
topology changes cannot hide whether the optimizer finding itself scales. The
model has no cells: it is the current flat Convergent graph with 40 private
nodes. Its exact configuration is V2048/E32/D24/N40/L10/degree3/R2, private
full routers, fixed ingress, active top-k 4, slot-conditioned terminal voting,
small-world seed zero, and 50% rewiring. The current parameter formula and
instantiated checkpoint both report **130,194 parameters**.

The planned trajectory is 20,000 total updates. The first 4,000 updates (20%)
are fully teacher forced and shared by both handoff branches. One branch then
uses monolithic AdamW; the matched treatment changes only the private-node
group to 1.30x. Both branches use slot-normalized coefficient 2, full route
reheat, complete optimizer-state continuity, epsilon `1e-8`, and constant LR.

### Test 08a -- shared 20% teacher phase

The compiled dense L4 run completed 4,000 updates in 178.84 trainer seconds
(190.40 seconds including compile and archive work), reaching 22.37
steps/second. Peak host RSS was 2,205 MiB; peak allocated/reserved CUDA memory
was 374/436 MiB. The best and latest checkpoints are both the final step.

Final overall PPL was 14.09. Content PPL was 39.42 with 20.05% top-1
accuracy, 50.75% top-5 accuracy, and 93.19% formation coverage. The curve was
still improving at the boundary, which is acceptable here: this checkpoint is
the deliberately early 20% handoff state, not a claimed teacher optimum.
Private-node updates remained useful but conservative at the endpoint: the
sampled private-node update ratio was `7.25e-4`, versus `1.17e-3` lexical,
`8.71e-4` lane/core, and `9.08e-4` route/gate.

The archive SHA-256 is
`84ea05f3fb2452f300343b3e053bd6d9c1be7a74a7437203d598997044452a8a`;
`best.pt` SHA-256 is
`bd7c559c0d355a48ddbceb9d83fb78ea28b50a58fb50bf09dbed80676cb9cfbe`.
One end-of-run copy was verified on Drive at
`MyDrive/GTR/experimental_checkpoints/optimizer_alignment_colab/08a_130k_n40_teacher20.zip`.

**Decision:** freeze this exact model, tokenizer, optimizer/data state, and
step-4,000 checkpoint for the monolithic-versus-node-priority transfer A/B.

### Test 08b -- monolithic AdamW handoff control

The control resumed the shared step-4,000 checkpoint with all model,
optimizer, controller, and data-pipeline state intact. It ran 16,000
additional token-handoff updates at constant LR `1e-3`, AdamW epsilon `1e-8`,
active-slot-normalized auxiliary coefficient 2, and the full branch-local
route reheat. All 130,194 parameters remained in one optimizer group.

The L4 completed the branch in 640.60 trainer seconds (645.50 seconds
including harness and archive work), sustaining 24.98 steps/second. Peak host
RSS was 2,189 MiB; peak allocated/reserved CUDA memory was 377.5/420 MiB.
The archive SHA-256 is
`dd8c1282e1a7899dce38fc36df5cf922cafe17c7e7e733bf01defc2881bfb98f`.
One verified end-of-run copy is on Drive at
`MyDrive/GTR/experimental_checkpoints/optimizer_alignment_colab/08b_130k_n40_handoff_monolithic.zip`.

The route reheat caused a large but recoverable initial shock. Primary PPL was
43.06 at global step 8,000, 30.16 at 10,000, 22.78 at 12,000, 16.68 at
16,000, 14.20 at 18,000, and 13.71 at 20,000. Fully unsupported content PPL
over those same later landmarks was 601.50, 358.03, 232.08, 230.31, 209.44,
and 223.00. The controller first advanced from one to two released slots at
global step 18,000; it did not reach a third slot.

At the final step, partially supported content PPL was 37.19 with 24.21%
top-1 accuracy, 55.81% top-5 accuracy, and 91.89% formation coverage. The
fully teacher-supported probe had content PPL 37.27, 24.03% accuracy, and
93.15% coverage. The fully unsupported probe had content PPL 223.00, 3.55%
accuracy, 22.99% top-5 accuracy, and only 11.30% formation coverage. The best
teacher checkpoint occurred at global step 19,500 with content PPL 36.63;
the selected unsupported checkpoint occurred at step 18,000 with free content
PPL 209.44.

**Interim decision:** monolithic AdamW is a competent supported-language
control, but it allocates too little adaptation to autonomous node behavior.
Its slow frontier transition and weak free-formation coverage leave substantial
room for the private-node-priority treatment. Do not select the treatment
until the matched Test 08c branch completes from the identical teacher and data
state.

### Test 08c -- private-node LR 1.30x transfer

This branch differed from Test 08b only by placing the 25,120 private-node
parameters in a native AdamW group at LR `0.0013`; lexical, lane/core, and
route/gate parameters remained at `0.001`. The trainer migrated all 22 Adam
moment states by parameter name and preserved exact model and data state. The
loss normalization, route schedule, epsilon, weight decay, clipping, and
controller were identical to the monolithic control. This was the first run
through the permanent first-class CLI path rather than the experimental probe
hooks.

The L4 completed 16,000 branch updates in 633.46 trainer seconds (642.63
seconds including child launch and archiving), sustaining 25.26 steps/second.
Peak child host RSS was 2,202 MiB. The archive SHA-256 is
`10fdde5746fa5262dea04e059bcb3197340ffb6a732c1fcd46a2aef2069b6d92`;
the verified Drive copy is
`MyDrive/GTR/experimental_checkpoints/optimizer_alignment_colab/08c_130k_n40_handoff_node130.zip`.

The treatment radically accelerated curriculum motion. It advanced to two
released slots at global step 10,500, then to three at 11,250, four at 12,000,
five at 12,750, six at 13,500, seven at 14,250, and eight at 15,250. It held
eight through step 20,000. Monolithic AdamW did not reach two slots until step
18,000 and never reached three.

At the final step, the fully unsupported probe reached content PPL 199.82,
5.06% accuracy, 24.03% top-5 accuracy, and 16.89% formation coverage. Its best
observed free content PPL was 170.24 at global step 19,750. Relative to the
monolithic endpoint, free accuracy improved 42%, formation coverage improved
50%, and the best observed free PPL improved 19%.

That autonomy gain was purchased with excessive forgetting. At step 20,000,
the fully teacher-supported probe had content PPL 71.99, 18.90% accuracy, and
93.58% coverage, versus PPL 37.27 and 24.03% accuracy for monolithic AdamW.
Partially supported content PPL was 53.40 with 19.87% accuracy and 81.53%
coverage, reflecting the much harder eight-slot frontier. Even the selected
step-19,750 checkpoint retained teacher content PPL only 66.03.

**Decision:** reject an unguarded permanent 1.30x private-node LR as the final
130k policy. It proves that the monolithic optimizer starves the autonomous
node pathway, but it crosses the retention frontier: teacher PPL nearly doubles
while the free-PPL gain remains modest. Treat `1.0x` and `1.30x` as the two
ends of a Pareto bracket. Test a moderated `1.15x` treatment from the same
checkpoint, then add an absolute teacher-retention gate if no static multiplier
simultaneously preserves supported language and advances autonomous slots.

### Test 08d -- private-node LR 1.15x transfer

This matched branch changed only the private-node multiplier from `1.30x` to
`1.15x`. It resumed the same step-4,000 model, optimizer, controller, and data
state as Tests 08b and 08c. The native grouped optimizer migrated all 22 Adam
moment states by parameter name. All loss, routing, clipping, and curriculum
settings remained fixed.

The L4 completed the 16,000-update branch in 649.21 trainer seconds (657.54
seconds including launch and archive work), sustaining 24.65 steps/second.
Peak child host RSS was 2,204 MiB. The archive SHA-256 is
`a172b459e8cba618128a48d88f6b3bf089878047e3e7b252db09f4ac66183628`;
the verified Drive copy is
`MyDrive/GTR/experimental_checkpoints/optimizer_alignment_colab/08d_130k_n40_handoff_node115.zip`.

Curriculum motion landed between the prior arms. The controller advanced to
two released slots at global step 11,750, three at 14,000, four at 15,750, and
five at 19,750. Thus `1.15x` released more than twice as many slots as the
monolithic control, but did not produce the rapid eight-slot cascade seen at
`1.30x`.

At step 20,000, partially supported content PPL was 47.39 with 21.31% top-1
accuracy, 52.22% top-5 accuracy, and 85.08% coverage. The fully
teacher-supported probe had content PPL 46.88, 20.77% accuracy, and 93.44%
coverage. The fully unsupported probe had content PPL 219.68, 4.30% accuracy,
21.92% top-5 accuracy, and 12.34% coverage. The best observed free content PPL
was 184.25 at global step 17,000 while four slots were released.

Relative to monolithic AdamW, `1.15x` improved final free accuracy by 21% and
advanced five rather than two slots, but final free PPL was essentially flat
and teacher PPL rose 26%. Relative to `1.30x`, it cut teacher PPL by 35% while
retaining meaningful curriculum acceleration, although it surrendered some
free accuracy and coverage. The endpoint immediately follows the fifth-slot
release, so its partial-support metric includes a fresh frontier shock.

**Decision:** retain `1.15x` as the best static policy observed so far, but do
not finalize it yet. Run the predeclared `1.20x` midpoint from the identical
checkpoint. Select on the joint frontier of full-teacher retention, unsupported
quality, released-slot progress, and throughput rather than slot count alone.

### Test 08e -- private-node LR 1.20x transfer

This branch changed only the private-node multiplier to `1.20x`, again
resuming the exact shared step-4,000 checkpoint with all 22 Adam moment states
migrated by name. The L4 completed 16,000 updates in 638.06 trainer seconds
(646.48 seconds including launch and archive work), sustaining 25.08
steps/second. Peak child host RSS was 2,205 MiB. The archive SHA-256 is
`f39c743aa7341d12d461438674cb3bffd13addb3decbd3a085bef41d2b726fea`;
the verified Drive copy is
`MyDrive/GTR/experimental_checkpoints/optimizer_alignment_colab/08e_130k_n40_handoff_node120.zip`.

The curriculum advanced faster than either neighboring treatment. It reached
two released slots at global step 10,250, three at 11,500, four at 12,250,
five at 14,500, six at 16,750, seven at 17,750, eight at 18,500, and nine at
19,250. The late advances occurred while absolute teacher-frontier accuracy
fell toward `0.15`; because unsupported and teacher-frontier accuracy declined
together, their ratio remained close to one and continued satisfying the
relative controller.

At step 20,000, partially supported content PPL was 54.49 with 19.40% top-1
accuracy, 48.06% top-5 accuracy, and 77.83% coverage. The fully
teacher-supported probe had content PPL 70.79, 19.23% accuracy, and 93.11%
coverage. The fully unsupported probe reached content PPL 163.09, 4.84%
accuracy, 24.89% top-5 accuracy, and 15.78% coverage. The best observed free
content PPL was 160.70 at global step 19,750; that checkpoint retained teacher
content PPL 63.42.

The autonomy result is the strongest in this static-rate sweep: final free PPL
improved 27% over monolithic AdamW and 18% over `1.30x`. Nevertheless,
full-teacher PPL was 90% worse than monolithic and essentially matched the
`1.30x` collapse. The non-monotonic curriculum timing also shows that the
`1.30x` arm's rapid release was not simply proportional to node learning rate.

**Decision:** no permanent static multiplier spans the desired frontier.
`1.15x` is the best unguarded static policy; `1.20x` is the best adaptation
rate but requires retention control. The controller's relative frontier ratio
is insufficient because both numerator and denominator can collapse together.
Add a native full-teacher content-PPL guard, anchored to the pre-handoff
baseline, then rerun `1.20x` with advancement capped at `1.25x` baseline PPL
and retreat triggered at `1.35x`. This is an event-gated curriculum response,
not a scheduled optimizer decay.

### Objective correction -- free generation is primary

The deployment objective is fully unsupported generation, not preservation of
teacher-forced perplexity. Teacher PPL remains a useful diagnostic of pathway
retention, but it is not a hard selection constraint. A retention guard should
only be adopted if it improves eventual free PPL, free accuracy, or free
formation coverage. Consequently, the planned guarded branch was deferred and
the private-node LR bracket was narrowed directly at `1.25x`.

### Test 08f -- private-node LR 1.25x transfer

This branch changed only the private-node multiplier to `1.25x` from the same
step-4,000 teacher checkpoint and exact data/optimizer state. It completed
16,000 branch updates in 645.43 trainer seconds (653.84 seconds including
launch and archive work), sustaining 24.79 steps/second. Peak child host RSS
was 2,206 MiB. The archive SHA-256 is
`c856e8d0550f56315592998d78c5dee6a7b845a441d9929b8923f4705a02c45d`;
the verified Drive copy is
`MyDrive/GTR/experimental_checkpoints/optimizer_alignment_colab/08f_130k_n40_handoff_node125.zip`.

The controller reached two released slots at global step 11,000, three at
12,000, four at 12,750, five at 13,500, six at 14,250, seven at 15,250, eight
at 16,250, nine at 18,750, and ten at 19,500. Thus it eventually transferred
faster than `1.20x` without the unstable early cascade of `1.30x`.

At step 20,000, partially supported content PPL was 56.50 with 18.22% top-1
accuracy and 76.04% coverage. Fully teacher-supported content PPL was 75.82
with 18.08% accuracy. Fully unsupported content PPL was 159.06 with 5.02%
top-1 accuracy, 24.68% top-5 accuracy, and 16.68% coverage.

The strongest free checkpoint occurred at global step 18,750, evaluated with
eight removed slots immediately before the controller advanced to nine. It
reached free content PPL 155.97, 5.09% top-1 accuracy, 25.00% top-5 accuracy,
and 19.62% formation coverage. It improves three of four free metrics over the
best `1.20x` checkpoint (PPL 160.70, 4.84% top-1, 25.29% top-5, and 16.18%
coverage), with only a small top-5 difference favoring `1.20x`.

**Decision:** provisionally select private-node LR `1.25x` for the free-first
objective. Continue the step-18,750 checkpoint with route temperature held at
its terminal `0.35`, branching constant `1.25x` against a `1.0x` node-LR
consolidation arm. This tests whether ongoing node-priority adaptation or a
post-transfer rate normalization produces the better long-horizon free result.

### Test 09a -- permanent 1.25x private-node continuation

The strongest Test 08f checkpoint at global step 18,750 was continued for an
identical 10,000 updates with route temperature fixed at `0.35`, exact data and
optimizer state restored, and the private-node group retained at `1.25x`.
Training took 401.63 seconds at 24.90 steps/second (409.94 seconds including
launch and archival). The archive SHA-256 is
`6d229b5f8e35fb113193e7e1507cc26c78d7ff0ffb56a080c289a503c843ce33`;
the verified Drive copy is
`MyDrive/GTR/experimental_checkpoints/optimizer_alignment_colab/09a_130k_node125_continue10k.zip`.

The controller advanced from nine to ten removed teacher tokens at step
21,500 and then stalled. At the matched global step 28,750, free content PPL
was 183.33 with 5.38% top-1 accuracy, 25.32% top-5 accuracy, and 23.64%
formation coverage. The best observed free content PPL was 154.67 at step
19,250. Across all 40 matched validation points from steps 19,000 through
28,750, trapezoidal mean free cross-entropy corresponds to geometric-mean PPL
180.40; mean free top-1 accuracy was 5.215%.

### Test 09b -- all-1.0x consolidation

The exact same step-18,750 checkpoint was independently continued for 10,000
updates with every component group at `1.0x`. Route temperature, data order,
optimizer moments, controller state, and all other controls matched Test 09a.
Training took 400.33 seconds at 24.98 steps/second (408.72 seconds including
launch and archival). The archive SHA-256 is
`70b58956436b6ac8ed7c62d0b70622083fed1c0bc54eb3cd364b4f5fa4af8b1b`;
the verified Drive copy is
`MyDrive/GTR/experimental_checkpoints/optimizer_alignment_colab/09b_130k_node100_consolidate10k.zip`.

This branch advanced to eleven removed tokens by step 23,250. At step 28,750,
free content PPL was 196.92 with 5.60% top-1 accuracy, 25.07% top-5 accuracy,
and 22.96% formation coverage. Its best saved unsupported checkpoint occurred
at step 19,250 with free content PPL 156.81. Its matched-curve geometric-mean
PPL was 180.74 and mean free top-1 accuracy was 5.158%.

The two curves are effectively tied: `1.25x` had lower free PPL at 19 of 40
matched steps and `1.0x` at 21 of 40, while their geometric-mean PPL differed
by only 0.19%. The permanent `1.25x` branch had slightly better curve-average
PPL and accuracy; `1.0x` released one additional slot and had better endpoint
top-1 accuracy. Neither produced sustained free-PPL improvement over the
source checkpoint.

**Decision:** retain `1.25x` as the established initial handoff/transfer rate,
but do not claim that either permanent `1.25x` or all-`1.0x` is an optimal
late continuation. Run two same-checkpoint controls: (1) preserve the same
relative `1.25:1` node priority at a lower absolute scale by setting private
nodes to `1.0x` and all shared/route groups to `0.8x`; and (2) directly test
slower node learning with private nodes at `0.8x` and all other groups at
`1.0x`. Rank both only by matched-step free metrics.

### Test 09c -- relative 1.25:1 node priority at 0.8x global scale

This branch preserved the same relative node advantage as Test 09a while
lowering its absolute scale: private nodes used `1.0x`; lexical, lane/core,
and route/gate groups used `0.8x`. It resumed the identical step-18,750 model,
all 22 AdamW states, exact data position, controller state, and fixed `0.35`
route temperature. Training took 404.66 seconds at 24.71 steps/second (413.09
seconds including launch and archival). The archive SHA-256 is
`81ae4de3429e581234a5c272c070532029a371aea0860eec195f953227443d83`;
the verified Drive copy is
`MyDrive/GTR/experimental_checkpoints/optimizer_alignment_colab/09c_130k_relative_node125_global080_10k.zip`.

The controller advanced to ten removed tokens at step 20,500 and eleven at
21,750. At global step 28,750, free content PPL was 191.64 with 5.16% top-1
accuracy, 24.82% top-5 accuracy, and 23.67% formation coverage. The lowest
logged free content PPL was 156.96 at step 21,250. Its matched-curve
geometric-mean PPL was 180.03 and mean free accuracy was 5.123%.

The lower global scale changed curriculum timing but did not create a clear
free-quality separation. It beat permanent `1.25x` on free PPL at 19 of 40
matched steps, and its geometric-mean advantage was only 0.20%; permanent
`1.25x` retained the better curve-average accuracy. Thus AdamW's absolute
scale matters enough to alter the trajectory, but preserving a `1.25:1`
relative node ratio is not itself the missing long-horizon policy.

**Decision:** do not select Test 09c. Complete the direct slower-node branch
before deciding whether late handoff should use a relative node disadvantage
or simply stop adapting the continuation at its best free checkpoint.

### Test 09d -- private nodes 0.8x, shared and route groups 1.0x

This direct driver/car test changed only the private-node multiplier relative
to the all-`1.0x` Test 09b control. Private nodes used `0.8x`; lexical,
lane/core, and route/gate groups remained at `1.0x`. The branch restored the
same step-18,750 model, all optimizer moments, exact data position, controller
state, and fixed `0.35` route temperature. Training took 407.88 seconds at
24.52 steps/second (416.25 seconds including launch and archival). The archive
SHA-256 is
`35b7c650e71504b0d26bec399a1386232088e9fe33e839e3665b1f0509f47f1a`;
the verified Drive copy is
`MyDrive/GTR/experimental_checkpoints/optimizer_alignment_colab/09d_130k_node080_shared100_10k.zip`.

The controller advanced to ten removed tokens at step 21,500 and eleven at
24,000. At global step 28,750, free content PPL was 193.86 with 5.31% top-1
accuracy, 25.54% top-5 accuracy, and 21.99% formation coverage. The endpoint
itself remained noisy and worse than the source checkpoint, but the complete
matched curve separated clearly from the controls.

Across steps 19,000 through 28,750, private-node `0.8x` achieved geometric-mean
free PPL 176.81, versus 180.40 for permanent node `1.25x`, 180.74 for all
`1.0x`, and 180.03 for node `1.0x`/shared `0.8x`. It had lower PPL than the
all-`1.0x` control at 33 of 40 matched checkpoints. Its mean free-CE advantage
over all-`1.0x` was 0.02173 nats, equivalent to a 2.15% geometric-PPL gain.
The late-window geometric-mean PPL was 184.73, versus 190.49, 191.83, and
191.04 for the other three branches. Mean free accuracy was 5.195%, and its
endpoint top-5 accuracy was the strongest of the four continuations.

No continuation improved sustainably over the step-18,750 source's free PPL;
all eventually drifted upward as training continued. Therefore Test 09d is a
better late stabilizer, not evidence that unlimited continuation is useful.

## Selected optimizer and training policy

The evidence supports a phase-specific optimizer rather than one permanent
multiplier:

1. **Teacher foundation:** all component groups at `1.0x` for the first 20%
   of the initial 20k template (4,000 updates here).
2. **Autonomy transfer:** private nodes at `1.25x`, all shared and route groups
   at `1.0x`, with active-slot-normalized free loss and staged route
   exploration. This was the strongest 130k policy at the matched 20k point.
3. **Route-led stabilization:** after the route-temperature schedule reaches
   `0.35` and the initial transfer plateaus, reduce private nodes to `0.8x`
   while retaining lexical, lane/core, and route/gate groups at `1.0x`.
4. **Free-first stopping:** select and stop by matched fully unsupported PPL,
   accuracy, top-5, and coverage. Teacher PPL remains telemetry only. If free
   PPL fails to improve over a patience window, preserve the best-free
   checkpoint rather than continuing merely to release more curriculum slots.

At this scale, the empirical switch region is global step 18k--20k, near nine
removed teacher tokens and after route temperature reaches its terminal value.
The architecture scaling screen will compare every candidate at exactly step
20,000 using the established first two phases. Longer promoted candidates will
then use the `0.8x` private-node stabilization phase. This preserves a clean
matched-step architecture comparison while applying the late optimizer result
where the ablation actually supports it.

### Cross-width validation -- reject `1/sqrt(D)` node-LR normalization

The 130k scaling screen subsequently tested whether the `1.25x` transfer rate
was accidentally tuned to D24. N40/D28 was replayed from its exact step-4,000
teacher checkpoint at `1.157275x`, while N40/D20 was replayed at `1.369306x`;
all other state and the 20,000-step horizon were unchanged.

Lowering D28 produced only a `0.00857`-nat endpoint gain (`184.44` to `182.87`
free PPL), while worsening curve CE from `5.74875` to `5.83924`, best PPL from
`176.07` to `182.22`, and final curriculum release from five tokens to two.
Raising D20 was clearly destructive: endpoint free PPL worsened from `173.45`
to `222.96` (`0.2511` nats), and the controller advanced once before failing
retention and retreating. Teacher PPL remained nearly unchanged (`34.65` to
`35.35`), again showing why it cannot select the optimizer.

The 64-point matched curves localize the failures. Lower-rate D28 won only 27
points and only 17 of the 41 points where both branches had identical
one-token visibility. Higher-rate D20 showed a small roughly `0.012`-nat
advantage in each early/middle window, then lost `0.0599` mean CE after step
15,250 and won only 6 of those final 20 comparisons. Thus the thin-node rate
is not too high to start learning; it is too high to remain a stable
long-horizon substrate.

**Decision:** keep the phase-specific policy above without a width-derived
multiplier. In particular, do not increase the private-node rate merely
because a node is thinner. The selected `1.25x` is an autonomy-transfer cap at
this scale, followed by `0.8x` stabilization; it is not a matrix-size
normalization law. Recalibrate rates at a materially larger budget only with
matched free-PPL curves and fixed-visibility controls.

### Curriculum control -- fixed one-slot visibility is not a universal baseline

The 130k matched-parameter diagonal was replayed from the same step-4,000
teacher checkpoints with exactly one response slot unsupported through step
20,000. Relative to dynamic release, endpoint free content PPL changed from
`171.13` to `243.92` at N40/D24, from `230.37` to `198.06` at N24/D28, and
from `196.25` to `200.63` at N64/D20. Curve-average free CE changed by
`+0.02166`, `-0.22276`, and `-0.01110` nats respectively (lower is better).

Teacher retention again failed as a selector. Fixed-one improved teacher PPL
from `43.81` to `34.28` at N40/D24 while making endpoint free PPL `42.5%`
worse; N64/D20 similarly improved teacher PPL slightly while losing free
quality. Only the fat-node N24/D28 architecture benefited, and it did so after
a long delayed transition rather than by maintaining a uniformly stronger
curve.

**Decision:** do not replace adaptive handoff with a universal fixed-exposure
foundation. Keep the first 20% fully teacher-forced, then permit adaptive
release, select by matched free metrics, and checkpoint the best-free state.
For scaling studies, treat curriculum response as an architecture interaction:
use fixed-visibility replay as a diagnostic control, not as the production
training policy.

### Cross-budget transfer -- 500k balanced architecture

The first two selected phases were transferred unchanged to a
coverage-normalized approximately 500k-parameter diagonal. N54/D63/L15,
N82/D49/L23, and N126/D39/L34 all used 4,000 all-`1.0x` teacher updates,
followed by 16,000 adaptive-handoff updates with private nodes at `1.25x`, all
other groups at `1.0x`, active-slot-normalized loss, and staged route reheat.

The balanced N82/D49 arm validated the transfer policy. It advanced to 12
released slots with no retreat, reached best logged free content PPL `147.07`,
and achieved curve CE `5.36958`. It beat the fatter N54/D63 arm at 62 of 64
common validation steps and the thinner N126/D39 arm at 53 of 64. The low-N
arm did not reach its second released slot until step 14,250 and ended with
five; the high-N arm plateaued at six. Architecture therefore changes the
handoff timescale even when total, reachable-private, node-plus-route, and
coverage budgets are matched.

The N54/D63 endpoint of `159.29` narrowly beat N82/D49's `161.57`, but only
after a one-step terminal surge. Over the final 16 validation points their
geometric-mean free PPLs were `196.85` and `162.47`. This reinforces the
selected free-first *curve and checkpoint* policy: do not tune optimizer rates
to a single noisy endpoint, teacher PPL, or release count. The experiment does
not overturn the rejection of `1/sqrt(D)` normalization and does not justify
inventing a new width formula from one diagonal. It does support using the
adaptive release trajectory itself as an architecture-sensitive signal for
when transfer has plateaued and the `0.8x` stabilization phase should begin.

The run also exposed a checkpoint-objective mismatch. The trainer's
`best_unsupported.pt` minimizes unsupported slot CE, including blank/formation
behavior, whereas the deployment objective under discussion is free content
PPL. Logged content-PPL minima were therefore not preserved by that file.
Retain the formation-aware checkpoint, but additionally preserve a separate
best fully unsupported content-CE checkpoint and use both metrics in final
selection. This is an objective-alignment correction, not a reason to discard
slot CE during training.

## Full-autonomy 1M diagnosis: slot isolation and component LR sweep

The later 1M investigation deliberately removed the curriculum as a confound.
Its N109/D64/L32/R2 model used vocabulary 2,048, degree-three small-world
routing, private full routers, fixed ingress, active top-k four, 32 response
slots, and 999,052 parameters. All 31 causal content slots were autonomous
from initialization. Free content PPL is pooled token NLL over all valid free
content positions; it is not an arithmetic mean of independently exponentiated
slot PPLs.

Two uninterrupted controls first established that the handoff transition was
not the sole failure:

- The teacher-only run improved to content PPL `27.10` at step 10,000, but
  then regressed monotonically to `444.20` at step 50,000 while content
  accuracy stayed near 25%. Constant-rate training is unstable even within
  one regime.
- The all-open shared-node run worsened from free content PPL `165.15` at
  step 5,000 to `208.42`, `344.94`, `647.93`, and `1,040.19` at steps
  10k, 15k, 20k, and 25k, ending at `5,886.18` at 50k. Accuracy and coverage
  rose while calibration collapsed. This is aggressive emission, not useful
  free-language convergence.

### Slot-owned private computation A/B

The architectural A/B divided all 109 private nodes into deterministic,
disjoint slot-owned pools while preserving shared physical routing and causal
node memory. Thirteen slots receive four private nodes and nineteen receive
three. Each slot also receives a private terminal-jury bias and evidence gain,
reusing the existing terminal-query parameter storage, so the parameter count
remains exactly 999,052 and active compute remains top-k four.

At step 5,000, slot isolation improved free PPL from `165.15` to `151.72`,
accuracy from 3.1% to 5.2%, top-5 to 25.7%, and coverage from 12.0% to 19.8%
at essentially unchanged throughput. At step 10,000 it reached 49.7% coverage
and 7.3% accuracy, versus 18.7% and 3.8% for the shared model, but PPL was
`328.34` versus `208.42`. By step 15,000, coverage reached 67.3% and accuracy
7.6%, while PPL overshot to `2,354.49` and terminal-query RMS reached `1.291`.

At matched approximately 50% coverage, isolation is dramatically better:
the isolated model reaches it near step 10k at PPL `328.34`; the shared model
does not reach it until about step 40k, at PPL `3,686.16`. Isolation therefore
removes a major cross-slot interference bottleneck and accelerates coherent
specialization, but the inherited learning rates turn that speed into severe
calibration overshoot.

The exact isolated anchors are stored under
`experimental_checkpoints/gator1m_n109_d64_l32_slot_isolated_seed0_50k`.
Their step-5k, 10k, and 15k archive SHA-256 values are respectively
`b17e7d8285b849685de3647283d8bc8c8a4b039e4df6c20974bb59ead4f44655`,
`ccb752afcf18c8ccc90ab1eb2077d750b64c838a389ff84b4527897946bbf965`,
and `fa4bd6ae0360145c548dfac3ceddb2d5ae0a04ab3ec3052c7fe73999551715dd`.

### Matched one-factor LR sweep

Every treatment below restored the exact isolated step-5k model, all 22 AdamW
states, and the exact portable data cursor, then trained to step 10,000. The
control multipliers were lexical `1.0`, lane/core `1.0`, private node `1.25`,
and route/gate `1.0` on base LR `0.001`. One component alone was halved or
doubled in each branch. The route/gate group currently includes the transport
router and terminal jury.

| Treatment | Changed multiplier | Free PPL | vs control | Free acc. | Top-5 | Coverage | Teacher PPL | Query RMS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Matched control | none | 328.34 | -- | 7.3% | 24.6% | 49.7% | 10,098.64 | 0.699 | reference |
| Route down | route `0.5` | 341.63 | +4.0% | 7.6% | 24.1% | 51.3% | 10,964.60 | 0.544 | reject; less jury growth without a PPL gain |
| Node down | private `0.625` | **273.88** | **-16.6%** | 7.2% | 23.5% | 48.1% | **4,419.87** | 0.652 | **clear winner** |
| Lane down | lane `0.5` | 345.53 | +5.2% | 7.0% | 24.5% | 51.5% | 8,731.96 | 0.673 | reject |
| Lexical down | lexical `0.5` | **287.99** | **-12.3%** | 7.0% | 24.1% | 45.2% | 6,406.09 | 0.755 | clear secondary win |
| Route up | route `2.0` | 329.43 | +0.3% | 7.3% | 24.0% | 51.3% | 9,098.25 | 0.873 | neutral PPL; increases jury magnitude |
| Node up | private `2.5` | 351.60 | +7.1% | 7.7% | 24.1% | 54.1% | 11,280.20 | 0.719 | reject; faster emission, worse calibration |
| Lane up | lane `2.0` | 310.54 | -5.4% | 7.3% | 24.6% | 48.0% | 23,584.63 | 0.692 | free-only gain, severe retention damage |
| Lexical up | lexical `2.0` | 410.70 | +25.1% | 7.4% | 23.7% | 54.3% | 30,029.24 | 0.616 | strongly reject |
| Combined validation | lexical `0.5`, private `0.625` | **247.28** | **-24.7%** | 6.5% | 24.6% | 40.9% | **2,563.23** | 0.704 | **combined winner** |

The symmetric node result is the strongest causal evidence in the study:
slower private nodes improve free PPL by 16.6%, while faster private nodes
worsen it by 7.1%. The analogous lexical result points in the same direction.
Higher accuracy and coverage in the high-rate branches do not offset their
worse PPL; they are becoming more confident and more willing to fill slots,
not learning a better conditional distribution.

This result qualifies the earlier 130k gradual-handoff policy. Private-node
priority helped while most slots remained teacher-supported, but it fights the
fully autonomous, slot-isolated regime. The trainer therefore needs a
regime-specific optimizer: faster node acquisition may remain useful during
supported transfer, followed by a decisive node and lexical slowdown when
broad autonomy is active. The final matched combination branch verifies that
the two independent gains are compatible: PPL improves another 9.7% beyond
node-down alone and 14.1% beyond lexical-down alone. Its reduced coverage and
unchanged top-5 show that the optimizer is trading premature emission for
substantially better calibration, which is the intended free-PPL objective.

Continuing the combined winner unchanged to step 15,000 produced the first
broad-coverage fork: 62.7% coverage, 7.4% accuracy, 23.6% top-5, and free PPL
`772.12`. PPL still regressed from step 10k, but remained 67.2% below the
unchanged isolated control's `2,354.49` at the same step. Terminal-query RMS
nevertheless reached `1.314`, essentially reproducing the control's jury
growth. Slower node and lexical learning therefore delays and reduces the
collapse but does not remove the remaining route/jury instability. The 15k
archive SHA-256 is
`fc4b52b941b40733d99defbcb9c636e53e08b85aba55551bb27f267a91c192d3`.

The logged 62.7% value is content-formation coverage, not route coverage, so a
native route trace was run before declaring the fork ready. At the exact
step-15k route temperature (`0.988658900`), all 104 statically reachable nodes
received validation route mass. Entropy-effective utilization was 56.04 nodes
at step 5k, 55.91 at 10k, and 57.72 at 15k; the nodes carrying 95% of route
mass rose from 55 to 58, while the largest node held only 6.11% at 15k. This is
genuinely broad node-route utilization. The complete diagnostic is
`experimental_checkpoints/gator1m_slot_lr_sweep/route_utilization_5k_15k.json`
with SHA-256
`be3c9513b962642b0c9e7e3aa014a654e397bf1791904535db495f7433ad62a9`.

Probe archives are stored under
`experimental_checkpoints/gator1m_slot_lr_sweep`. In table order after the
control, their SHA-256 values are:

- route down: `3978ac5ba629a8d99799bd2cb6537480ee8b0a0c999051d8fa22ea53644e679e`
- node down: `895a4192ab44dd30e9c172fbd47e6d3bbfd468cc5986b2a043f56e9714614628`
- lane down: `eba7937f2ac3745165fc0b4ac2527e8940252ceff1ebd3146df12a06a186f64d`
- lexical down: `dc7984f1ddd21e0646f25b5f79906983364a0377df3ca65f0b8b38f79c225c01`
- route up: `0bc74c952269282bff26fcf0fd42db9cc9faac422a9801d067e38c3f0e01ba92`
- node up: `3715e4a0c863092ebeaabe5575ffde268b522fa92e3fb24cc8d34cdafcb63597`
- lane up: `3e5cd64eeb22f9f6d160256c1b770111adec7525ae0a429dd52d664d9c9e6cd9`
- lexical up: `3e060ad3341ed3fe5ccd51a060a7aac262a2a92859b6c7ee7f8e0873544d902a`
- combined validation: `21cfae893497d1a4075fb068bb79ebaa577ad9e3673f0a43df9b1428b88ac544`

### Next topology tests

The current `private_full` router is only private over each node's legal
degree-three outgoing edges; it does not expose all destinations. The next
architecture probe will test the inverse topology curriculum proposed during
this study: broad candidate connectivity with sparse active top-k compute,
an exploration grace period, accumulated edge-use and loss-salience evidence,
gradual pruning with a protected backbone/minimum degree, and final route
consolidation.

Before implementing dynamic pruning, a matched continuation will fork from a
well-calibrated broad-coverage checkpoint. One arm will retain trainable route
transport; the other will freeze only transport parameters, leave the terminal
jury trainable, and hold route temperature fixed. This isolates whether route
drift prevents private nodes from consolidating around useful paths. A dense
candidate test must report router parameter and edge-scoring compute separately
from active-node compute; degree `N` with the current private edge tensor is a
small-scale falsification probe, not yet the intended scalable design.

The first matched freeze fork is complete. Both branches restored the exact
step-15k model, optimizer state, and data cursor; both held route temperature
at `0.988658900` and continued the combined slow-node/slow-lexical policy to
step 20k. With transport trainable, free PPL reached `3,829.64`, accuracy 7.8%,
coverage 66.7%, and query RMS `1.791`. Freezing 44,472 router/ingress/address
transport parameters while leaving the 2,208 terminal-jury parameters
trainable worsened PPL to `4,315.55`, with 7.7% accuracy, 67.2% coverage, and
query RMS `1.801`. The archive SHA-256 values are respectively
`2665b730a87ec18836d08ec6fb327bf675707a2b593914c6ca68586490b8c17c` and
`2a42dc83519d2dcf3dec28f3707ea99877e94b206e20ad0066c6bbdcda882835`.

**Decision:** reject permanent transport freezing at the already-regressing
15k state. Router adaptation is partly compensating for the moving
node/shared substrate at this point; transport drift alone is not the collapse
cause. This result does not reject the three-stage treatment below, because it
omitted the essential route-only organization phase against frozen nodes. It
also leaves terminal-jury drift open: query RMS continued growing when
transport was frozen.

A second scheduled treatment will use three-stage block-coordinate training:

1. Train node/expert and route blocks jointly until smoothed held-out free CE
   plateaus.
2. Freeze the node/expert block and train route transport against stationary
   node functions until a second free-CE plateau.
3. Freeze route transport, unfreeze the node/expert block, and train the nodes
   around the consolidated paths for the remaining budget.

Frozen blocks must retain their dormant AdamW moments so unfreezing does not
silently become an optimizer-reset experiment. Checkpoint ranking uses content
CE, while plateau decisions are reported in the equivalent content-PPL curve
at only the requested 5,000-step evaluations. After the alternating arm chooses
its two transition steps, continuous-joint and simpler freeze controls will
replay those exact boundaries and data positions. The
transport router and terminal jury will be separable controls: the primary
test freezes transport only, because freezing the verdict mechanism would
confound route consolidation with output calibration.

Phase B has produced the first positive result for this schedule. The exact
combined step-10k checkpoint (`247.28` free PPL) was continued for 5,000
updates with lexical, lane/core, and private-node parameters frozen; all four
optimizer groups and all 22 AdamW states remained checkpointed, while only the
46,680 route/jury parameters received gradients. At step 15k, free PPL was
`258.28`, accuracy 6.6%, top-5 23.8%, coverage 40.4%, teacher PPL `2,771.82`,
and query RMS `1.432`. Throughput rose from about 10.4 to 15.6 steps/second.

The matched joint continuation reached PPL `772.12` and 62.7% coverage at the
same step. Route-only training therefore reduces the 10k-to-15k CE regression
from `1.138` nats to only `0.0435` nats and prevents the premature coverage
expansion. By the requested 5k evaluation cadence, this is a second plateau and
triggers Phase C. The Phase-B archive SHA-256 is
`2f2b7be075ca00a73a2691e942fd51cadc784c3ff5e59716d6beb89f1556d936`.

Phase C will fork twice from this exact state: one arm freezes transport only
while leaving the jury trainable; the other freezes the complete route/jury
block. Both restore lexical, lane/core, and private-node learning and hold
temperature at the Phase-B endpoint. This extra arm resolves whether the jury
is part of the stationary computational circuit or a safe adaptive output
calibrator.

The initial two Phase-C arms reactivated the entire non-route block. With
transport frozen but jury trainable, step-20k free PPL was `797.31`, accuracy
7.6%, top-5 23.5%, and coverage 60.1%. Freezing the complete route/jury block
instead produced PPL `805.87`, accuracy 7.6%, top-5 23.8%, and coverage 58.7%.
The 1.1% PPL difference is negligible relative to the control gap, so jury
adaptation is not the main stability mechanism. Their archive SHA-256 values
are respectively
`51e29ce358e127d0d34ee2a65832671842b25d3c23c3dac4c162cba4b102c369` and
`621a5ae3d0cfa7eefa25bf8b471a1a275a6ce9069ffb50016a3e16ef7f43aca9`.

Those arms were broader than the requested schedule: they reactivated the
lexical shell and lane/core substrate as well as private nodes. A literal
brains-only Phase C therefore froze lexical, lane/core, route transport, and
jury while updating only 460,852 private-node parameters at `0.625x`. It
reached step-20k free PPL **`271.33`**, 7.4% accuracy, 24.4% top-5, and 41.4%
coverage. This is 92.9% lower PPL than continuous joint training's `3,829.64`
at the identical step count, equivalent to a `2.647`-nat/token CE advantage.
It is also nearly stationary across the complete alternating path:
`247.28` at the 10k joint exit, `258.28` after route-only Phase B, and
`271.33` after brains-only Phase C. The archive SHA-256 is
`19fd135e9ffe2d22e728bb7662d86ff05ad60fa808d867f19657fb5ce5f37726`.

**Interim decision:** the destructive Phase-C movement is primarily in the
shared lexical/lane substrate, not private-node learning or route transport.
This is a phase-structure result: same source, data order, total steps, and
private-node LR produce radically different outcomes depending on which
coupled subsystems move. A final matched arm adds only the 2,208 terminal-jury
parameters to brains-only training to determine whether verdict adaptation
offers a useful small correction.

That final arm reached PPL `282.87`, accuracy 7.1%, top-5 24.4%, coverage
43.7%, and query RMS `2.084`. It is 4.3% worse in PPL than brains-only, with
lower top-1 accuracy and substantially larger jury magnitude. Its archive
SHA-256 is
`bdd15a784a21b71f3b4c104afbb6cde865874ce6269a831b936d66e55316da29`.

**Phase-C selection:** freeze route transport, terminal jury, lexical shell,
and lane/core; update only private-node brains. Jury adaptation mainly adds
coverage/confidence and does not improve the held-out token distribution in
this phase. Validate the selected brains-only arm over a longer horizon and
then replay the complete schedule from the best pre-plateau checkpoint rather
than accepting the detection window's already-regressed state.

The selected brains-only arm was then continued unchanged from 20k to 25k.
Free PPL remained effectively flat at `279.12` while accuracy improved to
7.8%, top-5 to 24.7%, and coverage to 43.6%; query RMS stayed exactly frozen
at `1.432`. The complete selected trajectory is therefore `247.28` at 10k,
`258.28` after route-only 15k, `271.33` after brains-only 20k, and `279.12`
after brains-only 25k. The 25k archive SHA-256 is
`5a4589285c90971621a55bc8213c3dd3eab88d1ec5b57a97144acfedec3e6776`.

This validates stability, not yet renewed PPL descent. The next schedule replay
will roll back to the best pre-plateau 5k checkpoint before entering route-only
and brains-only phases. A production plateau detector should preserve and
restore the best-free checkpoint at a phase transition; it should not force the
next block to inherit the already-regressed endpoint used to detect plateau.

That rewind replay has now completed its route-only leg. It restored the exact
slot-isolated step-5k checkpoint (`151.72` free PPL), all 22 AdamW states, and
the portable-exact data cursor, then updated only the 46,680 route/jury
parameters for 5,000 fully autonomous steps. At step 10k, free PPL improved to
**`148.07`** rather than regressing, a 2.41% gain (`0.0244` nats/token), while
content accuracy and coverage held at 5.2% and 19.8%. Teacher PPL was `345.40`
and terminal-query RMS remained only `0.575`. The locally recovered archive is
`experimental_checkpoints/gator1m_alternating/early_phase_b_route_only_step10000.tgz`
with SHA-256
`14137e35f58e87bd7ead0822c690fd8814f5fc6142c6aec897a0043a7f57e3a5`.

This is stronger than the late 10k-to-15k route plateau: against the genuinely
best checkpoint, a stationary brain lets route/jury learning lower free CE.

The paired private-node-only continuation then froze lexical, lane/core,
route transport, and jury while preserving all optimizer states. At step 15k,
free PPL was **`154.66`**, only 4.45% (`0.0435` nats/token) above the route
checkpoint; teacher PPL improved from `345.40` to `328.13`, content accuracy
rose from 5.2% to 5.3%, and coverage rose from 19.8% to 21.4%. Terminal-query
RMS remained bit-for-bit fixed at `0.575232`. This endpoint is 93.4% below the
unchanged isolated run's `2,354.49` PPL and 80.0% below the combined-policy
joint run's `772.12` PPL at the same 15k step. The locally recovered archive is
`experimental_checkpoints/gator1m_alternating/early_phase_c_brains_only_step15000.tgz`
with SHA-256
`ba89e48632b54c39d64f458e71d8eacdc7fea1c688170ace27ccd14979bbd651`.

The clean 50k replay will now start from a new seed-0 initialization:
teacher-forced co-adaptation to a 5k-granularity PPL plateau, rewind to its best
checkpoint, teacher-forced route/jury-only adaptation to a second plateau and
rewind, one 5k private-node-only teacher runway, then the 60%-mastery token
handoff with only private nodes trainable. Route temperature will follow the
broad staged schedule during discovery and be frozen at the selected route
checkpoint for node consolidation.

The clean replay's first Phase-A interval is complete. From a fresh seed-0
initialization, all four component groups co-adapted under full teacher forcing
with slot isolation, lexical `0.5x`, private nodes `0.625x`, lane/core and
route/jury `1.0x`, and route temperature held broad at `1.25`. At step 5k,
teacher content PPL was **`47.59`**, content accuracy 18.3%, top-5 47.0%, and
formation coverage 90.6%; terminal-query RMS was `0.522`. This establishes the
first protected Phase-A rewind checkpoint rather than triggering a transition.
The resumable archive is
`experimental_checkpoints/gator1m_alternating/block_coordinate_segment01_retained5000_attempted5000.tgz`
with SHA-256
`fd41400c481f9d4d14df7ba6719ccb40fecfd99a74b44a1a7567484d8729d911`.

The second co-adaptation interval exposed the expected confidence collapse.
At the speculative step-10k endpoint, teacher content PPL worsened from
`47.5860` to `59.6473` (+25.35%, +`0.2259` nats/token), even though content
accuracy rose from 18.26% to 19.48% and coverage from 90.64% to 91.39%.
Terminal-query RMS grew from `0.522` to `0.834`, while the effective terminal
vote count narrowed from 4.93 to 2.72. The plateau controller therefore saved
the probe for diagnosis, restored the exact step-5k model/optimizer/data state,
and entered route/jury-only Phase B with retained step 5k and attempted compute
10k. Its recovery archive is
`experimental_checkpoints/gator1m_alternating/block_coordinate_segment02_retained5000_attempted10000.tgz`
with SHA-256
`ed155e44df0edb71eeeaee8fb862aefd19b3eadfa422ea78b474999608d4ca41`.
The discarded model itself was retained separately for parameter-drift and
gradient-conflict comparison as
`experimental_checkpoints/gator1m_alternating/block_coordinate_phase_a_plateau_probe_step10000.pt`
(SHA-256
`ecabc295e6b6c0c5e520323b80f461777df0e8bb33eb4ee50aad1175b831fd54`).

Phase B then replayed the same retained 5k-to-10k data interval with only the
46,680 route/jury parameters trainable. Teacher content PPL reached `48.1477`,
1.18% worse than the `47.5860` entry checkpoint, with accuracy 17.61%,
coverage 90.03%, and query RMS `0.848`. Under the predeclared 1% criterion this
is a route-phase plateau, so the controller again restored the step-5k best.
The selected route temperature is consequently the broad `1.25` value at that
checkpoint. This differs from the autonomous replay, where route-only learning
improved free PPL 2.41%; route utility is therefore regime-conditioned, not a
universally improving subproblem. Retained step remains 5k while cumulative
attempted compute is 15k.

The resumable archive is
`experimental_checkpoints/gator1m_alternating/block_coordinate_segment03_retained5000_attempted15000.tgz`
(SHA-256
`bf21a602678fd5c3ecdd5e187d41773464ae663e775cf02a2d7f2820c06ed9da`).
The discarded route checkpoint is
`experimental_checkpoints/gator1m_alternating/block_coordinate_phase_b_plateau_probe_step10000.pt`
(SHA-256
`107b4491cc2cc7399f1476ba301b6f069f2f5f9d82e58e214dbdd9fc979fc8d1`).

Phase C's one-interval teacher runway provides a three-way matched causal
comparison from the same step-5k model, Adam states, and data cursor. At the
step-10k endpoint:

| Trainable block from 5k to 10k | Teacher content PPL | vs 5k entry | Content accuracy | Coverage | Query RMS |
|---|---:|---:|---:|---:|---:|
| All blocks jointly | 59.6473 | +25.35% | 19.48% | 91.39% | 0.834 |
| Route/jury only | 48.1477 | +1.18% | 17.61% | 90.03% | 0.848 |
| **Private nodes only** | **47.5022** | **-0.18%** | 18.15% | 89.13% | **0.522 (frozen)** |

Private-node-only learning is the sole branch that improved teacher content
PPL, albeit slightly. It is 20.36% below joint co-adaptation and 1.34% below
route/jury-only at the identical retained step. This falsifies the simple claim
that node updates themselves cause the regression. The damaging term is
coupled movement—especially shared lexical/lane and jury confidence—around the
private computation. The experiment now continues from this retained node-only
checkpoint with all non-node parameters and route temperature (`1.25`) frozen.

The segment archive is
`experimental_checkpoints/gator1m_alternating/block_coordinate_segment04_retained10000_attempted20000.tgz`
with SHA-256
`bf787f82a83bacf3a540347c6eb8149e851238550f10fb37ca8e208ead6af879`.

The first node-only handoff interval reached retained step 15k without a
transition collapse. With one unsupported frontier token, partial content PPL
was `52.7771`; teacher PPL was `50.0612`; and a separate all-31-slots-free
evaluation measured PPL **`286.3186`**, 3.77% top-1, 19.51% top-5, and 12.23%
formation coverage. The frontier accuracy was 40.3% versus 52.7% at the same
address under teacher forcing, so both were below the 60% floors and the
controller correctly held at one released token. There were no advances or
retreats, and jury RMS remained frozen at `0.522424`.

Although not yet a low free PPL, `286.32` is already 87.8% below the unchanged
slot-isolated run's `2,354.49` and 62.9% below the joint slow-rate run's
`772.12` at matched step 15k. Unlike those controls, it retains a teacher PPL
near 50 rather than thousands. This is the intended stable frontier-learning
regime; more runway is required before release, not a lower mastery gate.
The archive is
`experimental_checkpoints/gator1m_alternating/block_coordinate_segment05_retained15000_attempted25000.tgz`
with SHA-256
`bc243ebad75295e6eae9b40675e9dab8f259740318e04d9f0456eb282545d490`.

The same one-frontier node-only regime continued to retained step 20k. Fully
free PPL improved from `286.32` to **`254.79`** (-11.0%), free top-5 rose from
19.51% to 20.4%, and free formation coverage rose from 12.23% to 16.6%.
Teacher PPL remained bounded at `50.86`; partial PPL was `53.29`; jury RMS was
still exactly `0.522424`. Frontier and teacher-frontier accuracies remained
40.3% and 52.7%, so the ratio EMA held at `0.764` and the 60% gate made no
move. This is useful progress rather than a calendar stall: global all-free CE
is descending while the model is honestly denied credit for mastering the
specific next frontier.

The step-20k archive is
`experimental_checkpoints/gator1m_alternating/block_coordinate_segment06_retained20000_attempted30000.tgz`
with SHA-256
`cf60deceba7271f2a0ac308c01f3742994e52e32b3e0e083950cff17d9daf3b5`.
