# Gator-Path implementation

This package is the executable Gator-Path (`GTR-P`) architecture track. It is
being built as small rollback-safe milestones rather than folded into the
validated Arrow implementation.

The current local-frontier surface contains:

- a fixed local ring topology with the five legal cell actions
  `stay, -1, +1, -2, +2`;
- the validated four-node Arrow graph and source-local action tables;
- factorized and joint local-action codecs; and
- a dense proof state containing only the current packet mass and carried
  payload at `[B,L,C,N]` locations;
- dense correctness mailboxes for the ten-cell proof, alongside the earlier
  sparse occupied-location mailbox primitives for the eventual packed engine;
- genuine soft forward participation by plausible local stewards early in
  training, annealing toward deterministic local handoffs;
- active-only edge-and-destination transforms selected from private
  `(cell, node)` parameter banks;
- one synchronous relay with old-snapshot reads and functional commits; and
- a shared-recurrent core whose relay depth does not duplicate parameters;
- learned dense-to-deterministic ingress for the first steward cell/node;
- a ten-cell simultaneous modular-prefix model with one persistent current
  packet frontier per token;
- private tied token input/readout tables for each cell, matching the
  specialization capacity of the original ten 980-parameter Arrow cells;
- receiver-specific relative-position mailbox reads inside each local
  cell/node, restoring Arrow's relation-aware `R` mechanism; and
- a resumable trainer with deterministic fresh evaluation, best/latest
  checkpoints, route diagnostics, and Colab-friendly progress logs.

The restored ten-cell frontier candidate has 9,908 trainable parameters,
within 0.92% of the validated 10,000-parameter Arrow. The earlier shared-I/O,
shared-memory frontier has 8,596 parameters. Relay weights are recurrently
shared, so raising relay depth increases computation without duplicating the
cell policies.

The packet address is immutable across relay updates. The proof engine carries
only the current frontier into the next relay and discards the previous
handoff. Every source cell transforms the packet mass it actually stewards,
then makes its own bounded next-cell/node decision. The transformed payload,
address, and new current location distribution carry forward directly; only
mailbox summaries are refreshed. No complete route is represented or retained.

An earlier hard-from-step-zero mode remains available for rollback tests. It
used a dense backward surrogate but committed only one forward steward. Seed 0
remained at 8.6% after 5,000 updates, versus 68.3% for the validated Arrow
reference. That falsified the surrogate-only interpretation of "start dense":
alternate cells must receive real packets and affect downstream computation
before useful local policies can emerge.

The earlier shared-I/O, non-relative current-frontier seeds 0, 1, and 2
reached 18.7%, 29.4%, and
13.5% at 5,000 steps (20.5% mean). All three left the random-loss floor and
naturally contracted from 40 effective cell/node locations to about 8. Learned
first-steward confidence reached 73--92% while different packet types still
used four to six effective ingress cells. One seed widened ingress when
progress slowed and then resumed learning. This validates the intended routing
behavior and packet-dependent ingress specialization, but it does not yet
recover Arrow's 50.7% three-seed mean learning speed.

Forcing exact top-1 downstream handoffs preserves two seeds almost perfectly.
Forcing exact top-1 ingress does not: it drops all three and collapses two near
chance. High soft first-steward confidence is therefore not yet operationally
deterministic. Use `--hard-ingress` and `--hard-handoffs` independently when
testing hardening curricula.

An abrupt seed-1 hard-ingress continuation recovered loss from 5.1811 to
2.5068 over 1,000 updates but remained at 8.9% chance accuracy. Use gradual
ingress-only annealing or soft/hard transfer next; sudden top-1 commitment is a
documented failed control.

Run a standalone finite forward/backward check from the repository root:

```bash
python gator_path_smoke.py
```

Run the ten-cell local-frontier learning proof:

```bash
python -u gator_path_prefix_train.py \
  --steps 10000 \
  --batch-size 128 \
  --length 7 \
  --cells 10 \
  --relays 4 \
  --top-k 10 \
  --log-every 500
```

The trainer defaults to the restored near-parity candidate:

```text
private_cell_io=True
relation_aware=True
parameters=9,908
```

The independently switchable capability A/Bs are:

| Private cell I/O | Relative reads | Parameters | Purpose |
|---|---|---:|---|
| No | No | 8,596 | Earlier Path frontier control |
| Yes | No | 9,748 | Restore cell capacity only |
| No | Yes | 8,756 | Restore receiver-specific reads only |
| Yes | Yes | 9,908 | Near-parity Arrow-capacity Path candidate |

Use `--no-private-cell-io` and `--no-relation-aware` to reproduce the earlier
control. These features change what each current steward can compute, but not
the routing contract: each cell still transforms only its current packet mass,
makes one bounded local next-hop decision, and carries the resulting state
forward without storing a complete route.

For the tighter pre-Path control, the repository also exposes
`gator_cells_vectorized.py --carry-state`. It preserves the exact
10,000-parameter Arrow model and changes only dense destination-owned neighbor
reconstruction into source-owned state handoff. This separates the carry
hypothesis from every other Path architectural change.

The dense carry control restored much of Arrow's capability but not its speed:
seed 0 reached 52.7% at 5k, while matched 10-cell throughput was 2.88 steps/s
for Arrow and 2.89 for carry. At 62 cells both measured 0.57 steps/s. Both
still execute every cell and rebuild every node memory.

`persistent_memory.py` is the next executable milestone. It carries
receiver-specific relative-memory numerators and denominators, subtracts
departing packet contributions, and adds transformed arrivals. Incremental
updates are tested against exact dense rebuilds and preserve finite gradients.
This removes the need to revisit empty cell/node locations once it is combined
with packed ownership and active-only transforms.

The model receives `[start, d1, ..., d7]` and forms all seven modular prefix
sums simultaneously. It does not receive a correct previous output token.
The initial output packets all contain the same blank payload; their immutable
addresses distinguish their requested formation coordinates. Learned ingress
begins as a distribution over first stewards and sharpens with temperature.
Thereafter each cell independently learns what to do to the packet mass it
currently owns and which neighbor should receive it next. Complete routes are
never represented or optimized as objects.

## Historical regional developmental trainer

The following regional and split-optimizer commands are preserved for
reproduction, not recommended as the current core Path proof. Final
orchestrator ownership proved separable from prefix capability: the graduated
router reached 99.9% ownership without improving task accuracy or loss. The
local-policy trainer above supersedes this branch for architecture validation.

The ten-cell 4/4/2, 2/6/2, and 1/7/2 experiments are available through:

```powershell
python -u gator_path_regional_train.py `
  --device cpu `
  --number-cells 1 `
  --orchestrator-cells 2 `
  --number-steps 400 `
  --arithmetic-steps 1200 `
  --integration-steps 1500 `
  --orchestrator-participates `
  --interleaved-skills `
  --call-return `
  --relays 10 `
  --arithmetic-learning-rate 0.003 `
  --integration-learning-rate 0.003 `
  --checkpoint-dir experimental_checkpoints/path_regional_1_7_2
```

Skill rehearsal activates only the orchestrator and the region relevant to
the current batch. The best regional checkpoint is saved separately as
`gator_path_regional_skills.pt` before integration begins. Use
`--resume-skill-checkpoint` to iterate on integration without repeating skill
development.

`--call-return` replaces a static regional mask with directional relay phases.
Packets are dispatched by the orchestrator, computed by one specialist,
returned toward cells 8–9, and consolidated before another specialist can be
called. Ten relays are the minimum for the current number-then-arithmetic
composition schedule.

Use `--learned-dispatch` to remove the scripted region order:

```powershell
python -u gator_path_regional_train.py `
  --number-cells 1 --orchestrator-cells 2 --relays 6 `
  --orchestrator-participates --interleaved-skills `
  --learned-dispatch --return-loss-weight 0.1 --return-relays 1 `
  --rehearse-loaded-skills --router-only-rehearsal `
  --resume-skill-checkpoint experimental_checkpoints/skills/gator_path_regional_skills.pt
```

The ownership mask permits the orchestrator to call either region while
specialists may only work locally or return. `--router-only-rehearsal` freezes
specialist transformations while adapting router heads and orchestrator
transforms. Route logs include correct-region call shares and final
orchestrator ownership.

Add `--gate-dispatch` to make those region calls size-independent. The
orchestrator then has exactly one action for the number gate and one for the
arithmetic gate, regardless of how many cells live behind either entrance.
Specialists retain local work actions and every specialist receives a direct
return action:

```powershell
python -u gator_path_regional_train.py `
  --number-cells 1 --orchestrator-cells 2 --relays 6 `
  --orchestrator-participates --interleaved-skills `
  --learned-dispatch --gate-dispatch `
  --return-loss-weight 0.1 --return-relays 1 `
  --rehearse-loaded-skills `
  --resume-skill-checkpoint experimental_checkpoints/skills/gator_path_regional_skills.pt
```

Gate dispatch is result-centric: the macro-router chooses a skill region, then
the region's private routing determines internal stewardship. The saved
checkpoint records the exact destination-action table used by the run.

For a dense-to-sparse curriculum, complete a broad-support warmup, resume its
best regional checkpoint with a fresh optimizer, and add
`--maximum-support 2`. This prevents a longer total step budget from
accidentally stretching the exploratory top-6 phase:

```powershell
python -u gator_path_regional_train.py `
  --learned-dispatch --gate-dispatch --maximum-support 2 `
  --rehearse-loaded-skills `
  --resume-skill-checkpoint experimental_checkpoints/gate_warmup/gator_path_regional_skills.pt
```

Long tiny-model runs can accumulate unstable Adam moments even when gradients
are clipped. `--nonfinite-restarts N` permits at most `N` automatic
recoveries: restore the best fresh-data checkpoint, clear optimizer state,
and continue. Every recovery is logged. The default remains zero so an
experiment never hides instability unless this recovery policy is requested.

Use `--optimizer-mode split` to optimize route heads with zero-momentum SGD
while retaining AdamW for token representations and transforms.
`--router-learning-rate` controls the SGD rate independently:

```powershell
python -u gator_path_regional_train.py `
  --optimizer-mode split --router-learning-rate 0.01 `
  --integration-learning-rate 0.001
```

This split exists because route changes reassign parameter responsibility
discretely: averaging old and new route gradients can describe neither valid
path. Transformation gradients remain gradual enough for AdamW.

Fixed route rates exposed a second problem: `0.01` stabilized final
orchestrator ownership near 50%. One `0.1` run reached roughly 75%, but that
curve has not reproduced and is recorded as an outlier rather than a result.
The optional ownership schedule tests whether beginning with a stronger
exploratory rate and lowering it as ownership forms is more stable:

```powershell
python -u gator_path_regional_train.py `
  --optimizer-mode split `
  --router-learning-rate 0.1 `
  --ownership-router-schedule `
  --router-minimum-learning-rate 0.01 `
  --router-ownership-start 0.5 `
  --router-ownership-target 0.8 `
  --integration-learning-rate 0.001
```

The trainer schedules against the highest ownership observed so far, not the
latest noisy measurement. Consequently, routing LR can only decrease during a
stage. With the defaults it remains `0.1` through 50% ownership, is about
`0.046`, `0.022`, and `0.015` at 60%, 70%, and 75%, respectively, and reaches
`0.01` at 80%. The transformation AdamW rate is unchanged.

If this schedule fails, the next clean A/B is scratch regional training with
split optimization enabled from the first update. If that also fails, use the
existing broad-support bootstrap and contract toward top-k/hard stewardship as
route confidence develops. Sparse deployment remains the target even if early
training must begin dense.

The first local study retained 100% number conversion and 41.8% modular
arithmetic accuracy simultaneously, but did not yet compose those skills into
the length-seven prefix task. See
`Papers/GATOR_PATH_REGIONAL_EXPERIMENT.md` for the complete result.

The complete architecture and milestone gates are defined in
[`Papers/GATOR_PATH_DESIGN_SPEC.md`](../Papers/GATOR_PATH_DESIGN_SPEC.md).
