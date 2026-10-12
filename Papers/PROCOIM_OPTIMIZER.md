# Procoim Optimizer MVP

Procoim (Progressive Continuous Improvement) is an experimental, forward-only
optimizer for Gator. It treats one generated-prefix example or microbatch as a
short optimization episode, improves its output slots with structured weight
perturbations, and protects progress already earned by solved slots.

The MVP asks whether GTR-L can learn above chance without reverse-mode
backpropagation. It is not expected to beat AdamW on throughput.

## Forward-only search

For each episode, Procoim:

1. Runs a deterministic baseline forward pass and records cross-entropy
   separately for every valid output slot.
2. Sets each slot's target to
   `max(loss_floor, (1 - reduction_fraction) * start_loss)`.
3. Selects one semantic parameter group and samples normalized Rademacher
   perturbations.
4. Evaluates antithetic `+epsilon` and `-epsilon` candidates sequentially with
   additional forward passes.
5. Rejects candidates that violate protected-slot tolerances and immediately
   accepts the first perturbation that reduces the current mean active-slot
   loss by at least `reduction_fraction` while also clearing the normalized
   active-gain and utility thresholds. `candidate_directions` is a maximum
   search allowance, not a bank that must be exhausted; even the opposite
   antithetic sign is skipped when the first sign qualifies.
6. Restores the probed block exactly, applies only the winning delta, and keeps
   that accepted weight update for future examples.

`max_candidate_banks` is a consecutive-stall budget: accepting a perturbation
resets it, so the independent `max_accepted_updates` limit remains reachable.
The semantic parameter-group cursor persists across episodes and checkpoints,
preventing short episodes from probing only the first group repeatedly.

The default requested reduction is deliberately small:

```text
reduction_fraction = 0.005  # 0.5% per slot, per episode
```

The same fraction is a hard acceptance floor for each persistent perturbation:
with the default, a candidate must reduce the current mean ACTIVE-slot loss by
at least 0.5%. Failed probes still count as steps, but accepting a smaller gain
cannot dilute the configured continuous-improvement goal.

This is exposed as `--procoim-reduction-fraction` so larger and smaller targets
can be swept without changing code.

## Optional search and retention controls

The optimizer exposes independent search controls and bounded cross-example
retention. Every control is disabled by default, preserving the original
random antithetic search trajectory and forward accounting.

- `--procoim-reuse-successful-directions` reserves one direction in a bank for
  the signed, normalized EMA of that parameter group's accepted directions.
  The cached direction is rescaled to the group's current probe radius.
- `--procoim-antithetic-estimate` combines the measured `+u/-u` utility
  differences into a finite-difference direction. The result is never trusted
  analytically: it must win an additional real forward evaluation, which is
  charged to `--steps`. Paired probes are retained and combined only when no
  ordinary perturbation in the bank clears the acceptance threshold.
- `--procoim-adaptive-probe-scale` keeps a separate radius for each parameter
  group. Accepted banks grow that group's radius up to the configured maximum;
  rejected banks shrink it toward the configured minimum.
- `--procoim-adaptive-group-selection` warm-starts every configured group,
  then allocates banks by deterministic gain-per-forward EMA plus an
  exploration bonus.
- `--procoim-directional-adam` measures a seeded basis with antithetic
  forwards, runs Adam in that low-dimensional coefficient space, and accepts
  the resulting direction only after a real forward check.
- The default directional basis is `rotating_factorized`: K defaults to 32,
  matrix-like tensors receive rank-1 factors, and each group rotates to a
  fresh deterministic basis after eight of its own candidate banks.
  Coefficient moments reset when their coordinate basis changes.
- `fixed_rademacher` remains available for reproducing the original fixed
  dense-basis experiments.
- `--procoim-anchor-buffer-size N` retains up to `N` slotwise replay examples.
- `--procoim-anchor-sentinel-count N` checks a rotating subset only after a
  candidate passes the current-example gate. Set it to zero to keep replay out
  of the inner search loop.
- `--procoim-consolidation-interval-examples N` accumulates accepted changes
  provisionally for `N` examples, then audits the window before promoting it
  to the trusted base weights.

### Rotating factorized search

A fixed directional basis restricts every accepted update to the same K
directions. Increasing K widens that subspace but does not remove the ceiling.
The rotating mode instead keeps a basis stable for a short coefficient-Adam
window, then deterministically regenerates a new one:

```text
K antithetic probes in basis epoch e
        ↓
project and forward-validate the coefficient update
        ↓
retain accepted movement in the model weights
        ↓
after N group-local banks, advance to epoch e + 1
```

For a matrix reshaped to `[rows, columns]`, each proposal is generated from
low-rank factors. Rank 1 stores conceptually `rows + columns` random signs
instead of a persistent dense matrix. Vectors use seeded Rademacher directions
because they have no useful matrix factorization.

Directions are regenerated one at a time from `(seed, group, epoch, index)`.
Procoim does not retain K full parameter-sized directions. Combining the
coefficient update also regenerates each direction into one dense accumulator.
Consequently:

```text
persistent directional state: O(groups * K)
peak proposal storage:         O(P)
retained K-direction bank:     none
```

The default controls are:

```text
--procoim-candidate-directions 32
--procoim-directional-basis-mode rotating_factorized
--procoim-directional-factor-rank 1
--procoim-directional-basis-rotation-banks 8
```

Training logs expose `group_basis_epochs=[...]`. Use
`--procoim-directional-basis-mode fixed_rademacher` for a direct fixed-basis
control with otherwise identical settings.


### Relaxed consolidation floor

Search and consolidation have separate progress thresholds. For example:

```text
search reduction target = 0.00010  # 0.01%
audit retained floor    = 0.00005  # 0.005%
```

Configure the latter with:

```bash
--procoim-anchor-minimum-retained-reduction-fraction 0.00005
```

The search still accepts a persistent perturbation only after it clears the
0.01% current-example target. At consolidation, the floor is measured across
all protected slots in the audited anchor bank:

```text
sum(start_loss - audited_loss) / sum(abs(start_loss))
```

This bank-level contract permits small cross-example redistribution instead
of requiring one scalar provisional update to improve every historical slot
simultaneously. It does not remove individual safety: a hard anchor slot still
vetoes an audit if its loss exceeds:

```text
start_loss + anchor_hard_regression_fraction * abs(start_loss)
           + anchor_absolute_tolerance
```

The audit tries the configured descending consolidation alphas and commits the
largest safe fraction. `consolidation_retained_reduction` is logged so a commit
or rollback can be traced directly to the retained-progress contract.

Procoim executes under `torch.inference_mode()` and never calls `backward()`.
Candidate comparisons use a deterministic search mode, dedicated perturbation
RNG state, fixed inputs and masks, and restored model buffers.

## What counts as a step

**Every model forward inside Procoim optimization counts as one training
step.** This includes:

- the episode baseline;
- every positive and negative candidate probe; and
- every current-anchor baseline and candidate-preservation probe;
- deterministic protection-confirmation forwards.
- every anchor forward in a consolidation audit.

`--steps` is an exact forward-call budget. An episode may stop before using its
normal bank or update allowance when the remaining budget is exhausted.
Episode count and accepted-update count are secondary metrics, not training
steps. Periodic fresh-validation forwards are reported separately and are not
charged to the Procoim search budget.

This accounting is essential when comparing Procoim with AdamW: one Procoim
episode can consume many steps even if it accepts only one update.

## Slot state machine

Slot protection is local to the current episode. Solved slots still participate
in all shared Gator communication; neither their hidden state nor shared model
parameters are frozen.

```text
ACTIVE -> SOFT_FROZEN -> PROTECTED
   ^            |
   +------------+
     regression above target
```

- `ACTIVE` slots drive the normalized-gap search objective.
- `SOFT_FROZEN` slots have reached their episode target. They stop driving
  exploration, contribute a soft regression penalty, and accumulate a
  stability streak across accepted updates.
- A soft-frozen slot that regresses above its target is reactivated as
  `ACTIVE`; its stability count is reset.
- `PROTECTED` slots are anchored to their best achieved loss. A candidate that
  exceeds the configured hard tolerance is vetoed. A later improvement moves
  the best-loss anchor forward.

If no active slot remains while slots are still soft-frozen, positive active
gain is undefined and no further perturbation can legally be accepted.
Procoim therefore uses deterministic no-op confirmation forwards to finish the
required stability streak. These confirmations change no parameters, still
validate the achieved losses, and each counts as a training step.

## GTR-L parameter groups

`GatorLinearPrefixModel.procoim_parameter_groups()` exposes disjoint semantic
blocks:

```text
routing
stream_mix
node_compute
readout
embedding
```

The default search schedule is:

```text
routing stream_mix node_compute readout
```

Embeddings are excluded for the initial prefix proof-of-concept. Include them
explicitly when needed:

```bash
--procoim-parameter-groups routing stream_mix node_compute readout embedding
```

With tied token embeddings and vocabulary weights, the shared parameter is
assigned to `embedding` exactly once. Omitting `embedding` therefore also
freezes that tied vocabulary matrix even though `readout` is enabled. This is
intentional for the MVP but should be considered when comparing tied and
untied models.

Models without `procoim_parameter_groups()` fall back to one
`all_parameters` block. In that fallback, the semantic group schedule has no
effect.

## Running the prefix MVP

GTR-L prefix trainer:

```bash
python gator_linear_prefix_train.py \
  --optimizer procoim \
  --steps 2000 \
  --batch-size 1 \
  --length 4 \
  --procoim-reduction-fraction 0.005
```

Original parallel-prefix benchmark:

```bash
python parallel_prefix_benchmark.py \
  --variant gtraw \
  --optimizer procoim \
  --steps 2000 \
  --batch-size 1 \
  --train-length 4 \
  --eval-lengths 4 \
  --procoim-reduction-fraction 0.005
```

Useful initial sweep controls include:

```text
--procoim-candidate-directions
--procoim-max-candidate-banks
--procoim-max-accepted-updates
--procoim-stability-required
--procoim-probe-scale
--procoim-minimum-active-gain
--procoim-soft-regression-fraction
--procoim-hard-regression-fraction
--procoim-completion-mode
--procoim-seed
```

The optimized-search ablations additionally expose:

```text
--procoim-reuse-successful-directions
--procoim-direction-momentum
--procoim-antithetic-estimate
--procoim-adaptive-probe-scale
--procoim-probe-scale-growth
--procoim-maximum-probe-scale
--procoim-adaptive-group-selection
--procoim-group-score-beta
--procoim-group-exploration
--procoim-anchor-buffer-size
--procoim-directional-adam
--procoim-directional-adam-learning-rate
--procoim-directional-basis-mode
--procoim-directional-factor-rank
--procoim-directional-basis-rotation-banks
--procoim-anchor-sentinel-count
--procoim-anchor-soft-regression-fraction
--procoim-anchor-hard-regression-fraction
--procoim-anchor-minimum-retained-reduction-fraction
--procoim-consolidation-interval-examples
--procoim-full-audit-anchor-count
--procoim-consolidation-alphas
--procoim-provisional-shrink-factor
--procoim-anchor-regression-fraction
--procoim-anchor-absolute-tolerance
--procoim-anchor-weight
--procoim-anchor-hard-protection
```

Use `--help` on either entrypoint for the complete set and current defaults.

## Calibrated proof-of-concept

The deterministic integration test uses the `PrefixTransformer980` control
for 1,000 length-one generated-prefix episodes with batch size 64, up to four
antithetic directions, one bank, and probe scale `0.02`. First-hit acceptance
uses 8,769 counted forwards instead of exhausting the 9,000-forward maximum.
Across the exhaustive 144-case validation set, CE falls from `4.2629` to
`2.8224` and accuracy moves from `12/144` (chance) to `13/144` (above chance).
Accepted changes persist across examples and at least some episode slots reach
protected state. A separate test runs a candidate bank through the smallest
GTR-L configuration, but does not yet claim robust multi-episode GTR-L
learning.

The default GTR-L probe scale is `0.03`. In the 40-node, seven-slot smoke,
`0.003` and `0.01` produced no perturbation large enough to clear the hard 0.5%
acceptance floor, while `0.03` did. A fixed scale remains fixed across stalls;
only the explicitly enabled adaptive-scale control changes group radii.

This is intentionally a weak success criterion for the prototype. It proves
that the forward-only path can produce persistent, generalizing improvement;
it does not show robust prefix mastery. Longer calibration runs sometimes
continued lowering CE while argmax accuracy regressed, so both metrics must
remain visible in future sweeps.

## Precision and execution mode

The MVP uses FP32 model parameters so small perturbations are not rounded away.
The GTR-L prefix entrypoint rejects AMP, compilation, non-constant learning-rate
schedulers, and route-auxiliary losses when Procoim is selected. No standard
PyTorch optimizer, gradient scaler, gradient clipping, or learning-rate
scheduler is used. Parameter `.grad` fields must be empty before and after an
episode.

## Logging and checkpoints

The GTR-L trainer reports forward-call step and throughput, episode forwards,
candidate and anchor forwards, accepted updates, active/soft/protected counts,
target hit rate, acceptance and hard-veto rates, stall count, accepted reused
and estimated directions, per-group probe scales, gain-per-forward estimates,
anchor-specific vetoes, acceptance counts, consolidation alpha, audit penalty,
and retained reduction. Source accepts and anchor vetoes are logged both per
episode and cumulatively. Fresh evaluation includes total accuracy, exact
formation accuracy, loss, and the full left-to-right `by_pos` vector.
It also prints `by_pos_ce`, the left-to-right validation cross-entropy vector,
so probability improvement remains visible before an argmax changes.
`acceptance_rate` is accepted updates divided by candidate forwards; when
anchors are enabled, also report accepted updates per total search forwards so
retention probes are not hidden from efficiency comparisons.

The benchmark trainer reports the same forward-step accounting with a smaller
episode summary.

Checkpoint payloads include:

- persistent accepted model weights;
- total forward calls, episodes, candidate forwards, and accepted updates;
- parameter-group acceptance counts and group-layout signature;
- per-group probe scales, gain statistics, bank counts, and signed direction
  momentum;
- dedicated perturbation-generator state for reproducible continuation; and
- the next semantic parameter-group position.

The GTR-L checkpoint payload also serializes its bounded trainer-side anchor
buffer on CPU.

`ProcoimTrainingEngine.load_state_dict()` restores this optimizer state and
checks that its saved step equals the cumulative forward count. The current
training entrypoints save Procoim state but do not yet expose a command-line
resume flag.

## MVP limitations

- Only generated-prefix slotwise cross-entropy is wired in. Language training
  needs a meaningful `[batch, output_slot]` loss decomposition; a single global
  alignment loss is not sufficient.
- Candidate forwards are evaluated sequentially. There is no
  `torch.func`/`vmap` candidate batching yet.
- Slot protection is episode-local. Cross-example anchors provide bounded
  behavioral replay and window consolidation, not permanent parameter
  reservation for any slot or example.
- The search is FP32-only and currently does not support `torch.compile` or
  mixed precision.
- A slot-loss provider is a pure evaluator by contract. Parameter mutation is
  detected and aborts the episode, but mutation of an unselected parameter by
  an invalid provider is not recoverable without a full-model snapshot.
- An episode may exhaust its forward, bank, or accepted-update budget without
  protecting every slot.
- The method trades activation memory for repeated forward compute. Forward-only
  does not imply wall-clock speed.
- Checkpoint resume is available through the engine API, but not yet through a
  launcher flag.

The core contract is: improve unresolved slots through forward
counterfactuals, preserve solved-slot gains within tolerance, and persist only
accepted shared-weight changes.
