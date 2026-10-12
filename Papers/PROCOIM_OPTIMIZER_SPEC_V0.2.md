# Progressive Continuous Improvement (Procoim) Optimizer

**Status:** Detailed prototype specification for Codex implementation
**Target codebase:** Gator / Graph Trace Relay (GTR)
**Version:** 0.2 design draft
**Training method:** Fully forward-only, perturbation-based optimization
**Initial benchmark:** Generated simultaneous parallel-prefix task
**Primary goal:** Determine whether Gator can learn through repeated forward counterfactuals, progressive per-slot protection, and persistent accepted parameter updates without reverse-mode backpropagation.

---

## 0. Executive summary

Procoim turns each training example into a short optimization episode. Gator performs a baseline formation, measures loss independently for every output slot, and repeatedly tries structured parameter perturbations. Candidate perturbations are accepted only when they improve the unresolved part of the formation while preserving gains already achieved by solved slots.

```text
one example or microbatch
        ↓
baseline forward formation
        ↓
record loss for every output slot
        ↓
propose structured parameter perturbations
        ↓
evaluate candidates using forward passes only
        ↓
accept the best candidate that improves unresolved slots
and respects protected-slot constraints
        ↓
promote sufficiently improved slots to protected status
        ↓
continue optimizing only the unresolved objective
        ↓
finish episode when all slots are protected or budget expires
```

There is no call to loss.backward(), no reverse traversal of the model, and no backward activation tape. Accepted parameter changes persist into later examples.

The defining rule is:

> **Freeze a slot's achieved objective, not the shared parameters that produced it.**

A protected slot:

- stops driving exploration;
- remains active in Gator's shared communication;
- may continue improving;
- and can veto candidate updates that destroy too much of its earned progress.

The first implementation should be deliberately simple: batch size one, generated prefix data, FP32 parameters, sequential candidate evaluation, antithetic perturbations, and direct retention of the winning perturbation. Candidate batching, language alignment, replay anchors, and momentum can be added only after the basic learning mechanism is demonstrated.

---

## 1. Motivation

Gator forms all output positions simultaneously. Each slot has its own hidden state, graph route, and participation across multiple work streams. Standard backpropagation collapses all slot errors into one scalar objective and distributes infinitesimal credit backward through the entire shared computation.

Procoim instead asks explicit forward counterfactuals:

```text
Did this change improve slot 3?
Did it help the unresolved formation overall?
Did it damage slots 0, 1, and 5 after they had already improved?
Which Gator subsystem produced the most useful forward change?
```

This approach is attractive for Gator because the architecture already exposes natural boundaries for structured perturbation:

- routing and stewardship;
- node transforms;
- work-stream mixing;
- receiver memory;
- formation readout;
- slot-address adapters;
- blank and EOS behavior.

The hypothesis is not that generic random search will beat backpropagation. The hypothesis is narrower:

> Gator's explicit slot structure and multiple paths may make forward counterfactual credit assignment unusually practical.

---

## 2. Goals

### 2.1 MVP goals

The MVP must demonstrate that:

1. training uses forward passes only;
2. per-slot loss reduction targets can be reached progressively;
3. solved slots can be protected without freezing shared parameters;
4. accepted perturbations persist into later examples;
5. fresh generated prefix accuracy rises above chance;
6. peak activation memory is lower than reverse-mode training.

### 2.2 Secondary goals

After the MVP works:

- compare total forward evaluations with AdamW updates;
- batch candidate perturbations on GPU;
- add replay anchors for cross-example retention;
- adapt Procoim to monotonic language formation;
- use Gator-specific route and stream perturbations;
- add forward-only momentum or directional Adam;
- test whether primary/corrector work streams still emerge.

### 2.3 Non-goals for the first prototype

The MVP does not need to:

- outperform AdamW in wall-clock time;
- train free-form language immediately;
- solve every episode completely;
- support mixed precision;
- support distributed training;
- permanently freeze parameters for individual slots;
- implement a biologically plausible learning rule;
- or remove all forgetting across examples.

The first question is simply:

> **Can Procoim make a Gator learn a generated simultaneous prefix task above chance without a backward pass?**

---

## 3. Terminology

**Episode**
Repeated forward optimization on one fixed example or microbatch.

**Slot**
One output position in the simultaneous formation.

**Active slot**
A valid output slot that has not yet achieved the configured loss reduction.

**Soft-frozen slot**
A slot that reached its target but has not yet demonstrated stability across several accepted shared updates.

**Protected slot**
A stable solved slot that no longer drives exploration and may veto excessively damaging candidate updates.

**Candidate**
One proposed structured change to a parameter group.

**Candidate bank**
A set of candidate perturbations evaluated from the same exact baseline parameter state.

**Protection anchor**
The best stored loss achieved by a soft-frozen or protected slot.

**Forward trial**
One candidate forward evaluation. Baseline forward evaluations must also be counted separately.

**Persistent update**
An accepted parameter change retained after the episode and carried into future examples.

---

## 4. Design principles

1. Forward-only means forward-only. No backward(), autograd.grad, reverse-mode hooks, or hidden gradient calculations.
2. Slots are objectives, not parameter owners. A slot cannot exclusively own shared Gator parameters.
3. Solved slots remain communicative. Protection must not remove their hidden states, messages, or routes.
4. Every candidate begins from the same baseline. Exact parameter restoration is mandatory.
5. Progress is measured relative to each slot's own starting loss. High-loss slots should not dominate merely because their CE scale is larger.
6. Protection is tolerant, not perfectly monotonic. Tiny regressions are acceptable; destructive regressions are not.
7. Search must be structured. Perturb complete functional blocks or low-dimensional directions, never every scalar independently.
8. Report forward work honestly. One Procoim episode may use dozens of forwards and is not comparable to one AdamW update.
9. Prefix first, language later. Language introduces alignment ambiguity that can obscure whether the optimizer works.
10. Determinism inside an episode is non-negotiable. Candidate comparisons are invalid if dropout or route noise differs.

---

## 5. Required Gator and task interfaces

### 5.1 Model forward

The optimizer requires a deterministic forward path that returns simultaneous logits:

```python
logits = model(inputs, **forward_kwargs)  # [B, O, V]
```

Optional diagnostic output is strongly recommended:

```python
output = model(
    inputs,
    return_diagnostics=True,
    deterministic_search=True,
)

logits = output.logits
routes = output.routes
stream_weights = output.stream_weights
hidden_states = output.hidden_states
```

Diagnostics are not required to score the MVP, but they will support later Gator-specific perturbations.

### 5.2 Slot-loss provider

Every task must provide a decomposed loss tensor:

```python
@dataclass
class SlotLossResult:
    losses: torch.Tensor       # [B, O], detached forward values
    valid_mask: torch.Tensor   # [B, O], bool
    aggregate: torch.Tensor    # scalar
    metadata: dict[str, Any]
```

For generated prefix:

```python
slot_loss = F.cross_entropy(
    logits.transpose(1, 2),
    targets,
    reduction="none",
)  # [B, O]
```

Padding and ignored output positions must have valid_mask=False.

### 5.3 Parameter groups

Gator should expose named groups:

```python
def procoim_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
    return {
        "routing": [...],
        "receiver_memory": [...],
        "node_compute": [...],
        "stream_mix": [...],
        "readout": [...],
        "embedding": [...],
    }
```

The grouping function must:

- return each trainable parameter exactly once;
- omit frozen parameters;
- preserve a deterministic group order;
- and optionally attach group metadata such as preferred perturbation scale.

If no grouping method exists, Procoim may create one all_parameters group for a smoke test, but this is not the intended final mode.

---

## 6. Per-slot objective formulation

For valid slot i, baseline loss is:

```text
ℓ_i^start
```

For requested fractional reduction rho:

```text
ℓ_i^target = max(ℓ_floor, (1 - ρ) · ℓ_i^start)
```

Example:

```text
start loss:          4.00
reduction fraction:  0.10
loss floor:          0.01
target loss:         3.60
```

### 6.1 Minimum absolute reduction

A very low-loss slot could satisfy a percentage target with a meaningless change. Optionally require:

```text
ℓ_i^target = min((1 - ρ) · ℓ_i^start, ℓ_i^start - Δ_min)
```

before applying the floor.

The MVP may set minimum_absolute_reduction=0.0, but the field should exist.

### 6.2 Normalized unresolved gap

Raw CE values can differ greatly. Normalize progress per slot:

```text
g_i = max(0, (ℓ_i - ℓ_i^target) / (ℓ_i^start - ℓ_i^target + ε))
```

Interpretation:

```text
g_i = 1.0  no progress from the episode baseline
g_i = 0.5  halfway to target
g_i = 0.0  target reached or exceeded
```

The active objective is:

```text
J_active = Σ_i [i ∈ ACTIVE] · w_i · g_i / (Σ_i [i ∈ ACTIVE] · w_i + ε)
```

Default w_i=1. Later modes may increase the weight of content tokens, late positions, or chronically difficult slots.

### 6.3 Aggregate episode target

Record total valid-slot loss at the start:

```text
L^start = mean_valid(ℓ_i^start)
```

Aggregate target:

```text
L^target = (1 - ρ_aggregate) · L^start
```

The default prefix completion criterion should still require all valid slots to reach protection. Aggregate reduction is logged and may be used in alternative completion modes.

---

## 7. Slot lifecycle

Each valid slot has one status:

```text
ACTIVE → SOFT_FROZEN → PROTECTED
                  ↘
                    ACTIVE  (reopened when stability fails)
```

An optional later state, LOCKED_FOR_EPISODE, may be added for experiments, but it should not exist in the MVP.

### 7.1 ACTIVE

An active slot:

- contributes to J_active;
- may improve or regress as shared parameters change;
- is eligible to become soft-frozen after reaching target;
- continues all normal Gator communication.

### 7.2 SOFT_FROZEN

A soft-frozen slot:

- has reached target_loss;
- no longer contributes to the active search objective;
- contributes a regression penalty;
- does not yet hard-veto candidate updates;
- must remain under target for stability_required accepted updates.

If it rises above target, it reopens as ACTIVE and its stability count resets.

### 7.3 PROTECTED

A protected slot:

- no longer drives exploration;
- remains in all relays and communication;
- may continue improving;
- stores its best loss as a protection anchor;
- hard-vetoes candidates that exceed its regression limit.

A protected slot is not permanently immutable. An optional reopen_protected mode may reopen it if repeated accepted updates leave it near the hard boundary, but this should be disabled in the MVP.

### 7.4 State stored per slot

```python
@dataclass
class ProcoimSlotState:
    status: torch.Tensor              # uint8 [B, O]
    start_loss: torch.Tensor          # float32 [B, O]
    target_loss: torch.Tensor         # float32 [B, O]
    current_loss: torch.Tensor        # float32 [B, O]
    best_loss: torch.Tensor           # float32 [B, O]
    stable_count: torch.Tensor        # int32 [B, O]
    valid_mask: torch.Tensor          # bool [B, O]
```

Keep this state on the same device as candidate loss evaluation unless CPU transfer is proven cheaper.

---

## 8. Protection without blocking shared learning

The main design challenge is protecting solved slots while allowing shared Gator weights to keep improving unresolved ones.

### 8.1 Achieved gain

For slot i:

```text
Δ_i = max(0, ℓ_i^start - ℓ_i^best)
```

### 8.2 Soft and hard tolerances

```text
ε_i^soft = ρ_soft · Δ_i + ε_abs
ε_i^hard = ρ_hard · Δ_i + ε_abs
```

Recommended relationship:

```text
soft_regression_fraction < hard_regression_fraction
```

Example:

```text
start loss:          4.0
best loss:           3.0
achieved gain:       1.0
soft fraction:       0.02
hard fraction:       0.08
absolute tolerance:  0.0001
soft margin:         0.0201
hard margin:         0.0801
```

### 8.3 Soft protection penalty

For soft-frozen and protected slots:

```text
r_i^soft = max(0, ℓ_i^cand - ℓ_i^best - ε_i^soft)
```

Aggregate penalty:

```text
J_protect = mean(r_i^soft)
```

### 8.4 Hard protection veto

A candidate is ineligible when any protected slot violates:

```text
ℓ_i^cand > ℓ_i^best + ε_i^hard
```

The comparison is against the best stored loss, not the previous candidate. This prevents cumulative drift.

### 8.5 Why parameters are never frozen per slot

A router matrix, node transform, or shared readout can contribute to every slot. Freezing it because slot 2 improved could make slot 7 impossible to solve. Procoim therefore freezes only objective participation and enforces output constraints through forward evaluation.

### 8.6 Optional local rescue state

A later extension may attach a temporary per-slot latent offset:

```text
h_i^effective = h_i + δ_i
```

When a slot first reaches target, its small delta_i can preserve the solution while shared parameters continue adapting. These offsets must later be distilled into shared parameters and reset. This is not part of the MVP because it changes the model state beyond ordinary parameters.

---

## 9. Candidate perturbation engine

### 9.1 Why structured perturbations

Perturbing every scalar independently would require an impossible number of forward trials. Procoim perturbs coherent functional blocks or low-dimensional directions.

Candidate unit examples:

- all parameters in the routing group;
- one router matrix;
- one node-program bank;
- one work-stream adapter family;
- one readout projection;
- one formation blank/EOS adapter;
- one low-rank direction shared across related matrices.

### 9.2 Antithetic directions

For selected group g, sample direction u and evaluate:

```text
θ_g⁺ = θ_g + ε_g · u
θ_g⁻ = θ_g - ε_g · u
```

Using +u and -u reduces directional noise and provides a crude local slope estimate.

### 9.3 Direction distributions

MVP default:

```text
Rademacher noise: each element is -1 or +1
```

Alternatives for later experiments:

- Gaussian direction;
- orthogonal candidate directions;
- low-rank matrix perturbations;
- row-wise or column-wise block perturbations;
- Gator route-specific bias perturbations;
- historical successful direction replay.

### 9.4 Direction normalization

For each parameter tensor p and raw noise u:

```python
u = u / (u.norm() + eps)
scale = max(p.detach().float().pow(2).mean().sqrt(), parameter_rms_floor)
delta = probe_scale * scale * u
```

For a multi-tensor group, normalize either:

- each tensor independently; or
- the entire flattened group as one vector.

Use per-tensor normalization in the MVP to prevent one large tensor from consuming all perturbation magnitude.

### 9.5 Exact restoration

Every candidate in one bank must start from the same parameters.

Required sequence:

1. clone the selected group into FP32 baseline buffers;
2. restore those buffers before each candidate;
3. apply candidate delta under torch.no_grad();
4. run deterministic forward evaluation;
5. record detached losses and diagnostics;
6. restore the exact baseline buffers;
7. apply only the winning delta after selection.

Do not rely only on add_(delta) followed by sub_(delta). Floating-point drift can accumulate across thousands of trials.

### 9.6 Candidate bank size

If candidate_directions=8 and antithetic=True, one bank evaluates 16 perturbed forwards.

Count and log:

```text
baseline forwards
candidate forwards
post-accept verification forwards
```

### 9.7 Parameter-group selection

MVP schedule is round-robin:

```text
routing
stream_mix
node_compute
readout
```

Optional weighted schedule:

```python
probability(group) ∝ acceptance_rate_ema[group] + exploration_floor
```

Do not use a pure greedy schedule initially. Every group needs some exploration.

### 9.8 Group-scale adaptation

Each parameter group maintains probe_scale.

Suggested rule:

```text
if bank produces accepted candidate:
    scale *= growth_factor
else:
    scale *= shrink_factor

clip to [min_probe_scale, max_probe_scale]
```

Recommended initial values:

```text
growth_factor: 1.05
shrink_factor: 0.70
```

A bank that triggers many hard vetoes should shrink faster.

---

## 10. Candidate scoring

For candidate k, compute:

### 10.1 Active gain

```text
G_k^active = J_active^current - J_active^candidate
```

### 10.2 Protection penalty

```text
P_k = J_protect^candidate
```

### 10.3 Optional global loss guard

A candidate may improve normalized active gaps while making total CE dramatically worse. Define:

```text
R_k^global = max(0, L_global^candidate - L_global^current - ε_global)
```

### 10.4 Candidate utility

```text
U_k = G_k^active - λ_protect · P_k - λ_global · R_k^global
```

### 10.5 Eligibility

A candidate is eligible only if:

```text
no hard protected-slot violation
active gain >= minimum_active_gain
utility > 0
all losses are finite
```

Optional active-slot cap:

```text
no active slot may regress more than active_regression_cap
```

This cap should be disabled initially because some active slots may need to temporarily regress while the aggregate improves.

### 10.6 Selection

Choose the eligible candidate with maximum utility. Ties are resolved by:

1. larger active gain;
2. smaller maximum protected regression;
3. smaller global loss;
4. deterministic candidate index.

### 10.7 Post-accept verification

After applying the winning delta permanently, run one verification forward from the accepted model state. This catches restoration or stateful-layer bugs.

The verified losses, not the candidate's cached losses, become the current episode state.

---

## 11. Forward-only persistent update policies

### 11.1 Mode A: direct winner retention, recommended MVP

Keep the winning candidate delta exactly:

```text
θ ← θ + δ_winner
```

Advantages:

- simplest;
- easy to debug;
- no hidden optimizer state;
- faithful to the proposed forward search.

Disadvantages:

- noisy;
- order-dependent;
- may require many forwards.

### 11.2 Mode B: antithetic directional estimate

From losses for +u and -u:

```text
d_g = (J(θ + ε·u) - J(θ - ε·u)) / (2ε)
```

Estimated gradient direction:

```text
ĝ_g = d_g · u
```

Apply a forward-only SGD or Adam-like update. This remains forward-only, but it is a later mode because protection constraints complicate combining multiple estimates.

### 11.3 Mode C: accumulated accepted direction

Accumulate accepted deltas during an episode, normalize them, and apply one consolidated update at episode end. This reduces repeated parameter drift but makes slot protection less immediate.

### 11.4 Recommendation

Implement only Mode A in the MVP. Design interfaces so Mode B can be added without rewriting episode logic.

---

## 12. Episode algorithm

### 12.1 High-level pseudocode

```python
@torch.no_grad()
def run_episode(model, batch, slot_loss_provider, config):
    search_context = make_deterministic_search_context(model, config)

    baseline = evaluate(model, batch, slot_loss_provider, search_context)
    state = initialize_episode_state(baseline, config)

    accepted_updates = 0
    candidate_banks = 0
    stall_banks = 0

    while not episode_complete(state, config):
        if accepted_updates >= config.max_accepted_updates:
            break
        if candidate_banks >= config.max_candidate_banks:
            break
        if stall_banks >= config.max_stall_banks:
            break

        group = select_parameter_group(model, state, config)
        bank = build_candidate_bank(model, group, state, config)

        results = evaluate_candidate_bank(
            model=model,
            batch=batch,
            group=group,
            bank=bank,
            state=state,
            slot_loss_provider=slot_loss_provider,
            search_context=search_context,
        )

        winner = select_winner(results, state, config)
        candidate_banks += 1

        if winner is None:
            adapt_scale_after_failure(group, results, config)
            stall_banks += 1
            continue

        apply_delta_exactly(group, winner.delta)
        verified = evaluate(model, batch, slot_loss_provider, search_context)

        if not verify_winner(verified, winner, state, config):
            restore_group(group)
            adapt_scale_after_failure(group, results, config)
            stall_banks += 1
            continue

        update_episode_state(state, verified, config)
        adapt_scale_after_success(group, winner, config)

        accepted_updates += 1
        stall_banks = 0

    final = evaluate(model, batch, slot_loss_provider, search_context)
    return build_episode_metrics(state, baseline, final, config)
```

### 12.2 Candidate-bank pseudocode

```python
@torch.no_grad()
def evaluate_candidate_bank(...):
    baseline_params = clone_group_parameters(group)
    results = []

    directions = sample_directions(group, config)

    for direction_id, direction in enumerate(directions):
        signs = (+1, -1) if config.antithetic else (+1,)

        for sign in signs:
            restore_group_parameters(group, baseline_params)
            delta = make_scaled_delta(group, direction, sign, config)
            add_delta(group, delta)

            trial = evaluate(
                model,
                batch,
                slot_loss_provider,
                search_context,
            )

            results.append(
                score_candidate(
                    trial=trial,
                    delta=delta,
                    direction_id=direction_id,
                    sign=sign,
                    episode_state=state,
                    config=config,
                )
            )

    restore_group_parameters(group, baseline_params)
    return results
```

---

## 13. Freeze and completion logic

After every accepted and verified update:

1. set current_loss from verified forward output;
2. update best_loss = min(best_loss, current_loss) for all valid slots;
3. promote active slots at or below target to SOFT_FROZEN;
4. increment soft-frozen stable_count while they remain below target;
5. reopen soft-frozen slots that rise above target;
6. promote soft-frozen slots after stability_required accepted updates;
7. retain protected slots unless optional reopening is enabled.

### 13.1 Completion modes

**all_slots**

All valid slots are PROTECTED.

Recommended for prefix MVP.

**aggregate_and_fraction**

Complete when:

```text
aggregate loss reduction target reached
and protected fraction >= minimum_protected_fraction
```

Useful for language where exact protection of every slot may be unrealistic.

**budget**

Always stop at configured trial or accepted-update budget.

Budget limits apply in every mode.

### 13.2 Slots already below floor

If baseline loss is already below loss_floor, mark the slot soft-frozen immediately and require the normal stability streak. Do not mark it protected without verification.

### 13.3 No active slots but soft-frozen remain

If all unresolved slots are soft-frozen but not yet protected, candidate search should continue only if required for stability. A cheaper option is to run no-op verification forwards. MVP recommendation:

```text
perform deterministic no-op forwards and advance stability counts
only if losses remain below target
```

This avoids perturbing a solved formation merely to satisfy the streak.

---

## 14. Deterministic search mode

Candidate evaluation is meaningless if stochasticity differs across candidates.

During one episode:

- execute under torch.no_grad();
- disable dropout;
- disable stochastic depth;
- freeze batch-norm running statistics;
- reuse identical route exploration noise;
- reuse identical stewardship exploration masks;
- reuse identical tokenization and padding;
- keep model train/eval semantics explicitly controlled;
- avoid stateful caches that mutate between trials.

### 14.1 RNG snapshot

Before baseline evaluation, capture:

```python
python_rng_state
numpy_rng_state
torch_cpu_rng_state
torch_cuda_rng_state_all
```

Restore the same model-forward RNG state before every candidate if the architecture cannot disable all stochastic features.

Use a separate RNG generator for candidate direction sampling so restoring forward RNG does not repeat candidate directions.

### 14.2 Stateful Gator caches

Block caches, route caches, or mutable formation buffers must be either:

- disabled during Procoim search;
- rebuilt from the same baseline before every candidate;
- or proven read-only and invariant.

No candidate may inherit cache state produced by a previous candidate.

---

## 15. Gator-specific parameter groups

Recommended default grouping:

### 15.1 routing

- node transition routers;
- route biases;
- stewardship logits;
- route temperature parameters if learned;
- work-stream ownership gates.

### 15.2 receiver_memory

- convolutional receiver kernels;
- relation-aware read biases;
- mailbox write/read projections;
- context mixing gates.

### 15.3 node_compute

- node-local transforms;
- shared node-program bases;
- node embeddings;
- residual update gates.

### 15.4 stream_mix

- work-stream communication projections;
- hidden-state fusion adapters;
- correction-stream gates;
- cross-stream message gates.

### 15.5 readout

- terminal fusion weights;
- output projection adapters;
- blank/EOS logits if separately parameterized;
- vocabulary head only in later language experiments.

### 15.6 embedding

- token embedding;
- slot-address embeddings;
- role embeddings.

Embeddings should remain frozen for the first generated-prefix test unless they are integral to the task.

### 15.7 Fine-grained subgroups

After the MVP, each broad group may be split by relay, node family, or work stream. Do not begin with hundreds of groups because search scheduling will dominate.

---

## 16. Prefix-task mode

Generated simultaneous parallel prefix is the canonical MVP because:

- every target slot has one exact answer;
- slot loss is unambiguous;
- fresh examples are unlimited;
- chance is known;
- Gator already has a validated backprop baseline;
- exact-formation accuracy is meaningful.

### 16.1 Suggested initial task

Use the smallest configuration that still tests interaction:

```text
modulus:                 12
training output length:  4 or 7
batch size:               1
fresh example each episode
formation readout:       simultaneous
```

### 16.2 Progressive test ladder

1. overfit one fixed example;
2. alternate between two fixed examples;
3. train on freshly generated examples;
4. evaluate on fresh generated examples;
5. scale from length 4 to length 7;
6. compare with AdamW baseline.

### 16.3 Prefix success metrics

- fresh token accuracy;
- exact-formation accuracy;
- validation CE;
- episodes to leave chance;
- total forward trials to leave chance;
- protected slots per episode;
- cross-example regression;
- peak memory;
- wall-clock time.

---

## 17. Generalization and retention across examples

Accepted updates persist, but slot protection is episode-local. Repeatedly optimizing one example can therefore overfit.

### 17.1 MVP behavior

Use one fresh generated example per episode. Do not add replay until the basic mechanism learns anything.

### 17.2 Anchor replay extension

Maintain a small buffer of prior examples:

```text
current example
+ K anchor examples
```

For every candidate, evaluate current-example utility and anchor regression.

Anchor guard:

```text
R_anchor = max(0, L_anchor^cand - L_anchor^best - ε_anchor)
```

Candidate utility becomes:

```text
U_k = G_k^active - λ_protect · P_k - λ_anchor · R_anchor
```

Initial anchor count should be 1–4, not a large replay batch.

### 17.3 Anchor selection

Possible strategies:

- reservoir sampling;
- most recently learned examples;
- highest observed forgetting;
- diverse rule/start combinations;
- examples that depend strongly on complementary streams.

### 17.4 Episode-to-episode baseline

Store validation snapshots periodically. Do not use validation data for candidate acceptance, only for external measurement.

---

## 18. Language mode and monotonic alignment

Language is explicitly outside the MVP because one exact target can have many valid paraphrases and output-slot alignment can shift.

### 18.1 Requirement: stable slotwise decomposition

Procoim requires a meaningful [B, O] loss. A single global sequence loss cannot support slot freezing.

For monotonic alignment:

1. compute the best alignment between output slots and target tokens;
2. assign matched-token cost to the emitting output slot;
3. assign insertion, blank, and EOS costs to concrete slots;
4. expose the chosen alignment map in metadata.

### 18.2 Freeze alignment inside a candidate bank

If alignment is recomputed independently for every candidate, candidates may appear better merely by shifting token assignment.

Recommended language behavior:

```text
baseline forward
→ compute baseline monotonic alignment path
→ freeze that alignment map for all candidates in one bank
→ accept winner
→ recompute alignment after verified accepted update
```

This keeps slot identity stable during direct candidate comparison while allowing the formation to realign after genuine progress.

### 18.3 Protected target identity

For protected language slots, store both:

- output slot index;
- aligned target-token index or blank/EOS role.

If a new alignment radically changes that mapping, the slot may need to reopen.

### 18.4 Sequence-level guard

A later language version should include a whole-formation guard for:

- monotonic aligned loss;
- repetition;
- blank overuse;
- EOS placement;
- optional semantic score.

Per-slot protection alone can preserve individually likely tokens while destroying sentence coherence.

### 18.5 Multiple acceptable answers

Future extensions may use:

- teacher probability distributions;
- multiple references;
- semantic reward models;
- canonical deterministic tasks before dialogue.

These are not part of the first implementation.

---

## 19. Performance and memory characteristics

### 19.1 Expected memory advantage

Without reverse-mode backpropagation, Procoim does not need to retain the complete activation graph for backward. Candidate forwards may release intermediate activations immediately.

Expected memory use includes:

- model parameters;
- current forward activations;
- cloned parameter group baseline;
- candidate direction buffers;
- per-slot episode state;
- optional anchor examples.

### 19.2 Expected compute disadvantage

The MVP may perform tens or hundreds of forwards per episode. It may be slower than AdamW despite using less memory.

Report both:

```text
episodes
accepted updates
candidate banks
candidate forwards
total forward passes
training examples consumed
wall-clock time
```

### 19.3 Candidate batching

After correctness is established, candidate variants can be vectorized using:

- torch.func.functional_call;
- vmap over candidate parameter sets;
- an expanded candidate batch dimension;
- static CUDA graphs for fixed shapes.

Do not mutate model parameters concurrently in a batched candidate implementation. Use functional parameter dictionaries.

### 19.4 Clone cost

Cloning a large parameter group every bank can dominate runtime. Later optimizations:

- persistent FP32 baseline buffers;
- perturbation overlays instead of physical copies;
- functional parameter application;
- smaller subgroup scheduling;
- master-weight views.

### 19.5 Mixed precision

MVP uses FP32. Later:

- keep FP32 master weights;
- evaluate forwards in BF16 where stable;
- apply perturbations to FP32 masters;
- ensure probe_scale is larger than BF16 quantization noise;
- compare candidate ranking consistency between FP32 and BF16.

---

## 20. Proposed code layout

Create:

```text
gator/optimizers/
    __init__.py
    procoim.py
    procoim_state.py
    procoim_candidates.py
    slot_loss.py
```

### 20.1 slot_loss.py

```python
from dataclasses import dataclass, field
from typing import Any, Protocol

@dataclass
class SlotLossResult:
    losses: torch.Tensor
    valid_mask: torch.Tensor
    aggregate: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

class SlotLossProvider(Protocol):
    def __call__(
        self,
        logits: torch.Tensor,
        batch: Any,
        *,
        fixed_metadata: dict[str, Any] | None = None,
    ) -> SlotLossResult: ...
```

### 20.2 procoim_state.py

```python
class SlotStatus(IntEnum):
    ACTIVE = 0
    SOFT_FROZEN = 1
    PROTECTED = 2

@dataclass
class ProcoimEpisodeState:
    status: torch.Tensor
    start_loss: torch.Tensor
    target_loss: torch.Tensor
    current_loss: torch.Tensor
    best_loss: torch.Tensor
    stable_count: torch.Tensor
    valid_mask: torch.Tensor
    accepted_updates: int = 0
    candidate_banks: int = 0
    candidate_forwards: int = 0
    stall_banks: int = 0
```

### 20.3 procoim_candidates.py

```python
@dataclass
class ParameterGroupSpec:
    name: str
    parameters: list[nn.Parameter]
    probe_scale: float
    min_probe_scale: float
    max_probe_scale: float

@dataclass
class CandidateDelta:
    group_name: str
    tensors: list[torch.Tensor]
    direction_id: int
    sign: int

@dataclass
class CandidateResult:
    delta: CandidateDelta
    slot_losses: torch.Tensor
    aggregate_loss: float
    active_gain: float
    protection_penalty: float
    global_regression: float
    max_protected_regression: float
    hard_violation: bool
    utility: float
    finite: bool
```

### 20.4 procoim.py

```python
class ProcoimOptimizer:
    def __init__(self, config: ProcoimConfig): ...

    def run_episode(
        self,
        model: nn.Module,
        batch: Any,
        slot_loss_provider: SlotLossProvider,
    ) -> dict[str, float]: ...

    def evaluate_baseline(...): ...
    def initialize_state(...): ...
    def select_parameter_group(...): ...
    def sample_candidate_bank(...): ...
    def evaluate_candidate_bank(...): ...
    def score_candidate(...): ...
    def select_winner(...): ...
    def apply_winner(...): ...
    def verify_winner(...): ...
    def update_slot_states(...): ...
    def episode_complete(...): ...
```

### 20.5 Training integration

Modify gator/training.py:

```python
if train_config.optimizer_name == "procoim":
    metrics = procoim_optimizer.run_episode(
        model=model,
        batch=batch,
        slot_loss_provider=task.slot_loss_provider,
    )
else:
    metrics = standard_backprop_step(...)
```

Do not instantiate a normal PyTorch optimizer in direct-winner MVP mode.

---

## 21. Configuration

```python
@dataclass
class ProcoimConfig:
    reduction_fraction: float = 0.10
    aggregate_reduction_fraction: float = 0.10
    minimum_absolute_reduction: float = 0.0
    loss_floor: float = 0.01

    candidate_directions: int = 8
    antithetic: bool = True
    max_candidate_banks: int = 16
    max_stall_banks: int = 6
    max_accepted_updates: int = 64

    stability_required: int = 3
    soft_regression_fraction: float = 0.02
    hard_regression_fraction: float = 0.08
    absolute_regression_tolerance: float = 1e-4

    protection_weight: float = 2.0
    global_regression_weight: float = 0.25
    global_regression_tolerance: float = 0.0
    minimum_active_gain: float = 1e-4
    active_regression_cap: float | None = None

    initial_probe_scale: float = 1e-3
    min_probe_scale: float = 1e-6
    max_probe_scale: float = 5e-2
    probe_growth_factor: float = 1.05
    probe_shrink_factor: float = 0.70
    parameter_rms_floor: float = 1e-3

    completion_mode: str = "all_slots"
    minimum_protected_fraction: float = 1.0

    deterministic_search: bool = True
    verify_accepted_candidate: bool = True
    batch_candidates: bool = False
    forward_dtype: str = "float32"

    parameter_group_schedule: tuple[str, ...] = (
        "routing",
        "stream_mix",
        "node_compute",
        "readout",
    )

    anchor_count: int = 0
    anchor_weight: float = 0.0
    anchor_tolerance: float = 0.0

    seed: int = 0
```

### 21.1 Example YAML

```yaml
optimizer:
  name: procoim
  reduction_fraction: 0.10
  aggregate_reduction_fraction: 0.10
  loss_floor: 0.01

  candidate_directions: 8
  antithetic: true
  max_candidate_banks: 16
  max_stall_banks: 6
  max_accepted_updates: 64

  stability_required: 3
  soft_regression_fraction: 0.02
  hard_regression_fraction: 0.08
  absolute_regression_tolerance: 0.0001

  protection_weight: 2.0
  global_regression_weight: 0.25
  minimum_active_gain: 0.0001

  initial_probe_scale: 0.001
  probe_growth_factor: 1.05
  probe_shrink_factor: 0.70

  parameter_group_schedule:
    - routing
    - stream_mix
    - node_compute
    - readout

  deterministic_search: true
  verify_accepted_candidate: true
  batch_candidates: false
  seed: 0
```

---

## 22. Logging and diagnostics

### 22.1 Per episode

```text
episode_index
example_id
start_aggregate_loss
end_aggregate_loss
aggregate_reduction
active_slots_start/end
soft_frozen_slots_start/end
protected_slots_start/end
slot_target_hit_rate
accepted_updates
candidate_banks
candidate_forwards
baseline_forwards
verification_forwards
acceptance_rate
hard_veto_rate
stall_banks
max_protected_regression
mean_protected_regression
mean_active_gain
best_candidate_utility
wall_clock_seconds
peak_allocated_memory
```

### 22.2 Per parameter group

```text
banks_attempted
candidates_evaluated
accepted_candidates
acceptance_rate
probe_scale
mean_active_gain
hard_veto_rate
mean_utility
```

### 22.3 Slot progression

At configurable intervals, log:

```text
slot index
start loss
target loss
best loss
current loss
status
stability count
first target-hit trial
first protected trial
```

### 22.4 External validation

On fresh generated examples:

```text
validation loss
token accuracy
exact formation accuracy
accuracy by position
step consistency if applicable
forward evaluations consumed
```

### 22.5 Honest comparison axes

Compare Procoim and AdamW by:

- examples consumed;
- forward-equivalent compute;
- wall-clock time;
- peak memory;
- validation accuracy;
- time and work to leave chance.

Do not compare one episode to one optimizer update without noting candidate forwards.

---

## 23. Checkpointing and resume

Checkpoint persistent training state:

```text
model parameters
Procoim configuration
per-group probe scales
per-group acceptance statistics
candidate RNG state
global episode index
anchor buffer if enabled
training data generator state
```

Episode-local slot state does not need to be checkpointed in the MVP. Resume at the next example. Later, long language episodes may support mid-episode resume.

Use atomic checkpoint writes because a Procoim run may contain many accepted micro-updates between ordinary logging steps.

---

## 24. Unit tests Codex must add

**Slot-loss tests**

1. Per-slot CE equals a manually calculated reference.
2. Invalid and padded positions never participate.
3. Aggregate loss uses only valid positions.

**Lifecycle tests**

4. Active slot reaches target and becomes soft-frozen.
5. Stability streak promotes soft-frozen to protected.
6. Soft-frozen slot above target reopens.
7. Protected slot can improve and updates its best anchor.

**Protection tests**

8. Candidate inside soft tolerance receives no penalty.
9. Candidate outside soft tolerance receives the correct penalty.
10. Candidate outside hard tolerance is vetoed.
11. Repeated accepted updates cannot cumulatively drift beyond best-loss protection.

**Candidate-engine tests**

12. Antithetic deltas are exact negatives.
13. Parameter restoration is bitwise exact in FP32 where possible.
14. Non-selected groups are unchanged.
15. Winning delta is applied exactly once.
16. Failed candidate bank leaves model parameters unchanged.
17. Candidate ranking is deterministic under fixed seed.

**Forward-only tests**

18. No parameter .grad is created.
19. torch.autograd.backward is never called.
20. torch.autograd.grad is never called.
21. Forward evaluation runs under torch.no_grad().

**Edge-case tests**

22. Already-low slot is handled correctly.
23. Zero valid slots raises a clear error.
24. NaN or Inf candidate is rejected.
25. All slots protected terminates the episode.
26. Stall budget terminates the episode.

---

## 25. Integration tests

### 25.1 Toy scalar model

Create a tiny deterministic model with two shared parameters and three output slots. Verify that Procoim:

- improves at least one slot;
- protects it;
- continues changing shared parameters;
- and does not violate protection.

### 25.2 Fixed prefix example

Train repeatedly on one generated prefix example for 100–500 episodes.

Expected:

- training loss falls;
- at least some slots become protected;
- no backward graph is created.

This tests optimizer mechanics, not generalization.

### 25.3 Two-example alternation

Alternate two different examples. Verify that accepted updates can improve both rather than permanently destroying the previous one.

### 25.4 Fresh generated prefix

Use fresh examples and a small Gator. Verify:

- validation accuracy rises above chance;
- accepted updates persist;
- protection works within episodes;
- memory is below backprop baseline.

Automated CI should not require near-perfect performance because stochastic search may vary. Require a statistically clear loss reduction under a fixed seed.

---

## 26. Experimental roadmap

**Phase 0: instrumentation**

- Add slot-loss provider.
- Add deterministic search mode.
- Add parameter groups.
- Add tests proving no backward use.

**Phase 1: one-example mechanics**

- Direct winner retention.
- Batch size one.
- Sequential candidate evaluation.
- FP32.
- Verify slot state transitions and protection.

**Phase 2: two-example retention**

- Alternate two fixed examples.
- Tune hard/soft tolerances.
- Add a one-example anchor only if necessary.

**Phase 3: generated prefix learning**

- Fresh examples.
- Measure validation accuracy.
- Sweep candidate count, probe scale, and threshold.
- Compare to chance and AdamW.

**Phase 4: Gator-specific search**

- Split parameter groups more finely.
- Prioritize groups by acceptance history.
- Add route-bias and stream-fusion perturbations.
- Test whether complementary streams emerge.

**Phase 5: candidate batching**

- Functional parameter calls.
- GPU candidate dimension.
- Static shapes and CUDA Graphs.
- Measure candidate forwards per second.

**Phase 6: language formation**

- Slotwise monotonic alignment decomposition.
- Stable alignment per candidate bank.
- Sequence-level guard.
- Anchor replay.
- Start with deterministic extraction or copying tasks before dialogue.

---

## 27. Failure modes and mitigations

### 27.1 No candidate ever improves

Possible causes:

- probe scale too small;
- probe scale too large;
- parameter group too broad;
- stochastic forward path;
- model is at a local plateau;
- candidate count too small.

Mitigation:

- adapt scale;
- split groups;
- increase directions;
- verify deterministic outputs;
- test a known easy fixed example.

### 27.2 Protection veto paralysis

As more slots protect, every candidate may be rejected.

Mitigation:

- use soft tolerance before hard protection;
- increase stability requirement;
- allow small relative regressions;
- shrink probe scale;
- perturb a more local parameter group;
- add local rescue offsets only in later experiments.

Do not silently relax hard protection inside an episode. Log and end the episode if necessary.

### 27.3 Slot swapping

One active slot improves while another repeatedly loses progress.

Mitigation:

- normalized active gaps;
- optional active regression cap;
- promote target-reaching slots quickly;
- track per-slot best loss even while active.

### 27.4 Easy-token capture

Spaces, punctuation, or common tokens protect immediately while content slots never improve.

Mitigation:

- slot weights;
- minimum absolute reduction;
- content-token metrics;
- later language-specific loss decomposition.

### 27.5 Overfitting one episode

Training loss improves but fresh validation does not.

Mitigation:

- fresh example every episode;
- fewer accepted updates per example;
- anchor replay;
- microbatch episodes;
- validation gating at experiment level, not candidate level.

### 27.6 Candidate comparison noise

Same candidate gives different scores.

Mitigation:

- deterministic search mode;
- fixed RNG snapshots;
- disable mutable caches;
- verify parameter restoration;
- use repeated candidate evaluation only for debugging.

### 27.7 Probe-scale collapse

Scale shrinks to near zero and updates become numerically meaningless.

Mitigation:

- minimum scale;
- group reset after long stalls;
- FP32 master weights;
- log delta-to-parameter RMS.

### 27.8 One group monopolizes search

Readout perturbations may yield easy short-term gains while routing never develops.

Mitigation:

- round-robin exploration floor;
- per-group bank quotas;
- compare validation behavior by group;
- freeze readout for early prefix experiments if needed.

---

## 28. Future extensions

### 28.1 Hierarchical Procoim

Protect at multiple levels:

```text
slot
phrase or output block
formation
example replay anchor
```

A candidate must improve the active lowest-level objective while satisfying higher-level guards.

### 28.2 Route-local perturbations

Use Gator diagnostics to perturb only routes that contribute strongly to unresolved slots.

Example:

```text
slot 6 unresolved
→ identify top contributing work streams and relays
→ perturb only their route logits
```

This remains forward-only and could reduce variance dramatically.

### 28.3 Temporary slot correction vectors

Forward search may optimize per-slot latent offsets first, then distill them into shared parameters. This separates fast local improvement from slower shared learning.

### 28.4 Region-level Procoim

For a regional Gator:

- optimize unresolved slots through one region at a time;
- protect region-level contributions;
- open orchestrator communication after local progress.

### 28.5 Forward directional momentum

Maintain exponential moving averages of successful directions per group:

```text
m_g ← β · m_g + (1 - β) · δ_g
```

Include m_g as one candidate direction in future banks.

### 28.6 Candidate covariance adaptation

Learn a low-rank distribution of successful perturbations rather than sampling isotropic noise. This resembles a small structured evolution strategy but remains Gator-group specific.

### 28.7 Hybrid optimizer

Possible research control:

```text
routing and stewardship: Procoim
node transforms:         AdamW
```

This is not pure Procoim, but it can determine which Gator subsystems benefit from forward-only credit.

### 28.8 Forward-mode automatic differentiation

A future variant could use JVPs to evaluate directional derivatives without reverse-mode backpropagation. This is conceptually separate from perturbation search and should receive a distinct mode name.

### 28.9 Parallel slot candidate ownership

Different candidate directions may target different unresolved slots in the same GPU batch. Candidate utility still uses aggregate protection, but proposal generation becomes slot-aware.

---

## 29. Codex implementation checklist

Codex should complete work in this order:

1. Read the existing training loop and task abstractions.
2. Identify the canonical generated-prefix entrypoint.
3. Add SlotLossResult and prefix slot-loss provider.
4. Add deterministic forward/search mode to Gator without altering standard training.
5. Add procoim_parameter_groups() with validation for duplicate parameters.
6. Implement configuration and episode-state dataclasses.
7. Implement exact group cloning and restoration.
8. Implement Rademacher antithetic candidate generation.
9. Implement candidate scoring, protection, and hard veto.
10. Implement slot lifecycle and completion logic.
11. Add optimizer_name="procoim" branch to training.
12. Add all forward-only and protection unit tests.
13. Run fixed-example smoke test.
14. Run two-example alternation.
15. Run fresh generated-prefix experiment.
16. Produce a report containing total forwards, validation accuracy, peak memory, and wall-clock time.

Codex must not:

- call backward in the Procoim branch;
- alter the existing AdamW path;
- add language alignment in the MVP;
- silently batch candidates before the sequential implementation is verified;
- or weaken protected-slot constraints to make a test pass.

---

## 30. Prototype acceptance criteria

The implementation is accepted as a functioning Procoim MVP when:

1. unit tests prove no reverse-mode gradient path is used;
2. candidate restoration is exact;
3. active slots can become soft-frozen and protected;
4. protected slots veto damaging updates;
5. shared parameters continue changing after some slots protect;
6. one fixed prefix example can be improved;
7. fresh generated-prefix validation rises above chance under a fixed experiment;
8. peak memory is measured against an AdamW baseline;
9. total forward evaluations are reported honestly;
10. existing standard Gator training remains unaffected.

A negative scientific result is acceptable if the implementation meets the mechanical contract and fresh validation does not improve. The purpose of the prototype is to test the optimizer, not to manufacture a success claim.

---

## 31. Architectural contract

An optimizer run should be called Progressive Continuous Improvement, or Procoim, only if all of the following are true:

1. It performs no reverse-mode backward pass.
2. It repeatedly evaluates forward perturbations on the same example or microbatch.
3. It measures loss separately for simultaneous output slots.
4. Each slot receives an episode-relative loss-reduction target.
5. Slots stop driving exploration after reaching their target.
6. Solved slots remain active in Gator's shared communication.
7. Shared parameter updates are allowed only when protected-slot gains remain within tolerance.
8. Candidate evaluations begin from an identical parameter baseline.
9. Accepted updates persist into future examples.
10. Training reports total forward work, not only episode count.

The central principle is:

> **Continuously improve the unresolved formation, protect progress already earned, and let shared Gator machinery keep learning through forward counterfactuals.**
