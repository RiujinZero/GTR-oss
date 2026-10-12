# GTR-L continuous token evolution

Status: **experimental**

Primary implementation: `gator_linear/evotoken.py`

130k Colab module: `gtrl_evotoken_colab.py`

Thin launch cell:
`Runbooks/colab_cells/gtrl_130k_evotoken_relay_drive_cell.py`

Module guide: `Runbooks/GTRL_EVOTOKEN_COLAB_MODULE.md`

## Motivation

The current 130k no-whitespace run is not strictly token-by-token
autoregression: it predicts an eight-token future block simultaneously.
However, it is still a one-shot predictor. Every output slot starts from the
same learned query, the four relays run once, and only the terminal prediction
is supervised. A wrong early interpretation cannot be exposed as an explicit
token belief and revised.

[EvoToken-DLM](https://arxiv.org/abs/2601.07351) replaces one-shot hard
unmasking with a trajectory:

```
MASK -> mask-aware soft token -> pure soft token -> hard decode
```

The released method reruns a complete pretrained masked diffusion backbone at
every refinement step. GTR-L has a useful structural advantage over an
autoregressive decoder: all output slots exist simultaneously and are
bidirectional *inside the active block*. With `block_causal=True`, earlier
blocks are immutable while the active block can read the full committed
prefix. The causal-prior mismatch called out by the EvoToken paper therefore
does not require reducing refinement to token-causal attention.

## Block-causal cache

Training partitions the entire window into fixed `output_slots`-wide blocks.
A vectorized mixer keeps this one training pass: forward scan state continues
across blocks, reverse scan state resets at each block boundary, and global
and routing-occupancy summaries advance only at complete block boundaries.
This is the exact reference for cached generation, not an inference-time
approximation.

Every relay owns a constant-size recurrent cache containing its convolution
tail, raw forward-scan carry, accumulated mailbox sum, optional linear-
attention key/value carries, and optional routing-occupancy sum. Absolute
addresses use the configured context length, so appending a block never
changes an earlier token's address. No token hidden-state history or
context-sized route tensor is stored.

This is called the block KV cache by analogy with Transformer decoding, but
it is deliberately broader than a Transformer key/value cache. Linear
attention contributes literal key/value carries when enabled; the mixer and
router also need their own sufficient statistics to make a cached block
identical to the vectorized block-causal reference. Cache storage therefore
scales with model state and relay count, not with the number of committed
tokens.

Refinement reads the committed cache without changing it. Once a block is
accepted, `append_cache` recomputes its discrete IDs as observed slots with
full write gates and returns a new cache atomically. Proposal embeddings,
mask-aware states, and the controlled-writing ramp are never persisted as
language history. This costs one additional short commit pass per generated
block. Cache eviction is intentionally unsupported because decayed scan sums
cannot be subtracted exactly; reaching the context limit requires reprefill.

The current training runner remains a fixed-boundary smoke experiment: it
scores the active block at one absolute position. A larger model
`context_length` creates cache headroom but does not by itself train proposals
at every later position. Multi-block generation therefore needs a subsequent
variable-prefix or unrolled-block curriculum. Prefill also currently accepts
only complete blocks; partial natural-language prompts require a future
masked partial-block design rather than silently padding recurrent sums.

## Two adaptations

### Relay-integrated refinement (primary)

Treat GTR-L's four relays as the paper's four trajectory steps.

After relay `r`:

1. Apply the existing shared coalition/vocabulary readout.
2. Mask illegal token classes.
3. Retain and renormalize the top three token probabilities.
4. Promote a selected subset of output positions from mask-aware to
   pure-soft state.
5. Form the probability-weighted token embedding.
6. Inject the change in token belief into every private stream at the output
   slot before relay `r+1`.
7. Apply cross-entropy at this relay as well as every later relay.

Occupancy is not reset. It continues through the normal relay trajectory, so a
revised token belief can change later routing without discarding the route
history already accumulated.

This costs four relays total, the same relay count as the one-shot model. The
extra work is four small shared vocabulary readouts and sparse top-k embedding
updates.

This is not mathematically identical to EvoToken. Each refinement step sees
one distinct relay rather than the entire shared backbone, and latent hidden
state remains live across steps.

### Full-pass refinement (fidelity control)

Rebuild ingress and rerun all four relays after every soft-token update. This
matches the paper's model/state boundary much more closely and detaches the
token IDs/probabilities between passes.

Four trajectory steps cost 16 relay evaluations. Its purpose is to establish
whether revisiting earlier machinery matters enough to justify the compute.

## Soft-token equations

For top-`K` legal candidates and renormalized probabilities:

```
distribution = sum_c probability[c] * token_embedding[token[c]]
```

An unresolved position receives:

```
alpha * mask_embedding + (1 - alpha) * distribution
```

A promoted position receives the pure `distribution`. Training samples
`alpha` independently from `U(0.5, 1.0)` per example and stage. Inference
starts at `alpha=0.7`.

Candidate IDs and probabilities are detached before feedback. The live
embedding table performs the next lookup, so it still receives ordinary
input-path gradient, but gradients cannot pass through top-k selection or a
previous relay's probability distribution.

## Masked-only CTS from scratch

The denoising curriculum produced a strong easy-token shortcut: copied visible
positions reduced total loss while all-mask outputs converged on punctuation
and a few frequent words. The v4 runner therefore trains continuous trajectory
supervision from step zero.

The v5 runner makes the output-block width a single sweep setting and
derives the rest of the experiment from it:

- four refinement stages remain fixed so relay depth is controlled;
- every output slot starts masked and is promoted exactly once;
- promotions are balanced across stages, including zero-promotion stages when
  there are fewer slots than relays;
- four slots use `(1, 1, 1, 1)`, two use `(1, 0, 1, 0)`, and larger or
  non-divisible widths are distributed as evenly as possible;
- top-3 soft beliefs and detached trajectory feedback remain unchanged.

The model context length, promotion schedule, checkpoint namespace, log
name, collapse threshold, sample denominators, and saved configuration all
derive from the module's `--output-slots` argument. Checkpoints are
width-specific and cannot silently resume into a different slot shape.

The notebook no longer embeds the launcher or worker. It imports
`gtrl_evotoken_colab.py` directly from the uploaded bundle and calls its
single `main()` entry point. Every path, model, trajectory, optimizer,
evaluation, checkpoint, compilation, and monitoring setting is exposed as a
CLI flag and JSON `RunConfig` field. Resolved run configurations are mirrored
to Drive, and module-managed workers save an exit checkpoint when stopped.

Cross-entropy is computed at every stage on every output position. The
general schedule API still supports partially visible trajectories and keeps
their loss weight at zero, but this width sweep starts all slots masked so the
comparison cannot receive loss credit for copying ground truth.

The run starts at `1e-3` and cosine-decays toward `1e-4` over one million
updates. Width-specific `cts_masked_sN_v5` checkpoint namespaces keep the
four-slot, two-slot, and future wider runs isolated from one another and from
older v4 state. BF16 autocast, full-graph compilation, global clipping, and
the empty-node gradient safeguard remain enabled.

Validation also reports the decoded modal token, its frequency in the target
batch, accuracy lift over predicting that token everywhere, and the number of
positions failing to beat that per-position baseline. After a 10,000-step
warmup, five consecutive evaluations with at least 40% modal predictions,
at most two percentage points of unigram lift, and at least four weak positions
emit a prominent collapse warning.

## Positional-starvation controls

The six-slot results motivate a distinction between ordinary capacity
starvation and **positional starvation**: added width or nodes can improve the
already-strong front while the tail receives little additional usable credit.
The runner now separates four possible causes: fixed physical tensor slots,
promotion order, competition in the shared loss, and weak or conflicting
gradient paths.

One important baseline detail is easy to miss: CTS training promotions were
already randomized before these controls were added. With the default
`--promotion-schedule-mode random`, every batch row samples an independent
permutation, every slot starts masked, and every physical slot is promoted
exactly once. The default is therefore not a fixed left-to-right training
schedule. Inference is different. The default
`--inference-promotion-mode confidence` promotes the most confident unresolved
positions, which can create a self-reinforcing bias if the front becomes
confident first. Forced inference schedules isolate that decode-time effect
from the training schedule.

### Address-aware logical-to-physical permutation

`--slot-permutation-mode rotate` or `shuffle` changes which physical tensor
slot carries each logical future-token position during training. The mapping
is shared by the batch and is recorded as
`map_physical_to_logical=[...]`. Crucially, the runner permutes both the target
and its logical absolute address. If physical slot `j` carries logical
position `p[j]`, it receives the address for `p[j]` as well as that target.
This makes the assignment identifiable to the model. Permuting labels alone
would instead create unannounced random label noise and would not be a valid
architectural control.

Training may therefore move a logical tail token through different physical
convolution, scan, and routing locations while preserving its semantic future
offset. Validation always returns to the identity mapping and reports the
ordinary logical left-to-right sequence. If the same physical locations stay
weak under permutation, the defect is physical or scheduling-related. If the
logical tail stays weak as it moves, the information available for later
future offsets is the likelier limitation. Slot permutation requires a
bidirectional active output block; it is rejected with fully causal
token-by-token attention.

### Launcher controls

All controls are opt-in. Their defaults reproduce the original v5 objective.

| flag | default | interpretation |
|---|---:|---|
| `--slot-permutation-mode {none,rotate,shuffle}` | `none` | Keep the identity physical/logical mapping, apply a random cyclic rotation per update, or apply a random full permutation per update. Addresses and targets move together. |
| `--promotion-schedule-mode {random,rotating,fixed}` | `random` | Choose each row's training promotion order independently at random, rotate a deterministic order across rows and updates, or always promote physical slots from left to right. `fixed` is primarily an adverse control. |
| `--inference-promotion-mode {confidence,rotating,fixed,tail_first}` | `confidence` | Promote unresolved slots by model confidence, balance forced promotion order across validation rows, force left-to-right order, or force the tail first. This affects evaluation/refinement selection rather than the training promotion sampler. |
| `--position-objective-coordinates {logical,physical}` | `logical` | Define balancing and tail masks by future-token offset or by physical tensor slot. |
| `--tail-loss-slots N` | `0` | Score only the final `N` positions on the selected coordinate axis. Zero disables the tail-only probe. |
| `--position-balance-gamma G` | `0.0` | Enable inverse-running-accuracy weighting. Larger `G` increases the preference for weak positions; zero disables it. |
| `--position-balance-decay D` | `0.99` | EMA decay for per-position final decoded training accuracy. |
| `--position-balance-epsilon E` | `0.02` | Stabilizer in the inverse-accuracy ratio. |
| `--position-balance-max-ratio R` | `3.0` | Cap a weak position's weight relative to the current strongest position. |
| `--worst-position-weight L` | `0.0` | Mix fraction for smooth worst-position loss; zero preserves the weighted mean objective. |
| `--worst-position-temperature T` | `0.5` | Log-sum-exp temperature; lower values approach a hard maximum over active positions. |
| `--position-gradient-every-evals N` | `0` | Run exact positional gradient diagnostics every `N` validation events. Zero disables them. |
| `--position-gradient-max-slots N` | `8` | Maximum number of evenly spaced positions included in a diagnostic event. |
| `--position-gradient-coordinates {logical,physical}` | `logical` | Order diagnostic losses by future-token offset or physical tensor slot. |

Position balancing tracks both logical and physical final-decoded accuracy
EMAs even when weighting is disabled. Once enabled, a position receives

```
min(max_ratio, ((best_accuracy + epsilon) /
                (position_accuracy + epsilon)) ** gamma)
```

with a minimum weight of one. Logical weights follow their targets through a
physical permutation. `tail_loss_slots` then zeros every position outside the
selected logical or physical tail. The worst-position term mixes the resulting
weighted mean with a position-weight-normalized log-sum-exp over active
per-position losses. Balance weights therefore remain effective even if the
worst-position mix is set to one:

```
loss = (1 - worst_weight) * weighted_mean + worst_weight * smooth_worst
```

Each evaluation also reports the current batch's mean one-based training
promotion stage in logical and physical coordinates plus the corresponding
canonical validation promotion stage. These fields make it possible to check
the scheduler allocation directly rather than infer it from accuracy alone.

These knobs should first be tested one at a time. Combining balancing,
worst-position pressure, and a tail-only mask immediately would make a gain
impossible to attribute.

### Gradient interpretation and cost

When enabled, each `pos_grad` line reports the unweighted masked trajectory CE
and exact parameter-gradient statistics for `shared` and for every relay. The
reported positions are evenly sampled when the slot count exceeds
`--position-gradient-max-slots`.

- `norm` is the absolute gradient norm contributed by that position to the
  named parameter group. A persistently tiny tail norm indicates weak credit
  reaching that group.
- `relative` divides each norm by the mean norm across the positions printed
  on the same line. Values below one are relatively starved; values above one
  dominate that group. Compare relative values within a line, not absolute
  magnitudes across differently sized groups.
- `cos_prefix[i]` is the cosine between position `i`'s gradient and the sum of
  all earlier selected-position gradients on that coordinate axis. Positive
  values are aligned, values near zero are independent, and negative values
  indicate interference. The first entry is `nan` because it has no prefix;
  zero-norm comparisons are also `nan`.

The diagnostic uses exact vector-Jacobian products without filling ordinary
`.grad` buffers or retaining one full parameter vector per position. It still
costs `2P-1` extra backward traversals for `P` selected positions and retains
the training graph until they finish. On a 100M model it should be sampled
infrequently, for example every ten validation events, rather than enabled at
every evaluation.

### Six-slot diagnostic matrix

Hold seed, data, vocabulary, context, model shape, and run length fixed. Add
`--position-gradient-every-evals 10 --position-gradient-max-slots 6` to the
baseline and the most informative arms, then change only the listed flag:

| arm | extra flag | question answered |
|---|---|---|
| baseline | none | Reproduce the six-slot front/tail gap with randomized training promotions and confidence inference. |
| physical permutation | `--slot-permutation-mode rotate` | Does weakness stay with a physical slot or follow the logical future offset? |
| deterministic promotion rotation | `--promotion-schedule-mode rotating` | Does removing random promotion-order variance change allocation? |
| forced inference rotation | `--inference-promotion-mode rotating` | Is confidence-based promotion amplifying an already-strong front? |
| balanced loss | `--position-balance-gamma 0.5` | Can capped inverse-accuracy credit lift weak positions without masking the front? |
| worst-position mixture | `--worst-position-weight 0.25` | Does preventing a mean-loss win from hiding the worst slot improve the tail? |
| tail probe | `--tail-loss-slots 2` | Can positions 4-5 learn rapidly when positions 0-3 receive no direct loss? |

`--inference-promotion-mode tail_first` is a useful short follow-up if forced
rotation helps. `shuffle` is a stronger permutation follow-up if cyclic
rotation changes the physical/logical diagnosis. Only after the single-factor
screen should the best allocation and objective controls be combined.

Nondefault permutation, promotion, balancing, worst-position, tail, and
coordinate modes automatically add descriptive suffixes plus a short signature
digest to checkpoint and log names. The digest keeps distinct numeric
ablations from colliding. Checkpoints also store a strict position-ablation
signature plus both accuracy EMAs. Resume rejects a signature mismatch even
when a custom experiment tag points at the wrong file, and a position-balanced
run cannot resume from a progressed checkpoint that lacks its EMA state.
Gradient-only diagnostic flags do not alter training semantics and therefore
do not create a new checkpoint namespace.

The tail-only arm is most diagnostic when it starts from the same trained
baseline. `--initial-model-checkpoint PATH` performs that weights-only warm
start when the new objective namespace has no checkpoint: it verifies the
tokenizer and loads the model, but starts with a fresh optimizer, zero completed
steps, fresh accuracy EMAs, and a new checkpoint namespace. Model shape,
topology, causal mode, and embedding tying must match the source checkpoint;
the objective and schedule controls may differ. The default `resume_source=auto`
then resumes an interrupted probe from its own checkpoint. Add
`--resume-source none` only to discard that progress and restart from the
baseline weights deliberately.

## Experiment arms

The Colab module exposes `--experiment-mode`:

| mode | loss | feedback | relay evaluations/update | purpose |
|---|---|---|---:|---|
| `relay` | every relay | soft belief delta | 4 | primary GTR-L adaptation |
| `relay_no_feedback` | every relay | none | 4 | isolates deep supervision |
| `full_pass` | every full pass | soft embeddings | 16 | paper-fidelity control |

The existing no-whitespace runner remains the one-shot baseline.

## Measurements

Every validation interval reports:

- historical-best refined accuracy;
- one-pass all-mask accuracy;
- accuracy at every relay/refinement stage;
- exact full-block match for the configured slot count;
- per-position accuracy;
- revision rate between stages;
- fraction of revisions that correct an error;
- top-k retained probability mass;
- invalid-token probability mass;
- modal output share;
- masked training loss versus diagnostic-only visible-token loss;
- all-mask ground-truth top-3 hit rate for one-pass and every relay;
- current learning rate and the per-stage promotion counts;
- global pre-clip norm, actual clip scale, and shell/attention/routing/per-relay
  gradient norms.

A convincing positive result requires more than a higher terminal raw
accuracy:

1. `relay` must beat `relay_no_feedback`, or the gain is only deep
   supervision.
2. Refined accuracy should exceed the matched one-pass all-mask probe.
3. Later stages should correct errors more often than they introduce them.
4. Initially masked loss must improve; visible-position loss is not optimized.
5. Modal share, unigram lift, weak-position count, and invalid mass must stay
   controlled.
6. The full-pass arm determines whether the efficient relay approximation
   leaves material quality on the table.

## Empty-node gradient safety

Hard routing leaves many node mailboxes exactly empty. Linear attention must
not normalize those empty numerators by a tiny epsilon: the forward result is
zero, but the surrogate derivative scales as the reciprocal epsilon and can
explode through straight-through routing. GTR-L now masks the empty branch to
an exact zero contribution and zero content gradient. Occupied low-mass nodes
use a `1e-3` denominator floor, bounding the remaining normalization gain.
Global norm clipping at 1.0 remains the final safety net rather than the
primary mechanism controlling this branch.

## Compatibility

The legacy `GatorLinearPrefixModel.forward()` remains unchanged when
`output_embeddings` is omitted. Relay refinement adds no learned parameters.
The same token table, slot types, addresses, streams, route graph, controlled
writing schedule, linear attention, hard routing, fusion head, and vocabulary
head are reused.

The no-whitespace tokenizer is preserved. Soft evolution is restricted by
tokenizer semantics: non-whitespace learned pieces and non-whitespace byte
fallback IDs are legal, while structural tokens, MASK, BLANK, and whitespace
bytes remain impossible. Validation data never changes that legal vocabulary.
