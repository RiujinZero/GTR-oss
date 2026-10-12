# GTR-L EvoToken Colab module

The EvoToken Colab launcher and trainer live in one module:

`gtrl_evotoken_colab.py`

The notebook cell only mounts Drive, imports that module directly from
`gtrl_evotoken_bundle.zip`, and calls `main(RUN_ARGS)`. The checked-in thin
cell is:

`Runbooks/colab_cells/gtrl_130k_evotoken_relay_drive_cell.py`

## Common sweep controls

The current four-slot run is the default. Its explicit arguments are:

```python
RUN_ARGS = [
    "--bundle",
    str(BUNDLE),
    "--output-slots",
    "4",
    "--experiment-mode",
    "relay",
    "--block-causal",
    "--interval-seconds",
    "900",
]
```

Changing only `--output-slots` derives the promotion schedule and isolated
v5 checkpoint/log names:

| output slots | promotion counts |
|---:|---|
| 2 | `1,0,1,0` |
| 4 | `1,1,1,1` |
| 8 | `2,2,2,2` |
| 10 | `3,2,3,2` |

Use a custom `--experiment-tag` whenever changing architecture or objective
settings. This prevents a shape-incompatible run from finding a checkpoint
created under the standard slot-only namespace.

## Positional-starvation ablations

The launcher can separate a weak logical tail from a weak physical execution
slot, promotion-order bias, and shared-loss competition. All new settings are
disabled by default, so the existing v5 command remains unchanged.

The default training promotion schedule is already randomized. Every row
starts with all output slots masked, samples an independent permutation, and
promotes each physical slot exactly once according to `--promotion-counts`.
There is no fixed left-to-right promotion boundary in the baseline training
sampler. The default inference path is distinct: it promotes the most
confident unresolved positions. If front positions become confident first,
`--inference-promotion-mode confidence` can reinforce that advantage even
though training promotions were balanced in expectation.

### Address-aware slot permutation

`--slot-permutation-mode rotate` applies a random cyclic logical-to-physical
assignment on every training update; `shuffle` uses a full random
permutation. A mapping printed as

```
map_physical_to_logical=[2, 3, 4, 5, 0, 1]
```

means physical output slot 0 is training logical future offset 2, physical
slot 1 is training offset 3, and so on. The runner moves each target together
with its logical absolute address. The model can therefore identify which future
offset it is predicting while that offset moves through different physical
convolution, scan, and routing locations. Shuffling labels without their
addresses would be random label noise, not a positional control.

Evaluation uses the canonical identity mapping, so `by_pos` remains the
ordinary logical left-to-right accuracy vector. If weakness moves or
equalizes after training permutation, fixed physical specialization is
implicated. If the logical tail remains weak while it moves across physical
slots, task information or credit assignment by future offset is implicated.
Permutation requires a bidirectional active output block and cannot be paired
with `--causal`.

### Flags and defaults

| flag | default | behavior |
|---|---:|---|
| `--slot-permutation-mode {none,rotate,shuffle}` | `none` | Identity mapping, random cyclic mapping per update, or random full mapping per update. Targets and logical addresses move together. |
| `--promotion-schedule-mode {random,rotating,fixed}` | `random` | Independent random training order per row, deterministic order rotated by update and row, or fixed physical left-to-right order. |
| `--inference-promotion-mode {confidence,rotating,fixed,tail_first}` | `confidence` | Confidence top-k selection, forced orders balanced across validation rows, fixed left-to-right order, or reverse tail-first order. |
| `--position-objective-coordinates {logical,physical}` | `logical` | Apply balance weights and the tail mask by future-token offset or physical tensor slot. |
| `--tail-loss-slots N` | `0` | Train only the final `N` positions on the chosen coordinate axis. Zero disables the probe. |
| `--position-balance-gamma G` | `0.0` | Exponent for inverse-running-accuracy weights. Zero disables balancing. |
| `--position-balance-decay D` | `0.99` | EMA decay for final decoded per-position training accuracy. |
| `--position-balance-epsilon E` | `0.02` | Accuracy-ratio stabilizer. |
| `--position-balance-max-ratio R` | `3.0` | Maximum weak-position weight; must be at least one. |
| `--worst-position-weight L` | `0.0` | Fraction of the objective assigned to smooth worst-position loss. |
| `--worst-position-temperature T` | `0.5` | Smooth-maximum temperature; lower is closer to a hard maximum. |
| `--position-gradient-every-evals N` | `0` | Run positional gradient diagnostics every `N` validation events. Zero disables them. |
| `--position-gradient-max-slots N` | `8` | Cap diagnostic positions; longer blocks are sampled evenly from front to tail. |
| `--position-gradient-coordinates {logical,physical}` | `logical` | Report gradient positions by logical future offset or physical execution slot. |

Both logical and physical accuracy EMAs are tracked and printed. When
balancing is enabled, the strongest EMA position has weight 1 and weaker
positions receive up to `--position-balance-max-ratio`, using

```
((best_accuracy + epsilon) / (position_accuracy + epsilon)) ** gamma
```

Logical weights follow the address-aware target mapping into physical tensor
slots. `--tail-loss-slots 2` zeros direct loss on every position except the
last two logical positions by default; set
`--position-objective-coordinates physical` to probe the final two physical
slots instead. The worst-position objective mixes this weighted mean with a
temperature-normalized smooth maximum using the same position weights. Start
with each control separately so its effect remains attributable.

### Reading the logs

Every evaluation prints a `position=` line containing the current
physical-to-logical map, masked training CE, applied weights, and accuracy EMA
in both coordinate systems. `train_promote_logical` and
`train_promote_physical` show the current batch's mean one-based scheduled
promotion stage on both axes. `infer_promote_stage` is the corresponding mean
for canonical validation positions. Under confidence inference, a larger tail
value shows that the model is delaying that position; forced schedules make
the allocation explicit.

With `--position-gradient-every-evals N`, the worker additionally prints one
`pos_grad` line for the non-relay `shared` parameters and one for every relay:

- `norm` is the exact gradient norm contributed by each printed position. A
  much smaller tail norm is direct evidence of weak credit reaching that
  parameter group.
- `relative` is each norm divided by the within-line mean. Below 1 means less
  positional credit than average and above 1 means more. Interpret it within
  each group; do not compare absolute `norm` values across differently sized
  groups.
- `cos_prefix[i]` compares position `i`'s gradient with the combined gradient
  of all earlier printed positions on the selected coordinate axis. Positive
  is aligned, near zero is independent, and negative is conflicting. The
  first value and zero-norm comparisons are `nan`.

The diagnostic operates on the unweighted per-position masked trajectory CE,
so a tail-only or balanced objective does not hide the underlying path. It
does not populate ordinary parameter gradients and does not store one full
gradient vector per position. It does retain the graph and performs `2P-1`
extra backward traversals for `P` selected positions. On the 100M model, use a
sparse setting such as `--position-gradient-every-evals 10` and keep
`--position-gradient-max-slots` no larger than the positions needed for the
diagnosis.

### Compact six-slot screen

Keep architecture, vocabulary, context, seed, data, and run length fixed. Use
the baseline command below and change only one control in each arm:

```python
BASE = [
    "--bundle", str(BUNDLE),
    "--output-slots", "6",
    "--position-gradient-every-evals", "10",
    "--position-gradient-max-slots", "6",
]
```

| arm | one added flag | interpretation |
|---|---|---|
| baseline | none | Reproduce the gap under random training promotions and confidence inference. |
| physical mapping | `--slot-permutation-mode rotate` | Tests whether weakness belongs to physical execution slots or logical future offsets. |
| training promotion | `--promotion-schedule-mode rotating` | Removes random-order variance while rotating early access evenly. |
| inference promotion | `--inference-promotion-mode rotating` | Removes confidence selection from validation refinement. |
| balanced loss | `--position-balance-gamma 0.5` | Gives weak EMA positions capped extra credit. |
| worst-position loss | `--worst-position-weight 0.25` | Prevents mean loss from improving while the worst slot remains flat. |
| tail-only probe | `--tail-loss-slots 2` | Tests whether positions 4-5 learn when positions 0-3 receive no direct loss. |

Run the arms independently before combining them. If forced inference rotation
helps, `--inference-promotion-mode tail_first` is the direct tail-first probe.
If cyclic mapping changes the result, `--slot-permutation-mode shuffle` is the
stronger physical-assignment control.

For the tail-only arm to be a fine-tuning probe rather than another from-scratch
run, point it at the baseline checkpoint while giving it a fresh optimizer and
step counter:

```python
BASE + [
    "--tail-loss-slots", "2",
    "--initial-model-checkpoint", "/content/drive/MyDrive/.../baseline.pt",
]
```

With the default `--resume-source auto`, the first launch warm-starts when the
new tail namespace is empty and later launches resume that tail checkpoint.
Add `--resume-source none` only when deliberately restarting the probe from its
baseline weights. A warm start loads only model weights and verifies the
tokenizer; optimizer state, EMAs, best metrics, and completed steps start
fresh. It also requires all model-architecture, topology, causal, and
embedding-tying settings to match the source checkpoint; only
training/refinement objectives and schedules may differ silently.

Nondefault slot permutation, training promotion, inference promotion,
balanced loss, worst-position loss, tail mask, and relevant physical-coordinate
settings receive automatic descriptive checkpoint/log suffixes plus a short
signature digest. The digest prevents different numeric settings such as two
balance exponents or temperatures from sharing artifact paths. A strict
position-ablation signature is also stored in every checkpoint and checked on
resume, so even a reused custom tag cannot silently load different objective
or schedule semantics. Position-balanced checkpoints preserve both accuracy
EMAs and reject a progressed resume if that state is missing. Gradient-
diagnostic frequency, slot cap, and coordinate settings are observational only
and do not change the training namespace.

## Discovering and changing settings

Every `RunConfig` field has a kebab-case command-line flag:

```python
main(["--help"])
```

Examples:

```python
main([
    "--bundle", str(BUNDLE),
    "--output-slots", "2",
    "--total-steps", "50000",
    "--checkpoint-every", "10000",
    "--batch-size", "16",
    "--learning-rate", "0.001",
    "--interval-seconds", "900",
])
```

```python
main([
    "--bundle", str(BUNDLE),
    "--experiment-tag", "s4_r3_width20_probe",
    "--output-slots", "4",
    "--relays", "3",
    "--trajectory-steps", "3",
    "--promotion-counts", "2,1,1",
    "--width", "20",
])
```

Boolean flags have positive and negative forms, such as `--compile` and
`--no-compile`, `--monitor` and `--no-monitor`, or `--preflight` and
`--no-preflight`.

`--block-causal` makes the context match chunked generation: each block has
bidirectional refinement inside itself, may read every earlier committed
block, and cannot modify an earlier block. The block width is
`--output-slots`, so `--observed` must be divisible by it. Enabled runs get a
separate `_block` checkpoint/log namespace; `--no-block-causal` preserves the
legacy bidirectional-window behavior and old names.

Token embeddings are untied by default so existing checkpoints remain
compatible. New experiments can share the input embedding and vocabulary
classifier with `--tie-token-embeddings`; use `--no-tie-token-embeddings` for
the control arm. When no explicit experiment tag is supplied, tied runs get a
separate `_tied` checkpoint/log namespace automatically.

For example, a hardware-aligned approximately 100M-parameter tied GTR-L with
256 nodes and a 20k vocabulary uses width 1504:

```python
main([
    "--bundle", str(BUNDLE),
    "--tie-token-embeddings",
    "--block-causal",
    "--nodes", "256",
    "--width", "1504",
    "--vocab", "20000",
    "--output-slots", "8",
    "--observed", "2040",
    "--batch-size", "1",
    "--eval-batch-size", "1",
])
```

For generation, build the fixed-size prefix state once with
`model.prefill_cache(ids, block_size=K)`, call `forward_cached` (or
`EvoTokenRefiner.forward_cached`) as many times as refinement needs, then call
`append_cache(cache, accepted_ids)` exactly once. Proposal calls are read-only.
Append recomputes accepted IDs as ordinary observed tokens with full writer
gates and returns a new cache. That clean commit is one extra `K`-token relay
pass per generated block; cache storage itself is constant with context
length. Sliding eviction is not supported, so reaching `context_length`
requires a fresh prefill.

### Block KV-cache contract

`GatorLinearBlockCache` stores `next_position`, the fixed block width, and one
immutable state record per relay. Each relay record contains:

- the left convolution tail needed by the next block;
- the forward decayed-scan carry and accumulated mailbox sum;
- linear-attention key/value carries when `--linear-attention` is enabled;
- cumulative routing occupancy when `--router-occupancy` is enabled.

This is a superset of a Transformer KV cache: the key/value tensors are only
one part of the recurrent state required by GTR-L's mixer and router. No
per-token hidden states, proposal embeddings, or route history are retained,
so cache storage stays constant as committed context grows.

The lifecycle is functional rather than in-place. `prefill_cache` and
`append_cache` return detached cache objects; `forward_cached` never advances
the input cache. This allows all EvoToken refinement passes to inspect the
same committed prefix and prevents rejected proposal state from leaking into
later blocks. Only accepted discrete IDs advance `next_position`.

Cache position, alignment, relay count, tensor shapes, and context overflow
are validated before use. Absolute position construction and long-lived sums
use FP32 arithmetic under BF16, then cast back to the model dtype. With a
fixed batch shape and block width, compiled proposal calls reuse the same
full graph at later positions. Cached route diagnostics describe only the
active block and are labeled `route_statistics_scope="active_block"`; they
must not be interpreted as full-prefix routing statistics.

By default, model capacity is exactly `observed + output_slots`. Set a larger
`--context-length` to reserve repeated-generation headroom without increasing
the sampled training window; this automatically adds `_ctxN` to the run
namespace. The present smoke trainer still supervises its proposal at one
fixed absolute boundary, so reliable long generation will need a later
variable-prefix/unrolled-block curriculum. `prefill_cache` currently requires
complete `K`-token blocks; arbitrary partial prompt blocks are not yet masked
or cached.

The module exposes paths for the bundle, extracted code, local work,
TinyStories cache, Drive root, logs, checkpoints, progress archive, PID file,
and run-configuration snapshots.

## JSON configuration

Long sweeps can keep settings in JSON and override individual values from the
cell:

```python
main([
    "--config",
    "/content/drive/MyDrive/gtrl_evotoken_relay/configs/sweep.json",
    "--output-slots",
    "2",
])
```

CLI values take precedence over JSON values. The fully resolved configuration
is written atomically beside the local run and to `run_configs/` under the
Drive root. Use `--dry-run` to validate and print it without extracting code
or launching:

```python
main([
    "--bundle", str(BUNDLE),
    "--output-slots", "2",
    "--dry-run",
])
```

## Process and checkpoint behavior

The launcher:

1. preserves complete local checkpoints and logs;
2. extracts the uploaded bundle;
3. runs the focused preflight;
4. writes the resolved run configuration;
5. starts a detached worker from the same module;
6. mirrors progress and new atomic checkpoints to Drive.

The worker handles `SIGTERM` and `SIGINT` by finishing the current update and
saving an exit checkpoint. Starting a new module-managed run therefore
preserves progress even when the previous sweep arm stopped between regular
checkpoint boundaries.

`--resume-source` accepts `auto`, `local`, `drive`, or `none`. `auto` preserves
the previous behavior: prefer a matching local checkpoint, then Drive.
