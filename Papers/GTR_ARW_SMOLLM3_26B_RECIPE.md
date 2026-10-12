# GTR-ARW SmolLM3 26B-scaled pretraining

`gator_language_train.py --training-recipe smollm3-26b` selects the dedicated
original GTR-ARW training loop. The optimizer schedule, three-stage source
weights, tokenizer, and token budget are scaled from the SmolLM3 recipe.

## Training recipe

- 4,096 total tokens per formation by default: observed GTR context plus
  output slots; `--smollm3-sequence-length` may change the total while scaling
  update, warmup, decay, corpus-stage, and token accounting boundaries to keep
  the 26B-token recipe equivalent
- effective batch fixed at 32 formations; physical batch and accumulation may
  vary as long as their product is 32, such as batch 1 / accumulation 32 or
  batch 2 / accumulation 16
- BF16 autocast with FP32 optimizer state and gradients
- AdamW `(beta1=0.9, beta2=0.95, eps=1e-8)`
- weight decay `0.1`, excluding the token embedding
- gradient clipping at norm `1.0`
- peak learning rate `2e-4` by default with 500-update linear warmup; override
  with `--smollm3-peak-lr`, for example `--smollm3-peak-lr 8e-5`
- stage transitions at updates 144,992 and 176,428
- linear decay to zero at update 198,365
- `meta-llama/Llama-3.2-1B` tokenizer by default

`--steps` is an absolute optimizer-update target. Checkpoints default to every
5,000 updates and are written atomically. The append-only timed log is
`CHECKPOINT_DIR/train.log`.

The training loop fails closed on numerical errors. It checks every microstep
loss before backward, checks the accumulated gradient norm before the optimizer
step, and verifies model parameters plus AdamW state before loading or writing a
checkpoint. A failure reports the next update, exact microstep, and selected
data source or accumulation source set. The pending gradients are cleared and
the optimizer step and checkpoint are skipped; training never intentionally
continues from or saves a non-finite state.

`optimized_gpu` now activates the same packed-auto controller in this dedicated
training loop as in standard language training. Every configured observation
interval, the final microbatch runs one eager route-density probe. Once an
entire relay suffix falls below the configured break-even threshold, the
controller latches that suffix to persistent packed stewardship execution and
stores the decision in checkpoints. Dense calls remain compiled; packed calls
are intentionally eager because their active-row shapes are dynamic.

Supplying only `--batch-size 2` automatically selects accumulation 16. An
explicit pair must multiply to 32.

## Factorized tied vocabulary

The SmolLM3 recipe defaults to a 256-wide lexical table even when Gator's
internal language width is much larger. Input tokens follow:

```text
token id -> tied [vocab, lexical_width] table -> lexical projection -> Gator
```

The output path reverses the same learned projection and then uses the same
token table for vocabulary logits. There is no separate output-head matrix.
This keeps all 128,256 Llama-tokenizer entries while preventing the vocabulary
from consuming almost the entire model.

For vocabulary size `V`, lexical width `E`, and language width `H`, the lexical
path contains approximately:

```text
V * E + E * H + V
```

parameters: one tied token table, one tied factor projection, and the output
bias. With `V=128256`, `E=256`, and `H=2304`, that is 33,551,616 parameters.

A tensor-friendly GTR-ARW configuration near 100M parameters is:

```bash
--cells 11 \
--nodes 4 \
--d-model 1152 \
--embedding-dim 2304 \
--lexical-embedding-dim 256 \
--relays 4 \
--neighbors 4 \
--node-topology asymmetric \
--cell-input-mode feature_gated \
--cell-output-mode grouped_concat \
--cell-transport-mode stewardship \
--receiver-memory-backend linear_scan
```

This model has 100,836,055 trainable parameters: 33,551,616 in the lexical
path and 67,284,439 in the Gator core. By comparison, the old
`d_model=128, embedding_dim=768, cells=10` configuration becomes a 34,603,054
parameter model after factorization; before factorization, 98.6M of its 100.1M
parameters were the vocabulary table and bias.

Use `--lexical-embedding-dim 0` to retain the legacy unfactorized tied table.
Changing lexical width changes checkpoint tensor shapes, so start a new
checkpoint directory. To resume an older unfactorized checkpoint with the new
launcher, explicitly pass `--lexical-embedding-dim 0`.

## Data path: Nanotron-compatible shards

Nanotron does not load the raw heterogeneous Hugging Face rows in its training
loop. DataTrove first tokenizes every logical source into `.ds` files containing
fixed-width token IDs; Nanotron then reads sequential windows from those files.
This is the preferred and closest-reproduction mode:

```bash
python gator_language_train.py \
  --training-recipe smollm3-26b \
  --smollm3-tokenized-data-root /content/smollm3_ds \
  --checkpoint-dir /content/drive/MyDrive/GTR/smollm3_26b
```

The root must contain one folder per logical recipe source, such as
`fineweb-edu/`, `stack-v2-Python/`, and `pull-requests/`. Each folder contains
one or more `.ds` files and may contain DataTrove `.metadata` files. Only one
shard file handle is open per selected source; token files are never loaded in
full. A checkpoint stores file index and token offset, not shard contents.

## Data path: direct Hugging Face streaming

When `--smollm3-tokenized-data-root` is omitted, the trainer uses an out-of-core
adapter for public Hugging Face datasets:

1. A source remains unopened until the deterministic mixture selects it.
2. At most one bounded token chunk per active source is held in RAM.
3. When the RAM chunk reaches `--smollm3-prefetch-tokens`, a shared worker
   prepares the next chunk as an `.npy` file on fast local disk.
4. The next chunk stays on disk until the current RAM chunk is exhausted.
5. `--smollm3-prefetch-workers` caps tokenization across all sources globally.
6. Only `--smollm3-max-open-streams` HF/PyArrow readers remain resident. Least-
   recently-used readers are closed after saving their exact native cursor.
7. Tokenizer truncation is applied while encoding, so a very large row cannot
   first materialize an unbounded Python token list.
8. `--smollm3-max-host-ram-gb` is a process-RSS guard. The loader evicts all
   resumable readers under pressure and exits clearly if RSS remains over the
   limit, rather than waiting for an opaque operating-system OOM kill.

Recommended Colab settings:

```bash
--smollm3-stream-cache-dir /content/smollm3_stream_cache \
--smollm3-token-chunk-tokens 65536 \
--smollm3-prefetch-tokens 8192 \
--smollm3-prefetch-workers 2 \
--smollm3-max-document-chars 262144 \
--smollm3-max-open-streams 2 \
--smollm3-max-host-ram-gb 20 \
--smollm3-memory-log-every-seconds 60
```

The cache belongs on the Colab VM or another fast local disk, not Google Drive.
Each run uses an isolated subdirectory and removes it on clean shutdown. Corpus
chunks are not copied to Drive.

Startup diagnostics are enabled without an extra flag. For every source's
first raw-HF chunk, the trainer reports dataset-open start/finish, first-row
columns, first-document tokenization, 8,192-token chunk milestones, and chunk
completion. During optimizer update zero it also reports every batch acquisition
and accumulation microstep, including separate first-forward/compile and
first-backward timings. Detailed startup messages stop after those bounded
events; ordinary long-run logging remains controlled by `--log-every`.

Direct-HF mode is operationally similar but is not byte-for-byte equivalent to
the private materialized SmolLM3 shards. In particular, the public
`bigcode/the-stack-v2` dataset exposes Software Heritage identifiers rather
than source contents. Direct mode therefore maps Stack logical sources to the
content-bearing gated `bigcode/starcoderdata` language folders. Its pinned
revision has no Swift directory, so Swift alone uses the content-bearing
`bigcode/the-stack-dedup/data/swift` shards. StarCoder2 extras use
`bigcode/starcoder2data-extras`. Its current public revision no
longer has the historical `pull_requests` and `jupyter_scripts` configurations,
so direct mode maps those small logical slots to the closest available
`issues` and `kaggle` content respectively. The aggregate StackExchange slot
uses DOLMino's public `stackexchange` configuration; the newer 2025 mirror
cannot represent that slot with one stream because it requires a separate
configuration for every site.

Final-stage public mappings use `nvidia/OpenMathInstruct-2`'s train split,
`nvidia/OpenMathReasoning`'s `default` configuration and `cot` split, and
`nvidia/OpenCodeReasoning`'s `default` configuration and `split_0` split. The
two TinyGSM-MIND production views use the combined public
`ostapeno/tinygsm-mind` train split. The original logical datasets can still
be used exactly through `.ds` mode. Public MegaMath rows are metadata
pointers, so direct mode uses FineMath content for those small mixture slots.
Use `.ds` mode when the original materialized sources are available.

## Mixture and stage continuity

Source selection matches Nanotron/Megatron's deterministic largest-mixture-
deficit schedule. It does not call `random.choices`. Every stage stores its
sample position and per-source counts. When a logical source appears in a later
stage, its dataset cursor is carried forward rather than restarting at row zero.

## Exact resume

Resume with the same model, tokenizer, and data settings:

```bash
--resume-checkpoint /content/drive/MyDrive/GTR/smollm3_26b/latest.pt
```

The checkpoint restores model weights, AdamW moments, update and accumulation
position, CPU/CUDA RNG, stage blend counts, and every source cursor. Direct-HF
mode also stores only the bounded current/document/next token remainders needed
to resume at the exact application-level token. Dataset rows and the full local
cache are never embedded in the checkpoint.

Changing token chunk size, prefetch threshold, document cap, source identity,
or `.ds` shard inventory invalidates exact data resume and is rejected.


Physical batch size and gradient accumulation may change at resume as long as
their product remains the recipe's 32-formation effective batch. For example,
a checkpoint written with batch 4 / accumulation 8 may resume with:

```bash
--batch-size 1 --gradient-accumulation-steps 32
```

Checkpoints are written only after complete optimizer updates. On resume, the
physical microstep counter is remapped to the new accumulation width while
model, optimizer, RNG, token progress, and exact dataset cursors are retained.
Changing to an effective batch other than 32 remains a hard error.


## Nonfinite accumulation recovery

An isolated nonfinite loss or gradient discards the complete accumulation
window without advancing the optimizer update, LR schedule, or token counter.
The default permits three consecutive recovered windows:

```bash
--smollm3-max-consecutive-nonfinite-windows 3
```

Every recovery is printed and written immediately to `train.log`, while
periodic lines retain the cumulative `nonfinite_windows=N` count. Gradient
failures report each affected parameter's name, shape, dtype, NaN/Inf counts,
and largest finite magnitude. If every element is finite, the report identifies
the largest gradient and marks aggregate norm overflow as the likely cause.
A successful optimizer update resets the consecutive counter. The next failure
after the configured consecutive recovery budget is fatal. Set the option to
zero for the original fail-fast behavior.

Dense and packed stewardship transport, grouped-concat readout, and receiver
memory share bounded-mass normalization. Empty destinations/groups remain exactly
zero. Transport and output groups carrying at least `1e-3` mass retain the
historical weighted mean; receiver memory uses a lower `1e-6` floor because its
denominator also includes ordinary per-node occupancy. Smaller positive paths are
attenuated linearly while keeping a bounded recovery gradient. Both floors sit
below their normal live traffic, including the configured one-percent scout budget.
This covers forced grouped-output routes whose soft probability reaches the BF16
underflow boundary: normalizing those microscopic masses literally creates a
`1 / mass` backward gain and poisons entire active cell slices even though the
forward pass remains finite.

### BF16 private-routing incident signature

The production failure appeared only after routing had sharpened. Forward loss
remained finite, but backward repeatedly produced NaNs in the same five of eight
cell slices before eventually spreading to all cells. Five live cells are the
expected signature when the dominant grouped-output route retains four cells and
the scheduler forces one microscopic steward to keep the other output group alive.

A local BF16 reproduction used a forced route mass of approximately `5.5e-40`.
Before bounded normalization, both grouped readout and `linear_scan` receiver
memory produced NaN/Inf gradients. After the fix, the route-logit gradient was
finite with maximum magnitude `10.44`, receiver gradients were finite, and the
94-test GTR-ARW/SmolLM3 regression selection passed. The resumed production run
then crossed the complete former update-1896-to-2004 failure cluster and continued
for more than 200 updates without another nonfinite window.

The discard/retry mechanism remains a guardrail and diagnostic, not the primary
fix. Repeated `nonfinite_window_discarded` events indicate another numerical path
still needs investigation; increasing the recovery budget should not be treated
as stabilization.


## Block-causal training and recurrent block cache

Original GTR-ARW now has an opt-in block-causal mode that is compatible with
parallel chunk generation. Committed blocks are causal: a later block cannot
alter an earlier block. Positions inside the active block remain bidirectional,
so Gator can still refine all output slots in parallel.

Enable it from the initial training run with:

```bash
--receiver-memory-backend linear_scan \
--block-causal \
--block-size 32 \
--position-norm-length 4096
```

Both original broadcast transport and dense stewardship transport are cache-aware:

```bash
--cell-transport-mode broadcast_mix
# or
--cell-transport-mode stewardship --stewardship-execution dense
```

`broadcast_mix` retains full cell participation and neighbor coupling without
ownership, handoff, nucleus pruning, or scout policy. It is the appropriate
block-causal setting for a behaviorally naked original GTR-ARW.

CUDA-graph compile modes require an explicit iteration boundary before every
accumulated micro-batch. The trainers call
`torch.compiler.cudagraph_mark_step_begin()` outside the compiled model. GTR-ARW
also evaluates and clones its tied token/lexical ingress outside graph capture;
otherwise the next replay can overwrite an activation still referenced by the
gradient-addition path from the transposed readout use of the same weights.
`max-autotune-no-cudagraphs` leaves this storage break disabled.
Compile GTR-ARW through `gator.compile_module`; a raw `torch.compile(model, ...)`
call bypasses the model safeguard.

Before promoting `max-autotune` on a target GPU, run:

```bash
python run_cuda_unit_gate.py tests.test_gator_arw_cudagraph
```

The gate performs three BF16 accumulated backwards, compares every cumulative
compiled gradient with an eager reference after every micro-batch, and verifies
that Inductor actually records and replays a CUDA-graph backward region. Keep
`max-autotune-no-cudagraphs` as the production mode until this target-GPU gate
passes.

`--block-size` defaults to `--response-tokens`. The total training formation
must be block-aligned. The fixed position reference must cover the largest
context planned for the checkpoint. For example, a 2,048-token first stage
that may later grow to 4,096 must use `--position-norm-length 4096` from its
first update; otherwise addresses learned at 2,048 would be rescaled later.

The cache is recurrent rather than a Transformer key/value tensor. Each relay
stores its linear receiver carry, cumulative node-traffic numerator, and write
mass, plus one scalar absolute position. Broadcast transport records full-mass
cell traffic; stewardship records ownership-weighted traffic. Cache storage is
constant in the committed context length. `prefill_cache()` commits observed
blocks, `forward_cached()` proposes one parallel output block without mutating
that cache, and `append_cache()` recomputes accepted token IDs as observed
context before advancing the immutable cache. Masked/empty observed slots have
zero write mass and therefore do not enter the receiver or traffic carries.

Packed stewardship is not yet cache-aware, so stewardship block-causal runs
require dense execution. Broadcast block-causal calls support BF16 and
compilation without stewardship execution settings.

### Growing 2,048 training to 4,096

Resume the same model at a strictly larger formation with:

```bash
--smollm3-sequence-length 4096 \
--smollm3-allow-context-growth \
--resume-checkpoint /path/to/latest.pt \
--block-causal \
--block-size 32 \
--position-norm-length 4096
```

The checkpoint stores `tokens_seen` and a token-equivalent `recipe_update`.
The physical AdamW update counter remains monotonic, while warmup, corpus stage,
decay, and the remaining run budget continue from the equivalent token point at
the new sequence length. Source blend counts and exact per-source token cursors
are retained; the next sample simply consumes a wider window from that cursor.
Context shrink is rejected, and growth is rejected unless explicitly enabled
and covered by the checkpoint's fixed position reference.

Without context growth, `--steps` remains identical to the physical optimizer
update target. During a growth run it denotes the token-equivalent recipe
target; logs show both `update` (physical) and `recipe_update` (schedule).
