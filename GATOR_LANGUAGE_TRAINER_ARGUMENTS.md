# Gator Language Trainer Argument Reference

This is the operator's manual for `gator_language_train.py`. It documents every
current command-line argument, its default, what it changes, and the direction
to move it during an experiment.

The trainer forms every response slot simultaneously. It can run either Gator
or a non-autoregressive Transformer through the same tokenizer, corpora,
formation losses, evaluation, optimizer, and data pipeline.

## Reading defaults correctly

Defaults in this document are the raw parser defaults. An
`--execution-preset` may replace several execution settings afterward.
Explicit low-level arguments always override the selected preset.

For example, this remains a valid diagnostic override:

```bash
--execution-preset optimized_gpu_safe --compile
```

uses all structural settings from `optimized_gpu_safe`, then explicitly turns
compilation back on. It is not the supported long-run Arrow fallback: compiled
Blackwell runs have eventually produced non-finite gradients and compilation
is currently deferred pending Gator-Path.

Use the startup record as the authoritative resolved configuration for a run.
It prints model shape, routing policy, execution plan, precision, compilation,
data preparation, and tokenizer settings.

## Minimal launch

Only the primary corpus is required:

```bash
python -u gator_language_train.py --data smoltalk_everyday.jsonl
```

The current Gator-S/Arrow language experiments use a more explicit shape:

```bash
python -u gator_language_train.py \
  --data smoltalk_everyday.jsonl \
  --vocab-size 2048 --context-tokens 64 \
  --response-tokens 40 --reference-tokens 32 \
  --embedding-dim 32 --d-model 8 --cells 62 \
  --lexical-mode communal \
  --cell-input-mode feature_gated \
  --cell-output-mode grouped_concat \
  --cell-transport-mode stewardship \
  --nodes 4 --relays 4 --neighbors 4 \
  --node-topology hybrid_hemispheric \
  --formation-objective monotonic \
  --alignment-warmup-fraction 0.05 \
  --alignment-max-shift 8 \
  --sequence-quality-penalty 0.01 \
  --execution-preset optimized_gpu_safe
```

## Run control, evaluation, and checkpoints

| Argument | Default | Effect |
|---|---:|---|
| `--checkpoint-dir PATH` | `experimental_checkpoints/language` | Destination for periodic/best/final checkpoints and logs produced by the trainer. |
| `--resume-checkpoint PATH` | none | Loads model, tokenizer, completed-step metadata, and compatible run state. `--steps` becomes the number of additional steps. |
| `--resume-optimizer` | off | Restores optimizer moments. The scheduler still starts a new phase, and `--learning-rate` is the new phase's initial rate. |
| `--steps N` | `20000` | Optimizer steps in this invocation. Warmup fractions and cosine scheduling use this phase length. |
| `--batch-size N` | `32` | Training examples per optimizer step. Higher values increase memory and reduce gradient noise. |
| `--eval-batch-size N` | `64` | Validation examples evaluated at once. It affects evaluation memory and speed, not learning. |
| `--eval-examples N` | `0` | Validation examples per report. `0` evaluates the complete validation set. |
| `--log-every N` | `1000` | Steps between evaluation, route diagnostics, samples, and checkpoint consideration. Lower values provide visibility but cost runtime. |
| `--logging-slim` | off | Reduces console output to a compact run header containing the validation count/fingerprint, selected loss/PPL/accuracy, semantic top-1/top-5, formation coverage, whitespace omission, essential routing metrics, compact pipeline timing/queue health, and `context`/`target`/`formed`. Evaluation and checkpoint contents are unchanged. |
| `--seed N` | `0` | Controls model initialization, split/sampling order, displayed samples, and other deterministic experiment choices. |
| `--validation-fraction X` | `0.1` | Fraction of the primary corpus reserved for validation. |
| `--repeat-training-examples N` | `0` | Memorization diagnostic. Restricts training to one deterministic SmolTalk subset and repeats it. `0` disables it. Streamed corpora are forbidden in this mode. |

Best checkpoints matter. Language runs have repeatedly reached a validation
minimum and then regressed while training loss continued falling. Compare best
checkpoints, not merely final checkpoints.

### Slim console output

Add:

```bash
--logging-slim
```

Each report becomes approximately:

```text
step=2000 steps_per_second=13.56 loss=3.1024 ppl=22.25 acc=0.284 exact=0.000
routing support=[62.00,31.20,22.10,17.40,14.80] effective=[62.00,27.10,18.60,13.20,11.40] beam=[2.21,2.08,1.94,1.82] stay=[0.20,0.18,0.17,0.19] dead=[0.00,0.00,0.02,0.05] packed=[0,0,0,0]
context='...'
target='...'
formed='...'
```

For `viterbi` and `monotonic`, slim `loss` is the selected aligned objective,
`ppl=exp(loss)`, and `acc` is aligned required-token accuracy. For rigid
objectives, `acc` and `exact` are the rigid slot metrics. Slim mode suppresses
pipeline internals, filter dictionaries, forensic alignment components,
stream counters, sample IDs, and physical `slots=` output. It does not skip any
evaluation or alter training.

## Primary and streamed training data

| Argument | Default | Effect |
|---|---:|---|
| `--data PATH` | required | Primary SmolTalk-style JSONL. It supplies training and the stable validation benchmark. |
| `--tinystories-data PATH` | none | TinyStories train data as the official delimiter-separated text, JSONL/JSON with a `text` field, Parquet, or a directory of those files. TinyStories is inserted first in tokenizer and stream order. |
| `--tinystories-validation-data PATH` | none | Optional distinct validation file/directory. Without it, the first validation stories are reserved from the training path and excluded from training. |
| `--tinystories-mix-fraction X` | `0.40` | Exact TinyStories share inside the named streaming mixture. Existing source weights are renormalized without changing their ratios. |
| `--tinystories-validation-stories N` | `512` | Number of complete stories used for the TinyStories validation view. |
| `--tinystories-tokenizer-stories N` | `50000` | Number of non-validation TinyStories exposed while fitting a scratch tokenizer. |
| `--tinystories-shuffle-buffer N` | `16384` | Shuffle reservoir for TinyStories continuation examples. |
| `--lmsys-data PATH` | none | Optional LMSYS JSONL source. |
| `--lmsys-weight X` | `0.9` | Legacy streamed-corpus probability. Used when `--streaming-weight` is omitted. |
| `--streaming-weight X` | unset | Total probability assigned to all streamed corpora. The primary corpus receives `1-X`. |
| `--streaming-exhaustion restart\|drop` | `restart` | A restart source drains a complete deduplicated shuffle-buffer epoch before beginning the next epoch; it never cycles merely to fill an oversized buffer. `drop` permanently removes the source after one traversal. Both require streamed rather than memory-loaded LMSYS when explicitly selecting `drop`. |
| `--lmsys-source-weight X` | `1.0` | LMSYS's relative weight *inside* the streamed mixture. Must be positive. |
| `--primary-stream-weight X` | `0.0` | Adds the primary SmolTalk training split to the named streamed mixture at relative weight `X`. Set `1` with LMSYS/Tulu/Preferences weights `4:3:2` and total streaming weight `1` to reproduce the established 10:40:30:20 corpus. Use epoch-safe `restart` to preserve those proportions or `drop` for the one-pass ablation. |
| `--primary-shuffle-buffer N` | `8192` | Shuffle reservoir for the optional primary SmolTalk stream. |
| `--chat-data PATH` | none; repeatable | Adds another training-only chat JSONL source. It does not replace the primary validation benchmark. |
| `--chat-source-weight X` | none; repeatable | Relative streamed-mixture weight corresponding to each `--chat-data`. If omitted for all sources, each receives `1.0`. |
| `--chat-tokenizer-conversations N` | `10000` | Rows sampled from each additional chat corpus when fitting a tokenizer from scratch. |
| `--chat-max-characters N` | unset | Rejects additional-chat rows above this decoded length. Falls back to `--lmsys-max-characters`. |
| `--chat-shuffle-buffer N` | `16384` | Streaming shuffle reservoir per additional corpus. Higher values improve mixing and use more RAM. |
| `--lmsys-validation-conversations N` | `512` | LMSYS conversations inspected to construct its auxiliary validation view. |
| `--lmsys-tokenizer-conversations N` | `50000` | LMSYS rows available while fitting a tokenizer from scratch. |
| `--lmsys-max-characters N` | unset | Rejects an LMSYS row when its complete decoded text exceeds this character count. |
| `--lmsys-load-mode {memory,stream}` | `memory` | Loads accepted LMSYS examples into RAM or streams them continuously. |
| `--lmsys-shuffle-buffer N` | `2048` | Shuffle reservoir used by streamed LMSYS loading. |
| `--lmsys-min-ascii-letter-ratio X` | `0.0` | Cheap English-oriented filter. Higher values reject rows with fewer ASCII letters. |
| `--lmsys-min-response-characters N` | `0` | Rejects short assistant responses before tokenization. |
| `--lmsys-require-user-prompt` | off | Requires the immediately preceding turn to be a user turn. Useful for removing malformed fragments. |
| `--lmsys-skip-first-assistant` | off | Drops the first LMSYS assistant reply when it is likely boilerplate. |
| `--skip-first-assistant` | off | Drops the first assistant reply in the primary conversation corpus. |
| `--require-complete-windows` | off | Rejects examples if either full context or response plus EOS would be token-truncated. This is the strict combination. |
| `--require-complete-context` | off | After whole-turn trimming, rejects contexts that still exceed the context window. |
| `--require-complete-response` | off | Rejects responses that cannot fit with EOS instead of training on a right-censored prefix. |
| `--trim-context-turns` | off | Removes complete oldest turns until context fits. It does not crop through a sentence. |
| `--deduplicate-examples` | off | Deduplicates normalized context/response pairs within each corpus traversal. |

### Mixture math

If `--streaming-weight 0.9` and the streamed source weights are:

```text
LMSYS=4, Tulu=3, SmolTalk Preferences=2
```

then approximately 10% of samples come from the primary corpus and the
remaining 90% is split 4:3:2 among the streamed sources.

Corpus diversity has been important for Gator. Small repeated subsets are
memorized cleanly, but held-out language quality collapses while specializing.
Treat `--repeat-training-examples` as a capacity diagnostic, not a normal
training recipe.

With TinyStories enabled at `0.40` and the established existing weights
`1:4:3:2`, the resolved named-stream shares are:

```text
TinyStories            40%
SmolTalk Everyday       6%
LMSYS                   24%
Tulu3                   18%
SmolTalk Preferences    12%
```

The trainer derives the TinyStories relative weight from the other weights and
prints the normalized `corpus_mix` at startup. TinyStories is parsed by
`<|endoftext|>` in the official `.txt`; newlines inside a story are not treated
as story boundaries. At context 64, each example uses the first two sentences
as its earliest prefix. Every additional 64 context slots makes one additional
two-sentence prefix boundary available. The response continuation similarly
grows by one sentence pair per 64 response slots. Normal token-window filters
remain authoritative, so no overlength prefix is silently cropped.

## Tokenizer and simultaneous formation window

| Argument | Default | Effect |
|---|---:|---|
| `--vocab-size N` | `2048` | Vocabulary entries, including control symbols. It strongly affects lexical parameter count. |
| `--context-tokens N` | `64` | Maximum input slots shown to the model. Larger windows increase receiver-memory work and activation memory. |
| `--response-tokens N` | `32` | Full reference/formation window. In `chunk_parallel`, only the selected chunk is instantiated as output slots. |
| `--reference-tokens N` | `0` | Maximum target pieces placed into the response window. `0` uses all output slots. A smaller value reserves physical alignment slack. |
| `--decode-mode {parallel,serial,chunk_parallel,teacher_forced,handoff,graduated_handoff,token_handoff,prefix_denoise}` | `parallel` | Selects blank simultaneous formation, the one-prefix/one-target serial diagnostic, parallel next-chunk formation, efficient causal teacher forcing, either whole-batch handoff controller, exact tokenwise teacher decay, or causal token/span prefix denoising. |
| `--chunk-size N` | `8` | Number of target tokens formed simultaneously by `chunk_parallel`. Earlier complete chunks are appended to the prompt; the selected chunk remains fully hidden. |
| `--handoff-start-content-accuracy X` | `0.60` | In `handoff` mode, use 100% teacher-forced batches below this smoothed teacher-forced validation content accuracy. |
| `--handoff-end-content-accuracy X` | `0.90` | In `handoff` mode, reach 0% teacher-forced batches at this smoothed content accuracy. |
| `--handoff-accuracy-ema-decay X` | `0.50` | Smooths validation feedback for `handoff`, `graduated_handoff`, and `prefix_denoise`. `0` uses only the latest evaluation. |
| `--graduated-baseline-fraction X` | `0.05` | Fraction of planned `graduated_handoff` steps spent alternating unsupported and teacher batches before locking the unsupported content-accuracy baseline. |
| `--graduated-accuracy-step X` | `0.05` | Every gain of this many unsupported content-accuracy points adds one unsupported batch between teacher batches. |
| `--graduated-release-accuracy X` | `0.90` | Smoothed unsupported content top-1 accuracy that permanently removes teacher batches. |
| `--token-handoff-start-removed N` | `1` | Shifted teacher tokens removed from every row at the first tokenwise curriculum level. |
| `--token-handoff-max-removed N` | `0` | Final removal level. `0` means all `response_tokens - 1` eligible teacher tokens. |
| `--token-handoff-advance-ratio X` | `0.95` | Unsupported-slot accuracy divided by teacher accuracy at the same addresses required for advancement. |
| `--token-handoff-retreat-ratio X` | `0.75` | Smoothed frontier ratio below which one teacher token is restored. |
| `--token-handoff-min-teacher-accuracy X` | `0.15` | Prevents a meaningless low/low accuracy ratio from advancing the frontier. |
| `--token-handoff-advance-patience N` | `3` | Consecutive successful evaluations required to remove one more token. |
| `--token-handoff-retreat-patience N` | `2` | Consecutive failed evaluations required to restore one token. |
| `--token-handoff-full-probe-fraction X` | `0.02` | Fraction of training batches receiving no answer teacher tokens at any curriculum level. |
| `--token-handoff-unforced-loss-weight X` | `1.0` | Extra cross-entropy weight applied only to output addresses whose immediate teacher token is missing. |
| `--prefix-retention-start X` | `0.95` | Initial fraction of eligible shifted answer-prefix tokens retained by `prefix_denoise`. |
| `--prefix-retention-floor X` | `0.50` | Retention reached mechanically before context/accuracy feedback may remove additional support. |
| `--prefix-denoise-warmup-steps N` | `100000` | Steps used to move from start retention to floor retention. This uses absolute completed steps and resumes cleanly. |
| `--prefix-zero-row-fraction X` | `0.05` | Fraction of training rows receiving no gold answer prefix. |
| `--prefix-short-row-fraction X` | `0.10` | Fraction of rows retaining only a short leading answer prefix. |
| `--prefix-short-max-tokens N` | `8` | Maximum number of visible shifted target tokens on a short-prefix row. |
| `--prefix-span-fraction X` | `0.50` | Fraction of ordinary rows using one contiguous missing span; the remainder use independent token dropout. |
| `--prefix-context-utility-start X` | `0.02` | Correct-context minus shuffled-context accuracy margin at which adaptive removal below the retention floor begins. |
| `--prefix-context-utility-end X` | `0.10` | Context-utility margin counted as fully ready for adaptive removal. |

For example, `--response-tokens 40 --reference-tokens 32` provides eight slots
in which the alignment objective may shift tokens right without deleting target
content.

`chunk_parallel` trains next-chunk rather than next-token decoding. For a
128-token reference and `--chunk-size 8`, each row selects one of up to sixteen
aligned chunks. The model input contains the original prompt followed by all
gold chunks completed before that boundary. The current eight-token chunk is
represented only by eight output queries and all eight targets are predicted
simultaneously with noncausal graph interaction:

```text
prompt + gold chunks [0:k] -> eight BLANK queries -> target chunk [k:k+8]
```

No token from the selected chunk is exposed to another slot in that chunk.
Training samples chunk indices uniformly per row. Validation assigns chunk
indices deterministically across examples so every logging pass is comparable
and reports oracle-prefix next-chunk accuracy. Deployment commits each
predicted chunk to context before forming the next; free-running chunk rollout
must therefore be evaluated separately from the oracle-prefix training metric.
The full reference window remains controlled by `--response-tokens`; it must
not be reduced to the chunk size.

`teacher_forced` is the proper serial-decoding training control for Convergent
Gator. Output slot zero receives BLANK; every later slot receives the preceding
correct reference token. The graph receiver kernel and traffic occupancy are
causally masked, so slot `i` cannot read a future output writer either directly
or indirectly through later-relay traffic. All active response positions
produce next-token losses in one forward pass, while inference remains
autoregressive. This mode currently requires `--architecture convergent`,
`--formation-objective slot_ce`, and `global` or `slot` terminal voting.

`handoff` uses the same safe causal teacher-forced forward, but changes the
training regime according to teacher-forced validation content accuracy:

```text
accuracy <= 0.60: teacher fraction = 1.00
accuracy  = 0.75: teacher fraction = 0.50
accuracy >= 0.90: teacher fraction = 0.00
```

Only complete batches change mode. A teacher batch supplies the shifted
correct response under causal graph memory; a parallel batch supplies no
target response tokens and uses noncausal blank formation. Randomly revealing
some correct output tokens inside a noncausal batch is intentionally not
supported because other slots could read future answers through graph memory.

`graduated_handoff` fixes the historical controller's main failure: it is
gated by unsupported validation content accuracy rather than teacher-forced
accuracy, and unsupported formation is present from step one. During the first
5% of the planned run, complete batches alternate:

```text
unsupported, teacher, unsupported, teacher, ...
```

The unsupported content-accuracy EMA at the end of that window becomes the
baseline. Every five percentage-point gain above baseline adds one unsupported
batch before the next teacher batch:

```text
baseline + 0 points: U, T
baseline + 5 points: U, U, T
baseline + 10 points: U, U, U, T
```

Curriculum levels advance monotonically. At 90% unsupported content top-1
accuracy the teacher is permanently released. The baseline, EMA, gain level,
cycle position, release latch, and batch counts are checkpointed so resume
continues the exact cadence.

`token_handoff` is the fine-grained bridge from a trained causal model to
unsupported parallel formation. Slot zero is intrinsically unsupported.
Every other output slot initially receives the preceding gold token, then the
controller replaces exactly one additional shifted teacher token per row with
the non-rendering BLANK token. Removed addresses rotate deterministically
across rows and steps so every slot learns without a permanent privileged
subset.

At each logging evaluation, the partially supported model and the fully
teacher-forced model are scored at the exact same removed content addresses.
The controller advances one token only when the smoothed accuracy ratio clears
the advance threshold for the configured patience. A separate lower threshold
can restore one token after sustained regression. Two percent of training
batches default to fully unsupported formation, keeping the final pathway
alive before the frontier reaches it. Unsupported addresses receive their own
additional CE term. Logs include the current visible/removed counts, frontier
and teacher-frontier accuracy, fully unsupported accuracy/PPL, controller
action and streaks, plus a `free_formed` qualitative sample. Checkpoints save
the complete frontier state and a separate `best_unsupported.pt`.

`prefix_denoise` is the revised curriculum after the D20 million-step run
exposed a gold-prefix continuation shortcut. It always uses the causal graph
mask, so revealing a shifted target token cannot leak it backward. Ordinary
rows use either independent token dropout or one contiguous missing span.
Five percent of rows default to a completely blank response prefix and ten
percent retain only one to eight leading prefix tokens.

The retained-token schedule moves from `0.95` to `0.50` during the first
100,000 completed steps regardless of validation accuracy, preventing the
old controller from remaining permanently teacher-forced. Further support
removal requires both:

```text
teacher content accuracy:           0.60 -> 0.90
correct - shuffled context accuracy: 0.02 -> 0.10
```

The lower readiness of those two signals controls the adaptive portion. If
either signal falls, causal support returns. Context utility compares correct
and shuffled prompts under the exact same deterministic partial-prefix mask at
the current retention; it is not measured with an intact gold answer. Logs
report requested and actual retention, row corruption types, full-teacher,
partial-prefix, and shuffled-context PPL/accuracy, context utility, both EMAs,
and readiness. Checkpoints include the complete controller state and save
`best_context.pt` using positive context utility times teacher content
accuracy.

The requested fraction is implemented with deterministic error diffusion, so
`0.50` alternates five teacher and five parallel batches in every ten-batch
window instead of adding another random-seed effect. Both validation modes are
evaluated at every logging interval. `best.pt` is selected by blank-parallel
validation loss, `best_teacher.pt` preserves the best causal control, and
`latest.pt` stores the controller EMA, mixture credit, and batch counts.
If free-form training lowers causal content accuracy, teacher support rises
again automatically.

The older `serial` diagnostic samples one prefix and one next-token target per
example. It is useful for probing but exposes far fewer labels per optimizer
step and should not be treated as the main serial baseline.

## Formation objectives and sequence quality

| Argument | Default | Effect |
|---|---:|---|
| `--formation-objective {slot_ce,blank_ce,viterbi,monotonic}` | `slot_ce` | Chooses rigid historical CE, explicit BLANK CE, detached best-path alignment, or differentiable forward-sum alignment. |
| `--alignment-warmup-fraction X` | `0.05` | Blends from rigid BLANK CE into Viterbi/monotonic alignment over this fraction of the phase. `0` activates alignment immediately. |
| `--alignment-max-shift N` | `4` | Maximum slots a target piece may move right. `-1` permits every legal monotonic placement. |
| `--sequence-quality-penalty X` | `0.0` | Overall weight on the whole-formation composition, length, and EOS auxiliary. |
| `--sequence-composition-weight X` | `1.0` | Relative weight for matching semantic token composition. |
| `--sequence-length-weight X` | `0.25` | Relative weight for expected rendered length. |
| `--sequence-eos-weight X` | `0.10` | Relative weight for complete-response EOS behavior. |
| `--whitespace-omission-penalty X` | `0.0` | Penalizes whitespace probability where a rigid active target is non-whitespace. |
| `--adjacent-whitespace-penalty X` | `0.0` | Penalizes consecutive whitespace probability in rigid active target slots. |

### Objective meanings

| Objective | Meaning | Use |
|---|---|---|
| `slot_ce` | Scores only historical active target slots at fixed addresses. | Compatibility and rigid baseline. |
| `blank_ce` | Scores target pieces, EOS where known, and BLANK in known unused slots. | Rigid simultaneous-formation baseline. |
| `viterbi` | Selects a detached best monotonic placement, then applies CE. | Hard aligned diagnostic. |
| `monotonic` | Marginalizes all legal emit-next-piece-or-BLANK paths. | Current flexible simultaneous-formation objective. |

Rigid whitespace penalties are intentionally rejected with `viterbi` and
`monotonic`; they would reintroduce a slot-specific assumption that alignment
is designed to remove.

Increasing alignment freedom can improve rendered placement but also makes the
task less anchored. Change `--alignment-max-shift` separately from model shape.

## Architecture selection and model width

| Argument | Default | Effect |
|---|---:|---|
| `--architecture {gator,convergent,transformer}` | `gator` | Selects cell-based Arrow, flat persistent-lane Convergent Gator, or a one-shot non-autoregressive Transformer under the same task and trainer. |
| `--d-model N` | `32` | Gator cell width, or Transformer hidden width. Current Arrow language cells use `8`. |
| `--embedding-dim N` | same as `d-model` | Shared language/lexical width. Current language experiments use `32` feeding width-8 cells. |
| `--cells N` | `10` | Number of Gator cells. More cells add specialized local compute, routing state, and activation cost. |
| `--nodes N` | `4` | Graph nodes inside each cell. More nodes add internal functional choices and node transforms. |
| `--relays N` | `4` | Recurrent graph handoff/computation rounds before the response is read out. More relays add sequential latent depth, not output-token steps. |
| `--neighbors N` | `4` | Neighboring cells reachable from each source cell. The handoff action set also includes self, so four neighbors means five actions. |
| `--node-topology {asymmetric,symmetric,hybrid_alternating,hybrid_hemispheric}` | `asymmetric` | Selects directed node masking, bidirectional node masking, relay-alternating masks, or cell-population hemispheres with complementary masks. |

Increasing cells, nodes, relays, and neighbors are not equivalent:

- More cells increases the population and specialization capacity.
- More nodes increases computation choices *inside* each cell.
- More relays gives every token more handoffs and latent computation depth.
- More neighbors increases each cell's immediate communication fanout.

Change one at a time for interpretable A/B tests.

### Convergent-only graph and voter controls

These arguments apply when `--architecture convergent`:

| Argument | Default | Effect |
|---|---:|---|
| `--graph-degree N` | `3` | Fixed outgoing graph connections per shared node. |
| `--lanes N` | `10` | Persistent complementary payload traces. This is distinct from shared node count. |
| `--connection-layout {ring,small_world,dense}` | `small_world` | Fixed shared graph layout. |
| `--connection-seed N` | `0` | Deterministic graph-rewiring seed. |
| `--connection-rewire X` | `0.5` | Fraction of eligible small-world edges replaced by shortcuts. |
| `--terminal-vote-mode {global,prompt,slot}` | `global` | Uses one static lane jury, one prompt-conditioned jury shared across the complete output, or one prompt-and-immutable-address-aware lane jury per simultaneous output slot. Prompt mode initializes as the exact global-vote forward. Slot mode adds a randomly initialized private four-value address key per lane, creating distinct juries immediately while preserving every shared tensor's seeded initialization. |
| `--router-mode {private_full,shared_low_rank}` | `private_full` | Keeps the legacy complete private router at every source node or factors packet interpretation through one shared low-rank trunk while retaining private node-edge keys and biases. |
| `--router-rank N` | `16` | Width of the shared routing trunk and edge-key space. Only used by `shared_low_rank`. |
| `--ingress-mode {fixed,learned}` | `fixed` | Starts every lane at its static portal or lets each packet choose among all node ingress keys. `learned` requires `shared_low_rank`. |
| `--active-node-top-k N` | unset | Retains at most `N` node locations per packet after ingress and every relay, renormalizes their mass, and dispatches only those node transforms. |
| `--active-node-mass-threshold X` | `0.0` | Drops retained node locations below probability `X` before renormalization. At least the highest-mass node always survives. |
| `--route-backend {dense,packed}` | `dense` | Keeps the complete `[B,T,L,N]` passport oracle or carries a fixed-capacity `(node_id, mass, valid)` frontier between relays. `packed` requires `shared_low_rank` routing, learned ingress, and `--active-node-top-k`. |
| `--packed-memory-backend {matrix,convolution,fft,fft_opaque}` | `matrix` | Selects the exact dense-node matrix oracle, exact direct grouped convolution, raw eager FFT, or compile-safe opaque FFT receiver memory. Every non-matrix choice requires `--route-backend packed`. |
| `--packed-memory-reader-backend {torch,triton}` | `torch` | Selects the portable expanded packed-reader oracle or the explicit fused CUDA/Triton normalization, gather, weighting, and reduction kernel with registered backward. `triton` is float32 CUDA-only in its first release and never silently falls back. |
| `--packed-memory-writer-backend {torch,triton}` | `torch` | Selects the portable writer oracle or fused CUDA/Triton writer construction and explicit adjoint. The Triton path requires packed `fft_opaque` memory, float32 CUDA tensors, and feature chunks no wider than 128; it never silently falls back. |
| `--packed-memory-feature-chunk N` | `32` | Number of hidden features processed in one packed convolution-memory pass. Smaller chunks reduce individual forward intermediates at the cost of more launches; autograd may still retain several chunks for backward. |
| `--packed-memory-compact-nodes` | off | Dynamically restricts packed convolution memory to the node union present in the current batch. This path is eager-only until fixed-cap compaction exists and is rejected with `--compile`. |
| `--adaptive-route-min-k N` | unset | Enables packet-local route width between this minimum and `--active-node-top-k`. Sharply routed packets narrow while uncertain packets retain more alternatives. Requires packed routing. |
| `--adaptive-route-mode {entropy,margin,hybrid}` | `hybrid` | Defines uncertainty from normalized route entropy, the top-one probability margin, or a mixture of both. Used only when `--adaptive-route-min-k` is set. |
| `--adaptive-route-entropy-mix X` | `0.5` | Entropy weight in `hybrid` adaptive routing; `0` uses the margin term and `1` uses entropy. Must lie in `[0,1]`. |
| `--compact-node-dispatch` | off | Removes zero-mass adaptive route rows before node GEMMs. The current dynamic implementation is eager-only and is rejected with `--compile`. |
| `--node-compute-mode {full_private,shared_adapter}` | `full_private` | Gives every node a complete private `D x D` transform or uses one shared `D x D` trunk plus private node embeddings, biases, and low-rank adapters. `shared_adapter` currently requires packed routing and is a model-capacity A/B, not an execution-only switch. |
| `--node-adapter-rank N` | `8` | Rank of each private down/up adapter when `--node-compute-mode shared_adapter` is selected. |
| `--ingress-groups N` | unset | Enables a balanced hierarchical learned-ingress partition with `N` coarse groups. Every graph node remains structurally reachable through its group. Requires packed learned ingress. |
| `--ingress-top-groups N` | `2` | Number of coarse ingress groups retained before scoring their node candidates. Must not exceed `--ingress-groups`; the resulting candidate count must be at least `--active-node-top-k`. |
| `--no-ingress-learned-groups` | off | Derives coarse group keys and biases from member-node keys instead of adding private group parameters. By default hierarchical ingress learns `groups * (router_rank + 1)` extra scalars. |

`prompt` and `slot` voting each add `lanes * d_model` parameters. Their query
starts at zero, making the initial forward exactly equal to `global`; ordinary
task gradients then learn context-sensitive lane confidence. `prompt` chooses
one jury for the entire response. `slot` applies the same tiny query to every
final lane/slot state, allowing different immutable output addresses to select
different coalitions without autoregressive token feedback.

The routing optimization experiment is enabled explicitly:

```bash
--router-mode shared_low_rank \
--router-rank 16 \
--ingress-mode learned \
--active-node-top-k 4 \
--active-node-mass-threshold 0.01
```

The low-rank factorization reduces router parameters while preserving private
source-edge identity. Learned ingress makes every node a possible first
steward. Active-node controls are packet-local and make the node-transform
stage sparse, but they intentionally alter the soft-routing estimator by
discarding small branches.

### Packed Convergent execution

The exact packed execution baseline is:

```bash
--router-mode shared_low_rank \
--ingress-mode learned \
--active-node-top-k 32 \
--route-backend packed \
--packed-memory-backend matrix
```

`matrix` preserves the receiver-memory equation as a transparent numerical
oracle, but it still builds dense node-indexed writer and memory tensors.
Here, exact means dense-versus-packed backend equivalence under the same
fixed-top-k route policy; top-k pruning itself is not equivalent to the
unpruned soft passport.
`convolution` and `fft` use the translation-invariant relative-position
kernel directly, avoid a quadratic `[T,T,N]` relation tensor, process hidden
features in bounded chunks, and gather only memories requested by packed
readers. Direct convolution may be faster at shorter context; FFT has better
asymptotic scaling and can still lose at small shapes. Select between them on
the target GPU rather than from a CPU or forward-only screen.

The current TorchInductor stack cannot code-generate the raw FFT path's
`complex64` intermediates. Raw `fft` therefore remains eager-only.
`fft_opaque` executes the same complete FFT receiver behind real-valued custom
operators in forward and backward, preventing those complex tensors from
entering the compiled graph. It recomputes the FFT receiver in backward, which
trades compute for a smaller retained forward graph. The trainer and 140M
runner reject only raw `fft + --compile` before model construction.

The fused writer/reader A/B is:

```bash
--route-backend packed \
--packed-memory-backend fft_opaque \
--packed-memory-reader-backend triton \
--packed-memory-writer-backend triton \
--packed-memory-feature-chunk 32 \
--compile
```

Explicit Triton selection requires CUDA, float32, Triton, and
`torch.library.triton_op`; it raises instead of changing the backend when any
requirement is missing. The fused kernels remove the K-expanded writer and
reader tensors; `torch` remains the reference for both boundaries. They do
not yet fuse active-node
selection, cuFFT outputs, collision merge, occupancy, or route transport.

Packed storage has a static maximum width so it can remain shape-stable under
compilation. Adaptive routing changes only the `valid` mask inside that
storage: route-count selection is discrete, while retained probability masses
remain differentiable. Static compiled kernels still process `Kmax`; adaptive
K becomes a compute saving only with eager compaction or a future bucketed or
mask-aware fused kernel. It is therefore an estimator/architecture A/B, not a
numerically transparent replacement for fixed top-k. Likewise,
`shared_adapter` and hierarchical ingress change learned parameters and must
be compared separately from execution backends.

The two compact flags currently use data-dependent eager grouping. They can
reduce work when a batch visits a small node union, but they cannot be combined
with `--compile`. Leaving them off does not restore dense passports: packed
frontiers still carry only their fixed-capacity node IDs and masses.

Compiled packed `full_private` node dispatch additionally requires CUDA
grouped MM, compute capability 8.0 or newer, and a BF16 row width aligned to
16 bytes. The trainer rejects unsupported devices/widths instead of falling
into a host-synchronized per-node loop inside a compiled graph. Use eager
execution or `shared_adapter` on an unsupported target. The current
full-private grouped node GEMM uses BF16 operands with FP32 master parameters,
routing, memory, and optimizer state; this is the same boundary used by the
frozen 140M control.

A larger graph does not automatically make active-row optimization useful.
The slim terminal line reports:

| Diagnostic | Meaning |
|---|---|
| `route_k=[...]` | Mean number of valid packed node routes at ingress and after each relay. |
| `route_uncertainty=[...]` | Mean adaptive uncertainty at ingress and after each relay; emitted only when adaptive routing is enabled. |
| `node_union` | Fraction of graph nodes touched by any evaluated packet at any recorded ingress/relay state. A value near one means the batch is sparse per packet but dense across the batch. |

These diagnostics require route tracing during evaluation; they are not
per-step optimizer statistics. `node_union`, in particular, should be measured
before considering a lazy active-row optimizer.

## Transformer-only shape

These arguments apply when `--architecture transformer`:

| Argument | Default | Effect |
|---|---:|---|
| `--transformer-heads N` | `2` | Self-attention heads. `d-model` must be divisible by this value. |
| `--transformer-ff-width N` | `76` | Hidden width of each feed-forward block. |
| `--transformer-layers N` | `4` | Noncausal encoder layers applied before simultaneous readout. |
| `--transformer-dropout X` | `0.0` | Transformer dropout probability. |

## Lexical input, cell views, and output voting

| Argument | Default | Effect |
|---|---:|---|
| `--lexical-mode {private_hash,communal}` | `private_hash` | Gives cells tiny private modulo-hash prototypes or one shared full vocabulary table. |
| `--cell-input-mode {shared_projection,feature_gated}` | `shared_projection` | Gives every cell the same narrow projected view or learns per-cell gates over the wider language embedding. |
| `--cell-output-mode {shared_pool,grouped_concat}` | `shared_pool` | Pools every cell into one width-`d_model` channel or forms several grouped channels and concatenates them before lexical readout. |
| `--cell-transport-mode {broadcast_mix,stewardship}` | `broadcast_mix` | Selects legacy neighboring broadcast mixing or token ownership/state/node handoffs. |

The validated language configuration is currently:

```bash
--lexical-mode communal \
--cell-input-mode feature_gated \
--cell-output-mode grouped_concat \
--cell-transport-mode stewardship
```

`feature_gated` prevents every cell from viewing the width-32 lexical embedding
through one identical width-8 keyhole. `grouped_concat` restores matching
output width instead of collapsing the population to one width-8 vote.

## Stewardship semantics

Stewardship has two nested routing decisions:

1. **Ownership nucleus:** how many cells continue carrying a token.
2. **Handoff beam:** how many local destinations each current steward tries.

These controls are related, but they are not interchangeable.

| Argument | Default | Turn it up to… | Turn it down to… |
|---|---:|---|---|
| `--stewardship-keep-mass X` | `0.9` | Retain more cells per token and broaden ownership. | Force a narrower ownership nucleus sooner. |
| `--stewardship-handoff-keep-mass X` | `0.9` | Retain more destination actions per steward. This is the cleanest “more handoff routing” lever. | Make confident stewards choose fewer destinations. |
| `--stewardship-min-active N` | `1` | Guarantee a wider minimum cell population for every token. | Permit very narrow familiar routes. |
| `--stewardship-max-active N` | `0` | `0` removes the cap; a larger explicit cap permits more cells. | Put a hard ceiling on active cells and compute. |
| `--stewardship-exploration X` | `0.01` | Reserve more probability for alternate paths and make narrowing more reversible. | Exploit learned routes more deterministically. |
| `--stewardship-scouts N` | `1` | Preserve more alternate branches outside the main nucleus. | Retain fewer exploratory alternatives. |
| `--stewardship-warmup-fraction X` | `0.05` | Keep all routes dense for longer before hard-forward pruning. | Begin sparse specialization earlier. |
| `--stewardship-temperature-start X` | `1.5` | Make early handoff distributions softer. | Make early choices sharper. |
| `--stewardship-temperature-end X` | `0.35` | Keep late handoffs softer and usually retain broader beams. | Make familiar late routes crisper. |
| `--stewardship-familiar-entropy-penalty X` | `0.001` | Apply stronger late pressure for familiar routes to sharpen. | Allow familiar routes to remain broader; `0` removes this auxiliary. |
| `--stewardship-balance-penalty X` | `0.01` | Push batch-marginal traffic away from population collapse. | Allow specialization/load imbalance to emerge more freely. |

### Routing recipes

Change one lever per clean A/B.

| Goal | First change |
|---|---|
| More destinations per current steward | `--stewardship-handoff-keep-mass 0.97` |
| More cells carrying each token | `--stewardship-keep-mass 0.97` |
| Hard minimum of eight stewards | `--stewardship-min-active 8` |
| Preserve more novel alternatives | `--stewardship-scouts 2` |
| Slightly stronger reversible exploration | `--stewardship-exploration 0.02` |
| Stop explicitly sharpening familiar routes | `--stewardship-familiar-entropy-penalty 0` |
| Keep late routing softer | `--stewardship-temperature-end 0.7` |

There is currently no direct stay/move penalty. `stay_rate` is measured, but the
loss does not reward leaving the current cell merely for the sake of motion.
That is deliberate: useful routing should improve the formation objective, not
become compulsory token tourism.

With four neighbors, each steward has five actions including self. A stay rate
near `0.20` is therefore approximately uniform; below `0.20` favors handoff,
and above `0.20` favors local retention.

### Logged stewardship diagnostics

| Diagnostic | Meaning |
|---|---|
| `support` | Mean number of nonzero cell routes per token at initialization and after every relay. |
| `effective` | Entropy-effective number of cells, which distinguishes one dominant owner plus dust from genuinely broad ownership. |
| `uncertainty` | Any-live-steward uncertainty used to widen novel routes. |
| `mean_uncertainty` | Ownership-weighted uncertainty used by the familiar-route auxiliary. |
| `handoff_beam` | Mean number of retained local actions per steward. |
| `widen` / `narrow` | Fraction of routes whose support expands or contracts at a handoff. |
| `stay` | Ownership-weighted probability of choosing the source cell itself. |
| `load_cv` | Coefficient of variation of cell traffic; larger means more uneven specialization. |
| `dead` | Fraction of cells receiving no traffic. |
| `packed` | Whether each relay used the packed dispatcher; it is not a FLOP measurement. |

Narrower cells may rely on stewardship more heavily because they have less
local transformation capacity. Compare cell widths using these diagnostics,
not only validation loss.

## Stewardship execution backends

These settings change implementation and performance, not intended model
semantics:

| Argument | Default | Effect |
|---|---:|---|
| `--stewardship-execution {dense,packed_auto}` | `dense` | Always evaluates dense routes or periodically latches a packed suffix when live support is low enough. |
| `--stewardship-packed-threshold X` | `0.5` | Packed execution may latch when the complete relay suffix stays at or below this active-route fraction. Higher values switch sooner. |
| `--packed-stewardship-backend {repack,persistent_bucketed}` | `persistent_bucketed` | Selects the legacy pack/unpack oracle or persistent power-of-two bucketed sparse suffix. |
| `--stewardship-execution-check-every N` | `100` | Steps between route-density observations. The decision is host-side and one-way once latched. |
| `--receiver-memory-backend {attention,convolution}` | `convolution` | Uses the quadratic semantic reference or exact normalized convolutional receiver read. |
| `--dense-router-backend {einsum,grouped_bmm}` | `einsum` | Selects dense router contraction. `grouped_bmm` can lose to transpose overhead and should remain benchmark-gated. |
| `--dense-node-backend {einsum,folded_bmm}` | `folded_bmm` | Selects expanded node transforms or an algebraically folded wide projection that avoids a large node-input tensor. |
| `--dense-kernel-padding N` | `1` | Pads feature dimensions to this multiple. `1` disables manual padding and is the safe default. |

`packed_auto` does not bounce between engines. It begins dense, observes
support on schedule, and permanently latches the earliest qualifying packed
relay suffix. Novel routes may widen semantically after pruning, but the
execution plan itself remains static for that controller.

## Execution presets and GPU memory

| Argument | Default | Effect |
|---|---:|---|
| `--execution-preset {manual,dense_gpu_safe,dense_gpu,optimized_gpu_safe,optimized_gpu}` | `manual` | Applies a tested group of parameter-free backend, pipeline, compilation, and precision settings. |
| `--max-vram-gb X` | unset | Restricts PyTorch's CUDA allocator to this many decimal GB. It is a cap, not a reservation or prediction. |

| Preset | Compilation | Precision | Stewardship execution |
|---|---|---|---|
| `manual` | individual default | individual default | individual default |
| `dense_gpu_safe` | off | strict FP32 | dense |
| `dense_gpu` | on | aggressive FP32 Tensor Core/TF32 | dense |
| `optimized_gpu_safe` | off | strict FP32 | `packed_auto` with structural optimizations |
| `optimized_gpu` | on | aggressive FP32 Tensor Core/TF32 | `packed_auto` with structural optimizations |

The historical fault-isolation ladder was:

```text
optimized_gpu_safe
    → optimized_gpu_safe --compile
    → optimized_gpu
```

Long compiled Blackwell runs still eventually produced genuine NaNs. Use
`optimized_gpu_safe` without `--compile` for the preserved Arrow fallback.
Compiled modes remain available for diagnostics and short profiling, but are
not the current long-run recommendation.

## Compilation and numerical precision

| Argument | Default | Effect |
|---|---:|---|
| `--compile` | off | Wraps the static dense model call in `torch.compile`. Initial compilation can take tens of seconds. |
| `--compile-mode MODE` | `default` | Selects TorchInductor's default, reduced-overhead, or autotuning policy. |
| `--compile-shape-padding` / `--no-compile-shape-padding` | backend default | Explicitly enables/disables Inductor matrix shape padding. Custom padding is compatible only with default compile mode. |
| `--float32-matmul-precision {highest,high,medium}` | `highest` | Controls internal FP32 matmul precision. `high` permits TF32 Tensor Cores on supported NVIDIA GPUs. |
| `--cudnn-tf32` / `--no-cudnn-tf32` | PyTorch default | Separately controls TF32 for cuDNN receiver-memory convolution. |

Matmul TF32 and cuDNN TF32 are separate switches. If an aggressive run becomes
unstable, isolate them separately rather than treating “TF32” as one setting.

The optimizer rejects genuine NaN/Inf gradients. If every gradient element is
finite but PyTorch's FP32 aggregate norm overflows, it recomputes only that
exceptional norm in FP64, clips, and continues.

## Optimizer and learning-rate schedule

| Argument | Default | Effect |
|---|---:|---|
| `--optimizer {adamw,adam,sgd}` | `adamw` | Update rule. |
| `--learning-rate X` | `0.008` | Initial LR for this phase. Explicit experiment commands commonly override it to `0.003`. |
| `--weight-decay X` | `0.0001` | Parameter shrinkage applied by the optimizer. |
| `--gradient-clip X` | `1.0` | Global L2 norm ceiling. Higher values permit larger updates; lower values suppress spikes. |
| `--scheduler {constant,cosine,milestone}` | `constant` | LR schedule for this invocation. |
| `--minimum-lr X` | `0.0` | Cosine schedule floor. |
| `--lr-milestones N...` | none | Step numbers at which milestone scheduling multiplies LR by `--lr-gamma`. |
| `--lr-gamma X` | `0.25` | Multiplicative LR change at each milestone. |

When resuming with `--resume-optimizer`, moments continue but the scheduler is
rebuilt for the new phase. This supports “rewind to best checkpoint, change
data/LR, continue” experiments without replaying an expired schedule.

## Batch preparation, transfer, and exact resume

| Argument | Default | Effect |
|---|---:|---|
| `--batch-pipeline {serial,prefetch}` | `prefetch` | Samples inline or overlaps CPU preparation with GPU model work. |
| `--prefetch-batches N` | `2` | Maximum prepared batches in the bounded queue. |
| `--pin-memory` / `--no-pin-memory` | on for CUDA | Page-locks queued CPU tensors for asynchronous copies. |
| `--non-blocking-transfer` / `--no-non-blocking-transfer` | on | Permits asynchronous copies from pinned memory. |
| `--dedicated-copy-stream` / `--no-dedicated-copy-stream` | on | Stages copies on a dedicated CUDA stream. |
| `--resume-data-state {auto,exact,portable_exact,reset}` | `auto` | Restores compatible state, requires strict file identity, permits an exact encoded-corpus restore after a portable file copy, or deliberately starts a new data phase. |
| `--tokenizer-workers N` | `0` | Ordered tokenization workers. `0` uses serial preparation. |
| `--tokenizer-worker-backend {threads,processes}` | `processes` | Parallel worker type when workers are enabled. |
| `--tokenizer-chunk-size N` | `256` | Examples assigned to each ordered tokenization work item. |
| `--tokenizer-pending-chunks N` | `0` | Maximum queued work chunks. `0` means twice the worker count. |

Prefetch diagnostics report queue occupancy, producer waits, sampling time,
transfer time, and failures. If the queue remains full and producer waits are
high, CPU preparation is keeping up; increasing workers or queue depth will not
speed the model.

`portable_exact` is not a data reset. It permits absolute path or modification
time to differ only after the saved validation fingerprint matches; the native
provider then verifies the complete encoded training-corpus fingerprint, task
geometry, queued batches, and RNG state before any update executes.

## Common experimental mistakes

1. **Changing route semantics and execution together.** First compare routing
   knobs under one execution engine; benchmark packing afterward.
2. **Calling higher keep mass “sparser.”** Higher keep mass retains *more*
   probability support and is generally broader.
3. **Confusing handoff breadth with ownership breadth.**
   `handoff-keep-mass` controls destinations per steward;
   `keep-mass` controls continuing steward cells per token.
4. **Reading `packed=0` as no stewardship.** It means the dense dispatcher
   executed the stewardship equations.
5. **Comparing final checkpoints after regression.** Use the best validation
   checkpoint and record the shape of the regression curve.
6. **Assuming formed text shows every slot.** Inspect both `formed=` and
   `slots=`; BLANK and whitespace can make a short rendered answer occupy a
   much wider physical formation.
7. **Assuming a preset prevents explicit overrides.** Explicit low-level flags
   win.
8. **Using memorization subsets as generalization evidence.** They measure
   capacity and optimization only.

## Keeping this reference synchronized

The parser remains authoritative:

```bash
python gator_language_train.py --help
```

When adding an argument, update this file in the same commit and add it to
checkpoint metadata when it affects model semantics, data identity, optimizer
continuation, or execution reproducibility.
