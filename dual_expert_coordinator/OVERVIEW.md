# Dual-Expert + Coordinator Architecture

## Background

The original goal was to find a single shared-topology Gator configuration
(~130K params) that could learn both language (smoltalk, target PPL ~50) and
modular-prefix-sum math jointly within a 50K-step budget on an L4, using
`chunk_parallel` decode mode as a "hybrid" formation strategy (parallel
formation within a chunk, autoregressive across chunks).

Six mechanically distinct configurations were tested against this goal:

1. `chunk_parallel` decode, language-favoring topology (`shared_low_rank` /
   `learned` / `small_world`), ~50/50 corpus mix.
2. Same as (1), corrected corpus-mixing math (90% math-weighted).
3. `chunk_parallel` decode, math-validated topology (`private_full` /
   `fixed` / `ring`, `relays=4`), 90% math-weighted.
4. Same as (3), ~99.9% math-weighted (near-pure math).
5. Sequential curriculum: 20K steps math-only, then 20K steps language-only,
   then 10K steps interleaved (transformer architecture).
6. `graduated_handoff` decode mode (gradual teacher-forcing release) on the
   math-validated topology, 90% math-weighted.

**Every configuration failed the same way**: math formation collapsed to a
degenerate constant output (e.g. `'aaaaaaa'`, `'eeeeeeee'`) with `exact=0.000`,
regardless of topology, corpus weighting, curriculum ordering, or decode
schedule. Configuration (5) additionally demonstrated catastrophic
forgetting — math capability learned in phase 1 did not survive phase 2's
language-only training, and could not be recovered by phase 3's interleave.

A further isolation test showed the failure wasn't specific to the hybrid
mixture at all: training the math-validated topology on **100% pure math
data** through the `gator_language_train.py` wrapper still plateaued at
PPL~8.8 (a "predict the marginal per-slot character frequency" degenerate
optimum) regardless of relay count (1 vs 4) or vocab size (2000 vs 263) —
whereas the same topology trained via `gator_convergent.py`'s **native**
training loop (fresh synthetic batches every step, no corpus repetition)
broke out of that exact plateau and reached genuine task competence. This
means the wrapper's finite, repeated-corpus JSONL pipeline creates a much
stickier local minimum than the native script's infinite fresh-batch
generator — not a topology or capacity problem.

## Conclusion: single shared topology is a dead end

No amount of hyperparameter search within a single shared network topology
produced a configuration that could learn both tasks jointly. The two tasks
appear to need incompatible internal dynamics:

- **Math** (modular prefix-sum) needs joint, simultaneous formation across
  all 7 output positions, driven by many relay rounds of message-passing
  over a `private_full` / `fixed` / `ring` graph, and benefits from
  data generated fresh every step rather than a finite repeated corpus.
- **Language** (smoltalk) needs genuinely serial/autoregressive formation,
  and trains cleanly on a `shared_low_rank` / `learned` / `small_world`
  topology with ordinary repeated-corpus training.

Forcing one topology/objective to do both, at any point on the sweep, always
resolved in favor of whichever task's optimization basin was more accessible
under the current schedule — usually language, with math collapsing to a
constant.

## New strategy: two frozen experts + a trainable coordinator

Rather than searching for a shared configuration, each task gets its own
independently-trained network, frozen after training, with only a small
router/coordinator trained on top to decide which expert to consult. This is
architecturally similar to a mixture-of-experts, except the "experts" differ
in **topology and training regime**, not just learned weights on an
otherwise-identical architecture.

### Math expert
- Trained via `gator_convergent.py`'s native loop directly (not the
  `gator_language_train.py` wrapper), which avoids the finite-corpus
  plateau entirely.
- Topology: `private_full` / `fixed` / `ring`, `degree=3`, `relays=4`,
  `nodes=51`, `width(d_model)=30`, `lanes=8`, `vocabulary_mode=private_lane`
  (16-token alphabet, `embedding_dim=d_model`).
- 81,656 parameters.
- 30,000 steps on fresh synthetic modular-prefix batches (7-token output).
- Result: ~64% per-character accuracy, ~4% exact-sequence match, still
  slowly improving when stopped (a proof-of-concept result, not a ceiling).
- Driver: `expert_math_native_run.py`.

### Language expert
- Trained via `gator_language_train.py` with `--decode-mode serial`
  (genuinely autoregressive, one token at a time — not `chunk_parallel`).
- Topology: `shared_low_rank` / `learned` / `small_world`, `degree=4`,
  `relays=4`, `nodes=53`, `d_model=31`, `lanes=8`, `vocabulary_mode=
  shared_projected` (2000-token vocab).
- 132,112 parameters.
- ~21,500 steps on the smoltalk corpus.
- Result: PPL 28-29, comfortably under the original PPL-50 target.
- Driver: `expert_language_serial_run.py`.

### Coordinator (router)
Because the two experts use **incompatible output spaces** (math: 16-symbol
alphabet, 7 fixed output slots; language: 2000-token vocab, up to 128 slots,
different tokenizer), blending their logits directly is not meaningful.
The coordinator therefore performs **hard dispatch** rather than continuous
mixing:

- A tiny classifier (character-histogram features → 2-layer MLP, 2,305
  parameters total) looks at the raw input prompt and predicts which expert
  should handle it.
- Trained on 8,000 interleaved labeled examples (4,000 math prompts, 4,000
  language prompts) drawn directly from each task's training corpus —
  reached 100% validation accuracy within ~20 epochs (the two domains are
  lexically trivial to distinguish: math prompts are a fixed `s<10 letters>`
  pattern, language prompts are natural-language text).
- At inference, the router's decision selects which frozen expert's own
  native decode pipeline runs on the full input — the other expert is never
  invoked.
- Spot-checked on held-out math probes and language validation prompts:
  100% correct dispatch, with high confidence in both directions
  (math scores ≥0.996, language scores ≤0.007).
- Driver: `coordinator_prototype.py`.

## Why this works where the shared topology didn't

The router's job — deciding which of two lexically distinct input
distributions a prompt belongs to — is a nearly trivial classification
problem, learnable in seconds with a two-layer MLP. All of the actual task
difficulty (learning math, learning language) is handled by networks that
never have to compromise with each other, because they're never jointly
optimized. This sidesteps every failure mode observed in the shared-topology
sweep: no corpus-mixing imbalance, no catastrophic forgetting, no
topology-vs-objective mismatch, no shared vocabulary dilution.

## Reusing an already-trained model as an expert (e.g. the 100M serial run)

The coordinator doesn't require experts to be trained specifically for this
prototype — any existing checkpoint can be dropped in as an expert, since
routing happens at the raw-prompt level, entirely outside each expert's own
vocab/tokenizer. To plug in a larger existing serial-language checkpoint
(e.g. the 100M-parameter run) in place of the 132K language expert here:

1. **Get a self-contained checkpoint.** Confirm it was saved via the normal
   `gator_language_train.py` checkpoint format (`{"model_config": ...,
   "model": ...}` — see `checkpoint_state()`). No special export step is
   needed; any `latest.pt` (or a numbered step checkpoint) from that run
   works as-is, including a mid-training checkpoint if you don't want to
   wait for the run to fully finish.
2. **Load it with `load_expert()`** from `coordinator_prototype.py`:
   - It already filters `model_config` down to the keys `ConvergentGator.
     __init__` accepts (via `inspect.signature`), so extra wrapper-only keys
     (e.g. `mask_token_id`) are silently dropped — no edits needed there.
   - It already strips a leading `core.` prefix from the state dict, which
     is what `ConvergentLanguageModel`-wrapped checkpoints (i.e. anything
     trained through `gator_language_train.py`, including the 100M) use.
   - Pass `config_overrides={"vocabulary_mode": "shared_projected"}` (as
     done for the existing language expert) — this is required for any
     `shared_projected`-vocab model; skip it only for a `private_lane`
     (16-token, math-style) expert.
3. **Do not retrain or fine-tune it.** `load_expert()` already calls
   `model.eval()` and sets `requires_grad_(False)` on every parameter —
   the point of freezing is that the expert's own training run (whatever
   step it stopped at) is untouched by anything that happens afterward.
4. **Re-validate the router, don't assume it transfers.** The existing
   router was trained (and hit 100% val accuracy) on math-vs-smoltalk,
   which are lexically almost trivially separable. Swapping in the 100M as
   the serial expert means the router needs to be retrained on whatever the
   *new* task pairing actually is (100M's corpus vs. the parallel expert's
   task). Reuse the same `Router` class and training loop in
   `coordinator_prototype.py`, just point `load_prompts()` at the new
   corpora — don't reuse the saved `coordinator_router.pt` weights, they're
   specific to the old pairing.
5. **Parallel-task expert is trained independently, same as the math
   expert here** — via `gator_convergent.py`'s native loop (see
   `expert_math_native_run.py` as a template), not the
   `gator_language_train.py` wrapper, to avoid the finite-corpus plateau
   documented above. Swap in whatever task-specific batch generator
   replaces `make_prefix_batch` if the parallel task isn't modular-prefix
   math.
6. **Vocab/tokenizer mismatch is expected and fine** for hard dispatch —
   the router only ever sees raw prompt text, never either expert's token
   ids, so a 20K-vocab serial expert and a 16-token parallel expert can
   coexist without any bridging work, as long as dispatch stays
   whole-prompt (no mid-generation handoff between experts).

## Known limitations / next steps

- This is a "one layer each" proof of concept, not a scaled system. The math
  expert's accuracy (64% char, 4% exact) has room to improve with more
  training steps — it was still climbing when stopped.
- The router currently does discrete lexical classification, not semantic
  understanding — it would need to be re-evaluated (or retrained) on inputs
  that don't cleanly separate on surface features (e.g. a language prompt
  that contains embedded math-like tokens). This is the main open risk when
  swapping in a larger, more general serial expert (e.g. the 100M model) —
  math-vs-smoltalk was a lexically easy 100%-accuracy case, but a
  general-purpose language model's corpus may not separate as cleanly from
  whatever the parallel expert's task domain is.
- The two experts use different tokenizers/vocabularies entirely; a
  production version would need a shared or bridged tokenizer if the router
  ever needs to hand off partial context between experts (this prototype
  only supports whole-prompt dispatch, not mid-generation handoff).
- No continuous blending/soft-routing was attempted, since the output
  spaces aren't directly comparable; if a future variant needs blending,
  it would require projecting both experts into a shared representation
  space first.
