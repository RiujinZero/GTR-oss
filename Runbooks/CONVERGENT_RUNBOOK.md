# Convergent Gator runbook

`--architecture convergent`: the single-graph node-routing language model
(`gator_convergent.py`'s `ConvergentGator`). This doc tracks the currently
validated 100M-scale "golden" config — the one actually running in
production as of 2026-07-26, resumed lineage `gator100m_serial_direct_golden_v2`,
step 1.16M+ of a 3.5M-step target (~700-800 tokens/param by the end).

See `Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md` for the full plateau/collapse
investigation this config's choices are justified by, and
`Papers/GATOR_LANGUAGE_EXPERIMENT.md` for the convergent-vs-Arrow (cell-ring)
comparison.

## Known findings that shape this config

- **Every PPL/eval metric reported for this run before 2026-07-27 was
  likely TinyStories-only, not overall quality.** `validation_examples` is
  built `[*tinystories, *smoltalk_everyday, *lmsys]` with no shuffling, and
  the eval loop truncated with a plain `examples[:limit]` — since
  TinyStories alone supplies more examples than `--eval-examples`, every
  reported PPL/accuracy/content_ppl number (and the displayed sample) came
  exclusively from TinyStories. `--chat-data`/`--prose-data` (tulu3,
  smoltalk_preference, wiki, openscience) contributed zero validation
  examples at all. Fixed (commit `2b43adb`): stratified eval sampling by
  source, plus `--chat-validation-conversations`/
  `--prose-validation-stories` to reserve held-out examples from those
  sources. **Add both flags to any resume going forward** (small positive
  defaults already apply, 64 each) — without them you're still blind to
  actual chat quality.
- Constant LR causes calibration drift (content PPL bottoms early then
  climbs while raw accuracy keeps improving) — an LR schedule has never
  actually been tried; this is still an open, high-priority experiment.
- `--decode-mode serial` was the one change that eliminated the
  improve-then-regress pattern at small scale (35k-52k params). Believed but
  **not fully re-confirmed** at 100M scale — this run is the first long-horizon
  test of that extrapolation.
- Free generation can look clean (no whitespace/constant-token collapse)
  while still being contextually wrong — the checkpoint at step 1.16M
  produces fluent but off-topic completions (drifts into TinyStories-style
  narrative regardless of prompt). Watch `formed=`/`target=` samples and a
  real free-generation check (`semantic_grade_checkpoint.py`), not just PPL.
- Raw prose corpora (TinyStories, Wikipedia, OpenScience) pull free
  generation toward their own style (narrative / fact-recitation) in
  proportion to their mix weight — turn-based chat corpora (LMSYS, Tulu3,
  SmolTalk) are what teaches actual conversational context-following.
  Current plan: hold at TinyStories 10% / Wikipedia 15% / OpenScience 15% /
  turn-based chat 60% until this run's PPL breaks into the 20s (currently
  ~32), then rebalance to 30% total prose / 70% turn-based chat.

## Architecture (97.7M params, verified from the live checkpoint's state_dict)

| Field | Value |
|---|---|
| `vocab_size` | 20000 (`gator-compact-v2` tokenizer) |
| `d_model` | 379 |
| `embedding_dim` | 200 |
| `n_nodes` | 643 |
| `degree` | 4 |
| `relays` | 2 |
| `lanes` | 1 |
| `router_mode` | `shared_low_rank`, `router_rank=96` |
| `ingress_mode` | `learned` (`ingress_top_groups=2`, `ingress_learned_groups=True`) |
| `route_backend` | `dense` |
| `node_compute_mode` | `full_private` |
| `adaptive_route_mode` | `hybrid` |
| `connection_layout` | `small_world`, `connection_seed=0`, `connection_rewire=0.5` |
| `terminal_vote_mode` | `prompt` |
| `active_node_top_k` | 32 |
| `context_tokens` / `response_tokens` / `reference_tokens` | 512 / 128 / 128 |
| `formation_objective` | `slot_ce` |
| `decode_mode` | `serial` |

## Data mixture (current, as of the wiki/openscience addition)

Interleaved streams, target shares of the total mix:

- TinyStories (`--tinystories-data` / `--tinystories-mix-fraction 0.10`)
- Wikipedia (`--prose-data wikipedia_en.jsonl --prose-source-weight 2.5`) — 15%
- OpenScience (`--prose-data openscience.jsonl --prose-source-weight 2.5`) — 15%
- SmolTalk Everyday (`--primary-stream-weight 1`) — turn-based chat
- LMSYS Chat 1M (`--lmsys-data --lmsys-source-weight 4`) — turn-based chat
- Tulu3 SFT (`--chat-data ... --chat-source-weight 3`) — turn-based chat
- SmolTalk Preference (`--chat-data ... --chat-source-weight 2`) — turn-based chat

`--prose-data`/`--prose-source-weight` are new flags added 2026-07-26 —
raw-text JSONL (`{"text": ...}` per row) run through the same
prefix/continuation pairing TinyStories already used
(`gator/data/tinystories.py`'s `iter_tinystories`/`tinystories_sentence_pairs`),
not the conversational `--chat-data` pipeline.

**Weight formula** used to hit an exact target share for a new stream without
touching the others: `new_weight = (sum of all other non-TinyStories stream
weights) / N`, where `N` solves `target_share = new_weight / (new_weight +
existing_other) * (1 - tinystories_fraction)` — see chat log 2026-07-26 for
the full derivation. For two new streams at 15% each with existing_other=10
(lmsys 4 + tulu3 3 + smoltalk_preference 2 + primary 1) and
`tinystories_mix_fraction=0.10`, that's `10/4 = 2.5` each.

## Training settings

- `--batch-size 4 --gradient-accumulation-steps 8` (32 effective batch,
  20,480 tokens/step at 640-token sequences)
- `--optimizer-groups monolithic`, `--learning-rate 3e-4`, `--weight-decay
  0.0001`, `--optimizer-epsilon 1e-8`, `--gradient-clip 1.0`, `--scheduler
  constant` (untested: replacing this with cosine/step decay is the single
  highest-priority open experiment)
- Route temperature: `route_temperature_schedule=legacy` (1.25 -> 0.55 ->
  0.35 cooling), runs automatically — do not hardcode a fixed temperature
  (a known bug in `run_gator_100m_staged_colab.py`, avoided entirely by
  calling `gator_language_train.py` directly instead of that staged wrapper)
- `--compile` enabled; mixed precision is fp32 (`matmul_precision=highest`)
  on this run, not bf16 — bf16 for convergent has not been validated the way
  it now has for Arrow (see `Runbooks/ARROW_RUNBOOK.md`'s bf16 fix)

## Colab cell

See `Runbooks/colab_cells/convergent_100m_launch_cell.py`. Paste the whole
file into one Colab cell after uploading `gator_colab_bundle.zip` to
`/content/drive/MyDrive/GTR/colab_bundles/`. Edit the `PATHS` block only.

Runtime model: `gator_language_train.py --steps N` runs N steps **this
invocation**, added on top of the resumed checkpoint's own step count — it
is not an absolute target. The cell loops in `SEGMENT_STEPS` chunks,
resuming from the last checkpoint each time, until `TARGET_STEPS` is
reached or the Colab runtime disconnects (rerun the same cell to resume from
the last Drive-synced checkpoint).

## Causal execution / KV-cache prerequisite: swap at inference, not training

`causal=True` (needed for any valid KV cache — see the memory scaling
section above) requires `terminal_vote_mode` in `{global, slot}`, not this
config's `prompt`. Tested both paths:

- **Training-time**: resumed with `--terminal-vote-mode global
  --position-norm-length 640 --force-causal`. PPL immediately jumped from
  ~32 to ~220 (bigger disruption than a frozen-weight sanity check
  predicted), and after a partial recovery to steady monotonic decline, it
  regressed back to oscillating. Inconclusive/risky — **not currently
  recommended**.
- **Inference-only** (no retraining): took a fresh snapshot of the native
  `prompt`-mode checkpoint and generated the same prompts both natively
  (causal=False) and with `terminal_vote_mode=global` + `causal=True`
  swapped in only for the forward pass (state_dict loads with 1 harmlessly
  unused key, `terminal_vote_query`). **Comparable quality, no
  degradation** — same generic/repetitive content either way, not worse.

**Conclusion: prefer swapping to causal only at inference/serving time.**
This gets the causal-execution prerequisite for a KV cache without paying
any training-instability cost — training continues natively in `prompt`
mode as before. Whether training itself should ever run causal remains an
open, unresolved question (one regression on one run isn't a controlled
study) — revisit only if there's a specific reason to want causally-trained
weights rather than a post-hoc swap.

## Open items

- Context length increase (user wants this eventually) will need memory-
  scaling mitigation first — not yet scoped. For Arrow, cell count was found
  to dominate activation memory far more than raw param count; the
  equivalent lever for convergent (route width / active_node_top_k /
  n_nodes) hasn't been characterized the same way.
- LR schedule (cosine/step decay) — still untested at any scale.
- Prose-corpus rebalance (30% prose / 70% turn-based chat) — gated on PPL
  reaching the 20s.
- KV-cache inference code itself — still not implemented (only the
  causal-execution prerequisite and the length-invariant position encoding
  groundwork have landed). Now expected to be built against inference-only
  causal swaps on the natively-trained checkpoint, not a causally-trained one.
