# Gator golden_v5 tokens/param milestones

Restarted from scratch (2026-07-28) after `golden_v4` accumulated too many
live config changes mid-run (K=1 -> 2 -> 4 sweep, `--eval-batch-size`
changes) to keep clean tokens/param accounting. See
`Papers/GATOR_GOLDEN_V4_MILESTONES.md` for that run's history — its K=1
finding (**total whitespace collapse at step 82,000/5 tokens/param, despite
an unremarkable PPL=44.56**) is still valid evidence and the whole reason
`--serial-positions-per-example` exists; it's just no longer this run's own
progression.

Same architecture as golden_v4: `causal=True` from step 0,
`terminal_vote_mode=global` from init, `position_norm_length=640` fixed
from the start, `lanes=4`. Params: **99,044,424**. Context: **620**.

**golden_v5 fixed config (locked from step 0, no more mid-run changes):**
- `--batch-size 8`
- `--serial-positions-per-example 4` (K=4 — stratified dense supervision
  across the response window, not golden_v4's original K=1 single-random-
  position sampling)
- `--eval-batch-size 16` (down from the default 64 — frees real VRAM
  headroom; the eval pass's memory footprint was found to dominate the
  post-first-log VRAM plateau independent of training batch/K, see
  golden_v4's notes)

`tokens/step = batch(8) x K(4) x context(620) = 19,840`

## Probe tool — LOCKED

Same as golden_v4: **always use `semantic_grade_checkpoint.py`**, never
`language_probe.py` (non-causal single-shot decode, produces false-alarm
repetition-garbage regardless of real checkpoint quality — see golden_v4's
doc for the full reasoning). True causal serial autoregressive rollout only.

```bash
python3 semantic_grade_checkpoint.py \
    --checkpoint /path/to/checkpoint_at_this_milestone.pt \
    --data /path/to/eval.jsonl \
    --num-examples 50 \
    --response-tokens 128 \
    --chunk-size 8
```

**Locked prompt set** (same 8 used throughout this whole session, for
comparability against golden_v2/golden_v4/Sonic):

```
What season is it right now?
Name an animal that says meow.
What is 2 + 2?
I need to rent a car. What types are available?
What are some popular team sports?
How often should a beginner run?
Once upon a time,
Explain how photosynthesis works.
```

## Milestones

| tokens/param | steps | step reached (date) | PPL | free-gen quality | notes |
|---|---|---|---|---|---|
| 5   | 24,961    | | | | |
| 10  | 49,922    | | | | |
| 20  | 99,843    | | | | |
| 40  | 199,686   | | | | |
| 100 | 499,216   | | | | |
| 200 | 998,432   | | | | |

Fill in "free-gen quality" with the same axis used elsewhere this session:
coherent-but-off-topic (TinyStories-drift-style) vs genuinely on-topic vs
repetition-collapse garbage vs memorized-phrase regurgitation vs whitespace
collapse. Always note PPL alongside the qualitative read — the entire point
of this log is watching whether PPL and real quality track together or
decouple, per the plateau study's central finding. **Milestone 1 (5
tokens/param) is the critical checkpoint given golden_v4 collapsed exactly
there** — don't skip it even though it's early.
