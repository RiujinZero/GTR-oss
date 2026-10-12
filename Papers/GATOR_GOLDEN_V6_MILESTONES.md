# Gator golden_v6 tokens/param milestones

Restarted (2026-07-28) from `golden_v5` at step 0 after an early qualitative
observation: enabling whitespace penalties made the model noticeably less
confident, suggesting golden_v4/v5's unremarkable PPL numbers (e.g. 44.56 at
a checkpoint that was later found to be TOTAL WHITESPACE COLLAPSE under
`semantic_grade_checkpoint.py`) were likely driven by the model being
extremely confident specifically on whitespace/separator positions —
easy, low-entropy predictions that drag average cross-entropy down a lot
even while the harder, actually-informative content positions are predicted
poorly. PPL is an average over all positions; it can't distinguish
"genuinely good everywhere" from "trivially confident on the easy majority,
bad on the hard minority." See `Papers/GATOR_GOLDEN_V4_MILESTONES.md` and
`Papers/GATOR_GOLDEN_V5_MILESTONES.md` for that history.

**golden_v6 config changes vs golden_v5 (three changed at once — noted for
the record since this makes later attribution ambiguous if quality
improves):**
- `--whitespace-omission-penalty 5` (penalize whitespace probability where
  the active target is non-whitespace)
- `--adjacent-whitespace-penalty 5` (penalize consecutive whitespace
  probability inside active target slots)
- `--lanes 1` (down from golden_v4/v5's `lanes=4`) — user's hypothesis:
  multiple lanes may have been converging on voting for the same "safe"
  high-frequency (whitespace) token, effectively ensembling toward
  overconfidence rather than diversifying, rather than lanes=4 genuinely
  helping causal training as first hypothesized in golden_v4's early PPL
  result. Also **changes the model's parameter count** (per-lane weight
  tensors shrink) — this run's params are NOT golden_v4/v5's 99,044,424.
- `--serial-positions-per-example 4` (K=4, kept from golden_v5)
- `--batch-size 8`, `--eval-batch-size 16` (kept from golden_v5)
- `--gradient-accumulation-steps 6` added (effective batch 48; does not
  change tokens/step, which is driven by `--batch-size`, not accumulation
  — see the Arrow batch-vs-accumulation lesson earlier this session)

Params: **97,648,349**. Context: **620**.
`tokens/step = batch(8) x K(4) x context(620) = 19,840` (unchanged from v5
since batch/K/context are unchanged).

## Probe tool — LOCKED

Same as golden_v4/v5: **always use `semantic_grade_checkpoint.py`**, never
`language_probe.py` (non-causal single-shot decode, produces false-alarm
repetition-garbage regardless of real checkpoint quality).

```bash
python3 semantic_grade_checkpoint.py \
    --checkpoint /path/to/checkpoint_at_this_milestone.pt \
    --data /path/to/eval.jsonl \
    --num-examples 50 \
    --response-tokens 128 \
    --chunk-size 8
```

**Locked prompt set** (same 8 used throughout this whole session):

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
| 5   | 24,609  | | | | |
| 10  | 49,218  | | | | |
| 20  | 98,436  | | | | |
| 40  | 196,871 | | | | |
| 100 | 492,178 | | | | |
| 200 | 984,356 | | | | |

**Milestone 1 (5 tokens/param) is the critical checkpoint** — golden_v4
collapsed into total whitespace exactly there. Fill in "free-gen quality"
with: coherent-but-off-topic (TinyStories-drift-style) vs genuinely
on-topic vs repetition-collapse garbage vs whitespace collapse vs
memorized-phrase regurgitation. Always note PPL alongside the qualitative
read, and specifically note whether PPL now reads *worse* than golden_v4/v5
despite (hopefully) better real generation — that would directly confirm
the "PPL was inflated by whitespace overconfidence" hypothesis this restart
is testing.
