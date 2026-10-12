# Gator golden_v7 tokens/param milestones

Restarted (2026-07-28) from scratch, wiping `golden_v6`'s progress, after a
fast small-scale ablation sweep (`~10M` params, vocab=20000, G4, 1500 steps
per config) FOUND and CONFIRMED the fix golden_v6 was searching for. See
`Papers/GATOR_GOLDEN_V4_MILESTONES.md`, `_V5_`, and `_V6_` for the full
history that led here.

## What the ablation found

User's original hypothesis was right the whole time: `decode_mode=serial`'s
training data construction samples only ONE random position in the response
window per row per step — sparse, scattered supervision that lets the model
fall back to reciting its whitespace-heavy unconditional prior instead of
genuinely conditioning on context. `--serial-positions-per-example` (K,
added this session) was built to fix exactly this. golden_v4/v5/v6 tried
K=1, 2, and 4 on the real production run without clearly breaking the
collapse — the ablation found why: **K needs to be at least 8, not 4**, at
least at this scale. Tested configs (each in isolation, 1500 steps):

| Config | content_acc | ws_omission | coverage | verdict |
|---|---|---|---|---|
| baseline (causal, lanes=1, K=1) | 0.000 | 1.000 | 0.000 | stuck |
| + whitespace penalty alone (K=1) | 0.000 | 1.000 | 0.000 | stuck |
| + K=8 alone (no penalty) | 0.000 | **0.455** | **0.545** | breaks blank-collapse, wrong content |
| non-causal control (K=1) | 0.000 | 1.000 | 0.000 | stuck (rules out causal-masking itself as the fix) |
| + lanes=4 (K=1) | 0.000 | 1.000 | 0.000 | stuck — lanes doesn't matter |
| + higher LR (K=1) | 0.000 | 1.000 | 0.000 | stuck — LR doesn't matter |
| **+ K=8 + whitespace penalty (5/5)** | **0.182** | **0.000** | **1.000** | **real non-zero content_acc, still climbing at step 1500** |

Only K=8 broke the whitespace-collapse local optimum; whitespace penalty,
lane count, and learning rate did nothing in isolation. K=8 combined with
the whitespace penalty gave genuine non-zero content accuracy. Neither
lever alone was sufficient — K=8 opens up real content coverage, the
penalty then helps the model actually get it right rather than just filling
with wrong content.

## golden_v7 config (extrapolated to production ~100M scale)

- `--serial-positions-per-example 8` (K=8 — the confirmed critical lever)
- `--whitespace-omission-penalty 5 --adjacent-whitespace-penalty 5`
- `--terminal-vote-mode global --lanes 1` (lanes=4 showed no benefit in the
  ablation; no reason to pay the extra param/compute cost)
- `--force-causal` (causal=True from init, matching v4-v6)
- `--position-norm-length 620` fixed from init
- `--batch-size 4` (the original extrapolated estimate — landed at ~93GB
  pre-eval and did fit, confirmed at real launch; an earlier report of
  needing batch=3 was corrected). **This is an
  extrapolation, not a measured data point at full scale** — K=8 was only
  tested at the tiny ablation model. Watch VRAM closely on the first few
  hundred real steps before trusting it.
- `--eval-batch-size 16` (frees real headroom; eval pass's memory footprint
  was found to dominate the post-first-log VRAM plateau independent of
  training batch/K, see golden_v4's notes)
- `--ingress-mode learned` (matches the ablation and every prior golden_vN)

Params: **97,648,349** (same lanes=1 architecture as golden_v6). Context:
**620**. Actual launched config: `--batch-size 4` (~93GB pre-eval, fit with
the eval-pass headroom; an earlier report of batch=3 was a correction that
was itself later corrected back to 4).
`tokens/step = batch(4) x K(8) x context(620) = 19,840` — identical to
golden_v6's rate, so the milestone step targets below match v6's table.

## Probe tool — LOCKED

Same as every prior golden_vN: **always use `semantic_grade_checkpoint.py`**,
never `language_probe.py` (non-causal single-shot decode, produces
false-alarm repetition-garbage regardless of real checkpoint quality).

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

**Milestone 1 (5 tokens/param) is still the critical checkpoint** — every
prior lineage (v4, v5, v6) either collapsed or was still stuck there. Fill
in "free-gen quality" with: coherent-but-off-topic (TinyStories-drift-style)
vs genuinely on-topic vs repetition-collapse garbage vs whitespace collapse
vs memorized-phrase regurgitation. Always note PPL alongside the
qualitative read and `content_acc` from the training log if available —
this is the first lineage with a real, ablation-confirmed reason to expect
`content_acc` to actually be non-zero at this checkpoint.
