# Gator golden_v4 tokens/param milestones

Tracks free-generation quality at fixed tokens/param checkpoints for the
`golden_v4` lineage (started 2026-07-28): `causal=True` from step 0,
`terminal_vote_mode=global` from init, `position_norm_length=640` fixed from
the start, `lanes=4` (vs golden_v2's `lanes=1`). See
`Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md` and `Runbooks/CONVERGENT_RUNBOOK.md`
for the broader context this run is testing (whether the earlier "training-time
causal execution regresses" finding was `lanes=1`-specific).

Params: **99,044,424**. Context: **620**. Batch: **10** (raised from an
initial 4 -> 6 -> 10 sweep this session; VRAM-maxed at 84/95GB, real
throughput ~23k tok/s, the practical ceiling for this hardware/config).

## Probe tool — LOCKED

**Always use `semantic_grade_checkpoint.py`** for every milestone in this
file, not `language_probe.py`. `language_probe.py` does a single non-causal
forward pass with all response slots filled blank at once (a task the model
was never trained to do standalone) and produces repetition-loop garbage
regardless of actual checkpoint quality — it false-alarmed on the golden_v2
checkpoint earlier this session (three different steps, all "broken") before
this was caught. `semantic_grade_checkpoint.py`'s `free_generate()` does true
causal, serial autoregressive rollout (one token at a time, fed back as
context, `causal=True`) and is what actually produced the reliable
step-1.16M and step-1.82M golden_v2 findings this session. Using a different
probe per milestone would make the results incomparable to each other.

Invocation (adjust `--checkpoint`/`--data` per milestone):

```bash
python3 semantic_grade_checkpoint.py \
    --checkpoint /path/to/checkpoint_at_this_milestone.pt \
    --data /path/to/eval.jsonl \
    --num-examples 50 \
    --response-tokens 128 \
    --chunk-size 8
```

Note: golden_v4 trains natively with `terminal_vote_mode=global` from init
(not the `prompt`->`global` inference-time swap golden_v2 needed), so
`causal=True` should just work directly — no swap workaround required here.

**Locked prompt set** (same 8 used throughout this session's checkpoint
quality checks, for direct comparability across milestones and against
golden_v2/Sonic):

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

## `--serial-positions-per-example` (K) — now an active tuning variable

After the milestone-1 total-whitespace-collapse result below, `gator_language_train.py`
gained `--serial-positions-per-example N` (2026-07-28): for `decode_mode=serial`,
expands each training row into N stratified prefix positions per step
instead of one random position per row, giving denser per-example
supervision at an ~N x compute cost (VRAM cost is much less than Nx in
practice — see notes below). Goal: find a K high enough that the model
stops defaulting to a whitespace-heavy global prior instead of genuinely
conditioning on context, without over-fitting to a single conclusion from
training-log samples alone (always confirm with the locked
`semantic_grade_checkpoint.py` probe, not the trainer's own eval prints,
which can look better than true causal free-generation).

K is being swept live on the running lineage (not a fixed hyperparameter
per milestone) - **record which K was active when each milestone below was
reached**, since it's a second axis alongside tokens/param now:

- K=1 (original, milestone 1 below): total whitespace collapse.
- K=2 (batch reduced 10->8, then 10->5 to compensate for the row-count
  multiply): steps/sec dropped (~3.63->2.5 at batch=8) but real throughput
  (rows/sec, the correct unit given each row is a full context-window
  forward pass) actually rose slightly (~36.3->40.0 rows/sec) - denser
  supervision at a small net throughput *gain*, not a cost.
- K=128 (full response window): OOM.
- K=4, K=16: being tested next (2026-07-28), informed by two VRAM findings:
  (1) VRAM did NOT scale with row count as expected from the tiny-model
  smoke test's ratio - two different K=2 configs (batch=8 and batch=5,
  16 vs 10 rows) landed at the *same* ~51GB post-eval-pass plateau, because
  `--eval-batch-size` (a fixed, K-independent setting, default 64) sets a
  memory high-water-mark via PyTorch's caching allocator (reserved memory
  never shrinks back down once claimed) that dominated the training-batch's
  own footprint at this scale; (2) cutting `--eval-batch-size` frees real
  headroom for a larger training K/batch than the naive VRAM-only estimate
  suggested.

## Milestones

`tokens/step = batch(10) x context(620) = 6,200` at the K=1 baseline batch
size - recompute per-row-count if batch/K change (see above). Step targets:

| tokens/param | steps | step reached (date) | K | PPL | free-gen quality | notes |
|---|---|---|---|---|---|---|
| 5   | 79,875    | 82,000 (2026-07-28) | 1 | 44.56 | **total whitespace collapse** — all 8 prompts generated pure blank/space output, `whitespace_fraction=1.000` on 8/8 examples, no actual text at all | Worse than golden_v2's known failure mode (which at least produces real words, just off-topic). PPL=44.56 is in a plausible/unremarkable range, giving no hint of this from the number alone — another clean PPL/quality decoupling instance. This is what motivated the `--serial-positions-per-example` fix above. |
| 10  | 159,749   | | | | | |
| 20  | 319,498   | | | | | |
| 40  | 638,996   | | | | | |
| 100 | 1,597,491 | | | | | |
| 200 | 3,194,981 | | | | | |

Fill in "free-gen quality" with the same axis used elsewhere this session:
coherent-but-off-topic (TinyStories-drift-style) vs genuinely on-topic vs
repetition-collapse garbage vs memorized-phrase regurgitation. Always note
PPL alongside the qualitative read — the entire point of this log is
watching whether PPL and real quality track together or decouple, per the
plateau study's central finding.
