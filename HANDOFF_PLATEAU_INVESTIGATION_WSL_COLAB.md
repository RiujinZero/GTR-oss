# Handoff: Gator Plateau Investigation → WSL + Colab

**Purpose of this file:** everything a new Claude Code session (running in WSL, using
Google Colab for GPU speed) needs to pick this investigation up without re-deriving
context. The full research write-up is in
[`Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md`](Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md) —
read that first for the findings. This file is about *state*: what's running, where the
data lives, and how to reproduce or continue each experiment.

**2026-07-23 update:** the root cause is now isolated — see section 6 of the study doc.
It is the `slot_ce` (default) decode mode's independent, simultaneous per-slot content
prediction with no conditioning on other slots' true values, not Gator's architecture, not
parameter count (severity actually scales *up* with size, contradicting this investigation's
original working assumption — see section 6.2), not cross-slot gradient conflict (directly
tested and refuted, section 6.3), and not positional/alignment ambiguity (`viterbi`/`monotonic`
tested and refuted, section 6.5). `serial` decode (true one-token-at-a-time autoregressive)
shows zero regression and is the best result of the whole study. The investigation shifted
from instability-hunting to a capability question (section 7): does Gator actually produce a
smarter model than a same-sized transformer once both are trained on a non-regressing decode
mode? Answered, with a correction: on the tiny 2,260-example corpus (~170 epochs) Gator beat
the transformer dramatically (35.73 vs. 245.22), but that gap was mostly the transformer
collapsing from extreme corpus repetition, not a fundamental weakness — rerunning both on a
50,000-example subset of `smoltalk_preference` (~7.7 epochs) made the transformer fully stable
(29.50 final) and narrowed Gator's edge to a real but much smaller ~28% (21.16 vs. 29.50). See
section 7.1 of the study doc for the full numbers and caveats. Next natural step (not started):
test at even lower repetition (a full pass over the 230k-example/617MB `smoltalk_preference`
file, or a genuinely large corpus) to see whether Gator's edge holds, grows, or shrinks further.

## 1. Where everything lives now

All experiment logs were copied out of the (session-ephemeral, now-inaccessible) Windows
Claude scratchpad into the actual repo at:

```
experimental_checkpoints/plateau_instability_study_20260723/
```

One subdirectory per run, each containing `run.log` (full training log, including
`step=...` metric lines and periodic `context=`/`target=`/`formed=`/`free_formed=`
generation samples). No model checkpoints (`.pt`) were copied — only text logs. If you need
the actual weights, they may still exist on the Windows filesystem; ask the user, since this
handoff doc cannot assume the Windows session/laptop is still reachable.

Training data used throughout: `C:\Users\user\Downloads\smoltalk_everyday.jsonl` (2,260
conversations, SmolTalk Everyday only, no other corpora mixed in). Confirm this file (or an
equivalent) exists in the new environment before trying to reproduce any run — it is not
tracked in git (`training_data/` is gitignored, and this file was in `Downloads`, outside the
repo).

## 2. Windows CPU runs — all three completed before handoff

All three background processes launched during this investigation finished on their own
before this handoff was written; there is nothing still running on the Windows laptop from
this study. Their final logs are already copied into
`experimental_checkpoints/plateau_instability_study_20260723/`.

| Run | Log (Windows path) | Repo copy (as of handoff) |
|---|---|---|
| `arrow_50k` — Arrow architecture, single jury (inherent), monolithic optimizer, 50,000-step target | n/a — **complete** | final content PPL 679.73 at step 50,000 (5.24h runtime); plateaued noisily 600–900, did not explode. Full log in repo copy. No need to resume/re-run unless testing beyond 50k steps. |
| `convergent_50k` — Convergent, multi-jury + isolation + staged optimizer, 50,000-step target | n/a — **complete** | final content PPL 702,730.51 at step 50,000 (5.44h runtime), still climbing at the final step with no plateau reached. Full log in repo copy. |
| `factorial_7_global_mono_ring_local` — the cells/topology test, 8,000-step target | `...\scratchpad\factorial_7_global_mono_ring_local\run.log` | **complete** — final content PPL 122.0 vs. cell 1's 99.3 (23% worse); this question is resolved, see study doc section 3.5 — no need to repeat on Colab |

**Before doing anything else in the new session**, ask the user whether these Windows
processes are still running and, if so, whether to let them finish, pull their latest logs
into the repo copy above, or kill them now that Colab will be faster. Do not assume; the
scratchpad paths above may already be gone if the Windows Claude session was closed.

The exact command lines for each (useful for exact reproduction or for resuming the same
work on Colab) are reconstructable from section 4 below — every flag is listed.

## 3. Why we're moving to Colab

CPU-only training on the Windows laptop was the bottleneck: a 61,770-param model got only
~2–5 steps/sec depending on concurrent load, meaning a full 50,000-step run took most of a
day. All findings in `Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md` are therefore from either
short (8,000-step) small-param (≤62k) runs, or partially-complete long runs. The next phase
should use Colab GPU to:

- finish the runs listed in section 2 faster, or re-launch them fresh on GPU;
- scale the same ablation matrix up 10–100x in params and steps to confirm the findings hold
  at closer-to-production scale (see `Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md` section 5
  for the specific recommended next experiments, in priority order — **the learning-rate-
  schedule test is listed first and is the cheapest, highest-value thing to try**).

## 4. Exact reproduction commands

All of these use `gator_language_train.py` from the repo root. Replace `--data` with wherever
`smoltalk_everyday.jsonl` lives in the new environment (e.g. a Drive-mounted path on Colab).
Common flags across every run below: `--skip-first-assistant --require-complete-context
--trim-context-turns --deduplicate-examples --learning-rate 0.001 --weight-decay 0.0001
--optimizer-epsilon 0.00000001 --gradient-clip 1.0 --scheduler constant --seed 0
--eval-examples 64 --logging-slim`. `--log-every` was 400–500 depending on the run
(see each command). Batch size 16 throughout.

### arrow_50k (Arrow/stewardship architecture)

```bash
python gator_language_train.py --data <smoltalk_everyday.jsonl> \
  --skip-first-assistant --require-complete-context --trim-context-turns --deduplicate-examples \
  --architecture gator \
  --vocab-size 1024 --context-tokens 32 --response-tokens 16 \
  --d-model 8 --embedding-dim 16 --cells 10 --nodes 4 --relays 2 --neighbors 2 --node-topology symmetric \
  --lexical-mode communal --cell-input-mode feature_gated --cell-output-mode shared_pool --cell-transport-mode stewardship \
  --decode-mode parallel --steps 50000 --batch-size 16 --eval-examples 64 --log-every 500 --logging-slim \
  --learning-rate 0.001 --weight-decay 0.0001 --optimizer-epsilon 0.00000001 --gradient-clip 1.0 --scheduler constant \
  --seed 0 --checkpoint-dir <dir>
```

### convergent_50k (full "modern" combo: multi-jury + isolation + staged optimizer)

```bash
python gator_language_train.py --data <smoltalk_everyday.jsonl> \
  --skip-first-assistant --require-complete-context --trim-context-turns --deduplicate-examples \
  --architecture convergent \
  --vocab-size 1024 --context-tokens 32 --response-tokens 16 \
  --d-model 32 --embedding-dim 12 --nodes 30 --graph-degree 4 --relays 2 --lanes 4 \
  --router-mode shared_low_rank --router-rank 8 --ingress-mode learned --active-node-top-k 4 --route-backend dense \
  --connection-layout small_world --connection-seed 0 --connection-rewire 0.5 \
  --terminal-vote-mode slot --slot-isolation-slots 16 \
  --optimizer-groups component_scaled --optimizer-private-node-lr-multiplier 1.25 \
  --decode-mode parallel --steps 50000 --batch-size 16 --eval-examples 64 --log-every 500 --logging-slim \
  --learning-rate 0.001 --weight-decay 0.0001 --optimizer-epsilon 0.00000001 --gradient-clip 1.0 --scheduler constant \
  --seed 0 --checkpoint-dir <dir>
```

### Five-cell factorial (35,324–35,856 params, 8,000 steps, `decode_mode=parallel`)

Shared base flags:

```bash
--architecture convergent \
--vocab-size 1024 --context-tokens 24 --response-tokens 12 \
--d-model 24 --embedding-dim 12 --nodes 20 --graph-degree 4 --relays 2 --lanes 4 \
--router-mode shared_low_rank --router-rank 8 --ingress-mode learned --active-node-top-k 4 --route-backend dense \
--connection-layout small_world --connection-seed 0 --connection-rewire 0.5 \
--decode-mode parallel --steps 8000 --batch-size 16 --eval-examples 64 --log-every 400 --logging-slim \
--learning-rate 0.001 --weight-decay 0.0001 --optimizer-epsilon 0.00000001 --gradient-clip 1.0 --scheduler constant --seed 0
```

Per-cell additions:

1. `--terminal-vote-mode global --optimizer-groups monolithic`
2. `--terminal-vote-mode slot --optimizer-groups monolithic`
3. `--terminal-vote-mode slot --optimizer-groups monolithic --slot-isolation-slots 12`
4. `--terminal-vote-mode slot --optimizer-groups component_scaled --optimizer-private-node-lr-multiplier 1.25 --slot-isolation-slots 12`
5. `--terminal-vote-mode global --optimizer-groups component_scaled --optimizer-private-node-lr-multiplier 1.25`

### Capacity-matched isolation pair (45,098–45,210 params, 8,000 steps)

Shared base flags:

```bash
--architecture convergent \
--vocab-size 512 --context-tokens 24 --response-tokens 12 \
--d-model 24 --embedding-dim 8 --nodes 48 --graph-degree 4 --relays 2 --lanes 4 \
--router-mode shared_low_rank --router-rank 6 --ingress-mode learned --active-node-top-k 4 --route-backend dense \
--connection-layout small_world --connection-seed 0 --connection-rewire 0.5 \
--decode-mode parallel --steps 8000 --batch-size 16 --eval-examples 64 --log-every 400 --logging-slim \
--learning-rate 0.001 --weight-decay 0.0001 --optimizer-epsilon 0.00000001 --gradient-clip 1.0 --scheduler constant \
--optimizer-groups monolithic --seed 0
```

Per-cell additions: (a) `--terminal-vote-mode global`; (b) `--terminal-vote-mode slot
--slot-isolation-slots 12`.

### Teacher-forced control

Identical to factorial cell 1, except `--decode-mode teacher_forced` instead of `parallel`.

### Ring/no-rewire connectivity-locality test (the "cells" question, section 3.5 of the study doc)

Identical to factorial cell 1, except `--graph-degree 2 --connection-layout ring
--connection-rewire 0.0` instead of `--graph-degree 4 --connection-layout small_world
--connection-rewire 0.5`. **This finished before handoff** (final content PPL 122.0 vs.
cell 1's 99.3 — 23% worse, not better). Cells/local connectivity are not protective at this
scale; do not re-run this on Colab unless testing a much larger node count specifically.

## 5. What "success" looks like for the next phase

Re-read `Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md` section 5 for the full prioritized list.
In short: (1) test an LR schedule instead of constant LR — untested anywhere in this whole
investigation and the most likely cheap fix; (2) confirm the gradual-vs-explosive distinction
between plain and "modern" configs reproduces at larger scale; (3) fix the hardcoded
route-temperature bug in `run_gator_100m_staged_colab.py` line ~423; (4) stop trusting PPL
alone for checkpoint selection — inspect generated text samples (`formed=`/`free_formed=`
lines in any `gator_language_train.py` log) before calling any treatment an improvement,
since this investigation twice found PPL and qualitative quality pointing in different
directions. The cells/topology question (previously item 2) is resolved — see section 2's
table above — and does not need to be repeated.
