# Gator Teacher-Forced Language Control

Date: 2026-07-19

## Why this control exists

The blank-parallel D400 language run remained near raw PPL 58. A matched
teacher-forced run rapidly reached approximately PPL 11. This establishes a
critical distinction:

- the graph, corpus pipeline, optimizer, and D400 parameterization can learn
  substantial language structure;
- asking 128 initially blank slots to form the response without preceding
  target tokens is the current bottleneck;
- teacher forcing is not itself simultaneous novel generation. Training
  predicts every response position in one forward pass, but each position
  receives the correct preceding response tokens through a causal mask.

The result is therefore a diagnostic control, not a claim that the current
model has solved simultaneous response formation.

## Frozen D400 run

The run must remain in this unique Drive directory and must not be reused:

```text
/content/drive/MyDrive/GTR/experimental_checkpoints/
teacher_forced_language_controls/
teacherforced_d400_n66_l10_k3_r2_c512_o128_b8_rw50_seed0_30k
```

```text
/content/drive/MyDrive/GTR/experimental_checkpoints/convergent_scaling/
convergent_teacherforced_d400_n66_l10_k3_r2_c512_o128_b8_rw50_seed0_30k
```

This is the authoritative completed directory. Do not reuse it.

After the process exits, freeze its checkpoints, embedded configuration, and
the exact source bundle with:

```bash
python -u archive_teacher_forced_colab_run.py \
  --run-dir /content/drive/MyDrive/GTR/experimental_checkpoints/convergent_scaling/convergent_teacherforced_d400_n66_l10_k3_r2_c512_o128_b8_rw50_seed0_30k \
  --source-bundle /content/drive/MyDrive/GTR/teacherforced_d400_source_bundle_sha21c0.zip
```

This writes `ARCHIVE_MANIFEST.json` beside the checkpoints and copies the
bundle used for the run to `SOURCE_BUNDLE.zip`. It refuses to replace an
existing frozen source bundle with different bytes.

The exact source revision used to launch D400 has been preserved separately
in Drive as `teacherforced_d400_source_bundle_sha21c0.zip`, file ID
`1LStf_b0vkgjagS2ADuUDb3iO0Xk9d1B3`. Its verified SHA-256 is
`21c0e0a710a5122a3bb2c793809bdcdb280f6e3c91a02b1c8d7766c6dc8f9a1e`
and its size is 538,867 bytes. Do not substitute the newer live bundle when
archiving this run.

```text
Final Drive directory: convergent_scaling/convergent_teacherforced_d400_n66_l10_k3_r2_c512_o128_b8_rw50_seed0_30k
best.pt SHA-256: pending
latest.pt SHA-256: pending
best step: 30,000
best validation loss/PPL: 2.4180 / 11.2238
final step metrics: accuracy 0.4051; top-5 0.6853; content accuracy 0.2288; content top-5 0.5802; content PPL 22.7513; exact 0.0015
GPU/runtime/steps per second: RTX PRO 6000 Blackwell Server Edition / 2,496.31 seconds / 12.02
```

Exact model and task configuration:

| Field | Value |
|---|---:|
| Architecture | Convergent Gator |
| Decode mode | teacher_forced |
| Trainable parameters | 14,360,642 |
| Vocabulary | 2,048 |
| Lexical embedding width | 32 |
| Payload/node width | 400 |
| Nodes | 66 |
| Lanes | 10 |
| Graph degree | 3 |
| Relay depth | 2 |
| Layout | small_world |
| Rewire fraction | 0.5 |
| Terminal vote | slot |
| Context tokens | 512 |
| Response targets | 128 |
| Batch size | 8 |
| Evaluation batch size | 4 |
| Seed | 0 |
| Learning rate | 0.003 |
| Schedule | cosine, 300k horizon, 0.0001 floor |
| Planned run | 30,000 updates |

The corpus is the non-repeating mixed stream:

```text
smoltalk_everyday : lmsys_chat_1m : tulu3_sft : smoltalk_preference
        1         :       4       :     3     :          2
```

Streaming exhaustion is `drop`, maximum accepted conversation length is
8,192 characters, and the usual complete-context, turn-trimming,
deduplication, English-ratio, and minimum-response filters are active.

The complete Colab output is also frozen locally at
`experimental_checkpoints/teacherforced_d400_n66_l10_k3_r2_c512_o128_b8_rw50_seed0_30k/colab_output_verbatim.txt`,
SHA-256
`9bbc68eaba50aaa0155887d8bedc1a126fcbcc3aa1c47613d5255c40eddac998`.

Raw PPL includes the easier structural/blank positions. Content-only PPL is
the more conservative language number and finished at `22.7513`. Likewise,
the printed `formed` samples are not free-running generations: every position
was predicted under its own correct target prefix. Concatenating those
argmaxes creates a diagnostic mosaic, not a causally self-consistent answer.

## D20 handoff probe with TinyStories curriculum

Slot voting adds one width-20 query to each of ten lanes, so the D20 model's
exact count is **130,402** parameters. The older 130,202 figure belongs to
global terminal voting.

This is no longer an exact corpus-controlled D20/D400 capacity A/B. Before the
D20 launch, TinyStories was added as the first tokenizer/training source and
assigned 40% of the named stream. The previous
SmolTalk/LMSYS/Tulu3/Preferences `1:4:3:2` ratio is retained inside the
remaining 60%, producing final shares `40:6:24:18:12`. Preserve the completed
D400 result as the pre-TinyStories teacher-forced control; do not retroactively
attribute a D20 difference solely to payload width.

The next run uses the adaptive handoff rather than remaining teacher-forced
forever. Until teacher-forced validation content accuracy exceeds 60%, it is
mathematically identical to the D20 causal control. From 60% to 90%, complete
blank-parallel batches replace teacher batches linearly. At 90%, the training
stream is fully blank-parallel.

Run it with:

```bash
python -u run_teacher_forced_language_colab.py \
  --d-model 20 \
  --decode-mode handoff \
  --steps 100000 \
  --log-every 1000
```

The runner refuses to reuse a nonempty checkpoint directory, validates the
parameter count before loading data, writes `RUN_SPEC.json`, and mirrors
merged stdout/stderr to `train.log`. Its default directory is:

```text
/content/drive/MyDrive/GTR/experimental_checkpoints/
teacher_forced_language_controls/
handoff60to90_d20_n66_l10_k3_r2_c512_o128_b8_rw50_seed0_100k
```

The D20 run still holds batch size, number of target labels per update, graph,
lanes, relays, terminal voter, seed, optimizer, and schedule fixed. A future
D400 rerun on the same TinyStories-first mixture is required before payload
width can again be treated as the only controlled variable.

At each log, the ordinary `step=` line is the blank-parallel target metric.
The following `handoff` line reports teacher content accuracy/PPL,
blank-parallel content accuracy/PPL, the smoothed controller accuracy, the
next teacher fraction, and cumulative teacher:parallel batch counts. The D400
30k control ended at only 22.88% teacher-forced content accuracy, so it would
still receive 100% teacher forcing under this policy. The handoff begins only
after the model demonstrates substantially stronger local language mastery;
the threshold must not be lowered merely to make the transition occur.

## Decision rule

Record the entire curve, not only final PPL.

- If D20 approaches the D400 teacher-forced curve, the 130k graph already has
  enough language capacity and blank formation is overwhelmingly the missing
  mechanism.
- If D20 learns contextual text but bottoms materially above D400, the gap
  measures capacity under the easier causal objective.
- If D20 fails while D400 succeeds, do not infer that Gator cannot learn
  language; infer a capacity threshold between these two widths.
- If D20 crosses 60% teacher-forced content accuracy, inspect whether its
  blank-parallel curve improves as teacher support decays. This is the direct
  test of learned-language-path transfer into simultaneous formation.

## D20 terminal outcome

The TinyStories-first D20 run completed one million updates but plateaued near
30--31% teacher-content accuracy. It never crossed 60%; therefore every
training batch remained teacher-forced and the adaptive handoff was never
activated. The later checkpoint autopsy below shows that this is specifically
a teacher-prefix shortcut plateau, not a clean causal-capacity result for
130,402 parameters. It does not answer whether a causally scaffolded Gator can
transfer into simultaneous blank formation.

Retain the best teacher checkpoint for the counterfactual and autoregressive
probes below. The next experiment should repair the support curriculum before
bracketing payload width; otherwise a larger model may simply learn the same
shortcut more confidently.

## Post-run checkpoint autopsy

The strongest checkpoint was not the final checkpoint. `best_teacher.pt` was
saved at step 330,000 with SHA-256
`1973dc29b033a9f8a0860e6da57fc477826d273b7a11418402e9e0fd06cfce73`.
Its teacher-forced content accuracy was 31.54%, content top-5 accuracy was
65.27%, and content PPL was 16.49. At step 1,000,000, content accuracy was
still 31.05%, but content PPL had deteriorated to 29.48. The long tail added
confidence error rather than capability.

The original interpretation was too generous: the 30--31% plateau is not
primarily evidence of 30% prompt understanding. A causal counterfactual probe
on held-out examples produced:

| Source | Correct prefix | Shuffled prompt | Masked prompt | Shuffled response prefix | Blank response prefix |
|---|---:|---:|---:|---:|---:|
| TinyStories | 30.25% | 27.63% | 23.44% | 1.36% | 0.97% |
| SmolTalk Everyday | 25.36% | 25.46% | 21.22% | 2.70% | 0.39% |
| LMSYS | 31.47% | 31.47% | 27.02% | 2.02% | 0.73% |

The model used the existence and broad style of the prompt somewhat, because
masking it reduced accuracy by four to seven points. It used prompt identity
very little: exchanging one prompt for another while retaining the correct
response prefix barely changed SmolTalk or LMSYS accuracy. By contrast,
corrupting the gold response history destroyed performance. The dominant
learned strategy was local continuation from the teacher-provided answer
prefix.

True autoregressive extraction confirmed the exposure-bias failure. Example
32-token rollouts collapsed into repetitions such as:

```text
prompt: What season is it right now?
output: I or or is is is is is is is is is is is is is

prompt: Tell me a short story about a dog.
output: I feel feel feel feel feel feel feel what what what are are are are are
```

This is not a slot-address bottleneck. Teacher accuracy was approximately flat
across the 128 response positions and did not degrade systematically with
longer contexts. It is also not principally a terminal-voter bottleneck:
roughly six of ten lanes contributed materially, the ensemble substantially
outperformed any individual lane, and only about four to seven percent of
target-token misses were attributable to the final vote overturning a lane
that preferred the target.

The data schedule introduced a second confound. The run used
`streaming_exhaustion=drop`. SmolTalk exhausted near 20k, Preferences near
140k, Tulu near 320k, and the remaining non-TinyStories source later
exhausted. The advertised `40:6:24:18:12` mixture therefore drifted toward
repeated TinyStories instead of remaining fixed. This likely contributed to
the post-330k calibration regression, although the context counterfactual
shows that the teacher-prefix shortcut was already the primary ceiling.

## Revised next-run rule

Do not merely lower the 60% whole-batch handoff gate. That would jump from a
shortcut the model can exploit directly to a blank-prefix regime it cannot
solve. Instead, train on a continuum of partially corrupted response prefixes:

1. Begin with 90--95% of eligible causal prefix tokens visible.
2. Independently drop tokens or contiguous spans to the true non-rendering
   `BLANK` token while preserving the causal mask.
3. Include a controlled share of zero-prefix and short-prefix cases so the
   first content tokens must be inferred from the prompt.
4. Adapt corruption using held-out context utility: correct-prompt
   loss/accuracy minus shuffled-prompt loss/accuracy.
5. Keep the named corpus proportions fixed with restartable streams, or stop
   when a one-pass source exhausts.
6. Save and rewind to the best content-PPL checkpoint when accuracy is flat
   and calibration regresses.

The gate for reducing support should require both continuation competence and
positive semantic context utility. Raw teacher accuracy alone rewards the
shortcut discovered in this run.

## Implemented retry

The D20 retry is now encoded as `--decode-mode prefix_denoise`. It retains the
same 130,402-parameter D20/N66/L10/R2/slot-voter architecture and changes only
the causal-support curriculum and stream exhaustion policy.

The default schedule retains 95% of eligible shifted response tokens
initially, reaches a 50% floor after 100k completed steps, assigns 5% of rows
no response prefix, assigns 10% a one-to-eight-token prefix, and divides the
remaining rows evenly between independent token dropout and contiguous span
dropout. Removal below the 50% floor requires both 60--90% teacher content
accuracy and a 2--10-point correct-context versus shuffled-context accuracy
margin. The weaker normalized signal controls the transition, so lost context
utility restores support.

The new run uses `streaming_exhaustion=restart`, evaluates shuffled-context
teacher performance at every report, checkpoints the controller state, and
writes `best_context.pt` in addition to the historical best checkpoints.

## July 20 outcome and graduated restart

The D20 prefix-denoising run did not escape. At step 810k, unsupported content
top-1 accuracy remained 0%, unsupported content PPL was approximately 397,
and unsupported content top-5 was approximately 15%. Full teacher support
reached only about 26.5% content top-1 accuracy. This was worse than the
historical pure-parallel formation signal and showed that a teacher-dominant
curriculum could suppress rather than bootstrap Gator's native simultaneous
formation.

The next D20 and D400 controls therefore start from fresh weights with
`--decode-mode graduated_handoff`. The first 5% of planned steps alternates
unsupported and causal teacher batches. That window establishes the
unsupported content-accuracy baseline. Every five-point unsupported gain adds
one additional unsupported batch between teacher batches, and 90% unsupported
accuracy permanently removes teacher support. The controller never uses
teacher accuracy as its release signal.
