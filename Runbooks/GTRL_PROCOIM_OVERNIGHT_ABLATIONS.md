# GTR-L Procoim overnight ablations

This is the handoff protocol for the first optimized-search Procoim sweep on
the seven-slot generated simultaneous-prefix task. Run the matrix serially and
do not change model code during the sweep.

## Experiment contract

Keep these settings fixed unless a row below explicitly changes one:

```text
model                 GTR-L prefix
output slots          7
streams               10
nodes                 40
width                 8
relays/layers         4
batch size            1
evaluation batch      1024
precision             FP32
compile               off
candidate directions  up to 8 antithetic pairs; stop on first eligible probe
candidate banks       16 consecutive stalls
accepted-update cap   64
slot reduction        0.5% per episode and per accepted perturbation
probe scale           0.03
stability streak      3 accepted updates
device                 CPU
```

Every Procoim model forward is one training step. Baselines, positive and
negative probes, explicit antithetic estimates, anchor evaluations, and
stability confirmations all consume the `--steps` budget. Fresh validation is
reported separately and is not charged. Compare equal forward budgets, not
episodes or accepted updates.

Procoim is intentionally FP32-only in this prototype. Do not enable AMP,
BF16, FP16, or `torch.compile`. Candidate batching with `vmap` is not part of
this bundle. Do not run two variants concurrently on the same machine.

## Gate zero: record the completed controls

The legacy U128 seed-0 control completed at final/best accuracy `0.0864`, loss
`2.6916`, and `29.75` overall steps/s. Its accepted-update ceiling rarely bound,
so use U64 for every new row. The controls below predate first-hit acceptance
and the hard per-update reduction gate; retain them as context, not same-code
baselines.

The completed legacy 64-update controls are:

| seed | final accuracy | best accuracy | final loss | steps/s |
|---:|---:|---:|---:|---:|
| 0 | 0.0845 | 0.0866 | 2.6963 | 28.62 |
| 1 | 0.0868 | 0.0884 | 2.6959 | 43.91 |
| 2 | 0.0809 | 0.0873 | 2.6748 | 39.77 |

Chance token accuracy is `1/12 = 0.08333`. Accuracy is noisy enough that loss,
best accuracy, final accuracy, and the seven-position vector must all be kept.
The seed-2 control is the warning case: its loss improved while final argmax
accuracy fell below chance.

## Colab setup

The bundle is expected at:

```text
/content/drive/MyDrive/GTR-L/gtrl_evotoken_relay/gtrl_evotoken_bundle.zip
```

Mount Drive, extract the bundle, and verify the two Procoim imports before
launching anything:

```python
from google.colab import drive
from pathlib import Path
import shutil
import sys
import zipfile

drive.mount("/content/drive")

bundle = Path(
    "/content/drive/MyDrive/GTR-L/gtrl_evotoken_relay/"
    "gtrl_evotoken_bundle.zip"
)
source = Path("/content/gtrl_procoim_source")
if not bundle.is_file():
    raise FileNotFoundError(bundle)
if source.exists():
    shutil.rmtree(source)
source.mkdir(parents=True)
with zipfile.ZipFile(bundle) as archive:
    bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"corrupt bundle member: {bad_member}")
    archive.extractall(source)

sys.path.insert(0, str(source))
from gator.optimizers import ProcoimConfig, ProcoimOptimizer

print(bundle)
print(source)
print(ProcoimConfig().reduction_fraction)
```

The final line must be `0.005`.

## Serial runner

Save the following as `/content/run_procoim_ablation.sh`, or run it in one
`%%bash` cell. The function refuses to overwrite a log. Output and the small
latest checkpoint are written directly to Drive.

```bash
set -euo pipefail

SOURCE_ROOT="${PROCOIM_SOURCE_ROOT:-/content/gtrl_procoim_source}"
OUT_ROOT="${PROCOIM_OUT_ROOT:-/content/drive/MyDrive/GTR-L/gtrl_evotoken_relay/procoim_overnight_20260804}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$SOURCE_ROOT"
mkdir -p "$OUT_ROOT"

run_named() {
    local stage=$1
    local steps=$2
    local seed=$3
    local name=$4
    local log_every=500
    local accepted_cap=64
    local -a flags=()

    if (( steps > 10000 )); then
        log_every=5000
    fi

    case "$name" in
        b64)
            ;;
        d16)
            flags+=(--procoim-candidate-directions 16)
            ;;
        d32)
            flags+=(--procoim-candidate-directions 32)
            ;;
        r)
            flags+=(
                --procoim-reuse-successful-directions
                --procoim-direction-momentum 0.9
            )
            ;;
        e)
            flags+=(--procoim-antithetic-estimate)
            ;;
        s)
            flags+=(
                --procoim-adaptive-probe-scale
                --procoim-probe-scale-growth 1.1
                --procoim-maximum-probe-scale 0.1
            )
            ;;
        g)
            flags+=(
                --procoim-adaptive-group-selection
                --procoim-group-score-beta 0.9
                --procoim-group-exploration 0.001
            )
            ;;
        a1)
            flags+=(
                --procoim-anchor-buffer-size 1
                --procoim-anchor-regression-fraction 0.02
                --procoim-anchor-absolute-tolerance 0.0001
                --procoim-anchor-weight 1.0
                --procoim-anchor-hard-protection
            )
            ;;
        re)
            flags+=(
                --procoim-reuse-successful-directions
                --procoim-direction-momentum 0.9
                --procoim-antithetic-estimate
            )
            ;;
        sg)
            flags+=(
                --procoim-adaptive-probe-scale
                --procoim-probe-scale-growth 1.1
                --procoim-maximum-probe-scale 0.1
                --procoim-adaptive-group-selection
                --procoim-group-score-beta 0.9
                --procoim-group-exploration 0.001
            )
            ;;
        resg|resga1)
            flags+=(
                --procoim-reuse-successful-directions
                --procoim-direction-momentum 0.9
                --procoim-antithetic-estimate
                --procoim-adaptive-probe-scale
                --procoim-probe-scale-growth 1.1
                --procoim-maximum-probe-scale 0.1
                --procoim-adaptive-group-selection
                --procoim-group-score-beta 0.9
                --procoim-group-exploration 0.001
            )
            if [[ "$name" == resga1 ]]; then
                flags+=(
                    --procoim-anchor-buffer-size 1
                    --procoim-anchor-regression-fraction 0.02
                    --procoim-anchor-absolute-tolerance 0.0001
                    --procoim-anchor-weight 1.0
                    --procoim-anchor-hard-protection
                )
            fi
            ;;
        *)
            echo "unknown variant: $name" >&2
            return 2
            ;;
    esac

    local out="$OUT_ROOT/${stage}_s${seed}_${name}"
    if [[ -e "$out/train.log" ]]; then
        echo "refusing to overwrite $out/train.log" >&2
        return 2
    fi
    mkdir -p "$out"

    "$PYTHON_BIN" -u gator_linear_prefix_train.py \
        --optimizer procoim \
        --steps "$steps" \
        --batch-size 1 \
        --eval-batch-size 1024 \
        --eval-seed 12345 \
        --length 7 \
        --streams 10 \
        --nodes 40 \
        --width 8 \
        --relays 4 \
        --degree 6 \
        --buses 4 \
        --mailbox-channels 4 \
        --mailbox-width 8 \
        --basis-rank 4 \
        --address-width 4 \
        --router-rank 8 \
        --conv-kernel 3 \
        --graph-layout small_world \
        --graph-seed 0 \
        --graph-rewire 0.5 \
        --device cpu \
        --amp-dtype none \
        --scheduler constant \
        --log-every "$log_every" \
        --checkpoint-dir "$out" \
        --procoim-reduction-fraction 0.005 \
        --procoim-candidate-directions 8 \
        --procoim-antithetic \
        --procoim-max-candidate-banks 16 \
        --procoim-max-accepted-updates "$accepted_cap" \
        --procoim-stability-required 3 \
        --procoim-probe-scale 0.03 \
        --procoim-probe-scale-decay 0.5 \
        --procoim-minimum-probe-scale 0.000001 \
        --no-procoim-reuse-successful-directions \
        --no-procoim-antithetic-estimate \
        --no-procoim-adaptive-probe-scale \
        --no-procoim-adaptive-group-selection \
        --procoim-anchor-buffer-size 0 \
        --seed "$seed" \
        --procoim-seed "$seed" \
        "${flags[@]}" \
        2>&1 | tee "$out/train.log"
}
```

The repeated negative feature flags make the baseline independent of future
default changes. Variant flags occur later on the command line and therefore
enable only the requested feature.

## Stage 1: 10k-forward triage

Run these serially at seed 0:

```bash
run_named triage10k 10000 0 b64
run_named triage10k 10000 0 d16
run_named triage10k 10000 0 d32
run_named triage10k 10000 0 r
run_named triage10k 10000 0 e
run_named triage10k 10000 0 s
run_named triage10k 10000 0 g
run_named triage10k 10000 0 a1
run_named triage10k 10000 0 re
run_named triage10k 10000 0 sg
run_named triage10k 10000 0 resg
run_named triage10k 10000 0 resga1
```

All `step=0` values, including `by_pos`, must be identical. Stop the sweep and
report the mismatch if they are not.

For every row record:

- initial and final validation loss;
- final and best token accuracy;
- all seven `by_pos` accuracies and `by_pos_ce` losses, the minimum accuracy,
  and the mean accuracy and CE of positions 4–6;
- steps/second and runtime;
- accepted updates per 1,000 forwards;
- candidate, anchor, and stability-forward shares;
- hard-veto rate, cumulative anchor vetoes, and stalls;
- cumulative reused-direction and antithetic-estimate accepts;
- per-group acceptance counts, probe scales, and gain-per-forward values; and
- minimum node load, minimum stream weight, and stream agreement.

Use the same-seed `b64` row as the triage baseline. A simple ranking score is:

```text
score = (baseline_loss - variant_loss)
      + 0.50 * (variant_accuracy - baseline_accuracy)
      + 0.25 * (variant_tail3_mean - baseline_tail3_mean)
```

Hard-reject a row if any of these are true:

- it crashes, reports a non-finite value, or does not end at exactly 10,000;
- final loss is more than `0.015` worse than baseline;
- final accuracy is more than `0.015` worse than baseline;
- `min_node < 0.002` or `min_stream < 0.03`;
- adaptive scales remain at `1e-6` for the final two logs without a validation
  advantage; or
- one adaptive group receives over 85% of banks without a validation
  advantage.

Normally reject throughput below 70% of baseline. The `a1` and `resga1` rows
may remain as retention diagnostics below that threshold, but promote an
anchor row only if it earns at least `0.015` lower loss, `0.010` higher
accuracy, or `0.010` higher tail-three mean. Do not test buffers larger than
one tonight.

Promote the top three novel rows with `score >= 0.002`. A row may also qualify
when loss is no more than `0.003` worse and accuracy or tail-three mean is at
least `0.005` better. Verify that each enabled mechanism actually fired:

- `r`: `reuse_accepts_total > 0`, unless it has a clear validation win;
- `e`: `estimate_accepts_total > 0`, unless it has a clear validation win;
- `s`: group scales moved and did not permanently collapse;
- `g`: bank allocation departed from round-robin without starving a group; and
- `a1`: `anchor_forwards` and `anchor_vetoes_total` are nonzero.

## Stage 2: fresh 100k-forward runs

The trainer does not expose CLI resume. Do not continue a 10k checkpoint.
Start every promoted 100k run from a fresh initialization.

Run a fresh same-code seed-0 baseline and the top three novel variants:

```bash
run_named full100k 100000 0 b64
run_named full100k 100000 0 "$WIN1"
run_named full100k 100000 0 "$WIN2"
run_named full100k 100000 0 "$WIN3"
```

Then test seed dependence with a seed-1 baseline and the best two survivors:

```bash
run_named full100k 100000 1 b64
run_named full100k 100000 1 "$WIN1"
run_named full100k 100000 1 "$WIN2"
```

If at least 90 minutes remain, run seed 2 for the current winner:

```bash
run_named full100k 100000 2 "$WIN1"
```

At 25k and 50k, stop a survivor if its score against the corresponding
same-seed baseline curve is below `-0.015` for two consecutive validation
logs, or if persistent scale/group collapse has no validation advantage.

A 100k variant is a win when either:

- loss improves by at least `0.010` and accuracy is no more than `0.003`
  worse; or
- accuracy improves by at least `0.005` and loss is no more than `0.005`
  worse.

Also require wall-clock loss reduction per second to be at least 90% of the
same-seed baseline and the improvement direction to repeat at seed 1. A robust
winner should have a mean two-seed score of at least `0.005`. Report tail-three
movement separately even if aggregate accuracy is flat.

## Failure and checkpoint rules

- Never overwrite an existing run directory.
- The saved `gator_linear_prefix_latest.pt` is an audit checkpoint, not a
  supported CLI resume point in this prototype.
- If Colab disconnects, preserve the partial directory and restart from zero
  under a new name ending in `_retry1`.
- Do not load only the model and optimizer state while omitting anchors. Anchor
  batches are stored separately and require device-aware reconstruction.
- Do not loosen protected-slot or anchor hard tolerances after seeing a
  rejection. That would change the experiment.
- Do not edit the optimizer between rows. If a code defect appears, stop the
  matrix and report the exact command, last complete log line, and traceback.

## Morning report

Provide one table with:

```text
variant | seed | forward steps | final loss | final acc | best acc |
tail3 mean | min position | exact | steps/s | accepted/1k |
anchor share | hard-veto rate | score | decision
```

Then include:

1. the completed legacy U64 and U128 control summaries;
2. the full 10k ranking;
3. each 100k curve at 25k, 50k, and 100k;
4. the final `by_pos` vector for every 100k run;
5. the final `by_pos_ce` vector for every 100k run;
6. which optional mechanisms actually produced accepted updates;
7. whether any result repeated at seed 1; and
8. the recommended next configuration, or a clear conclusion that none beat
   the same-code random antithetic U64 first-hit baseline.
