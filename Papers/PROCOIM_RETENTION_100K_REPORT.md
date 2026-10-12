# Procoim Cross-Example Retention: 100k Report

Date: 2026-08-05

## Question

Can a 0.01% current-example search target plus a relaxed 0.005%
cross-example audit floor prevent Procoim from overwriting prior progress and
turn its early four-slot accuracy increase into continued learning?

## Matched configuration

```text
model parameters:             46,701
streams / nodes / width:      10 / 40 / 8
relays / output slots:        4 / 4
routing:                      hard
scan normalization:           disabled
temperature:                  fixed at 1.0
loss vocabulary:              full
batch size:                   1
optimizer policy:             directional Adam
direction count:              8
directional Adam LR:          0.03
probe scale:                  0.03
episode reduction target:     0.00010 (0.01%)
anchor retained floor:        0.00005 (0.005%)
anchor capacity:              64
inner-loop sentinels:         0
provisional window:           16 examples
maximum accepted updates:     64 per episode
seed:                         0
budget:                       100,000 model forwards
```

Every model forward used by Procoim counts as one training step. Validation
forwards are reported separately.

## Audit correction

The first implementation applied the 0.005% floor independently to every
protected historical slot. That remained an impossible intersection:

- A 10k run accepted 175 provisional updates but committed zero of six
  windows and returned exactly to its initial model.
- A diagnostic tried commit fractions down through `alpha=0.0078125`; no
  nonzero scalar interpolation preserved every slot's individual floor.

The corrected audit treats the requested floor as net retained progress over
the protected anchor bank:

```text
retained_reduction =
    sum(start_loss - audited_loss) / sum(abs(start_loss))
```

An audited window is eligible only when:

```text
retained_reduction >= 0.00005
```

Individual examples may therefore redistribute a small amount of progress.
A separate catastrophe bound still vetoes a hard anchor slot above:

```text
start_loss
    + anchor_hard_regression_fraction * abs(start_loss)
    + anchor_absolute_tolerance
```

This distinction matters. Before the catastrophe correction, the first bank
retained 0.5059% net improvement—more than 100 times the requested floor—but
was still rejected by the old tolerance tied to each tiny saved gain.

## Validation before the long run

The corrected 10k run produced:

```text
full windows committed:       4
full windows audit-rejected:  0
commit alphas:                1.0, 1.0, 0.5, 1.0
best accuracy:                0.0879
final accuracy:               0.0791
throughput:                   60.1 forwards/s
```

The only rollback was an unfinished eight-example provisional window at the
global forward budget, not an audit veto.

## 100k result

| Forward step | Accuracy | Eval CE | Valid-vocab CE | Committed | Rejected |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.064 | 2.7730 | 2.5316 | 0 | 0 |
| 5,019 | 0.088 | 2.7642 | 2.5314 | 2 | 0 |
| 20,013 | 0.087 | 2.7424 | 2.5261 | 7 | 1 |
| 40,167 | 0.086 | 2.7390 | 2.5221 | 11 | 4 |
| 60,142 | 0.083 | 2.7376 | 2.5196 | 17 | 4 |
| 85,216 | 0.085 | 2.7193 | 2.5210 | 25 | 6 |
| 95,063 | 0.085 | 2.7211 | 2.5219 | 28 | 8 |
| 100,000 | 0.084 | 2.7227 | 2.5226 | 28 | 9 |

After restoring the last trusted base from the unfinished final window:

```text
final accuracy:               0.0830
best accuracy:                0.0879
exact formation accuracy:     0.0000
escaped chance at:            never
final eval CE:                2.7212
final valid-vocabulary CE:    2.5219
final by-position accuracy:   [0.064, 0.086, 0.096, 0.086]
accepted updates:             1,080
episodes:                     607
runtime:                      1,519.91 seconds
throughput:                   65.79 forwards/s
```

## Verdict

Cross-example retention now works mechanically. Useful provisional windows
are committed, destructive windows are rejected, partial commit alphas are
used, and the model no longer resets to its initial weights after every
window.

The learning breakout did not survive the 100k test. Accuracy finished at
approximately the 1/12 chance rate, exact match remained zero, and the best
accuracy was only 8.79%. The CE reduction from 2.7730 to 2.7212 is real, but
the update policy did not turn that probability-margin improvement into
correct argmax classes.

The current bottleneck is therefore no longer cross-example overwriting. It is
the geometry and credit assignment of the proposed update direction. The next
diagnostic should trace a matched AdamW run every 500 steps, preserving full
small-model weights, interval deltas, gradients seen by AdamW, and first/second
moments. Comparing those vectors with Procoim's directional basis should show
which mathematical structure the perturbation policy is missing.

## Artifacts

```text
stdout log:
C:\tmp\procoim_bankaudit_cat10_100k.stdout.log

checkpoint:
C:\Users\user\AppData\Local\Temp\gtrl_procoim_fix_20260805\
slot4_dadam_k8_r0001_bankaudit00005_cat10_c16_seed0_100k\
checkpoint\gator_linear_prefix_latest.pt
```
