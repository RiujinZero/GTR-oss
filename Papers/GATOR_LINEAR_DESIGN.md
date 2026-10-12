# GTR-L: linear-scaling Gator

Status: **under validation** (started 2026-07-29, branch `gator_sparse`).
Implementation: `gator_linear/` + `gator_linear_prefix_train.py`.

## Why

`gator_convergent`'s memory scales roughly **quadratically with context**
(measured 2026-07-29: 512 -> 1024 costs 3.4x, 1024 -> 2048 costs 3.7x; OOM at
3072 on a 95 GB card). The bottleneck is `_receiver_kernel()`
(`gator_convergent.py:1213`), which under `--route-backend dense`
materialises an all-pairs `[T, T, n_nodes]` relation tensor once per relay.
That is what caps production context at 620 tokens, and it is a property of
the *operator*, not of precision -- an fp8 readout changed peak memory by
0.001% (see `JOURNAL.md`, 2026-07-29).

## Principle

> Dense hypotheses, linear communication.

Gator-P failed because it made *routes* sparse before useful terrain existed.
The correction is that "all nodes receive gradient" does not require every
node to compare against every other node. Keep every node softly active;
make the machinery that *moves and mixes* that activity linear.

The cell ablations say what must survive: separate hidden states, separate
route distributions, multiple evolving interpretations, communication between
them, and late complementary fusion. Arbitrary all-to-all matrices and
duplicated cell machinery are negotiable.

## What replaces what

| Quadratic operator | Linear replacement | Cost |
|---|---|---|
| receiver compares every writer, `O(L^2)` | grouped conv + prefix/suffix scans + global summary | `O(L N)` |
| every source node scores every destination, `O(N^2)` | fixed degree-`d` graph + `M` low-rank buses | `O(N d + N M)` |
| every stream mixes with every stream, `O(S^2)` | shared `C`-channel node mailboxes | `O(S N)` |
| full transform per node, `O(N D^2)` | rank-`r` shared program basis | `O(r D^2 + N r)` |
| vocab projection per stream, `O(S L V)` | fuse hidden states, then one head | `O(L S + L V)` |

State: `H [B,L,S,D]` private stream states, `Q [B,L,S,N]` soft occupancy,
`A [L,Da]` immutable addresses, mailboxes `[B,L,N,C,Dm]`.

## Measured scaling

CPU, fwd+bwd, prototype config (`streams=10, width=8, relays=4, degree=6`):

| axis | change | time | linear would be | quadratic would be |
|---|---|---|---|---|
| nodes 40 -> 320 | 8x | **8.39x** | 8x | 64x |
| length 16 -> 256 | 16x | **17.29x** | 16x | 256x |
| streams 5 -> 40 | 8x | **3.76x** | 8x | 64x |

The design target is met on all three axes. Sequence length is the important
one: 17x instead of 256x is the entire reason this branch exists.

## Three failure modes found during bring-up

Both produced the *same* symptom -- loss pinned at exactly `ln(12) = 2.4849`,
accuracy at chance, on the prefix task -- and both are worth recording
because neither is obvious from the design document.

### 1. Broad support must not mean *uniform* support

First implementation initialised ingress keys at `std=0.02` with
`ingress_temperature=4.0`, giving occupancy entropy `3.672` against a maximum
of `3.689` -- effectively flat. That is degenerate, not safe: if every node
holds the same content, the graph cannot address anything, and the mailbox
read returns the same context to every stream regardless of route.

The prefix task makes the requirement concrete. Output slot `j` must read the
partial sum over observed slots `1..j`. That is only reachable if observed
slot `i` deposits its message on a *distinguishable* node subset and slot `j`
reads exactly the subset for `i <= j`. Uniform `Q` destroys that mechanism.

Fix: `ingress_temperature=1.0`, keys at `1/sqrt(width)`, stream bias at
`std=0.5` -- entropy `3.12`, every node still far above zero, but slot and
stream occupancy profiles genuinely differ.

The non-negotiable is "no node is starved of gradient", **not** "occupancy is
flat". Those are different constraints and the second one is harmful.

### 2. A shared global bus becomes a homogenising pipe

Second implementation had one shared `bus_in [N,M]` / `bus_out [M,N]` pair
initialised at `std=0.02`, so `softmax(bus_out)` was near-uniform over nodes,
and `bus_gate` sat at `0.726`. Result: ~73% of route mass was replaced by a
near-uniform redistribution at every relay, and occupancy entropy snapped
back to `3.66` even though ingress produced `3.12`. The addressing built at
ingress was being erased by relay 1.

This is exactly the "one giant homogenizing soup pipe" the design warns
about. Three fixes, all in `LinearRelay.reset_parameters`:

* buses are **per stream** (`[S,N,M]` / `[S,M,N]`), so they stay plural and
  specialised instead of applying one shared redistribution to everyone;
* bus factors initialise at `std=1.0`, so each bus moves mass somewhere
  specific rather than smearing it;
* `bus_gate.bias = -2.0`, so the bus starts ~12% open and local transport
  dominates until long-distance movement is actually earned.

A related third guard: `edge_bias[:, 0] = 2.0` favours the self-loop at
initialisation. Repeated *soft* random-walk steps over a fixed graph converge
toward the stationary distribution, which washes out addressing within a
couple of relays; biasing toward staying put lets the router learn movement
deliberately rather than fighting diffusion from step zero.

### 3. The mailbox must be mass-normalised or evidence never arrives

The most severe of the three, and invisible from loss curves alone -- it
looks identical to the other two (chance accuracy, `loss = ln(12)`).

Diagnosis came from a direct causal test rather than more training: perturb
one observed token, measure the change in output log-probabilities. The
answer was **`0.0000` for every (observed slot, output slot) pair**. The
model could not see its own input at all, so "not learning" was never a
tuning problem.

Cause: `node_input = sum_s Q[s,n] * message[s]` with broad `Q` carries
magnitude `~S/N` -- at `S=10, N=40` that is `~0.08` of the private residual
stream. `gator_convergent._node_memory` avoids this by returning
`numerator / denominator`, i.e. dividing by the mass actually deposited at
the node; the first GTR-L implementation dropped that normalisation. Each
relay then attenuated the evidence further, leaving `~1e-4` at the readout.

Three coupled fixes:

* divide the mailbox by `Q.sum(streams)` per node, restoring the
  numerator/denominator contract;
* `LayerNorm` the mailbox context so it is on the same scale as the private
  state it competes with;
* `LayerNorm` the program-basis update. `E_{n~Q}[a_n]` shrinks as
  `1/sqrt(N_effective)` under broad occupancy, so without this the relay
  weakens purely as a function of node count -- which would also have
  silently changed behaviour in the 40 -> 160 node scaling test.

Post-fix sensitivity is `~0.007` per observed slot with gradient reaching
every observed embedding row.

**Process lesson**: a causal sensitivity probe (perturb input, measure output
delta) diagnosed in one run what additional training runs could not have
distinguished from a bad learning rate. Reach for it before tuning whenever a
new architecture sits exactly at chance.

### 4. A plain cumulative scan cannot address by distance

Not a coding bug -- a misreading of the design that cost real expressivity.
"Forward scan" was first implemented as a plain cumulative mean. That weights
every earlier slot identically, so a receiver cannot distinguish slot `l-1`
from slot `l-30`, and *all* addressing burden falls on the node dimension.
The reference model gets distance selectivity for free from its `[L, L, N]`
relation kernel; dropping to a uniform prefix silently gave that up.

`decayed_scan()` restores it: `h[l] = sum_{l'<=l} decay^(l-l') x[l']` with a
learned per-channel decay initialised across a geometric spread, so short-
range channels resolve neighbours and long-range channels approximate a
running total. It is evaluated in fixed-size chunks, so the transient is
`[chunk, chunk, C]` rather than `[L, L, C]` and the cost stays `O(L)`.
Verified exact against the naive recurrence and, importantly,
**chunk-size independent** -- that equivalence is what makes the linear
blocking legitimate rather than a hidden quadratic.

## Capacity is confirmed; the architecture is not the problem

An overfit probe on 16 fixed examples reaches **loss 0.0002 / accuracy 1.000
by step 300**. Evidence reaches the readout, the readout can express the
target, and gradients are healthy. Any remaining failure to reach 99% on
fresh batches is therefore a *training-dynamics* question (schedule, budget,
routing temperature), not a structural one.

This also corrects an error of process on my part: several "still at chance"
conclusions were drawn from 600-1500 step CPU runs, when the reference itself
needs ~20-30k steps to reach mastery on this task. Short runs could not have
shown success even if the model were perfect.

## Validation plan

1. **10k prefix** (this gate): match the existing Gator-Path result --
   escape chance at a comparable step, reach >=99% token accuracy, no dead
   streams or nodes, linear 40 -> 80 -> 160 node scaling.
2. **130k Gator**: language, same acceptance shape.
3. **1M Gator**: language, plus a real context-length memory curve against
   `gator_convergent` to confirm the quadratic term is actually gone.

### 5. Receptive field: `conv_kernel=3` cannot reach the target

The prefix target needs a **step at a fixed relative distance**. With 8
observed slots, output slot `j` sits at `p = 8 + j` and needs observed
`i <= j + 1`, so the required writers lie at distance
`d = p - i` in `[7, 8 + j]`. The lower bound is `7` for *every* output slot,
so the kernel the task wants is a single fixed step at `d >= 7`.

Now count the reach actually available. A depthwise convolution of width `k`
reaches `+/-(k-1)/2` per relay, so `R` relays compose to about
`R * (k-1) / 2`:

| `conv_kernel` | reach per relay | reach over 4 relays | covers `d >= 7`? |
|---|---|---|---|
| 3 | 1 | 4 | **no** |
| 9 | 4 | 16 | yes |
| 15 | 7 | 28 | yes |

`conv_kernel=3` therefore cannot construct the required kernel *at any
relay depth used here*. The only operators with longer reach are the decayed
scans, and those are monotonic in distance: representing a step requires
approximating it as `K(d) = sum_c w_c * decay_c^d`, an exponential-sum
approximation to a discontinuity that needs large alternating weights and is
badly conditioned.

This reframes the earlier collapse. Occupancy flattening and stream collapse
are real and were correctly diagnosed, but they may be *consequences* rather
than the root cause: if no setting of the parameters can express the target,
gradient descent settles on the marginal, and an unused routing mechanism
has no pressure keeping it structured.

Widening the convolution is squarely within the design ("grouped
convolution") and stays `O(L * k)` with `k` a fixed hyperparameter, in the
same sense as an SSM's convolution width. It only fails to be linear if `k`
must grow with `L`, which is a property of the *task's* dependency range,
not of the operator.

## Diagnostics that earned their keep

Three probes did more than any amount of retraining, and are worth reaching
for first on any new architecture that sits at chance:

1. **Causal sensitivity** -- perturb one input token, measure the mean change
   in output log-probabilities. Reading exactly `0.0000` proved the model
   could not see its own input, which is indistinguishable from a bad
   learning rate if you only watch loss curves.
2. **Overfit a tiny fixed batch** -- if 16 examples cannot be driven to zero
   loss, the blocker is structural. Reaching `1.000` accuracy by step 300
   settled that the design is sound and moved the question to schedules.
3. **Scan-vs-naive equivalence at several chunk sizes** -- confirms a
   blocked/chunked linear kernel is actually computing the intended
   recurrence and is not silently quadratic.

## Acceptance criteria (gate 1)

* escapes chance at a similar training step to the reference;
* >= 99% token accuracy on simultaneous prefix;
* retains primary/corrector fusion synergy (no single stream dominating);
* no dead streams or nodes early in training;
* memory and runtime scale approximately linearly 40 -> 80 -> 160 nodes.

## Non-negotiables

Do not: hard top-1 route at initialisation; force low route support;
supervise every stream as a complete solver; prune on final vote weight
alone; average private hidden states.
