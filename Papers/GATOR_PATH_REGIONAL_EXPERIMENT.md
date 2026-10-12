# Gator-Path Regional Development Experiment

## Question

Can a ten-cell Gator-Path avoid cold sparse routes by developing useful
regions before asking an orchestrator to compose them?

The ten cells were divided into contiguous regions. Skill packets were trained
with regional access masks before the bridges were opened for the length-seven
parallel-prefix task.

## Skill definitions

The first proof deliberately uses small, exact subtasks:

- **Number:** convert raw delta token `d` into canonical number token `d + 4`.
- **Arithmetic:** compute `(canonical_value + raw_delta) mod 12`.
- **Orchestrator:** receive skill packets first and learn which regional bridge
  should handle their work.

These do not individually solve parallel prefix. They only establish whether
reachable regions can offer useful transformations instead of random,
untrained machinery.

## Developmental topology

The best tested allocation was:

| Region | Cells | Role |
|---|---:|---|
| Number | 1 | Raw-to-canonical number representation |
| Arithmetic | 7 | Modular binary operation |
| Orchestrator | 2 | Initial packet stewardship and regional handoff |

During skill rehearsal, a task-specific mask activates only:

```text
orchestrator + number
```

or:

```text
orchestrator + arithmetic
```

The inactive skill region receives no packets and executes no selected cell
transform. Number and arithmetic batches are shuffled into one interleaved
training stage to prevent shared router and readout state from forgetting the
earlier skill.

During integration, both orchestrator bridges are opened while direct
number-to-arithmetic access remains prohibited.

## Local results

All values below use fresh generated evaluation batches. Random token accuracy
is `1 / 12 = 8.33%`.

| Configuration | Number | Arithmetic | Prefix observation |
|---|---:|---:|---|
| 4/4/2, sequential, isolated skills | 100% at 300 steps | 9.8% at 300 steps | Stayed near chance |
| 2/6/2, sequential, stage optimizers | 54.9% retained | 53.3% retained | Stayed near chance |
| 1/7/2, sequential, orchestrator participates | 6.6% retained | 46.3% retained | Number skill was overwritten |
| 1/7/2, interleaved, both bridges open | 55.5% retained | 27.5% retained | Prefix peaked around 9.2% |
| **1/7/2, interleaved, task-active regions** | **100% retained** | **41.8% retained** | Prefix peaked around 9.6% |

The strongest regional run used:

```text
number updates:       400
arithmetic updates:  1200
skill learning rate: 0.003
route support:        annealed 6 → 2
relays:               4
payload width:        8
parameters:           7,964
```

At its best regional checkpoint:

```text
number accuracy:       1.000
arithmetic accuracy:   0.418
number cross-region:   0.099
arithmetic cross:      0.490
```

This proves the immediate hypothesis: a ten-cell Path can contain multiple
nontrivial, simultaneously retained regional skills. Its routes no longer
lead only to barren random transforms.

## Negative integration result

Opening both bridges and training the length-seven prefix objective for
1,500–2,000 local steps did not produce compositional transfer. Accuracy
remained statistically near chance even when:

- the orchestrator participated in both skill regimes;
- skill batches were interleaved;
- only the relevant region was active for each skill batch;
- prefix packets began in the orchestrator;
- broad route support was annealed from six paths to two; and
- each developmental stage restored its best checkpoint.

Regional competence and regional composition are therefore separate gates.
The former now passes; the latter does not.

## Mandatory orchestrator return

The first regional masks prohibited direct number-to-arithmetic edges, but
they did not guarantee that a specialist returned packet ownership. The
protocol is now relay-phased and directional:

```text
orchestrator dispatch
    → specialist-local compute
    → distance-reducing return
    → orchestrator consolidation
```

For a composed number-then-arithmetic call, at least ten relays are required:

```text
0  number dispatch
1  number compute
2  number return
3  orchestrator consolidation
4  arithmetic dispatch
5  arithmetic compute
6  arithmetic return step 1
7  arithmetic return step 2
8  arithmetic return step 3
9  orchestrator consolidation
```

The extra arithmetic return windows cover the worst legal two-cell inward
move made during its compute relay. Tests assert that every packet finishes in
orchestrator cells 8–9 and that no relay opens a specialist-to-specialist
shortcut.

An eight-relay skill-only run retained:

```text
number accuracy:       0.490
arithmetic accuracy:   0.381
skill route crossings: 0.250
```

`0.250` is exactly two ownership crossings over eight relays: dispatch and
return. A ten-relay composed integration produces a stable crossing fraction
of `0.400`, corresponding to the four required boundary crossings for two
calls. It remained near chance on full length-seven prefix after 500
integration steps. The ownership protocol works; repeated scan composition
remains unlearned.

## Learned dispatch result

The relay-phased call schedule was useful scaffolding but did not test
orchestrator judgment because it named the called region in advance. Learned
dispatch replaces it with one static ownership law:

```text
orchestrator packet:
    stay, call number, or call arithmetic

specialist packet:
    work locally or return to orchestrator

forbidden:
    specialist → different specialist
```

A small late-relay distance loss rewards packets for returning home without
specifying which region to call or when to call it.

Portable specialists were trained with packets already at callable boundary
interfaces. Four-relay isolated pretraining produced:

```text
number region:      1.000
arithmetic region:  0.469
cross-region work:  0.000
```

Those transformations were frozen. Only router heads and orchestrator-cell
transforms adapted during six-relay dispatch rehearsal. The best result was:

```text
number accuracy through dispatch:      0.604
arithmetic accuracy through dispatch:  0.252
number-task calls to number region:     0.055
arithmetic-task calls to arithmetic:    0.817
```

The orchestrator learned nontrivial dispatch and preserved some specialist
capability, but it did not learn a balanced region choice. It mostly called
the seven-cell arithmetic region, even for number tasks.

This reveals a topology-cardinality bias. The current router chooses local
cell actions directly. From orchestrator boundary cells, the seven-cell region
offers more legal offsets than the one-cell number region, so it receives more
initial probability mass and more gradient opportunities.

The next dispatcher should be hierarchical:

```text
choose region: [stay, number, arithmetic]
        ↓
choose legal cell/node inside selected region
```

Region probability must be normalized independently of the number of cells or
edges inside that region. This is an architectural correction, not a
task-specific route label.

## Gate-cell dispatch result

Gate dispatch implements that correction without adding a dense region-sized
router. Each specialist region exposes exactly one public entrance:

```text
orchestrator action 3 -> number gate
orchestrator action 4 -> arithmetic gate

specialist actions 0-3 -> local work
specialist action 4    -> orchestrator return
```

The orchestrator therefore chooses between two skill identities, not between
one number cell and seven arithmetic cells. Changing a region's internal size
does not change the number of public call actions. Once a gate receives a
packet, private regional routing decides which internal cell should steward
it. Any specialist may return directly to the orchestrator; specialists still
cannot transfer directly between regions.

Using the same portable 1/7/2 checkpoint and six relays, a clean 1,600-step
router-only rehearsal with frozen specialist transforms produced:

```text
number accuracy through dispatch:      0.457
arithmetic accuracy through dispatch:  0.301
number-task calls to number gate:       0.531
arithmetic-task calls to arithmetic:    0.599
```

The comparable ungated frozen-specialist run reached 0.604 number accuracy,
0.252 arithmetic accuracy, called number correctly on only 0.055 of
number-task calls, and called arithmetic correctly on 0.817 of arithmetic
calls. Gate dispatch therefore traded some number-task accuracy for arithmetic
accuracy while raising the harmonic skill score slightly from 0.353 to 0.363.
More importantly, it removed the overwhelming region-cardinality preference
and made the two call rates substantially more balanced.

Repurposing specialist action 4 as a universal return changes an edge type
that the portable regions had learned under the original ring. The frozen
control therefore measures both the macro-routing correction and that
interface mismatch. When specialist and orchestrator transforms were allowed
to adapt, the same gate curriculum reached:

```text
number accuracy through dispatch:      0.746
arithmetic accuracy through dispatch:  0.346
number-task calls to number gate:       0.908
arithmetic-task calls to arithmetic:    0.402
```

This adaptive result is not a topology-only A/B, but it shows that both skills
can exceed the ungated routed accuracies after learning the gate interface.
Arithmetic selection is not yet reliable, so learned macro-dispatch is
promising rather than solved. Full seven-output prefix remained near chance
after 500 integration steps, as expected before repeated regional composition
is learned.

If cold gate routing remains difficult, the fallback curriculum is dense
exploration followed by gate contraction: begin with broad legal support,
measure successful paths, and anneal toward the two gate actions only after
route signal materializes. This matches Arrow stewardship behavior observed
in a live 62-cell, 500k-step language run: active cell use contracted to about
45 cells and briefly reached 33. This is an encouraging training observation,
not yet a controlled Path result.

## 20k dense-to-sparse regional curriculum

Simply stretching the broad-support schedule to 20,000 steps was unstable.
Top-6 runs stopped on non-finite gradients after approximately 2,300 updates
at both `0.003` and `0.001` learning rates. Removing the return auxiliary
delayed failure to approximately 4,500 updates but did not solve it.

The successful curriculum separated exploration from refinement:

```text
phase 1:  1,600 broad-support gate warmup
phase 2: 18,400 fixed-top-2 continuation
```

The sparse continuation immediately improved routed arithmetic from 34.6% to
51.4% within 500 updates. At 2,000 sparse updates it reached 84.6% number and
73.0% arithmetic. The complete 20,000-update curriculum retained:

```text
number accuracy through dispatch:      0.920
arithmetic accuracy through dispatch:  0.932
harmonic skill accuracy:                0.926
best individual number accuracy:       0.926
best individual arithmetic accuracy:   0.934
```

Long sparse training still accumulated unstable optimizer moments. A bounded
recovery policy restored the best fresh-data checkpoint, cleared Adam state,
and continued. Four recoveries occurred in the final 15,245-update segment.
This preserved the best joint state instead of ending or corrupting the run.

The traffic diagnostics are not per-example route correctness. They measure
the share of all orchestrator-to-specialist crossings that entered a named
gate across every packet and relay. High arithmetic accuracy sometimes
coexisted with less than half of call traffic entering the arithmetic gate,
which indicates multi-region, multi-handoff routes rather than a single
mixture-of-experts choice.

A 500-step full-prefix probe from the final regional checkpoint remained at
chance (`0.59/7`, 8.4%). Regional competence is now solved for this experiment;
repeated orchestration remains a separate learning problem. The next run gives
the full prefix objective its own 20,000-update budget from the retained
92.0%/93.2% regional checkpoint.

## Global-Adam orchestration control

The dedicated prefix run used six relays, fixed top-2 support, AdamW over all
parameters, and the retained 92.0%/93.2% regional checkpoint. It did not match
the dense Arrow phase transition:

```text
step 5,000:   accuracy 0.085, loss 2.4870, final ownership 0.518
step 10,000:  accuracy 0.087, loss 2.4883, final ownership 0.602
step 10,500:  accuracy 0.103, loss 2.4746, final ownership 0.796
step 15,000:  accuracy 0.151, loss 2.3295, final ownership 0.768
step 17,000:  accuracy 0.169, loss 2.2803, final ownership 0.764
```

The run was stopped at 17,000 updates after establishing the curve. Exact
sequence accuracy remained zero. Learning began only after final orchestrator
ownership crossed approximately 0.75 near step 10,500. Before that threshold,
loss converged almost exactly to the uniform twelve-class value `ln(12)`.

This supports an optimizer/path mismatch. Dense Arrow evaluates nearly every
cell every update, so Adam moments describe a relatively stationary
computation. Path routing changes which parameters own a token. Gradients
collected before a routing change can become stale immediately afterward:
there is no meaningful average of two different discrete computational paths.

The next controlled A/B uses:

```text
cell and node router heads:  SGD, zero momentum
all token transformations:  AdamW
```

The split optimizer keeps independent state and can reset routing without
erasing gradual transformation history. A later path-stability curriculum
should alternate exploration and committed transformation windows rather than
allowing successful paths to thrash every update.

## Fixed split-optimizer result and ownership schedule

The fixed-rate split optimizer was smoother than global AdamW but plateaued.
With route SGD at `0.01`, final orchestrator ownership settled near `0.50`.
One route-SGD `0.1` run moved ownership above `0.70` near step 7,500 and then
stalled near `0.75`, but subsequent attempts did not reproduce that curve.
The `0.1` trajectory is therefore an unconfirmed outlier, not evidence that a
larger fixed route rate reliably improves ownership. No fixed split setting
solved the prefix task. The defensible result is narrower: split optimization
in its current form plateaus, and route-rate sensitivity remains unresolved.

The next A/B therefore uses an ownership-conditioned route rate. Zero-momentum
SGD starts at `0.1` while the route is uncertain, then decays log-linearly to
`0.01` between 50% and 80% final orchestrator ownership. The schedule consumes
the maximum ownership observed so far, making it one-way despite noisy
evaluation:

```text
ownership <= 0.50: route LR 0.100
ownership =  0.60: route LR 0.046
ownership =  0.70: route LR 0.022
ownership =  0.75: route LR 0.015
ownership >= 0.80: route LR 0.010
```

AdamW for token representations and transforms remains at `0.001`. This is a
stability intervention, not a success claim; prefix accuracy and loss remain
the deciding metrics.

### Decision sequence after the graduated-rate probe

Change one training assumption at a time:

1. Run the ownership-conditioned schedule from the retained regional
   checkpoint. Require reproducible improvement across seeds; do not promote a
   single high-ownership trajectory.
2. If it plateaus, retrain the number and arithmetic regions from random
   initialization with split optimization active from the first update. This
   tests whether switching optimizer regimes after regional learning creates
   incompatible route and transformation histories.
3. If scratch split training also fails, adopt broad dense-support development
   as the default bootstrap. Contract support only after routes acquire useful
   signal, and allow unfamiliar inputs to reopen broader support.

The third outcome is not a rejection of sparse Gator inference. It means
sparsity is developmental: early training pays for broad exploration, while
established paths progressively reduce active support. The contraction should
follow route confidence or demonstrated path success rather than wall-clock
step alone. Dense execution must not remain the permanent deployed behavior.

Likely remaining issues:

1. Four relays may be insufficient for dispatch, skill execution, return, and
   a logarithmic parallel scan.
2. Prefix cross-entropy does not explicitly preserve the pretrained skill
   behavior while bridge routes are being learned.
3. Shared token, mailbox, and readout projections still permit interference
   between otherwise private cell banks.
4. The current packet state has no explicit call/return or destination-role
   marker for a regional handoff.
5. A binary arithmetic skill is not automatically a parallel-scan operator.
6. One number call and one arithmetic call do not provide the repeated or
   tree-structured calls required to form seven prefix outputs.

## Compute interpretation

Inactive regions do not execute selected packet transforms. Current Path
forward cost is already approximately proportional to active packets,
selected support, and relay count—not total cell population.

The local skill stages ran around 10–11 steps/s while prefix integration ran
around 6–7 steps/s, but skill examples contain only two or three packets while
prefix examples contain fifteen. This is not yet a controlled proof that
regional masking alone increases throughput.

The remaining dense cost is optimizer traversal over large parameter banks.
At scale, regional parameters and optimizer states should be physically
sharded so an inactive region incurs neither forward work nor optimizer work.

## Next composition gate

Before returning to full prefix formation, train a direct two-region handoff:

```text
raw delta
    → orchestrator
    → number region
    → canonical value
    → orchestrator
    → arithmetic region with accumulator
    → canonical sum
```

Require fresh-data accuracy above both isolated regions and verify the expected
bridge sequence. Once a single learned call/return chain works, expand it to a
two-output scan and then lengths four and seven.
