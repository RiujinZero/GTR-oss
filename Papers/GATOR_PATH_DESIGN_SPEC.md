# Gator-Path / GTR-P Design Specification

**Status:** Active experimental architecture  
**Family:** Graph Trace Relay / Gator  
**Baseline preserved at:** Arrow commit `187ed97`  
**Purpose:** Replace replicated all-cell refinement with explicit packet paths  
**Date:** July 18, 2026

## 1. Decision

Gator-Path moves ahead of Arrow-specific performance optimization.

Current cellular Arrow proves that persistent graph state, address anchoring,
controlled writing, recurrent routing, and simultaneous formation can learn
useful algorithms. It does not yet execute the original GTR thesis directly:
each slot is still represented in every cell, receiver memory is rebuilt
densely, every destination node is evaluated, and sparse responsibility is
learned inside a largely dense execution graph.

Gator-Path changes the unit of computation from:

```text
reconstruct every slot across the graph
```

to:

```text
move every slot through the graph
```

Each context token or output slot is one persistent packet identity. It has an
immutable formation address, mutable payload state, and a weighted frontier of
explicit graph stewards. Every relay reads local traffic, lets each active
steward select legal local handoffs, applies selected receiving transforms, and
commits all packet movements synchronously.

The frontier is intentionally elastic:

```text
uncertain / novel packet -> explore several local stewards
familiar packet          -> use a narrow specialized coalition
```

Multiple surviving branches may be complementary computations, not competing
guesses. They can remain separate, merge when they meet, or jointly contribute
to the final packet state. Hard top-1 is an inference endpoint the model may
earn, not a defining requirement imposed at initialization.

Branch payload separation is essential. A soft node passport paired with one
shared hidden vector is still one semantic lane: node transforms are averaged
back into the same payload after every relay. Successful ten-cell Arrow
supplied ten persistent lanes because every cell retained its own `H_c`. Path
must preserve that functional width directly as branch records rather than
relying on permanent cell replication.

A complete route is never a learned or stored object. It emerges from repeated
local policies:

```text
cell A: recognize, transform, hand to B
cell B: recognize the changed packet, transform, hand to C
cell C: recognize, transform, hand to D
```

The sequence `A -> B -> C -> D` may be logged after the fact, but no parameter,
cache, ranker, or route memory represents it. The trace is an observation of
cell behavior, not model state.

Arrow remains intact as the validated reference until Path reproduces the
parallel-prefix result.

## 2. Architectural contract

A model is Gator-Path only when all of these are true:

1. A slot has one persistent packet identity, not an obligatory permanent
   replica in every graph site.
2. Every packet branch has an explicit graph location and ownership mass.
3. Every relay dynamically selects legal local next hops.
4. Forward execution applies only the currently retained frontier transforms;
   dense exploration is allowed while that frontier is still broad.
5. Packet interaction is local to occupied graph locations.
6. All reads use the previous relay snapshot.
7. All moves commit simultaneously.
8. Immutable output addresses remain separate from mutable payload.
9. All output positions decode simultaneously after the final relay.
10. No predicted output token is fed back during the same formation pass.
11. A bounded weighted set of complementary packet branches may survive each
    relay; top-1 is not required.
12. Transformation and handoff policies train jointly under one task
    optimizer in the reference proof.
13. Routing uncertainty may widen support for unfamiliar work and learned
    confidence may narrow support for familiar work.
14. No parameter, cache, or optimizer state stores or ranks complete routes.

Dense `[B,C,L,D]` execution may serve as a training oracle while support is
broad, but it is not the intended mature execution form. The release gate is
that learned inactive support can be omitted without changing formation
quality materially.

## 3. Core state

For batch `B`, packet count `L`, payload width `D`, and address width `Da`:

| State | Shape | Meaning |
|---|---:|---|
| address | `[L,Da]` | Immutable formation coordinate shared by branches of a slot |
| branch payload | `[P,D]` | Mutable state of each retained packet branch |
| packet id | `[P]` | Batch/slot identity tying branches to one logical packet |
| site | `[P]` | Current flat or hierarchical graph steward |
| mass | `[P]` | Ownership/contribution weight |
| active | `[P]` | Valid retained-branch mask |
| write gate | `[P]` | Mailbox contribution |

The optimized mature engine must not expand a narrow learned frontier over the
total graph population. Early training may evaluate a broad or fully dense set
of local alternatives to discover useful coalitions. Those alternatives are
exploration activations, not persistent packet branches, and are discarded
before the next relay.

## 4. Graph representation

A graph site may be represented directly by one flat `site_id` or factored as
`(cell_id, node_id)`. The factorization is an execution and parameter-sharing
choice; local handoff behavior is the architectural requirement.

The first topology uses:

- a ring of cells,
- four cell neighbors plus self,
- four internal nodes,
- the validated Arrow node adjacency,
- factorized cell and node actions.

For degree four, legal cell offsets are:

```text
[0, -1, +1, -2, +2]
```

The node action table is conditioned on the current node and contains only
legal destinations. The optimized router does not score impossible edges and
does not score every cell.

Factorized routing is the first implementation:

```text
cell action × legal destination-node action
```

Joint action identifiers remain available for dispatch grouping and trace
encoding.

## 5. Packet initialization

Context packets receive:

- lexical/task input payload,
- immutable sequence address,
- a learned deterministic ingress decision,
- full write permission.

Output packets receive:

- blank/query payload,
- immutable output address,
- a learned deterministic ingress decision,
- the validated controlled-write schedule.

Ingress begins as a genuine dense differentiable current location over the
small proof population. Plausible first stewards therefore receive real packet
mass and participate in downstream computation. Temperature and learned local
policies sharpen this distribution until the same familiar packet shape
selects the same first steward. At larger population sizes ingress becomes
hierarchical or local; it must not require permanent global all-cell compute.

## 6. Relay semantics

The current proof carries only a packet's present frontier, never its route
history. For packet `i`, current location `x`, and local destination `y`:

\[
m_{i,x}^r = Read(h_{i,x}^r, a_i, Mailbox(x))
\]

\[
p_{i,x\rightarrow y}^r = Route_x(h_{i,x}^r,m_{i,x}^r,a_i)
\]

\[
\tilde h_{i,x}^r =
Norm(h_{i,x}^r + Transform_x(h_{i,x}^r,m_{i,x}^r,a_i))
\]

\[
q_{i,y}^{r+1} = \sum_x q_{i,x}^r p_{i,x\rightarrow y}^r
\]

\[
h_{i,y}^{r+1} =
\frac{\sum_x q_{i,x}^r p_{i,x\rightarrow y}^r\tilde h_{i,x}^r}
{\epsilon+q_{i,y}^{r+1}}
\]

Each relay discards the previous handoff after committing this new current
frontier. The following single-location form describes the eventual hard
deployment limit, where each `q` has annealed to one steward:

Every relay consumes one immutable snapshot:

```text
snapshot packets
    → aggregate occupied-location mailboxes
    → read current local mailbox
    → score legal local actions
    → select top-k paths
    → apply selected path transforms
    → update payload/location
    → commit all packets simultaneously
```

For packet `i`:

\[
m_i^r = Read(h_i^r, a_i, Mailbox(\ell_i^r))
\]

\[
e_i^r = Route(h_i^r, m_i^r, a_i, \ell_i^r)
\]

\[
h_i^{r+1} = Norm(h_i^r + Transform_{e_i^r}(h_i^r,m_i^r,a_i))
\]

\[
\ell_i^{r+1} = Destination(\ell_i^r,e_i^r)
\]

The packet arriving at a destination carries its updated payload directly into
the next relay and reads that destination's traffic there. It is never rebuilt
from token embeddings or reconstructed across cells. Only the derived sparse
mailbox aggregation is refreshed because packet occupancy changed.

## 7. Sparse local mailboxes

Packets interact because their paths meet.

The baseline mailbox stores, per occupied `(batch, cell, node)` location:

- total write mass,
- weighted payload sum,
- weighted address sum,
- optional separate context/output sums.

The read is receiver-specific through a small function of the packet payload,
address, mailbox means, and mass. It must not construct receiver-by-writer
`L×L` tensors.

### Storage correction

A dense correctness engine may materialize:

```text
[B,C,N,Dm]
```

The scalable engine must store only occupied locations:

```text
O(B×L×K×Dm)
```

using bounded segmented reduction, a fixed-capacity hash workspace, or sorted
location keys. Mailbox allocation may not grow with all cells.

Controlled writing initially remains:

```text
context packets: 1
output relay 0: 0
output relay 1: 1/3
output relay 2: 2/3
output relay 3+: 1
padding: 0
```

## 8. Routing and novelty

Exploration is local, bounded, and ephemeral.

Initial proof training evaluates the complete small legal action set of every
current steward carrying nonzero packet mass. These alternatives participate
in the forward pass, so downstream task loss can distinguish what each
candidate cell actually does. As local weights specialize and temperature
falls, the current frontier sharpens around useful handoffs. Only the resulting
current location/payload frontier survives into the next relay; the previous
handoff and every complete route history are discarded.

Required controls:

- bounded local candidate count,
- route temperature,
- optional straight-through top-1 deployment routing,
- uncertainty threshold or schedule,
- deterministic tie behavior,
- optional load balance disabled by default.

The first ten-cell proof uses one AdamW optimizer over both routing and
transformation weights, ordinary prefix cross-entropy, and no ownership loss.
A separate routing optimizer, explicit path objective, or route regularizer is
added only for a measured pathology.

## 9. Selected path computation

The route is chosen before expensive computation.

The first transform belongs to the current steward. The selected handoff is an
input to that transform; the destination cell processes the carried result on
the following relay:

```text
small edge/offset adapter
    → current-steward transform conditioned on chosen handoff
    → residual packet update
```

One parameter-efficient form is:

\[
\Delta h =
F_{\text{current cell,current node}}\left(
h + W_m m + W_a a + E_{cell\ action} + E_{node\ edge}
\right)
\]

Deployment evaluates one selected transform. Development may evaluate the
bounded local candidates only to construct a surrogate backward signal; its
forward value is exactly the current steward's selected handoff transform. The engine must
never evaluate every cell in the population.

Normalization begins with Arrow-compatible behavior, then receives a separate
performance/quality ablation because width-8 normalization can be expensive.

## 10. Trace policy

Trace is diagnostic-only. A route is the observed sequence of local steward
choices, not a learned representation and not an input to later routers or the
decoder. Full route histories are retained only when explicitly requested for
evaluation; ordinary training carries the packet state without accumulating
diagnostic route/mailbox objects.

## 11. Simultaneous readout

Only output packets are decoded:

\[
r_i=h_i^R+W_a a_i
\]

There is no global vote over replicated cell states because each output is one
persistent packet.

## 12. Reference and sparse engines

### Reference engine

Purpose:

- readable equations,
- correctness,
- gradient inspection,
- deterministic tests.

It may use dense mailbox tables and evaluate the small legal local action set,
but it may not replicate packet state across all cells.

### Sparse engine

One relay should become a small number of large operations:

```text
location-key mailbox reduction
router + bounded top-k
pack by selected cell/node/transform
grouped selected transforms
scatter and synchronous commit
```

It must not materialize dense cell or destination tensors before packing.

Given identical selected routes, parameters, and input packets, reference and
sparse engines must agree on:

- payloads,
- locations,
- logits,
- optional trace,
- parameter gradients,
- input-payload gradients.

## 13. Scaling contract

Path removes Arrow's primary activation factor:

```text
Arrow mutable state: [B,C,L,D]
Path mutable state:  [B,L,K,D]
```

Increasing total cells may increase stored parameters and topology metadata,
but ordinary packet activations and mailbox work must depend on occupied route
count, not total population.

This is necessary but not sufficient for a 500M model. Hundreds of thousands
of fully independent postage-stamp cells would receive too little traffic per
batch. Later scaling must test:

- regional cell banks,
- shared archetypes plus private adapters,
- hierarchical regions,
- local links plus sparse long-range links,
- curriculum that grows reachable regions,
- percentage of parameters receiving gradients.

No 500M design is accepted until active-parameter coverage and grouped-GEMM
occupancy are measured.

## 14. Validation gates

### Gate A: topology and packet state

- Legal ring and node actions are deterministic.
- Top-1 and top-k locations validate.
- Snapshot/commit cannot mutate the previous relay.
- Mutable state has no total-cell axis.

### Gate B: mailboxes

- Local aggregation matches a simple loop oracle.
- Empty locations are finite zero.
- Padding and zero write gates contribute nothing.
- Packets at different locations cannot communicate.
- Gradients are finite.

### Gate C: executable relay

- Every packet reads the old snapshot.
- Locations update only after all route decisions.
- Only legal destinations occur.
- One forward/backward smoke is finite.
- Payload shape remains `[B,L,D]`.

### Gate D: single-cell prefix

- Above chance reliably.
- Approaches the isolated Arrow result at the same training budget.

### Gate E: ten-cell ring

- Reproduce approximately 99% fresh generated token accuracy on length 7.
- Strong exact formation accuracy.
- Nontrivial path diversity.
- No dense all-cell packet replication.

### Regional development result

Cold sparse routing and compositional routing are now treated as separate
problems. A ten-cell developmental trainer divides the population into number,
arithmetic, and orchestrator regions, pretrains useful local behavior, and
then opens orchestrator-mediated bridges.

The strongest local 1/7/2 run retained 100% number conversion and 41.8%
modular arithmetic accuracy simultaneously. This passes the "non-wasteland"
regional competence gate. It did not yet transfer to length-seven parallel
prefix, so learned call/return composition is the next required gate before
claiming a successful ten-cell Path.

Regional ownership now follows a hard call/return invariant:

```text
orchestrator → specialist → orchestrator
```

Specialists never transfer packets directly to another specialist. Directional
relay masks force dispatch, local compute, distance-reducing return, and
orchestrator consolidation. A two-specialist composed call requires ten
relays in the current ring and is tested to finish with every packet owned by
the orchestrator.

The subsequent learned-dispatch experiment removed the scripted call order.
It retained 60.4% number and 25.2% arithmetic accuracy through an orchestrator,
but the seven-cell arithmetic region dominated calls.

Gate-cell dispatch now gives every region exactly one public orchestrator
action regardless of its internal cell count. Every specialist has one direct
return action, while its remaining actions stay regional. In the clean
frozen-specialist control, correct number calls rose from 5.5% to 53.1% and
correct arithmetic calls became 59.9%; routed accuracies were 45.7% number and
30.1% arithmetic, for a slightly higher harmonic skill score than ungated
dispatch. Allowing the gate interface and specialist transforms to adapt
reached 74.6% number and 34.6% arithmetic accuracy. The adaptive run is not a
topology-only comparison, but demonstrates that the new interface can recover
and exceed both ungated routed skills. Result-centric macro routing is
materially improved but not solved.

The accepted fallback curriculum begins with broad dense support and contracts
to gate-only calls after successful routes emerge. Familiar work should become
sparse; novelty or uncertainty must be able to widen support again. Region
size must never alter the number of macro choices exposed to the orchestrator.

The first 20k curriculum validates that policy. Extending top-6 exploration
for thousands of updates repeatedly destabilized, while a 1,600-step broad
warmup followed by fixed top-2 refinement retained 92.0% number and 93.2%
arithmetic accuracy through learned gates. Best-checkpoint rewind plus
optimizer-state reset was required four times during the long sparse phase.
The result establishes strong regional competence but not composition: a
500-step seven-output prefix probe remained at chance. Full orchestration must
therefore receive its own training stage and budget.

A global-Adam regional prefix control subsequently remained at the random
floor through 10,000 updates. Task learning began later and reached 16.9%
accuracy by step 17,000. At the time this was correlated with final
orchestrator ownership and motivated optimizer separation.

The first fixed split-optimizer probes did not establish a successful routing
rate. A rate of `0.01` plateaued near 50% final orchestrator ownership. One
`0.1` run reached roughly 75%, but that trajectory has not reproduced and is
classified as an outlier pending replication. Path training supports an
optional graduated policy that was tested as the next diagnostic: explore with
a high route SGD rate while ownership is low, then monotonically anneal toward
a lower rate as the best observed ownership rises. The transformation AdamW
rate did not follow this schedule.

The graduated route rate later drove ownership to 99.9% without improving
prefix accuracy or loss. With the return auxiliary removed, accuracy and loss
improved weakly while ownership fell again. Ownership is therefore not a
capability objective, and the split-optimizer/regional-orchestration branch is
superseded for the core prefix proof.

The corrected proof returns to the original local-policy vision: one optimizer,
task loss only, learned ingress, a genuine dense current frontier that may
anneal toward a small contribution-validated coalition, and carried packet
state. Cells jointly learn what to do and where to send the resulting packet.
Complete routes are emergent rollouts, not objects optimized as a whole.
Top-1 is accepted only when the learned computation survives it; it is not a
goal imposed independently of task quality.

The first attempted correction committed a hard steward from step zero and
used soft alternatives only as a backward surrogate. Seed 0 remained at 8.6%
prefix accuracy after 5,000 updates, while the validated ten-cell Arrow seed 0
had reached 68.3% at the same step. Backward exposure alone therefore does not
seed useful local skills: alternate cells must participate in the actual
forward computation early enough for task loss to compare their consequences.

The resulting dense-current-frontier seed 0 reached 18.7% accuracy and loss
2.1477 at 5,000 steps. It narrowed from 40 effective cell/node locations to
8.3, reopened ingress when loss stalled, and then resumed improving while the
downstream frontier stayed narrow. This validates the local behavioral
mechanism but does not yet pass the Arrow capability gate: the historical
10,000-parameter Arrow seed 0 reached 68.3% at the same step.

Seeds 1 and 2 reached 29.4% and 13.5%, for a three-seed Path mean of 20.5%
versus Arrow's 50.7%. All Path seeds left the random-loss floor and finished
with only 8.3--8.8 effective cell/node locations. Their learned first-steward
confidence ranged from 73.4% to 92.1% while the marginal ingress population
still covered roughly four to six effective cells. Ordinary optimization
therefore makes first-cell probabilities packet-dependent and sharp. High
confidence alone is neither capability nor discrete readiness: the most
confident seed was the weakest.

A hard-route audit found that hard downstream handoffs preserve seeds 0 and 2
almost exactly, while hard ingress drops all three models substantially and
collapses seeds 0 and 1 near chance. Residual non-top-1 ingress mass still
carries essential computation. A deterministic first cell is therefore not a
valid universal release criterion for this architecture. Gate E remains open
until a parameter-matched frontier closes the accuracy gap and demonstrates a
measured quality/sparsity trade-off across retained support sizes.

An abrupt hard-ingress continuation of the strongest seed used the same AdamW
state for another 1,000 updates. Loss recovered from 5.1811 to 2.5068, but
accuracy remained at chance (8.9%). Exact first stewardship is therefore a
training-transfer problem, not a switch to flip after soft training. The next
attempt must anneal or distill cooperative ingress into one selected steward
while preserving the local-policy architecture.

### Restoring the missing Arrow organs

The three-seed frontier audit exposed two representation regressions relative
to the validated ten-cell Arrow rather than a failure of local handoffs:

1. All ten Path cells shared one 16-by-8 token table. The original stitched
   Arrow gave every 980-parameter cell its own tied token input/readout table.
2. Path constructed one sequence-averaged mailbox per cell/node. The validated
   GTR-ARW cell used receiver-specific relative-position reads; output slot
   `i` and output slot `j` therefore saw different views of the same node.

Both are now independently switchable without changing the route semantics.
Private cell I/O raises the model from 8,596 to 9,748 parameters.
Receiver-specific relative reads alone raise it to 8,756. Together they yield
9,908 parameters, within 0.92% of the 10,000-parameter Arrow reference.

For relation-aware reads, cell/node `(c,n)` now computes:

```text
score[b,i,j,c,n] =
    log mass[b,j,c,n]
  + beta[c,n] dot phi(i-j)
  + log write_gate[b,j]

memory[b,i,c,n] =
    sum_j softmax_j(score)[b,i,j,c,n] * payload[b,j,c,n]
```

This is local stewardship, not global Transformer attention. A receiver can
read only packet state currently present in its steward cell/node. The learned
relative basis changes which local writer matters to that receiver. The
current frontier, transformed payload, and local next-hop decision still carry
forward once per relay; no complete route is scored, stored, or replayed.

The clean validation order is capacity-only, relation-only, then both, using
seeds 0, 1, and 2 sequentially. Concurrent local seeds are prohibited because
each process materializes its own dense training frontier and evaluation
batch; the first attempted parallel launch exhausted host RAM.

### Minimal Arrow-Carry control

The closest causal A/B does not approximate Arrow inside the Path engine. It
starts directly from `VectorizedGatorCellRing` and preserves:

- all ten private 980-parameter GTR-ARW cells;
- private tied embeddings and readouts;
- address anchoring, relative node reads, and controlled writing;
- four recurrent relays and node transition state;
- four-neighbor ring geometry;
- output voting, optimizer, objective, initialization, and exactly 10,000
  trainable parameters.

Only the cross-cell relay commit changes. Validated Arrow is
destination-owned: each destination gathers its neighbor states and applies
the destination's left/right gates. The corrected Arrow-Carry projects source
states before transport but preserves the destination's left/right acceptance
gates. The historical source-owned diagnostic applies an outgoing source gate
and is retained only under `--source-carry`.

Linear projection commutes with the ring average, so original Arrow and
destination-owned carry are algebraically equivalent even after private gates
specialize. Their state dicts are identical and interchangeable. This makes
the corrected `--carry-state` a tighter test of the carry hypothesis than the
broader Path rewrite:

```bash
python -u gator_cells_vectorized.py \
  --steps 5000 \
  --seed 0 \
  --neighbors 4 \
  --carry-state
```

This control does not yet add packet-conditioned cell routing. If source-owned
dispatch is tested again, it must be an additional offer decision rather than
a replacement for destination-owned acceptance.

#### Observed Arrow-Carry result

Seed 0 reached 24.3% at step 2,000 and 42.4% at step 3,000, ahead of the
historical Arrow seed-0 curve at those checkpoints (12.3% and 37.9%). It then
reached 47.8% at step 4,000 and 52.8% at step 5,000, below Arrow's 55.4% and
68.3%. Carried source ownership therefore restores the phase transition but
changes its later slope. It is the closest Path result to Arrow so far, not a
completed parity result.

Matched throughput was 2.88 versus 2.89 steps/second for 10-cell Arrow and
Arrow-Carry. Scaling both to 62 cells produced 0.57 steps/second for each.
This is expected after inspecting the computation: both still execute all
cells and rebuild all node traffic. State handoff semantics alone do not skip
work.

#### 2026-07-18 causal diagnosis: gate ownership, not carry itself

Lockstep gradient forensics identified the first exact divergence between
validated Arrow and source-owned Arrow-Carry. The two models were initialized
from the same state dictionary and trained on the same batches:

- initial maximum logit difference: approximately `7e-7`;
- initial loss difference: zero at displayed precision;
- every parameter gradient agreed to approximately `1e-8` except
  `neighbor_gates`;
- the `neighbor_gates` maximum gradient difference was approximately `0.06`;
- the two gate-gradient vectors were nearly orthogonal;
- one AdamW update separated gate values by `0.016`.

The reason is semantic. Original Arrow assigns the left/right gate to the
**destination** cell. A destination learns which neighboring computation it
should accept. The first Arrow-Carry implementation assigned that gate to the
**source** cell. A source learned what it should broadcast. Those operations
are forward-equivalent only while all gates have the same value. Their credit
assignment is different immediately, before their forward values visibly
separate.

Two interventions isolate this cause:

1. Freezing the gates keeps original Arrow and source-owned carry matched for
   at least 25 optimizer updates.
2. Tying all cell gates to one shared value also keeps them matched for at
   least 25 updates.

The restored `--destination-carry` control computes each projected source
message once, carries it to its receivers, and lets each receiving cell apply
its own gate. With arbitrary nonuniform private gates, this is algebraically
equivalent to original Arrow. Forward values and **all** parameter gradients
match within numerical tolerance. In a 25-step independent-optimizer
lockstep, losses matched to six decimals and logits stayed within `1e-6`.

The preserved mastered seed-0 checkpoint supplies the stronger functional
control. On the same deterministic 1,024-example evaluation batch:

| Handoff interpretation | Token accuracy | Exact sequence | Loss |
|---|---:|---:|---:|
| Original destination gather | 99.665% | 97.852% | 0.010192 |
| Destination-owned carry | 99.665% | 97.852% | 0.010192 |
| Source-owned carry | 10.659% | 0.000% | 14.808550 |

Destination-owned carry preserves the mastered computation at displayed
precision. Merely reinterpreting the already-trained private acceptance gates
as source broadcast gates destroys it. This rules out the proposition that
carried state is inherently incompatible with Arrow's acquired algorithm.

Independent from-scratch CPU runs of the locked reference and
destination-owned form both entered the same approximately 51--52% basin
rather than reproducing the original compiled-CUDA seed-0 mastery. Their first
1,000-step metrics matched, then algebraically harmless floating-point
reordering sent the chaotic optimizer trajectories apart. Because the locked
OG reference also failed to reproduce its historical CUDA curve, that run is
evidence of backend/trajectory sensitivity, not a destination-carry
capability regression. Full training comparisons must report backend,
compilation, and seed in addition to nominal model initialization.

The locked reference completed at 51.0% token accuracy, 0% exact, and loss
1.3163. Its step-20k gradient norm spiked to 31.35, reinforcing that this local
trajectory was oscillatory rather than a clean converged capability ceiling.

The corrected destination-owned carry control completed at 51.8% token
accuracy, 0.1% exact, and loss 1.0529. On the same forensic batch it selected
the same terminal cell identities as the local OG circuit (cells 5 and 9),
concentrated 99.8% of terminal mass on them, and still required every
intermediate cell. Thus parity holds at three levels: equations/gradients,
mastered-checkpoint function, and a full from-scratch circuit trajectory.

AdamW is the amplifier. With default epsilon `1e-8`, initial gradient
differences below `6e-8` produced a `1.32e-5` parameter difference after one
step. Raising epsilon to `1e-6` reduced the step-25 maximum difference to
`1.30e-6`; epsilon `1e-4` reduced it to `2.38e-7`. This does not prove the
larger-epsilon optimizer learns better, but it establishes why algebraic
equivalence alone does not guarantee identical stochastic acquisition curves.
The optimizer must be treated as part of the routing architecture's effective
dynamics.

This changes the interpretation of the earlier failure:

> State persistence did not erase Arrow's capability. Reassigning local
> routing responsibility from receiver acceptance to source broadcast changed
> the learning problem.

Destination-owned carry is therefore the semantic baseline for future sparse
work. Packet-conditioned source dispatch may still be added, but it must not
silently replace the receiver's learned acceptance decision. A faithful sparse
handoff can carry both decisions:

```text
source:      which candidate packet should I offer?
destination: should I accept and integrate this offered packet?
```

No full route is stored. Both decisions remain local and are learned through
ordinary end-task gradients.

#### Complementary paths are functional width

The first persistent active-only implementation selected one cell/node path
per token and evaluated only that selected transform. It remained at chance
through 4,000 updates. Doubling relay depth from four to eight remained at
chance through 3,000 updates. Carrying two complementary paths also remained
at chance through 2,000 updates.

These failures do not show that sparse local handoffs are impossible. They
show that hard pruning at initialization removed a capability used by dense
Arrow: the token is processed simultaneously by multiple private cells, and
their outputs are combined. Those branches are complementary computational
features, not merely alternative guesses at one route. Relay depth cannot
replace missing parallel width when the selected path has not yet learned
which branches are disposable.

The mastered seed-0 audit clarifies the desired endpoint. Individual active
node strands had nearly deterministic next-hop transitions (about
`1.02--1.06` effective edges), while roughly three complementary node strands
remained functionally necessary. The top two final node locations held 99.23%
of mass but were insufficient; the top three held 99.95% and nearly preserved
mastery. Sparse specialization therefore means a small deterministic
**coalition of traces**, not necessarily one trace.

Cell-level behavior was different. Although two cells owned almost all final
readout weight, removing any one of the ten cells from intermediate relay
execution reduced accuracy by at least 16.8 points; several zero-vote cells
caused losses above 70 points. OG Arrow learned a distributed ten-cell
intermediate circuit and a sparse terminal readout. The Path goal is not to
pretend those zero-vote cells were idle. It is to make active-cell support
packet-conditioned and train it broadly enough that a smaller mature coalition
can inherit or consolidate their functions.

Terminal cooperation is continuous score fusion, not majority voting. In the
mastered seed-0 checkpoint:

```text
final_logits ~= 0.658 * logits(cell 0)
             + 0.320 * logits(cell 4)
             + small residual terms
```

Each terminal emits a full vocabulary score vector, so there is no two-cell
tie to break. One terminal can supply the dominant interpretation while
another shifts close token alternatives. The next Path readout should preserve
learned mass-weighted terminal fusion rather than forcing one terminal or
counting discrete token choices.

This reinterprets the apparent cell advantage. A 40-node monolith with one
shared `H` has one payload lane despite 40 location probabilities. Ten
four-node cells have ten payload lanes. Cells may help because they prevent
premature averaging of complementary computations, not because a cell wrapper
is intrinsically required. A vision-complete flat Path must carry several
independent `(site, payload, mass)` records.

The next sparse design must therefore begin with dense complementary support,
learn contribution/acceptance scores, and prune only after measured branch
utility becomes unequal. Sparsity is an acquired execution policy, not a safe
initial condition.

#### Keep node refinement separate from graph-site handoff

The first persistent active prototype also overloaded one routing decision.
Its selected internal node edge indexed `active_cell_offsets`, so choosing the
next node simultaneously chose whether and where the packet moved in the cell
ring.

Validated Arrow did not have this coupling:

```text
inside each cell: choose/mix destination-node transforms
between cells:    offer transformed state to ring neighbors
at receiver:      apply destination-owned acceptance
```

An internal specialization decision and an inter-site stewardship decision
serve different purposes. Tying them means a node cannot independently learn
"apply transform 2 but remain local" versus "apply transform 2 and offer this
result to the right region." It also replaces Arrow's simultaneous neighbor
offers with one selected move.

The restored Path relay must therefore be two-stage:

1. Active stewards refine packet branches through their local node graph.
2. A separate packet-conditioned local handoff proposes neighboring graph
   sites, whose destination-owned acceptance determines retained arrivals.

Both remain local choices and ordinary learned weights; no complete route is
represented. This factorization restores Arrow's independent sources of
functional width while still allowing mature sparse execution.

#### Carry failure root-cause ledger

| Change from OG Arrow | Evidence | Status for next iteration |
|---|---|---|
| Source owns the neighbor gate | First-step gate-gradient difference about `0.06`, nearly orthogonal; frozen/tied gates restore lockstep | Rejected; preserve destination acceptance |
| One hard payload lane | Hard active model stayed near chance through 4k; every OG cell is ablation-important | Rejected; begin with several real forward lanes |
| Two hard payload lanes | Two-path model stayed near chance through 2k | Rejected as an initial frontier |
| Hard node support from initialization | Mastered OG falls to 41.3% at top-1 and 77.5% at top-2 | Prune only after measured consolidation |
| Internal node edge also selects next cell | Removes OG's independent node refinement and neighbor communication decisions | Rejected; use a separate local handoff |
| Merge arrivals immediately into one payload | Destroys the persistent semantic lanes supplied by OG cells | Retain branch payload records; merge explicitly |
| Carry persistent memory statistics | Dense-rebuild equivalence and finite-gradient tests pass | Not identified as a capability failure |
| Hoist relay-invariant tensor construction | Forward-equivalent within about `2e-6`; tests pass | Safe math optimization, subject to backend trajectory variance |

This ledger narrows the diagnosis: carry-forward state is not itself the
problem. Capability was lost when the carry rewrites changed responsibility,
functional lane width, or independent routing decisions while attempting to
obtain sparsity immediately.

#### Convergent flat-frontier proof

`gator_convergent.py` implements the first direct test of the lane hypothesis:

```text
parameters=10,010
permanent cells=0
shared graph nodes=40
persistent payload lanes=10
payload width=8
node degree=3
relays=4
```

For degree three, the exact trainable count is

\[
P(K,N,D)
=
K(2D^2+31D+1)
+
N(D^2+8D+28),
\]

where \(K\) is persistent payload lanes, \(N\) is shared graph nodes, and
\(D\) is payload width. This exposes lanes and graph sites as distinct scaling
axes; increasing a legacy “cell” count changed both at once.

For arbitrary fixed connection degree \(G\), the generalized count is

\[
P(K,N,D,G)
=K(2D^2+31D+1)
+N\left[D^2+2D+4+G(2D+G+5)\right].
\]

Changing connection endpoints or rewiring probability at fixed \(G\) costs no
parameters. Increasing degree adds router/address capacity and execution work,
so endpoint layout should be swept at degree three before degree is allowed to
grow.

Ten logical branches begin at evenly spaced nodes
`[0,4,8,...,36]`. Every branch carries its own `H` and soft node passport
across relays. Each lane retains private tied token I/O, address and relay
projections, normalization, and a tiny two-layer adapter. Graph-site router
rows and node transforms are shared across lanes. This reallocates the budget
away from the first prototype's oversized global router and preserves the
private feature views that OG Arrow's cells used. Branches occupying or
approaching the same node interact through receiver-specific node memory.
After the final relay, learned terminal masses fuse each lane's complete
16-token score vector.

This is intentionally fixed at ten lanes for the capability proof. It does not
yet widen lane count on novelty or physically pack a narrowed frontier.
Passing prefix acquisition would show that cell wrappers are unnecessary once
their essential contribution—persistent branch payload separation—is made
explicit. Failing while 10x4 Arrow succeeds would identify private
cell-specific parameters or structured cross-lane communication as additional
necessary organs.

The proof retains dense `[B,T,K,N]` location tensors as a correctness oracle,
but receiver memory is factorized without a lane-expanded score tensor and
router/node transforms execute only reachable graph rows. With degree-three
routing, each seeded lane can reach at most nine nodes after four relays; the
other lane/node pairs are already omitted from transform execution.

A ten-step CPU optimization smoke of the private-lane model is finite:

```text
step=10 loss=2.5204 accuracy=8.3%
terminal_effective=9.99/10
node_effective=3.72
interval throughput approximately 2.1 steps/s under background load
```

This establishes execution and gradient health only. The 20k acquisition run
was the capability gate. It completed at `92.1%` token accuracy, `61.0%`
exact-sequence accuracy, loss `0.2211`, and `3.56` local CPU steps/s. Terminal
support narrowed to `3.04` effective lanes and node support to `1.39` without
hard pruning.

On a separate 4,096-example probe, no dominant lane solved the task alone:
the best lane reached `27.8%`, the best two `44.3%`, and the best three
`91.2%`. Their weights were approximately `39.5%`, `31.5%`, and `28.6%`.
This is converging complementary computation, not alternative whole routes
competing for selection.

The result exceeds the reproducible local OG Arrow control by `41.1` accuracy
points and is therefore the current matched leader. The historical mastered
Arrow checkpoint is retained as an upper-bound observation because OG could
not reproduce that trajectory in the current environment. The next matched
controls bracket recurrent depth at one and eight relays while holding
parameters, batches, seed, optimizer, lanes, and graph fixed. One relay keeps
all ten parallel traces but reduces each lane's reachable degree-three graph
support from 9 nodes to 3; eight relays expands it to 17.

The default comparison still combines depth with relay-time scheduling.
Controlled writing reaches full output-slot mass over the configured depth,
and sinusoidal step codes are normalized over that same depth. For a
strict extra-computation A/B, instantiate both models with:

```text
relay_code_span=8
write_warmup_relays=4
```

Tests establish that the R=4 trajectory is then exactly the first four hops of
R=8 when weights and inputs match. Any downstream difference comes from the
four additional recurrent hops rather than different early phase codes or
write ramps.

Depth also controls when independent traces can interact. With starts spaced
four nodes apart and radius-one graph moves, lane supports are disjoint at
states \(q^0\) and \(q^1\), first overlap at \(q^2\), and become visible to
shared node memory only in the third relay. R=1 and R=2 therefore execute
parallel private traces but cannot coordinate through graph traffic. R=3 is
the first true multi-trace interaction depth; larger R adds further
coordination rounds. This collision horizon is now tested explicitly and
reported by the checkpoint analyzer.

Default depth configurations also change the output-write ramp: R=1 uses
`[0]`, R=2 uses `[0,1]`, and R=3 uses `[0,0.5,1]`. Accordingly, the
operational R=1-to-R=2 difference includes the first opportunity for formed
outputs to write into memory, not just private recurrent refinement. A strict
collision audit must give every depth a common `relay_code_span` and
`write_warmup_relays`, so common-prefix hops are identical.

However, R=1 is not a local-information baseline. All sequence positions in a
lane currently share that lane's initial node. The receiver-relative memory
read can aggregate every prefix input for every output coordinate before any
route movement occurs. Its `69.5%` token accuracy at 10k confirms that a
single relay can exploit this direct global within-lane view. The present
prefix result proves simultaneous structured decoding, but it does not force
the model to discover a logarithmic propagation algorithm. A stronger
algorithmic benchmark must distribute position ingress over graph sites or
mask the initial memory radius so information can only arrive through
handoffs.

The completed isolated R=1 control reached `84.7%` token accuracy,
`30.3%` exact accuracy, and `0.4446` loss at 20k, averaging `19.15`
steps/s. Its `6.76` effective terminal lanes versus `1.10` effective node
locations show how a non-coordinating version compensates: it recruits a
broad coalition of independently learned traces. R=4 used fewer terminal
specialists but more than doubled exact-sequence accuracy (`61.0%`),
consistent with recurrent graph interaction improving global formation
coherence.

Operational R=2 was substantially more sample-efficient: `85.6%` token,
`36.1%` exact, and `0.4064` loss at 10k, with `9.90` steps/s end to end.
It beat R=1's 20k result and R=4/R=8 at the same 10k update count. This
rules out a simple monotonic “more relays learn better” law. Recurrent depth
offers greater computational reach, but deeper credit assignment can delay
acquisition; R=2 is the current early-learning optimum under the operational
depth-specific schedules.

The trained R=2 trace validates the collision argument. Pairwise lane overlap
was `0.0` at \(q^0\) and \(q^1\) and appeared only at terminal \(q^2\)
(`0.209867`), which is never consumed by another memory read. Its best lane
alone reached just `25.5%`, whereas its top six terminal lanes reached
`85.7%`. R=2 succeeds through complementary independent traces fused at
readout; R=3 is the first configuration that can consume their overlap inside
the graph.

Extending the native R=2 model to 20k did not recover mastery. It finished at
`91.0%` token / `50.7%` exact, with a best logged point of `91.7%` /
`55.5%` at 16k. Thus its dramatic early-learning advantage is real but does
not establish equal asymptotic capability. R=1 and R=2 both approached a
roughly 92% token ceiling by their tested runways.

Operational R=3 finished 10k at `71.3%` token, `8.5%` exact, and `0.8225`
loss, nearly identical to R=4 at that update and far behind R=2. Its terminal
support prematurely narrowed to `2.60` lanes at 5k, then reopened to `4.27`
by 10k. First collision therefore adds both coordination opportunity and
interference; the optimizer must learn useful meetings before graph contact
pays for itself. The trained passports had pairwise overlap `0.626731` at
\(q^2\), consumed by the third relay, and `0.786597` at terminal \(q^3\);
the negative early result cannot be dismissed as failure to meet.

Longer training did not repair R=3. It ended 20k at `82.2%` token / `25.0%`
exact / `0.5049` loss, with `4.71` effective terminal lanes. In contrast,
R=4 narrowed toward three terminal specialists while climbing to 98.2%.
The evidence supports a four-stage protocol: private formation, routed
expansion, cross-trace reconciliation, then commitment. Three relays reach
reconciliation but expose the unresolved coalition directly to readout.

The zero-parameter `bootstrap_context_mode=self` control now enforces that
condition without changing lane count, graph topology, width, or parameter
budget. On relay zero, each packet receives only its own payload as context.
Shared receiver-relative node memory opens from relay one onward. Therefore
an output packet cannot read any condition packet before at least one
handoff; packets must move into overlapping graph support before information
can transfer. The default remains `shared` so historical checkpoints retain
their exact semantics.

#### A mastered route uses every relay

Truncating the preserved 99% Arrow checkpoint while retaining its original
four-relay phase codes and write schedule yielded:

```text
trained hops executed     1       2       3       4
token accuracy          12.1%   20.9%   58.3%   99.76%
exact accuracy           0.0%    0.0%    3.2%   98.49%
```

The final relay is a learned convergence/commit operation, not redundant
iteration. The checkpoint distributes its computation across the full route.
This distinguishes two questions that must not be conflated: a trained R=4
solution requires all four of its hops, while a separately trained R=1 or
R=2 model may discover a different shallow algorithm given sufficient
runway.

Mastered Arrow makes the phase behavior explicit. After hop three its seven
positions were only
`61.96,88.72,88.92,70.36,47.27,31.74,19.29%` correct. Hop four transformed
them to at least `99.22%` correct at every position. The final relay is a
global convergence stage that repairs an incomplete, distance-degraded
formation.

The reproducible Convergent R=4 checkpoint independently showed `9.5%`,
`10.4%`, `35.3%`, and `91.2%` token accuracy after one through four trained
hops. Both model families therefore place a decisive commitment operation in
the final relay. Native shallow models and truncated deep models are
functionally different algorithms.

R=1 did not do so within 30k. It ended at `92.0%` token / `56.3%` exact,
with a best logged point of `92.3%` / `57.5%` at 29k. It can trade more
training and a wider terminal coalition for much of R=4's token accuracy,
but it did not recover Arrow mastery or equal the deeper model's formation
coherence within the tested runway.

The residual shallow errors are structured by dependency distance. Best R=1
accuracy fell from `99.95%` at output zero to `82.67%` at output five. R=4
at 24k raised the corresponding late position to `90.04%` and improved
long-prefix step consistency from `72.80%` to `85.21%`. Relay depth is
therefore buying reliable propagation over longer prefixes, not a uniform
classification-margin improvement.

R=4 ultimately escaped the common shallow plateau. Its 20k-to-30k
continuation reached `98.2%` token / `88.5%` exact / `0.0638` loss, versus
R=1 and R=2 best token accuracies near `92%`. This is direct evidence that
consumed cross-trace communication plus a fourth commitment stage raises the
attainable solution. The constant `0.008` learning rate produced substantial
late oscillation, motivating a preserved-best-state cooldown rather than an
architectural restart.

That cooldown restored the predefined mastery gate. Resuming the best R=4
state and its optimizer/batch RNG at learning rate `0.002` reached `99.2%`
token / `95.2%` exact / `0.0263` loss at step 31k, then held `99.3%` /
`95.6%` / `0.0235` at step 32k and `99.4%` / `95.9%` / `0.0223` at
step 33k. The flat carried-state graph has therefore
recovered Arrow-level token mastery in the current environment; the remaining
gap is exact-sequence error, not failure to learn the prefix algorithm.

On a separate 4,096-example probe, the step-32k checkpoint scored `99.29%`
token / `95.68%` exact. Only three terminal lanes had material fusion weight
(`40.64%`, `29.93%`, and `29.13%`), but none was a complete solver: the best
lane alone scored `33.25%`, the best pair `82.65%`, and all three `99.27%`.
The learned unit is therefore a convergent coalition of complementary traces,
not a winner-take-all particle. Executing only the first one, two, or three
trained relays scored `9.61%`, `12.19%`, and `34.59%`; the fourth restored
`99.35%`. The commitment discontinuity survived mastery and cooldown.

#### Restoration verdict

The carry-state investigation now separates four previously conflated causes:

1. **The first Arrow-Carry rewrite changed ownership semantics.** Arrow's
   gates are destination-owned acceptance gates. Making them source-owned
   altered credit assignment. Correct destination-owned carry is
   algebraically equivalent to ordinary Arrow, including parameter gradients;
   carry itself was not the regression.
2. **Path collapsed a cooperative wave into a premature particle.** Historical
   Arrow and Convergent both require several low-mass branches. A single hard
   steward, explicit complete-route ranking, coupled cell/node handoffs, and
   collision averaging removed complementary computation before it
   specialized.
3. **Adding cells did not add usable reach.** In the stitched ring a terminal
   cell sees at most \(\min(C,1+KR)\) cells; out-of-horizon cells receive no
   terminal gradient. Uniform voting also dilutes each cell's contribution,
   while the language build compressed many cell outputs into four width-eight
   summaries. Nominal parameters therefore grew faster than trainable,
   observable capacity.
4. **The restored primitive is a shared graph with persistent parallel
   traces.** Ten width-eight payload lanes move through one 40-node graph,
   keep both payload and location state between relays, retain soft
   complementary routes, and fuse continuously at readout. Four relays plus
   a late learning-rate cooldown recover mastery without explicit route
   storage or permanent cell wrappers.

This preserves the original vision: a route exists only as the rollout of
local learned handoff decisions. What must be carried is packet state and its
probabilistic passport—not a named route object, and not a full recomputation
of a stitched cell population.

#### Connection graphs replace geometric neighbors

The restored flat graph no longer requires a ring. `ConvergentGator` now
stores an explicit table of legal outgoing connections:

```text
destinations[source_node, connection_slot] -> destination_node
```

The packet-conditioned router chooses among connection slots. It does not
learn or store a complete route. Arbitrary fixed connections therefore
preserve the local policy:

```text
inspect packet + local traffic
        -> transform packet
        -> choose one or several legal outgoing connections
        -> carry payload and passport forward
```

The active execution path computes exact graph closure from every lane's start
node. It no longer assumes that reachable nodes form a contiguous interval
around a ring location. Padded closure rows are masked before routing and node
transforms, so optimized reachable execution remains equivalent to evaluating
all nodes.

Three layouts are currently exposed:

- `ring`: self plus geometrically local connections;
- `small_world`: self and a directed successor backbone, with a controlled
  fraction of the remaining ring edges replaced by fixed random shortcuts;
- `dense`: every node is a legal destination, intended as an upper-bound
  control rather than the sparse production topology.

`--connection-rewire` creates an equal-parameter topology continuum.
Rewiring changes only endpoint buffers, not trainable parameter count. At
40 nodes, degree three, ten lanes, and four relays, static seed-zero geometry
was:

| Layout | Rewire | Diameter | Mean nodes reachable per lane after R4 | Lane pairs colliding by R3 |
|---|---:|---:|---:|---:|
| Ring | 0% | 20 | 9.0 | 22.2% |
| Small-world | 10% | 31 | 9.9 | 31.1% |
| Small-world | 25% | 19 | 12.8 | 57.8% |
| Small-world | 50% | 12 | 17.5 | 88.9% |
| Small-world | 100% | 10 | 23.0 | 97.8% |

Sparse rewiring is not automatically better. The 10% directed instance
actually increased diameter because it removed useful reverse ring edges
without adding enough shortcuts. At the other extreme, full rewiring makes
almost every lane meet by relay three and may homogenize complementary
traces. The capability experiment must therefore sweep topology rather than
declare maximum reach the winner.

The first matched A/B holds seed, batches, parameters, width, lanes, relays,
optimizer, and task fixed while testing 25%, 50%, and 100% rewiring against
the preserved ring curve. Report learning speed, final accuracy, exact
accuracy, realized node entropy, lane collision, and steps per second.

The scaling failure becomes structural at 80 nodes. With ten evenly seeded
lanes and four relays, the degree-three ring produced no pairwise lane overlap
through relay three. Its first contact occurred only in terminal \(q^4\),
after the last memory read, so the nominal R=4 model would behave like an
independent terminal ensemble. At the same node count and degree, 25%
rewiring produced overlap after relay two and connected 33.3% of lane pairs
by relay three; relay four can consume and reconcile that traffic.

| 80-node layout | Diameter | Lane-pair overlap after R2 | After R3 | After R4 |
|---|---:|---:|---:|---:|
| Ring | 40 | 0.0% | 0.0% | 22.2% |
| Small-world, 25% | 22 | 8.9% | 33.3% | 66.7% |
| Small-world, 50% | 15 | 20.0% | 55.6% | 86.7% |
| Small-world, 100% | 12 | 37.8% | 91.1% | 100.0% |

This establishes a topology scaling invariant: at least some cross-lane
overlap must occur no later than \(q^{R-1}\), leaving a subsequent relay to
consume it. Increasing node count without reducing graph distance violates
that invariant even though parameter count and nominal relay depth increase.

For an evenly seeded degree-three ring, adjacent lane starts are approximately
\(N/K\) nodes apart and each trace expands one node in each direction per
relay. The minimum consumed-contact condition is therefore

\[
\frac{N}{K}\leq 2(R-1)
\qquad\Longleftrightarrow\qquad
N\leq 2K(R-1).
\]

With \(K=10,R=4\), the local ring can support at most approximately 60 nodes
before adjacent traces cannot meet in time. Forty nodes satisfy the bound;
80 nodes do not. This is a hard geometric scaling ceiling, not an optimizer
problem. Long-range connections, additional lanes, or more relays are
required beyond it.

The first trained topology control used 25% rewiring at 40 nodes. It was
matched exactly to the preserved ring run: seed-zero initialization, portable
batch stream, 10,010 trainable parameters, ten lanes, width eight, degree
three, four relays, AdamW, LR `0.008`, and cross-entropy only.

| Step | Ring token accuracy | 25%-rewired token accuracy |
|---:|---:|---:|
| 1,000 | **18.3%** | 8.7% |
| 2,000 | **35.4%** | 35.1% |
| 3,000 | 45.1% | **58.8%** |
| 4,000 | 48.9% | **73.2%** |
| 5,000 | 54.0% | **79.8%** |

At 5k, exact-sequence accuracy improved from `0.4%` to `21.2%`, and loss
fell from `1.1774` to `0.5980`. Rewiring incurred approximately 8--9% lower
throughput over the matched first 5k window (`3.47` versus about `3.79`
steps/s), because each lane executed more reachable node rows. Despite that
cost, it exceeded the ring's 5k capability before step 3k and therefore
improved wall-clock time to the same accuracy.

The connected checkpoint retained the same three-trace solution structure:
terminal weights were approximately `38.2%`, `30.1%`, and `29.2%`. The best
single trace scored only `26.5%`, the best two `44.3%`, and the first three
reconstructed `78.7%`. Connections accelerated formation and cross-trace
coordination without collapsing the complementary coalition into one path.

Doubling rewiring to 50% while retaining four relays was strongly
antagonistic: the model finished 5k at only `38.6%` token / `0.1%` exact.
The same 50% connection graph with two relays reached `62.3%` / `2.8%` in
only 457.8 seconds. R4/25% remained better per update (`79.8%`), but required
1,440.6 seconds. Connectivity and relay depth are therefore coupled:

```text
more connections -> wider frontier per hop
more relays      -> repeated traversal of that widening frontier
combined excess -> premature mixing and superlinear execution cost
```

The next compute-matched control resumes R2/50% to 16k total updates, roughly
the number it can execute in the wall time of a 5k R4/25% run.

That control succeeded. R2/50% reached `98.0%` token / `87.1%` exact /
`0.0691` loss at 16k. Its combined runtime was 1,494.8 seconds, only 3.8%
longer than R4/25%'s 1,440.6-second 5k run. At essentially equal wall time,
the shallower connected graph gained 18.2 token-accuracy points and 65.9
exact-sequence points. It exceeded the deeper model's endpoint by step 7k,
after approximately 10.6 total minutes.

The best-checkpoint audit shows that shallower execution did not reduce the
model to one serial route. One terminal lane received 47.8% of the vote but
achieved only 55.6% accuracy alone; eight lanes were needed to reproduce the
full 98.0% result. Removing the dominant lane reduced accuracy to 70.3%.
R2/50% is therefore a faster complementary-path solution: broader spatial
contact replaces recurrent depth while several traces still converge at
readout.

This is direct evidence for spatial/temporal compensation:

\[
\text{useful communication}
\neq \text{relay count alone};
\qquad
\text{it depends on graph reach per relay and collision timing}.
\]

The fastest successful configuration currently uses enough shortcut
connectivity to move and reconcile information in two hops, avoiding the
superlinearly expensive R3/R4 frontiers.

The matched one-shot Transformer control clarifies the compute result. A
10,020-parameter, four-layer Transformer ran at 34.68 steps/second, versus
10.61 for the 10,010-parameter R2/50% Gator, but ended its identical 16k
portable run at `8.5%` token / `0.0%` exact / `2.4852` loss. Gator ended at
`98.0%` / `87.1%` / `0.0691`. Thus the Transformer won the isolated
update-throughput measurement by 3.27 times while failing to acquire the
algorithm. Previous Transformer scaling screens showed that increasing it to
roughly 70k parameters still required more than 20k updates, eroding the raw
speed advantage before matching capability.

The operative scaling metric is consequently:

\[
\text{time-to-capability}
=
\frac{\text{updates-to-threshold}}{\text{updates per second}},
\]

not updates per second in isolation. A fast model that never crosses the
capability threshold has no finite measured time-to-capability within the
experiment.

#### Tied-vocabulary Convergent language shell

The first Gate-G transfer preserves persistent complementary lanes and the
shared connected graph while replacing ten private 16-token tables with one
tied language vocabulary:

\[
\begin{aligned}
e_t &= E[x_t],\\
h_{t,k}^{(0)} &= W^{in}_k e_t + A_k(a_t),\\
\bar e_o &= \sum_k \pi_k W^{out}_k h_{o,k}^{(R)},\\
\ell_o &= \bar e_o E^\top + b.
\end{aligned}
\]

Each lane therefore learns a distinct input and output lexical view, but the
2,048-row table is stored once. Voting before the tied vocabulary projection
is exactly linear-equivalent to voting over lane vocabulary logits and avoids
materializing the latter.

For vocabulary \(V\), shared embedding width \(E\), lane width \(D\), lanes
\(K\), nodes \(N\), and degree \(G\), the language count is:

\[
P =
VE+V
+K(2D^2+15D+1+2DE)
+N[D^2+2D+4+G(2D+G+5)].
\]

The first scale uses `(V,E,D,K,N,G,R) =
(2048,32,20,10,66,3,2)`, totaling 130,202 parameters. Its fixed graph is
small-world seed zero with 50% rewiring. At 66 nodes, 6.7% of lane pairs
already overlap after relay one, preserving contact that relay two can consume
while expanding the graph beyond the 40-node prefix proof.

Lane count and graph workspace must scale together. Define the static
frontier-density proxy

\[
\rho_{\mathrm{frontier}}=\frac{K\,F(G,R)}{N},
\]

where \(F(G,R)\) is the mean distinct-node frontier reached by one lane before
the final read. The validated degree-three, R2 L10/N66 geometry has
\(F\approx6\) and \(\rho\approx0.91\). Doubling lanes to 20 while retaining
66 nodes doubles density to approximately 1.82 and nearly saturates the graph
(64/66 nodes covered at R2). It also doubles each lane's expected number of
colliding peers from about 2.6 to 5.4. The provisional balanced family is
therefore \(N\approx6.6K\) for this degree/depth.

At `(V,E,D,G)=(2048,32,20,3)`, the count reduces to

\[
P=67{,}584+2{,}381K+588N.
\]

Thus a balanced twenty-lane scale is L20/N132 with 192,820 parameters, while
an approximately 154k parameter-matched balance is L14/N90 with 153,838
parameters. Treat L20/N66 as a deliberate lane-congestion ablation, not a
generic capacity scale. Width governs transformation capacity per occupied
node and requires a separate controlled sweep.

Frontier density is only a first-order proxy. The actual capability may depend
on forming the right terminal voter coalition. The current implementation
learns one global vector

\[
w=\operatorname{softmax}(v_{\mathrm{terminal}})
\]

and uses the same lane weights for every prompt, output slot, and token class.
Measure both entropy-effective voters

\[
K_H=\exp\left(-\sum_k w_k\log w_k\right)
\]

and concentration-effective voters

\[
K_2=\frac{1}{\sum_k w_k^2}.
\]

These statistics describe vote concentration, not functional contribution.
Also ablate the top-\(k\) voters and each voter individually on a frozen
validation manifest. Prefix experiments already showed why: three material
lanes could be complementary even when no single lane was accurate.

The language scaling hypothesis therefore has three coupled quantities:

1. raw lanes supply candidate semantic hypotheses;
2. consumed graph contact lets compatible hypotheses exchange information;
3. terminal fusion retains a useful number of complementary voters.

Too much contact can homogenize candidates; too little contact leaves voters
unable to converge; too few effective voters removes complementary evidence.
`analyze_convergent_votes.py` reports the global coalition directly from any
prefix or language checkpoint. A later architecture may make terminal votes
input-dependent, but the static coalition should be measured before adding
that capacity.

#### Persistent language-floor bottleneck audit

Repeated language runs cluster near the same aggregate PPL floor despite
substantial changes to cells, nodes, lanes, topology, and total parameters.
Two interface constraints remained nearly invariant:

1. token identity enters and leaves through a shared width-32 lexical space;
2. one global terminal vote vector is reused for every prompt and output slot.

The historical 1M population scale retained width-32 input/output bandwidth
and barely improved over its 130k counterpart. Increasing internal population
therefore did not test whether lexical rank or terminal selection was limiting.
The 130,072-parameter one-shot Transformer used a tied width-38 interface and
reached PPL 31.45, proving that the cleaned corpus/evaluator can move below the
nominal Gator floor, although its greedy semantic behavior remained poor.

The next isolation matrix must hold L16/N90, R2, degree three, rewiring,
training stream, and validation fingerprint fixed:

| Condition | Terminal voter | Lexical width | Question |
|---|---|---:|---|
| A | Global static | 32 | Reproduce the floor |
| B | Prompt-conditioned | 32 | Is static coalition selection limiting? |
| C | Prompt-and-slot-conditioned | 32 | Is per-position formation control limiting? |
| D | Global static | 64 | Is lexical/output rank limiting? |
| E | Slot-conditioned | 64 | Do formation and lexical rank interact? |

Condition B should add only a tiny gate relative to the base model and is the
first clean test. Its lane-specific confidence query adds exactly
\(K D=320\) parameters to L16/N90, for 158,920 total. The query initializes
to zero, making the prompt-vote forward exactly equal to global voting before
learning. One prompt-specific coalition is reused across every output slot.
Condition C reuses the same width-D query at every final lane/slot state and
adds one private four-value address key per lane. At L16 this adds 64 more
parameters, for 158,984 total. It produces one coalition per immutable,
output-relative address while keeping all positions simultaneous; no
predicted token is emitted or fed back. The jury's coordinate system spans
the output formation independently of context length. The direct address term
prevents repeated relay mixing from erasing the only signal that tells the
juries apart.

Condition D is intentionally larger: for L16/N90 it raises the count from
158,600 to 244,616 because both the tied vocabulary and lane-specific
input/output projections widen. Do not interpret D as a pure total-parameter
law; compare it with a compute-width scale that retains lexical width 32.

Aggregate PPL is not sufficient. The matched Transformer already showed that
PPL can improve while greedy outputs remain generic and under-formed. Report
content-only PPL/top-k, formation coverage, response length, BLANK/whitespace
rates, fixed semantic probes, and effective/top-\(k\) voter contribution
beside every condition.

Padding is excluded from node writes and occupancy. A true tokenizer BLANK
embedding initializes every output query, and forbidden structural IDs are
masked at the tied readout. The context-mask invariance test changes every
padded token ID and requires bitwise-equivalent output logits.

#### Exact relay execution audit

The Convergent reference now has an exact optimized eager path. Relay-invariant
relative kernels, step codes, write schedules, and address views are computed
once per forward. Router semantics and occupancy are contracted separately,
node embeddings are folded into node-transform offsets, and receiver
numerator/denominator mixing shares one grouped matrix multiplication.

Against the committed semantic implementation, the optimized complete model
agreed within `4.77e-7` in FP32 outputs and `5.17e-7` relative in the worst
observed parameter gradient. At batch 32 on one CPU thread, throughput rose
from `7.12` to `8.82` steps/s for four relays and from `2.70` to `3.71`
steps/s for eight relays.

The audit exposed an important invariant. Convergent's inherited node weights
are indexed `[input, output]`; the reusable grouped-linear helper accepts
`[output, input]`. Since both dimensions are width eight, an omitted transpose
silently changed every node function without causing a shape error. Dedicated
tests now lock this orientation.

Normalized receiver convolution was mathematically suitable but approximately
`1.7x` slower at the prefix workload's length nine. Short formation tasks keep
the quadratic receiver kernel; convolution remains appropriate only for a
separately benchmarked longer-context regime.

#### Route-support requirement after convergence

The four-relay checkpoint's `1.39` effective node count does not license a
top-1 path. Renormalizing every relay to top-1, top-2, top-3, top-4, and top-5
support yielded `20.5%`, `49.7%`, `72.6%`, `87.7%`, and `91.2%` token
accuracy respectively, against `91.45%` for the soft oracle. Top-7 was
numerically identical to the oracle on the measured batch.

This confirms that a trace is sparse but still wave-like: low-mass branches
can carry necessary corrections before reconverging. Physical execution
should begin dense, learn the basin, and then cap support around a validated
top-k. It must not equate entropy's effective count with the number of branches
safe to retain. Top-2 preserved `96.73%` of final probability mass while
falling to `49.7%` accuracy; top-4 preserved `99.70%` while still losing
`3.74` accuracy points. Pruning must measure counterfactual contribution, not
rank branches by mass alone.

Naively gathering one node-weight tensor for every retained
`(batch,token,lane)` branch is not the physical solution. Although that
prototype matched the dense-pruned logits and gradients, it slowed a
single-thread batch-32 benchmark from `7.70` to `5.19` steps/s at four relays
and from `3.12` to `2.27` at eight. A useful sparse backend must sort or bucket
packets by node, apply each selected node transform to a contiguous group, and
scatter the results back.

Pruning must also vary over relay time. Applying top-2 only at the final relay
retained `90.62%` token accuracy, but applying it throughout collapsed to
`49.71%`. A schedule of `[3,5,6,4]` retained `91.43%` token and `57.42%`
exact accuracy against the soft `[3,5,7,9]` oracle's `91.45%/58.20%`, while
reducing node-transform rows by 25%. Gator should preserve exploratory
alternatives during early handoffs and converge near final commitment.

For the current degree-three ring, reachable node-transform work per lane is

\[
\sum_{r=1}^{R}(2r+1)=R(R+2),
\]

and router-source work is \(R^2\), until the graph saturates. Four relays
therefore visit `24/16` transform/router rows; eight visit `80/64`. A top-5
frontier would bound the eight-relay counts near `38/34`, making late learned
packing especially valuable at greater recurrent depth. The present top-k
ablation changes support but still executes the dense reachable kernels; a
packed implementation is required to realize the theoretical saving.

#### Carrying node memory and transformed state

The speed-producing state must include both packet and node state:

```text
packet = (payload, immutable address, cell owner, node owner)
node   = (relative-memory numerator, relative-memory denominator)
```

For receiver `i`, a writer `j` at `(c,n)` contributes:

```text
w(i,j,c,n) = exp(beta[c,n] dot phi(i-j) - fixed_max(i,c,n)) * write(j)
num(i,c,n) += w * payload(j)
den(i,c,n) += w
```

When the packet moves, its old contribution is subtracted and its transformed
arrival is added. The fixed per-receiver/location maximum keeps exponentiation
stable while preserving the normalized memory ratio across incremental
updates. This makes update work proportional to changed packets and sequence
receivers, independent of the number of empty cell/node locations.

The executable `persistent_memory` milestone proves incremental equality with
a dense rebuild and finite backpropagation. It intentionally retains a dense
persistent state tensor for the correctness gate. The next milestone packs
nonempty node states and groups packets by current owner so dormant cell/node
transforms are not invoked.

### Gate F: hard top-1 and packed engine

- Hard-route behavior remains capable.
- Reference/packed equivalence passes for fixed selections.
- Peak activation growth remains nearly flat as total cells rise.
- Sparse execution beats the reference at measured support.

### Gate G: language

Only after the prefix and sparse-engine gates pass should Path enter the
language shell.

## 15. Rollback milestones

Each row must be a separate executable commit:

| Milestone | Required artifact | Commit |
|---|---|---|
| Arrow fallback | Compile failures documented; strict eager fallback bundled | `187ed97` |
| P0 | Corrected spec and milestone ledger | `c9cb416` |
| P1 | Topology and validated packet state | `37afb07` |
| P2 | Sparse mailbox and local routing primitives | `3ce1293` |
| P3 | Executable synchronous relay/model smoke | `ec9fe67` |
| P4 | Single-cell parallel-prefix trainer | `914bec4` |
| P5 | Ten-cell ring reproduction | pending |
| P6 | Hard top-1 route A/B | pending |
| P7 | Bounded packed engine and equivalence | pending |
| P8 | Language-shell integration | pending |

If a later milestone fails, return to the last passing commit without modifying
Arrow history.

## 16. Transferable optimization work

Retain from the Arrow performance program:

- bounded packed workspaces,
- top-k selection,
- grouped active-cell transforms,
- fused segmented reductions,
- host-barrier removal,
- optimizer fusion,
- allocation/liveness discipline,
- checkpoint/recompute techniques.

Defer or replace:

- Arrow convolution receiver fusion,
- all-cell dense-prefix staging,
- dense destination-node materialization,
- global replicated ownership tensors.

## 17. Immediate implementation order

1. Fixed topology and legal action tables.
2. Persistent current-frontier packet state with no route history.
3. Dense proof mailbox followed by sparse occupied-location mailbox.
4. Dense-to-deterministic local router with real forward participation.
5. Selected edge-plus-destination transform.
6. Synchronous reference relay.
7. Minimal simultaneous model/readout.
8. Single-cell prefix trainer.
9. Ten-cell degree-four ring.
10. Learned dense-to-deterministic ingress and hard deployment choice.
11. Bounded sparse dispatch.
12. Diagnostic trace inspection.
13. Language shell.

The first proof obligation is not language. It is whether an actual packet-path
model can reproduce the generated parallel-prefix result while avoiding
Arrow's dense replicated state.
