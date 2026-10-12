# GTR-L architecture

Status: **implemented and under active language-model validation**

Primary implementation:

- `gator_linear/core.py` — streams, nodes, relays, routing, sequence mixing,
  and the recurrent block cache
- `gator_linear/prefix.py` — token shell, simultaneous output slots, coalition
  readout, and cache lifecycle
- `gator_linear/evotoken.py` — continuous soft-token refinement
- `gtrl_evotoken_colab.py` — current language-training launcher

Related documents:

- `Papers/GATOR_LINEAR_DESIGN.md` — original linear-scaling design history and
  bring-up findings
- `Papers/GATOR_LINEAR_EVOTOKEN.md` — EvoToken experiments and diagnostics
- `Papers/GTRL_CONFIGS.md` — measured production-size configurations
- `Papers/GTRL_INTEL_ARC_B70_ROADMAP.md` — Intel GPU optimization roadmap

## 1. What GTR-L is

GTR-L is a Gator language architecture designed to predict a **block of
tokens simultaneously** while avoiding the context-quadratic receiver
operator used by GTR-Convergent.

Its design principle is:

> **Dense hypotheses, linear communication.**

GTR-L keeps several private interpretations of every sequence position. These
are called **streams**. Each stream also carries a soft or hard distribution
over a shared set of **nodes**. Streams communicate by writing messages into
node mailboxes, mixing those mailboxes along the sequence, and reading them
back through their own node occupancies.

The private stream states are never averaged together inside the core. They
remain separate through every relay and are combined only by a learned
coalition readout at the end.

For a batch `B`, sequence length `L`, stream count `S`, model width `D`, and
node count `N`, the two central state tensors are:

```text
private hidden state H : [B, L, S, D]
node occupancy Q       : [B, L, S, N]
```

`H` says what each stream currently believes. `Q` says where each
position/stream pair is working in the shared node graph.

## 2. Terminology

| Term | Meaning |
|---|---|
| **slot** | One sequence position. Output slots are future positions predicted together. |
| **stream** | A private, persistent interpretation of every slot. A stream is closer to a parallel solver than to an attention head. |
| **node** | A shared routing, mailbox, and program coordinate. A node is not a full `D`-wide network and does not own a private hidden state. |
| **degree** | Number of legal outgoing graph edges per node, including the self-loop in the implemented graph layouts. |
| **bus** | A low-rank, per-stream global route that can move occupancy beyond local graph neighbors. |
| **relay** | One sequential GTR-L processing layer: write, mix, read, route, transport, transform, and update. Relays have distinct parameters. |
| **mailbox channel** | One independently gated message role within a node mailbox. |
| **EvoToken phase** | One prediction/evolution step. In relay mode, phase `r` reads out after relay `r`. |
| **promotion** | Moving an output slot from mask-aware soft state to pure soft-token state. |
| **active block** | The output chunk currently being predicted and refined. |
| **committed prefix** | Previously accepted discrete token blocks represented by recurrent cache state. |

The distinction between nodes and streams is important:

- increasing **streams** adds more private hypotheses and increases the
  `[B,L,S,D]` state;
- increasing **nodes** expands the shared routing/program space and increases
  the `[B,L,S,N]` occupancy;
- increasing **width** increases the representational size of every stream;
- increasing **relays** adds sequential processing depth and a new set of
  relay parameters.

## 3. End-to-end dataflow

```text
observed token IDs                  K output proposal slots
        |                           (MASK/query/soft token)
        +-------------+----------------------+
                      |
        token + slot-type + address embeddings
                      |
            add a learned stream identity
                      |
              H0 [B, L, S, D]
                      |
        broad learned node ingress -> Q0
                      |
          +-----------v-----------+
          |       relay 1         | -> optional stage-1 logits
          +-----------+-----------+
                      |
          +-----------v-----------+
          |       relay 2         | -> optional stage-2 logits
          +-----------+-----------+
                      |
                     ...
                      |
          +-----------v-----------+
          |       relay R         | -> final-stage logits
          +-----------+-----------+
                      |
       per-stream adapter + learned fusion weights
                      |
             one shared vocabulary head
                      |
              logits [B, K, V]
```

The ordinary one-pass path reads logits only after the last relay. EvoToken
relay mode reuses the same readout after every relay and may inject the change
in soft-token belief before the next relay.

## 4. Input shell and output slots

For language modeling, an example is split into:

```text
observed prefix: [B, observed_length]
future target:   [B, output_slots]
```

The complete training window is:

```text
context_length = observed_length + output_slots
```

Every observed token is embedded normally. Each active output slot begins
from either:

- a learned output query in the one-pass path;
- the MASK token embedding;
- or an EvoToken soft embedding derived from the model's current token
  distribution.

Every slot also receives:

1. an observed/proposal slot-type embedding;
2. a projected immutable sinusoidal address;
3. a learned stream embedding.

This produces one private `D`-wide state for every `(slot, stream)` pair.
Logical output addresses can be permuted together with their targets for
positional-starvation experiments, but the normal language path uses their
natural left-to-right order.

## 5. Initial node occupancy

The core projects each private hidden state against learned node ingress keys:

```text
ingress_logits[b,l,s,n]
    = dot(H[b,l,s], ingress_key[n])
    + node_bias[n]
    + stream_node_bias[s,n]

Q0 = softmax(ingress_logits / ingress_temperature)
```

The desired ingress is **broad but not uniform**:

- broad occupancy lets every node receive gradient;
- structured occupancy lets different positions and streams address different
  node subsets.

Perfectly uniform occupancy is degenerate because every node receives nearly
the same content and the graph loses its ability to address information.

## 6. One relay

Each relay advances both the private hidden state and node occupancy. It
performs eight conceptual operations.

### 6.1 Write private messages

Each stream projects its private state into `C` mailbox channels of width
`Dm`:

```text
message: [B, L, S, C, Dm]
```

The shared projection is supplemented by optional low-rank private
stream adapters. A learned write gate determines how strongly each stream
uses each message.

Messages are deposited according to the current occupancy:

```text
node_input[b,l,n,c,m]
    = sum_s Q[b,l,s,n] * message[b,l,s,c,m]
```

The deposit is normalized by mean node mass. This restores useful signal
scale without erasing the relative mass differences that make nodes
addressable.

### 6.2 Mix mailboxes along the sequence

For every node and mailbox channel, `LinearSequenceMixer` builds four context
views:

1. **local convolution** — fixed-width depthwise convolution;
2. **forward decayed scan** — learned timescales from local to context-wide;
3. **reverse decayed scan** — future context when the active region permits
   it;
4. **global summary** — whole-sequence or prefix/block summary.

The four views are transformed and summed without constructing an
`[L,L,N]` receiver-by-writer tensor.

The decayed forward scan has the form:

```text
state[l] = decay * state[l-1] + input[l]
```

Different channels begin with geometrically distributed timescales, allowing
some channels to specialize in nearby evidence and others to retain
long-range evidence.

### 6.3 Read node memory

Each stream reads the mixed mailbox through its own occupancy:

```text
read[b,l,s,c,m]
    = sum_n Q[b,l,s,n] * node_memory[b,l,n,c,m]
```

Per-stream channel gates, a shared projection, optional private low-rank read
adapters, and LayerNorm turn the mailbox read into a `D`-wide context vector.

This is communication through a residual mailbox. It is not hidden-state
averaging: every stream retains its own private state.

### 6.4 Optional linear attention

When enabled, the relay adds a content-addressed, retention-style read. It
maintains decayed key/value sufficient statistics and queries them using the
current hidden state.

This supplies content-dependent retrieval without materializing a dense
attention matrix. In the cached path its key/value and key-normalizer states
become part of the relay cache.

### 6.5 Score legal graph edges

The router receives:

```text
private hidden + mailbox context + immutable address
```

Optionally it also receives a running node-occupancy summary. It projects this
state to a small router rank and scores only the `degree` legal outgoing
edges of each source node.

For the usual constant-degree graph:

```text
edge probabilities: [B, L, S, N, degree]
```

`degree` therefore really is graph connectivity. It controls how many local
destinations each node can choose on one relay. Increasing degree improves
one-hop connectivity but increases routing work and memory. It does not make
the node itself wider.

### 6.6 Move occupancy locally and globally

Local transport scatters source-node mass along the legal graph edges. A
separate low-rank global bus supplies long-distance movement:

```text
nodes -> M buses -> nodes
```

The buses are per stream, preventing every stream from receiving the same
homogenized global redistribution. A learned gate mixes local graph transport
with global bus transport. The resulting occupancy is renormalized.

Routing may remain soft or use a straight-through hard commitment:

```text
forward:  one-hot argmax occupancy
backward: soft occupancy gradient
```

Current production-size GTR-L configurations enable hard routing.

### 6.7 Select a node program

Nodes do not each own a full `D x D` transform. Instead, every relay owns a
small shared basis of `basis_rank` transforms. Each node owns only a
coefficient vector over that basis.

The new occupancy selects the expected coefficients:

```text
coeff[b,l,s,r] = sum_n Q_next[b,l,s,n] * node_coeff[n,r]
```

The relay evaluates the shared basis on the private signal and combines the
results using those coefficients. This preserves node-specific computation
without paying for `N` complete width-by-width transforms.

### 6.8 Update private state

The selected program produces a normalized delta. A learned gate applies it
residually:

```text
H_next = LayerNorm(H + sigmoid(update_gate(H)) * delta)
```

Optional heterogeneous-stream masks can restrict different streams to
different active widths. The mask is reapplied after each relay so LayerNorm
cannot repopulate inactive dimensions.

## 7. Controlled writing

Output proposal slots can read from relay 1, but their early contents are
not yet trustworthy enough to write into shared memory.

With controlled writing enabled:

- observed/committed tokens write at full strength;
- active output slots start with write strength zero;
- output write strength increases across relays;
- reads, routing, and private residual updates remain active throughout.

For `R` relays, the default output write fractions are approximately:

```text
0 / (R-1), 1 / (R-1), ..., (R-1) / (R-1)
```

For four relays this is:

```text
0.00, 0.33, 0.67, 1.00
```

This prevents blank or mask-like proposal states from polluting the shared
mailboxes before they have absorbed evidence.

## 8. Coalition readout

After the final relay—or after every relay under EvoToken—GTR-L reads the
active output slots.

Each stream first receives a learned full-width adapter. A scalar fusion head
then assigns softmax weights across streams independently at every slot:

```text
fusion_weight: [B, L, S]
fused_output:  [B, output_slots, D]
```

Only after fusion is the vocabulary projection applied:

```text
logits: [B, output_slots, vocabulary]
```

This avoids paying for a separate full vocabulary projection for every
stream. It also makes streams a coalition: each stream may be imperfect alone
while their learned combination is stronger.

The vocabulary head can share its weight with the input token embedding. Tied
embeddings:

- remove a second `[vocabulary, D]` parameter matrix;
- keep input embeddings, EvoToken feedback embeddings, and output
  classification in the same token geometry;
- are enabled in the current 100M and 500M configurations.

## 9. Causality modes

`causal` and `block_causal` are mutually exclusive.

### 9.1 Bidirectional whole-window mode

With both flags disabled, local mixing is symmetric, both decayed scans are
available, and the global summary sees the whole supplied sequence. This is
useful for simultaneous tasks but is not suitable for cached language
generation.

### 9.2 Fully causal mode

With `causal=True`:

- convolution reads only the current and earlier slots;
- the reverse scan is removed;
- the global summary becomes a running prefix mean.

This supports token-causal computation, but EvoToken refinement rejects it
because active output positions need bidirectional interaction.

### 9.3 Block-causal mode

With `block_causal=True`, the sequence is partitioned into fixed blocks whose
width equals the generated chunk size:

- a block can read every committed earlier block;
- positions inside the current block can interact bidirectionally;
- information cannot flow backward from a later block into an earlier block;
- forward recurrent state continues across blocks;
- reverse scan state resets at every block boundary.

This is the intended language-generation mode. It combines a causal history
with simultaneous refinement inside the active output chunk.

For a four-token Gator:

```text
committed block 0 -> committed block 1 -> ... -> active block
                                                   [t, t+1, t+2, t+3]
                                                        bidirectional
```

The observed length must be divisible by `output_slots` in the current
vectorized training and cache implementation.

## 10. EvoToken refinement

EvoToken is a refinement policy layered on top of GTR-L. It does not add
learned model parameters.

The state trajectory is:

```text
MASK -> mask-aware soft token -> pure soft token -> hard decoded ID
```

### 10.1 Sparse soft-token belief

At each phase, the shared vocabulary head produces logits. Illegal tokenizer
classes are masked, the top `K` legal candidates are retained, and their
probabilities are renormalized.

The distribution embedding is:

```text
soft_embedding
    = sum_candidate probability[candidate]
                    * token_embedding[candidate]
```

An unresolved slot receives a mixture of MASK and the soft distribution:

```text
mask_aware = alpha * mask_embedding
           + (1 - alpha) * soft_embedding
```

A promoted slot receives the pure `soft_embedding`.

Candidate IDs and probabilities are detached between phases. The live token
embedding table still receives ordinary input-path gradient, but gradient does
not pass through top-k selection or back through an earlier phase's
probability distribution.

### 10.2 Relay-integrated mode

`mode=relay` treats each relay as one EvoToken phase:

```text
relay r
  -> stage-r logits
  -> top-k soft belief
  -> scheduled promotion
  -> inject embedding delta
  -> relay r+1
```

Hidden state and occupancy continue across phases. The injected soft-token
delta can therefore affect later mailbox writes, routing, and node-program
selection without restarting the model.

For `R` relays this costs `R` relay evaluations, plus intermediate vocabulary
readouts and top-k evolution. Current code requires:

```text
trajectory_steps == relays
```

### 10.3 Full-pass mode

`mode=full_pass` reruns the complete relay stack after every token evolution:

```text
full GTR-L pass -> evolve embeddings -> full GTR-L pass -> ...
```

For four relays and four phases this costs 16 relay evaluations. It is a
paper-fidelity control, not the preferred GTR-L execution path. Current
experiments found the compounding slowdown large and did not find a decisive
quality improvement.

### 10.4 Promotion counts and order

`promotion_counts` specifies how many unresolved output positions become pure
at each phase:

```text
length(promotion_counts) == trajectory_steps
sum(promotion_counts)    == output_slots
```

Examples for four slots and four phases:

```text
(1, 1, 1, 1)  balanced default
(2, 1, 1, 0)  all promotions occur before the terminal phase
(2, 2, 0, 0)  two early promotion groups
(4, 0, 0, 0)  all slots receive one early correction opportunity
```

During training, the default order is randomized independently for each batch
row. At inference, the default order promotes the most confident unresolved
positions. Counts and ordering are separate controls.

The launcher exposes them as:

```text
--promotion-counts 4,0,0,0
--promotion-schedule-mode random
--inference-promotion-mode confidence
```

### 10.5 Terminal-promotion limitation

The implemented ordering inside a relay phase is:

```text
relay -> logits -> evolve/promote -> optional feedback into the next relay
```

Feedback is injected only when another phase exists. A promotion assigned to
the final phase can update the historical-best decoded bookkeeping, but its
new embedding is never consumed by another relay.

Therefore `(1,1,1,1)` has only three usable feedback transitions. The fourth
promotion does not receive a correction pass. Front-loaded schedules avoid
wasting the last transition.

Early experiments also indicate that GTR-L often obtains nearly all of its
terminal token accuracy within the first one or two phases. This is an active
finding rather than a guaranteed architectural property. Any early-exit
decision should compare:

- mean token accuracy;
- exact full-block accuracy;
- per-position accuracy, especially tail slots;
- modal-token share and unigram lift;
- revision rate and the fraction of revisions that are actually corrective.

### 10.6 Training objective

The current continuous-trajectory objective supervises logits from every
phase. In the all-mask CTS regime, every output position begins masked, so
the model cannot reduce its loss by copying initially visible ground-truth
targets.

The trainer can additionally apply:

- position-balanced weights;
- a smooth worst-position term;
- tail-only loss;
- logical/physical slot permutation;
- exact per-position gradient diagnostics.

These are experimental allocation controls, not required parts of the GTR-L
core.

## 11. Block cache and parallel generation

GTR-L uses a recurrent **block cache**. It is called a block KV cache by
analogy with Transformers, but it stores every sufficient statistic required
by the linear mixer, router, and optional linear attention.

Each relay cache contains:

- the local convolution tail;
- forward decayed-scan state;
- accumulated node-mailbox sum;
- optional linear-attention key/value and key-normalizer states;
- optional accumulated occupancy summary.

The top-level cache also stores:

- the absolute position of the next block as a scalar tensor;
- the fixed block size;
- one cache entry per relay.

The cache is constant-size with respect to the number of committed tokens.
It does not retain every prior token's hidden state or key/value vectors.

### 11.1 Generation lifecycle

```text
1. prefill_cache(committed_ids)
      Encode complete committed blocks.

2. forward_cached(cache, proposal embeddings)
      Refine one active block against the immutable committed cache.
      The proposal does not mutate history.

3. choose/accept discrete IDs

4. append_cache(cache, accepted_ids)
      Recompute the accepted IDs as ordinary committed tokens with full
      write strength and return a new cache atomically.

5. repeat from step 2
```

Mask embeddings, soft proposal states, and controlled-writing states are
never committed as language history.

The explicit commit pass costs one additional short relay pass per accepted
block. The payoff is that the next `output_slots` tokens are proposed
together rather than one token at a time.

### 11.2 Current cache limitations

- Cache eviction is not implemented. Once `context_length` is reached, the
  prefix must be rebuilt or a future approximate/sliding state design must be
  introduced.
- Prefill currently requires complete blocks.
- Partial natural-language prompts need an explicit partial-block policy;
  silently padding would alter recurrent sums.
- The current fixed-boundary trainer does not by itself train every possible
  proposal position. Robust multi-block generation needs variable-prefix or
  unrolled-block training.

## 12. Scaling behavior

GTR-L was created to remove three expensive all-pairs operators:

| Dense construction avoided | GTR-L replacement |
|---|---|
| all sequence-position pairs | local convolution + recurrent decayed scans + summaries |
| all source/destination node pairs | fixed-degree graph + low-rank buses |
| all stream pairs | shared node mailboxes |

With degree, channel counts, ranks, and convolution width held fixed, the
communication machinery scales linearly in sequence length and node count.
The complete cost still contains products such as `L*S*N`; “linear” means
linear in each scaling axis when the others are fixed, not constant work.

### 12.1 What width changes

Width is the expensive capacity axis:

- every stream state is `D` wide;
- relay program-basis transforms contain `D x D` matrices;
- the coalition stream adapters contain full-width matrices;
- vocabulary embedding/head cost is proportional to `V*D`.

Parameter count and dense-matmul work therefore grow roughly quadratically
with width once the `D x D` terms dominate.

For the current tied-vocabulary, 16-stream, 128-node, four-relay production
topology:

```text
parameters(D) = 36*D^2 + 21,188*D + 140,773
```

This gives:

```text
D=1,392 ->  99,390,373 parameters
D=3,440 -> 499,037,093 parameters
```

That formula is configuration-specific. Changing streams, relays, ranks,
nodes, or vocabulary changes its coefficients.

### 12.2 What nodes change

Increasing nodes primarily adds:

- occupancy activation state `[B,L,S,N]`;
- edge scoring and transport over `N*degree`;
- node program coefficients;
- ingress keys and biases;
- per-stream bus factors;
- mailbox work for each node.

It does **not** allocate a separate `D`-wide hidden state or full `D x D`
network per node. Node scaling is therefore much cheaper in parameters than
width scaling, although occupancy and mailbox activations can still consume
substantial VRAM at long context.

### 12.3 What streams change

Increasing streams adds another private hidden trajectory across the complete
sequence. It increases:

- `[B,L,S,D]` private activations;
- `[B,L,S,N]` occupancy;
- private write/read adapters and routing biases;
- per-stream buses;
- full-width coalition stream adapters.

Streams are therefore not free ensemble members. Their value depends on
remaining complementary rather than converging to duplicate solutions.

### 12.4 What relays change

Relays are independently parameterized layers. Reducing relay count reduces:

- program-basis matrices;
- write/read/router projections;
- sequence mixers;
- graph and bus routing parameters;
- optional linear-attention projections;
- both compute depth and parameter count.

This is why relay count affects GTR-L model size. It is not merely a request
to repeat one shared layer.

### 12.5 What slots and context change

`output_slots` and `context_length` do not directly add learned parameters.
They change:

- activation memory;
- sequence-mixing work;
- output-head work;
- cache block size;
- the difficulty of learning later future positions;
- exact-sequence probability.

More slots increase potential decode parallelism, but they also move the tail
farther into the future. If later positions depend on earlier generated
positions, high block accuracy requires the front slots to become extremely
reliable and the model to use active-block communication effectively.

## 13. Dominant operations on a GPU

GTR-L is a mixed workload:

1. **large BF16 dense contractions**
   - shared node-program basis transforms;
   - shell stream adapters;
   - vocabulary embedding/readout;
   - write/read and routing projections;
2. **long recurrent or scan-like operations**
   - forward/reverse decayed sequence scans;
   - optional linear-attention scans;
3. **bandwidth-sensitive tensor operations**
   - occupancy-weighted mailbox writes and reads;
   - route scatter-add;
   - normalizations, gates, and elementwise state updates;
4. **small reductions and softmaxes**
   - degree-local route choice;
   - node buses;
   - stream fusion;
   - EvoToken top-k selection.

At large width, dense BF16 matmuls dominate arithmetic. At large context,
stream/node activations, scans, and data movement become increasingly
important. Good hardware therefore needs both strong matrix throughput and
adequate memory bandwidth/capacity; peak tensor-core throughput alone does not
describe the workload.

Current NVIDIA runs use BF16 autocast and full-graph `torch.compile`. Intel
support requires backend-specific scan, scatter, and fusion work described in
`Papers/GTRL_INTEL_ARC_B70_ROADMAP.md`.

## 14. Comparison with a decoder-only Transformer

| Property | Decoder-only Transformer | GTR-L |
|---|---|---|
| Primary state | one hidden state per token/layer | several private stream states plus node occupancy |
| Token interaction | dense/self attention or another token mixer | node mailboxes, local convolution, decayed scans, optional linear attention |
| Node routing | none | fixed-degree graph plus low-rank global buses |
| Layer computation | feed-forward network per layer | occupancy-selected shared program basis per relay |
| Output generation | normally one token at a time | one active block in parallel |
| Standard cache | key/value history grows with committed length | recurrent sufficient state is constant-size in committed length |
| Active future block | causally ordered | bidirectional inside a block, causal across blocks |
| Vocabulary head | one shared head | one shared head after stream coalition fusion |

GTR-L's parallel block output is a potential latency advantage, not a
guaranteed speedup. It performs more work per sequence position than a single
Transformer stream, and accepted blocks require a commit pass. The advantage
is realized only when block accuracy is high enough that several useful
tokens are obtained per proposal.

## 15. Current architectural risks

### 15.1 Positional starvation

Later output slots can remain weak even when shared width or node count is
increased. This suggests that capacity may be allocated preferentially to
easier front positions rather than reaching the tail.

The current diagnostic suite distinguishes:

- fixed physical-slot weakness;
- logical future-distance difficulty;
- promotion-order bias;
- shared-loss gradient competition;
- weak or conflicting tail gradients.

### 15.2 Refinement saturation

Recent experiments show mixed gains from EvoToken phases, no decisive gain
from full-pass refinement, and little extra token accuracy after roughly one
or two relay phases. This may mean:

- the masked continuous-feedback objective is more valuable than repeated
  refinement;
- later relays are useful as architectural depth but not as additional token
  evolution phases;
- or stage-average accuracy is hiding smaller exact/tail improvements.

Front-loaded promotion and stage-2 early-exit evaluation are the cheapest
ways to separate these explanations.

### 15.3 Degenerate token modes

Language runs must be monitored for punctuation, whitespace, or frequent-word
collapse. Mean accuracy alone is not enough. The current trainer reports:

- modal token and modal share;
- target frequency of that modal token;
- lift over the modal baseline;
- weak positions;
- per-position accuracy;
- exact block accuracy;
- stage-wise accuracy and top-3 accuracy.

EvoToken-trained GTR-L has so far been more resistant to the immediate
degeneracy observed in earlier autoregressive Gator variants, but this remains
an empirical result rather than a proof of stability.

### 15.4 Empty-node gradients

Hard routing can leave node mailboxes exactly empty. Normalizing an empty
linear-attention accumulator by a tiny epsilon creates a zero forward value
with an enormous surrogate derivative.

The implementation detects the empty branch, returns an exact zero
contribution with zero content gradient, floors occupied low-mass
denominators, and retains global gradient clipping as a final guard.

## 16. Configuration map

The most consequential architecture settings are:

| Setting | Primary effect |
|---|---|
| `vocab_size` | embedding/head size and classification difficulty |
| `width` | stream capacity; dominant parameter and BF16 matmul scaling axis |
| `streams` | number of private interpretations |
| `nodes` | routing/program address space |
| `relays` | processing depth and EvoToken relay phases |
| `degree` | local graph connectivity and route-scoring cost |
| `buses` | low-rank global route capacity |
| `mailbox_channels` | number of separately gated communication roles |
| `mailbox_width` | payload width per mailbox channel |
| `basis_rank` | number of shared node-program transforms |
| `router_rank` | low-rank edge-scoring dimension |
| `conv_kernel` | fixed local sequence receptive field per relay |
| `linear_attention` | optional content-addressed recurrent memory |
| `hard_routing` | one-hot forward occupancy with soft backward gradients |
| `controlled_writing` | prevents immature output proposals from writing early |
| `tie_token_embeddings` | shares input embedding and vocabulary classifier |
| `causal` | token-causal mixing; incompatible with EvoToken |
| `block_causal` | causal committed blocks with bidirectional active blocks and cache |
| `output_slots` | generated chunk size; runtime/task setting rather than parameter axis |
| `context_length` | maximum full window and address reference length |

## 17. Architectural invariants

Changes to GTR-L should preserve these properties unless an experiment
explicitly intends to remove one:

1. Private stream hidden states remain separate through the core.
2. Streams communicate through residual messages, not hidden-state averaging.
3. Node support is broad enough for gradient but structured enough for
   addressing.
4. Local routing scores only legal fixed-degree edges.
5. Long-distance routing uses low-rank buses rather than an `N x N` learned
   matrix.
6. Sequence mixing does not materialize an `L x L` relation tensor.
7. Node-specific computation uses a shared program basis rather than one full
   transform per node.
8. Vocabulary projection occurs after coalition fusion, not once per stream.
9. Block-causal proposals never mutate committed cache state.
10. Only accepted discrete tokens are appended to language history.

