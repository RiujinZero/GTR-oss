# Gator Language Experiment 1: Simultaneous Response Formation

Architecture definitions and invariants are maintained in
[`../ARCHITECTURE.md`](../ARCHITECTURE.md).

## Question

Can a small GTR-ARW cell network learn held-out linguistic structure while
forming a response window simultaneously rather than decoding autoregressively?

## Dataset

SmolTalk Everyday contains 2,260 conversations, 17,504 dialogue turns, and
8,624 assistant responses. The deterministic seed-zero split is performed by
whole conversation:

- Training: 2,034 conversations and 5,726 scored assistant responses
- Validation: 226 conversations and 638 scored assistant responses

No assistant turn from a validation conversation is used to train either the
model or tokenizer.

The overnight mixture adds LMSYS Chat 1M at a 90% sampling weight. Its first
512 conversations (954 assistant responses in the inspected copy) are held out
for validation. Remaining conversations are packed into 64-token contexts and
32-token response targets in CPU memory. SmolTalk supplies the other 10% so its
short everyday style remains represented without dominating the run.

A full integrity scan found 346 malformed LMSYS rows containing empty role
turns (0.035% of the million-row file). The LMSYS adapter skips those rows;
strict parsing remains enabled for SmolTalk and other curated inputs.
The supplied file contains 983,244 valid conversations and 1,973,235 assistant
responses after filtering. Packing all fixed windows should occupy roughly
0.76 GB of CPU tensor storage.

### Context-window audit

The apparent sentence fragments in logged LMSYS contexts are often created by
our packer rather than stored that way in the source row. The current collator
encodes every preceding turn and then takes exactly:

```python
context_ids[-64:]
```

For a long prompt this removes its beginning, potentially including the BOS and
`<user>` role marker. Raw rows that logged as fragments such as an address tail
or `and Downstream products...` begin with complete instructions in the source
file.

A July 16 audit of the first 10,000 LMSYS rows found 20,227 assistant examples:

| Preceding text length | Fraction of assistant examples |
|---|---:|
| More than 256 characters | 60.2% |
| More than 512 characters | 50.0% |
| More than 1,024 characters | 40.3% |
| Immediately preceding user turn over 256 characters | 20.0% |
| Immediately preceding user turn over 512 characters | 11.6% |

Characters are not tokenizer tokens, so these are diagnostic rather than exact
truncation percentages. They nevertheless establish that a large fraction of
the corpus is structurally mismatched to a 64-token context. The largest
observed preceding context in this sample was 566,002 characters.

This means the private-hash and communal 62-cell runs do not cleanly test
language architecture alone. At a 90% LMSYS sampling weight they also test
whether a 64-token model can learn from millions of tail-cropped, heterogeneous
prompts. Lack of improvement must not yet be assigned solely to Gator topology
or lexical organization.

### Selected short-row control

The next run uses a deliberately simple rule:

- Context allocation: 512 tokens
- LMSYS row filter: skip the complete decoded `text` field when it exceeds 512
  characters
- Response allocation: 32 simultaneous tokens
- Lexical mode: communal
- Node topology: symmetric
- Cells: 62
- VRAM cap: 50 GB

Filtering occurs before dialogue parsing, tokenizer sampling, packing, and
train/validation construction. The first 512 eligible filtered conversations
are validation data; training and tokenizer construction skip those same
eligible conversations rather than merely skipping 512 raw file lines.

At context 64, response 32, and batch 32, the observed allocation was 5.4 GB.
Receiver-specific memory is quadratic in total slots, so the launch estimate is:

\[
5.4\ \mathrm{GB}\times\frac{8}{32}\times
\left(\frac{512+32}{64+32}\right)^2
\approx 43.4\ \mathrm{GB}.
\]

Batch 8 was therefore selected as the initial cap-aware setting. The launch command also
sets `--max-vram-gb 50`, which limits PyTorch's caching allocator to the
corresponding fraction of physical GPU memory. Record actual peak VRAM and
reduce the batch immediately if compilation or non-allocator CUDA overhead
approaches 50 GB. The raw 512-character rule does not mathematically guarantee 512 or
fewer tokenizer IDs for arbitrary Unicode, but it removes the extreme long-row
tail-cropping problem with a transparent, reproducible filter.

The actual operator-observed allocation for the compiled 512-context, 32-slot,
batch-8 run was approximately **17 GB**, far below the 43.4 GB estimate. The
estimate incorrectly treated the full earlier 5.4 GB observation as though all
of it scaled with the quadratic receiver-memory term and assumed every
theoretical intermediate remained resident. Fixed allocations, buffer reuse,
and compiled lifetime planning make that model too pessimistic. Use 17 GB as
the empirical reference for this exact configuration; preserve the old
calculation only as a documented upper-bound error.

The first assistant response is excluded from every SmolTalk conversation in
the corrected run. The original corpus repeats `Hello! How can I help you today?`
2,022 times in training and 226 times in validation, allowing a context-free
template to obtain approximately 26% exact accuracy.

## Tokenization

The historical compact tokenizer had 2,048 entries:

- Six structural tokens
- 256 lossless UTF-8 byte fallback tokens
- 1,786 frequent text pieces learned only from the training split

This avoids a large general-purpose vocabulary and guarantees that validation
text never becomes unknown.

Tokenizer format version 2 now reserves the final requested vocabulary row for
one instance-specific, non-rendering BLANK without renumbering version-1
special, byte, or retained learned-piece IDs. At requested vocabulary 2,048 it
contains six original structural IDs, 256 byte IDs, 1,785 learned pieces, and
one BLANK. Gator initializes output queries from BLANK. Version-1 checkpoints
remain loadable but cannot silently use alignment training.

## Initial wide-cell model

| Property | Value |
|---|---:|
| Parameters | 130,000 |
| Gator cells | 10 |
| Cell neighbors | 4 |
| Graph nodes per cell | 4 |
| Relay hops | 4 |
| Hidden width | 32 |
| Context window | 64 tokens |
| Simultaneous response window | 32 tokens |
| Vocabulary | 2,048 tokens |

The token embedding is shared across cells. Each cell retains its own graph
router, node transformations, immutable address projections, relative memory
bias, and route state. This prevents vocabulary embeddings from being repeated
ten times while preserving specialized graph computation.

## Stitched postage-stamp model

The overnight LMSYS mixture uses a split-width design. A shared 32-dimensional
language shell retains vocabulary capacity, while the graph computation returns
to the original eight-dimensional postage-stamp width.

| Component | Parameters |
|---|---:|
| Shared language shell | 68,136 |
| Shared coupling projections and norm | 160 |
| One literal width-8, four-node Gator cell | 980 |
| 62 stitched cells | 60,760 |
| Per-cell coupling gates | 248 |
| **Total** | **129,304** |

This increases the network from 10 cells and 40 node instances to 62 cells and
248 node instances without increasing the model budget. Each cell retains its
own routes, nodes, address projections, relative memory, and gates. Vocabulary
embeddings remain shared because duplicating a 2,048-token table inside every
postage-stamp cell would spend parameters on lexical storage rather than graph
computation. A private 16-by-8 lexical-prototype table can remain inside every
cell and be selected by `token_id % 16`; this preserves the original math
cell's 128-parameter lexical component while the shared embedding supplies
full-vocabulary meaning. At vocabulary 2,048, however, it forces 128 unrelated
tokens into each private bucket. The `--lexical-mode communal` A/B removes that
hashed overlay and substitutes one learned identity vector per cell. With all
else fixed at 62 cells, the communal model has 121,864 parameters rather than
129,304. The first promising wide language pilot used communal vocabulary.

The scaling rule is deliberately conservative: preserve the validated width-8,
four-node, four-relay 980-parameter cell and add more cells until an experiment
demonstrates that changing the local unit is beneficial.

### Node-mask A/B and hybrids

The successful prefix cells used the original asymmetric four-node mask, while
the wide language pilot that produced multiple grammatical, contextually
plausible formations used a symmetric cyclic mask. Node topology is therefore
an explicit zero-parameter A/B:

- `asymmetric`: the validated staged-routing mask
- `symmetric`: equal self/left/right connectivity for every node
- `hybrid_alternating`: 31 symmetric and 31 asymmetric cells interleaved
- `hybrid_hemispheric`: one contiguous 31-cell hemisphere of each type

Alternating hybrid maximizes local contact between road systems. Hemispheric
hybrid allows each topology to cooperate internally and exchange information at
the two boundaries of the cell ring. The all-symmetric run comes first.

## Objective and evaluation

Every configured output query is initialized from BLANK and predicted in one
forward pass. The first alignment screen uses 40 physical output slots with at
most 32 reference pieces, leaving eight real placement slots. The model does not
observe its own intermediate token predictions. PAD,
BOS, MASK, USER, and ASSISTANT are forbidden response classes; EOS, BLANK,
bytes, and learned text remain legal. A clipped target is explicitly tracked as
a right-censored prefix with no synthetic EOS.

The implemented objective generation exposes five ordered gates, with each
previous stage retained as a matched A/B and GPU evidence still pending:

1. Preserve the tested context/target padding, complete-versus-censored EOS,
   whitespace, rendering, and response-vocabulary semantics.
2. Test rigid content/EOS plus BLANK-tail cross-entropy before allowing
   alignment freedom.
3. Compare Viterbi/forward-sum best-monotonic-aligned diagnostics under the
   rigid controls.
4. Compare detached Viterbi aligned CE and differentiable monotonic forward-sum
   training with those controls.
5. Add the small whole-sequence semantic-composition, length, and EOS term as an
   independent low-weight A/B.

Monotonic alignment may insert BLANK anywhere but must consume every reference
token exactly once and in order. It does not use repeat-collapsing CTC
semantics. `reference_tokens` can be lower than physical `response_tokens` so
long examples retain actual placement slack; the first full-stack launcher
uses 32 reference pieces across 40 output slots. The full design is specified in
[`GATOR_FORMATION_ALIGNMENT_ROADMAP.md`](GATOR_FORMATION_ALIGNMENT_ROADMAP.md).

Report every 2,000 steps over the full validation set:

- Held-out token accuracy
- Exact response-window accuracy
- Held-out cross-entropy and perplexity
- Rigid and best-aligned loss/accuracy, BLANK statistics,
  complete-response EOS accuracy, and post-EOS violations
- One randomly selected context, target, and formed response
- Learning rate and gradient norm

Use cosine learning-rate decay from 0.003 to 0.0001. The first run accidentally
used the optimizer API's constant-schedule default, so its printed 0.003 value
was genuinely flat rather than a logging error.

The first scientific signal is sustained held-out loss reduction below the
uniform baseline of `ln(2048) = 7.625`. Coherent spelling, local syntax, and
slot-to-slot agreement are stronger secondary signals. Exact accuracy is
expected to remain near zero early because all target positions must agree.
Until the verification gate passes, apparently short or empty output must be
inspected with diagnostic rendering that exposes every structural prediction.

## Recorded language runs

The following table separates completed measurements, interrupted screens, and
active experiments. Exact accuracy includes very short targets, so it must not
be interpreted as complete 32-slot formation without inspecting target length.

| ID | Cells / width | Lexical mode | Node mask | Budget | Final or latest held-out result | Status |
|---|---:|---|---|---:|---|---|
| LANG-WIDE-COMMUNAL-S0 | 10 / 32 | Communal | Symmetric | 20k | approximately 43% token accuracy, 26.6% exact, loss approximately 3.17, PPL approximately 24 | Completed pilot; greeting-confounded |
| LANG-62-HASH-ASYM-S0 | 62 / 8 | `private_hash` | Asymmetric | 100k | 27.5% token, 0.6% exact, loss 3.7253, PPL 41.48 | Completed |
| LANG-62-HASH-SYM-S0 | 62 / 8 | `private_hash` | Symmetric | 42k observed | 27.3% token, 0.6% exact, loss 3.7499, PPL 42.52 | Interrupted screen |
| LANG-62-COMMUNAL-SYM-CTX512-S0 | 62 / 8 | `communal`, shared input | Symmetric | Exploratory | Formed only approximately 1–3 words | Stopped sanity run |
| LANG-62-COMMUNAL-SYM-CTX64-SMOL-S0 | 62 / 8 | `communal`, shared input | Symmetric | Exploratory | SmolTalk-only; still formed only approximately 1–3 words | Stopped sanity run |
| LANG-62-COMMUNAL-GATED-CTX64-SMOL-S0 | 62 / 8 | `communal`, feature-gated input | Symmetric | Exploratory | Contextually plausible but wrong formations returned; capped around 4–5 displayed words | Input semantics improved; output remained short |
| LANG-62-COMMUNAL-GATED-GROUPED-CTX64-SMOL-S0 | 62 / 8 | Feature-gated input, four-channel output | Symmetric | 50k complete | Best: 32.7%, loss 3.7461, PPL 42.36; final: 30.2%, loss 4.1408, PPL 62.85 | Learned early, then validation regressed |
| LANG-62-COMMUNAL-GATED-GROUPED-CTX64-SMOL-A-S0 | 62 / 8 | Feature-gated input, four-channel output | Asymmetric | 50k complete | Best loss: 3.7607, PPL 42.98 at 18k; final: 30.7%, loss 4.0844, PPL 59.40 | No peak benefit; later peak and substantial regression |
| LANG-62-COMMUNAL-GATED-GROUPED-CTX64-SMOL-ALT-S0 | 62 / 8 | Feature-gated input, four-channel output | Hybrid alternating | 50k complete | Best: 32.7%, loss 3.7463, PPL 42.36 at 12k; final: 30.1%, loss 4.1271, PPL 62.00 | Matched symmetric peak, faster middle regression |
| LANG-62-COMMUNAL-GATED-GROUPED-CTX64-SMOL-HEMI-S0 | 62 / 8 | Feature-gated input, four-channel output | Hybrid hemispheric | 50k complete | Best loss: 3.7639, PPL 43.12 at 13k; final: 30.4%, loss 4.0411, PPL 56.89 | Slightly higher minimum, strongest retention |
| LANG-62-COMMUNAL-GATED-GROUPED-CTX64-SMOL-HEMI-N8-S0 | 62 / 8 | Feature-gated input, four-channel output | Hemispheric, 8 neighbors | 25k complete | Best loss: 3.7640, PPL 43.12 at 16k; final PPL 44.18 | No floor improvement; scheduler-confounded retention |
| LANG-62-COMMUNAL-GATED-GROUPED-CTX64-SMOL-HEMI-R6-S0 | 62 / 8 | Feature-gated input, four-channel output | Hemispheric, 4 neighbors, 6 relays | 25k complete | Best loss: 3.7495, PPL 42.50 at 15k; final PPL 43.42 | First depth-driven floor improvement |
| LANG-1038-COMMUNAL-GATED-GROUPED-CTX64-SMOL-HEMI-S0 | 1,038 / 8 | Feature-gated input, four-channel output | Hemispheric, 4 neighbors, 4 relays | 25k complete | Best loss: 3.7416, PPL 42.16 at 16k; final PPL 43.02 | New numerical best, but same qualitative basin at ~10x lower throughput |

Gator-S stewardship is implemented as a capacity-matched population transport
at 124,618 parameters, but no held-out language-quality curve has yet been
recorded for it. Its mechanism and dense/packed equivalence checks are
implementation evidence, not an additional row in the language results table.

The wide pilot's exact score was dominated by the repeated greeting and is not
evidence of general language learning. After random sample display was fixed,
however, several held-out predictions were grammatical and contextually
plausible while factually or semantically wrong. Examples included formations
beginning `The main the the...` for an ear-function question and `Yes, there
are several...` for a travel/yoga question. These are qualitative observations,
not a substitute for held-out loss or conditioning controls.

The corrected 62-cell private-hash runs mostly formed one high-frequency first
word followed by visually empty, usually whitespace-filled, or repetitive
slots. Switching the node mask from
asymmetric to symmetric did not materially change held-out metrics through
42,000 steps. This makes the private modulo-16 lexical overlay a stronger
suspect than node symmetry: with vocabulary 2,048, every private bucket aliases
128 unrelated tokens. The first communal run held cell count, width, relays,
neighbors, symmetric mask, data, and optimizer fixed and removed only that
overlay. Its lower parameter count, 121,864 versus 129,304, was intentional. It
did not show an obvious early improvement, which exposed the long-row
tail-cropping confound. The 512-character/512-context control still collapsed
to a few formed words. Removing LMSYS entirely and returning to context 64 did
not restore longer formation in the shared-input 62-cell model. LMSYS messiness
and context length were real pipeline concerns but are not sufficient
explanations for the current collapse.

Inspection exposed a population bottleneck: one shared `32 -> 8` projection is
computed before the cell axis, so every cell begins with the same eight-number
summary. The next SmolTalk-only A/B adds a 32-feature gate per cell before that
projection. It adds 1,984 parameters and is exactly equivalent at
initialization, allowing the experiment to test whether differentiated cell
views emerge during learning.

The feature-gated input restored qualitatively plausible, context-sensitive but
wrong answers, demonstrating that differentiated cell views carry useful
semantic information. Formations remained capped around four or five displayed
words depending on spaces and punctuation. The next inspection found the
symmetric output problem: every cell was globally pooled into one width-8
vector before an 8-to-32 linear projection. The new grouped readout keeps four
independent interleaved cell populations, pools each to width 8, and
concatenates them into width 32. This tests a matched principle: the population
receives multiple language views and must return multiple language channels.

One of the longest repeatedly observed coherent formations was `Yes, there are
several`. Under the compact tokenizer this is plausibly eight pieces (`Yes`,
comma, space, `there`, space, `are`, space, `several`), matching the pooled
hidden width. Hidden width does not impose a mathematical eight-token output
limit because every slot retains its own state, so this is not proof of a hard
cap. It is nevertheless a useful qualitative signature to test against the
new width-32 grouped population readout.

### Progressive formation-span acquisition

Across the language experiments, Gator did not immediately attempt to fill the
entire 32-slot response window with coherent text. Early checkpoints generally
formed one word. Later checkpoints added words gradually, with plausible
phrases emerging only after shorter formations had stabilized. The observed
behavior resembles progressive recruitment of output slots:

```text
one stable word
    -> short phrase
    -> longer locally coherent formation
    -> refinement of additional slots
```

This is noteworthy because every output slot is computed on every forward
pass; the growth is learned coordination, not autoregressive decoding or an
external length curriculum. It may reflect the network first finding a robust
low-dimensional formation and then extending agreement across more addressed
slots.

This remains an observation, not yet a demonstrated Gator-specific law.
Predicted EOS can shorten displayed text, spaces and punctuation consume
tokenizer positions, and isolated samples are noisy. Future logs must therefore
measure formation span directly at every evaluation:

- Mean and median predicted pieces before the first EOS
- 10th, 50th, and 90th percentile predicted lengths
- Fraction reaching at least 1, 2, 4, 8, 16, and 32 pieces
- Mean displayed words and maximum coherent displayed words
- Predicted length versus target length
- Accuracy by output position

Plotting these against steps will show whether language formation expands as a
smooth frontier, in discrete transitions, or merely follows changing EOS
calibration.

### Grouped-readout complete 50k curve

The first 124,616-parameter feature-gated/grouped-output run used only filtered
SmolTalk, context 64, 32 simultaneous response slots, and no LMSYS examples.
The preserved complete log reports:

| Step | Accuracy | Loss | PPL | Representative formation |
|---:|---:|---:|---:|---|
| 1,000 | 30.8% | 3.9898 | 54.04 | `Yes` |
| 7,000 | 32.3% | 3.8150 | 45.37 | `No,` |
| 8,000 | 32.3% | 3.7998 | 44.69 | `You're welcome! I hope` |
| 13,000 | 32.7% | **3.7461** | **42.36** | `Yes, there are` |
| 19,000 | 32.5% | 3.7601 | 42.95 | `It's recommended` |
| 24,000 | 32.3% | 3.7953 | 44.49 | noisy short formation |
| 32,000 | 31.9% | 3.8746 | 48.16 | `The Col l` |
| 42,000 | 30.7% | 4.0490 | 57.34 | `Start simple way to` |
| 48,000 | 30.2% | 4.1228 | 61.73 | `Currently, sustainable fashion` |
| 50,000 | 30.2% | 4.1408 | 62.85 | noisy short formation |

Accuracy improved from 30.8% to a best logged 32.7%, while loss fell from
3.9898 to 3.7461 and PPL from 54.04 to 42.36 before wobbling. Exact-window
accuracy remained 0%, which is unsurprising for a simultaneous 32-slot target
at this stage.

The best quantitative region was approximately steps 11k–14k. Validation then
regressed almost continuously despite the decaying learning rate, reaching
loss 4.1408 and PPL 62.85 at 50k. This is consistent with overfitting the small
5,726-response training set, optimization drift, or both. Best-checkpoint
retention and early stopping are mandatory for topology comparison.

The complete run took 687.26 seconds at 72.75 steps/second on the NVIDIA RTX
PRO 6000 Blackwell Server Edition.

Several random samples matched conversational intent: gratitude produced
`You're welcome! I hope`, a rest-day question produced `It's recommended`, and
a comparison of acceleration with velocity began `No,`. Other cases had wrong
polarity or generic continuations. This is consistent with contextual
conditioning but does not prove it; shuffled-context and blank-context
evaluation on the same checkpoint remain mandatory.

### Symmetric versus asymmetric topology

The matched asymmetric run changed only the node topology. It reached its best
logged loss/PPL at step 18k (3.7607/42.98), compared with the symmetric run's
better and earlier minimum at step 13k (3.7461/42.36). The best logged
accuracy was 32.5% for asymmetric versus 32.7% for symmetric. At step 30k the
curves were effectively tied (PPL 47.39 asymmetric and 47.35 symmetric), so
the apparent mid-run asymmetric destabilization was not a material quantitative
separation at that checkpoint.

Both runs regressed after their best region. By step 50k, asymmetric retained
slightly better aggregate validation metrics (30.7%, PPL 59.40) than symmetric
(30.2%, PPL 62.85), although its late samples also showed repetitive-letter
degeneration. The clean conclusion is therefore narrower than “asymmetric is
less stable”: asymmetric provided no peak-quality or learning-speed advantage,
while neither topology prevented late validation regression. Symmetric remains
the leading topology by best checkpoint; hybrids should be judged on peak,
time-to-peak, late retention, and qualitative formation separately.

The hybrid-alternating run then reached loss 3.7463 and PPL 42.36 at step
12k, statistically indistinguishable at log precision from symmetric's 3.7461
and 42.36 at step 13k. Its retention was worse through the middle: at step 31k
it had reached PPL 50.43, while symmetric and asymmetric were still below 49
near that region. It ended at 30.1% accuracy and PPL 62.00, between the two
other final PPL values but below both final accuracies. The topology changed
the path and retention of learning more than the best reachable basin. Logged
formations remained context-sensitive even during regression, suggesting that
decoded behavior persisted while token probabilities and fine-grained slot
coordination deteriorated.

The completed hemispheric run provided the clearest stability result. Its best
loss/PPL was 3.7639/43.12 at step 13k, slightly above the tightly clustered
best minima of the other topologies. It then remained near PPL 43–44 through
step 25k and crossed `best + 2 PPL` at step 26k. At matched checkpoints it was
best by step 25k (44.48), step 31k (47.52), and step 50k (56.89). The other
final PPL values were 59.40 asymmetric, 62.00 alternating, and 62.85
symmetric. Thus hemispheric partitioning sacrifices approximately 0.76 PPL at
the single best logged point but materially improves retention under prolonged
training. Topology is best described here as a stability and convergence lever.

### Neighbor breadth and relay depth

The eight-neighbor hemispheric screen retained 124,616 parameters and four
relays. It reached the same PPL 43.12 floor at step 16k and ended at 44.18 at
25k. Increasing connectivity therefore did not improve the best solution. A
scheduler caveat applies: the cosine schedule was tied to the 25k run length,
so it decayed twice as quickly as in the earlier 50k four-neighbor baseline.
Peak-floor replication remains informative, but convergence and retention are
not a fully controlled neighbor-only comparison.

Returning to four neighbors and increasing recurrent depth from four to six
relays produced the first lower hemispheric basin. The parameter count remained
124,616. PPL reached 42.50 at step 15k, stayed near 42.5–43.1 through step 21k,
and finished at 43.42. This nearly matched the global symmetric minimum of
42.36 while retaining the hemispheric mask's stability. Runtime was 476.36
seconds for 25k steps, or 52.48 steps/second. Compared with the four-relay
hemispheric throughput of 73.17 steps/second, 50% more recurrent work reduced
throughput by approximately 28%, indicating sublinear wall-time growth on the
Blackwell GPU for this configuration.

This relay result is promising but preliminary. It does not yet establish that
additional relays are the preferred scaling direction. Validation requires a
matched four-relay/25k scheduler control, additional relay counts, repeated
seeds, and reporting quality against wall time and total relay computation.
The six-relay run improved PPL, but its 52.48 steps/second throughput must be
included in any claim of architectural efficiency.

### One-million-parameter population scaling

Scaling from 62 to 1,038 width-8 cells produced 999,112 parameters while
holding vocabulary, language width, topology, neighbors, relays, data, batch,
and 25k cosine schedule fixed. The model reached a new numerical minimum of
PPL 42.16 at step 16k and ended at 43.02. This is only 0.20 PPL below the prior
global minimum of 42.36 and did not materially expand visible formation length.
Late examples remained short, such as `Yes, there are many`, `You can start
to`, and `You're welcome! I hope you`, followed by many whitespace slots.

Runtime was 3,545.85 seconds at 7.05 steps/second. Relative to the 62-cell,
four-relay hemispheric throughput of 73.17 steps/second, training was about
10.4 times slower. The result is therefore a null practical scaling result:
roughly eight times total parameters and 16.7 times as many cells remained in
the same qualitative/PPL basin for a very large compute cost. Return to the
124,616-parameter model for rapid diagnosis of training, formation span, and
the fixed four-group static-vote readout before scaling population again.

### Whitespace-token confound

The closing quote in logged `formed=...` strings appeared far to the right
because Python printed the complete string representation, including many
trailing spaces. Gator was not leaving those slots unused: it was repeatedly
predicting learned whitespace pieces after the visible words.

The compact tokenizer currently separates text using:

```python
r"\s+|[\w]+|[^\w\s]+"
```

Consequently, ordinary spaces are independent, extremely frequent target
classes. A model can raise all-token accuracy substantially by learning the
alternating word/space structure without predicting the content words. The
reported best 32.7% accuracy is therefore not interpretable as 32.7% semantic
token accuracy until target-class contributions are measured. The reduction in
held-out loss/PPL and contextually plausible content remain useful signals, but
they are also affected by the easy whitespace class.

This observation refines progressive formation-span acquisition. Early Gator
may fill most uncertain slots with whitespace and then replace some of those
positions with content as training progresses. It is not necessarily expanding
from true empty or EOS-terminated output.

Required diagnostics for the preserved checkpoint are now:

- Accuracy on all scored pieces
- Accuracy excluding whitespace-only target pieces
- Accuracy on words, punctuation, EOS, and whitespace separately
- Predicted whitespace fraction by output position and training step
- Predicted EOS position and fraction of examples with no EOS
- Non-whitespace predicted-piece count before EOS
- Most frequent predicted IDs and their decoded forms
- Under tokenizer version 2: predicted BLANK rate by position, BLANK-versus-
  whitespace confusion, and non-BLANK emissions after aligned EOS

The completed topology sweep intentionally kept this tokenizer unchanged. That
preserves a clean relative A/B because symmetric, asymmetric, and both hybrids
share the same whitespace shortcut, data, seed, and training order. Raw
accuracy remains unsuitable as semantic accuracy, but relative topology curves
can still be compared. Its former plan to move directly to a leading-whitespace
tokenizer has been superseded by the explicit PAD/EOS/whitespace audit and
versioned BLANK/alignment program.

The curve also supports progressive formation-span acquisition: outputs began
as one word and later formed short multi-word phrases. It has not yet shown
full use of the width-32 grouped output. The best quantitative region in this
partial log was approximately steps 11k–14k; later results oscillated rather
than improving monotonically.

Raw logs are retained as:

- `experimental_checkpoints/gtr-arw-130k-assym-100ksteps.txt`
- `experimental_checkpoints/gtr-arw-130k-sym-42ksteps.txt`

## Hardware observation and comparison standard

The completed 100k private-hash log reports 18,582.35 seconds and 5.38
steps/second. During subsequent operation, a 50k language screen was estimated
at roughly four hours on a T4 and roughly forty minutes on an RTX Pro 6000.
That approximately sixfold operator-observed difference is encouraging but is
not yet a controlled hardware benchmark: compilation, precision settings,
software version, and exact model variant must be held constant before claiming
a hardware scaling law.

Architecture comparisons must be quality-matched. Steps/second alone is not a
useful victory condition. For Gator and Transformer controls, report the time,
training tokens, memory, and eventually energy required to reach the same
held-out loss, perplexity, token accuracy, exact-window accuracy, and
qualitative conditioning threshold. If a model never reaches a threshold within
the declared budget, record time-to-threshold as not reached rather than
crediting its raw throughput.

## Expansion sequence

### Current objective sequence

1. Pass the PAD, EOS, whitespace, rendering, and response-vocabulary audit.
2. Add tokenizer-versioned BLANK and run the rigid all-slot BLANK-tail control.
3. Measure best-monotonic-aligned accuracy and loss without optimizing them.
4. A/B rigid, detached-Viterbi, and forward-sum monotonic training objectives.
5. Add the low-weight whole-sequence composition/length/EOS term only after
   alignment wins.
6. Run the same objective code on Gator-S and the matched one-shot Transformer,
   locking model, seed, corpus order, stewardship schedule, and optimizer.

### Historical pre-alignment plan

The following list records the earlier topology/data expansion plan. Several
items have been completed or superseded; it is retained for provenance rather
than presented as the current queue.

1. Preserve the current symmetric run as the topology reference under tokenizer
   version 1.
2. Sweep only node topology in this order: asymmetric, hybrid alternating,
   hybrid hemispheric.
3. Compare normal, shuffled, and blank contexts for every topology's best and
   final checkpoints.
4. After the four-mask sweep, attach whitespace to following lexical pieces and
   rerun the winning mask plus symmetric under tokenizer version 2.
5. Replicate the winning corrected topology across seeds.
6. Add an LMSYS short-context filter that admits only examples fitting the
   context window without truncation; record the retained fraction.
7. Replace blind tail cropping with turn-aware packing that always preserves
   role markers and the most recent user turn.
8. Compare four and eight shared relay hops without changing the local cell.
9. Increase context length separately only after the no-truncation control,
   because receiver-specific memory makes longer contexts more costly.
10. Increase response formation to 64 tokens.
11. Add a causal Gator diagnostic baseline using the same split and tokenizer.
12. Compare model-generated samples and held-out perplexity at matched parameter
   and training budgets.

The previous topology screens are considered tainted for language comparison.
They combined different masks with long-row tail truncation, heterogeneous LMSYS
noise, identical shared width-8 inputs, and a single width-8 population readout.
They remain historical engineering results but cannot answer which node mask is
best for the corrected language architecture.

## Monotonic-alignment continuation update (2026-07-17)

The first Gator-S run with true BLANK and forward-sum monotonic alignment began
showing coordinated held-out improvement after roughly 12k steps. By step 16k,
aligned loss was 2.8619, required-token aligned loss 3.7459, required accuracy
30.0%, aligned content accuracy 8.5%, content F1 14.5%, and edit accuracy
31.0%. Because required-token likelihood and semantic/edit metrics improved
with the marginal alignment loss, this turn cannot be attributed to easier
BLANK placement alone. Outputs were still repetitive and under-length, EOS was
poorly learned, and exact response accuracy remained zero.

The next capability probe continues the best checkpoint on a broader
training-only mixture while preserving the existing validation benchmark.
Tulu3 SFT and SmolTalk Preferences use the same flat transcript schema and
passed a 20,000-row parser audit without errors. The 300k-step overnight phase
uses 10% SmolTalk Everyday and a 90% streamed share split
LMSYS:Tulu3:SmolTalk-Preferences at 4:3:2. It preserves Adam moments, restarts
the cosine schedule at LR 0.001, and retains the same model, tokenizer,
formation objective, maximum alignment shift, and response geometry.

## Follow-up: vectorize per-cell computation (CORRECTION: already done)

**Correction (2026-07-26, later the same day):** the section below was
written after reading `gator_cells.py`/`gator_cells_vectorized.py`, which
*do* have a sequential Python loop over cells — but those are disconnected
legacy prototype scripts, not what `--architecture gator` actually runs.
The real production implementation, `gator/models/language.py`
(`GatorLanguageModel`), has **no Python loop over cells at all**: every
cell's private weights are stored as batched tensors with a leading cell
dimension (e.g. `self.node_weight = nn.Parameter(torch.empty(c, n, d, d))`),
and the forward pass routes through `gator/models/dense_kernels.py`'s
`cell_linear`/`cell_node_linear`, which reshape the cell axis into batched
`bmm`/`einsum` calls (a `grouped_bmm` backend is available specifically to
give CUDA/`torch.compile` regular GEMMs instead of many implicit cell-local
contractions), plus `fold_cell_node_parameters`, which algebraically folds
the per-node affine transforms into fewer, wider GEMMs. This vectorization
work was already done — the original text below is preserved for the
record but its premise was wrong.

---

The cell architecture (`--architecture gator`) is the preferred design (it
resists the degenerate-constant-output collapse that the single-shared-graph
`convergent` architecture repeatedly hits under joint/parallel-formation
training — see `dual_expert_coordinator/OVERVIEW.md`), but it has never been
optimized for scale. Each of the `--cells` cells currently runs its own
independent forward pass as a separate sequential Python-level loop iteration
(`for c, cell in enumerate(self.cells): ...` — see the per-cell `_local_relay`/
`_memory` methods), even though every cell is structurally identical and
differs only in its private per-cell weights (node transforms, router,
relative-node bias, etc.) plus the coupling gates. This is both a real memory
cost and a real speed penalty (Python-loop + per-cell kernel-launch overhead
multiplied by `--cells`), and is the main reason this architecture was set
aside in favor of `convergent`'s single shared graph.

Shared infrastructure (the communal token embedding) is already factored out.
The remaining, unexplored optimization is to vectorize the *private* per-cell
math itself: stack all cells' private weight tensors along a new leading
"cell" dimension and replace the sequential loop with a single batched
einsum/bmm, the same way `gator_cells_vectorized.py` ("Vectorized 10-cell
Gator ring with checkpoint compatibility") already attempts for the smaller
10-cell math-only ring. This should preserve the property that seems to matter
for collapse-resistance — each cell still computes a genuinely independent
forward pass and independent logits, combined only at the final vote — while
removing the Python-loop and per-cell kernel-launch cost that currently makes
scaling `--cells` expensive. This has not been attempted for the language-
capable (`communal`/`feature_gated`/`grouped_concat`) variant and is the
natural next step before trying to scale the cell architecture further.

## Synthesis: why `--architecture gator` resists the collapse that `convergent` does not

A same-day, direct A/B (2026-07-26) makes this an evidence-backed claim rather
than a guess. Six configurations of `--architecture convergent` (the current
primary single-shared-graph implementation) were tested earlier the same day
under `chunk_parallel`/`parallel` decode with a joint math+language corpus:
varying corpus weighting, curriculum ordering, topology choice, `slot_ce` vs.
`monotonic`/`viterbi` formation objectives, a whitespace-omission penalty, and
a sequence-composition penalty. **Every single one collapsed to a degenerate
constant output** (a single repeated character or token, e.g. `'aaaaaaa'`,
`'eeeeeeee'`, or pure whitespace) regardless of which objective or penalty was
used — penalizing one collapse mode simply relocated the exploit to a
different constant token, never fixed it.

The same day, a from-scratch reproduction of the historical
`LANG-62-COMMUNAL-GATED-GROUPED-CTX64-SMOL-HEMI-S0`-class config
(`--architecture gator`, 124,618 parameters: `--cells 62 --d-model 8
--embedding-dim 32 --lexical-mode communal --cell-input-mode feature_gated
--cell-output-mode grouped_concat --node-topology hybrid_hemispheric
--cell-transport-mode stewardship --stewardship-min-active 4
--stewardship-max-active 62 --stewardship-keep-mass 0.97
--stewardship-handoff-keep-mass 0.97 --stewardship-scouts 1
--response-tokens 40 --reference-tokens 32`) was tested under both
`slot_ce` and `monotonic` formation objectives. **Neither ever collapsed to a
degenerate constant** — formed output was consistently multiple distinct real
vocabulary tokens (e.g. `'The sorry'`, `'To s the simple'`, `'Sure!"'`),
even during a stretch where `monotonic` training destabilized the scalar PPL
metric itself (77.66 → 136.96 over steps 1,000–8,000). That destabilization
episode is itself a useful data point: **raw-metric instability and semantic
collapse are decoupled here** — PPL got worse while the formed text stayed
qualitatively healthy, the opposite of every `convergent` collapse observed
today, where PPL and semantic content degraded together into the same fixed
point.

Because the formation objective was varied independently in both
architectures without changing this outcome, the objective is not the
determining factor. What differs structurally between the two architectures:

1. **Redundant independent computation vs. one shared path.** `gator` runs
   `--cells` fully independent forward passes (each with its own routing,
   node computation, and logits) and combines them only at the final
   softmax-weighted vote. `convergent` routes everything through one shared
   graph with a single (or few) terminal jury per output slot. A degenerate
   shortcut in `gator` has to be independently reached by enough of the 62
   separately-computed cells to dominate the weighted sum; in `convergent`
   there is only one computation to collapse into a shortcut at all.
2. **Stewardship's adaptive active-cell nucleus.** `--cell-transport-mode
   stewardship` with `--stewardship-min-active`/`--stewardship-max-active`
   keeps a shifting, live subset of cells participating each step (not a
   fixed shared computation every cell must funnel through), which
   structurally prevents any single computational path from becoming the
   *only* path the way a single shared graph's route/jury can.

Both point toward the same underlying mechanism: **redundant, independently-
computed diversity feeding into a final combination step resists collapse;
centralizing that computation into one shared graph (done in `convergent` for
good, separate scaling reasons — see the vectorization follow-up above)
removes the redundancy that appears to be doing the collapse-resistance
work.** This reframes the whole plateau-instability investigation: the
single-shared-graph design may have traded away collapse-resistance in
exchange for the scaling/parameter-efficiency gains documented earlier in
this repo, and that trade was never explicitly evaluated at the time it was
made (`convergent` was validated on the modular-prefix math task in
isolation, which does not exercise this failure mode the way joint or
long-form language training does).

This is not yet a proven causal mechanism — only a same-day, same-corpus,
same-hyperparameter-family correlation across every configuration tested so
far. The two live candidate follow-ups are: (1) test `convergent` with an
explicit multi-jury ensemble-style terminal vote (multiple independently
scored juries combined by weighted vote, rather than one jury per slot) to
see whether that alone restores collapse-resistance without reintroducing
per-cell replication cost; (2) test the vectorized `gator` cell computation
(see the follow-up section above) at a scale closer to `convergent`'s typical
parameter budget, to see whether the collapse-resistance property survives
once the per-cell Python-loop cost is removed and cell/node counts can be
raised further.

A direct `slot_ce` re-run of the same 124,618-parameter config (identical
`--cells 62`/`hybrid_hemispheric`/stewardship setup, only the formation
objective changed back from `monotonic`) reached PPL 67.16 by step 2,000 with
no sign of collapse either — consistent with the objective being irrelevant
to this specific architecture's collapse-resistance, matching the `monotonic`
result above.

## Next goal: scaling laws for the cell architecture itself

Given the collapse-resistance evidence above, the cell architecture
(`--architecture gator`) is now the preferred design, but its own scaling
question has never been systematically studied the way `convergent`'s
node/width ratio was (see `Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md` section
9). Every cell-scaling step taken historically was ad hoc: the original
980-parameter four-node/width-8 cell was simply replicated and stitched
together in larger rings (10 cells, then 62, then experiments up to 1,038 —
see the results table above), without ever asking whether a *different*
per-cell shape (more/fewer nodes per cell, a different width, a different
relay count) scales better than "more copies of the same fixed 980-parameter
cell." The `GATOR_PLATEAU_INSTABILITY_STUDY.md` node/width ratio (~1.7) and
embedding-dim-vs-vocab-size sizing rule were both derived for `convergent`'s
single shared graph and have not been checked against the cell architecture's
very different scaling axes: cell count, nodes-per-cell, per-cell width, and
whatever relationship (if any) holds between them at a fixed total parameter
budget. This is the open question for the next phase of cell-architecture
work: is there an analogous "golden ratio" for per-cell shape that scales
better than naively stitching together more instances of the original
950-ish-parameter cell. (The vectorization concern originally noted here
does not block this — see the correction at the top of the follow-up
section above: the production `GatorLanguageModel` forward pass is already
fully vectorized across cells, no Python loop involved.)

## Design note: what else could be shared across cells (not implemented)

`gator/models/language.py`'s `GatorLanguageModel` already shares some
machinery across all `--cells` cells and keeps the rest private per cell.
Listing what's already shared, what's private, and each private block's
actual parameter cost (computed for the validated 62-cell/`n_nodes=4`/
`d_model=8`/`embedding_dim=32` config) to prioritize further sharing:

**Already shared (one copy, not `c` of them):**
- `token_embedding` — the vocabulary table itself (this is what `--lexical-mode
  communal` means; `--lexical-mode private_hash` instead gives every cell a
  private 16-row prototype table, `lexical_prototypes`, selected by
  `token_id % 16` — a deliberately tiny private overlay on top of the still-
  shared full embedding, not a full private vocabulary).
- `input_projection` / `output_projection` — the embedding-width-to-cell-width
  projections (one shared `nn.Linear`, not per-cell).
- `handoff_weight` (stewardship transport mode only) — **the existing
  precedent for this whole design note**: the routing-decision *weight matrix*
  is a single shared `(actions, router_inputs)` tensor with no cell dimension
  at all, while `handoff_bias` (a `(c, actions)` tensor) stays private per
  cell. Same shared function, different per-cell bias/inputs, so the routing
  decision itself still differs per cell.

**Private per cell, with rough parameter cost at the reference config
(`c=62, n=4, d=8, embedding_dim=32`; edges `= n²=16`; router input width
`= 2d+n=20`):**

| tensor | shape (per cell) | total params (×62) | share candidate? |
| --- | --- | ---: | --- |
| `router_weight` | `(edges, router_inputs)` = (16,20) | 19,840 | **yes — highest priority** |
| `node_weight` | `(n, d, d)` = (4,8,8) | 15,872 | no — likely the core of collapse-resistance, see below |
| `address_to_edges` | `(edges, 4)` = (16,4) | 3,968 | yes — positional, not semantic |
| `address_to_state` | `(d, 4)` = (8,4) | 1,984 | yes |
| `address_to_readout` | `(d, 4)` = (8,4) | 1,984 | yes |
| `step_projection` | `(d, 4)` = (8,4) | 1,984 | yes |
| `node_embedding` | `(n, d)` = (4,8) | 1,984 | maybe (small either way) |
| `node_bias` | `(n, d)` = (4,8) | 1,984 | maybe |
| `input_feature_gates` | `(embedding_dim,)` = (32,) | 1,984 | no — this *is* the mechanism that differentiates each cell's view of the input (see the "population bottleneck" discussion above); sharing it would collapse cell views back together |
| `router_bias` | `(edges,)` = (16,) | 992 | no — keep private (mirrors the `handoff_weight`/`handoff_bias` split: shared weight, private bias) |
| `relative_node_bias` | `(n, 4)` = (4,4) | 992 | yes |
| `norm_weight` + `norm_bias` | `(d,)` ×2 = (8,)×2 | 992 | yes |
| `handoff_bias` (stewardship) | `(actions,)` = (5,) | 310 | no — private by design, see above |
| `cell_embedding` (communal lexical mode) | `(d,)` = (8,) | 496 | no — this is the cell's identity/bias in the shared embedding space, analogous to `input_feature_gates` |
| `start_logits` | `(n,)` = (4,) | 248 | yes (tiny either way) |

**Recommendation, in priority order (highest expected savings first, not yet
implemented):**

1. **Share `router_weight`, keep `router_bias` private** — the single
   biggest lever (≈19.8K of the ≈56.5K total private-parameter budget at this
   reference config, more than a third). This exactly mirrors the existing
   `handoff_weight`/`handoff_bias` precedent already in the codebase: the
   routing *function* is shared, but since each cell's local state/context/
   occupancy inputs differ, the routing *decision* still differs per cell —
   computation stays independent, only the learned function is tied.
2. **Share the four small address/step projections** (`address_to_state`,
   `address_to_edges`, `address_to_readout`, `step_projection` — ≈9.9K
   combined). These encode *position in the sequence*, not cell-specific
   semantics — there's no obvious reason every cell needs its own notion of
   "where token 17 is." Low risk, moderate savings.
3. **Share `relative_node_bias`, `norm_weight`/`norm_bias`, `start_logits`**
   (≈2.2K combined) — small individually, but no clear reason to keep private
   either; low-priority cleanup rather than a real savings lever.
4. **Do not share `node_weight`/`node_bias`/`node_embedding`** (≈19.8K
   combined, the actual per-node affine transform) — this is the closest
   thing in the architecture to "what each cell actually computes," and is
   the most likely candidate for *being* the source of the redundant-
   independent-computation property tied to collapse-resistance in the
   synthesis above. If a cheaper version is wanted, prefer a **low-rank
   shared-base-plus-private-adapter** factoring (a shared `node_weight` plus
   a small per-cell low-rank correction, LoRA-style) over full sharing, so
   each cell keeps a genuinely different transform without paying the full
   per-cell parameter cost.
5. **Do not share `input_feature_gates`, `router_bias`, `handoff_bias`,
   `cell_embedding`** — each of these is specifically *the* mechanism that
   differentiates one cell from another in its respective subsystem (input
   view, routing decision, transport decision, embedding identity); sharing
   any of them would directly undo the differentiation it exists to provide.

Combining recommendations 1–3 would cut roughly 32K of parameters (≈57% of
the current per-cell-scaled budget) at this reference config while — per the
reasoning above — leaving untouched the specific tensors hypothesized to
matter for collapse-resistance (`node_weight`, `input_feature_gates`,
`router_bias`, `handoff_bias`, `cell_embedding`). That freed budget could
either shrink the model at fixed capability, or be reinvested into a wider
`node_weight`/more cells at the same total parameter budget — directly
feeding into the cell-count-vs-per-cell-shape sweep below. **This is a
design proposal only; none of it has been implemented or tested yet.**

## First cell-shape/cell-count sweep (2026-07-26, later the same day)

A 3×3 grid (`n_cells` in {12, 16, 20} × `d_model` in {4, 8, 16}, holding
`n_nodes=4`, `embedding_dim=16`, `relays=4`, `neighbors=4`,
`cell_output_mode=grouped_concat`, `cell_transport_mode=stewardship` fixed —
same architecture family as the validated 124,618-param reproduction above,
just smaller and with `embedding_dim` reduced from 32 to 16 to reach a ~50K
budget) was trained for 3,000 steps each (SmolTalk-only, `slot_ce`,
`--mixed-precision bf16`, same optimizer/LR as the earlier validation run).
This grid answers both the per-cell-shape question (task: vary `d_model` at
fixed `n_cells`) and the cell-count question (vary `n_cells` at fixed
`d_model`) simultaneously, since they're the same two sweep axes:

| n_cells | d_model | parameters | PPL @ 3,000 steps |
| ---: | ---: | ---: | ---: |
| 12 | 4 | 40,844 | 74.49 |
| 12 | 8 | 45,896 | 67.23 |
| 12 | 16 | 60,336 | 64.82 |
| 16 | 4 | 42,720 | 69.94 |
| 16 | 8 | 49,420 | 65.01 |
| 16 | 16 | 68,692 | 64.94 |
| 20 | 4 | 44,596 | 70.48 |
| **20** | **8** | **52,944** | **64.29 (best)** |
| 20 | 16 | 77,048 | 65.34 |

**Findings:**

1. **`d_model=8` is a consistent efficiency knee, independent of `n_cells`.**
   At every cell count tested, `d_model=8` matches or beats `d_model=16`
   despite `d_model=16` costing 31–75% more parameters (e.g. at `n_cells=16`:
   68,692 params at `d_model=16` gets PPL 64.94, essentially *identical* to
   49,420 params at `d_model=8` getting PPL 65.01 — 39% more parameters for
   no measurable gain). `d_model=4` is clearly worse than `d_model=8` at every
   cell count (a 5–7 point PPL gap). This is analogous to `convergent`'s
   embedding-dim-vs-vocab-size knee in `GATOR_PLATEAU_INSTABILITY_STUDY.md`
   section 9 — there's a real per-cell-width sweet spot, and it isn't "wider
   is always better."
2. **More cells helps, but not monotonically at every width.** At fixed
   `d_model=8`, going 12→16→20 cells is monotonically better (67.23→65.01→
   64.29). At fixed `d_model=4`, the same cell-count increase improves then
   *flattens/slightly regresses* (74.49→69.94→70.48) — narrow cells seem to
   run out of useful capacity to add via more copies alone. This suggests
   cell count and per-cell width aren't fully independent axes: how much
   benefit more cells buys depends on whether each cell already has enough
   width to do something worth replicating.
3. **Best config in this grid:** `n_cells=20, d_model=8` (52,944 params,
   PPL 64.29) — closest to the reference config's ratio (which uses
   `n_cells=62, d_model=8` at a larger `embedding_dim=32`) rather than either
   extreme (few-wide-cells or many-narrow-cells).

**Caveats:** this is a single 3,000-step read per config (not the full
20,000+ step trajectory validated for the reference config above), on
SmolTalk-only data with `slot_ce`, and only two axes (`n_cells`, `d_model`)
at fixed `n_nodes=4`. It establishes a first-pass "don't go below `d_model=8`
for this task/vocab, and prefer scaling via cell count over cell width past
that point" rule of thumb, not a fully validated golden ratio. A proper
follow-up should: (a) extend the best candidates to a longer step budget to
confirm the ranking holds past the point where the historical curve showed
regression (11k–14k steps and beyond, per the earlier results table), (b)
sweep `n_nodes` as a third axis (held fixed at 4 here), and (c) test whether
the knee point shifts with `embedding_dim`/`vocab_size`, the way `convergent`'s
embedding_dim needed to scale with vocab_size rather than d_model.

## 100M-scale memory ceiling: cell count is the real cost driver, not raw parameter count

Following the small-scale sweep above, an attempt to scale Arrow to a genuine
100M-parameter, `context=512`/`vocab=20000` config (matching a realistic
production target) hit a hard memory wall on a ~95 GiB GPU (G4), and the
investigation revealed something the small-scale sweep alone didn't surface:
**the number of cells is a direct multiplier on every activation tensor's
shape** (`[batch, cells, tokens, ...]`), not just a parameter-count
contributor — so at long context, cell count matters far more to memory
than total parameter count does.

**What OOM'd:** `n_cells=400, d_model=256` (114.8M params, `shared_pool`
output, `communal`/`feature_gated`) OOM'd at `context=512`/`vocab=20000`
regardless of batch size (tried 8, 4, and 1 — the attempted allocation size
was *identical* across all three, ruling out batch size as the driver),
`--mixed-precision bf16`, and `--compile`. Halving the target down to
170 cells (~51M params) and then 124 cells (~24M params) still OOM'd or
barely fit: 124-cell/d_model=192 (23.7M params) was the first config that
actually completed, at **~64 GiB allocated / ~94 GiB reserved — essentially
the entire card — and only 0.11 steps/sec.** Parameter count alone is a
poor predictor of whether an Arrow config fits in memory.

**`--decode-mode chunk_parallel` is not wired up for `--architecture gator`
at all** — the trainer explicitly rejects it ("currently requires
--architecture convergent"). This is a separate, real gap from the memory
ceiling, not yet addressed.

**Why packed execution didn't save it:** `stewardship_execution=packed_auto`
is designed to skip evaluating dead routes once a live/dead split has been
*observed* over several steps (`StewardshipExecutionController`, a one-way
dense-to-packed latch) — it assumes the initial dense pass is affordable and
only becomes a bottleneck as training naturally sparsifies. It cannot help a
config that OOMs on its very first dense forward pass, before any
observation window completes. This is a real design constraint worth
remembering: packed execution is a mid-training optimization, not a
step-zero one.

**Fix 1 — activation checkpointing (partial win, implemented):** added an
opt-in `--activation-checkpointing` flag (`GatorLanguageConfig
.activation_checkpointing`, default `False`) that wraps the single most
memory-hungry per-relay segment (`GatorLanguageModel._dense_relay_transform`
— the router/node-affine computation that materializes
`[batch, cells, tokens, nodes, nodes]`-shaped intermediates) in
`torch.utils.checkpoint`, recomputing it during backward instead of keeping
it resident. This segment has no side effects (the diagnostic trace-list
appends all live outside it, and are only active when `diagnostic_mode !=
"none"`, i.e. periodic eval — never during the checkpointed segment itself),
so it's safe to checkpoint unconditionally when the flag is set. Testing it
on the 170-cell/51M config that OOM'd: **the OOM point measurably moved**
further into the forward pass (into `_dense_stewardship_context`'s
receiver-memory computation, a segment not checkpointed), confirming
checkpointing genuinely reduced memory pressure where applied — but a
second, independently-expensive computation remained, so this alone did not
fix that specific config. Chasing every individual expensive segment with
its own checkpoint has diminishing returns; a real fix would need to
checkpoint (or otherwise reduce) the receiver-memory computation too.

**Fix 2 — fewer, bigger cells (validated, and stronger than first thought):**
re-shaping the same ~100M-parameter budget from "many small cells" toward
"few large cells" was tested at four points, and the result is a clean
**monotonic win on both memory and throughput simultaneously**, all the way
down to the topology's practical floor:

| `n_cells` | `d_model` | params | steps/sec | peak reserved |
| ---: | ---: | ---: | ---: | ---: |
| 400 | 256 | 114.8M | OOM | >95 GiB |
| 40 | 768 | 100.4M | 0.07 | 92.4 GiB |
| 10 | 1536 | 99.9M | 0.13 | 56.8 GiB |
| 4 | 2432 | 100.2M | 0.14 | 34.2 GiB |

(All at `context=512`, `vocab=20000`, `batch=4`, `--mixed-precision bf16`,
`--compile`, `stewardship_execution=packed_auto`, `neighbors` set to the
minimum valid value for each cell count.) An earlier draft of this section,
written after only the first two rows, concluded fewer/bigger cells "trades
memory for throughput" — **that was wrong**, based on comparing configs at
different parameter counts. The controlled four-point comparison above,
all near the same ~100M budget, shows both memory *and* speed improving
together as cell count drops, all the way to 4 (`hybrid_hemispheric`'s
practical minimum, since it requires an even cell count). This makes sense
once you separate the two cost mechanisms cleanly: cell count multiplies
every activation tensor's `[batch, cells, tokens, ...]` shape directly
(the memory-dominant term at this context length), while `d_model` mostly
costs *parameters* (`node_weight` scales as `d²`) and FLOPs, not activation
memory — and at these current cell counts, the FLOPs cost of a wider
`d_model` is evidently still cheaper than the memory-bandwidth cost of
materializing many more cells' worth of activations at `context=512`. This
directly validates the user's standing hypothesis from earlier tonight: if
the cell *ensemble/voting* structure is what resists collapse, a genuinely
small, "tensor-friendly" cell count (as low as ~4, not ~40) may be both the
most practical *and* the fastest production shape — not a compromise at all.

**This has not been pushed further down** (e.g. `n_cells=2`, or non-hybrid
topologies without the even-cell-count constraint) or tested at longer
context/larger batch to see where the trend eventually reverses (it must,
at the extreme — a single cell has no ensemble left, and arbitrarily large
`d_model` costs arbitrarily many parameters and FLOPs) — that boundary is
the natural next question.

**Open follow-ups:**
- Find where the "fewer cells" trend actually reverses (test below 4,
  and at larger context/batch where the balance may shift).
- Wire up `chunk_parallel` for `--architecture gator` (currently rejected
  outright).
- Extend activation checkpointing to `_dense_stewardship_context`'s
  receiver-memory computation, the second bottleneck identified above (may
  matter less now that the fewer-cells fix reduces memory pressure overall).
- A genuine cell-axis chunking scheme (process cells in groups, only
  materializing one group's activations at a time) was considered but not
  attempted: cells exchange information with their ring neighbors each
  relay via the handoff mechanism (`ring_handoff_destinations`, a local/
  banded interaction, not all-to-all), so naive disjoint chunking would
  break correctness at chunk boundaries unless a halo-overlap or similar
  boundary-handling scheme is added. This is a real, larger follow-up, not
  attempted tonight given the correctness risk.
- Systematically sweep the cell-count-vs-width tradeoff at the 100M scale
  specifically (only two points were tested: 400×256 and 40×768) to find
  where the memory-fits/throughput-acceptable region actually is, rather
  than jumping straight from one extreme to the other.
