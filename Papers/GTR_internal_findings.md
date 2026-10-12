# Graph Trace Relay: Preliminary Findings on Simultaneous Sequence Formation

**Internal research note — updated August 10, 2026**

## Abstract

We evaluated Graph Trace Relay (GTR), a recurrent dynamically routed graph network, as a 980-parameter mechanism for simultaneous sequence formation. Unlike autoregressive language models, the fixed-slot GTR variants receive a small conditioning sequence, refine all output slots together through recurrent graph relays, and emit the complete output once.

On a deterministic modular-sequence task, fixed-slot GTR improved from 24.6% position accuracy after 2,000 steps to 52.6% after 10,000 steps, demonstrating that a complete fourteen-token structure can be partially assembled from two condition tokens without autoregressive feedback. On a harder random parallel-prefix task, a relation-aware GTR variant reached approximately 21–22% held-out accuracy at length 7, while equally trained one-shot Transformers from 980 through 39,758 parameters remained near the 8.3% chance floor. A 20,726-parameter Transformer could perfectly memorize 100 fixed examples but failed to generalize beyond them.

These are single-seed exploratory results, not yet evidence of a general scaling law. They nevertheless identify receiver-specific graph reads as the first GTR component to produce a measurable generalization advantage on simultaneous prefix formation.

## 1. Architecture

Each token or output slot carries a mutable hidden state and a distribution over graph nodes. At every relay hop, all slots:

1. Write into graph-node memories.
2. Read graph context.
3. Recompute source-to-destination transitions.
4. Move their node-location distributions.
5. Apply destination-node transformations.

All sequence positions update simultaneously. Relay depth remains recurrent, but the model does not emit or feed back intermediate output tokens.

The principal model variants were:

- **GTR-0:** fixed output slots with shared graph-node memory.
- **GTR-A:** immutable continuous slot addresses reinjected into routing, node computation, and readout.
- **GTR-AW:** address anchoring plus controlled writing, where condition tokens seed memory before empty, MASK-initialized output queries contribute fully.
- **GTR-AR:** address anchoring plus receiver-specific relative node reads.
- **GTR-ARW:** address anchoring, receiver-specific relative reads, and controlled writing.

Every principal GTR variant contains exactly 980 trainable parameters. Learned positional tables were replaced by fixed four-dimensional coordinate codes and three learned projections without increasing the parameter budget.

### 1.1 Gator-S token stewardship

The later Gator-S transport keeps Arrow as the local cell and replaces static
population mixing with explicit per-token ownership. Each live steward applies
its Arrow computation, chooses `stay` or a legal ring neighbor, and transports
ownership mass, branch state, and the persistent node passport together.
Uncertain stewards may retain several local exits; confident routes narrow while
a low-mass scout keeps alternative paths recoverable. Final ownership determines
which cells contribute to each output slot.

This mechanism is implemented in dense and packed-equivalent forms at 124,618
parameters for the current 62-cell reference, two parameters above its
`broadcast_mix` control. Conservation, gradient, padding-isolation, and
dense/packed equivalence tests are implementation evidence only. No held-out
language-quality or GPU-efficiency conclusion has yet been established for
Gator-S.

## 2. Fixed-slot modular formation

The model observes only a rule token and a starting data token. It must emit the remaining fourteen-token modular sequence simultaneously.

### 2.1 Training-duration result

| Training steps | Correct positions | Position accuracy | Exact sequences | Loss |
|---:|---:|---:|---:|---:|
| 2,000 | 3.44/14 | 24.6% | 0% | 1.7631 |
| 10,000 | 7.37/14 | 52.6% | 2.3% | 0.8685 |

The model received no teacher-forced intermediate outputs. It therefore had to infer the modular rule and assign all fourteen output positions in one latent formation process. The improvement was material and had not fully plateaued by 10,000 steps.

Qualitative errors often contained the correct token cycle with local phase slips. This motivated explicit treatment of slot addressability and receiver-specific reads.

## 3. Parallel-prefix formation

The harder benchmark uses fresh random modular increments:

\[
y_t = x_0 + \sum_{j=1}^{t} d_j \pmod {12}.
\]

The input is `[start, d1, ..., dL]`; all prefix targets `[y1, ..., yL]` are emitted simultaneously. Because fresh random increments are used for every example, incomplete aggregation provides essentially no predictive signal. Chance token accuracy is approximately 8.3%.

### 3.1 Matched 10,000-step comparison at length 7

| Model | Parameters | Layers/relays | Correct | Accuracy | Loss |
|---|---:|---:|---:|---:|---:|
| One-shot Transformer | 980 | 1 layer | 0.59/7 | 8.5% | 2.4852 |
| GTR-AW | 980 | 4 relays | 0.59/7 | 8.4% | 2.4854 |
| **GTR-ARW** | **980** | **4 relays** | **1.50/7** | **21.4%** | **2.1385** |

GTR-ARW showed an abrupt acquisition transition:

| Step | Accuracy |
|---:|---:|
| 1,000 | 8.6% |
| 2,000 | 9.0% |
| 3,000 | 8.5% |
| 4,000 | 10.8% |
| 5,000 | 20.8% |
| 7,000 | 22.0% |
| 10,000 | 21.9% |

This curve is compatible with an S-shaped acquisition event, but a single seed cannot distinguish a repeatable phase transition from optimization instability.

GTR-AW remained at chance. The effective intervention was therefore not address anchoring or controlled writing alone, but receiver-specific relative graph reads. Each output position needed a distinct view of the shared workspace to aggregate its corresponding prefix.

### 3.2 Curriculum from length 7 to 14

GTR-ARW was trained for 55,000 steps. Prefix length increased from 7 to 14 in equal stages over the first 80% of training and remained at 14 for the final 20%.

Performance followed the moving training length:

| Current length | Approximate accuracy after adaptation |
|---:|---:|
| 7 | 21.0% |
| 8 | 19.5% |
| 9 | 18.6% |
| 10 | 17.2% |
| 11 | 16.5% |
| 12 | 16.1% |
| 13 | 15.4% |
| 14 | 15.2% |

Final held-out length-14 accuracy was 14.9%, or 2.08 correct positions out of 14. Final evaluation at lengths 7–12 returned to approximately chance, showing catastrophic forgetting and length-specific adaptation rather than a length-general parallel-scan algorithm.

## 4. Transformer parameter scaling

Four-layer one-shot Transformers were trained for 5,000 steps on fresh length-7 prefix examples. The screening criterion was greater than 10% held-out accuracy.

| Parameters | Accuracy | Outcome |
|---:|---:|---|
| 806 | 8.21% | Chance |
| 2,204 | 8.32% | Chance |
| 3,698 | 8.47% | Chance |
| 10,484 | 8.3% | Chance |
| 20,726 | 8.5% | Chance |
| 39,758 | 8.3% | Chance |

An approximately 79,826-parameter run was started but intentionally stopped to prioritize a memorization diagnostic. No result should be attributed to that candidate.

At the 5,000-step checkpoint, the 980-parameter GTR-ARW achieved 20.8%, while tested four-layer Transformers up to 39,758 parameters remained at chance. This suggests a substantial parameter-efficiency difference on this benchmark, subject to replication.

## 5. Transformer memorization control

To verify that the Transformer implementation and optimizer were functional, the 20,726-parameter, four-layer Transformer was repeatedly trained on the same 100 examples for 1,000 steps.

| Step | Training token accuracy | Exact training sequences |
|---:|---:|---:|
| 100 | 11.6% | 0% |
| 200 | 72.4% | 12% |
| 300 | 96.9% | 80% |
| 400 | 99.1% | 95% |
| 600 | 100% | 100% |

After perfect memorization, held-out accuracy was 8.6% with a loss of 13.7866. The Transformer had ample capacity to store the finite dataset but did not discover a procedure that generalized to fresh prefix examples.

This control argues against a broken implementation or an inability to optimize the network at all. The observed failure concerns algorithmic generalization.

## 6. Interpretation

The strongest current result is narrow but meaningful:

> At equal training budget, a 980-parameter receiver-relative GTR learned above-chance simultaneous modular-prefix formation, while tested one-shot Transformers with up to approximately forty times as many parameters remained at chance.

The result supports three provisional hypotheses:

1. **Receiver-specific graph reads are essential.** Shared node memory contains formation-level information but does not tell each output slot which portion it needs.
2. **Learning may be thresholded.** GTR-ARW remained near chance for several thousand steps before a sharp increase around steps 4,000–5,000.
3. **The learned procedure is not length-general.** Curriculum training moved competence to longer lengths but erased shorter-length behavior.

The data do not yet demonstrate exponential GTR learning, logarithmic Transformer learning, or superior general performance. They demonstrate a task-specific inductive-bias advantage worth investigating.

## 7. Experimental caveat and correction

An early variant-parser bug checked whether the character `r` appeared anywhere in model names beginning with `gtr`. This unintentionally enabled relation-aware reads in variants initially labeled non-relative. The exact duplicate AW/ARW curves exposed the problem.

The parser was corrected to use explicit variant membership. Earlier architecture-sanity and relay-sweep CSVs must not be used to compare relative with non-relative variants. The following results are valid after correction:

- The 10,000-step GTR-AW prefix result.
- The 10,000-step GTR-ARW prefix result.
- The 55,000-step GTR-ARW curriculum.
- Transformer scaling and memorization results.

## 8. Early next-experiment list (historical)

1. Repeat the GTR-ARW acquisition curve across paired seeds.
2. Save dense checkpoints around steps 3,000–6,000 to characterize the apparent S-curve.
3. Resume Transformer screening at approximately 80k parameters, then sweep two- and one-layer models below 100k.
4. Mix prior prefix lengths during curriculum training to separate catastrophic forgetting from failed extrapolation.
5. Probe whether relative-read parameters learn powers-of-two displacement patterns expected from parallel scan.
6. Vectorize node transformations, profile receiver-memory construction, and tune CPU thread allocation before large multi-seed runs.
7. Compare parameter count, wall-clock time, FLOPs, and sample efficiency separately; equal parameter count does not imply equal compute.

## 9. Subsequent stitched-cell result

A later ten-cell, degree-four, vectorized Gator experiment substantially exceeded the early results summarized above. All models used 10,000 parameters, four local nodes per cell, four relays, and four neighboring cells.

| Seed | Final token accuracy | Exact sequences | Final loss | Interpretation |
|---:|---:|---:|---:|---|
| 0 | 99.8% | 98.8% | 0.0096 | Near-complete acquisition |
| 1 | 91.3% | 59.6% | 0.2515 | Early but oscillatory acquisition |
| 2 | 41.0% | 0.0% | 1.5217 | Slow continuing consolidation |

The topology learned above chance in all three seeds, but acquisition timing and convergence rate varied dramatically. Seed 2 continued improving slowly through 20,000 steps and should not yet be interpreted as a fixed plateau. These results motivate a formal optimizer API, resume support, best-checkpoint retention, and fixed-set diagnostics.

## 10. Larger Transformer screen

A four-layer Transformer with 1,006,154 parameters was trained for 20,000
steps on fresh length-7 prefix examples. It converged to the uniform predictor:

| Metric | Result |
|---|---:|
| Final token accuracy | 8.3% |
| Best logged accuracy | 8.8% |
| Exact sequences | 0% |
| Final loss | 2.4856 |
| Runtime | 236.0 seconds |

The chance loss is `ln(12) = 2.4849`; gradient norm declined from 0.8702 at
step 1,000 to values near 0.03 late in training. This was stable convergence to
an uninformative solution, not numerical divergence. Observed 4M- and 16M-scale
curiosity screens also failed the 20%-by-10k candidate criterion, but complete
final logs were not retained and precise values are not reported.

## 11. First simultaneous-language pilot and correction

The first language model used 130,000 parameters, ten width-32 cells, a
2,048-token compact vocabulary, 64 context tokens, and 32 simultaneous response
slots. On unfiltered SmolTalk it reached approximately 43% token accuracy,
26.6% exact windows, and perplexity near 24.

Those headline metrics were confounded. The exact response
`Hello! How can I help you today?` occurred 2,022 times in training and 226
times in validation; 226/864 = 26.16%, almost exactly the reported exact score.
The model had learned a coordinated default formation, not established
contextual language understanding. The corrected protocol excludes every
SmolTalk conversation's first assistant response, prints random validation
turns, evaluates the full set, and requires shuffled- and blank-context
ablations before claiming conditioning.

The training corpus now mixes 90% LMSYS Chat and 10% filtered SmolTalk. A full
scan found 983,244 valid LMSYS conversations, 1,973,235 assistant responses, and
346 malformed rows that are skipped. Fixed windows occupy approximately 0.76 GB
when packed as CPU `int32` tensors.

## 12. Language architecture progression and lexical correction

The validated atomic unit remains the original 980-parameter GTR-ARW cell:
width 8, four nodes, four relays, anchored addresses, controlled writing, and
receiver-specific relative reads. The first 62-cell, 129,304-parameter language
model contained:

- A shared 32-dimensional, 2,048-token language shell
- 62 literal 980-parameter cells
- 248 total graph-node instances
- Degree-four cell coupling
- Private 16-by-8 lexical prototypes in every cell

The private prototypes preserve the original math cell's 128 lexical
parameters; `token_id % 16` selects the local prototype while the shared
embedding supplies full-vocabulary semantics. At vocabulary 2,048, however,
this aliases 128 unrelated tokens into each of only 16 private buckets.

The asymmetric private-hash run completed 100k steps at 27.5% token accuracy,
loss 3.7253, and PPL 41.48. A symmetric private-hash screen reached 27.3%, loss
3.7499, and PPL 42.52 at 42k. Both generally emitted a frequent first word and
then visually empty, usually whitespace-filled, or repetitive slots. Symmetric masking alone therefore did not
restore the wide pilot's qualitative multi-token formations.

The next clean run uses `lexical_mode=communal`: the same 62 width-8 cells,
symmetric node graph, four relays, degree-four coupling, data, and optimizer,
but no modulo-16 private tables. Learned cell identities replace the tables,
giving 121,864 parameters. This is intentionally not budget-matched; it isolates
lexical organization before additional cells consume the released budget.

Subsequent work added per-cell feature gates and a grouped width-32 readout,
creating a 124,616-parameter broadcast reference. Gator-S replaces that static
population shell with token stewardship at 124,618 parameters. The current
language-objective investigation uses this capacity-matched pair rather than
returning to the historical private-hash configuration.

## 13. Current next experiments

The alignment-aware language shell and its audit suite are implemented; the
current queue is the controlled training matrix:

1. Preserve the audited PAD, EOS, whitespace, rendering, and legal-response
   semantics as invariants.
2. Run matched `slot_ce` and rigid `blank_ce` controls.
3. Compare their best-aligned diagnostics before changing the training loss.
4. Run detached `viterbi` and differentiable `monotonic` aligned objectives.
5. Add the modest whole-sequence composition/length/EOS auxiliary as a separate
   low-weight A/B after alignment has a matched result.

Every stage retains the previous stage as a control and keeps architecture,
data, seed, optimizer, and stewardship schedule fixed. See
[`GATOR_FORMATION_ALIGNMENT_ROADMAP.md`](GATOR_FORMATION_ALIGNMENT_ROADMAP.md).

The earlier queue was to finish the communal lexical A/B, compare normal,
shuffled, and visually empty contexts, preserve width 8/four-node cells,
compare cell versus node scaling, and add checkpoint/route diagnostics. Those
items are historical provenance rather than the current experiment ordering.

### 13.1 Deferred open issue: late-training destabilization

Flagged 2026-08-10 during the node/`d_model`/routing-capacity scaling sweep
(`gator_cells_vectorized.py`, see `Runbooks/GTR_RUN_REGISTRY.md`) as
a separate problem to come back to, not something to chase mid-sweep. Two
independent runs both showed a late-training grad-norm spike coinciding with
an accuracy regression, on two different architectures in the same GTR
family:

- **GTR-ARW, `arw_nodes8_trace_seed0`** (local WSL, `--nodes 8`, otherwise the
  10,000-param mastery reference config): `acc=0.956` at step 16,000, then
  dropped to `acc=0.705` at step 17,000 with `grad_norm` spiking to 72.98
  (vs. a 14-38 range over the preceding several thousand steps).
- **GTR-L, `gtrl_len7_adamw_trace_seed0_gatebias3_noreadoutbias_routerocc_LR3e-4_slots4`**
  (Colab, plain AdamW, `--length 4`): peaked at `valid_acc=0.557`/
  `valid_exact=0.082` at step 407,000. This run was believed killed at step
  457,500 (`grad_norm` spike to 80.1) but turned out to still be running —
  re-checked at step 700,500, it had continued declining for another ~240k
  steps to `valid_acc=0.326`/`valid_exact=0.012`, with a much larger
  destabilization event at step 594,500 where `grad_norm` spiked to
  **1,055.38** (more than 13x the 80.1 figure first flagged, and two orders
  of magnitude above the single/low-double-digit `grad_norm` typical of the
  rest of the run). The last 25 logged points (steps 688,500-700,500) show
  the run oscillating in a degraded regime (`valid_acc` 0.26-0.38, several
  more secondary `grad_norm` spikes in the 40-130 range) with no sign of
  recovering toward the 0.557 peak. This is now the clearest evidence in the
  project for this issue's severity, not just its existence.

Both are constant-learning-rate runs, and `Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md`
(a root-cause study on the same *symptom* — late-training regression under
constant LR — in the older, larger production Gator language model) already
found LR schedule to be a real, if partial, mitigation there, plus a batch-
size sensitivity that scales with width (section 8-9 of that study). Whether
the same mechanism explains this GTR-ARW/GTR-L instance, or whether it's
architecture-specific to the cell-ring/linear-shell family, is untested and
deliberately not being investigated right now — the current sweep's purpose
is node-fattening and routing capacity, not stability. Worth a dedicated pass
later: does a decaying/cosine LR schedule suppress it here the way it did for
the older Gator model, and does the spike's severity scale with `d_model`/
`n_nodes`/`routing_dim` the way plateau severity scaled with parameter count
in that study?

## 14. Performance comparison rule

The completed 100k private-hash language log reports 18,582.35 seconds and
5.38 steps/second. Operator estimates place a 50k language screen near four
hours on T4 and forty minutes on RTX Pro 6000, an apparent approximately 6x
improvement. This is a provisional observation, not a controlled GPU scaling
result, because the exact software, compilation, precision, and model settings
were not locked.

Gator and Transformer must be compared by time to matched held-out quality,
not raw steps/second. Required axes are validation loss, perplexity, token
accuracy, exact-window accuracy, training tokens, wall-clock time, memory, and
eventually energy. A faster architecture that remains at chance has not won a
useful efficiency comparison.

### Progressive output-span observation

During simultaneous language training, Gator initially emitted approximately
one coherent word and then gradually added words across later checkpoints. It
did not immediately exploit the complete 32-slot output allocation. Since all
slots are computed together and no predicted token is fed back, this may be an
emergent expansion of coordinated formation span across fixed addresses.

The observation is qualitative and can be confounded by predicted EOS and
tokenization. Subsequent runs should report the distribution of predicted
piece counts before EOS, threshold-crossing steps for 1/2/4/8/16/32 pieces,
target-normalized length, and position-wise accuracy.

Later inspection confirmed that many visually blank positions were standalone
whitespace predictions. Because whitespace is an independent frequent token in
the compact tokenizer, raw token accuracy is inflated and progressive span may
partly represent content tokens replacing a whitespace canvas. Corrected runs
must attach whitespace to lexical pieces and report content-only accuracy,
whitespace prediction rate, and EOS behavior before topology comparisons.

### Formation-alignment diagnosis

The historical `slot_ce` language objective requires reference token `j` to
occupy output slot `j`. For a short complete target it scores EOS and ignores
the PAD tail, even though every tail slot emits a distribution. Version-1 runs
also initialized output queries from MASK. That control can reward a useful
prefix or whitespace canvas while hiding address-phase slips and malformed tail
predictions.

Tokenizer V2 now reserves a distinct non-rendering BLANK without renumbering
existing ID regions, and V2 Gator initializes output queries from it. Rigid
BLANK-tail CE, best-aligned diagnostics, detached
Viterbi CE, differentiable forward-sum alignment, and the optional
composition/length/EOS auxiliary are implemented behind separate controls.
They remain output-shell experiments, not established Gator improvements;
the next evidence comes from matched GPU training runs.

## 15. Reproducibility artifacts

Primary scripts and results are stored in the parent GTR project:

- `gtr_fixed_slots.py`
- `gtr_a_experiments.py`
- `parallel_prefix_benchmark.py`
- `transformer_prefix_scaling.py`
- `transformer_memorization.py`
- `gator/models/stewardship.py`
- `gator/alignment.py`
- `tests/test_alignment.py`
- `tests/test_language_semantics.py`
- `tests/test_formation_objectives.py`
- `tests/test_stewardship.py`
- `Papers/GATOR_FORMATION_ALIGNMENT_ROADMAP.md`
- `gtrarw_curriculum_checkpoint.pt`
- `experimental_checkpoints/gtr-arw-130k-assym-100ksteps.txt`
- `experimental_checkpoints/gtr-arw-130k-sym-42ksteps.txt`
