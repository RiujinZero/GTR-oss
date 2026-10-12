# Gator Formation Alignment Roadmap

**Status:** implemented language-output stack; training evidence pending, 2026-07-16
**Scope:** simultaneous response formation for Gator and matched one-shot baselines

## Abstract

The current language system predicts every response slot simultaneously, but
its training target still assumes that reference token `j` must occupy output
slot `j`. This is stricter than the architectural claim requires. It also
leaves every slot after the reference EOS outside the cross-entropy mask, so a
model may form a useful visible prefix while filling the unscored tail with
standalone whitespace or another non-rendering special token.

The next experiment generation separates five changes:

1. Verify padding, EOS, whitespace, and response-vocabulary masks.
2. Add a true non-rendering `BLANK` token distinct from `PAD` and `MASK`.
3. Add best-monotonic-aligned accuracy and loss as diagnostics only.
4. Replace rigid slot cross-entropy with a monotonic latent-alignment loss.
5. Add a low-weight whole-sequence composition/length/EOS objective.

Each stage must retain the previous stage as a matched A/B. No topology,
stewardship, corpus, tokenizer-piece inventory, or optimizer change belongs in
the same comparison.

## 1. Current behavior and the identified mismatch

The pre-alignment implementation had six structural symbols:

| Symbol | Current role | Rendering | Training treatment |
|---|---|---|---|
| `PAD` | Left-pad context and right-pad target tensors | Hidden | Context mask excludes it; target mask excludes it |
| `BOS` | Begin encoded dialogue context | Hidden | Valid context token |
| `EOS` | End a complete assistant target | Stops decoding | Scored only when it fits in the target window |
| `MASK` | Initialize all output queries | Hidden | Not a reference target |
| `USER` | Mark user turns | Rendered as a role label | Valid context token |
| `ASSISTANT` | Mark assistant turns | Rendered as a role label | Valid context token |

That implementation had no distinct `BLANK`. `MASK` did double conceptual duty
as an output-query seed and a visually empty symbol, while response logits
could predict `PAD`, `BOS`, `MASK`, `USER`, or `ASSISTANT`.

Targets are encoded as response pieces followed by EOS. A short complete
target is right-padded with `PAD`, and the padded tail is removed from the CE
mask. A response longer than the formation window is clipped and may contain
no EOS. This creates two legitimate target types that must remain distinct:

- **Complete:** content followed by one EOS.
- **Right-censored:** a full scored prefix whose continuation lies outside the
  formation window; no EOS assertion is made.

The central defect is not ordinary tensor padding. It is that unused output
slots are currently *unscored*. At inference those slots still emit logits.
The model can therefore learn a prefix without learning what empty formation
space means.

## 2. Required token-semantics contract

Tokenizer format `gator-compact-v2` now distinguishes four concepts:

| Token class | May appear in context | May be predicted in response | Visible text | Loss role |
|---|---:|---:|---:|---|
| `PAD` | Only as batch padding | No | No | Always ignored |
| `MASK` | Optional corruption/query marker | No | No | Never means empty output |
| `BLANK` | May initialize output slots | Yes | No | A real aligned emission |
| `EOS` | No ordinary context use | Yes | No; terminates rendering | Required once for complete targets |
| Whitespace piece | Yes | Yes | Yes | Ordinary textual content |

`BLANK` is not whitespace, padding, or EOS. Removing `BLANK` emissions from a
formation must not remove ordinary spaces. `PAD`, `BOS`, `MASK`, `USER`, and
`ASSISTANT` must be masked out of response logits before response loss and
decoding. EOS, BLANK, byte tokens, and learned text pieces remain legal.

For backward compatibility, existing compact-tokenizer IDs are not
renumbered. Version 2 reserves the final requested vocabulary row for one
instance-specific BLANK after the existing learned-piece region and records
the format in serialized state. Version-1 checkpoints remain loadable but
cannot use alignment training until explicitly migrated.

Normal rendering removes BLANK and stops at the first EOS. Diagnostic rendering
must have an option that displays `<blank>`, `<eos>`, and forbidden-special
predictions so apparently short output cannot hide a malformed tail.

## 3. Verification gate before changing the objective

The following tests are release-blocking for the next experiment generation.

### 3.1 Padding

- Changing IDs under a false context mask cannot change logits, routes, route
  gradients, loss, or metrics.
- `PAD` has zero Gator-S ownership and never enters node memory.
- Target tensor padding cannot contribute to any loss or metric.
- Response logits assign zero probability to `PAD` and other forbidden
  structural tokens after masking.

### 3.2 EOS and censoring

- A complete reference contains exactly one scored EOS.
- Slots after an aligned EOS are trained as BLANK, not ignored.
- A clipped reference is explicitly tagged `right_censored`; absence of EOS is
  neither rewarded nor penalized for it.
- Greedy rendering stops at first EOS, while diagnostics report missing EOS,
  multiple-EOS probability, EOS-position error, and non-BLANK emissions after
  EOS.

### 3.3 Whitespace

- Learned whitespace-only pieces and single-byte ASCII whitespace fall inside
  the whitespace mask.
- `PAD`, `MASK`, `BLANK`, and EOS never fall inside the whitespace mask.
- Whitespace omission and adjacency terms never touch tensor padding or BLANK.
- Ordinary isolated spaces remain valid content and survive alignment collapse.
- Content metrics exclude whitespace targets but do not misclassify hidden
  specials as content.

### 3.4 Rendering

- Plain rendering hides BLANK and structural symbols and stops at EOS.
- Diagnostic rendering exposes every slot.
- A formation containing text, interleaved BLANK, whitespace, and EOS round
  trips to the expected visible string.

No learning curve should be interpreted until this gate passes.

## 4. Stage A: BLANK with the existing rigid slot loss

The first causal intervention is intentionally small. Keep reference piece
`j` anchored to slot `j`, but replace the ignored tail of a complete target
with scored BLANK labels:

\[
\tilde y = [y_1,\ldots,y_L,\operatorname{EOS},
             \operatorname{BLANK},\ldots,\operatorname{BLANK}].
\]

All `T` output slots participate for complete targets. Right-censored targets
use the full reference cap, receive no synthetic EOS, and leave later physical
slots unobserved rather than falsely labeling an unknown continuation BLANK.
This experiment answers whether short-formation/whitespace behavior primarily
came from an untrained tail on examples whose endpoint is actually known.

Record separate CE for required text/EOS positions and BLANK positions. A
single aggregate loss can otherwise look better merely because BLANK is easy.

## 5. Stage B: best-aligned metrics without training on them

Let `T` be output slots and `y=(y_1,\ldots,y_L)` the reference sequence after
removing tensor padding. For complete targets, `y_L=EOS`. Define
`A(y,T)` as every length-`T` sequence made by inserting BLANK anywhere while
preserving the order of every reference token exactly once. Complete references
require BLANK after EOS. Right-censored prefixes enter a free absorbing state
after their last observed token, so unknown continuation slots add zero loss.

### 5.1 Viterbi-aligned negative log likelihood

\[
L_{\text{vit}}
=
-\frac{1}{T}
\max_{a\in A(y,T)}
\sum_{t=1}^{T}\log p_t(a_t).
\]

This is a diagnostic at first. It asks how well the model supports its best
legal monotonic placement, even if the rigid addresses are shifted.

### 5.2 Forward-sum aligned negative log likelihood

\[
L_{\text{align}}
=
-\frac{1}{T}
\log\sum_{a\in A(y,T)}
\prod_{t=1}^{T}p_t(a_t).
\]

It can be computed by a two-transition dynamic program. At each slot, either
emit BLANK without advancing the reference or emit the next reference token
and advance by one:

\[
\alpha_{t,j}=\operatorname{logsumexp}\left(
\alpha_{t-1,j}+\log p_t(\operatorname{BLANK}),
\alpha_{t-1,j-1}+\log p_t(y_j)
\right).
\]

Unlike standard repeat-collapsing CTC semantics, this formulation consumes
each reference token explicitly. Adjacent identical text tokens therefore do
not require a separator BLANK.

### 5.3 Greedy best-aligned accuracy

Collapse predicted BLANK, stop at EOS for complete responses, and compute a
minimum-edit monotonic alignment to the reference. Report:

- exact matches `M`
- substitutions `S`
- insertions `I`
- deletions `D`
- `aligned_accuracy = M / (M + S + I + D)`
- aligned precision, recall, and F1 for non-whitespace content
- `aligned_exact`, where the collapsed token sequence exactly matches the
  complete reference

Raw rigid-slot accuracy remains in the log. A large gap between aligned and
rigid accuracy is direct evidence of address-phase slips; a small gap means the
problem lies elsewhere.

## 6. Stage C: replace rigid CE with monotonic latent alignment

Begin with detached Viterbi aligned CE as the lowest-risk bridge:

1. Find the highest-probability legal alignment without gradient.
2. Treat that alignment's target/BLANK emissions as temporary slot labels.
3. Apply CE to all complete-target slots and only the observed aligned prefix
   of right-censored examples.

Then A/B it against the differentiable forward-sum objective. The forward-sum
loss is the intended endpoint because it rewards all viable monotonic
formations rather than committing early to one path.

If a transition is needed, anneal from rigid BLANK-tail CE to aligned loss:

\[
L=(1-\gamma)L_{\text{rigid}}+\gamma L_{\text{align}},
\qquad \gamma:0\rightarrow1.
\]

The schedule is an optimization control, not an architectural requirement.
Do not change stewardship pruning or routing temperature during this A/B.

Alignment needs slack. If target prefixes are allowed to consume all `T`
slots, no alternate placement exists. Separate `reference_tokens` from
`output_slots`, for example at first keeping at most 32 reference pieces while
forming into 40 slots. Report both numbers and the fraction of examples with
zero, low, and ample alignment slack. This separation is implemented through
`--reference-tokens` and `--response-tokens`.

Safeguards:

- Normalize loss per output slot and also report per required reference token.
- Track BLANK posterior mass by position and target length.
- Reject impossible batches where `L > T` rather than silently dropping
  reference tokens inside the loss.
- Preserve the complete/right-censored distinction.
- Keep a fixed rigid-CE baseline and identical batch order for every seed.

## 7. Stage D: a modest whole-sequence quality term

The implemented sequence auxiliary is parameter-free and deliberately weaker
than a learned critic. It checks three global properties while leaving local
ordering to the monotonic objective.

For renderable non-whitespace semantic tokens, compare normalized expected
token counts with reference counts through Hellinger distance:

\[
L_{\text{composition}}
=\frac12\sum_v
\left(\sqrt{\hat q_v+\epsilon}-\sqrt{q_v+\epsilon}\right)^2.
\]

Match expected non-BLANK emissions to the known reference length:

\[
\hat L=\sum_t(1-p_t(\operatorname{BLANK})),\qquad
L_{\text{length}}=\operatorname{Huber}((\hat L-L)/T,0).
\]

Semantic composition and EOS count are scored only when the endpoint is known.
For right-censored references, extra emitted length and unknown continuation
composition are free; only under-forming the observed prefix is penalized. For
complete references, encourage one total EOS emission:

\[
L_{\text{EOS}}=\left(\sum_t p_t(\operatorname{EOS})-1\right)^2.
\]

The implemented aggregate is configurable and defaults internally to:

\[
L_{\text{sequence}}
=L_{\text{composition}}+0.25L_{\text{length}}+0.1L_{\text{EOS}}.
\]

Start with an overall weight near `0.01` and retain plain selected formation
loss as the checkpoint-selection metric. This auxiliary discourages generic
token repetition, missing content, incorrect length, and absent termination
without supplying a task-specific recurrence. It may penalize a valid
paraphrase against a single reference, so it must remain modest.

The implemented total objective is:

\[
L_{\text{total}}
=
L_{\text{formation}}
+\lambda_{\text{sequence}}L_{\text{sequence}}
+\lambda_{\text{route}}L_{\text{route}}
+\lambda_{\text{balance}}L_{\text{balance}}.
\]

Whitespace penalties should be reevaluated after BLANK-tail supervision. They
may become unnecessary and must not be silently carried forward.

## 8. Clean experiment matrix

| ID | BLANK tail | Aligned metrics | Training loss | Sequence term |
|---|---:|---:|---|---:|
| `FORM-0` | No | No | Current masked rigid CE | No |
| `FORM-B` | Yes | Yes, diagnostic | Rigid all-slot CE | No |
| `FORM-V` | Yes | Yes | Detached Viterbi aligned CE | No |
| `FORM-M` | Yes | Yes | Forward-sum monotonic alignment | No |
| `FORM-MQ` | Yes | Yes | Forward-sum monotonic alignment | Composition/length/EOS |

Lock model, seed, corpus order, output-slot count, reference-token cap,
optimizer, stewardship schedule, and checkpoint policy. Use at least three
seeds before selecting an objective. The first screen may use the 124,618-
parameter Gator-S reference, but the same objective code must also run on the
one-shot Transformer control.

Primary selection metrics:

1. Forward-sum aligned validation NLL.
2. Content-aligned F1 and aligned exactness.
3. Complete-response EOS accuracy and post-EOS violation rate.
4. Visible non-whitespace formation length versus reference length.
5. Raw rigid-slot CE/accuracy for continuity with earlier logs.
6. Wall time, peak VRAM, and routes/second.

## 9. Implementation boundaries

The work should become generic API modules rather than trainer-local code:

- `gator/tokenization.py`: versioned BLANK semantics and debug rendering.
- `gator/tasks/language.py`: target completeness metadata and response masks.
- `gator/alignment.py`: Viterbi, forward-sum, collapse, edit metrics, and
  whole-sequence composition/length/EOS aggregation.
- `gator_language_train.py`: CLI composition, logging, and checkpoint policy.
- `tests/test_language_semantics.py`: PAD/EOS/whitespace/BLANK invariants.
- `tests/test_alignment.py`: exhaustive tiny-DP checks and finite gradients.

Alignment is a language-output shell. It does not alter Arrow node routing,
Gator-S stewardship, simultaneous decoding, or the no-token-feedback contract.
All output logits are still produced in one forward pass.

## 10. Claim boundary

This implementation is motivated by a concrete masking and address-assignment
mismatch; it is not evidence that alignment will solve language. A positive result would
show that Gator forms the correct ordered token sequence without requiring a
particular token-to-slot placement. It would not by itself establish semantic
understanding, open-ended generation quality, or an architecture-wide scaling
law.
