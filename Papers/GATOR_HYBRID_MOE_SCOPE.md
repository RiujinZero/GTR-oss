# Hybrid multi-expert Gator scope (2026-07-28)

## Goal

Scale toward a ~2B-parameter "virtual" model built from many independently
trained, frozen, specialized ~100M expert networks plus a single trainable
router — rather than one dense ~2B model — to fit realistic training-resource
constraints (each expert trains alone, sequentially, on hardware that could
never hold a real 2B model).

## Prior art already in this repo

`dual_expert_coordinator/` (`OVERVIEW.md`) is a working 2-expert proof of
concept: a math Gator and a language Gator, each frozen after independent
training, with a tiny character-histogram MLP router doing hard dispatch
(route to exactly one expert per prompt, no blending). Key findings from
that work that carry forward:

- **A single shared topology cannot learn structurally different tasks
  jointly** — extensive sweeps (6 configurations) forcing one Gator to learn
  both math and language always resolved in favor of one task, with the
  other collapsing to a degenerate constant output. This is *why* the
  separate-frozen-experts design exists at all, not just a resource-saving
  choice.
- **Experts must be trained via `gator_convergent.py`'s native training
  loop, not the `gator_language_train.py` wrapper**, for tasks that need
  fresh synthetic batches every step (e.g. math) — the wrapper's finite,
  repeated-corpus JSONL pipeline creates a stickier local minimum that
  native fresh-batch generation escapes.
- **Hard dispatch on raw prompt text sidesteps vocab/tokenizer mismatches
  entirely** — the router never touches any expert's tokens/embeddings,
  only raw text in and (eventually) text out. This is the key property that
  makes heterogeneous experts (different vocabs, different architectures
  even) composable at all.
- Any existing checkpoint (e.g. a production 100M run) can be dropped in as
  a frozen expert with no retraining — see `load_expert()` in
  `coordinator_prototype.py`.

## What's changing for the scaled-up version

1. **Domain split, not just a 2-way split**: math, code, science, dialogue,
   writing, (vision/audio later as genuinely separate modalities). Roughly
   17-19 experts at ~100M params each to hit a ~2B total budget alongside a
   larger router.
2. **Agentic and reasoning are NOT separate experts** — they're capability
   axes that cut across content domains (e.g. "write code that solves a
   differential equation using tool calls" is code+math+agentic at once).
   Decision: fold agentic/reasoning training data into every content
   expert's own training regime, rather than giving them dedicated experts
   that a hard-dispatch router would have to choose between (losing
   whichever skill it doesn't pick).
3. **The router needs real capacity, not a histogram MLP** — content
   domains won't be as lexically trivial to separate as math-vs-language
   was. Current thinking: the router should be a small **transformer**
   (this repo already has `OneShotTransformerLanguageModel` in
   `gator/models` — worth evaluating as a starting point before building a
   new transformer implementation), while experts stay Gators.
4. **The router becomes an orchestrator, not just a classifier**: it needs
   to (a) detect when a prompt needs more than one expert, (b) call
   experts sequentially, reading back their **text** output (not
   logits/embeddings — this is what avoids the vocab-bridging problem the
   original prototype flagged as a blocker for any future blending), and
   (c) decide whether to call another expert or synthesize a final answer.
   Architecturally this is a tool-calling agent loop with each Gator expert
   as a callable "tool", not a single-shot dispatch decision.
5. **Sonic (the custom hedgehog SSM project, `Documents/Sonic v1.4 CPP/`)
   is also an available option for expert (or router?) architecture** —
   not yet decided where it fits. Its geometry-memory subsystem is a real
   architectural differentiator (currently being tested independently for
   the GTR corpus-sanity-check project, see `Papers/GATOR_GOLDEN_V7_MILESTONES.md`
   and memory notes on that side project).

## Open question (blocking further design): Gator vs. transformer vs. Sonic capability profile

Before deciding which architecture handles which role (expert vs. router,
which domains go to which architecture), we need a real comparison of what
each one is actually good/bad at:

- **Gator (Convergent)**: node-routing graph, proven for the domain-expert
  role in the existing prototype. Known weaknesses this session: PPL/quality
  decoupling (fluent-but-off-topic drift, whitespace-collapse under strict
  causal training), architecture-level overhead vs. a plain SSM (measured
  slower than Sonic per-lever, though Sonic's own quality is unconfirmed at
  matched training volume — see `GATOR_GOLDEN_V7_MILESTONES.md`).
- **Transformer** (`OneShotTransformerLanguageModel`, or an external
  architecture): the default choice for the router given it needs strong
  general language understanding to make cross-domain routing decisions and
  potentially synthesize multiple experts' outputs. Not yet benchmarked
  in this codebase's context.
- **Sonic**: custom hedgehog SSM, linear-time in sequence length, has its
  own optional geometry-memory retrieval subsystem (currently disabled for
  the fair corpus comparison in progress). Real-world throughput measured
  slower than Convergent Gator per-lever this session, cause not yet
  isolated (candidates: architecture width/depth choice, remaining
  background geometry overhead, or a genuine SSM-vs-graph-routing
  difference at this specific config).

**Next concrete step**: define what "capability" actually means for this
comparison (raw language-modeling quality at matched params/tokens? training
efficiency/steps-per-second? something task-specific like math or code
competence? long-context handling?) and design a fair, matched-scale
benchmark across all three architectures before committing router/expert
architecture choices.

## Staged build-out plan (once the architecture question above is resolved)

1. Router does single-expert hard dispatch only (same shape as the existing
   2-expert prototype, real transformer instead of histogram MLP, real
   domain-separated training data for however many domains are decided).
2. Add "does this need more than one expert" detection as a second
   capability, once (1) is validated.
3. Add the actual multi-expert call-and-synthesize loop last, once routing
   itself is trustworthy on its own.
