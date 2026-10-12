# GTR-ARW vs GTR-L: tracing a token's path

Written 2026-07-30, after a session of single-variable experiments (learning
rate, scan normalisation, receptive field, contact/homogenisation, hard
routing) found and fixed five real bugs but still left a sharp wall between
length=2 (learns) and length=3 (never escapes chance) on the simultaneous
modular-prefix task. Rather than keep varying levers, this traces a token's
actual path through the validated reference and through GTR-L side by side,
looking for a structural mismatch instead of a tunable one.

Two reference files describe the same mechanism at two scales:

* `gtr_a_experiments.py`, `GTRFixedSlots` (`gtrarw` variant) -- the atomic
  980-parameter Arrow cell: 4 nodes, `d_model=8`, 4 relays, address-anchored,
  relation-aware, controlled-writing. This is the cleanest unit to trace.
* `gator_path/frontier.py`, `LocalPacketFrontier` -- ten of that exact cell
  replicated and connected in a ring (`num_cells=10`, `cell_offsets=(0,-1,1,
  -2,2)`). This is the ~10k-parameter reference this whole investigation has
  been chasing. Every mechanism below is present in *both* files, which is
  why they're read together -- a match at both scales is not a coincidence.

`gator_linear/core.py` is GTR-L, the linear-scaling replacement under test.

## Stage-by-stage token path

### 1. Ingress -- how a slot gets its starting node distribution

**GTR-ARW (`gtr_a_experiments.py:163-170`):**
```python
q = F.softmax(self.start_logits / temperature, -1).expand(batch, length, n_nodes)
```
Every slot -- regardless of its token, regardless of context -- starts from
the *identical*, content-independent, globally learned `start_logits`. There
is no per-token ingress routing in the base cell at all.

**GTR-L (`core.py:865-882`, `initial_occupancy`):** content-conditioned --
`logits = einsum(hidden, ingress_key) + ingress_bias + stream_ingress_bias`,
softmax over nodes. GTR-L's ingress is *more* sophisticated here, not less.
**Not a missing mechanism -- skip.**

### 2. Per-relay memory read -- how a slot learns what's at a node

**GTR-ARW (`_memory`, lines 130-146), relation-aware branch:**
```python
rel_bias = einsum("ijc,nc->ijn", relative_code(length), relative_node_bias)
scores = log(q)[:, None, :, :] + rel_bias[None] + log(write)[None, None, :, None]
alpha = softmax(scores, dim=2)          # receiver i attends over writer j
node_memory = einsum("bijn,bjd->bind", alpha, h)
```
This is a genuine **`[L, L]` receiver/writer attention**, restricted per node,
with a *learned* relative-position bias per node. It is the direct analogue
of the `[T,T,N]` relation kernel in `gator_convergent` -- exactly the
quadratic operator GTR-L exists to remove.

**GTR-L (`LinearSequenceMixer.forward`, `core.py:378-450`):** grouped
convolution (kernel 9) + per-channel *decayed* scans, `O(L)`. This is a
known, deliberate trade, already documented in
`GATOR_LINEAR_DESIGN.md` and `GATOR_LINEAR_DIAGNOSTIC.md` -- a monotonic
decay cannot represent an arbitrary learned relative kernel. **A real,
accepted gap, not a bug** -- but worth remembering it is the single largest
expressivity cut in the whole redesign, and it is *exactly* where a 2-step
task starts needing more than a 1-step task can get away with.

### 3. Routing -- what decides where mass moves next

**GTR-ARW (lines 174-192):**
```python
edge_logits = self.router(cat[h, context, occupancy])
if anchored: edge_logits = edge_logits + address_edges[None]
edge_logits = edge_logits.masked_fill(~adjacency, -1e9)
transition = softmax(edge_logits / temperature, -1)
```
Two router inputs GTR-L does not have:

* **`occupancy = q.mean(dim=1, keepdim=True)`** -- the current *global load*
  per node, fed directly into the router as a feature. The router can see
  how crowded a node already is and route accordingly. This is present in
  `frontier.py` too (`features = cat(payload, context, address, occupancy)`,
  `frontier.py:269-272`) at the 10-cell scale.
* **`address_edges = address_to_edges(address)`** -- a learned, purely
  positional (content-independent) additive bias on edge logits: "at
  absolute sequence position i, prefer edge A over B," baked in from the
  slot's own fixed address code.

**GTR-L (`LinearRelay.forward`, `core.py:766-778`):**
```python
router_input = torch.cat((hidden, context, address), dim=-1)
edge_logits = einsum(query, edge_key) + edge_bias + stream_edge_bias
```
No occupancy term anywhere in `router_input`. `edge_bias` is a plain
`[nodes, degree]` parameter -- not conditioned on the slot's absolute
position at all, unlike GTR-ARW's `address_edges`.

**This is the most actionable finding in this document.** GTR-L's router is
*structurally blind to how crowded a node currently is*. Every load-balancing
signal it has is indirect: an auxiliary `route_balance_loss` added well after
the fact, which we watched get gamed twice today -- first by opening the bus
fully (near-uniform redistribution trivially "balances" the loss), then,
with the bus shut, by collapsing onto a single node (a route-entropy shortcut
the balance loss wasn't weighted heavily enough to prevent). The reference
never needed that auxiliary loss because the router can *see* the state
it's supposed to be balancing and adjust directly, every step, as an input
rather than as an externally-imposed penalty on the output.

### 4. Node transform -- how a node's local computation differs from another's

**GTR-ARW (lines 199-204):**
```python
self.node_transforms = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(n_nodes))
destinations = stack([tanh(transform(base + node_embedding[i])) for i, transform in enumerate(node_transforms)], 2)
h = norm(h + (q_next.unsqueeze(-1) * destinations).sum(2))
```
**Every node gets its own complete, independent `D x D` transform.** At
4 nodes / 980 params this is affordable outright. `frontier.py` does the same
at 10 cells x 4 nodes: `transform_weight: [cells, nodes, D, D]`, again a full
private matrix per location, not shared.

**GTR-L (`core.py:610-624`, `809-820`):**
```python
basis = einsum(signal, self.basis_weight)     # basis_weight: [rank, D, D], rank=4 shared
coefficients = einsum(next_occupancy, self.node_coefficients)   # [nodes, rank]
delta = einsum(coefficients, basis)
```
A **shared rank-4 basis**, selected by a per-node coefficient vector, instead
of a full private transform per node. This is a deliberate design choice
(`GATOR_LINEAR_DESIGN.md` section 6: "instead of evaluating 40 or 400
complete node transforms per stream, evaluate 4-8 shared programs") made
specifically to keep parameters linear in node count as the graph scales past
what a 4-node Arrow cell would ever need. It was never validated against a
task the reference could do easily.

**This is the second most actionable finding.** A rank-4 shared basis is a
real capacity cut relative to full private per-node transforms, and it was
introduced for a scaling reason that has nothing to do with whether it's
*sufficient* for the accumulation task at hand. Worth testing directly:
raise `basis_rank` (cheap, a few lines of config) or, as a sharper test, give
GTR-L full private per-node transforms at small node count to see if the
wall moves.

### 5. Controlled writing -- the schedule that keeps blank slots quiet early

**GTR-ARW (`_memory`, lines 133-135):**
```python
if self.controlled_writing:
    fraction = relay / max(self.relays - 1, 1)
    write[observed_length:] = fraction
```
Output (blank) slots write **zero** mass into shared memory at relay 0, and
linearly ramp to full write weight by the final relay. `frontier.py`'s
`write_gate_schedule` (`prefix.py:459-488`) implements the identical ramp at
the 10-cell scale, and its own docstring calls it "the validated
context-first controlled-writing schedule" -- i.e. this was tested and kept,
not incidental.

**GTR-L (`LinearRelay.forward`, `core.py:696-698`):**
```python
gate = torch.sigmoid(self.write_gate(hidden)).unsqueeze(-1)
message = message * gate
```
A fully learned, unconstrained sigmoid gate, live from step 0, with no
schedule distinguishing context slots from blank/output slots. Every output
slot starts from the *identical* additive `output_query` embedding
(`prefix.py:121-124`) -- so at relay 0, before any relay-driven
differentiation, every blank slot is trying to write the same
near-degenerate content into the shared node mailboxes, and nothing stops
it. The reference explicitly silences exactly this pathway for the first
relay and only opens it once the blank slots have had a chance to receive
real context through reads.

**This is a plausible contributor to the collapse pattern observed all day**
(every stream converging on a shared constant) even though it wasn't
identified as the cause during today's diagnosis -- it was never tested in
isolation.

## Summary table

| stage | GTR-ARW / reference | GTR-L | verdict |
|---|---|---|---|
| ingress | content-independent, global `start_logits` | content-conditioned | GTR-L already exceeds this |
| cross-position memory read | learned `[L,L]` relative attention per node | linear decayed scan, `O(L)` | known, accepted expressivity cut |
| router inputs | `hidden, context, occupancy` (+ positional edge bias) | `hidden, context, address` -- **no occupancy** | **missing mechanism** |
| node computation | full private `D x D` per node | shared rank-4 basis + per-node coefficients | **capacity cut, untested against this task** |
| blank-slot writing | scheduled ramp, silent at relay 0 | fully learned, live from step 0 | **missing mechanism** |
| terminal readout bias | absent when `relation_aware=True` | present by default (we already found removing it helps) | GTR-L's own experiment already agrees with the reference here |

## What this changes about today's priorities

Everything tried today (learning rate, scan normalisation, receptive field,
route-loss weights, heterogeneous stream widths, hard routing) touched
*learned* parameters and hyperparameters. None of it added a router input
the model structurally cannot currently see, and none of it imposed a
schedule the reference relies on. The two structural candidates above --
**occupancy fed into the router, and a controlled-writing schedule for
blank slots** -- are cheap to implement and were never actually tested,
because until this trace they hadn't been identified as missing at all.

Suggested order, cheapest and most load-bearing first:

1. **Add occupancy to the router's input.** Small, mechanical change:
   compute the current per-node marginal load and concatenate it into
   `router_input` alongside `hidden`/`context`/`address`. Directly targets
   the bus-saturation and single-node-collapse failure modes observed twice
   today, and might make `route_balance_loss` unnecessary rather than
   fighting it.
2. **Add the controlled-writing schedule.** Silence (or steeply discount)
   blank-slot writes at early relays, ramping to full weight by the final
   relay -- exactly the reference's `write_gate_schedule`. Cheap, a few
   lines in the trainer/relay call site.
3. **Test node capacity.** Either raise `basis_rank` substantially or, as a
   sharper isolating test, give small node counts a full private transform
   (no shared basis) to see whether the rank-4 compression is the binding
   constraint at this scale, independent of (1) and (2).

None of these are hard-routing-style single-parameter flags -- they're small
but real structural additions, which is exactly the category of thing a
token-path trace was supposed to surface and single-variable sweeps were not
going to find by accident.
