# Gator Scaling and Resource-ROI Study

**Status:** active Colab L4 study  
**Primary objective:** minimize fully unsupported/free content perplexity  
**Reference task:** 2,048-token vocabulary, 64 context slots, 32 simultaneous
response slots, batch 32, seed 0

## Hypothesis

Gator's route system behaves like a driver and each private node behaves like
the vehicle executing the selected maneuver. The routes must adapt quickly
enough to choose useful paths without becoming unstable, while node weights
should change more slowly and each node must have enough width to implement a
meaningfully distinct transformation.

This predicts a node-count by node-width interaction. Increasing node count
alone may add route choices without adding useful transformations. If so, the
minimum useful node width should rise with node count, and a model with many
thin nodes should lose to a matched-parameter model with fewer, wider nodes.
It also predicts that route/gate learning may need to be faster than private
node-weight learning once the initial autonomous-transfer phase is complete.

## Experimental invariants

Architecture candidates are compared at identical global steps, not at each
candidate's individually best step. Every run uses the same corpus, tokenizer
policy, validation examples, seed, batch size, optimizer state-continuity
rules, and curriculum controls.

Each initial candidate receives:

1. 4,000 full-teacher updates with all AdamW groups at `1.0x`.
2. 16,000 token-handoff updates with active-slot-normalized unsupported loss,
   private-node LR `1.25x`, and the established route-temperature schedule.
3. A fully unsupported evaluation every 250 updates, independent of the
   candidate's current number of released teacher slots.

The handoff stage resumes `latest.pt` from exactly step 4,000 rather than the
best teacher checkpoint. This prevents differing teacher-best steps from
silently changing total training horizon.

### Width-normalized optimizer control

The primary surface intentionally holds the selected D24 optimizer policy
fixed, but equal AdamW multipliers do not guarantee equal functional node
motion across widths. With layer-normalized inputs, the RMS change induced by
a per-element matrix update is expected to scale approximately with
`LR * sqrt(D)`. A constant `1.25x` private-node multiplier may therefore move
D28 nodes too aggressively and D20 nodes too slowly, accidentally making D24
look architecturally optimal.

Before fitting a width law, replay the N40 edge widths from their identical
step-4,000 teacher checkpoints with

`private_multiplier(D) = 1.25 * sqrt(24 / D)`.

This gives `1.369x` at D20, leaves D24 at `1.25x`, and gives `1.157x` at D28;
all other settings, optimizer moments, data order, and global steps remain
unchanged. If these controls materially rescue either edge, report an
optimizer-by-width interaction and use normalized rates in the larger-scale
validation. If they do not, the simpler fixed multiplier remains preferred.

## Node-count by node-width surface

The first screen is a complete 3x3 surface:

| Nodes | Width 20 | Width 24 | Width 28 |
|---:|---:|---:|---:|
| 24 | 105,746 | 117,458 | **130,578** |
| 40 | 115,154 | **130,194** | 147,154 |
| 64 | **129,266** | 149,298 | 172,018 |

Entries are exact trainable parameter counts. The bold diagonal compares
approximately 130k-parameter allocations directly: 24 wider nodes, the
40-node reference, and 64 thinner nodes. Fixed-width columns measure the
marginal value of nodes when per-node capacity is held fixed; fixed-node rows
measure the marginal value of width when route choice is held fixed. Together
these slices identify an interaction that a one-dimensional sweep cannot.

### Fixed-routing geometry control

Node count is not varied in isolation from graph geometry when degree and
relay depth remain fixed. Before observing the quality outcomes, the exact
degree-3, R2 small-world reachable sets were measured for the ten fixed
ingress lanes:

| Nodes | Mean nodes reachable per lane | Union nodes | Union coverage | Lane-pair overlap |
|---:|---:|---:|---:|---:|
| 24 | 6.1 | 24 | 100.0% | 64.4% |
| 40 | 6.0 | 39 | 97.5% | 46.7% |
| 64 | 6.3 | 47 | 73.4% | 24.4% |

These are legal static reachable sets before learned mass and active top-k
selection, not realized route utilization. They show two distinct scaling
pressures: the 24-node graph forces much greater inter-lane collision, while
the 64-node graph leaves more than one quarter of its nodes unreachable by R2.
The 40-node reference happens to sit near full graph coverage with moderate
lane overlap. Therefore a diagonal win by N40/D24 cannot be attributed to
node width alone.

If N64 is weak, follow it with a routing-capacity rescue before concluding
that node count has poor ROI. At N64, degree 4 raises static R2 union coverage
to 92.2%; keeping degree 3 and adding a third relay raises it to 93.8%.
Those changes activate `26,196` and `26,640` private parameters respectively,
versus only `20,868` under degree-3 R2, without widening any node. Degree 4
adds route parameters and fanout; R3 adds no parameters but incurs another
communication/computation round.
Comparing those rescues against wider N64 nodes will separate insufficient
per-node transformation capacity from insufficient communication radius.

### Vocabulary tax and effective brain size

At this scale, total parameters substantially overstate the node substrate.
The matched diagonal allocates parameters as follows:

| Candidate | Lexical | Lane/core | Private nodes | R2-reachable private | Route/gate | Node + route |
|---|---:|---:|---:|---:|---:|---:|
| N24/D28 | 85,504 | 19,880 | 20,256 | 20,256 | 4,938 | 25,194 |
| N40/D24 | 82,944 | 15,120 | 25,120 | 24,492 | 7,010 | 32,130 |
| N64/D20 | 80,384 | 11,000 | 28,416 | 20,868 | 9,466 | 37,882 |

Thus roughly 62--65% of each nominal 130k model is the lexical interface, and
only 20--28k parameters are private-node transformations. After excluding
nodes that no ingress lane can reach within R2, the matched diagonal contains
only about 20--24.5k *participating* private parameters. The N64/D20 headline
count is especially deceptive: 7,548 of its private parameters sit in 17
statically unreachable nodes and receive no task path. Equal total params do
not mean equal node, active-node, or routing budgets. For the fixed ten-lane task, the
exact total is

`P(N,D) = 67,634 + (N + 20)D^2 + (800 + 8N)D + 28N`,

while each private node contains `D^2 + 2D + 4` parameters. A raw `N/D` ratio
is therefore only a candidate explanatory variable; the study must also test
private capacity per node, aggregate private capacity, and route capacity.
At the small surface widths, each node contains only `444` parameters at D20,
`628` at D24, and `844` at D28. Moving D20 to D24 is consequently a `41.4%`
increase in each node's transformation substrate even though the width label
changes by only four.

A naive approximately 370k fixed-ten-lane continuation would use N54/D56
(`370,202` params), N80/D48 (`369,394`), and N122/D40 (`369,290`). It is not a
clean primary validation: static R2 union coverage falls from `75.9%` to
`60.0%` to `47.5%` across that diagonal. That would make the high-N arm a test
of topology starvation as much as node-to-width ratio.

The primary cross-budget validation will therefore use an approximately 500k
coverage-normalized diagonal:

| Candidate | N | D | Lanes | Total | Private nodes | R2-reachable private | Node + route | R2 union coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Low N/D | 54 | 63 | 15 | 505,383 | 221,346 | 209,049 | 244,074 | 94.4% |
| Mid N/D | 82 | 49 | 23 | 499,627 | 205,246 | 195,234 | 232,564 | 95.1% |
| High N/D | 126 | 39 | 34 | 511,748 | 201,978 | 193,963 | 235,982 | 96.0% |

These candidates reproduce the small diagonal's N/D regimes (`0.857`,
`1.67`, and `3.23`) while keeping total size within `2.4%`, statically
reachable private capacity within `7.8%`, node-plus-route capacity within
`4.9%`, and R2 coverage within `1.6` percentage points. Their node-to-lane
ratios are also close (`3.60`, `3.57`, and `3.71`). Scaling lanes with nodes is
an intentional whole-system normalization rather than a claim that lanes are
irrelevant; a later fixed-N/D lane ablation must measure their own ROI. The
fixed-ten-lane diagonal remains a useful topology-stress control, not the
deciding nonlinear-ratio experiment.

The trainable widths are intentionally selected by the matching constraints,
not by tensor-core alignment. BF16 grouped node compute pads D63 to 64, D49 to
56, and D39 to 40. That padding changes no model parameters or logits beyond
the established BF16 tolerance, but D49 pays a conspicuous physical-compute
tax. Quality can still test the ratio hypothesis; latency/VRAM conclusions
must report the padded widths, and the winning region should receive an
aligned-width confirmation before becoming a deployment recipe.

## Surface results

### 10a -- N40/D24 reference

The 130,194-parameter reference completed the exact 4,000-teacher plus
16,000-handoff schedule. At global step 20,000, fully unsupported content PPL
was `171.13` (CE `5.1424`), top-1 accuracy `4.70%`, top-5 accuracy `24.64%`,
and formation coverage `14.63%`. Seven of 31 eligible teacher tokens had been
released. The best sampled free PPL was `168.99` at step 19,000; normalized
free-CE curve AUC was `5.52561`, so candidate ranking must use the full curve
and matched endpoint rather than that isolated minimum.

The two stages totaled `778.52` training seconds at `25.69` steps/s. Native
training peaked at `395,802,112` allocated CUDA bytes and `440,401,920`
reserved bytes. Compiled batch-1 fully free inference produced all 32 output
slots in `8.82 ms` mean (`8.77 ms` median), or `3,628` simultaneous slots/s.
The completed archive was hash-verified and preserved on Drive as
`scaling_roi_colab/10a_n40_d24.zip`.

This fresh trajectory released fewer slots and had higher free PPL than the
historical 08f run despite matching its nominal configuration. Small early
numerical differences can compound through the adaptive handoff controller;
therefore only within-sweep matched candidates are ranked, and the planned
common-release replay remains necessary if diagonal exposure differs.

### 10b -- N24/D28 matched-parameter diagonal

The 130,578-parameter wider/fewer-node candidate completed at free content PPL
`230.37` (CE `5.4397`), top-1 accuracy `3.55%`, top-5 accuracy `23.06%`, and
formation coverage `8.07%`. It released three teacher tokens. Relative to
N40/D24, endpoint PPL was `34.6%` higher and free CE was `0.2973` nats worse;
free-CE curve AUC was also worse (`6.17120` versus `5.52561`). This is not yet
a universal ratio claim, but it decisively rejects equal quality for these two
approximately 130k allocations under the selected policy.

The objective mismatch is unusually clear. At step 20,000, N24/D28 had
*better* teacher content PPL (`35.33` versus `43.81`) and better partial content
PPL (`37.73` versus `46.81`) while its fully free PPL was much worse. It did
not release a second slot until step 19,000, whereas N40/D24 began advancing at
step 12,250. Teacher and partially supported quality therefore cannot stand in
for autonomous formation quality.

N24/D28 is a useful efficiency point despite losing on quality. It averaged
`27.12` training steps/s, peaked at `342,017,536` allocated CUDA bytes, and
had `8.25 ms` median batch-1 free latency, compared with `25.69` steps/s,
`395,802,112` bytes, and `8.77 ms` for N40/D24. The quality loss must be priced
against those compute savings rather than obscured by the equal total parameter
count. Its Drive archive SHA-256 is
`2d16f9903cfa3749689c9b5efc42ebc136041f2c3dfbc1554acfeef095647b94`.

The endpoint differs by four released slots, crossing the preregistered
common-exposure threshold. A fixed-visibility diagonal replay is therefore
required before assigning the entire gap to architecture rather than an
architecture-by-curriculum feedback loop.

### 10c -- N64/D20 matched-parameter diagonal

The 129,266-parameter many/thin candidate completed at free content PPL
`196.25` (CE `5.2784`), top-1 accuracy `3.37%`, top-5 accuracy `22.96%`, and
formation coverage `10.51%`. It released two teacher tokens. This places it
between the other diagonal candidates at the common endpoint: `14.7%` worse
PPL than N40/D24 but `14.8%` better than N24/D28. Its free-CE curve AUC was
`6.10512`, much worse than N40/D24 and only modestly better than N24/D28,
because most of its free-quality gain arrived in the final 1,500 updates.

Teacher telemetry again ranked in the opposite direction from deployment
quality. N64/D20 had the diagonal's best endpoint teacher content PPL
(`33.23`) and partial content PPL (`33.73`), yet it did not release its second
slot until step 19,750. Across the diagonal, teacher PPL ranks N64/D20,
N24/D28, N40/D24 while free PPL ranks N40/D24, N64/D20, N24/D28. This is direct
evidence that teacher quality is not a valid proxy for Gator's simultaneous
unsupported formation objective.

The candidate averaged `24.18` training steps/s and peaked at `484,580,864`
allocated CUDA bytes. Batch-1 free inference had `8.56 ms` median latency.
More nodes were therefore substantially heavier during training than the
fewer/wider allocation, even at nearly identical total parameters. Its Drive
archive SHA-256 is
`45774c9efd6dc2ef14b55177e0bda65c75615a5c7dc103a49b11ad490771bfe7`.

The diagonal establishes a local interior winner, not a fixed-ratio law. The
N64 result is still confounded by width and static reachability: its private
nodes are only D20 and 26.6% of nodes are unreachable from the ten ingress
lanes by R2. A degree/relay or lane-count rescue is required to determine
whether the remaining gap is per-node capacity, communication radius, or
their interaction. Its five-slot exposure difference from N40 also confirms
the need for the common fixed-visibility replay.

### 10d -- N24/D24 fixed-width node-count contrast

The 117,458-parameter N24/D24 candidate completed at free content PPL
`164.57` (CE `5.1034`), top-1 accuracy `4.70%`, top-5 accuracy `24.50%`, and
formation coverage `13.24%`, with seven teacher tokens released. It therefore
beat the same-width N40/D24 reference's endpoint PPL (`171.13`) while using
`9.8%` fewer total parameters, `40.0%` fewer private-node parameters
(`15,072` versus `25,120`), and `39.6%` fewer node-plus-route parameters
(`19,394` versus `32,130`). The endpoint CE gain was approximately `0.0391`
nats. This rules out a monotonic "more private brain is better" explanation at
D24 and shows why the full interaction surface is necessary.

N24/D24 learned later rather than dominating throughout. Its normalized
free-CE curve AUC was `5.67278`, worse than N40/D24's `5.52561`, but its best
sampled free PPL reached `156.60` at step 19,000. The matching seven-slot
endpoint removes curriculum exposure as the obvious explanation for this
specific contrast. Static R2 union coverage is also nearly saturated in both
models, so the result points toward excess routing/search burden or
insufficient per-node capacity to justify the extra N40 population, not merely
dead nodes.

The smaller population also improved resource ROI: `26.55` versus `25.69`
training steps/s, `316,311,552` versus `395,802,112` peak allocated CUDA
bytes, and `8.21 ms` versus `8.77 ms` median batch-1 free latency. Its
hash-verified Drive archive is `scaling_roi_colab/10d_n24_d24.zip` with
SHA-256 `1002a42c2eefdc8a48a982a7785f8da2b4d975b1db9f44c5cb1d2148c9aaa5fa`.

### 10e -- N64/D24 fixed-width node-count contrast

The 149,298-parameter N64/D24 candidate completed at free content PPL
`201.70` (CE `5.3068`), top-1 accuracy `4.81%`, top-5 accuracy `22.67%`, and
formation coverage `16.36%`, with six teacher tokens released. Its best
sampled free PPL was much better (`165.79` at step 13,750), but the endpoint
regressed sharply. Normalized free-CE curve AUC was `5.61338`: worse than
N40/D24's `5.52561`, although better than N24/D24's later-learning `5.67278`.

At fixed D24, increasing N40 to N64 added `60.0%` more installed private-node
parameters (`40,192` versus `25,120`) and `20.5%` more statically reachable
private parameters (`29,516` versus `24,492`), yet endpoint free PPL worsened
by `17.9%`. The candidate formed nonblank content broadly and achieved the
column's highest endpoint coverage, but distributed probability over the
wrong content more heavily. This separates formation breadth from lexical
quality and shows that additional reachable node substrate is not
automatically useful capacity.

Teacher telemetry again gave the opposite recommendation. N64/D24's teacher
content PPL was `35.33`, substantially better than N40/D24's `43.81`, while
its autonomous endpoint was substantially worse. The early free-PPL minimum
followed by degradation is consistent with a growing route/search burden or
semantic drift across a larger node population. Width normalization cannot
explain this fixed-D contrast; topology rescue and utilized-node traces are
the required controls.

N64/D24 averaged `24.63` training steps/s and peaked at `511,031,808`
allocated CUDA bytes, versus `25.69` steps/s and `395,802,112` bytes for
N40/D24. Batch-1 free inference reached `8.93 ms` median latency. Thus the
extra nodes cost `29.1%` more peak training allocation while reducing endpoint
quality. Its Drive archive SHA-256 is
`5fba642c8cf775346a900506193dc41f2617e6e81538aaaf53a24800ff477c61`.

### 10f -- N40/D20 fixed-node width contrast

The 115,154-parameter N40/D20 arm completed at free content PPL `173.45`
(CE `5.1559`), top-1 accuracy `3.80%`, top-5 accuracy `23.42%`, and formation
coverage `8.07%`. It was only `1.36%` worse in endpoint PPL than N40/D24, but
lost more clearly on accuracy and formation coverage. Its free-CE curve AUC
was also worse (`5.63524` versus `5.52561`), and its best sampled free PPL was
`170.97` at step 19,000.

The controller exposed why the scalar endpoint is insufficient. N40/D20
advanced to two released slots, regressed, and restored the slot at step
10,000; it finished with only one removed token while N40/D24 sustained seven.
Despite receiving substantially more teacher support, D20 did not beat D24's
fully free quality. Its teacher content PPL was much better (`34.65` versus
`43.81`), again demonstrating that supported-path quality can improve while
autonomous transfer worsens. At 444 parameters per node, the node substrate
appears too thin to preserve the frontier behavior needed for the route
contract as support is removed.

The nominal savings did not translate into proportional hardware savings.
D20 has `29.3%` fewer private and statically reachable-private parameters than
D24 at N40, but BF16 node compute pads D20 to the same physical width 24. It
trained at `25.48` versus `25.69` steps/s, peaked at `371,714,560` allocated
CUDA bytes (`6.1%` below D24), and reached `8.45 ms` median batch-1 latency
(`3.6%` below D24). The Drive archive SHA-256 is
`debe2015aafc56720c04a8939e65938b40ac8e95b8fb196dd8731d96125a8460`.

### 10g -- N40/D28 fixed-node width contrast

The 147,154-parameter N40/D28 arm completed at free content PPL `184.44`
(CE `5.2176`), top-1 accuracy `4.02%`, top-5 accuracy `23.49%`, and formation
coverage `10.58%`. Its best sampled free PPL was `176.07` at step 19,000 and
its normalized free-CE curve AUC was `5.74875`. It sustained five released
teacher tokens, a clear curriculum improvement over D20's one, but it did not
match D24's seven released tokens or deployment quality. Relative to N40/D24,
the endpoint PPL was `7.8%` worse and the curve AUC was also worse (`5.74875`
versus `5.52561`).

This makes D24 the observed interior winner along the fixed-N40 width row,
despite D28 supplying `34.4%` more private-node capacity (`33,760` versus
`25,120`) and the same increase in statically reachable private capacity
(`32,916` versus `24,492`). The extra width was not useless: D28 recovered
from very poor early free formation, eventually released five slots, and
finished with higher free top-1 accuracy than D20. However, its additional
node substrate did not convert into better matched-step free PPL than D24.

The result is still optimizer-confounded in a specific, preregistered way.
Holding the private-node multiplier at `1.25x` increases expected functional
node motion with width, so D28 may have been updated about
`sqrt(28 / 24) = 1.08x` more aggressively than D24. The scheduled replay from
the exact teacher checkpoint lowers only D28's private multiplier to `1.157x`.
That completed control is reported below and did not materially rescue D28.

N40/D28 averaged `24.70` training steps/s and peaked at `431,139,328`
allocated CUDA bytes. Batch-1 free inference reached `8.55 ms` median latency;
that small apparent latency advantage over D24 is within the range where
kernel-selection noise must be checked with repeats. The hash-verified Drive
archive is `scaling_roi_colab/10g_n40_d28.zip` with SHA-256
`38524b18185e4d658704f38769ae446006e9418000f08125936de598fc400cd7`.

### 10h -- N24/D20 fixed-node width contrast

The 105,746-parameter N24/D20 arm completed at free content PPL `182.42`
(CE `5.2063`), top-1 accuracy `3.44%`, top-5 accuracy `23.60%`, and formation
coverage `8.36%`. Its best sampled free PPL was `176.69` at step 19,500 and
its normalized free-CE curve AUC was `5.80985`. It finished with one released
teacher token. Along the N24 row, D24 remains the observed interior winner:
D20 was `10.8%` worse than N24/D24 at the matched endpoint, while D28 was
`40.0%` worse. The full curve has the same ordering (`5.67278`, `5.80985`,
and `6.17120` for D24, D20, and D28 respectively).

Only `10,656` parameters, or `10.1%` of the headline model, are private-node
weights. All 24 nodes are statically reachable, so unlike the N64 arms this
result is not explained by dead topology. Increasing N24 from D20 to D24 adds
only `11.1%` headline parameters but adds `41.4%` private capacity per node
and aggregate private capacity. The accompanying free-CE gain is `0.1030`
nats. Increasing again to D28 adds another `34.4%` private capacity but loses
quality, reinforcing a local optimum rather than monotonic returns from raw
node substrate.

Supported-path telemetry again points the wrong way. N24/D20 ended at teacher
content PPL `33.38` and partial content PPL `32.70`, both better than
N24/D24's `42.28` and `41.41`, while fully unsupported quality was worse and
the controller released six fewer slots. The thin nodes can fit the heavily
supported trajectory but transfer it to autonomous simultaneous formation
less effectively.

N24/D20 was the surface's lightest and fastest arm so far: `28.01` aggregate
training steps/s, `301,314,048` peak allocated CUDA bytes, and `7.92 ms`
median batch-1 free latency (`4,032` simultaneous slots/s). Those savings are
real, but they buy lower autonomous quality than D24. Its hash-verified Drive
archive is `scaling_roi_colab/10h_n24_d20.zip` with SHA-256
`82350d92c38c5ae905934f2762836b27acf7d6b3a4bc51a07b80fd48ab3009d3`.

### 10i -- N64/D28 fixed-node width contrast

The final 172,018-parameter arm completed at free content PPL `184.92`
(CE `5.2199`), top-1 accuracy `4.30%`, top-5 accuracy `23.21%`, and formation
coverage `12.37%`, with four teacher tokens released. Its best sampled free
PPL was much better (`161.38` at step 19,250), while normalized free-CE curve
AUC was `5.68056`. At N64, D28 therefore won the matched endpoint by `5.8%`
over D20 and `8.3%` over D24. The curve ranking is less decisive: D24's
`5.61338` AUC remained better because it learned earlier before regressing,
whereas D28 improved abruptly near the end. A best-checkpoint-only comparison
would conceal that reversal in training dynamics.

The candidate installed `54,016` private-node parameters, but R2 exposed only
`39,668` in 47 reachable nodes. The validation trace touched 45 nodes, yet
route-mass entropy was equivalent to only `18.55` nodes or `15,654` private
parameters; 18 nodes (`15,192` parameters) carried 95% of route mass. Thus
only `29.0%` of installed private capacity behaved like uniformly used
capacity. N40/D28 actually had slightly more entropy-effective private
capacity (`16,091`) and virtually identical endpoint PPL (`184.44`) despite
installing 20,256 fewer private weights. The extra N64 population is mostly a
low-mass tail under this topology and lane count.

N64/D28 averaged `25.81` aggregate training steps/s and peaked at
`554,998,272` allocated CUDA bytes. Batch-1 free inference reached `8.58 ms`
median latency (`3,730` simultaneous slots/s). Relative to N40/D28, it used
`28.7%` more peak training allocation without a matched-endpoint quality gain;
the small throughput differences are run-level kernel noise rather than a
credible scaling advantage. Its hash-verified Drive archive is
`scaling_roi_colab/10i_n64_d28.zip` with SHA-256
`a9eb05ed6efe8e179bc40ee6df5d3e34742392c8b71788bb50f6b359f0ae630c`.

### 14a -- width-normalized N40/D28 private-node rate

The D28 replay resumed the exact step-4,000 teacher checkpoint and changed
only the private-node multiplier from `1.25x` to `1.157275x`. At step 20,000,
free content PPL was `182.87` (CE `5.20878`) versus `184.44` (CE `5.21735`)
for the original run: a negligible `0.00857`-nat gain. Every trajectory
diagnostic moved the other way. Mean free-CE curve AUC worsened from `5.74875`
to `5.83924`, best sampled PPL worsened from `176.07` to `182.22`, free
top-1 accuracy fell from `4.02%` to `3.44%`, and the controller released only
two tokens rather than five.

Across all 64 exact matched evaluation steps, the lower-rate control won only
27. During the 41 points where both branches still exposed exactly one
released token, it won 17. Its mean free-CE disadvantage was `0.2244` nats
over steps 4,250--10,000 and remained a small disadvantage in both subsequent
windows. The worse AUC is therefore not merely an artifact of the branches'
later curriculum divergence.

The tiny endpoint movement is therefore not evidence that D28 was previously
overdriven. It is smaller than the run-level variation implied by the
oscillatory free curve, while the poorer curve and slower autonomous release
are consistent. The control averaged `25.97` aggregate steps/s, peaked at
`428,647,936` allocated CUDA bytes, and reached `8.23 ms` median batch-1 free
latency. Its archive SHA-256 is
`5038049bbe8299cc0c08cc08683b87f8930e1a600bfb8ade195e287f87eb399a`.

### 14b -- width-normalized N40/D20 private-node rate

The reciprocal D20 replay resumed its exact step-4,000 teacher checkpoint and
raised only the private-node multiplier from `1.25x` to `1.369306x`. This was
strongly harmful: matched-step free content PPL rose from `173.45` to `222.96`,
a `28.5%` regression or `0.2511` nats. Mean free-CE curve AUC also worsened
from `5.63524` to `5.64610`, and best sampled PPL worsened from `170.97` to
`173.32`. Both the baseline and control briefly advanced to two released
tokens and then retreated; the higher-rate branch did so 250 steps later in
each direction. It ended with one released token, `3.05%` free top-1
accuracy, and `6.99%` free formation coverage.

The higher rate won 30 of 64 individual matched PPL samples and posted tiny
mean-CE gains of about `0.012` nats in each of the first two windows. The
ordering reversed decisively after step 15,250: it won only 6 of the final 20
points and lost `0.0599` mean CE. The damage is consequently a late
overshoot/retention failure rather than an inability to begin learning.

Teacher content PPL remained close to the original (`35.35` versus `34.65`),
so supported-path quality would have largely hidden the autonomous damage.
This is exactly the failure mode the free metric is intended to expose. The
control averaged `26.06` aggregate steps/s, peaked at `371,410,432` allocated
CUDA bytes, and reached `8.17 ms` median batch-1 free latency. Its archive
SHA-256 is
`059a2d93dbe02e5ce6e2f9aeb197e0172238f2ed782c1fad33cff6b486fc1939`.

Together, 14a and 14b reject the proposed `1/sqrt(D)` private-node LR
normalization. It catastrophically overdrives the thinner nodes and does not
credibly rescue the wider nodes. The simpler fixed `1.25x` handoff multiplier
therefore remains the selected cross-width policy for this screening budget.
The result also argues against treating per-element functional RMS as the
optimizer's controlling scale: Adam moments, node specialization, routing,
and the adaptive handoff boundary dominate that approximation here.

### Completed surface synthesis

Matched-step free content PPL across the completed surface is:

| Nodes | D20 | D24 | D28 |
|---:|---:|---:|---:|
| 24 | 182.42 | **164.57** | 230.37 |
| 40 | 173.45 | **171.13** | 184.44 |
| 64 | 196.25 | 201.70 | **184.92** |

The width optimum moves right as node count rises. Increasing N24 to N64
changes free CE by `-0.0731` nats at D20 and `-0.2034` nats at D24 (both
losses), but by `+0.2198` nats at D28 (a gain). Conversely, widening D20 to
D28 changes free CE by `-0.2334` nats at N24, `-0.0615` at N40, and `+0.0595`
at N64. The difference-in-differences interaction is therefore strongly
positive at `+0.2929` nats: additional nodes become useful only when each node
is sufficiently capable, directly supporting the coupled-scaling hypothesis.

The local quadratic screening fit does *not* support a fixed N/D ratio. Its
endpoint ridge is approximately `D proportional to N^0.164`, predicting widths
`20.85`, `22.66`, and `24.48` at N24, N40, and N64. The full-curve fit gives a
similar exponent (`0.130`) and widths `22.13`, `23.64`, and `25.13`. These
exponents are descriptive, not a scaling law: the grid is narrow, endpoint
curriculum exposure differs, and N64 has reduced static coverage. The matched
width-normalized LR controls did not rescue either edge and therefore do not
explain away the interior-width pattern. The surface establishes that the
observed relationship is nonlinear and much shallower than constant N/D. The
fixed-visibility validation below confirms an architecture-by-curriculum
interaction; the larger-budget validation remains necessary before treating
the fitted exponent as a scaling law.

### Fixed-visibility diagonal -- one released slot

The matched-parameter diagonal was replayed from the exact step-4,000 teacher
checkpoints with optimizer moments and portable data position restored. Every
branch kept exactly one of 31 scored response slots unsupported for all
16,000 handoff updates; optimizer rates, route-temperature schedule, data,
seed, and the step-20,000 endpoint matched the dynamic-release runs.

| Architecture | Dynamic endpoint PPL | Fixed-one endpoint PPL | Fixed-one CE gain | Dynamic / fixed curve CE | Dynamic / fixed teacher PPL |
|---|---:|---:|---:|---:|---:|
| N40/D24 | **171.13** | 243.92 | -0.3544 | **5.52561** / 5.54727 | 43.81 / **34.28** |
| N24/D28 | 230.37 | **198.06** | +0.1511 | 6.17120 / **5.94844** | **35.33** / 36.28 |
| N64/D20 | **196.25** | 200.63 | -0.0220 | 6.10512 / **6.09402** | 33.23 / **32.70** |

Fixed visibility is therefore not a generally better curriculum. It strongly
damages the N40/D24 endpoint and is effectively neutral at N64/D20, while it
materially helps the fattest-per-node N24/D28 arm. Even where teacher PPL
improves--N40/D24 and N64/D20--free endpoint quality does not. This directly
rejects teacher retention as a selection objective.

The N24/D28 improvement is nevertheless important: it proves that the
surface is partly an architecture-by-release interaction rather than a pure
capacity ranking. That branch learned free decoding very late, finishing at
its best observed PPL, whereas N40/D24 reached a much better best-free point
(`184.31`) and then regressed to `243.92`. N64/D20 spent most of training with
free PPL above 300 before recovering only to its dynamic baseline. The
release controller should therefore remain adaptive and free-metric-driven;
a universal fixed visibility level would favor one node/width regime while
silently handicapping another. Future scaling fits should either include
curriculum state as a covariate or compare both dynamic-release and
common-exposure curves for promoted configurations.

### N64 topology rescue -- fanout beats relay depth

The weak N64/D20 diagonal result was replayed with two coverage rescues under
the same 4,000-teacher plus 16,000-handoff schedule and selected optimizer
policy. Degree-4 R2 increases candidate fanout while retaining four active
nodes per relay. Degree-3 R3 instead adds a complete third relay without
adding parameters.

| Topology | Parameters | Static union coverage | Endpoint free PPL | Best free PPL | Curve CE | Released slots | Train steps/s | Peak allocated | Median latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Degree-3 R2 baseline | 129,266 | 73.4% | 196.25 | 194.98 | 6.10512 | 2 | 24.18 | 462.13 MiB | 8.564 ms |
| **Degree-4 R2** | 132,594 | 92.2% | **167.68** | **160.88** | **5.46998** | 8 | **25.59** | **463.06 MiB** | 8.797 ms |
| Degree-3 R3 | 129,266 | 93.8% | 204.82 | 192.89 | 5.93325 | 1 | 18.79 | 657.48 MiB | 11.831 ms |

Degree-4 R2 is a decisive rescue, not merely a static-coverage improvement.
Relative to the baseline it improves endpoint CE by `0.1574` nats and the
full-curve CE by `0.6351` nats, while adding only `3,328` route parameters,
`0.92 MiB` of peak allocation, and `2.7%` median inference latency. Measured
training throughput was slightly higher rather than lower. It also lets the
adaptive controller advance to eight unsupported slots, versus two for the
baseline. At this scale, expanding the legal route choices while holding
active top-k fixed is exceptionally high-ROI.

R3 demonstrates that static coverage is not a sufficient topology metric.
It reaches one more node than degree-4 R2, but loses `0.0427` endpoint CE to
the baseline, trains `22.3%` slower, consumes `42.3%` more peak allocation,
and raises median latency by `38.2%`. The extra transformation/communication
round is therefore not a cheap substitute for fanout. Its endpoint teacher
content PPL (`31.97`) is the best of the three while its endpoint free PPL is
the worst, providing another direct warning against teacher-metric selection.

This result revises the local interpretation of node ROI. N64 was not simply
too populous or too thin; degree-3 R2 denied the router access to enough of
its installed substrate. When union coverage is poor, increase low-cost
candidate fanout before deleting nodes or widening them. Increase relay depth
only when an explicit depth ablation justifies its substantial sequential
compute and memory cost.

### Coverage-normalized 500k diagonal -- the balanced ratio wins

The cross-budget diagonal completed with total parameters, installed
node-plus-route capacity, reachable private capacity, node-to-lane ratio, and
static R2 union coverage closely matched. It therefore removes the fixed-lane
coverage collapse that confounded the small high-node arm. Every candidate
used the same 4,000-step teacher foundation, 16,000-step adaptive handoff,
active top-k four, degree-3 R2 topology, and selected `1.25x` private-node
transfer rate.

| Regime | N / D / lanes | N/D | Params | Endpoint free PPL | Best logged free PPL | Curve CE | Released slots | Effective private | Effective node + route | Train steps/s | Peak allocated | Median latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fewer, fatter nodes | 54 / 63 / 15 | 0.857 | 505,383 | **159.29** | 159.29 | 5.68779 | 5 | 101,119 | 123,847 | **20.28** | **0.925 GiB** | **8.513 ms** |
| **Balanced** | **82 / 49 / 23** | **1.673** | **499,627** | 161.57 | **147.07** | **5.36958** | **12** | **103,586** | **130,904** | 15.40 | 1.214 GiB | 9.480 ms |
| More, thinner nodes | 126 / 39 / 34 | 3.231 | 511,748 | 177.57 | 160.81 | 5.56672 | 6 | 85,864 | 119,868 | 11.43 | 1.641 GiB | 10.672 ms |

The low-N arm's endpoint lead over the balanced arm is only `0.0142` CE nats
and is not representative of the matched trajectory. N82/D49 beat N54/D63
at 62 of 64 common validation steps. Over the final 16 points their geometric
mean free PPLs were `162.47` and `196.85`; N54/D63 surged from `199.49` to
`159.29` only at the final evaluation, exactly when it advanced from four to
five released slots. N82/D49 also beat N126/D39 at 53 of 64 points, by
`0.2133` mean CE nats across the complete curve and `0.2179` over the final 16
points. Thus N82/D49 is the robust quality and time-to-quality winner, while
N54/D63 is the resource-constrained endpoint option. A single final-step PPL
would select the wrong general training trajectory.

The route-utilization probe explains the shape of the result. N54/D63,
N82/D49, and N126/D39 used `24.67`, `41.38`, and `53.56` entropy-effective
nodes, but those nodes represented `101.1k`, `103.6k`, and only `85.9k`
effective private parameters. The corresponding shares of installed private
capacity were `45.7%`, `50.5%`, and `42.5%`. The high-node arm therefore did
spread mass across more nodes, but each node was too thin to preserve total
usable transformation capacity. Matching installed and statically reachable
brain did not match dynamically usable brain. This directly supports the
coupled-scaling hypothesis: extra nodes cannot be funded by thinning every
node one-for-one.

Routing remained corridor-like rather than prompt-specialized. Lane-to-node
mutual information rose from `3.67` to `4.24` to `4.80` bits across the
diagonal, while validation-example mutual information stayed only `0.029` to
`0.039` bits. Scaling lanes restored coverage, but much of the larger routing
space continued to encode lane identity rather than example-specific expert
choice. A fixed-N/D lane ablation is still required before attributing all of
the measured systems tax to node count alone.

The balanced point reproduces the small N40/D24 reference ratio almost
exactly: N grows by `2.05x`, D by `2.04x`, and private-node capacity by
`8.17x`. This is consistent with a provisional balanced-brain rule
`N/D approximately 1.7`, under which private capacity `N*D^2` grows cubically
as N and D scale together. It is stronger evidence than the confounded local
quadratic ridge, but two budgets do not establish a universal scaling law.
The promoted search region is approximately N/D `1.3`--`2.0`, with static
coverage held above about `90%`; do not extrapolate either extreme.

There is also a checkpoint-policy mismatch. `best_unsupported.pt` minimizes
fully unsupported slot CE, not content CE. Consequently the saved
best-unsupported checkpoints did not preserve the logged content-PPL minima
for any arm (most materially, `160.81` at step 12,500 for N126/D39). Keep the
existing formation-aware checkpoint, but also save a distinct
`best_free_content.pt` keyed to fully unsupported content CE when free PPL is
the deployment objective.

Entropy-effective private-node parameters reveal how small the operative
substrate actually is:

| Nodes | D20 | D24 | D28 |
|---:|---:|---:|---:|
| 24 | 6,489 | 11,721 | 14,192 |
| 40 | 8,848 | 13,242 | 16,091 |
| 64 | 11,470 | 10,698 | 15,654 |

The nine nominal 106k--172k models therefore deploy only about 6.5k--16.1k
entropy-equivalent private weights. More installed or even statically
reachable capacity is not monotonic with quality. The router generally
touches almost every reachable node, but assigns a long tail negligible mass.
Adding each model's route/gate controller to that effective private substrate
gives an operative node-plus-route budget of only about `10.2k`--`28.3k`;
the N40/D24 reference is `20,252` parameters. Thus describing these as
roughly 20k-parameter brains inside 100k-plus lexical shells is quantitatively
fair, provided "brain" is understood as an entropy-weighted routing and node
substrate rather than a formal effective-parameter theorem.
Across candidates, node identity carries `2.58`--`3.26` bits about lane,
only `0.059`--`0.181` bits about output slot, and just `0.023`--`0.070` bits
about validation example. Routing is overwhelmingly organized into stable
lane corridors rather than prompt-conditioned specialists. This makes lane
count, communication coverage, and route-mass balance first-class scaling
variables; raw node count is not usable brain by itself.

The resource Pareto set is correspondingly small. N24/D24 has the best
matched endpoint and dominates every larger model on both quality and peak
training memory. N40/D24 retains the best free-CE learning-curve AUC and is a
useful interior training-efficiency reference. N24/D20 is the cheapest/fastest
endpoint, while N40/D20 remains a quality-versus-size tradeoff. No N64 model
is resource-Pareto-efficient under degree-3 R2 with ten lanes. Topology rescue
must precede any conclusion that N64 itself lacks value.

### Cross-budget resource-compromise rule

Within the original 130k surface, the observed resource-cap envelope has only
two VRAM/latency breakpoints.
N24/D20 is the winner at the smallest measured budget (`0.281 GiB` peak,
`7.920 ms` median), and N24/D24 becomes the quality winner at `0.295 GiB` and
`8.212 ms`. No larger candidate in that surface improves endpoint PPL under
any larger observed cap. By headline parameters there is one additional narrow
breakpoint: N40/D20 wins from `115,154` through `117,457` parameters before
N24/D24 becomes feasible.

Adjacent reductions show why raw nodes are currently the first compromise
knob. Reducing N64 to N40 saves `107.6`--`118.1 MiB` of peak allocation across
the three widths and improves endpoint CE at every width (`0.0026`--`0.1644`
nats). At D24, N64-to-N40 also *increases* entropy-effective private capacity
by about `2,544` parameters because route mass becomes less diluted. Reducing
N40/D24 to N24/D24 saves another `75.8 MiB` and improves endpoint CE by
`0.0390` nats, although N40 retains the better full-curve CE. In contrast,
shrinking N40/D24 to D20 saves only `23.0 MiB` and loses `0.0135` endpoint CE
plus `0.1096` curve CE.

The deployment rule is therefore conditional on topology. First bring
static union coverage into the useful regime with candidate fanout while
holding active top-k fixed; degree-4 R2 recovered N64 for almost no measured
resource cost. Next move an over-populated design toward N/D about `1.7` by
removing nodes and reinvesting enough of the savings in width; N126/D39 to
N82/D49 improved endpoint and curve quality while saving `437.2 MiB`, raising
training throughput by `34.7%`, and cutting median latency by `11.2%`. Under a
hard VRAM or latency cap, move farther toward fewer, fatter nodes: N54/D63
saved another `296.1 MiB`, trained `31.7%` faster, and cut latency by `10.2%`
relative to N82/D49, at the cost of a `0.3182`-nat worse curve and much slower
teacher release. Preserve enough width for each remaining node to execute a
distinct transformation. Relay depth is currently the last topology knob,
not the first.

### Implementation control -- unaligned full-private widths

The first N24/D28 attempt exposed an implementation cliff rather than an
architectural cost. Full-private Convergent node dispatch used grouped BF16
matrix multiplication only when the native width already had a 16-byte
stride. Width 28 therefore fell into the portable Python loop over nodes,
although the shared-adapter implementation already zero-padded such widths.
The interrupted fallback teacher stage took `401.86 s` at `9.95` steps/s.

The native full-private path now zero-pads input and weight feature dimensions
to the next grouped-MM stride, slices the padded outputs, and adds no trainable
parameters. An L4 forward/backward comparison against the portable reference
at D28 measured `0.22%` relative output error, `0.31%` relative weight-gradient
error, and maximum absolute output error `0.00107`, matching the intended BF16
execution regime. The interrupted run was excluded from all quality results.

The first clean optimized D28 teacher rerun completed in `145.53 s` at `27.48`
steps/s: a `2.76x` throughput improvement with `351,788,032` peak allocated
CUDA bytes. After a runtime restart, the final included rerun improved to
`135.59 s` at `29.50` steps/s and `342,016,512` peak allocated bytes. Aligned
D24 follows the same grouped path as before. This control is essential for ROI
analysis: D20 and D28 now measure padded tensor-core compute rather than
Python-dispatch overhead.

## Quality measurements

The primary measurements are:

- free content PPL and free cross-entropy `ln(PPL)` at every common step;
- free content top-1 and top-5 accuracy;
- free formation coverage;
- trapezoidal curve AUC over free cross-entropy, accuracy, and coverage;
- current released-slot count as curriculum telemetry only.

Headline and static-reachability counts are supplemented by a native route
trace probe over the same 128 validation examples. For every candidate it
reports (a) the union of private nodes selected at least once, (b) the
entropy-equivalent number of nodes carrying route mass, and (c) the minimum
node count carrying 95% of route mass. Multiplying each count by the exact
private parameters per node produces installed, statically reachable,
validation-union, entropy-effective, and 95%-mass estimates of the actual node
substrate. Output-slot, lane, and validation-example mutual information with
node identity distinguish useful specialization from a large but uniformly
interchangeable or idle population. The probe reads existing `return_trace`
locations under fully unsupported formation and does not alter training.

An end-to-end 128-example probe on N40/D24 matched the checkpoint's full
validation fingerprint and exposed a still smaller effective substrate. Of
`25,120` installed private parameters, `24,492` were statically reachable,
`23,864` were selected at least once in the sample, but route-mass entropy was
equivalent to only `21.09` nodes or approximately `13,242` private parameters.
Twenty-one nodes (`13,188` private parameters) carried 95% of route mass. Node
identity was strongly lane-specific (`3.21` bits of mutual information) but
weakly output-slot-specific (`0.106` bits) and prompt-specific (`0.059` bits).
This preliminary result suggests that the dense router is primarily assigning
stable lane corridors rather than discovering many prompt-conditioned node
specialists. It is a complete matched evaluation for one candidate, not a
surface conclusion; the same measurement across all nine candidates remains
required.

Ranking uses matched-step endpoint quality and curve AUC. Best-checkpoint
quality is reported diagnostically but cannot replace a matched-step result.
Teacher-forced PPL is telemetry and is not a selection objective.

The adaptive handoff rules are identical, but their realized teacher exposure
can differ because stronger candidates may release slots sooner. The main
surface therefore measures end-to-end architecture quality under the selected
training policy. If matched-parameter candidates differ by more than two
released slots at the endpoint, or their free-CE ranking is close, replay a
common teacher-visibility schedule on the diagonal candidates. That control
separates raw architectural capacity from an architecture-by-curriculum
interaction before inferring a scaling law.

## Resource and ROI measurements

Every candidate records:

- exact trainable parameters;
- lexical, lane/core, private-node, and route/gate parameter counts, including
  private-node and node-plus-route fractions of the headline total;
- aggregate training steps/second and wall time;
- peak CUDA allocated and reserved memory during native compiled training;
- compiled batch-1 inference latency after warmup;
- throughput for one 64-token context producing all 32 output slots with every
  causal response writer set to BLANK (zero teacher tokens visible).

For candidate `c` relative to reference `r`, quality gain is expressed as

`delta_CE = ln(PPL_r) - ln(PPL_c)`.

Positive values are improvements. Marginal experiments report `delta_CE` per
added headline parameter, private-node parameter, statically reachable
private-node parameter, node-plus-route parameter, training second, peak-VRAM
byte, and millisecond of batch-1 latency. Separate
parameter Pareto frontiers prevent the roughly 20k-parameter "brain" inside a
nominal 130k model from being hidden by its lexical shell. Configurations are
also placed on a Pareto frontier over free quality, peak VRAM, and latency; a
dominated model is not recommended merely because one raw metric improved.

## Route-semantic retention probe

The refined hypothesis is a two-timescale contract failure: route parameters
can preserve their values while rapid private-node updates change the state
transition that made a route useful. The router then points to the same node,
but that node no longer performs the maneuver for which the route was learned.
This predicts functional route forgetting without literal route-parameter
forgetting.

Fork a common stabilized checkpoint into short, exactly matched branches:

1. freeze route/gate, lexical, and lane/core parameters and update only private
   nodes at the low and high node rates;
2. freeze private nodes and shared transforms and update only route/gate
   parameters as a route-adaptation-speed control;
3. retain an unmodified checkpoint as the zero-update reference.

On fixed examples, preserve the full per-relay node-location distributions and
measure top-k route overlap, Jensen-Shannon route divergence, private-node
functional drift on fixed probe states, and fully unsupported CE after every
short interval. With fixed ingress and two relays, the first transition is a
useful control: node updates should primarily disturb later routing because
the first private-node transformation has not yet occurred.

The hypothesis is supported only if the high-node-rate branch causes greater
later-relay route drift and free-CE damage than the low-node-rate branch while
the frozen-node route-only branch learns useful route changes quickly. If high
node rate changes routes without harming free quality, or improves both, then
the late `0.8x` result is ordinary optimization/regularization rather than
catastrophic route-semantic forgetting.

### Retrospective continuation probe

A preliminary non-causal probe compared all four 10,000-step continuation
endpoints against their identical step-18,750 source. It used eight fixed free
contexts, top-4 route distributions at each relay, and 256 fixed node-function
probe states:

| Branch | Node/shared rate | Node function rel. RMS | Node parameter rel. RMS | Relay-2 JS | Relay-2 argmax agreement |
|---|---:|---:|---:|---:|---:|
| 09a | 1.25 / 1.00 | 0.4858 | 0.4646 | 0.09586 | 77.15% |
| 09b | 1.00 / 1.00 | 0.4497 | 0.3944 | 0.09523 | 70.62% |
| 09c | 1.00 / 0.80 | 0.4487 | 0.3999 | **0.06992** | **78.87%** |
| 09d | 0.80 / 1.00 | **0.4008** | **0.3380** | 0.09060 | 71.13% |

Reducing the node rate from `1.25x` to `0.8x` reduced private-node functional
drift by about `17.5%` and parameter drift by about `27.3%`, consistent with a
slower-changing computational substrate. Route retention was not monotonic:
the globally slower 09c branch preserved second-relay distributions best.
Because route and shared parameters moved in every historical branch, this is
evidence for node drift but not proof of catastrophic route-semantic
forgetting. The frozen-component experiment above remains required.

## Bounded-driver ablation

After selecting the strongest node-count/width region, fork one common
checkpoint and vary only the route/gate learning-rate multiplier around the
selected late node multiplier. Keep lexical and lane/core rates, route
temperature, optimizer moments, data order, and every architecture setting
fixed. The initial bracket is route/gate `0.75x`, `1.0x`, and `1.25x` with
private nodes at the selected stabilization rate.

This distinguishes insufficient steering response from unstable steering.
Alongside matched-step free CE and curve AUC, record route/gate update ratio,
route entropy/effective support, route churn, winning-lane diversity, and
coverage. A lower route rate losing to `1.0x` would show that routes must react;
`1.25x` degrading stability or free quality would establish the useful upper
bound. Add group-specific clipping only if this sweep exposes damaging route
update spikes that the existing global gradient clip does not control.

## Follow-up structural knob order

After selecting the strongest node-width region, matched-step one-axis
ablations will test:

1. lanes/channels;
2. active route top-k;
3. graph degree;
4. relay depth;
5. small-world rewiring.

Each axis includes a matched-parameter allocation where possible and a
marginal-growth comparison where it is not. The final recommendations will
separate:

- quality-maximizing scaling;
- VRAM-constrained scaling;
- batch-1-latency-constrained scaling;
- the first knob to reduce when a deployment limit is exceeded.

The strongest inferred ratio will then be validated at a larger parameter
budget on the 22 GiB L4 before being treated as a scaling rule.
