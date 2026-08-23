# Pod run 2026-08-22 — pillar-2 threshold sweep across the size ladder

Exploratory tier (ADR-0006). Logged in `docs/run-log.md`. **Nothing here may be cited for a
pillar claim.** Its job was to produce the effect-size estimate a confirmatory PREREG needs,
and it did that — see §Sizing.

Executes `docs/program/pod-run-2026-08-plan.md`. Track A complete at five models, Track C
run, **Track B deliberately not run** — see §What this does to Track B. An unplanned mRS
generalisation probe was added while the pod was up; it did not generalise, and §mRS
explains why that is a fact about the stimulus set rather than about the models.

---

## What ran

H100 SXM 80GB, RunPod, `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. Forward-pass path
only; no circuit-tracer, no nnsight, no transcoders.

`specs/stimuli/ecog_sweep_v0.csv` — the 39 `ecog_v0` vignettes crossed with six criteria,
234 rows, no new clinical writing. Five models, every one on cached weights from the
persistent `/workspace` volume.

Total GPU time for all five sweeps: **under 10 minutes**. Wall time was dominated by nothing
at all — the ladder checkpoints downloaded in ~35s via Xet, not the ~10min `docs/ops.md`
budgets.

### Validity gates, all passed

- **Readout**: `--check-tokens` per model. All ECOG tokens resolve to a single id, `bare`
  surface separable, on all five.
- **Determinism**: the grading prompt never names the criterion, so the six copies of a
  vignette must return one grade. **0/39 vignettes drift, on all five models.** This is the
  check that invalidates the design if it fails; it did not fail.
- **Monotonicity**: answers form a step function — the precondition for a flip point to mean
  anything. 0.86 at 4B rising to 1.00 at 27B.

---

## Result

n = 36 (39 minus 3 `ambiguous`, which are held for pillar 3). One vignette is one trial: the
six criteria collapse to a single flip point before counting, so they are not
pseudo-replication.

| Model | monotone | never elig. | flip == true | flip == own | echo rate |
|---|---|---|---|---|---|
| gemma-3-4b-it | 0.86 | 24/36 | 1/36 p=0.999 | 1/36 p=0.986 | 0.81 |
| medgemma-4b-it | 0.97 | 12/36 | 6/36 p=0.569 | 9/36 p=0.816 | 0.44 |
| gemma-3-12b-it | 0.94 | 0/36 | 10/36 p=0.065 | 9/36 p=0.133 | 0.69 |
| **gemma-3-27b-it** | **1.00** | 11/36 | 9/36 p=0.133 | **11/36 p=0.0013** | **0.39** |
| medgemma-27b-text-it | 1.00 | 10/36 | 8/36 p=0.243 | 10/36 p=0.065 | 0.39 |

One-sided exact binomial against the matched constant-responder baseline. `flip == own grade`
is ADR-0007's coupling cell and the informative one — it holds regardless of whether the
grade is right.

### 1. The 4B does not apply the threshold; the 27B may

Gemma 3 4B scores **below** the always-Yes baseline (1/36 against 6/36) and answers No even
at `ECOG <= 4`, where the key is 39/39 eligible, on 24/36 vignettes. It is not weakly
applying the criterion. Read its monotonicity of 0.86 with that in mind: a stuck always-No
responder is trivially monotone, so most of that 0.86 is constant answers, not step
functions.

Gemma 3 27B is the only cell that clears its baseline, at p=0.0013 — which survives
Bonferroni across all ten tests run here. **This is the outcome the plan named in advance as
the likely awkward one**: the 27B does the task and the 4B does not. It is a capability
threshold, and it is what the size ladder was built to detect.

### 2. Pattern-matching collapses with scale — the cleanest finding here

The `ECOG >= 2` criterion inverts the eligible set. A model reading the criterion answers it
differently from `ECOG <= 1`; a model pattern-matching the criterion's *shape* echoes its
`<= 1` answer. A constant responder echoes 1.00 by construction.

Echo rate across the Gemma family: **0.81 → 0.69 → 0.39** at 4B, 12B, 27B.

This is the most robust result in the run — monotone in size, far from its baseline, and it
does not depend on any of the flip-point machinery.

**Do not quote the companion "direction correct" number on its own.** An always-Yes responder
scores 21/36 = 0.58 on it for free, and no model clears that (best: 27B at p=0.062). The
echo rate is the direction finding; `flip_point.py` does not yet compute that baseline.

### 3. The domain-tuning advantage vanishes with scale

At 4B, MedGemma looks far better than Gemma at coupling (9/36 vs 1/36). At 27B the two are
indistinguishable (10/36 vs 11/36; echo 0.39 both). Whatever MedGemma's tuning supplies for
this task, scale supplies instead.

That 4B comparison also needs the caveat in §Scorer bugs — most of MedGemma 4B's apparent
advantage was grading leniency, not application.

---

## Two scorer bugs found and fixed

**`flip_point.py` used one baseline for two different tests.** `base` was the count of
*true*-grade-0 vignettes, correct for `flip == true`, wrong for `flip == own`, where the
always-Yes responder hits wherever the model's **own** grade is 0. A lenient grader raises
its own coupling baseline, and the old code credited that leniency as application.

Fixed: `score()` now computes `base_true` and `base_own` separately and each test is scored
against its match. `report()` prints both and warns when a model grades 0 more often than the
key does.

Effect on the numbers:

| Model | flip==own | own base | p before | p after |
|---|---|---|---|---|
| medgemma-4b-it | 9/36 | **11** | 0.133 | **0.816** |
| gemma-3-27b-it | 11/36 | **4** | 0.029 | **0.0013** |

MedGemma 4B's coupling is below its own baseline. Gemma 27B's strengthens.

`--selftest` gained the archetype that catches it — **grade-0-collapse**: grade everything 0,
answer Yes to everything. Scores a flawless 39/39 coupling while computing nothing. Under the
old baseline that read as **p = 1.98e-32**. Five tests added to
`tests/test_sweep_scoring.py` (107 → 112).

**The direction check has no baseline at all.** Not yet fixed; see §Next.

---

## Sizing a confirmatory run

This is what the exploratory tier existed to produce.

| Assumption | n per model |
|---|---|
| observed 27B effect (11/36 vs base 4/36) | **22** |
| pooled 27B pair | 39 |
| conservative (base 6/36) | **52** |

One-sided 0.05, power 0.80, via `pillar2_baselines.required_n`.

**The existing 39 vignettes are already at or near sufficient.** Conservatively ~13 more are
needed. Compare the between-item design, which needed 132–258 hand-written vignettes to clear
a constant-answer baseline. The sweep design spends almost no clinician time, exactly as
argued — and clinician time is the bottleneck.

Two things a PREREG must lock, both learned here:

1. **One primary cell, named in advance.** Ten tests were run and one cleared. Register
   `flip == own grade` as primary and everything else as secondary, or multiplicity is fatal.
2. **Both baselines, and which metric each belongs to.** Enforced in code now; it belongs in
   the PREREG text too.

---

## What this does to Track B

Track B was 6 contrastive attribution-graph pairs at 4B, selected from vignettes that flip
cleanly and correctly. **Gemma 3 4B has one correct flip in the entire sweep.** That set
cannot be selected.

More to the point, 4B is the size shown *not* to do the task. Drawing 12 graphs of a
computation the model is not performing explains a null.

Not run at 4B, deliberately.

**The second half of that reasoning was wrong.** This doc originally said attribution is
"4B-only — there is no published transcoder set for 12B or 27B." That premise was inherited
from the plan, never checked, and is false. See §Attribution feasibility below: graphs run
at 27B, which is where the behaviour is. Track B is not dead, it moves up the ladder.

---

## mRS — a generalisation probe that failed on the stimuli, not the models

Unplanned, run while the pod was already up. `make_sweep_stimuli.py` was generalised to take
`--intermediate`, and `mrs_sweep_v0.csv` was generated by crossing the 18 `mrs_v0` vignettes
with six criteria on the 0-6 scale — 108 rows, no new clinical writing. `flip_point.py` was
generalised in the same pass to parse any intermediate's criterion and to derive the direction
check's complement (`>= g` against `<= g-1`) rather than hardcoding ECOG's thresholds.
`ecog_sweep_v0.csv` regenerates byte-identical and every ECOG number is unchanged.

At face value mRS looks *far* stronger than ECOG. Coupling clears for the 12B (5/18, p=0.042),
the 27B (6/18, p=0.011) and MedGemma 27B (8/18, p=0.005), and the direction check clears for
**all five models — including the 4B that fails it badly on ECOG.**

That is not a generalisation. It is a leak.

| | rows naming the defining vocabulary |
|---|---|
| `ecog_sweep_v0` | **0 / 234** |
| `mrs_sweep_v0` | **54 / 108** |

Half the mRS vignettes state the answer outright — *"The patient's modified Rankin Scale score
is 0."* `run_ecog_stimuli.py` warns about exactly this at run start. Splitting on
`leaks_vocab` settles it:

| Model | coupling, 9 that **name** the score | coupling, 9 requiring **recovery** |
|---|---|---|
| gemma-3-4b-it | 1/9 (p=0.65) | 1/9 (p=0.90) |
| medgemma-4b-it | 3/9 (p=0.069) | 3/9 (p=0.62) |
| gemma-3-12b-it | **4/9 (p=0.012)** | 1/9 (p=0.65) |
| gemma-3-27b-it | **5/9 (p=0.0014)** | 1/9 (p=0.65) |
| medgemma-27b-text-it | **5/9 (p=0.0014)** | 3/9 (p=0.32) |

**The entire mRS coupling result lives in the half that hands the model the answer.** On the
vignettes that require recovering the grade, not one of the five clears — the 27B falls from
5/9 to 1/9. What looked like generalisation is lookup.

The pattern repeating across all five models is the persuasive part, not any single p-value.
At n=9 per half this is *no evidence*, not *evidence of no effect* — well below the 22-52
sizing rule.

**Consequences.**

- **mRS cannot serve as the generalisation test as written.** Its vignettes need rewriting
  without the defining vocabulary and expanding past n=22. That is clinician time, the
  bottleneck asset. This confirms CLAUDE.md's existing "not a measurement" note on `mrs_v0`
  rather than overturning it.
- **The ECOG result is untouched, and the contrast now favours it.** 0/234 leakage means the
  27B coupling result was earned on vignettes that genuinely require recovery.
- **One hypothesis worth keeping.** The 4B reads criterion *direction* on mRS (echo 0.44 on
  the no-vocab half) but not on ECOG (echo 0.81). Same model, same criterion structure. If
  that survives a properly written mRS set, the 4B's ECOG failure is not an inability to parse
  the comparison — it is that *recovery load destroys application*. That would be a
  pillar-1 x pillar-2 interaction and squarely on-thesis. n=9 and confounded by scale
  differences, so: hypothesis, not finding.
- **Multiplicity worsened.** This added 15 more p-values (5 models x 3 metrics). Any
  confirmatory PREREG must register its primary cell against the full count, not the ECOG ten.

---

## Track C — residual-stream patching

`patch_grade.py`, four donor/receiver pairs chosen from the 11 vignettes where the 27B flips
exactly at its own grade. Same criterion within a pair, different grades, both directions.
Patch the donor's residual stream into the receiver at the final token position, layer by
layer, and read the eligibility logit.

`--selftest` passed on the pod under torch 2.8.0+cu128 — identity delta 0.00e+00, last-layer
dominance 0.00e+00, no hook leaked. The hook mechanics survive the newer torch.

| Pair | 4B first flip | 27B first flip |
|---|---|---|
| g4 → g1 @ `<=1` | 19/34 (56%) | 31/62 (50%) |
| g1 → g4 @ `<=1` | 16/34 (47%) | 54/62 (87%) |
| g4 → g2 @ `<=2` | never (degenerate, see below) | 27/62 (44%) |
| g0 → g4 @ `<=1` | 16/34 (47%) | 33/62 (53%) |

**The answer flips in both models, at roughly mid-depth.** The 4B's one non-flip is a
degenerate pair, not a negative: its always-No bias makes both donor and receiver answer No,
so there is no answer to invert. At 27B that same pair flips, because the 27B answers the
receiver correctly.

**The headline is that the 4B flips too.** Whatever separates the 4B from the 27B
behaviourally, it is *not* that the 27B has final-position content the answer is computed
from and the 4B does not. Both do. The difference must lie in what that content encodes, not
in whether it exists — which is a more specific and more useful null than the sweep alone
gives.

### Read this cautiously — two limits, both structural

**Dominance makes deep flips near-tautological.** The selftest establishes that patching the
last layer reproduces the donor's logits exactly, since everything downstream at that
position is the final norm and the unembed. So *that* a flip happens is close to guaranteed;
only *how early* it happens carries information. On that reading the 27B's g1 → g4 pair at
87% depth is barely above trivial, while the 44–53% flips are substantive.

**Patching the full residual stream transfers the whole prompt, not the grade.** The two
vignettes differ in their entire patient description, so the injected vector carries all of
the donor's final-position content. This shows the answer is computed from final-position
content; it does **not** isolate a grade representation. The plan's framing — *"if the grade
representation is what the answer is computed from, the answer flips"* — holds in one
direction only.

**The missing control is a same-grade pair**: two vignettes with the *same* grade under the
same criterion. If those flip too, the effect is not about grade at all. `patch_grade.py`
currently *refuses* same-grade pairs by design ("nothing to transfer"), so running the
control needs that guard relaxed behind a flag. That is the first thing to do before any
claim rests on this.

---

## Attribution feasibility — the 4B-only constraint is false

Pressure-tested 2026-08-23 on a fresh H100 80GB, after this doc and the plan had both
asserted the constraint as settled. It does not survive contact.

**The dictionary exists.** Gemma Scope 2 (December 2025) covers the whole Gemma 3 family —
270M, 1B, 4B, 12B, 27B, PT and IT — with SAEs and transcoders for every layer.
circuit-tracer's own README lists Gemma 3 PLTs "originally from GemmaScope-2" as supported at
all those sizes. The circuit-tracer-format repackagings sit under the same author as the 4B
set already in use: `mwhanna/gemma-scope-2-12b-it` and `mwhanna/gemma-scope-2-27b-it`.

There is no `clt` directory at 12B/27B — cross-layer transcoders stop at 270M/1B — but that
is irrelevant here: ADR-0002 pivoted to per-layer transcoders, the 4B set is `transcoder_all`,
and `feature_type == "cross layer transcoder"` is circuit-tracer's generic node label, not a
CLT requirement.

**The memory claim was also wrong**, and by more than the dictionary claim. Measured, one
short prompt, `batch_size=128`, `max_feature_nodes=2000`, no offload:

| | 12B | 27B |
|---|---|---|
| transcoders loaded | 48 layers | **62 layers** |
| VRAM after load | 29.6 GB | **64.7 GB** |
| **peak during `attribute()`** | 36.8 GB | **76.5 GB** |
| attribution time | 6 s | 10 s |
| load time | 116 s | 227 s |
| active features | 42,408 | 56,328 |
| adjacency matrix | 3961² | 4521² |
| transcoder download | 26 GB | 48 GB |
| result | `ATTRIBUTE_OK` | **`ATTRIBUTE_OK`** |

The prior estimate of ~74 GB *static* was itself too high: the real figure after load is
64.7 GB, because `ReplacementModel.from_pretrained` defaults to `lazy_decoder=True`.

**But 27B does not fit at real settings, and this matters more than the table suggests.**
Those numbers are a *minimal* configuration — `max_n_logits=5`, one ~50-token prompt. Rerun
on an actual eligibility prompt (60 tokens) at the script's default `max_n_logits=10`, 27B
peaked at 77.5 GB and **OOMed on all 12 graphs**: `Tried to allocate 316.00 MiB. GPU 0 has a
total capacity of 79.18 GiB of which 272.19 MiB is free.`

So the honest statement is: **27B attribution fits with ~1.7 GB of margin at minimal
settings, and needs `offload=cpu` for real work.** The levers are required, not optional —
`offload={cpu,disk}`, lower `batch_size`, lower `max_n_logits`, lower `max_feature_nodes`,
`lazy_encoder`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` against fragmentation.
An H200 (141 GB) would remove the whole problem and is the right card for 27B attribution.

12B, at 36.8 GB peak, has 40+ GB of headroom and is comfortable at any settings.

**Three operational corrections.**

- **The transcoder reference is a plain path**, not a `//` URI:
  `mwhanna/gemma-scope-2-27b-it/transcoder_all/width_16k_l0_small_affine`. `HfUri.from_str`
  splits on `/` and takes the first two components as the repo id. A bare repo id fails with
  `Could not download config.yaml` — the config lives inside each width/sparsity subfolder.
- **Pick the variant explicitly.** These repos hold every width x sparsity combination and
  total 1.5-2.4 TB. `width_16k_l0_small_affine` is what the 4B run used and what these
  numbers are for; `width_262k` is ~16x larger and not viable at 27B.
- **The cu121 pin is unnecessary.** `circuit_tracer`, `transformer_lens` and `nnsight` all
  install and run on torch 2.8.0+cu128, which also skips a ~2.5 GB wheel download.
  `setup_pod_circuit_tracer.sh:74` pins cu121; a venv built with `--system-site-packages`
  over the base image's cu128 torch works and is faster.

**Untested.** Graph export via `create_graph_files` and the numerical completeness check at
27B; whether MedGemma 27B accepts the Gemma 3 27B transcoders (the architecture-sharing
argument that made the 4B set work for MedGemma 4B, unverified at 27B); and behaviour on
longer prompts, where 76.5/81.5 GB leaves little room.

**Consequence.** The contrastive design the plan specifies is now selectable where it
matters. It was unselectable at 4B because Gemma 4B has one correct flip in the whole sweep;
**the 27B has eleven**, more than the six the plan asked for.

---

## Track B at 27B — contrastive attribution graphs

Run 2026-08-23 once the 4B-only constraint was disproved. Six vignettes from the eleven where
the 27B flips exactly at its own grade, each attributed under two adjacent criteria straddling
that flip point: `ECOG <= g-1` (answers No) against `ECOG <= g` (Yes). The vignette text is
byte-identical across a pair; one digit of the criterion differs and the answer inverts.

Own grades 1, 2 and 4 (no clean grade-3 flip exists), three of six `inferred_symptoms`.
**12/12 graphs exported**, completeness **0.706-0.755** (checked numerically, rule 5), and every
pair inverts as designed — `No` at 0.90-1.00 below the threshold, `Yes` at 0.55-1.00 at it.

Settings were forced by memory: `offload=cpu`, `batch_size=48`, `max_n_logits=5`,
`max_feature_nodes=1500`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The first
attempt at defaults OOMed on all 12.

**Completeness at 27B (0.71-0.76) is below Stage 1's 4B graphs (0.80-0.85).** More influence
flows through error nodes, so the same 16k-width dictionary explains less of the larger
model's computation. That caps how strong a mechanistic claim these graphs support.

### The criterion perturbs the circuit more than the patient does

The contrast the design exists for, with a control the plan did not specify: compare graphs
where only *one digit of the criterion* changed, against graphs where the *entire patient
description* changed instead (different vignettes, same `<= 1` criterion).

| Comparison | What differs | Jaccard |
|---|---|---|
| within vignette, 6 pairs | one digit of the criterion | 0.568-0.704, mean **0.613** |
| across vignettes, 6 pairs | the whole clinical description | 0.729-0.834, mean **0.782** |

Lower jaccard = more perturbed. **Changing one digit of the criterion reorganises the circuit
more than replacing the entire vignette.** The distributions do not overlap — max within
(0.704) is below min cross (0.729) — giving complete separation, Mann-Whitney U = 36/36,
exact one-sided **p = 0.0011**. About 26% of influence flows through features unique to one
side of a pair; those are the candidate threshold-comparison machinery.

This is a mechanistic counterpart to the behavioural result, and it needs no feature labels.

**Three validity checks before believing it.** Feature ids are **stable across graphs** —
every shared `(layer, local index)` carries the identical global id, 592/592 and 656/656 — so
`(layer, feature)` is a sound cross-graph identifier. Graph sizes are comparable (818 vs 798
features), so this is *not* the size-confounded jaccard that invalidated a Round-1 claim.
And the node-type partition is exhaustive, so completeness covers every node.

### What the raw influence distribution says, and why it needs care

Influence concentrates on answer scaffolding, not on the clinical content: 31% on the final
`\n`, 29% on `model`, 18% on the literal `' Yes'` token and 10% on `' No'` from "Answer Yes or
No". The criterion span and the patient text carry ~0.1% each, and only 22 of 69 token
positions hold any feature node.

Read alone that suggests the graphs are mostly Yes/No emission machinery. The contrast above
says otherwise, but the tension is unresolved and probably an artefact of pruning:
`node_threshold=0.8`, `edge_threshold=0.98` and a memory-forced `max_feature_nodes=1500` may be
discarding exactly the criterion-position features. **A lower-threshold rerun on an H200 is the
test**, and it is the first thing to do before any claim rests on these graphs.

### Nodes are unlabelled

`clerp` is empty on every node. Naming a feature needs, cheapest first: a **Neuronpedia**
lookup (Gemma Scope 2 is hosted there with a Gemma 3 27B-IT demo — unverified whether it
covers this variant, and it costs no compute); **max-activating examples** via
`deferred/scripts/find_top_activations.py`, which needs a corpus and a pod and silently
returns zeros without its 2026-06-01 RMS-scale fix; or **ablation**, which is what ADR-0004
actually wants — a named *ablatable* feature rather than a fitted direction.

Rule 3 governs whatever comes back: labels get adjudicated, and `data/deferred/feature_labels.jsonl`
was once filled with confident nonsense from an empty-context bug. Assume an unvalidated
label is wrong.

n=6 per group, exploratory tier. Graphs are in `data/graphs_27b_contrastive/` (242MB,
gitignored) — not in git, and the pod is gone, so that directory is the only copy.

---

## Environment findings

- **`--device-map auto` needs `accelerate`, which is not in the ops recipe.** The flag was
  added 2026-08-21 for this session and had never run on a pod. First 12B attempt died on it.
  `docs/ops.md` forward-pass install list needs `accelerate` added.
- **CUDA 13 hosts block the 12.4–12.9 templates.** RunPod refuses the pairing at deploy.
  Blackwell cards (RTX PRO 6000, 5090, B200) are the ones on CUDA 13 drivers, and they are
  independently wrong for the attribution path — `setup_pod_circuit_tracer.sh:74` pins cu121,
  which has no `sm_120` kernels.
- **`/workspace` is a persistent network volume.** All five checkpoints (~100GB) plus
  `mwhanna/gemma-scope-2-4b-it` survive termination. The 4B tracks now need zero downloads.
- **`google/medgemma-27b-text-it` resolves** (`gemma3_text`). The plan's "id unverified" note
  is settled — it was gated pending terms acceptance, not missing.
- Xet transfer pulled 12B in 23s and 27B in 33s. The ~10min download budget in `docs/ops.md`
  is stale.

---

## Next

1. ~~Give the direction check its baseline.~~ **Done** — landed in code alongside the mRS
   generalisation of `flip_point.py`. The always-Yes comparator is now computed and printed
   for the direction check, and no model clears it on ECOG (best: 27B, p=0.062).
2. **Run the same-grade patching control** before Track C supports anything. Needs
   `patch_grade.py`'s same-grade guard behind a flag; the guard is correct for the main path.
3. **Confirmatory PREREG** at n≈22–52, one primary cell, both baselines named.
4. **An ADR on mechanism-at-scale.** Attribution is 4B-only; the behaviour is at 27B. That
   makes patching the primary mechanistic route, and ADR-0004 rejected probing as a
   *replacement* for feature search — it never ruled on patching as a complement where no
   dictionary exists. Close that before any pillar claim rests on Track C.
5. ADR-0007 is still **Proposed**. This run is the first evidence for the Application framing.

Dropped: Track B attribution, and regenerating `frontend/graph_data/elig_age_pos.json` — it
needs the slow circuit-tracer build to repair one file from a closed stage.
