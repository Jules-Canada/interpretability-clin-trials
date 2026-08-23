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

More to the point, 4B is now the size shown *not* to do the task, and attribution is 4B-only
— there is no published transcoder set for 12B or 27B, and `run_graphs_ct.py` is
single-device with no sharding. Drawing 12 graphs of a computation the model is not
performing explains a null.

Not run, deliberately. The plan reaches the same verdict: *"the answer to that is Track C,
not a larger transcoder."*

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
