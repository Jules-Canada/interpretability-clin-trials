# CLT Init Convergence Test: PT vs IT Target (MedGemma 4B)

## Question

Does targeting `medgemma-4b-it` activations (vs `medgemma-4b-pt`) materially change CLT training dynamics when initialized from Gemma Scope 2 transcoders, on top of the base Gemma 3 → medical fine-tune shift the init already absorbs?

## Why 4B, not 27B

MedGemma 27B is released **IT-only** — no PT variant exists. A clean base-vs-IT comparison can only be run at 4B. The 4B result is the only direct empirical evidence available; qualitative conclusions should transfer to 27B even if magnitudes don't.

This also makes 4B the right place to validate the full pipeline before committing 27B compute (see Two-stage strategy below).

## Why this matters

We need to choose `medgemma-4b-it` vs `medgemma-27b-it` as the backbone for clinical trial eligibility attribution graphs. The IT-shift cost is one input to that decision. Equally important: this experiment is the first place we'll learn whether Gemma Scope 2 transcoders (trained on base Gemma 3) transfer to MedGemma (medical fine-tune of Gemma 3) without retraining from scratch. That answer is upstream of every downstream `ignis` decision.

## Design

Two CLT-Forge runs at MedGemma-4B, identical except for activation source:

- **Run A** — target: `medgemma-4b-pt`
- **Run B** — target: `medgemma-4b-it`

Held constant across both:

- Init: Gemma Scope 2 pretrained transcoder weights (trained on Gemma 3 — see pre-run checks)
- Data: same prompts, same batches, same ordering
- Hyperparameters: identical
- Logging: identical

## Pilot first — do not commit full training run

- 3–5k steps per run
- One representative mid-layer + one late layer (late layers carry most IT-induced shift)
- ~10–20% of the planned training corpus
- Goal: detect divergence in loss / feature-death curves early

If curves diverge meaningfully → quantify the cost, choose variant on scientific grounds.
If curves track within noise → variant choice is unconstrained by training cost.

## Metrics to log

- Per-step reconstruction loss (MSE on residual stream)
- Feature activation rate per batch
- Dead feature count over training
- L0 sparsity (if CLT-Forge logs it natively)
- Per-layer breakdown of all of the above

## Decision rule (specify before seeing curves)

- IT-target loss plateau >5–10% above PT-target plateau → IT incurs real cost
- IT-target time-to-target-loss materially slower → same conclusion
- Otherwise → curves equivalent for our purposes

## Pre-run checks

1. **Gemma Scope 2 → MedGemma 4B init mapping.** Gemma Scope 2 transcoders were trained on base Gemma 3 activations (same architecture generation as MedGemma). Confirm d_model (2560) and layer count (34) match between the Gemma Scope 2 4B transcoders and MedGemma 4B. Key open question: Gemma Scope 2 CLTs are only available for 270M and 1B — the 4B repo lists a `clt` folder but the announcement only confirms single-layer transcoders at 4B. Inspect the 4B `clt` artifacts directly to determine if they are true cross-layer transcoders or a subset. If only single-layer transcoders are available at 4B, document how to use them as init for a cross-layer CLT (encoder weights transfer directly; cross-layer decoder matrices would start random). This blocks everything downstream.
2. Confirm `medgemma-4b-pt` and `medgemma-4b-it` weights are accessible
3. Verify tokenizer compatibility between PT and IT variants (should be identical)
4. Verify activation extraction pipeline handles both backbones with no code path differences
5. **Chat template handling.** IT applies chat templates wrapping the user message. Either strip the template for the IT run or apply equivalent framing to the PT run. Activations must come from comparable token positions. Document the choice in the config.

## Kickoff tasks for the Claude Code session

1. Validate Gemma Scope 2 → MedGemma 4B init mapping (pre-run check #1) — **this is the blocking task**
2. Create matched CLT-Forge configs differing only in activation source → `experiments/configs/`
3. Confirm CLT-Forge already logs the metrics above; add missing instrumentation if needed
4. Implement chat template handling per pre-run check #5
5. Run pilot for both variants on selected layers
6. Plot loss curves and feature-death curves side by side
7. Write decision memo → `experiments/results/pt_vs_it_convergence_4b.md`

## Two-stage strategy

**Stage 1 (this experiment) — 4B for methodology and pipeline validation:**
- Validate Gemma Scope 2 → MedGemma init compatibility (same Gemma 3 arch, medical fine-tune shift only)
- Run the PT-vs-IT convergence test
- Exercise the full pipeline (activation extraction → CLT training → attribution graphs on eligibility prompts) end-to-end
- Iterate cheaply on tokenization, layer indexing, hyperparameters
- Pipeline bugs surface here at ~10–15% of 27B cost

**Stage 2 — 27B-IT for the headline result:**
- Once Stage 1 validates the pipeline and answers the IT-shift question, commit to `medgemma-27b-it`
- Deployed-clinical-AI framing is stronger at 27B (the variant Google positions for serious clinical use)
- The 27B-IT-only constraint becomes irrelevant once Stage 1 has decided IT is the right target

## Compute scaling reference (4B → 27B)

| Cost component | Scaling factor |
|---|---|
| Forward pass / activation extraction | ~6–7× |
| CLT trainable parameters | ~8–10× |
| Activation storage per token | ~4× |
| Model weights in bf16 | ~7× (~8GB → ~54GB) |
| End-to-end wall-clock per training run | ~5–10× |

Hardware step-change: 4B fits comfortably on a single 24GB GPU; 27B requires A100 80GB or multi-GPU sharding.

## Out of scope

- Direct PT-vs-IT comparison at 27B (impossible — no 27B PT variant exists)
- Attribution graph quality comparison between variants (separate experiment, downstream)
- Qualitative feature interpretability comparison (requires longer-trained CLTs)
- Alternative init strategies (holding init constant is the point of this experiment)

## Expected outcomes

- **Init compatibility check fails** → blocking issue; need an alternative init strategy (random init, partial layer mapping, or single-layer-only init with random cross-layer decoders) before any convergence test is meaningful. This outcome is itself important to learn early.
- **Convergence equivalent (PT ≈ IT training cost)** → choose IT on scientific grounds with no compute penalty; document the negative result as a methods contribution.
- **IT incurs measurable cost** → quantify it; the quantification itself is a contribution to the CLT transfer literature; choose variant by weighing scientific question against compute budget.
