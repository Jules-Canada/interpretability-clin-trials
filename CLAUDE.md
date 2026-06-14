# Attribution Graph Replication — Clinical Trials

## Project Goal

Replicate the Cross-Layer Transcoder (CLT) methodology from:
> *Circuit Tracing: Revealing Computational Graphs in Language Models*
> https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Then apply attribution graphs to clinical trial prompts to identify features involved
in reasoning about eligibility criteria, adverse events, and endpoint inference.

## Long-Term Goals

1. **Complete Pythia-410m proof-of-concept** — finish current training run, generate
   graphs for all 14 clinical prompts, label features, produce notebook readout.

2. **Scale to MedGemma + MIMIC-IV** — apply CLT to a medically-trained model on real clinical
   notes. Requires PhysioNet credentialing (long-lead item — start early).
   **Active path (revised 2026-06-09): use circuit-tracer (nnsight backend) + Gemma Scope 2
   per-layer transcoders for Gemma 3 4B. This retires `graphs/build.py` and from-scratch CLT
   training. Custom CLT is deferred, not killed. See Phase 2 Models section for full plan.**

3. **Find cross-trial generalisable features** — which features fire consistently across
   trial types vs. which are condition-specific? This is the core scientific question.

4. **Publish and communicate** — target ML4H or CHIL workshop; write at least one public
   technical post. Keep the GitHub repo as a clean public reference implementation.

These goals serve two purposes: contributing to mechanistic interpretability in the
clinical domain, and building a public portfolio for a career pivot into clinical AI.

---

## Models

### Phase 1 — Proof of concept (complete)
**EleutherAI/pythia-410m** (24 layers, d_model=1024, d_mlp=4096)
- Corpus: The Pile (monology/pile-uncopyrighted), 5M tokens
- CLT: n_features=2048, 50k steps, H100
- Key finding: syntactic features, not clinical — expected for a general model

### Phase 2 — Medical domain (active path: circuit-tracer + Gemma Scope 2 per-layer transcoders)

**Decision (revised 2026-06-09): Generate attribution graphs on MedGemma directly using
Anthropic's `circuit-tracer` (v0.3.1+, nnsight backend) + Gemma Scope 2's pretrained Gemma 3 4B
per-layer transcoders (PLTs). Do NOT train a CLT from scratch as the starting point. Spend effort
on a causally-validated clinical finding, not on transcoder engineering. Method = instrument;
finding = paper.**

**Why this path only became viable Dec 2025–Jan 2026:**
- `circuit-tracer` open-sourced 29 May 2025 — supported Gemma-2-2b / Llama-3.2-1b only.
- Gemma Scope 2 (Gemma 3 transcoders incl. 4B) released ~Dec 2025.
- `circuit-tracer` v0.3.1 (~Jan 2026) added an nnsight backend → works on *any* Transformers
  model, including Gemma 3 PLTs (PT & IT) for 270M–27B.
Building from scratch was reasonable before that window. It is not the right starting point now.

**Decode target resolved:** Gemma Scope 2 transcoders reconstruct the MLP **output** (`d_model`)
with affine skip connections — the canonical choice. This moots the old `d_mlp`-vs-`d_model`
debate (our old CLT used the non-standard `d_mlp` / pre-W_out target). Our old `graphs/build.py`
is *incompatible* with skip-transcoders (no skip term, wrong decode target) — the library
replaces it.

**PLT, not CLT, at 4B:** no pretrained 4B CLT exists (Gemma Scope 2 CLTs stop at 1B). Expect
longer attribution paths / more features than a CLT would give. Fine for a first paper; "a CLT
tightens this" is the future-work hook.

**Width / expansion (d_model=2560):** 16k = 6.4×, 64k = 25× (recommended), 256k = 100×. Our old
1024 = 0.4× (undercomplete — not credible). 4096 = 1.6× — still below the smallest width Google
shipped for this family.

**Off-distribution caveat:** Gemma Scope 2 transcoders were trained on base/IT Gemma 3, not
MedGemma. Running on the medical fine-tune degrades fidelity on exactly the medical-specific
features. The *size* of that gap is itself a publishable finding.

**Staged plan (cheap → expensive):**
- **Stage 0** — hours, ~free. Run `circuit-tracer` on base Gemma 3 4B-IT (hosted Neuronpedia is
  fine). Reproduce a known graph, push 2–3 clinical prompts through. Validates prompt format +
  target-token setup with zero custom code.
- **Stage 1** — a weekend, 1× H100, ~$10–30. Self-host `circuit-tracer` + nnsight, load
  MedGemma-4B, apply Gemma Scope 2 4B PLTs (16k or 64k fits alongside the ~8 GB model on one
  80 GB card). Run the prompt set. Deliverable: first MedGemma graphs + a measured
  off-distribution completeness gap. **Decision gate.**
- **Stage 2** — days, 1–2× H100, *only if* Stage 1 buries the medical computation in error nodes.
  Warm-start the Gemma Scope 2 transcoder, lightly domain-adapt on existing medical tokens
  (adaptation, not from-scratch). Rebuild graphs.

**On training our own CLT — deferred, not infeasible (verified 2026-06-09):**
"Can't" was wrong. We *can* train one; it is the expensive corner, not the next step:
- No warm-start at 4B → necessarily from-scratch (Gemma Scope 2's warm-start code is unreleased).
- Credible width from-scratch needs compute + tokens: ~26B params at ≥16k / `d_model` decode
  (8×H100-class node, cf. CLT-Forge for Llama-1B at expansion 32) and ~1–5B training tokens.
  We're at ~120M — the token gap is the real bind, forcing a corpus build first.
- CLT-on-Gemma-3 attribution is unproven in `circuit-tracer` (CLT support exists but not
  validated on Gemma 3 post-norms).
- It's a refinement, not a prerequisite — buys tighter graphs over the PLT, not graphs per se.
Revisit when (a) Stage 1/2 confirm a finding worth tightening, or (b) the CLT itself becomes the
contribution (a "MedScope" open-artifact play). Sequenced to a later stage, not killed.

**IT/PT axis (resolved):** the old IT-vs-PT dilemma was an artifact of training one home-grown
CLT. Gemma Scope 2 ships *both* PT and IT transcoder suites (IT finetuned on chat rollouts —
OpenAssistant/oasst1, LMSYS-Chat-1M). Rule: **IT model → IT transcoders → chat template,
consistently** (feeding raw text to IT transcoders reintroduces the Phase 6 OOD failure).
Sequence Gemma-3-4B-IT first (in-distribution, de-risks the method), then MedGemma-4B as point #2;
the delta between them is itself a publishable finding.

**google/medgemma-4b-pt** (Gemma 3 4B, 34 layers, d_model=2560, d_mlp=10240 GeGLU)
- Confirmed base model: `google/gemma-3-4b-pt` (verified via HuggingFace model tree)
- Gated model — requires HuggingFace terms acceptance before downloading
- Our old trained CLT: `checkpoints/medgemma-4b-1024/clt_inference.pt` (n_features=1024, L0~91)
  — deferred artifact, not used on the active path.

See `docs/ignis_approach_handoff.md` for the full handoff. The autograd `graphs/build.py`
pipeline (documented in "Attribution Graph Facts" below) is **superseded background** on this
path — kept for regression-test history and any future custom-CLT work.

---

## Stack

| Component | Library |
|---|---|
| Model loading & hooks | `transformer_lens` |
| Tensor ops | `torch`, `einops` |
| Experiment tracking | `wandb` |
| Type safety | `jaxtyping`, `beartype` |
| Testing | `pytest` |
| Visualization frontend | `anthropics/attribution-graphs-frontend` (cloned to `./frontend/`) |

Python 3.11. All deps managed via `pyproject.toml`.

---

## Repo Layout

```
.
├── CLAUDE.md                  # This file — always read first
├── pyproject.toml
├── frontend/                  # Cloned: github.com/anthropics/attribution-graphs-frontend
│
├── clt/
│   ├── __init__.py
│   ├── model.py               # CrossLayerTranscoder class
│   ├── train.py               # train_step() and train() — model-agnostic training loop
│   ├── loader.py              # ActivationLoader protocol, LiveActivationLoader, HDF5ActivationLoader
│   └── config.py              # CLTConfig (architecture) and TrainConfig (training) dataclasses
│
├── graphs/
│   ├── __init__.py
│   ├── build.py               # Attribution graph construction
│   ├── prune.py               # Graph pruning (top-K nodes/edges)
│   └── export.py              # Serialize to frontend JSON schema
│
├── interventions/
│   ├── __init__.py
│   └── patch.py               # Feature steering & patching experiments
│
├── prompts/
│   ├── eligibility.py         # Eligibility criteria prompts
│   ├── adverse_events.py      # AE attribution prompts
│   └── endpoints.py           # Endpoint inference prompts
│
├── scripts/
│   ├── extract_activations.py # Dump residual stream + MLP outputs to disk
│   ├── train_clt.py           # Entry point: train CLT
│   ├── run_graph.py           # Entry point: build + export attribution graph
│   └── run_intervention.py    # Entry point: patching experiment
│
├── viz/
│   ├── features.py            # Feature activation plots (heatmap, top-k bar chart, L0 curves)
│   └── graphs.py              # Attribution graph plots (node contributions, layer flow)
│
├── notebooks/
│   ├── 01_training_diagnostics.ipynb   # Loss curves, L0 sparsity — developer use
│   └── 02_feature_exploration.ipynb    # Prompt → features → labels — non-technical readout
│
└── tests/
    ├── test_clt_toy.py        # 2-layer toy model end-to-end pipeline test
    ├── test_attribution.py    # Graph construction unit tests
    └── test_export.py         # Frontend JSON schema validation
```

---

## Key Architectural Facts (read before touching clt/model.py)

From the paper (§ Building an Interpretable Replacement Model):

- The CLT has **L encoder matrices**, one per layer. Each reads from the residual stream
  `x_l` at its layer using a linear encoder + **JumpReLU** nonlinearity.
- Each feature at layer `l'` contributes to **all subsequent MLP outputs** via separate
  decoder matrices `W_dec[l' → l]` for each `l >= l'`.
- MLP output reconstruction at layer `l`:
  `y_hat_l = sum_{l'=1}^{l} W_dec[l'→l] @ a_l'`
- Training loss = MSE reconstruction (summed over layers) + L1 sparsity penalty.
- JumpReLU: zero below threshold `θ`, linear above. Threshold is a learned parameter.

---

## Attribution Graph Facts (read before touching graphs/)

- **Freeze** every multiplicative nonlinearity (attention pattern, LN/RMSNorm scale,
  GeGLU gate) before computing attributions. This makes the model linear in the
  residual stream and gives a well-defined chain rule.
- Nodes: CLT features, token embeddings, reconstruction errors, attention head
  cross-position contributions, output logits.
- Edges: linear effects between nodes. Feature pre-activation = sum of input edges.
- Pruning: keep top-K nodes/edges by contribution to target token logit. See §Appendix:
  Graph Pruning for the exact algorithm.
- Export format must match `anthropics/attribution-graphs-frontend` JSON schema.
  Check `frontend/README.md` for the schema spec before writing `graphs/export.py`.

> **Superseded background.** The autograd `graphs/build.py` pipeline (one frozen-nonlinearity
> backward pass for all edge gradients, architecture-agnostic post-norm handling, the
> `v·r_L == logit−b_U` sanity check, Pythia-70m completeness 0.8996) is retired on the active
> path — `circuit-tracer` replaces it. Full write-up moved to `docs/clt_era_archive.md`.
> Regression test still lives in `tests/test_attribution_completeness.py`.

---

## Development Rules

1. **Always start with the toy model test** (`tests/test_clt_toy.py`) before running on
   Pythia. The toy model is a 2-layer, d_model=64 transformer. If the pipeline doesn't
   pass on the toy model, don't scale up.

2. **Log everything to wandb** during CLT training: per-layer reconstruction MSE,
   sparsity (L0 norm of activations), total loss. Group runs by model size.

3. **No magic numbers.** All hyperparameters (L1 coefficient λ, JumpReLU threshold θ,
   pruning K) live in `clt/config.py` as fields of `CLTConfig`.

4. **Shapes in comments.** Every tensor that passes between modules should have its
   shape annotated in the line above, e.g.:
   ```python
   # (batch, seq, d_model)
   x = hook_point.hook(resid_pre)
   ```

5. **Do not modify the frontend submodule** except to update the JSON it reads.

6. **Every milestone gets a notebook.** Each item in the Current Status checklist must have
   a corresponding notebook in `notebooks/` that renders its outputs visually before the
   next milestone begins. Notebooks are the primary format for non-technical readouts —
   use plain-English section headers and captions, not just code and plots.

7. **Use `viz/` for all figures.** Never call matplotlib directly in scripts or notebooks
   without going through a function in `viz/features.py` or `viz/graphs.py`. This keeps
   figures consistent and reusable. Add to `viz/` as new plot types are needed.

8. **Verify completeness before calling graphs valid.** After building any graph, check that
   `completeness >= 0.5`. If it is below that threshold, the T matrix is missing paths
   (almost certainly attention Jacobians) and the graph does not answer "what caused the logit."
   Rendering in the frontend is not sufficient — completeness must be checked numerically.

---

## Clinical Trial Prompt Guidelines

Prompts live in `prompts/`. Each file exports a list of `TrialPrompt` dicts:

```python
TrialPrompt = TypedDict('TrialPrompt', {
    'id': str,               # e.g. "eligibility_nsclc_001"
    'text': str,             # the prompt text
    'target_token': str,     # token whose logit we trace (e.g. "eligible", "yes")
    'domain_tags': list[str] # e.g. ["oncology", "NSCLC", "eligibility"]
})
```

Start with **10–15 prompts per category**. Prioritize:
- Eligibility: NSCLC, breast cancer, renal cell carcinoma (oncology focus)
- Adverse events: hematologic toxicity, hepatotoxicity
- Endpoints: PFS, OS, ORR definitions

When labeling features found in attribution graphs, record labels in
`prompts/feature_labels.jsonl` (one JSON object per line, keyed by feature index).

---

## Pod Setup (RunPod — do this in order on every fresh instance)

### Active path — Stage 1 (circuit-tracer + Gemma Scope 2 transcoders)
No corpus, no HDF5, no CLT checkpoint — just pretrained transcoders.
```bash
git clone https://YOUR_TOKEN@github.com/Jules-Canada/interpretability-clin-trials.git ignis
cd ignis
export HF_TOKEN=YOUR_HF_TOKEN          # accept terms for gemma-3-4b-it, MedGemma, gemma-2-2b
bash scripts/setup_pod_circuit_tracer.sh
source .venv/bin/activate
python scripts/run_graphs_ct.py --probe          # confirm circuit-tracer API (instant, no GPU)
python scripts/run_graphs_ct.py --smoke          # gemma-2-2b known graph (API end-to-end)
python scripts/sweep_eligibility.py --model google/gemma-3-4b-it   # behavioral gate
python scripts/run_graphs_ct.py --model google/gemma-3-4b-it \
    --transcoders mwhanna/gemma-scope-2-4b-it    # eligibility graphs; repeat for MedGemma
# SCP back from Mac: frontend/graph_data/*.json and data/eligibility_sweep_*.json
```
H100 (80GB) — nnsight is less memory-efficient; A10 likely too small for 4B here.

### Deferred path — CLT training (from-scratch, not the active path)
```bash
# 1. Clone repo
git clone https://YOUR_TOKEN@github.com/Jules-Canada/interpretability-clin-trials.git
cd interpretability-clin-trials

# 2. Fix torchvision conflict (breaks transformer_lens import)
pip uninstall torchvision torchaudio -y

# 3. Set HF token then run setup script — it handles venv, torch, deps,
#    HF login, AND symlinks data/ + checkpoints/ to /workspace so the
#    20GB root disk doesn't fill up. Both setup_pod.sh and
#    setup_pod_medgemma.sh do this.
export HF_TOKEN=YOUR_HF_TOKEN
bash scripts/setup_pod_medgemma.sh   # or setup_pod.sh for non-MedGemma
source .venv/bin/activate            # also auto-added to .bashrc for new shells

# 4. SCP corpus + checkpoint from Mac (paths land on /workspace via symlinks
#    set up in step 3). Run from Mac:
#    scp -P <PORT> -i ~/.ssh/id_ed25519 \
#        ~/Desktop/protocol_corpus/ct_corpus/protocols.jsonl \
#        root@<IP>:interpretability-clin-trials/data/protocols.jsonl
#    scp -P <PORT> -i ~/.ssh/id_ed25519 \
#        ~/Desktop/ignis/checkpoints/medgemma-4b-1024/clt_inference.pt \
#        root@<IP>:interpretability-clin-trials/checkpoints/medgemma-4b-1024/
```

**Known gotchas:**
- `huggingface-cli login` and `hf login` are both broken on some images — `setup_pod*.sh` falls back to `hf auth login` automatically; if both fail, drop in: `python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"`
- `torchvision` must be uninstalled before importing `transformer_lens` or you get a segfault
- **Root disk is 20GB on RunPod, /workspace is huge** — never put corpus, HDF5, model checkpoints, or HF cache on root. The setup script handles this via symlinks; if you ever set up a pod manually, replicate those symlinks first or scp will fail at ~50% with "write remote: Failure"
- For graph generation only: A10 (24GB) is sufficient, no need for H100
- Repo dir is `interpretability-clin-trials` (the GitHub repo name), not `ignis` (the Mac local dir)

---

## Compute Notes

- CLT training on Pythia-410m: expect ~4–8 GPU-hours on a single A100 for a reasonable
  run. Use gradient checkpointing if VRAM is tight.
- Activation extraction: extract and cache residual streams + MLP outputs to disk first
  (`scripts/extract_activations.py`), then train CLT offline. Don't re-forward Pythia
  on every CLT training step.
- For local dev/testing, use `pythia-70m` (6 layers) — fast enough to iterate on CPU.

### Pre-termination checklist (run before killing any pod instance)

**If `run_pipeline.sh` completed fully, Step 4 already ran — skip to scp.**
**If the pipeline was interrupted, run `bash scripts/pre_terminate.sh` first.** It handles
checkpoint stripping, `collect_graph_features`, and `find_top_activations` in one shot,
then prints the exact scp commands to run. See `docs/pipeline_lessons.md` for why.

scp commands (run from your Mac):
```
INSTANCE=ubuntu@<ip>
scp "$INSTANCE:interpretability-clin-trials/frontend/graph_data/*.json" frontend/graph_data/
scp "$INSTANCE:interpretability-clin-trials/checkpoints/pythia-410m-2048/clt_inference.pt" checkpoints/pythia-410m-2048/
scp "$INSTANCE:interpretability-clin-trials/data/feature_activations.jsonl" data/
scp "$INSTANCE:interpretability-clin-trials/data/graph_features.json" data/
```

**Rule: run all 4 scp commands before terminating. The HDF5 stays on the instance and is
re-extracted next time. Use `--resid_only` for find_top_activations runs (~491GB on disk);
use full extraction (resid + mlp_post, ~2.5TB) only for CLT training — needs a larger disk.
`feature_labels.jsonl` is generated locally after scp (run `python scripts/label_features.py --resume`).**

---

## Current Status (as of 2026-05-18)

### What works

- **Autograd attribution pipeline validated** on MedGemma-4B-pt: 14 answer-telegraphing
  prompts achieved completeness 0.55–0.81, and 7 Track B medical-factual prompts achieved
  0.76–0.90. Phase 4 graphs + feature data moved to `deferred/phase4_telegraphing/`
  (scientifically weak — prompts telegraph the answer). Track B graphs need regeneration.
- **Autograd attribution pipeline** (`graphs/build.py`) — architecture-agnostic, handles
  Gemma 3 post-norms. Tested on Pythia-70m (0.90) and MedGemma-4B (0.55–0.81).
  Regression-locked in `tests/test_attribution_completeness.py`.
- **Contrastive readout** implemented in `graphs/build.py` — `contrastive=(pos_ids, neg_ids)`
  parameter for binary-decision targets. Untested at graph level (see "What doesn't work").
- **Contrastive screening** (`scripts/screen_contrastive.py`) — measures `logit(Yes) - logit(No)`
  per prompt and pair separation gap. Results in `data/contrastive_screen_*.json`.
  Weak statistical signal: easy prompts have larger mean gap than categorical (0.68 vs 0.12),
  but ranges overlap and 5/18 categorical pairs don't separate. Not a clean result.
- **PhysioNet credentialing** for MIMIC-IV (granted 2026-05-08).

### What doesn't work yet

- **Contrastive graphs** — CLT L0~91 projects ~0.55-logit constant No-bias onto the Yes-No
  direction. For single-token targets (logit ~20+) this is <3% noise; for contrastive
  targets (logit-diff ~0.2) it's >100% noise. Needs sparser CLT (L0~20-30).
- **IT model CLT** — trained on raw corpus activations, OOD at inference on short prompts.
  Completeness 0.20–0.47 (all below threshold). Artifacts moved to `deferred/phase6_it/`.
  IT path is not ruled out methodologically but needs format-matched training data.
- **Track B PT graphs regenerated (2026-05-30)** — 8 medical-factual cloze prompts,
  completeness 0.55–0.90. Graphs + labeled features in repo.
- **Feature labeling pipeline validated (2026-06-01)** — `find_top_activations` bug fixed
  (missing RMS normalization), 110 features labeled. Late-layer prompt (Sonnet, concept-level)
  produces much better labels than early-layer prompt (Haiku, token-level).
- **Graph layer distribution problem:** 92/110 features are L0 (structural/formatting).
  Only 15 features at L5+, which is where medical concepts live. Cross-layer circuits
  (the paper's key contribution) are not visible in current graphs. Root cause: L0~91
  sparsity floods the pruner budget with weakly-contributing L0 features.
- **Notebook 03** — needs rewrite to use current labeled graphs.

### Next steps

1. **Sparsity/corpus experiment — DONE (negative result).** Short CLT runs varying sparsity
   (L0~91 vs L0~20-30) and corpus (protocol JSONL vs PubMed) did not meaningfully shift the
   layer distribution; structural L0 features still dominate. Reinforced the decision to stop
   investing in from-scratch CLT training. See `scripts/compare_layer_distribution.py`.

2. **Stage 0 (hosted Neuronpedia) — DONE, provisional lead.** Confirmed Neuronpedia *does*
   do on-demand Gemma-3-4b-it graphs (public docs say otherwise). Hosted UI limits: no chat
   template, 64-token cap, mangles pasted special tokens → had to use raw-cloze prompts
   (`… Eligible? Answer:`). Behavioral sweep showed the model **ignores the stated age bounds**
   (effective eligible band ≈[16,97] vs stated [18,75]; approves 80/90/95; non-monotonic age-5
   reversal). UNTRUSTWORTHY — raw text on an IT model is OOD; must be reproduced under proper
   chat formatting before it's a result. The age-bound failure is a *candidate* clinical
   failure-mode finding (handoff candidate #1).

3. **Stage 1 (self-hosted circuit-tracer) — scripts ready, run on H100.** Plan executed
   2026-06-14: `scripts/setup_pod_circuit_tracer.sh` (env: circuit-tracer + nnsight, no CLT
   deps), `scripts/sweep_eligibility.py` (forward-pass behavioral GATE — re-test the age-bound
   finding under the real Gemma-3 chat template), `scripts/run_graphs_ct.py` (`--probe` /
   `--smoke` / real batch over `ELIGIBILITY_PAIRS`). Round 1 = **gemma-3-4b-it** (in-distribution,
   transcoders `mwhanna/gemma-scope-2-4b-it`) **+ MedGemma-4b** (off-distribution, same transcoder
   set — the gap is a finding). Target = predicted Yes/No (single-target; both Yes & No captured
   via top-K logits = free dual-logit read). True contrastive `logit(Yes)−logit(No)` deferred
   (no native circuit-tracer support — revisit after round 1). Replacement-score ≥0.5 is the
   completeness-gate analog (Dev Rule 8). **First behavior locked: eligibility yes/no.**
   `prompts/eligibility.py` holds the 5 matched contrastive pairs (`ELIGIBILITY_PAIRS`,
   decision-primitive ladder) + `to_chat` / `POS|NEG_TOKEN_IDS`. Pod run order: setup →
   `run_graphs_ct.py --probe` → `--smoke` → `sweep_eligibility.py` → `run_graphs_ct.py` (it,
   then MedGemma).

4. **Visual abstract for non-ML audience:** Create a figure explaining the pipeline
   (model → transcoder → features → attribution graph → labeled circuit) in plain language.
   Target audience: clinicians, clinical trialists, non-ML collaborators. Needed for
   paper submission and any public-facing communication.

5. **Notebook 03 + paper draft:** Write up results (graphs, labeled features, layer distribution
   finding, prompt engineering for feature labeling). Publishable as a methods/negative-result
   contribution; the MedGemma off-distribution gap (Stage 1) strengthens it.

### Completed milestones (detail in git history)

- Phase 1: Pythia-410m proof-of-concept (CLT + 14 graphs + feature labels + notebook 02)
- Phase 2-3: Autograd attribution rewrite (post-norm fix, architecture-agnostic)
- Phase 4: MedGemma-4B-pt batch run (14 clinical graphs, completeness validated)
- Phase 5: Contrastive methodology (screening script, Track B validation)
- Phase 6: IT model attempt (failed — OOD mismatch, moved to `deferred/phase6_it/`)

## Findings So Far

- **CLT feasibility re-verified (2026-06-09).** Training a credible CLT for Gemma 3 4B is not
  infeasible but is the expensive corner: no 4B warm-start (Gemma Scope 2 CLTs stop at 1B; their
  warm-start code is unreleased) → from-scratch only, ~26B params at ≥16k/`d_model` decode
  (8×H100-class), ~1–5B tokens (we have ~120M — the token gap is the real bind), and
  CLT-on-Gemma-3 attribution is unproven in `circuit-tracer`. Conclusion: PLT-first via
  circuit-tracer + Gemma Scope 2 is the right next step; custom CLT is a later refinement, not a
  prerequisite. See Phase 2 Models + `docs/ignis_approach_handoff.md`.
- **The circuit tracing paper used an instruction-tuned model (Claude 3 Sonnet), not a base model.**
  IT models are valid targets, but the CLT training data must match the inference format.
  Anthropic trained on the same distribution Claude processes at inference. Our IT attempt
  (Phase 6) failed because the CLT was trained on raw corpus activations, which are OOD for
  short QA prompts. PT models don't have this problem: raw text IS the inference format.
- torchvision/torchaudio conflict on pods: pins torch==2.5.1, incompatible with torch 2.11.0.
  Removed from setup_pod.sh; uninstall manually on existing instances.
- `frontend/` is tracked as a gitlink (embedded repo), not a proper submodule. Contents won't
  clone with the outer repo. Convert with `git submodule add` if needed.
- Frontend util.js rewrites all absolute paths to transformer-circuits.pub — added localhost
  check to skip rewrite for local development.
- Clinical trial protocol corpus: 49,002 docs from ClinicalTrials.gov, avg ~26k tokens/doc,
  JSONL with `full_text` field. Use --text_field full_text with extract_activations.py.
  2M tokens covers ~77 documents — sufficient for a proof-of-concept CLT run.
  Local copy on Julie's Mac: `/Users/juliecannon/Desktop/protocol_corpus/ct_corpus/protocols.jsonl`
  (6.0 GB). Not in the repo — scp this up to `data/protocols.jsonl` on every fresh pod.
- **Contrastive readout / dictionary sparsity (still relevant to the active path).** For
  binary-decision targets, trace `mean(logit[pos_ids]) − mean(logit[neg_ids])` so features
  pushing equally toward both answers cancel, decontaminating scaffold/format artifacts
  (standard for IOI / sparse feature circuits). Contrastive targets need *sparser*
  dictionaries than single-token targets: our L0~91 CLT projected a constant ~0.55-logit
  No-bias onto the Yes−No direction — <3% noise for single-token targets (logit ~20+, where
  completeness was 0.76–0.90) but >100% noise for contrastive targets (logit-diff ~0.2). This
  is the main argument for the lower-L0 Gemma Scope 2 PLTs (L0 ∈ {10,50,150}) on the active
  path. Implementation history (the retired `graphs/build.py contrastive=` param) is in
  `docs/clt_era_archive.md`.
- **Contrastive screening shows weak signal, not clean discrimination.**
  `medgemma-4b-pt` with the `Eligible?\nAnswer:` scaffold showed no dynamic range on
  absolute p_agg (easy controls ≈ categorical ≈ 0.3). On the contrastive metric
  (`logit(Yes) - logit(No)`), easy prompts have larger mean gap than categorical
  (0.677 vs 0.123), but the ranges overlap substantially — some easy gaps (0.15) are
  smaller than some categorical gaps (0.58), and 5/18 categorical pairs don't separate
  at all. The contrastive metric reveals more structure than absolute-p, but does not
  cleanly discriminate difficulty.
- **Gemma tokenizer splits many medical terms** (imatinib, trastuzumab, protamine, Hodgkin,
  troponin) into multi-token sequences. Single-token medical targets that work: metformin,
  pancreas, heart, embolism, plexus, gallbladder, K. Design prompts around these.
- **Feature labeling prompt matters enormously for late-layer features (2026-06-01).**
  Token-level prompt ("what token pattern?") + Haiku labeled L23F450 as "conjunction and
  preposition tokens in medical test enumeration." Concept-level prompt ("what concept does
  this encode?") + Sonnet labeled the same feature as "diabetes diagnostic biomarkers
  including HbA1c, fasting glucose, and oral glucose tolerance tests." The activation data
  was identical — only the prompt and model changed. Late-layer features encode abstract
  concepts that require semantic interpretation, not token-pattern matching.
  `label_features.py` now uses Haiku+token-prompt for L0-4, Sonnet+concept-prompt for L5+.
- **L0 features dominate graphs at L0~91 sparsity.** 92/110 graph features are Layer 0,
  almost all structural (TOC dots, section numbering, consent language, whitespace). Only
  15 features at L5+ across all 8 graphs. Cross-layer feature→feature circuits (the
  paper's central contribution) are invisible. Two possible causes: (1) L0~91 floods the
  pruner, (2) the protocol corpus is formatting-heavy. Experiment needed to disentangle.
- **Pod cost estimates must include overhead.** GPU-time estimates are routinely 2-3× too
  low. Real pod session = setup (~20min) + SCP (~15min each way for 12GB checkpoint) +
  compute + debugging disk/driver issues. Budget 2-3hrs wall time per session.
