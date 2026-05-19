# Prompt re-screen checklist

Goal: get **true** `p_agg` (full softmax, surface-form aggregated) for the
categorical set and the easy control set, replacing the top-3 lower-bound
estimate. See auto-memory `project_prompt_design_flaw` for why.

> **Precision matters for the diffuseness question.** The 2026-05-16 v2 screen
> was run in bfloat16 on MPS. 18/36 categorical prompts had distinct tokens at
> bit-identical probability — bf16 logit quantization, which mechanically
> flattens the softmax. **Any "no dynamic range" conclusion must come from a
> `--dtype float32` run.** bf16/fp16 is fine for the single-token vs aggregation
> question but NOT for reading the shape of the distribution. `screen_prompts.py`
> now flags this (`quant_tie_count` in the JSON, loud stderr banner); a clean
> float32 run should report `quant_tie_count: 0`. float32 weights need ~16 GB —
> use a pod, not the 17 GB Mac.

Screening needs only `google/medgemma-4b-pt` + `transformers`/`torch`. No CLT,
no HDF5, no `/workspace`. ~8 GB in float16.

## Option 1 — Local Mac (bf16 only — NOT valid for the diffuseness question)

Use this only for the single-token-vs-aggregation check. The 17 GB Mac cannot
hold float32 weights (~16 GB), so a local run is bf16/fp16 and will trip the
quantization-tie flag — do not read dynamic range from it. For the diffuseness
control use Option 2 (pod, float32).

HF is already logged in as JulesCan with MedGemma terms accepted (auto-memory
`hf_access`). Requires ≥16 GB unified memory (4B fp16 ≈ 8 GB) and ~8 GB free
disk for the model download.

```bash
cd ~/Desktop/ignis
# float32 on CPU is the safest numerically; MPS fp16 also works if memory allows.
python scripts/screen_prompts.py --prompts prompts/categorical_prompts.py --var_name CATEGORICAL_PROMPTS --model_name google/medgemma-4b-pt --device auto --dtype float16 --min_prob 0.2 --output data/categorical_screen_v2.json

python scripts/screen_prompts.py --prompts prompts/categorical_prompts.py --var_name EASY_INCLUSION_PROMPTS --model_name google/medgemma-4b-pt --device auto --dtype float16 --min_prob 0.2 --output data/easy_screen.json
```

If MPS gives NaNs, re-run with `--device cpu --dtype float32` (slower but 46
prompts is only a few minutes).

## Option 2 — RunPod (fallback if Mac memory is short)

Any 16 GB+ GPU (A10/24 GB is plenty; no H100 needed).

```bash
git clone https://YOUR_TOKEN@github.com/Jules-Canada/interpretability-clin-trials.git
cd interpretability-clin-trials
pip uninstall torchvision torchaudio -y          # avoids the known segfault
export HF_TOKEN=YOUR_HF_TOKEN
bash scripts/setup_pod_medgemma.sh               # handles gated-model HF login
# (HDF5/checkpoint scp steps in CLAUDE.md are NOT needed for screening)

# float32 = the precision control. New filenames so the bf16 v2 record is kept
# for the before/after comparison. Expect quant_tie_count: 0 in both JSONs.
python scripts/screen_prompts.py --prompts prompts/categorical_prompts.py --var_name CATEGORICAL_PROMPTS --model_name google/medgemma-4b-pt --dtype float32 --min_prob 0.2 --output data/categorical_screen_fp32.json

python scripts/screen_prompts.py --prompts prompts/categorical_prompts.py --var_name EASY_INCLUSION_PROMPTS --model_name google/medgemma-4b-pt --dtype float32 --min_prob 0.2 --output data/easy_screen_fp32.json
```

scp both JSONs back, then terminate:

```bash
INSTANCE=ubuntu@<ip>
scp "$INSTANCE:interpretability-clin-trials/data/categorical_screen_fp32.json" data/
scp "$INSTANCE:interpretability-clin-trials/data/easy_screen_fp32.json" data/
```

## Reading the result

**First check `quant_tie_count` in both JSONs. It must be 0.** If it is nonzero
the run is still quantization-flattened and the spread below is not
interpretable — re-run in float32 on a bigger box.

Do **not** just count PASS/FAIL. The scientific read is the **spread**:

- `easy_screen_fp32.json` `p_agg` is the dynamic-range ceiling on this scaffold.
  - If easy ≈ 0.3 and categorical ≈ 0.25 → scaffold/base-model is the ceiling;
    the categorical set is fine and a few-shot/scaffold change is needed to lift
    *all* margins (not a per-prompt difficulty problem).
  - If easy ≈ 0.7+ and categorical ≈ 0.3 → there is real headroom and the
    categorical numbers reflect genuine task difficulty, not a measurement floor.
- Keep `data/categorical_screen.json` (original top-3) and
  `data/categorical_screen_v2.json` (bf16, quant-flattened) as the before
  records; write the precision-clean re-run to `*_fp32.json`.
- Compare per-pair `p_agg` to the lower-bound table from the 2026-05-16 session;
  the surgical cluster (`cat_surgical_*`) is the one to watch — it had no
  yes-variant in top-3, so its true `p_agg` is currently unknown.
