# Prompt re-screen checklist

Goal: get **true** `p_agg` (full softmax, surface-form aggregated) for the
categorical set and the easy control set, replacing the top-3 lower-bound
estimate. See auto-memory `project_prompt_design_flaw` for why.

Screening needs only `google/medgemma-4b-pt` + `transformers`/`torch`. No CLT,
no HDF5, no `/workspace`. ~8 GB in float16.

## Option 1 — Local Mac (preferred: zero pod cost)

HF is already logged in as JulesCan with MedGemma terms accepted (auto-memory
`hf_access`). Requires ≥16 GB unified memory (4B fp16 ≈ 8 GB) and ~8 GB free
disk for the model download.

```bash
cd ~/Desktop/ignis
# float32 on CPU is the safest numerically; MPS fp16 also works if memory allows.
python scripts/screen_prompts.py \
  --prompts prompts/categorical_prompts.py --var_name CATEGORICAL_PROMPTS \
  --model_name google/medgemma-4b-pt --device auto --dtype float16 \
  --min_prob 0.2 --output data/categorical_screen_v2.json

python scripts/screen_prompts.py \
  --prompts prompts/categorical_prompts.py --var_name EASY_INCLUSION_PROMPTS \
  --model_name google/medgemma-4b-pt --device auto --dtype float16 \
  --min_prob 0.2 --output data/easy_screen.json
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

python scripts/screen_prompts.py \
  --prompts prompts/categorical_prompts.py --var_name CATEGORICAL_PROMPTS \
  --model_name google/medgemma-4b-pt --dtype float16 \
  --min_prob 0.2 --output data/categorical_screen_v2.json

python scripts/screen_prompts.py \
  --prompts prompts/categorical_prompts.py --var_name EASY_INCLUSION_PROMPTS \
  --model_name google/medgemma-4b-pt --dtype float16 \
  --min_prob 0.2 --output data/easy_screen.json
```

scp both JSONs back, then terminate:

```bash
INSTANCE=ubuntu@<ip>
scp "$INSTANCE:interpretability-clin-trials/data/categorical_screen_v2.json" data/
scp "$INSTANCE:interpretability-clin-trials/data/easy_screen.json" data/
```

## Reading the result

Do **not** just count PASS/FAIL. The scientific read is the **spread**:

- `easy_screen.json` `p_agg` is the dynamic-range ceiling on this scaffold.
  - If easy ≈ 0.3 and categorical ≈ 0.25 → scaffold/base-model is the ceiling;
    the categorical set is fine and a few-shot/scaffold change is needed to lift
    *all* margins (not a per-prompt difficulty problem).
  - If easy ≈ 0.7+ and categorical ≈ 0.3 → there is real headroom and the
    categorical numbers reflect genuine task difficulty, not a measurement floor.
- Keep `data/categorical_screen.json` (the original top-3 record) for the
  before/after comparison; write the re-run to `*_v2.json`.
- Compare per-pair `p_agg` to the lower-bound table from the 2026-05-16 session;
  the surgical cluster (`cat_surgical_*`) is the one to watch — it had no
  yes-variant in top-3, so its true `p_agg` is currently unknown.
