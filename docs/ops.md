# Ops — pods, scripts, environment

Execution-time detail. Nothing here should be needed to decide *what* to work on; see
`docs/program/thesis.md` for that.

Full pre-Strata operational history: `deferred/docs/claude_md_ignis_era.md`.

---

## Pod setup — active path (Stage 1: circuit-tracer + Gemma Scope 2)

No corpus, no HDF5, no CLT checkpoint. Pretrained transcoders only.

```bash
git clone https://YOUR_TOKEN@github.com/Jules-Canada/interpretability-clin-trials.git ignis
cd ignis
export HF_TOKEN=YOUR_HF_TOKEN     # accept terms: gemma-3-4b-it, MedGemma, gemma-2-2b
bash scripts/setup_pod_circuit_tracer.sh
source .venv/bin/activate
python scripts/run_graphs_ct.py --probe    # confirm circuit-tracer API — instant, no GPU
python scripts/run_graphs_ct.py --smoke    # gemma-2-2b known graph — API end-to-end
python scripts/sweep_eligibility.py --model google/gemma-3-4b-it        # behavioral gate
python scripts/run_graphs_ct.py --model google/gemma-3-4b-it \
    --transcoders mwhanna/gemma-scope-2-4b-it                           # then MedGemma
```

Run the four in that order on a fresh pod. `--probe` before anything else: it prints the
installed `circuit-tracer` signatures, and the API has moved between versions.

H100 80GB. nnsight is not memory-efficient; A10 is too small for 4B on this path.

`setup_pod_circuit_tracer.sh` deliberately does **not** `pip install -e .` — Stage 1 needs
none of the CLT dependencies. It installs `circuit-tracer` from git plus `nnsight`,
`jinja2`, `hf_transfer`, and torch from the cu121 index.

**SCP back before terminating:** `frontend/graph_data/*.json`, `data/eligibility_sweep_*.json`,
`data/ecog_v0_results_*.json`.

---

## Pod setup — forward-pass only (sweeps, ECOG stimuli, probes)

`sweep_eligibility.py` and `run_ecog_stimuli.py` need **no circuit-tracer, no nnsight, no
transcoders** — they are plain `AutoModelForCausalLM` forward passes. Skip
`setup_pod_circuit_tracer.sh` (its slowest step is building circuit-tracer from git) unless
the same session also runs attribution:

```bash
git clone https://YOUR_TOKEN@github.com/Jules-Canada/interpretability-clin-trials.git ignis
cd ignis
export HF_HOME=/workspace/.cache/huggingface     # 20GB root disk fills otherwise
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # sm_70..sm_120
pip install transformers jinja2 hf_transfer      # jinja2: apply_chat_template
                                                 # hf_transfer: RunPod sets the env var
                                                 #   but omits the package -> downloads die
huggingface-cli login --token "$HF_TOKEN"

# Fresh-pod check: wrong wheel, wrong driver, or not the GPU you booked. Two
# seconds, and it fails here instead of after a 10-minute checkpoint download.
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print(torch.zeros(1).cuda() + 1)"

python scripts/run_ecog_stimuli.py --dry-run                          # no GPU, no download
python scripts/run_ecog_stimuli.py --model google/medgemma-4b-it
python scripts/run_ecog_stimuli.py --model google/gemma-3-4b-it       # matched baseline
```

**24GB is enough** (A10 / L4 / 4090 / 5090): ~8.6GB of bf16 weights plus a ~47MB logits
tensor. The "A10 is too small for 4B" note above is about nnsight holding all-layer
activations, and does not apply here. Take the H100 only if bundling attribution into the
same session.

The cu128 wheel above is the one to use on **any** modern card — it carries kernels for
sm_70 through sm_120, so it covers Turing through Blackwell and is not a 50-series special
case. Its real requirement is a recent NVIDIA driver, which fails loudly at CUDA init.
Older wheels are the trap: cu121 and cu126 predate `sm_120`, install without complaint on a
5090, and then die at the first forward pass with `CUDA capability sm_120 is not compatible
with the current PyTorch installation`. The CLT-path recipe above still pins cu121 because
`transformer_lens` constrains it; that pin is not a reason to use cu121 here.

Wall time is **download-bound, not compute-bound**: 36 forward passes over ~90-token prompts
run in seconds; pulling both 4B checkpoints is ~10 min and ~17–20GB of `/workspace` cache.

Run both models in the **same session**. `dtype` is derived from the device (bf16 on CUDA,
fp32 otherwise), and on the existing 36-prompt `categorical_screen` pair that shift moves
`p_target` by up to 0.016 — no decision crossed p=0.5, but split the two models across
machines and part of any MedGemma-vs-Gemma gap you report is precision, not tuning.

---

## Pod setup — deferred path (CLT training)

Only for from-scratch CLT work, which is not the active path.

```bash
pip uninstall torchvision torchaudio -y      # BEFORE anything imports transformer_lens
export HF_TOKEN=YOUR_HF_TOKEN
bash scripts/setup_pod_medgemma.sh           # or setup_pod.sh for non-MedGemma
source .venv/bin/activate
```

The setup script symlinks `data/` and `checkpoints/` to `/workspace`. Then scp the corpus
and checkpoint from the Mac:

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 \
    ~/Desktop/protocol_corpus/ct_corpus/protocols.jsonl \
    root@<IP>:interpretability-clin-trials/data/protocols.jsonl
```

### Pre-termination

If `run_pipeline.sh` completed, step 4 already ran — go straight to scp. If it was
interrupted, `bash scripts/pre_terminate.sh` does checkpoint stripping,
`collect_graph_features`, and `find_top_activations` in one pass, then prints the scp
commands. Rationale in `docs/pipeline_lessons.md`.

The HDF5 stays on the instance and is re-extracted next time. `--resid_only` for
`find_top_activations` (~491GB); full extraction (resid + mlp_post, ~2.5TB) only for CLT
training, and that needs a bigger disk.

---

## Gotchas

- **Root disk is 20GB on RunPod; `/workspace` is large.** Never put corpus, HDF5,
  checkpoints, or the HF cache on root. The setup scripts symlink around this. Set a pod up
  by hand without replicating the symlinks and scp dies at ~50% with `write remote: Failure`.
- **`torchvision` segfaults `transformer_lens`.** It pins an older torch. Both setup scripts
  skip it; uninstall manually on any pre-existing instance.
- **`huggingface-cli login` and `hf login` are broken on some images.** The setup scripts
  fall back to `hf auth login`. If both fail:
  `python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"`
- **Repo directory on the pod is `interpretability-clin-trials`** (the GitHub name), not
  `strata-repo` or `ignis` (local names).
- **Torch is unpinned.** `pyproject.toml` says `torch>=2.2`; both setup scripts install
  whatever is current on the cu121 index. Fine so far, but a driver/torch break will show up
  as a fresh-pod failure with no local reproduction.
- **Budget 2–3h wall time per pod session.** Setup ~20min, scp ~15min each way for a 12GB
  checkpoint, plus compute and debugging. GPU-time estimates alone run 2–3× low.

---

## Compute notes

- Graph generation only: A10 (24GB) suffices on the CLT path. Stage 1 needs the H100.
- Local dev/testing on the CLT path: `pythia-70m` (6 layers), fast on CPU.
- CLT training on Pythia-410m was ~4–8 GPU-hours on one A100.
