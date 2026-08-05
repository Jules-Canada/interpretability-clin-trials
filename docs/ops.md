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

**SCP back before terminating:** `frontend/graph_data/*.json`, `data/eligibility_sweep_*.json`.

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
