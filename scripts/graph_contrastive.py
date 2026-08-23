"""Track B contrastive attribution graphs at 27B.

Same vignette under two adjacent criteria straddling its flip point:
`ECOG <= g-1` (the model answers No) against `ECOG <= g` (Yes). The vignette
text is byte-identical across the pair; one digit in the criterion differs and
the answer inverts. Whatever differs between the two graphs is the threshold
comparison — the Application computation itself, isolated.

Pairs are chosen from vignettes where the 27B flips exactly at its own grade,
so each graph explains a behaviour measured in the same session.
"""
import argparse, csv, json, sys, time, traceback
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_graphs_ct import run_one, _import_attr, _import_export  # noqa: E402

# vignette -> own grade g; graphs are drawn at <= g-1 and <= g
PAIRS = [("E005", 1), ("E010", 2), ("E011", 1), ("E023", 1), ("E024", 1), ("E036", 4)]

# The sweep's prompt, verbatim — graphs must explain the behaviour that was measured.
ELIG = ("You are screening a patient for a clinical trial.\n"
        "Inclusion: {crit}.\n"
        "Patient: {patient}\n"
        "Is the patient eligible for the trial? Answer Yes or No.")

# Prompt-order control. In the sweep's ordering the criterion sits ~40 tokens
# upstream of the readout, and attribution measures influence *on the answer
# logit* — so a criterion feature's effect is heavily mediated and it barely gets
# selected into the graph (8 of 11,170 nodes, 0.07%, on the 2026-08-23 run).
# This variant moves the criterion adjacent to the question, changing only the
# mediation distance. Everything else, including the vignette text, is identical.
# NOT the sweep's prompt: behaviour under it is unmeasured, so read it as a
# methods control on node selection, not as a pillar-2 result.
ELIG_CRIT_LAST = ("You are screening a patient for a clinical trial.\n"
                  "Patient: {patient}\n"
                  "Inclusion: {crit}.\n"
                  "Is the patient eligible for the trial? Answer Yes or No.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--transcoders", required=True)
    ap.add_argument("--stimuli", default="specs/stimuli/ecog_sweep_v0.csv")
    ap.add_argument("--output-dir", default="data/graphs_27b_contrastive")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-feature-nodes", type=int, default=2000)
    ap.add_argument("--max-n-logits", type=int, default=10)
    ap.add_argument("--node-threshold", type=float, default=0.8)
    ap.add_argument("--edge-threshold", type=float, default=0.98)
    ap.add_argument("--offload", default=None)
    ap.add_argument("--criterion-last", action="store_true",
                    help="prompt-order control: put the criterion next to the question "
                         "instead of ~40 tokens upstream (see ELIG_CRIT_LAST)")
    a = ap.parse_args()
    template = ELIG_CRIT_LAST if a.criterion_last else ELIG
    print("prompt order: %s" % ("criterion-last (CONTROL)" if a.criterion_last
                                else "sweep verbatim"), flush=True)

    rows = {r["id"]: r for r in csv.DictReader(open(a.stimuli, newline="", encoding="utf-8-sig"))}
    ReplacementModel, attribute = _import_attr()
    create_graph_files = _import_export()

    print("loading %s\n  transcoders=%s" % (a.model, a.transcoders), flush=True)
    t0 = time.time()
    model = ReplacementModel.from_pretrained(
        a.model, a.transcoders, backend="nnsight", dtype=torch.bfloat16)
    print("  loaded in %.0fs" % (time.time() - t0), flush=True)
    print("  after load: %.1f GB" % (torch.cuda.memory_allocated() / 1024**3), flush=True)

    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    todo = [(v, g, t) for v, g in PAIRS for t in (g - 1, g)]
    for i, (vig, g, t) in enumerate(todo):
        rid = "%s_LE%d" % (vig, t)
        row = rows.get(rid)
        if row is None:
            print("  MISSING %s" % rid, flush=True)
            results.append({"id": rid, "status": "missing"})
            continue
        slug = "contrast_%s_g%d_le%d" % (vig, g, t)
        body = template.format(crit=row["eligibility_criterion"], patient=row["vignette_text"])
        chat = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
        ids = model.tokenizer(chat, add_special_tokens=False)["input_ids"]
        print("[%d/%d] %s  (own grade %d, criterion <= %d, %s)"
              % (i + 1, len(todo), slug, g, t, row["lexical_distance"]), flush=True)
        try:
            r = run_one(model, attribute, create_graph_files, slug=slug,
                        attr_input=ids, display="%s tokens=%d" % (rid, len(ids)),
                        out_dir=a.output_dir, max_n_logits=a.max_n_logits,
                        node_threshold=a.node_threshold, edge_threshold=a.edge_threshold,
                        batch_size=a.batch_size, offload=a.offload,
                        max_feature_nodes=a.max_feature_nodes)
            r.update(vignette=vig, own_grade=g, threshold=t,
                     lexical_distance=row["lexical_distance"],
                     expected="Yes" if t >= g else "No",
                     peak_gb=torch.cuda.max_memory_allocated() / 1024**3)
            results.append(r)
        except Exception as e:
            traceback.print_exc()
            results.append({"id": slug, "status": "failed", "error": str(e)[:300]})
        print("  peak so far %.1f GB" % (torch.cuda.max_memory_allocated() / 1024**3), flush=True)
        print(flush=True)

    out = Path(a.output_dir) / "contrastive_summary.json"
    out.write_text(json.dumps({"model": a.model, "transcoders": a.transcoders,
                               "prompt_order": "criterion_last" if a.criterion_last else "sweep",
                               "results": results}, indent=1))
    ok = sum(r.get("status") == "ok" for r in results)
    print("=== %d/%d graphs ok -> %s" % (ok, len(results), out))


if __name__ == "__main__":
    main()
