# ignis — L0 structural-feature diagnosis + fix plan

## Context
MedGemma-4B CLT (`checkpoints/medgemma-4b-1024/clt_inference.pt`). Attribution graphs on 8 clinical
prompts surface 110 features; 92 are layer 0. Investigating why the graphs are dominated by structural
features instead of clinical reasoning.

## Diagnosis (from feature labels, not assumed)
The L0 features are **corpus/data-quality artifacts**, not pure CLT depth-collapse:

- **~40 of 92 L0 features are document scaffolding**: TOC dot-leaders / page-number fillers
  (F179, F405, F646, F651, F686, F708, F741), section numbering & headers
  (F100, F120, F270, F317, F569, F610, F824, F867), newline/section-break tokens duplicated 4×
  (F172, F363, F424, F806), separators/whitespace (F377, F395, F471, F700, F790), form-field
  underscores (F958). Heavy redundancy — dictionary capacity wasted on near-duplicate layout detectors.
- **2 extraction-corruption features**: F446 (garbled/cipher-like), F977 (dollar-sign in malformed text).
  Smoking gun that `ct_pdf_extractor.py` leaks junk into training.
- **~10 generic syntactic/function-word features** — low value, not corpus-specific.
- **~40 shallow medical-vocab features** (breast, tracheal, physiologic, dosage, drug morphemes).
  These legitimately belong at L0 — NOT the problem.

Key points:
- **"92/110 structural" overstates it.** The real issue is ~40 formatting + corruption features
  crowding clinical content out of the graphs.
- **Late layers are healthy.** L2–L25 contain genuine conceptual clinical features (subclavian access,
  squamous cell carcinoma, signaling pathways, HbA1c, cardiac dysfunction, targeted-therapy combos,
  adverse events). So the CLT learned good structure — it's being crowded out, not absent.
- Primary driver = **corpus formatting + extraction quality**. Secondary = CLT early-layer attribution
  bias. Fix corpus first; it's cheaper and less confounded than swapping to PubMed.

## Tasks (ordered; cheap → expensive)

1. **Inspect existing checkpoint (zero compute).** Dump per-layer L0 / alive-feature counts from
   `clt_inference.pt`. Confirms whether late layers are sparse/dead vs healthy. Expect healthy per above.

2. **Fix extraction corruption in `ct_pdf_extractor.py`.** Trace source of garbled/cipher tokens and
   stray `$` / malformed sequences (cf. F446, F977). Add validation/filtering. Required for the public
   HF corpus release regardless.

3. **Add corpus-cleaning preprocessing** (new step before activation extraction):
   - Drop TOC pages and page-number footers/headers.
   - Strip section-numbering scaffolding and form-field underscore runs.
   - Collapse repeated separators / blank lines / dot-leader runs.
   - Keep narrative protocol text (eligibility, methods, endpoints).

4. **Audit the 8 prompts** (`prompts/eligibility.py`, `prompts/clinical.py`, `prompts/categorical.py`)
   for embedded protocol formatting (criteria lists, headers, newlines). If present, rewrite as clean
   clinical questions. Zero compute; stops structure tokens dominating attribution directly.

5. **Retrain CLT on cleaned corpus, same config** (1e-2, 5k steps to start). This is a clean
   raw-vs-cleaned A/B isolating *formatting* — do NOT switch to PubMed (confounds formatting w/ domain).

6. **Re-run** `collect_graph_features` + `find_top_activations` on the same 8 prompts. Metric:
   fraction of graph features that are formatting/corruption, before vs after. Target: sharp drop,
   clinical-content features populate the graphs.

7. **(Architectural control)** Train a jointly-trained PLT (loss summed across layers) on the same
   cached activations; compare CLT-vs-PLT layer distribution on the 8 prompts. Secondary effect, but
   it's the publishable methods control.

## Acceptance check
- L0 formatting/corruption feature fraction drops materially after cleaning + retrain.
- Attribution graphs for clinical prompts are populated by clinical-content features, not layout.

## Caveat
"Cleaning pulls formatting features out of the graphs" is the **hypothesis**, not a result. Don't write
it up as settled until step 6 shows the drop.
