"""
prompts/categorical_prompts.py

Categorical-reasoning eligibility prompts. Each prompt pairs a clinical criterion
with a patient profile and asks Yes/No — the answer is NOT telegraphed in the
prompt. Paired pos/neg cases hold everything constant except the categorical
axis being probed (drug class, histology, anatomic site, surgical consequence,
allergy cross-reactivity, receptor status).

Replacement for the legacy `prompts/trial_prompts.json` set whose phrasing
encoded the answer (see auto-memory: project_prompt_design_flaw).
"""

from __future__ import annotations

from typing import TypedDict


class TrialPrompt(TypedDict):
    id: str
    text: str
    target_token: str
    domain_tags: list[str]


CATEGORICAL_PROMPTS: list[TrialPrompt] = [

    # ----------------------------------------------------------------------
    # Category 1: Drug class membership (steroids / glucocorticoids)
    # Probes: drug name → pharmacological class mapping
    # ----------------------------------------------------------------------

    # --- Pair 1: dexamethasone (synthetic glucocorticoid) ---
    {
        "id": "cat_steroid_001_pos",
        "text": (
            "Criterion: no current systemic corticosteroid use.\n"
            "Patient: aspirin 81mg, metoprolol 25mg, atorvastatin 40mg.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "drug_class", "steroid", "pharmacology"],
    },
    {
        "id": "cat_steroid_001_neg",
        "text": (
            "Criterion: no current systemic corticosteroid use.\n"
            "Patient: aspirin 81mg, metoprolol 25mg, dexamethasone 4mg daily.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "drug_class", "steroid", "pharmacology"],
    },

    # --- Pair 2: prednisone (different steroid, same class) ---
    {
        "id": "cat_steroid_002_pos",
        "text": (
            "Criterion: no chronic glucocorticoid therapy.\n"
            "Patient: metformin, lisinopril, atorvastatin.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "drug_class", "steroid", "pharmacology"],
    },
    {
        "id": "cat_steroid_002_neg",
        "text": (
            "Criterion: no chronic glucocorticoid therapy.\n"
            "Patient: metformin, lisinopril, prednisone 10mg daily.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "drug_class", "steroid", "pharmacology"],
    },

    # --- Pair 3: hydrocortisone vs NSAID distractor ---
    {
        "id": "cat_steroid_003_pos",
        "text": (
            "Criterion: no systemic steroid within 14 days.\n"
            "Patient: ibuprofen 600mg PRN, omeprazole 20mg daily.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "drug_class", "steroid", "NSAID_distractor"],
    },
    {
        "id": "cat_steroid_003_neg",
        "text": (
            "Criterion: no systemic steroid within 14 days.\n"
            "Patient: ibuprofen 600mg PRN, hydrocortisone 20mg daily.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "drug_class", "steroid", "NSAID_distractor"],
    },

    # ----------------------------------------------------------------------
    # Category 2: Histology / disease-equivalence terms
    # Probes: synonym recognition, taxonomy of disease names
    # ----------------------------------------------------------------------

    # --- Pair 1: NSCLC = adenocarcinoma; ≠ small cell ---
    {
        "id": "cat_histology_001_pos",
        "text": (
            "Criterion: non-small cell lung cancer only.\n"
            "Patient: lung adenocarcinoma, stage IV.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "histology", "NSCLC", "oncology"],
    },
    {
        "id": "cat_histology_001_neg",
        "text": (
            "Criterion: non-small cell lung cancer only.\n"
            "Patient: small cell lung cancer, extensive stage.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "histology", "NSCLC", "oncology"],
    },

    # --- Pair 2: HCC ≠ cholangiocarcinoma (both hepatic primary) ---
    {
        "id": "cat_histology_002_pos",
        "text": (
            "Criterion: hepatocellular carcinoma required.\n"
            "Patient: HCC, BCLC stage B, Child-Pugh A.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "histology", "HCC", "oncology", "hepatic"],
    },
    {
        "id": "cat_histology_002_neg",
        "text": (
            "Criterion: hepatocellular carcinoma required.\n"
            "Patient: intrahepatic cholangiocarcinoma, Child-Pugh A.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "histology", "HCC", "oncology", "hepatic"],
    },

    # --- Pair 3: TNBC requires ER/PR/HER2 all negative ---
    {
        "id": "cat_histology_003_pos",
        "text": (
            "Criterion: triple-negative breast cancer required.\n"
            "Patient: invasive ductal carcinoma, ER 0%, PR 0%, HER2 IHC 0.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "histology", "TNBC", "oncology"],
    },
    {
        "id": "cat_histology_003_neg",
        "text": (
            "Criterion: triple-negative breast cancer required.\n"
            "Patient: invasive ductal carcinoma, ER 95%, PR 80%, HER2 IHC 0.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "histology", "TNBC", "oncology"],
    },

    # ----------------------------------------------------------------------
    # Category 3: Anatomic site classification (visceral / non-visceral / CNS)
    # Probes: organ → anatomic category taxonomy
    # ----------------------------------------------------------------------

    # --- Pair 1: visceral disease required (liver+lung = yes, bone = no) ---
    {
        "id": "cat_site_001_pos",
        "text": (
            "Criterion: visceral metastatic disease required.\n"
            "Patient: liver and lung metastases on imaging.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "anatomic_site", "visceral", "metastatic"],
    },
    {
        "id": "cat_site_001_neg",
        "text": (
            "Criterion: visceral metastatic disease required.\n"
            "Patient: bone-only metastases on imaging.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "anatomic_site", "visceral", "metastatic"],
    },

    # --- Pair 2: bone-only cohort (inverse axis) ---
    {
        "id": "cat_site_002_pos",
        "text": (
            "Criterion: bone-only metastatic disease.\n"
            "Patient: multiple osseous metastases, no visceral involvement.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "anatomic_site", "bone_only", "metastatic"],
    },
    {
        "id": "cat_site_002_neg",
        "text": (
            "Criterion: bone-only metastatic disease.\n"
            "Patient: bone and hepatic metastases on staging scans.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "anatomic_site", "bone_only", "metastatic"],
    },

    # --- Pair 3: CNS metastases exclusion ---
    {
        "id": "cat_site_003_pos",
        "text": (
            "Criterion: no active CNS metastases.\n"
            "Patient: pulmonary and hepatic metastases, brain MRI negative.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "anatomic_site", "CNS", "brain_mets"],
    },
    {
        "id": "cat_site_003_neg",
        "text": (
            "Criterion: no active CNS metastases.\n"
            "Patient: pulmonary metastases and untreated cerebellar lesion.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "anatomic_site", "CNS", "brain_mets"],
    },

    # ----------------------------------------------------------------------
    # Category 4: Surgical history implications
    # Probes: surgical fact → physiological consequence inference
    # ----------------------------------------------------------------------

    # --- Pair 1: splenectomy → asplenia ---
    {
        "id": "cat_surgical_001_pos",
        "text": (
            "Criterion: functional spleen required.\n"
            "Patient: native spleen, no prior abdominal surgery.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "surgical_history", "spleen", "physiology"],
    },
    {
        "id": "cat_surgical_001_neg",
        "text": (
            "Criterion: functional spleen required.\n"
            "Patient: s/p splenectomy following MVA, 2018.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "surgical_history", "spleen", "physiology"],
    },

    # --- Pair 2: cholecystectomy → no gallbladder ---
    {
        "id": "cat_surgical_002_pos",
        "text": (
            "Criterion: native gallbladder required for imaging.\n"
            "Patient: no prior abdominal surgery.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "surgical_history", "gallbladder", "imaging"],
    },
    {
        "id": "cat_surgical_002_neg",
        "text": (
            "Criterion: native gallbladder required for imaging.\n"
            "Patient: s/p laparoscopic cholecystectomy, 2015.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "surgical_history", "gallbladder", "imaging"],
    },

    # --- Pair 3: gastrectomy → impaired oral absorption ---
    {
        "id": "cat_surgical_003_pos",
        "text": (
            "Criterion: ability to absorb oral medications.\n"
            "Patient: no prior gastrointestinal surgery.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "surgical_history", "GI", "absorption"],
    },
    {
        "id": "cat_surgical_003_neg",
        "text": (
            "Criterion: ability to absorb oral medications.\n"
            "Patient: s/p total gastrectomy with Roux-en-Y reconstruction.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "surgical_history", "GI", "absorption"],
    },

    # ----------------------------------------------------------------------
    # Category 5: Allergy cross-reactivity (drug class hierarchies)
    # Probes: allergen → drug class hierarchy
    # ----------------------------------------------------------------------

    # --- Pair 1: β-lactam allergy ---
    {
        "id": "cat_allergy_001_pos",
        "text": (
            "Criterion: no β-lactam antibiotic allergy.\n"
            "Patient: documented sulfa rash, 2010.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "allergy", "beta_lactam", "drug_class"],
    },
    {
        "id": "cat_allergy_001_neg",
        "text": (
            "Criterion: no β-lactam antibiotic allergy.\n"
            "Patient: documented penicillin anaphylaxis, 2015.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "allergy", "beta_lactam", "drug_class"],
    },

    # --- Pair 2: sulfonamide allergy (amoxicillin is distractor) ---
    {
        "id": "cat_allergy_002_pos",
        "text": (
            "Criterion: no sulfonamide allergy.\n"
            "Patient: documented amoxicillin rash, 2018.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "allergy", "sulfonamide", "drug_class"],
    },
    {
        "id": "cat_allergy_002_neg",
        "text": (
            "Criterion: no sulfonamide allergy.\n"
            "Patient: Stevens-Johnson syndrome from TMP-SMX, 2012.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "allergy", "sulfonamide", "drug_class"],
    },

    # --- Pair 3: NSAID hypersensitivity (penicillin distractor) ---
    {
        "id": "cat_allergy_003_pos",
        "text": (
            "Criterion: no NSAID hypersensitivity.\n"
            "Patient: penicillin allergy (urticaria), no other drug allergies.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "allergy", "NSAID", "drug_class"],
    },
    {
        "id": "cat_allergy_003_neg",
        "text": (
            "Criterion: no NSAID hypersensitivity.\n"
            "Patient: aspirin-induced bronchospasm, Samter's triad.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "allergy", "NSAID", "drug_class"],
    },

    # ----------------------------------------------------------------------
    # Category 6: Receptor / biomarker status interpretation
    # Probes: composite categorical reasoning (e.g., HER2 status = f(IHC, FISH))
    # ----------------------------------------------------------------------

    # --- Pair 1: HER2 IHC 2+ + FISH conditional ---
    {
        "id": "cat_receptor_001_pos",
        "text": (
            "Criterion: HER2-negative breast cancer required.\n"
            "Patient: HER2 IHC 2+, FISH not amplified (HER2/CEP17 ratio 1.4).\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "receptor", "HER2", "conditional", "biomarker"],
    },
    {
        "id": "cat_receptor_001_neg",
        "text": (
            "Criterion: HER2-negative breast cancer required.\n"
            "Patient: HER2 IHC 2+, FISH amplified (HER2/CEP17 ratio 4.2).\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "receptor", "HER2", "conditional", "biomarker"],
    },

    # --- Pair 2: clean HER2 endpoints (IHC 0 vs IHC 3+) ---
    {
        "id": "cat_receptor_002_pos",
        "text": (
            "Criterion: HER2-negative breast cancer required.\n"
            "Patient: HER2 IHC 0 by ASCO/CAP guidelines.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "receptor", "HER2", "biomarker"],
    },
    {
        "id": "cat_receptor_002_neg",
        "text": (
            "Criterion: HER2-negative breast cancer required.\n"
            "Patient: HER2 IHC 3+ (uniform circumferential membrane staining).\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "receptor", "HER2", "biomarker"],
    },

    # --- Pair 3: HR status (ER ≥ 1% threshold is the categorical cutoff) ---
    {
        "id": "cat_receptor_003_pos",
        "text": (
            "Criterion: hormone receptor positive disease required.\n"
            "Patient: ER 95%, PR 80%, HER2 IHC 1+.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "Yes",
        "domain_tags": ["categorical", "receptor", "HR", "ER", "biomarker"],
    },
    {
        "id": "cat_receptor_003_neg",
        "text": (
            "Criterion: hormone receptor positive disease required.\n"
            "Patient: ER < 1%, PR 0%, HER2 IHC 0.\n"
            "Eligible (answer Yes or No):"
        ),
        "target_token": "No",
        "domain_tags": ["categorical", "receptor", "HR", "ER", "biomarker"],
    },
]
