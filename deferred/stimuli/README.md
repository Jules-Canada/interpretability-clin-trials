# Deferred stimulus material

## ecog_v0_distant_DEFERRED.csv

20 vignettes written 2026-08-05 as a "held-out" set, deferred the same day.

Two reasons:

1. **"Held-out" was the wrong term.** Nothing is trained here, so there is no train/test
   split in the usual sense. It was a second experimental condition, not a holdout.
2. **It was not actually distant.** Measured against the main set: paired rows shared 4x
   more vocabulary than unrelated rows (jaccard 0.123 vs 0.028), and the discriminating
   stems survived the rewrite — "work" in 44% of main rows and 35% of these, "self" in 28%
   and 30%. What changed was register (`nursing note`, `clinic letter`, `in her own words`),
   not the words the grade actually turns on. Same author, same source, so the manipulation
   was framing rather than lexis.

The rows are good vignettes. They are kept because they become usable under either of:

- a **proscribed-word list** for the distant condition, so a grade-2 vignette cannot use
  *work*, *job*, *shift*, or *employment* and must convey lost work capacity another way.
  Enforceable in the instrument, same shape as the existing `leaks_vocab` check.
- **real clinical text** from a different source. Note that MIMIC is not a clean fix: it is
  ICU rather than ambulatory oncology, it carries a DUA that blocks committing the text
  here, and its presence in pretraining corpora is unknown.

Paraphrase generalisation is still tested without this file: all 12 paraphrase sets are
self-contained pairs inside `specs/stimuli/ecog_v0.csv`.
