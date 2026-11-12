# Grounding eval helpers

Wrappers and analysis scripts around `scripts/eval_grounding.py` (reference-free grounding:
each extracted value is checked against the paper's own text, no gold labels needed).
Run everything from anywhere; scripts `cd` to the repo root. Outputs go to `data/eval/` (gitignored).

API keys: loaded from `$METHY_ENV_FILE` if set, otherwise `<repo>/.env`. Example for a side worktree:
`METHY_ENV_FILE=/path/to/main/checkout/.env scripts/grounding_eval/run_paired.sh`

| Script | Purpose |
|---|---|
| `run_grounding.sh` | Env-loading wrapper; passes all args to `eval_grounding.py` |
| `run_paired.sh` → `paired_review_eval.py` | **Controlled test.** Extract once per paper, grade the draft and the reviewed version (spies on `review_extraction`; repo code untouched). Writes `data/eval/clean/q*.json` |
| `analyze_paired.py` | Pairs checks before/after review: kept/removed × correct/wrong, catch rate, removal precision |
| `run_arms.sh` | Older two-arm run (review on vs `--no-review`, separate extractions). Confounded by run-to-run nondeterminism; kept for reference |
| `compare_arms.py` | Aggregates the two arms on shared PMIDs |
| `inspect_sample_size.py` | Confirms ungrounded `sample_size` values are absent from the source text (raw and comma-formatted) |
| `build_reviewer_base.py` | Builds `data/eval/reviewer_base.json` for `eval_reviewer.py` from the unreviewed drafts + full source text of the no-review run |
| `run_reviewer_eval.sh` | Env-loading wrapper for `scripts/eval_reviewer.py --base data/eval/reviewer_base.json` (e.g. `--n-per-type 10`) |

## Results (Nov 2026, 3 queries × top-15 = 45 records, 39 unique papers)

How the papers were chosen and graded:
- **No curated set and no gold labels.** The 45 records are whatever the agent's own PubMed search returned
  (top 15 after the stage-1 filter) for three queries. 6 PMIDs appear under two queries, so 39 unique papers.
  1. `结直肠癌血浆cfDNA甲基化，需要健康对照` (CRC, plasma cfDNA, healthy controls)
  2. `肝细胞癌血浆cfDNA甲基化数据，要有健康人对照` (HCC, plasma cfDNA, healthy controls)
  3. `结直肠癌组织和癌旁组织的DNA甲基化数据` (CRC, tumour vs adjacent tissue)
- **Grading is deterministic string/regex matching** against the text the extractor was shown (no LLM judge).
  "Grounded" means the value appears in that text (for AUC: near an AUC term). It does not mean the value is
  correct, and it says nothing about what the extractor missed.
- **Source text:** 31 records used PMC full text, 14 fell back to the abstract (11 unique papers). The fallback
  is availability, not a key or rate-limit problem: an `elink pubmed_pmc` check shows 0 of those 11 are in PMC,
  while all 28 unique full-text papers are.

- Overall grounding 85.8% (169/197) before review, 85.7% after.
- By field: dataset_ids 100%, sample_type 100%, markers 97%, auc ~76%, **sample_size 50%** (48.9 / 51.1 / 50.0 across three runs).
- Reviewer (`tools/extraction_reviewer.py`): changed 1 of 173 values, caught 0 of 27 ungrounded ones.
  The apparent effect in the two-arm run was nondeterminism, not the reviewer.

## Reviewer comparison: planted errors (Nov 2026, glm-5.2, 39 base papers = the unique papers above, 10 per type)

`run_reviewer_eval.sh --n-per-type 10` → baseline = `tools/extraction_reviewer.py`, checklist = `tools/checklist_reviewer.py`.

| Planted error | Baseline | Checklist |
|---|---|---|
| auc_fabricated | removed 10/10, right flag 0/10 | removed 10/10, right check 10/10 |
| auc_mislabeled | removed 10/10, right flag 0/10 | removed 10/10, right check 10/10 |
| sample_type_swap | flagged 2/10 | flagged 10/10 |
| reference_accession | removed 10/10, flagged 10/10 | removed 10/10, flagged 10/10 |
| clean (10 untouched) | 0 flags | 4 flagged (see below) |
| latency | 13 s/record | 31 s/record |

Notes:
- `eval_reviewer.py` scores recall by flags only (baseline 0.30 vs checklist 1.00). That understates the
  baseline, which often silently nulls the planted value; the "removed" column scores the outcome instead.
- The checklist's clean-arm "false positives" were cross-checked against the grounding grades: 3 of 4 are
  **real** ungrounded sample sizes (derived sums such as 34+33=67 not stated in the text). Of the 10 clean
  papers, 4 had an ungrounded sample size; checklist flagged 3, baseline 0.
- 1 true false alarm (PMID 37918207): the string-match backstop overrode the model's PASS and nulled AUCs
  stored as a composite string ("0.937 (EC), 0.968 (GC), ...") whose numbers do appear in the text.
- No planted sample-size mutation exists yet, although that is the dominant real-world failure.

## Next steps
- Widen the sample: more queries and cancer types, and a small hand-verified set to measure correctness and misses.
- Add a `sample_size_derived` mutation (replace a stated count with a subgroup sum) to `eval_reviewer.py`.
- Make the C1 backstop parse numbers out of composite AUC strings before declaring them absent.
- Re-run both evals, then wire `review_by_checklist` into the pipeline behind a config flag.
- Move presence checks (sample size stated? AUC stated?) into deterministic code; reserve the LLM for judgment calls.
- Require a verbatim evidence quote (and shown arithmetic for derived counts) per extracted value.
