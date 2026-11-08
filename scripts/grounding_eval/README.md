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

## Results (Nov 2026, 3 queries × top-15, 45 papers, full text)

- Overall grounding 85.8% (169/197) before review, 85.7% after.
- By field: dataset_ids 100%, sample_type 100%, markers 97%, auc ~76%, **sample_size 50%** (48.9 / 51.1 / 50.0 across three runs).
- Reviewer (`tools/extraction_reviewer.py`): changed 1 of 173 values, caught 0 of 27 ungrounded ones.
  The apparent effect in the two-arm run was nondeterminism, not the reviewer.

## Next steps
- Run `scripts/eval_reviewer.py` (planted-error meta-eval) on `tools/checklist_reviewer.py` vs the current reviewer.
- Move presence checks (sample size stated? AUC stated?) into deterministic code; reserve the LLM for judgment calls.
- Require a verbatim evidence quote (and shown arithmetic for derived counts) per extracted value.
