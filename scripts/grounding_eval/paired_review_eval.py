"""Paired grounding eval: ONE extraction per paper, graded before and after review.
Removes the non-determinism confound of the two-arm run. Does not modify repo code."""
import sys, json, copy, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # repo root (scripts/grounding_eval/ -> ../..)
sys.path.insert(0, str(ROOT))
import scripts.eval_grounding as eg

_real_review = eg.review_extraction
DRAFTS = []
def _spy(abstract, draft_extraction, llm):
    DRAFTS.append(copy.deepcopy(draft_extraction))      # snapshot before reviewer sees it
    return _real_review(abstract=abstract, draft_extraction=draft_extraction, llm=llm)
eg.review_extraction = _spy

QUERIES = ["结直肠癌血浆cfDNA甲基化，需要健康对照",
           "肝细胞癌血浆cfDNA甲基化数据，要有健康人对照",
           "结直肠癌组织和癌旁组织的DNA甲基化数据"]
OUT = ROOT / "data" / "eval" / "clean"; OUT.mkdir(parents=True, exist_ok=True)
W, S = eg._DEFAULT_AUC_WINDOW, False

for qi, q in enumerate(QUERIES, 1):
    print(f"===== QUERY {qi}: {q}", flush=True)
    DRAFTS.clear()
    reviewed = eg.collect_records(q, 15, review=True, use_fulltext=True)
    assert len(DRAFTS) == len(reviewed), (len(DRAFTS), len(reviewed))
    drafts = []
    for d, r in zip(DRAFTS, reviewed):
        d = dict(d)
        d["_source_text"] = r["_source_text"]; d["abstract"] = r["abstract"]
        d.setdefault("source_basis", r.get("source_basis")); d.setdefault("pmid", r.get("pmid"))
        drafts.append(d)
    json.dump({"query": q,
               "draft":    [eg.grade_record(r, W, S) for r in drafts],
               "reviewed": [eg.grade_record(r, W, S) for r in reviewed]},
              open(OUT / f"q{qi}.json", "w"), ensure_ascii=False, indent=1)
    print(f"  saved q{qi}.json ({len(reviewed)} papers)", flush=True)
print("ALL DONE")
