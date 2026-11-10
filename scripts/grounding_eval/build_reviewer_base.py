"""Build a base-record file for scripts/eval_reviewer.py from the no-review grounding run.

Uses the UNREVIEWED draft extractions (data/eval/arm_noreview) and the full source text each
draft was extracted from (_source_text), falling back to the abstract. Output:
data/eval/reviewer_base.json  ->  python scripts/eval_reviewer.py --base data/eval/reviewer_base.json
"""
import os, glob, json
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
out, seen = [], set()
for f in sorted(glob.glob("data/eval/arm_noreview/*.json")):
    for r in json.load(open(f))["records"]:
        pmid = str(r.get("pmid"))
        if pmid in seen:
            continue
        text = (r.get("_source_text") or r.get("abstract") or "").strip()
        if not text:
            continue
        seen.add(pmid)
        ext = {k: v for k, v in r.items() if not k.startswith("_") and k != "abstract"}
        out.append({"pmid": pmid, "source_text": text, "extraction": ext,
                    "source_basis": r.get("source_basis")})
json.dump({"records": out}, open("data/eval/reviewer_base.json", "w"), ensure_ascii=False)
lens = sorted(len(r["source_text"]) for r in out)
print(f"{len(out)} records; text chars median={lens[len(lens)//2]} max={lens[-1]}")
