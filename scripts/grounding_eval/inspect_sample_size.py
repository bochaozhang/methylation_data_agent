import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # run from repo root
import json,glob
rows=[]
for f in sorted(glob.glob("data/eval/arm_noreview/*.json")):
    d=json.load(open(f))
    for g in d["graded"]:
        txt=g.get("_source_text","") or ""
        for ch in g["checks"]:
            if ch["field"]=="sample_size" and ch["verdict"]=="UNGROUNDED":
                v=str(ch["value"])
                rows.append((g["pmid"],ch["key"],v,ch["reason"][:60],
                             "RAW-PRESENT" if v in txt else "absent",
                             "COMMA-FORM" if f"{int(v):,}" in txt else ""))
print("ungrounded sample_size instances:",len(rows))
print("%-10s %-20s %-7s %-12s %s"%("PMID","key","value","raw-in-text","comma-form"))
for r in rows[:18]: print("%-10s %-20s %-7s %-12s %s"%(r[0],r[1],r[2],r[4],r[5]))
import collections
print("\nraw-present tally:",collections.Counter(r[4] for r in rows))
print("comma-form tally:",collections.Counter(r[5] for r in rows))
