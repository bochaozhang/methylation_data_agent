import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # run from repo root
import json, glob, collections
def ckey(c): return (c["field"], c["key"], json.dumps(c["value"], ensure_ascii=False, sort_keys=True))
perf = {n: collections.defaultdict(lambda: [0, 0]) for n in ("draft", "reviewed")}
trans = collections.Counter(); trans_f = collections.defaultdict(collections.Counter)
ex = collections.defaultdict(list); papers = 0
for f in sorted(glob.glob("data/eval/clean/q*.json")):
    d = json.load(open(f))
    for dr, rv in zip(d["draft"], d["reviewed"]):
        assert dr["pmid"] == rv["pmid"]; papers += 1
        for n, g in (("draft", dr), ("reviewed", rv)):
            for c in g["checks"]:
                if c["verdict"] == "GROUNDED": perf[n][c["field"]][0] += 1
                elif c["verdict"] == "UNGROUNDED": perf[n][c["field"]][1] += 1
        D = {ckey(c): c["verdict"] for c in dr["checks"] if c["verdict"] != "NOT_APPLICABLE"}
        R = {ckey(c): c["verdict"] for c in rv["checks"] if c["verdict"] != "NOT_APPLICABLE"}
        for k, v in D.items():
            t = ("kept_" if k in R else "removed_") + ("correct" if v == "GROUNDED" else "wrong")
            trans[t] += 1; trans_f[k[0]][t] += 1
            if t.startswith("removed") and len(ex[t]) < 8: ex[t].append((dr["pmid"], k[1], k[2]))
        for k, v in R.items():
            if k not in D:
                t = "added_grounded" if v == "GROUNDED" else "added_ungrounded"
                trans[t] += 1; trans_f[k[0]][t] += 1
                if len(ex[t]) < 8: ex[t].append((rv["pmid"], k[1], k[2]))
print(f"PAIRED RUN: {papers} papers, each extracted ONCE, graded before and after review\n")
print("%-18s %-22s %-22s %s" % ("", "DRAFT (pre-review)", "REVIEWED", "delta"))
G = {"draft": [0, 0], "reviewed": [0, 0]}
for fld in sorted(set(perf["draft"]) | set(perf["reviewed"])):
    row = []
    for n in ("draft", "reviewed"):
        g, u = perf[n].get(fld, [0, 0]); G[n][0] += g; G[n][1] += u
        row.append((g, u, g + u, 100 * g / (g + u) if g + u else None))
    fmt = lambda r: "%3d/%-3d %6s" % (r[0], r[2], "%.1f%%" % r[3] if r[3] is not None else "n/a")
    dl = "%+.1fpp" % (row[1][3] - row[0][3]) if None not in (row[0][3], row[1][3]) else "-"
    print("%-18s %-22s %-22s %s" % (fld, fmt(row[0]), fmt(row[1]), dl))
rd = 100 * G["draft"][0] / sum(G["draft"]); rr = 100 * G["reviewed"][0] / sum(G["reviewed"])
print("-" * 76)
print("%-18s %3d/%-3d %6.1f%%        %3d/%-3d %6.1f%%        %+.1fpp" % ("OVERALL", G["draft"][0], sum(G["draft"]), rd, G["reviewed"][0], sum(G["reviewed"]), rr, rr - rd))
print("\nWHAT THE REVIEWER DID TO EACH CHECKABLE VALUE")
for t in ("kept_correct", "kept_wrong", "removed_correct", "removed_wrong", "added_grounded", "added_ungrounded"):
    print("  %-18s %4d" % (t, trans[t]))
rc, rw, kw = trans["removed_correct"], trans["removed_wrong"], trans["kept_wrong"]
if rc + rw: print(f"\n  removal precision : {rw}/{rc+rw} = {100*rw/(rc+rw):.1f}% of deletions targeted a wrong value")
if rw + kw: print(f"  catch rate        : {rw}/{rw+kw} = {100*rw/(rw+kw):.1f}% of wrong draft values were removed")
print("\nPER FIELD"); print("  %-18s %6s %6s %6s %6s %6s %6s" % ("field", "keepOK", "keepX", "delOK", "delX", "addOK", "addX"))
for fld, c in sorted(trans_f.items()):
    print("  %-18s %6d %6d %6d %6d %6d %6d" % (fld, c["kept_correct"], c["kept_wrong"], c["removed_correct"], c["removed_wrong"], c["added_grounded"], c["added_ungrounded"]))
for t in ("removed_correct", "removed_wrong", "added_grounded", "added_ungrounded"):
    if ex[t]:
        print(f"\n  examples — {t}:"); [print("    PMID %-9s %-22s %s" % e) for e in ex[t]]
