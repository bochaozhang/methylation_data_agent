import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # run from repo root
import json,glob,collections
def load(arm):
    out={}
    for f in sorted(glob.glob("data/eval/%s/*.json"%arm)):
        d=json.load(open(f)); out[d["meta"]["query"]]={g["pmid"]:g for g in d["graded"]}
    return out
A=load("arm_review"); B=load("arm_noreview")
common={}
for q in A:
    if q in B:
        c=set(A[q])&set(B[q]); common[q]=c
        print("  %-26s reviewON=%2d OFF=%2d common=%2d"%(q[:24],len(A[q]),len(B[q]),len(c)))
def tally(arm,common):
    pf=collections.defaultdict(lambda:[0,0]); papers=0
    for q,pm in common.items():
        for p in pm:
            papers+=1
            for ch in arm[q][p]["checks"]:
                v=ch["verdict"]
                if v=="GROUNDED": pf[ch["field"]][0]+=1
                elif v=="UNGROUNDED": pf[ch["field"]][1]+=1
    return pf,papers
pa,na=tally(A,common); pb,nb=tally(B,common)
print("\nCommon papers compared: %d"%na)
print("%-18s %-24s %-24s %s"%("","reviewer OFF","reviewer ON","delta"))
print("%-18s %5s %5s %5s %8s   %5s %5s %5s %8s"%("FIELD","gr","ung","n","rate","gr","ung","n","rate"))
tg=tu=og=ou=0
for f in sorted(set(pa)|set(pb)):
    bg,bu=pb.get(f,[0,0]); ag,au=pa.get(f,[0,0]); bn=bg+bu; an=ag+au
    br="%.1f%%"%(100*bg/bn) if bn else "n/a"
    ar="%.1f%%"%(100*ag/an) if an else "n/a"
    dl="%+.1fpp"%(100*ag/an-100*bg/bn) if (an and bn) else "-"
    print("%-18s %5d %5d %5d %8s   %5d %5d %5d %8s   %s"%(f,bg,bu,bn,br,ag,au,an,ar,dl))
    og+=bg; ou+=bu; tg+=ag; tu+=au
bn=og+ou; an=tg+tu
print("-"*92)
print("%-18s %5d %5d %5d %7.1f%%   %5d %5d %5d %7.1f%%   %+.1fpp"%("OVERALL",og,ou,bn,100*og/bn,tg,tu,an,100*tg/an,100*tg/an-100*og/bn))
bas=collections.defaultdict(lambda:[0,0])
for q,pm in common.items():
    for p in pm:
        b=A[q][p]["source_basis"]
        for ch in A[q][p]["checks"]:
            if ch["verdict"]=="GROUNDED": bas[b][0]+=1
            elif ch["verdict"]=="UNGROUNDED": bas[b][1]+=1
print("\nsource_basis (reviewer ON):")
for b,(g,u) in sorted(bas.items()):
    print("  %-10s grounded=%-4d ungrounded=%-4d n=%-4d rate=%.1f%%"%(b,g,u,g+u,100*g/(g+u) if g+u else 0))
