#!/usr/bin/env python3
"""Join planner traces and benchmark evidence without changing runtime semantics."""
import argparse,csv,os,subprocess,sys,tempfile
def rows(path):
    with open(path,newline="") as f:return list(csv.DictReader(f,delimiter="\t"))
def chosen(plan,decision):
    return next((r for r in plan if r["decision"]==decision and r["chosen"]=="1"),None)
def summary(evidence,workload="transformer-tight",mode="cached"):
    return next((r for r in evidence if r["record"]=="summary" and r["workload"]==workload and r["profile_mode"]==mode),None)
def med(row,col):return float(row[col].split("/")[0])
def verdict(memory_pct,perf_pct,threshold=10):
    if memory_pct is None or perf_pct is None:return"insufficient evidence"
    mp=abs(memory_pct)>=1;pp=abs(perf_pct)>=threshold
    if not mp and not pp:return"within noise"
    if memory_pct<=-1 and perf_pct<=-threshold:return"win"
    if memory_pct>=1 and perf_pct>=threshold:return"loss"
    return"tradeoff"
def produce(plan,cf,evidence,baseline,threshold):
    out=[]
    bench=summary(evidence) if evidence else None;base=summary(baseline) if baseline else None
    perf=(med(bench,"train_step_ns")/med(base,"train_step_ns")-1)*100 if bench and base else None
    for subject in sorted({r["decision"] for r in plan}):
        cur=chosen(plan,subject);other=chosen(cf,subject) if cf else None
        if not cur:continue
        before_peak=int(other["peak_bytes"]) if other else int(cur["peak_bytes"]);after_peak=int(cur["peak_bytes"])
        memory=(after_peak/before_peak-1)*100 if before_peak else None
        local=None
        alternatives=[r for r in plan if r["decision"]==subject]
        if subject=="head_merge_layout" and len(alternatives)>=2:
            standard=next((r for r in alternatives if r["alternative"]=="standard"),alternatives[0]);layout=next((r for r in alternatives if r["alternative"]=="layout-aware"),alternatives[1]);reference=float(standard["compute_cost"]);local=(float(layout["compute_cost"])/reference-1)*100 if reference else None
        # Only a changed executable layout decision may be associated with an
        # external before/after train-step comparison. Save/remat is planning-only today.
        causal_perf=perf if subject=="head_merge_layout" and base and bench and base["layout_decision"]!=bench["layout_decision"] else None
        out.append({"decision":subject,"chosen":cur["alternative"],"counterfactual":other["alternative"] if other else"unavailable","peak_before":before_peak,"peak_after":after_peak,"memory_pct":memory,"local_pct":local,"end_to_end_pct":causal_perf,"verdict":verdict(memory,causal_perf,threshold),"source":cur["source"] or"default","confidence":"uncertain" if cur["uncertain"]=="1" else"confident","reason":cur["reason"]})
    return out
def fmt(v):return"unavailable" if v is None else f"{v:+.2f}%"
def render(report,human,machine):
    f=open(human,"w") if human else sys.stdout
    for r in report:f.write(f"decision: {r['decision']}\n  chosen: {r['chosen']}\n  counterfactual: {r['counterfactual']}\n  memory: {r['peak_before']} -> {r['peak_after']} bytes ({fmt(r['memory_pct'])})\n  local effect: {fmt(r['local_pct'])}\n  end-to-end effect: {fmt(r['end_to_end_pct'])}\n  evidence: source={r['source']} confidence={r['confidence']}\n  verdict: {r['verdict']}\n")
    if human:f.close()
    if machine:
        fields=["decision","chosen","counterfactual","peak_before","peak_after","memory_pct","local_pct","end_to_end_pct","source","confidence","verdict","reason"]
        with open(machine,"w",newline="") as f:w=csv.DictWriter(f,fields,delimiter="\t",lineterminator="\n");w.writeheader();w.writerows(report)
def self_test():
    cases=[(-10,-12,"win"),(10,12,"loss"),(-10,12,"tradeoff"),(.2,3,"within noise")]
    for m,p,want in cases:
        if verdict(m,p)!=want:raise AssertionError((m,p,verdict(m,p),want))
    if verdict(-10,None)!="insufficient evidence":raise AssertionError("missing evidence invented")
    print("tensorctl report self-test passed: win/loss/tradeoff/within-noise/insufficient-evidence")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("model",nargs="?",default="transformer");ap.add_argument("--tensorctl");ap.add_argument("--memory-budget",type=int,default=5000);ap.add_argument("--counterfactual-budget",type=int,default=8192);ap.add_argument("--evidence");ap.add_argument("--baseline");ap.add_argument("--human");ap.add_argument("--machine");ap.add_argument("--threshold",type=float,default=10);ap.add_argument("--self-test",action="store_true");a=ap.parse_args()
    if a.self_test:self_test();return
    if a.model!="transformer" or not a.tensorctl:ap.error("transformer and --tensorctl are required")
    with tempfile.TemporaryDirectory(prefix="tensorctl-report-") as d:
        pa,pb=os.path.join(d,"current.tsv"),os.path.join(d,"counterfactual.tsv");env=dict(os.environ,XDG_CACHE_HOME=os.path.join(d,"cache"))
        for budget,path in ((a.memory_budget,pa),(a.counterfactual_budget,pb)):
            subprocess.run([a.tensorctl,"plan","transformer",f"--memory-budget={budget}","--export",path],check=True,stdout=subprocess.DEVNULL,env=env)
        report=produce(rows(pa),rows(pb),rows(a.evidence) if a.evidence else None,rows(a.baseline) if a.baseline else None,a.threshold);render(report,a.human,a.machine)
if __name__=="__main__":
    try:main()
    except (OSError,ValueError,subprocess.CalledProcessError) as e:print(f"tensorctl report: {e}",file=sys.stderr);raise SystemExit(1)
