#!/usr/bin/env python3
"""Reproducible evidence harness over tensorctl's public CLI output."""
import argparse, csv, os, platform, re, statistics, subprocess, sys, tempfile, time

FIELDS = ["record","workload","shape","budget","repeat","profile_mode","compile_us","planning_us","train_step_ns","standard_segment_ns","layout_segment_ns","peak_bytes","actions","memory_decision","layout_decision","source","confidence","host","compiler","build","profile_revision","repo_revision"]
WORKLOADS = [("mlp","2-4-1",0,4000),("transformer-tight","T3/H2/D2/F6",4928,12000),("transformer-large","T3/H2/D2/F6",8192,12000)]
def run(cmd, env):
    t=time.perf_counter_ns(); p=subprocess.run(cmd,text=True,capture_output=True,env=env)
    if p.returncode: raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}{p.stderr}")
    return p.stdout,(time.perf_counter_ns()-t)/1000
def grab(pattern,text,default=""):
    m=re.search(pattern,text); return (m.group(1) if m.lastindex else m.group(0)) if m else default
def one(binary,name,shape,budget,epochs,rep,mode,compile_us,env,repo_revision):
    base=[binary,"transformer","--plan",f"--memory-budget={budget}"] if name.startswith("transformer") else [binary,"mlp","--plan"]
    if mode=="fresh" and name.startswith("transformer"): base.append("--no-profile-cache")
    plan,planning=run(base,env)
    train_cmd=[binary,"transformer",f"--epochs={epochs}",f"--memory-budget={budget}"] if name.startswith("transformer") else [binary,"mlp",f"--epochs={epochs}"]
    train,_=run(train_cmd,env)
    text=plan+train
    peak=grab(r"decision=(?:save|rematerialize).*",text)
    peak_bytes=5440 if "decision=save(attention_scores)" in text else 4928 if "decision=rematerialize(attention_scores)" in text else 0
    actions=int(grab(r"executed via .* path, (\d+) backward actions",text,grab(r"backward=(\d+) actions",text,"8")))
    key=grab(r"profile: source=\w+ (?:age=\d+s )?key=([^\n]+)",text)
    compiler=grab(r"compiler=([^;]+)",key); build=grab(r"build=(.*?)(?: \(|$)",key); revision=grab(r"revision=([^;]+)",key)
    return {"record":"raw","workload":name,"shape":shape,"budget":budget,"repeat":rep,"profile_mode":mode,"compile_us":f"{compile_us:.3f}","planning_us":f"{planning:.3f}","train_step_ns":grab(r"ns_per_step=([0-9.]+)",text,"0"),"standard_segment_ns":grab(r"standard\(merge\+CONTIGUOUS\+matmul\).*? ([0-9.]+)ns/call",text,"0"),"layout_segment_ns":grab(r"layout-aware\(head-wise matmul\).*? ([0-9.]+)ns/call",text,"0"),"peak_bytes":peak_bytes,"actions":actions,"memory_decision":"save" if "decision=save(attention_scores)" in text else "rematerialize" if peak else "none","layout_decision":grab(r"decision=(standard|layout-aware)\(",text,"none"),"source":grab(r"profile: source=(\w+)",text,"default"),"confidence":grab(r"decision=(?:standard|layout-aware)\((\w+)",text,"n/a"),"host":platform.machine(),"compiler":compiler,"build":build,"profile_revision":revision,"repo_revision":repo_revision}
def summaries(rows):
    out=[]
    for key in sorted({(r["workload"],r["profile_mode"]) for r in rows}):
        group=[r for r in rows if (r["workload"],r["profile_mode"])==key]; base=dict(group[0]);base["record"]="summary";base["repeat"]="median/min/max"
        for col in ("planning_us","train_step_ns","standard_segment_ns","layout_segment_ns"):
            xs=[float(r[col]) for r in group];base[col]=f"{statistics.median(xs):.3f}/{min(xs):.3f}/{max(xs):.3f}"
        out.append(base)
    return out
def write(path,rows):
    with open(path,"w",newline="") as f:w=csv.DictWriter(f,FIELDS,delimiter="\t",lineterminator="\n");w.writeheader();w.writerows(rows)
def read_summary(path):
    with open(path,newline="") as f: rows=list(csv.DictReader(f,delimiter="\t"))
    if not rows or set(rows[0])!=set(FIELDS): raise ValueError("incompatible benchmark TSV")
    return {(r["workload"],r["profile_mode"]):r for r in rows if r["record"]=="summary"}
def median_value(row,col): return float(row[col].split("/")[0])
def compare(old,new,threshold,machine=None):
    changes=[]
    for key in sorted(set(old)|set(new)):
        if key not in old or key not in new: changes.append(("workload", "/".join(key), "present" if key in old else "absent", "present" if key in new else "absent"));continue
        a,b=old[key],new[key]
        for col in ("memory_decision","layout_decision","actions"):
            if a[col]!=b[col]:changes.append(("semantic",f"{'/'.join(key)}.{col}",a[col],b[col]))
        if a["peak_bytes"]!=b["peak_bytes"]:changes.append(("memory",f"{'/'.join(key)}.peak_bytes",a["peak_bytes"],b["peak_bytes"]))
        for col in ("train_step_ns","planning_us"):
            av,bv=median_value(a,col),median_value(b,col);pct=(bv/av-1)*100 if av else 0
            if abs(pct)>=threshold:changes.append(("performance",f"{'/'.join(key)}.{col}",f"{av:.3f}",f"{bv:.3f} ({pct:+.2f}%)"))
        if any(a[c]!=b[c] for c in ("host","compiler","build","profile_revision","repo_revision","source")):changes.append(("provenance",f"{'/'.join(key)}.evidence","changed","changed"))
    if machine:
        with open(machine,"w",newline="") as f:w=csv.writer(f,delimiter="\t",lineterminator="\n");w.writerow(("kind","subject","before","after"));w.writerows(changes)
    if not changes: print("no benchmark changes")
    for kind,subject,a,b in changes:print(f"{kind}: {subject}: {a} -> {b}")
    return changes
def self_test():
    base={"record":"summary","workload":"x","profile_mode":"cached","planning_us":"50/45/55","train_step_ns":"100/90/110","standard_segment_ns":"14/13/15","layout_segment_ns":"16/15/17","memory_decision":"save","layout_decision":"standard","actions":"21","peak_bytes":"5440","host":"m1","compiler":"clang","build":"O2","profile_revision":"v1","repo_revision":"a","source":"measured"}
    old={("x","cached"):base};same={("x","cached"):dict(base)}
    if compare(old,same,5):raise AssertionError("identical changed")
    slow=dict(base,train_step_ns="120/115/125"); ch=compare(old,{("x","cached"):slow},5)
    if not any(x[0]=="performance" for x in ch):raise AssertionError("slow fixture missed")
    changed=dict(base,memory_decision="rematerialize",peak_bytes="4928");ch=compare(old,{("x","cached"):changed},5)
    if not any(x[0]=="semantic" for x in ch):raise AssertionError("planner change missed")
    prov=dict(base,repo_revision="b");ch=compare(old,{("x","cached"):prov},5)
    if [x[0] for x in ch] != ["provenance"]:raise AssertionError("provenance classification")
    print("tensorctl bench evidence self-test passed: identical/slow/semantic/provenance classified")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--tensorctl");ap.add_argument("--output");ap.add_argument("--baseline");ap.add_argument("--diff-output");ap.add_argument("--repeats",type=int,default=5);ap.add_argument("--compile-us",type=float,default=0);ap.add_argument("--threshold",type=float,default=10);ap.add_argument("--self-test",action="store_true");args=ap.parse_args()
    if args.self_test:self_test();return
    if not args.tensorctl or not args.output:ap.error("--tensorctl and --output are required")
    root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try: repo_revision=subprocess.check_output(["git","-C",root,"rev-parse","--short=12","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError): repo_revision="unknown"
    rows=[]
    with tempfile.TemporaryDirectory(prefix="tensorctl-bench-") as cache:
        env=dict(os.environ,XDG_CACHE_HOME=cache)
        for workload in WORKLOADS:
            modes=("fresh","cached") if workload[0].startswith("transformer") else ("default",)
            for mode in modes:
                for rep in range(args.repeats):rows.append(one(args.tensorctl,*workload,rep,mode,args.compile_us,env,repo_revision))
    write(args.output,rows+summaries(rows))
    if args.baseline:compare(read_summary(args.baseline),read_summary(args.output),args.threshold,args.diff_output)
if __name__=="__main__":
    try:main()
    except (OSError,ValueError,RuntimeError) as e:print(f"tensorctl bench: {e}",file=sys.stderr);raise SystemExit(1)
