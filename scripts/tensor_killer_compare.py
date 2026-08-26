#!/usr/bin/env python3
import argparse, os, statistics, subprocess, time
from pathlib import Path
def parse(s):
    return dict(line.split('\t',1) for line in s.strip().splitlines() if '\t' in line)
def tree_bytes(p):
    p=Path(p);return p.stat().st_size if p.is_file() else sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
def run(cmd,env=None):return subprocess.run(cmd,text=True,capture_output=True,check=True,env=env).stdout
ap=argparse.ArgumentParser();ap.add_argument('--native',required=True);ap.add_argument('--python',required=True);ap.add_argument('--runner',required=True);ap.add_argument('--model',required=True);a=ap.parse_args();m=Path(a.model)
common=[str(m/'w1.bin'),str(m/'b1.bin'),str(m/'w2.bin'),str(m/'b2.bin'),str(m/'inputs.bin'),str(m/'labels.bin')]
cmds={'native':[a.native,*common,'1000'],'onnxruntime':[a.python,a.runner,'onnxruntime','--model',a.model,'--reps','1000'],'tinygrad':[a.python,a.runner,'tinygrad','--model',a.model,'--reps','1000']}
rows={}
for name,cmd in cmds.items():
    warm=parse(run(cmd));cold=[]
    for _ in range(7):
        e=os.environ.copy();e['KILLER_SKIP_ACCURACY']='1';cc=cmd[:-1]+['1'] if name=='native' else cmd[:-1]+['1'];t=time.perf_counter_ns();run(cc,e);cold.append(time.perf_counter_ns()-t)
    rows[name]={'warm_ns':float(warm['warm_median_ns']),'accuracy':float(warm['accuracy']),'rss':int(warm['peak_rss_bytes']),'checksum':float(warm['checksum']),'cold_ns':statistics.median(cold)}
base=min(rows['onnxruntime']['warm_ns'],rows['tinygrad']['warm_ns']);coldbase=min(rows['onnxruntime']['cold_ns'],rows['tinygrad']['cold_ns']);rssbase=min(rows['onnxruntime']['rss'],rows['tinygrad']['rss'])
model_bytes=(m/'model.onnx').stat().st_size;native_bytes=Path(a.native).stat().st_size+sum((m/n).stat().st_size for n in ('w1.bin','b1.bin','w2.bin','b2.bin'));rows['native']['deploy_bytes']=native_bytes
for name,module in [('onnxruntime','onnxruntime'),('tinygrad','tinygrad')]:
    loc=run([a.python,'-c',f'import {module};print({module}.__path__[0])']).strip();rows[name]['deploy_bytes']=tree_bytes(loc)+Path(a.python).stat().st_size+model_bytes
ok=max(abs(rows['native']['checksum']-rows[x]['checksum']) for x in ('onnxruntime','tinygrad'))<=1e-2*max(1,abs(rows['native']['checksum'])) and len({round(v['accuracy'],6) for v in rows.values()})==1
wins=[]
if rows['native']['warm_ns']<=.7*base:wins.append('latency>=30%')
if rows['native']['cold_ns']<=.5*coldbase:wins.append('startup>=2x')
if rows['native']['rss']<=.5*rssbase:wins.append('rss>=2x')
if rows['native']['deploy_bytes']<=.5*min(rows['onnxruntime']['deploy_bytes'],rows['tinygrad']['deploy_bytes']):wins.append('footprint>=2x')
print('engine\tcold_ns\twarm_batch1_ns\tpeak_rss_bytes\tdeploy_bytes\taccuracy\tchecksum')
for n,v in rows.items():print(f"{n}\t{v['cold_ns']:.0f}\t{v['warm_ns']:.0f}\t{v['rss']}\t{v['deploy_bytes']}\t{v['accuracy']:.4f}\t{v['checksum']:.6g}")
print(f"correctness\t{'pass' if ok else 'FAIL'}");print(f"verdict\t{'PRODUCT HYPOTHESIS SURVIVES: '+','.join(wins) if ok and wins else 'NO MATERIAL ADVANTAGE'}")
raise SystemExit(0 if ok else 3)
