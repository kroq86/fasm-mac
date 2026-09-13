#!/usr/bin/env python3
"""Checked launcher for the native ordering handoff. Standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = {'sender.safetensors','receiver.safetensors','bridge.safetensors','reference.safetensors','vocab.json','merges.txt','cases.txt'}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def validate_assets(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest.get('schema') != 'gpt2-distilgpt2-ordering-v1' or manifest.get('layers') != [1,3,5,7,9,11]
        or manifest.get('dtype') != 'F32' or manifest.get('rank') != 8 or manifest.get('cases') != 32
        or manifest.get('layout') != 'position,head*64+dim' or manifest.get('tolerance') != {'atol':1e-3,'rtol':1e-4}):
        raise ValueError('unsupported bundle contract')
    files = manifest.get('files',{})
    if not isinstance(files,dict) or set(files) != FILES: raise ValueError('incomplete or unknown bundle files')
    for name, sha in files.items():
        if not isinstance(sha,str) or len(sha)!=64 or digest(directory/name)!=sha:
            raise ValueError(f'fingerprint mismatch: {name}')
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--binary',type=Path,default=ROOT/'fasm/build/out/tensor-kv-handoff')
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--sequence',help='exactly 32 A/B symbols; emits one class token')
    g.add_argument('--verify',action='store_true',help='compare all 32 original dev cases against reference')
    a=p.parse_args()
    if a.sequence is not None and (len(a.sequence)!=32 or set(a.sequence)-{'A','B'}): p.error('sequence must contain exactly 32 A/B symbols')
    try:
        m=validate_assets(a.assets)
        command=[str(a.binary.resolve()),str(a.assets.resolve()),m['files']['sender.safetensors'],m['files']['receiver.safetensors'],m['files']['bridge.safetensors']]
        if a.verify: command+=['--verify',str((a.assets/'cases.txt').resolve()),str((a.assets/'reference.safetensors').resolve())]
        else: command+=['--sequence',a.sequence,'unused']
        return subprocess.run(command,check=False).returncode
    except (OSError,ValueError,TypeError) as e:
        print(f'kv-handoff: {e}',file=sys.stderr); return 2

if __name__=='__main__': sys.exit(main())
