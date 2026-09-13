#!/usr/bin/env python3
"""Export a frozen ordering checkpoint + reference bundle. NO TRAINING.

Requires an external torch/transformers/safetensors environment. Native runtime
requires none of them. Source model files are pinned; receiver export changes
container/names only, never shape/layout/values. Output must be a new directory.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SENDER_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
RECEIVER_SHA = "ecbb4e22dd2b9dcc43b2622e1b87ebb9361fb31e496b98ea01a38785c1dbaa01"
BRIDGE_SHA = "dd7ddbdbc68503be5257313335fd21dce34c35ba66b16429cfa173eafbca7f7f"

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sender', type=Path, required=True, help='original GPT-2 snapshot directory')
    p.add_argument('--receiver', type=Path, required=True, help='original DistilGPT2 directory')
    p.add_argument('--bridge', type=Path, default=ROOT/'scratchpad/gpt2_distilgpt2_latent_bridge/phase1_kv_cache_minigunpoint_32token_bridge.pt')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): p.error('output exists; use a new directory')
    for file, sha in [(a.sender/'model.safetensors', SENDER_SHA), (a.receiver/'pytorch_model.bin', RECEIVER_SHA), (a.bridge, BRIDGE_SHA)]:
        if digest(file) != sha: raise RuntimeError(f'fingerprint mismatch: {file}')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import save_file
    spec = importlib.util.spec_from_file_location('ordering', ROOT/'scratchpad/gpt2_distilgpt2_latent_bridge/phase1_kv_cache_minigunpoint_32token_ablations.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    # CPU eager is a deliberately fixed oracle, separate from native Accelerate.
    m.DEVICE = 'cpu'; torch.set_num_threads(4); torch.manual_seed(m.SEED)
    ck = torch.load(a.bridge, map_location='cpu', weights_only=True)
    assert ck['sender_layer_selection'] == [1,3,5,7,9,11]
    assert (ck['rank'],ck['n_head'],ck['head_dim']) == (8,12,64)
    tok = AutoTokenizer.from_pretrained(a.receiver, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(a.sender, local_files_only=True, attn_implementation='eager').eval()
    receiver = AutoModelForCausalLM.from_pretrained(a.receiver, local_files_only=True, attn_implementation='eager').eval()
    assert sender.config.n_layer == 12 and receiver.config.n_layer == 6
    adapter = m.LowRankKVAdapter(6,64,12,8,m.SEED).eval()
    adapter.load_state_dict(ck['state_dict'])
    for model in (sender, receiver, adapter):
        for param in model.parameters(): param.requires_grad_(False)
    docs = m.select_docs(m.generate_pool(200), 'dev', 32)
    assert sorted(''.join(s) for s,_ in docs) == ck['dev_docs_keys']
    refs = {}; cases = []
    with torch.no_grad():
        for i, (seq, cls) in enumerate(docs):
            keys, values, target = m.build_batch(sender,tok,[(seq,cls)],ck['sender_layer_selection'])
            ak, av = adapter.forward_keys(keys), adapter.forward_values(values)
            for layer in range(6):
                for kind, tensors in [('source_k',keys),('source_v',values),('adapt_k',ak),('adapt_v',av)]:
                    refs[f'case{i}.{kind}.{layer}'] = tensors[layer][0].permute(1,0,2).reshape(-1,768).contiguous()
            q = m.query_ids(tok,'Class:',1)
            logits = m.next_token_logits(receiver,q,m.make_cache(ak,av))[0]
            refs[f'case{i}.logits'] = logits.contiguous()
            cases.append(f"{''.join(seq)} {int(target.item())}\n")
            print(f'reference {i+1}/32: predicted={logits.argmax().item()} target={target.item()}',flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kv-export-',dir=a.out.parent) as td:
        out = Path(td)/'bundle'; out.mkdir()
        shutil.copyfile(a.sender/'model.safetensors',out/'sender.safetensors')
        # Named, lossless conversion from the pinned original receiver checkpoint.
        save_file({k:v.detach().cpu().clone().contiguous() for k,v in receiver.transformer.state_dict().items()},str(out/'receiver.safetensors'))
        save_file({k:v.detach().cpu().clone().contiguous() for k,v in adapter.state_dict().items()},str(out/'bridge.safetensors'))
        save_file(refs,str(out/'reference.safetensors'))
        for name in ('vocab.json','merges.txt'): shutil.copyfile(a.receiver/name,out/name)
        (out/'cases.txt').write_text(''.join(cases))
        files = {f.name:digest(f) for f in sorted(out.iterdir())}
        manifest = {'schema':'gpt2-distilgpt2-ordering-v1','files':files,'source_sha256':{'sender':SENDER_SHA,'receiver':RECEIVER_SHA,'bridge':BRIDGE_SHA},
            'oracle':{'torch':torch.__version__,'transformers':transformers.__version__,'device':'cpu','attention':'eager'},
            'layout':'position,head*64+dim','dtype':'F32','layers':[1,3,5,7,9,11],'rank':8,'cases':32,
            'split':'original dev, NOT independent test','training_performed':False,
            'tolerance':{'atol':1e-3,'rtol':1e-4}, 'preparer_sha256':digest(__file__)}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        if a.out.exists(): raise RuntimeError('output appeared during export')
        out.rename(a.out)
    print(f'prepared {a.out}')

if __name__ == '__main__': main()
