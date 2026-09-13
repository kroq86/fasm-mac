#!/usr/bin/env python3
"""Read-only, standard-library artifact ledger. Not a statistical rerun."""
import argparse
import hashlib
import json
from pathlib import Path
from check_xor_design import diagnose

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'scratchpad/gpt2_distilgpt2_latent_bridge/'
SOURCES = {
    'ordering': ('phase1_kv_cache_minigunpoint_32token_checkpoint_result.json', ['dev', 'rank', 'seed', 'epochs_run', 'trainable_parameter_count', 'verdict']),
    'ablations': ('phase1_kv_cache_minigunpoint_32token_ablations_result.json', ['baseline_correct', 'baseline_shuffled_document', 'k_vs_v', 'cross_layer', 'head_alignment', 'persistence_delay', 'scale_mismatch', 'noise_robustness', 'receiver_context_conflict']),
    'lookup': ('phase1_kv_cache_sql_lookup_binding_result.json', ['test', 'train_docs', 'dev_docs', 'test_docs', 'rank', 'seed', 'lr', 'epochs_run', 'trainable_parameter_count', 'verdict']),
    'two_key': ('phase1_kv_cache_multifact_2key_batched_result.json', ['dev', 'epochs_run', 'seed', 'verdict']),
    'xor': ('phase1_kv_cache_compositionality_32token_result.json', ['dev', 'epochs_run', 'best_dev_acc_epoch', 'seed', 'verdict']),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ledger():
    result = {'schema': 1, 'kind': 'source consistency, not new experimental evidence', 'sources': {}}
    for label, (name, fields) in SOURCES.items():
        relative = PREFIX + name
        path = ROOT / relative
        data = json.loads(path.read_text())
        result['sources'][label] = {'path': relative, 'sha256': sha(path), 'values': {key: data[key] for key in fields}}
    extras = [
        PREFIX + 'phase1_kv_cache_minigunpoint_32token_bridge.pt',
        PREFIX + 'phase1_kv_cache_minigunpoint_32token.py',
        PREFIX + 'phase1_kv_cache_minigunpoint_32token_ablations.py',
        PREFIX + 'phase1_kv_cache_sql_lookup_binding.py',
        PREFIX + 'phase1_kv_cache_multifact_2key_batched.py',
        PREFIX + 'phase1_kv_cache_compositionality_32token.py',
        'fasm/spikes/tensor_gpt2_handoff.h',
        'fasm/spikes/tensor_kv_handoff.c',
        'scripts/prepare_kv_handoff.py',
        'scripts/check_kv_handoff.sh',
        'scratchpad/kv_transfer_audit_20260913/native_gate.log',
    ]
    result['files'] = {path: sha(ROOT / path) for path in extras}
    log = (ROOT / extras[-1]).read_text()
    expected = 'PASS cases=32 correct=31 split=original_dev worst_abs=0.00103759766 atol=0.001 rtol=0.0001'
    if expected not in log or 'native KV handoff gate passed' not in log:
        raise ValueError('native evidence differs from manuscript result')
    result['native_summary'] = expected
    result['xor_design_audit'] = diagnose()
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = ledger()
    if args.check:
        expected = json.loads(Path(__file__).with_name('evidence.json').read_text())
        if data != expected:
            raise SystemExit('FAIL: source artifacts changed; review manuscript before updating ledger')
        print('PASS: manuscript source ledger matches pinned result artifacts and native log')
    else:
        print(json.dumps(data, indent=2, sort_keys=True))
