#!/usr/bin/env python3
"""Diagnose the XOR half-swap pairing design without torch or model execution.

Execute only the named data-generation functions/constants from repository code.
The build_batch stub returns constructed documents, not tensors. No training,
dataset repair beyond the tracked source-code fix, or historical result
rewriting occurs.

Historical note: the design originally paired doc i's half1 with doc
(i+1 mod n)'s half2, which made every half-swap recombination land in class 1
(the defect this script used to assert). The tracked source has since been
patched to pair doc i with doc (i+2 mod n), which balances the recombined
labels for the current select_docs ordering; that is now verified here, not
assumed. The historical 14/32 half-swap score in evidence.json's "xor" source
was produced under the earlier, unbalanced pairing and is not rerun here.
"""
import ast
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'scratchpad/gpt2_distilgpt2_latent_bridge/phase1_kv_cache_compositionality_32token.py'

def diagnose():
    tree = ast.parse(SOURCE.read_text())
    functions = {'sha', 'bucket', 'make_half', 'make_doc', 'generate_pool', 'select_docs', 'build_half_swap_batch'}
    constants = {'SEED', 'SEQ_LEN', 'HALF', 'SUB', 'SUB_K_RANGE'}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants:
            selected.append(node)
    assert {n.name for n in selected if isinstance(n, ast.FunctionDef)} == functions
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + selected, type_ignores=[])
    env = {'hashlib': hashlib, 'random': random, 'Counter': Counter, 'build_batch': lambda sender, tokenizer, docs: docs}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), env)
    docs = env['select_docs'](env['generate_pool'](1000), 'dev', 32)
    swapped = env['build_half_swap_batch'](None, None, docs)
    original = dict(sorted(Counter(f'{d[1]},{d[2]}' for d in docs).items()))
    labels = dict(sorted(Counter(str(d[3]) for d in swapped).items()))
    assert original == {'0,0': 8, '0,1': 8, '1,0': 8, '1,1': 8}
    balanced = labels == {'0': 16, '1': 16}
    best_constant_correct = max(labels.get('0', 0), labels.get('1', 0))
    return {
        'original_combinations': original,
        'half_swap_labels': labels,
        'balanced': balanced,
        'best_constant_predictor_correct': best_constant_correct,
        'count': 32,
        'verdict': 'BALANCED_AS_COMPOSITION_TEST' if balanced else 'INVALID_AS_BALANCED_COMPOSITION_TEST',
        'scope': (
            'data-design diagnosis of the CURRENT tracked pairing; the historical '
            '14/32 half-swap score in evidence.json was produced under an earlier, '
            'unbalanced pairing and is not rerun by this script'
        ),
    }

if __name__ == '__main__':
    print(json.dumps(diagnose(), indent=2, sort_keys=True))
