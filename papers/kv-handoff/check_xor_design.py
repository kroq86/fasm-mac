#!/usr/bin/env python3
"""Reproduce the archived XOR pairing defect without torch or model execution.

Execute only the named data-generation functions/constants from repository code.
The build_batch stub returns constructed documents, not tensors. No training,
dataset repair, or historical result rewriting occurs.
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
    env = {'hashlib': hashlib, 'random': random, 'build_batch': lambda sender, tokenizer, docs: docs}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), env)
    docs = env['select_docs'](env['generate_pool'](1000), 'dev', 32)
    swapped = env['build_half_swap_batch'](None, None, docs)
    original = dict(sorted(Counter(f'{d[1]},{d[2]}' for d in docs).items()))
    labels = dict(sorted(Counter(str(d[3]) for d in swapped).items()))
    assert original == {'0,0': 8, '0,1': 8, '1,0': 8, '1,1': 8}
    assert labels == {'1': 32}, 'Pairing changed: reassess manuscript rather than silently accepting new design'
    return {'original_combinations': original, 'half_swap_labels': labels,
            'constant_one_correct': 32, 'count': 32,
            'verdict': 'INVALID_AS_BALANCED_COMPOSITION_TEST',
            'scope': 'data-design diagnosis; historical 14/32 score not rerun'}

if __name__ == '__main__':
    print(json.dumps(diagnose(), indent=2, sort_keys=True))
