#!/usr/bin/env python3
"""Read-only, standard-library artifact ledger. Not a statistical rerun."""
import argparse
import hashlib
import json
from pathlib import Path
from check_xor_design import diagnose
from tost_equivalence import analyze as tost_analyze

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'scratchpad/gpt2_distilgpt2_latent_bridge/'
SOURCES = {
    'ordering': ('phase1_kv_cache_minigunpoint_32token_checkpoint_result.json', ['dev', 'rank', 'seed', 'epochs_run', 'trainable_parameter_count', 'verdict']),
    'ablations': ('phase1_kv_cache_minigunpoint_32token_ablations_result.json', ['baseline_correct', 'baseline_shuffled_document', 'k_vs_v', 'cross_layer', 'head_alignment', 'persistence_delay', 'scale_mismatch', 'noise_robustness', 'receiver_context_conflict']),
    'lookup': ('phase1_kv_cache_sql_lookup_binding_result.json', ['test', 'train_docs', 'dev_docs', 'test_docs', 'rank', 'seed', 'lr', 'epochs_run', 'trainable_parameter_count', 'verdict']),
    'two_key': ('phase1_kv_cache_multifact_2key_batched_result.json', ['dev', 'epochs_run', 'seed', 'verdict']),
    'xor': ('phase1_kv_cache_compositionality_32token_result.json', ['dev', 'epochs_run', 'best_dev_acc_epoch', 'seed', 'verdict']),
    'moment_matched': ('phase1_kv_cache_minigunpoint_32token_moment_matched_result.json', ['seed', 'correct', 'moment_matched_random', 'moment_matched_minus_correct', 'reference_historical_arms']),
    'latency_baseline': ('phase1_kv_cache_minigunpoint_32token_latency_baseline_result.json', ['method', 'repeats', 'batch_size', 'sequence_length_tokens', 'reprefill_length_tokens', 'cache_transfer_path', 'receiver_reprefill_path', 'reprefill_over_cache_transfer_speedup_median']),
    'attention_cosine': ('phase1_kv_cache_minigunpoint_32token_attention_cosine_result.json', ['method', 'accuracy_reference', 'attention_output_cosine_to_correct']),
    'xor_repaired': ('phase1_kv_cache_compositionality_32token_repaired_result.json', ['seed', 'epochs_run', 'best_dev_acc_epoch', 'still_improving_at_cutoff', 'accuracy_above_chance', 'beats_half1_only', 'beats_half2_only', 'swap_above_chance', 'verdict', 'dev']),
    'template_family': ('phase1_kv_cache_minigunpoint_32token_template_family_result.json', ['trained_symbols', 'alternate_symbols', 'trained_template', 'alternate_template']),
    'lookup_replay': ('phase1_kv_cache_sql_lookup_binding_replay_result.json', ['seed', 'epochs_run', 'stopped_early', 'test', 'correct_beats_shuffled', 'correct_beats_wrong_binding', 'verdict']),
    'lookup_natural_language': ('phase1_kv_cache_sql_lookup_binding_natural_language_result.json', ['seed', 'epochs_run', 'stopped_early', 'test', 'correct_beats_shuffled', 'correct_beats_wrong_binding', 'verdict']),
    'lookup_natural_language_seed_variance': ('phase1_kv_cache_sql_lookup_binding_natural_language_seed_variance_result.json', ['method', 'seeds', 'runs', 'accuracy_mean', 'accuracy_min', 'accuracy_max', 'accuracy_range_in_32ths', 'verdicts']),
    'lookup_seed_variance': ('phase1_kv_cache_sql_lookup_binding_seed_variance_result.json', ['method', 'seeds', 'runs', 'accuracy_mean', 'accuracy_min', 'accuracy_max', 'accuracy_range_in_32ths', 'verdicts']),
    'lookup_natural_language_rank_sweep': ('phase1_kv_cache_sql_lookup_binding_natural_language_rank_sweep_result.json', ['mechanism', 'method', 'seeds_tested', 'ranks_tested', 'runs']),
    'lookup_natural_language_lr_sweep': ('phase1_kv_cache_sql_lookup_binding_natural_language_lr_sweep_result.json', ['mechanism', 'method', 'seeds_tested', 'learning_rates_tested', 'runs']),
    'lookup_natural_language_tuned_lr_seed_variance': ('phase1_kv_cache_sql_lookup_binding_natural_language_tuned_lr_seed_variance_result.json', ['method', 'tuned_lr', 'seeds', 'runs', 'accuracy_mean', 'accuracy_min', 'accuracy_max', 'accuracy_range_in_32ths', 'verdicts']),
    'lookup_document_form_matrix': ('phase1_kv_cache_sql_lookup_binding_document_form_matrix_result.json', ['mechanism', 'method', 'tuned_lr', 'seeds', 'form_token_lengths', 'forms']),
    'seed_variance': ('phase1_kv_cache_minigunpoint_32token_seed_variance_result.json', ['method', 'seeds', 'runs', 'accuracy_mean', 'accuracy_min', 'accuracy_max', 'accuracy_range_in_32ths']),
    'length_variation': ('phase1_kv_cache_minigunpoint_32token_length_variation_result.json', ['method', 'lengths_tested', 'docs_per_length', 'trained_length_32', 'length_16', 'length_48']),
    'amortized_latency': ('phase1_kv_cache_minigunpoint_32token_amortized_latency_threads1_batch32_result.json', ['method', 'threads', 'batch_size', 'fixed_cost_sender_plus_adapter', 'marginal_cost_cache_query', 'marginal_cost_reprefill', 'breakeven_query_count', 'totals_by_query_count']),
    'amortized_latency_robustness': ('phase1_kv_cache_minigunpoint_32token_amortized_latency_robustness_result.json', ['configs', 'breakeven_range', 'n1_cache_transfer_faster_in_any_config']),
    'amortized_latency_scenario_matrix': ('phase1_kv_cache_minigunpoint_32token_amortized_latency_scenario_matrix_result.json', ['matrix']),
    'lookup_multiquery_latency': ('phase1_kv_cache_sql_lookup_binding_multiquery_latency_result.json', ['method', 'names', 'repeats', 'threads', 'fixed_cost_sender_plus_adapter', 'marginal_cost_cache_query', 'marginal_cost_reprefill', 'breakeven_query_count', 'sunk_cost_speedup_x']),
    'lookup_multiquery_latency_threadsdefault': ('phase1_kv_cache_sql_lookup_binding_multiquery_latency_threadsdefault_result.json', ['method', 'names', 'repeats', 'threads', 'fixed_cost_sender_plus_adapter', 'marginal_cost_cache_query', 'marginal_cost_reprefill', 'breakeven_query_count', 'sunk_cost_speedup_x']),
}
NATIVE_LATENCY_FIELDS = ['host_environment', 'method', 'repeats', 'sequence', 'via_checked_launcher', 'via_direct_binary']
NATIVE_LATENCY_PATH = 'scratchpad/kv_transfer_audit_20260913/native_kv_handoff_latency_result.json'
NATIVE_LATENCY_FIXED_SHA256_PATH = 'scratchpad/kv_transfer_audit_20260913/native_kv_handoff_latency_fixed_sha256_result.json'
NATIVE_LATENCY_ARM64_PATH = 'scratchpad/kv_transfer_audit_20260913/native_kv_handoff_latency_arm64_result.json'
NATIVE_AMORTIZED_LATENCY_FIELDS = ['host_environment', 'method', 'warmup', 'repeats', 'fixed_cost_sender_plus_adapter', 'marginal_cost_cache_query']
NATIVE_AMORTIZED_LATENCY_ARM64_PATH = 'scratchpad/kv_transfer_audit_20260913/native_amortized_latency_arm64_result.json'
NATIVE_AMORTIZED_LATENCY_X86_64_PATH = 'scratchpad/kv_transfer_audit_20260913/native_amortized_latency_x86_64_result.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ledger():
    result = {'schema': 1, 'kind': 'source consistency, not new experimental evidence', 'sources': {}}
    for label, (name, fields) in SOURCES.items():
        relative = PREFIX + name
        path = ROOT / relative
        data = json.loads(path.read_text())
        result['sources'][label] = {'path': relative, 'sha256': sha(path), 'values': {key: data[key] for key in fields}}
    for label, rel_path in (
        ('native_latency', NATIVE_LATENCY_PATH),
        ('native_latency_fixed_sha256', NATIVE_LATENCY_FIXED_SHA256_PATH),
        ('native_latency_arm64', NATIVE_LATENCY_ARM64_PATH),
    ):
        full_path = ROOT / rel_path
        data = json.loads(full_path.read_text())
        result['sources'][label] = {
            'path': rel_path, 'sha256': sha(full_path),
            'values': {key: data[key] for key in NATIVE_LATENCY_FIELDS},
        }
    for label, rel_path in (
        ('native_amortized_latency_arm64', NATIVE_AMORTIZED_LATENCY_ARM64_PATH),
        ('native_amortized_latency_x86_64', NATIVE_AMORTIZED_LATENCY_X86_64_PATH),
    ):
        full_path = ROOT / rel_path
        data = json.loads(full_path.read_text())
        result['sources'][label] = {
            'path': rel_path, 'sha256': sha(full_path),
            'values': {key: data[key] for key in NATIVE_AMORTIZED_LATENCY_FIELDS},
        }
    extras = [
        PREFIX + 'phase1_kv_cache_minigunpoint_32token_bridge.pt',
        PREFIX + 'phase1_kv_cache_minigunpoint_32token.py',
        PREFIX + 'phase1_kv_cache_minigunpoint_32token_ablations.py',
        PREFIX + 'phase1_kv_cache_minigunpoint_32token_moment_matched.py',
        PREFIX + 'phase1_kv_cache_minigunpoint_32token_latency_baseline.py',
        PREFIX + 'phase1_kv_cache_minigunpoint_32token_attention_cosine.py',
        PREFIX + 'phase1_kv_cache_minigunpoint_32token_template_family.py',
        PREFIX + 'phase1_kv_cache_minigunpoint_32token_seed_variance.py',
        PREFIX + 'phase1_kv_cache_minigunpoint_32token_length_variation.py',
        PREFIX + 'phase1_kv_cache_minigunpoint_32token_amortized_latency.py',
        PREFIX + 'phase1_kv_cache_sql_lookup_binding_multiquery_latency.py',
        PREFIX + 'phase1_kv_cache_sql_lookup_binding.py',
        PREFIX + 'phase1_kv_cache_sql_lookup_binding_natural_language.py',
        PREFIX + 'phase1_kv_cache_sql_lookup_binding_natural_language_seed_variance.py',
        PREFIX + 'phase1_kv_cache_sql_lookup_binding_seed_variance.py',
        PREFIX + 'phase1_kv_cache_sql_lookup_binding_natural_language_rank_sweep.py',
        PREFIX + 'phase1_kv_cache_sql_lookup_binding_natural_language_lr_sweep.py',
        PREFIX + 'phase1_kv_cache_sql_lookup_binding_natural_language_tuned_lr_seed_variance.py',
        PREFIX + 'phase1_kv_cache_sql_lookup_binding_document_form_matrix.py',
        PREFIX + 'phase1_kv_cache_multifact_2key_batched.py',
        PREFIX + 'phase1_kv_cache_compositionality_32token.py',
        'fasm/spikes/tensor_gpt2_handoff.h',
        'fasm/spikes/tensor_kv_handoff.c',
        'fasm/spikes/tensor_sha256.h',
        'fasm/spikes/tensor_transformer_executor_f32_native.c',
        'scripts/prepare_kv_handoff.py',
        'scripts/check_kv_handoff.sh',
        'scripts/build-kv-handoff-arm64.sh',
        'scripts/bench_kv_handoff_native_latency.py',
        'papers/kv-handoff/tost_equivalence.py',
        'scratchpad/kv_transfer_audit_20260913/sha256_isolation_bench.c',
        'scratchpad/kv_transfer_audit_20260913/native_amortized_latency.c',
        'scratchpad/kv_transfer_audit_20260913/native_gate.log',
    ]
    result['files'] = {path: sha(ROOT / path) for path in extras}
    log = (ROOT / extras[-1]).read_text()
    expected = 'PASS cases=32 correct=31 split=original_dev worst_abs=0.00103759766 atol=0.001 rtol=0.0001'
    if expected not in log or 'native KV handoff gate passed' not in log:
        raise ValueError('native evidence differs from manuscript result')
    result['native_summary'] = expected
    result['xor_design_audit'] = diagnose()
    result['tost_equivalence'] = tost_analyze()
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
