#!/usr/bin/env python3
"""TOST-style equivalence reframing of the already-computed scale/noise
sensitivity checks for the ordering adapter.

Read-only, standard-library reanalysis: no model inference, no new bootstrap
resampling. It reads the bootstrap 95% CIs already stored in
phase1_kv_cache_minigunpoint_32token_ablations_result.json for the
scale-mismatch and noise-robustness conditions and reports, for each, the
smallest symmetric margin m such that the stored CI lies within [-m, m] --
the margin at which a two-one-sided-test (TOST) equivalence claim would be
licensed by this already-computed interval. This follows Cheng et al.'s
margin-sensitivity-ladder presentation (arXiv:2608.04893, Appendix A)
without adopting their specific anchor (their margin comes from a system's
own claimed aggregate gain; no such figure exists here to anchor to), so
this reports the ladder itself rather than asserting one preferred margin.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABLATIONS = ROOT / 'scratchpad/gpt2_distilgpt2_latent_bridge/phase1_kv_cache_minigunpoint_32token_ablations_result.json'


def smallest_passing_margin(ci):
    lo, hi = ci
    return max(abs(lo), abs(hi))


def analyze():
    data = json.loads(ABLATIONS.read_text())
    baseline_nll = data['baseline_correct']['nll']
    conditions = {
        'scale_0.5_vs_1.0': data['scale_mismatch']['scale_0.5_vs_1.0_ci'],
        'scale_2.0_vs_1.0': data['scale_mismatch']['scale_2.0_vs_1.0_ci'],
        'noise_plus_10pct_vs_clean': data['noise_robustness']['noisy_vs_clean_ci'],
    }
    result = {
        'schema': 1,
        'kind': 'read-only TOST reframing of stored bootstrap CIs; no new model inference or resampling',
        'baseline_correct_nll': baseline_nll,
        'conditions': {},
    }
    for name, entry in conditions.items():
        ci = entry['bootstrap_95pct_ci']
        margin = smallest_passing_margin(ci)
        result['conditions'][name] = {
            'other_minus_base_nll': entry['other_minus_base_nll'],
            'bootstrap_95pct_ci': ci,
            'smallest_passing_margin_nll': margin,
            'margin_as_fraction_of_baseline_nll': margin / baseline_nll,
        }
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(), indent=2, sort_keys=True))
