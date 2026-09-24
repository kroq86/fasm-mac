#include "adam.hpp"
#include "arena_search.hpp"
#include "pool_search.hpp"
#include "search.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL " << message << '\n';
    return 1;
}

eml_sr::SearchResult run(const std::vector<eml_sr::DataPoint>& data, int depth, eml_sr::MemoLifetime lifetime, double goal) {
    eml_sr::SearchOptions opts{};
    opts.profile = true;
    opts.memo_lifetime = lifetime;
    return eml_sr::search_best(data, depth, opts, goal);
}

}  // namespace

int main() {
    const std::vector<eml_sr::DataPoint> exp_points = {
        {0.1, std::exp(0.1)},
        {0.5, std::exp(0.5)},
        {1.0, std::exp(1.0)},
    };
    const std::vector<eml_sr::DataPoint> ln_points = {
        {1.0, 0.0},
        {2.0, std::log(2.0)},
        {3.0, std::log(3.0)},
        {std::exp(1.0), 1.0},
    };
    const std::vector<eml_sr::DataPoint> poly_points = {
        {1.0, 2.0},
        {2.0, 5.0},
        {3.0, 10.0},
        {4.0, 17.0},
    };

    const eml_sr::SearchResult exp_a = run(exp_points, 1, eml_sr::MemoLifetime::PerCandidate, 1e-9);
    const eml_sr::SearchResult exp_b = run(exp_points, 1, eml_sr::MemoLifetime::PerCandidate, 1e-9);
    const eml_sr::SearchResult exp_global = run(exp_points, 1, eml_sr::MemoLifetime::SearchGlobal, 1e-9);
    if (exp_a.rpn != "x 1 eml" || exp_b.rpn != exp_a.rpn || exp_global.rpn != exp_a.rpn) {
        return fail("exp rpn diverged: " + exp_a.rpn);
    }
    if (exp_a.mse > 1e-12 || exp_global.mse > 1e-12) {
        return fail("exp mse");
    }
    if (exp_a.stats.candidates_evaled != exp_global.stats.candidates_evaled) {
        return fail("exp candidate count differs across memo lifetime");
    }

    const eml_sr::SearchResult ln_per = run(ln_points, 3, eml_sr::MemoLifetime::PerCandidate, 1e-9);
    const eml_sr::SearchResult ln_per2 = run(ln_points, 3, eml_sr::MemoLifetime::PerCandidate, 1e-9);
    const eml_sr::SearchResult ln_global = run(ln_points, 3, eml_sr::MemoLifetime::SearchGlobal, 1e-9);
    if (ln_per.rpn != "1 1 x eml 1 eml eml" || ln_per2.rpn != ln_per.rpn || ln_global.rpn != ln_per.rpn) {
        return fail("ln rpn diverged: " + ln_per.rpn);
    }
    if (ln_per.mse > 1e-12 || ln_global.mse > 1e-12) {
        return fail("ln mse");
    }
    if (ln_per.stats.candidates_evaled != ln_global.stats.candidates_evaled) {
        return fail("ln candidate count differs across memo lifetime");
    }

    const eml_sr::SearchResult poly = run(poly_points, 4, eml_sr::MemoLifetime::PerCandidate, 0.0);
    const eml_sr::SearchResult poly2 = run(poly_points, 4, eml_sr::MemoLifetime::PerCandidate, 0.0);
    if (poly.rpn != "f f x eml eml x f eml eml" || poly2.rpn != poly.rpn) {
        return fail("poly rpn: " + poly.rpn);
    }
    if (std::abs(poly.mse - 0.156508) > 1e-4 || std::abs(poly2.mse - poly.mse) > 0.0) {
        return fail("poly mse");
    }
    if (poly.stats.candidates_evaled != poly2.stats.candidates_evaled || poly.stats.candidates_evaled == 0) {
        return fail("poly candidate count unstable");
    }
    if (poly.stats.max_memo_entries == 0 || poly.stats.max_memo_entries > 250000) {
        return fail("per-candidate memo unbounded: " + std::to_string(poly.stats.max_memo_entries));
    }
    if (ln_global.stats.max_memo_entries <= ln_per.stats.max_memo_entries) {
        return fail("global memo did not outgrow per-candidate memo on ln");
    }

    std::cout << "eml memo regression passed"
              << " poly_candidates=" << poly.stats.candidates_evaled
              << " poly_max_memo=" << poly.stats.max_memo_entries
              << " ln_per_max_memo=" << ln_per.stats.max_memo_entries
              << " ln_global_max_memo=" << ln_global.stats.max_memo_entries
              << '\n';
    return 0;
}
