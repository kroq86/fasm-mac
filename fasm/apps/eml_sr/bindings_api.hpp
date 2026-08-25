#pragma once

#include "adam.hpp"
#include "pool_search.hpp"
#include "arena_search.hpp"

namespace eml_sr {

struct FitConfig {
    int max_depth{3};
    SearchMethod method{SearchMethod::Enumerate};
    EvalDomain domain{EvalDomain::Complex};
    int jobs{1};
    bool profile{false};
    int adam_epochs{2000};
    double adam_lr{0.05};
};

inline SearchResult fit_api(const std::vector<DataPoint>& data, const FitConfig& config) {
    SearchOptions opts{};
    opts.domain = config.domain;
    opts.jobs = config.jobs;
    opts.profile = config.profile;
    opts.method = config.method;
    opts.adam_epochs = config.adam_epochs;
    opts.adam_lr = config.adam_lr;
    return search_best(data, config.max_depth, opts);
}

inline double predict_tree(const Tree& tree, double x) {
    EvalContext ctx{};
    return eval_real_ctx(tree, x, ctx);
}

inline std::string tree_to_dot(const Tree& tree) {
    return to_dot(tree, {});
}

}  // namespace eml_sr
