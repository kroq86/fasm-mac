#pragma once

#include "search.hpp"

#include <algorithm>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace eml_sr {

inline void try_update_best(SearchResult& best, const Tree& candidate, double mse) {
    if (mse < best.mse) {
        best.tree = candidate;
        best.mse = mse;
        best.rpn = to_rpn(candidate);
        ++best.stats.best_update_count;
    }
}

inline bool consider_candidate(
    SearchResult& best,
    Tree candidate,
    const std::vector<DataPoint>& data,
    EvalContext& ctx,
    const std::array<double, 4>& probe_x,
    std::unordered_map<std::uint64_t, double>& seen_sigs,
    std::vector<Tree>* pool_out = nullptr) {
    (void)probe_x;
    (void)seen_sigs;
    (void)pool_out;
    ++best.stats.candidates_evaled;

    optimize_f_params_ctx(candidate, data, ctx, best.mse);
    const double mse = mse_for_tree_ctx(candidate, data, ctx, best.mse);
    try_update_best(best, candidate, mse);
    return std::isfinite(mse);
}

inline SearchResult pool_search_best(
    const std::vector<DataPoint>& data,
    int max_eml_nodes,
    const SearchOptions& opts,
    double goal_mse) {
    SearchResult best{};
    EvalContext ctx{};
    ctx.domain = opts.domain;
    ctx.stats = opts.profile ? &best.stats : nullptr;

    const std::array<double, 4> probe_x = probe_x_values(data);
    std::unordered_map<std::uint64_t, double> seen_sigs;

    std::vector<std::vector<Tree>> pools(static_cast<std::size_t>(max_eml_nodes) + 1U);
    pools[0].push_back(leaf_tree(LeafKind::One));
    pools[0].push_back(leaf_tree(LeafKind::X));
    pools[0].push_back(leaf_tree(LeafKind::F));

    for (int eml_nodes = 1; eml_nodes <= max_eml_nodes; ++eml_nodes) {
        if (goal_mse > 0.0 && best.mse <= goal_mse) {
            break;
        }
        if (best.mse <= 0.0) {
            break;
        }

        std::vector<Tree> depth_pool;
        depth_pool.reserve(512);

        for (int left_eml = 0; left_eml < eml_nodes; ++left_eml) {
            const int right_eml = eml_nodes - 1 - left_eml;
            const auto& left_pool = pools[static_cast<std::size_t>(left_eml)];
            const auto& right_pool = pools[static_cast<std::size_t>(right_eml)];
            for (const Tree& left : left_pool) {
                for (const Tree& right : right_pool) {
                    ++best.stats.forms_seen;
                    Tree candidate = eml_tree(left, right);
                    consider_candidate(best, candidate, data, ctx, probe_x, seen_sigs, &depth_pool);
                    if (goal_mse > 0.0 && best.mse <= goal_mse) {
                        return best;
                    }
                }
            }
        }

        std::sort(depth_pool.begin(), depth_pool.end(), [](const Tree& a, const Tree& b) {
            return to_rpn(a) < to_rpn(b);
        });
        depth_pool.erase(
            std::unique(depth_pool.begin(), depth_pool.end(), [](const Tree& a, const Tree& b) {
                return to_rpn(a) == to_rpn(b);
            }),
            depth_pool.end());
        if (depth_pool.size() > 256) {
            depth_pool.resize(256);
        }
        pools[static_cast<std::size_t>(eml_nodes)] = std::move(depth_pool);

        gen_shapes(eml_nodes, [&](const Tree& shape) {
            std::vector<int> leaves;
            Tree base = shape;
            collect_leaf_indices(base, base.root, leaves);
            const std::size_t leaf_count = leaves.size();
            if (leaf_count >= 13) {
                return;
            }

            const std::uint32_t binary_limit = 1U << leaf_count;
#ifdef _OPENMP
            if (opts.jobs > 1) {
#pragma omp parallel if (opts.jobs > 1) num_threads(opts.jobs)
                {
                    EvalContext local_ctx = ctx;
                    SearchResult local_best = best;
                    std::unordered_map<std::uint64_t, double> local_seen = seen_sigs;
#pragma omp for schedule(dynamic)
                    for (std::int64_t mask = 0; mask < static_cast<std::int64_t>(binary_limit); ++mask) {
                        Tree candidate = base;
                        for (std::size_t i = 0; i < leaf_count; ++i) {
                            candidate.nodes[static_cast<std::size_t>(leaves[i])].leaf =
                                (static_cast<std::uint32_t>(mask) >> i) & 1U ? LeafKind::One : LeafKind::X;
                        }
                        consider_candidate(local_best, std::move(candidate), data, local_ctx, probe_x, local_seen);
                    }
#pragma omp critical
                    {
                        if (local_best.mse < best.mse) {
                            best = local_best;
                        }
                        for (const auto& [key, value] : local_seen) {
                            seen_sigs[key] = value;
                        }
                    }
                }
            } else
#endif
            {
                for (std::uint32_t mask = 0; mask < binary_limit; ++mask) {
                    Tree candidate = base;
                    for (std::size_t i = 0; i < leaf_count; ++i) {
                        candidate.nodes[static_cast<std::size_t>(leaves[i])].leaf =
                            (mask >> i) & 1U ? LeafKind::One : LeafKind::X;
                    }
                    consider_candidate(best, std::move(candidate), data, ctx, probe_x, seen_sigs);
                }
            }

            const std::uint32_t limit = static_cast<std::uint32_t>(std::pow(3, leaf_count));
            for (std::uint32_t assignment = 0; assignment < limit; ++assignment) {
                if (!assignment_has_f(assignment, leaf_count)) {
                    continue;
                }
                Tree candidate = with_leaf_assignment(base, assignment);
                consider_candidate(best, std::move(candidate), data, ctx, probe_x, seen_sigs);
            }
        });
    }

    return best;
}

inline SearchResult search_best_legacy(
    const std::vector<DataPoint>& data,
    int max_eml_nodes,
    const SearchOptions& opts,
    double goal_mse) {
    SearchResult best{};
    EvalContext ctx{};
    ctx.domain = opts.domain;
    ctx.stats = opts.profile ? &best.stats : nullptr;

    for (int eml_nodes = 1; eml_nodes <= max_eml_nodes; ++eml_nodes) {
        gen_shapes(eml_nodes, [&](const Tree& shape) {
            if (goal_mse > 0.0 && best.mse <= goal_mse) {
                return;
            }
            ++best.stats.forms_seen;

            std::vector<int> leaves;
            Tree base = shape;
            collect_leaf_indices(base, base.root, leaves);
            const std::size_t leaf_count = leaves.size();
            if (leaf_count >= 13) {
                return;
            }

            const std::uint32_t binary_limit = 1U << leaf_count;
            for (std::uint32_t mask = 0; mask < binary_limit; ++mask) {
                Tree candidate = base;
                for (std::size_t i = 0; i < leaf_count; ++i) {
                    candidate.nodes[static_cast<std::size_t>(leaves[i])].leaf =
                        (mask >> i) & 1U ? LeafKind::One : LeafKind::X;
                }
                ++best.stats.candidates_evaled;
                const double mse = mse_for_tree_ctx(candidate, data, ctx, best.mse);
                if (mse < best.mse) {
                    best.tree = candidate;
                    best.mse = mse;
                    best.rpn = to_rpn(candidate);
                    ++best.stats.best_update_count;
                }
            }

            const std::uint32_t limit = static_cast<std::uint32_t>(std::pow(3, leaf_count));
            for (std::uint32_t assignment = 0; assignment < limit; ++assignment) {
                if (!assignment_has_f(assignment, leaf_count)) {
                    continue;
                }
                Tree candidate = with_leaf_assignment(base, assignment);
                optimize_f_params_ctx(candidate, data, ctx, best.mse);
                ++best.stats.candidates_evaled;
                const double mse = mse_for_tree_ctx(candidate, data, ctx, best.mse);
                if (mse < best.mse) {
                    best.tree = candidate;
                    best.mse = mse;
                    best.rpn = to_rpn(candidate);
                    ++best.stats.best_update_count;
                }
            }
        });
        if (goal_mse > 0.0 && best.mse <= goal_mse) {
            break;
        }
    }
    return best;
}

}  // namespace eml_sr
