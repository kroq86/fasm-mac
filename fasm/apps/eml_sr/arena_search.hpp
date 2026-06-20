#pragma once

#include "expr_arena.hpp"
#include "search.hpp"

#include <functional>
#include <optional>

namespace eml_sr {

inline std::uint32_t arena_pow_u32(std::uint32_t base, std::uint32_t exp) {
    std::uint32_t out = 1;
    for (std::uint32_t i = 0; i < exp; ++i) {
        out *= base;
    }
    return out;
}

inline void gen_arena_shapes(
    ExprArena& arena,
    int eml_nodes,
    const std::function<void(ExprId)>& emit) {
    if (eml_nodes == 0) {
        emit(arena.intern_leaf(LeafKind::One));
        return;
    }
    for (int left_eml = 0; left_eml < eml_nodes; ++left_eml) {
        const int right_eml = eml_nodes - 1 - left_eml;
        gen_arena_shapes(arena, left_eml, [&](ExprId left) {
            gen_arena_shapes(arena, right_eml, [&](ExprId right) {
                emit(arena.intern_eml(left, right));
            });
        });
    }
}

inline ExprId instantiate_leaf_assignment(
    ExprArena& arena,
    ExprId shape,
    std::uint32_t assignment,
    std::uint32_t& leaf_index) {
    const ExprNode& n = arena.node(shape);
    if (n.kind == ExprKind::Leaf) {
        const std::uint32_t kind = (assignment / arena_pow_u32(3U, leaf_index)) % 3U;
        ++leaf_index;
        if (kind == 0U) {
            return arena.intern_leaf(LeafKind::One);
        }
        if (kind == 1U) {
            return arena.intern_leaf(LeafKind::X);
        }
        return arena.intern_leaf(LeafKind::F, 1.0);
    }
    const ExprId left = instantiate_leaf_assignment(arena, n.left, assignment, leaf_index);
    const ExprId right = instantiate_leaf_assignment(arena, n.right, assignment, leaf_index);
    return arena.intern_eml(left, right);
}

inline void collect_arena_f_values(
    const ExprArena& arena,
    ExprId id,
    std::vector<double>& out) {
    const ExprNode& n = arena.node(id);
    if (n.kind == ExprKind::Leaf) {
        if (n.leaf == LeafKind::F) {
            out.push_back(n.f_value);
        }
        return;
    }
    collect_arena_f_values(arena, n.left, out);
    collect_arena_f_values(arena, n.right, out);
}

inline ExprId with_arena_f_values(
    ExprArena& arena,
    ExprId id,
    const std::vector<double>& values,
    std::size_t& value_index) {
    const ExprNode& n = arena.node(id);
    if (n.kind == ExprKind::Leaf) {
        if (n.leaf == LeafKind::F) {
            const double value = values[value_index++];
            return arena.intern_leaf(LeafKind::F, value);
        }
        return arena.intern_leaf(n.leaf, n.f_value);
    }
    const ExprId left = with_arena_f_values(arena, n.left, values, value_index);
    const ExprId right = with_arena_f_values(arena, n.right, values, value_index);
    return arena.intern_eml(left, right);
}

inline ExprId with_arena_f_values(
    ExprArena& arena,
    ExprId id,
    const std::vector<double>& values) {
    std::size_t value_index = 0;
    return with_arena_f_values(arena, id, values, value_index);
}

inline double mse_for_expr_ctx(
    const ExprArena& arena,
    ExprId id,
    const std::vector<DataPoint>& data,
    EvalContext& ctx,
    ArenaEvalMemo& memo,
    double prune_above = std::numeric_limits<double>::infinity()) {
    if (id == kInvalidExprId || data.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    const double inv_n = 1.0 / static_cast<double>(data.size());
    double sum = 0.0;
    for (const DataPoint& pt : data) {
        const double pred = eval_expr_real(arena, id, pt.x, ctx, memo);
        if (!std::isfinite(pred)) {
            if (ctx.stats) {
                ++ctx.stats->pruned_by_numeric;
            }
            return std::numeric_limits<double>::infinity();
        }
        const double d = pred - pt.y;
        sum += d * d;
        if (sum * inv_n > prune_above) {
            if (ctx.stats) {
                ++ctx.stats->pruned_by_partial_mse;
            }
            return std::numeric_limits<double>::infinity();
        }
    }
    return sum * inv_n;
}

inline ExprId optimize_f_params_expr(
    ExprArena& arena,
    ExprId id,
    const std::vector<DataPoint>& data,
    EvalContext& ctx,
    ArenaEvalMemo& memo,
    double prune_above = std::numeric_limits<double>::infinity()) {
    std::vector<double> values;
    collect_arena_f_values(arena, id, values);
    if (values.empty()) {
        return id;
    }

    ExprId best_id = id;
    for (int pass = 0; pass < 8; ++pass) {
        bool improved = false;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const double start = values[i];
            double best_f = start;
            double best_mse = mse_for_expr_ctx(arena, best_id, data, ctx, memo, prune_above);
            const double probes[] = {
                start * 0.25, start * 0.5, start * 2.0, start * 4.0,
                start + 0.1,  start - 0.1,  start + 1.0, start - 1.0,
                0.5, 1.0, 2.0, 3.0, 4.0, 5.0,
            };
            for (const double trial_raw : probes) {
                const double trial = trial_raw < 1e-6 ? 1e-6 : trial_raw;
                std::vector<double> trial_values = values;
                trial_values[i] = trial;
                const ExprId trial_id = with_arena_f_values(arena, best_id, trial_values);
                const double trial_mse = mse_for_expr_ctx(arena, trial_id, data, ctx, memo, best_mse);
                if (trial_mse < best_mse) {
                    best_mse = trial_mse;
                    best_f = trial;
                    best_id = trial_id;
                    values = std::move(trial_values);
                    improved = true;
                    if (best_mse <= prune_above) {
                        prune_above = best_mse;
                    }
                }
            }
            values[i] = best_f;
        }
        if (!improved) {
            break;
        }
    }
    return best_id;
}

inline SearchResult arena_search_best(
    const std::vector<DataPoint>& data,
    int max_eml_nodes,
    const SearchOptions& opts,
    double goal_mse) {
    ExprArena arena{};
    EvalContext ctx{};
    ctx.domain = opts.domain;

    SearchResult best{};
    ctx.stats = opts.profile ? &best.stats : nullptr;
    ArenaEvalMemo memo{};

    for (int eml_nodes = 1; eml_nodes <= max_eml_nodes; ++eml_nodes) {
        gen_arena_shapes(arena, eml_nodes, [&](ExprId shape) {
            if (goal_mse > 0.0 && best.mse <= goal_mse) {
                return;
            }
            const ExprNode& shape_node = arena.node(shape);
            if (shape_node.leaf_count >= 13) {
                return;
            }
            ++best.stats.forms_seen;

            const std::uint32_t limit = arena_pow_u32(3U, shape_node.leaf_count);
            for (std::uint32_t assignment = 0; assignment < limit; ++assignment) {
                std::uint32_t leaf_index = 0;
                ExprId candidate = instantiate_leaf_assignment(arena, shape, assignment, leaf_index);
                candidate = optimize_f_params_expr(arena, candidate, data, ctx, memo, best.mse);
                ++best.stats.candidates_evaled;
                const double mse = mse_for_expr_ctx(arena, candidate, data, ctx, memo, best.mse);
                if (mse < best.mse) {
                    best.mse = mse;
                    best.rpn = arena.to_rpn(candidate);
                    best.tree = expr_to_tree(arena, candidate);
                    ++best.stats.best_update_count;
                    if (goal_mse > 0.0 && best.mse <= goal_mse) {
                        return;
                    }
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
