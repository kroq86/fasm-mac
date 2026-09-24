#include "../../apps/eml_sr/arena_search.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "arena smoke failed: %s\n", message);
        return false;
    }
    return true;
}

bool near(double a, double b) {
    return std::abs(a - b) <= 1e-9 * (1.0 + std::abs(a) + std::abs(b));
}

}  // namespace

int main() {
    eml_sr::ExprArena arena{};
    const eml_sr::ExprId x1 = arena.intern_leaf(eml_sr::LeafKind::X);
    const eml_sr::ExprId x2 = arena.intern_leaf(eml_sr::LeafKind::X);
    const eml_sr::ExprId one = arena.intern_leaf(eml_sr::LeafKind::One);
    const eml_sr::ExprId exp1 = arena.intern_eml(x1, one);
    const eml_sr::ExprId exp2 = arena.intern_eml(x2, one);

    if (!check(x1 == x2, "identical leaves must share id")) {
        return 1;
    }
    if (!check(exp1 == exp2, "identical eml subtrees must share id")) {
        return 1;
    }
    if (!check(arena.size() == 3, "arena should contain x, 1, and eml(x,1)")) {
        return 1;
    }
    if (!check(arena.to_rpn(exp1) == "x 1 eml", "rpn for exp tree")) {
        return 1;
    }

    const std::size_t stable_size = arena.size();
    std::string scoped_rpn;
    eml_sr::Tree scoped_tree;
    eml_sr::ExprId scoped_f = eml_sr::kInvalidExprId;
    eml_sr::ExprId scoped_expr = eml_sr::kInvalidExprId;
    {
        eml_sr::ExprArenaScope scope(arena);
        scoped_f = arena.intern_leaf(eml_sr::LeafKind::F, 2.0);
        scoped_expr = arena.intern_eml(exp1, scoped_f);
        scoped_rpn = arena.to_rpn(scoped_expr);
        scoped_tree = eml_sr::expr_to_tree(arena, scoped_expr);
        if (!check(arena.size() == stable_size + 2, "scope should add temporary nodes")) {
            return 1;
        }
    }
    if (!check(arena.size() == stable_size, "scope destructor should rollback temporary nodes")) {
        return 1;
    }
    if (!check(scoped_rpn == "x 1 eml f eml", "copied scoped rpn survives rollback")) {
        return 1;
    }
    if (!check(eml_sr::to_rpn(scoped_tree) == "x 1 eml f eml", "copied scoped tree survives rollback")) {
        return 1;
    }
    const eml_sr::ExprId scoped_f_again = arena.intern_leaf(eml_sr::LeafKind::F, 2.0);
    const eml_sr::ExprId scoped_expr_again = arena.intern_eml(exp1, scoped_f_again);
    if (!check(scoped_f_again == scoped_f && scoped_expr_again == scoped_expr, "rollback should remove keys for canonical reuse")) {
        return 1;
    }

    eml_sr::EvalContext ctx{};
    eml_sr::ArenaEvalMemo memo{};
    const double got = eml_sr::eval_expr_real(arena, exp1, 0.5, ctx, memo);
    if (!check(near(got, std::exp(0.5)), "arena eval exp")) {
        return 1;
    }

    const eml_sr::Tree tree = eml_sr::expr_to_tree(arena, exp1);
    if (!check(eml_sr::to_rpn(tree) == "x 1 eml", "expr to tree rpn")) {
        return 1;
    }
    eml_sr::ExprArena arena2{};
    const eml_sr::ExprId roundtrip = eml_sr::tree_to_expr(tree, arena2);
    if (!check(arena2.to_rpn(roundtrip) == "x 1 eml", "tree to expr rpn")) {
        return 1;
    }

    eml_sr::SearchOptions opts{};
    opts.hall_of_fame_size = 3;
    const std::vector<eml_sr::DataPoint> exp_data{{0.1, std::exp(0.1)}, {0.5, std::exp(0.5)}, {1.0, std::exp(1.0)}};
    const eml_sr::SearchResult exp_result = eml_sr::arena_search_best(exp_data, 1, opts, 1e-9);
    if (!check(exp_result.mse <= 1e-9 && exp_result.rpn == "x 1 eml", "arena search exp parity")) {
        return 1;
    }
    if (!check(!exp_result.equations.empty() && exp_result.equations.size() <= opts.hall_of_fame_size, "arena hof size")) {
        return 1;
    }
    if (!check(exp_result.equations[0].rpn == exp_result.rpn && near(exp_result.equations[0].mse, exp_result.mse), "arena hof best row")) {
        return 1;
    }

    const std::vector<eml_sr::DataPoint> ln_data{{1.0, 0.0}, {2.0, std::log(2.0)}, {3.0, std::log(3.0)}, {std::exp(1.0), 1.0}};
    const eml_sr::SearchResult ln_result = eml_sr::arena_search_best(ln_data, 3, opts, 1e-9);
    if (!check(ln_result.mse <= 1e-9 && ln_result.rpn == "1 1 x eml 1 eml eml", "arena search ln parity")) {
        return 1;
    }

    std::puts("OK eml_sr expr arena");
    return 0;
}
