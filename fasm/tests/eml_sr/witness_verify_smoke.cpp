#include "../../apps/eml_sr/witness_verify.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Case {
    const char* name;
    const char* target;
    const char* witness;
    std::vector<eml_sr::witness::Env> envs;
    bool skip_nonreal_target{false};
};

std::vector<eml_sr::witness::Env> binary_envs() {
    std::vector<eml_sr::witness::Env> out;
    for (const auto& pair : eml_sr::witness::anchor_pairs()) {
        out.push_back(eml_sr::witness::binary_env(pair.x, pair.y));
    }
    return out;
}

bool run_case(const Case& test) {
    if (eml_sr::witness::verify_identity(
            test.target,
            test.witness,
            test.envs,
            1e-9,
            test.skip_nonreal_target)) {
        return true;
    }
    std::fprintf(
        stderr,
        "witness failed: %s\n  target:  %s\n  witness: %s\n",
        test.name,
        test.target,
        test.witness);
    return false;
}

}  // namespace

int main() {
    using eml_sr::witness::Env;

    const std::vector<Env> no_env{{}};
    const std::vector<Env> unary = eml_sr::witness::unary_envs();
    const std::vector<Env> binary = binary_envs();

    const Case cases[] = {
        {"E", "E", "EML[1, 1]", no_env},
        {"Exp", "Exp[x]", "EML[x, 1]", unary},
        {"Log", "Log[x]", "EML[1, Exp[EML[1, x]]]", unary},
        {"Subtract", "Subtract[x, y]", "EML[Log[x], Exp[y]]", binary},
        {"-1", "-1", "Subtract[Log[1], 1]", no_env},
        {"2", "2", "Subtract[1, -1]", no_env},
        {"Minus", "Minus[x]", "Subtract[Log[1], x]", unary},
        {"Plus", "Plus[x, y]", "Subtract[x, Minus[y]]", binary},
        {"Inv", "Inv[x]", "Exp[Minus[Log[x]]]", unary},
        {"Times", "Times[x, y]", "Exp[Plus[Log[x], Log[y]]]", binary},
        {"Sqr", "Sqr[x]", "Times[x, x]", unary},
        {"Divide", "Divide[x, y]", "Times[x, Inv[y]]", binary},
        {"Half", "Half[x]", "Divide[x, 2]", unary},
        {"Avg", "Avg[x, y]", "Half[Plus[x, y]]", binary},
        {"Sqrt", "Sqrt[x]", "Exp[Half[Log[x]]]", unary},
        {"Power", "Power[x, y]", "Exp[Times[y, Log[x]]]", binary},
        {"LogBase", "Log[x, y]", "Divide[Log[y], Log[x]]", binary},
        {"Pi", "Pi", "Sqrt[Minus[Sqr[Log[-1]]]]", no_env},
        {"Hypot", "Hypot[x, y]", "Sqrt[Plus[Sqr[x], Sqr[y]]]", binary},
        {"LogisticSigmoid", "LogisticSigmoid[x]", "Inv[EML[Minus[x], Exp[-1]]]", unary, true},
        {"Cosh", "Cosh[x]", "Avg[Exp[x], Exp[Minus[x]]]", unary, true},
        {"Sinh", "Sinh[x]", "EML[x, Exp[Cosh[x]]]", unary, true},
        {"Tanh", "Tanh[x]", "Divide[Sinh[x], Cosh[x]]", unary, true},
        {"Cos", "Cos[x]", "Cosh[Sqrt[Minus[Sqr[x]]]]", unary, true},
        {"Sin", "Sin[x]", "Cos[Subtract[x, Half[Pi]]]", unary, true},
        {"Tan", "Tan[x]", "Divide[Sin[x], Cos[x]]", unary, true},
        {"ArcSinh", "ArcSinh[x]", "Log[Plus[x, Hypot[-1, x]]]", unary, true},
        {"ArcCosh", "ArcCosh[x]", "ArcSinh[Hypot[x, Sqrt[-1]]]", unary, true},
        {"ArcCos", "ArcCos[x]", "ArcCosh[Cos[ArcCosh[x]]]", unary, true},
        {"ArcTanh", "ArcTanh[x]", "ArcSinh[Inv[Tan[ArcCos[x]]]]", unary, true},
        {"ArcSin", "ArcSin[x]", "Subtract[Half[Pi], ArcCos[x]]", unary, true},
        {"ArcTan", "ArcTan[x]", "ArcSin[Tanh[ArcSinh[x]]]", unary, true},
    };

    for (const Case& test : cases) {
        if (!run_case(test)) {
            return 1;
        }
    }

    std::puts("OK eml_sr witness verifier");
    return 0;
}
