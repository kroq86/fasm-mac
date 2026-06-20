#include "../../apps/eml_sr/eml_compiler.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Case {
    const char* name;
    const char* source;
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
    const auto compiled = eml_sr::compiler::compile_source(test.source);
    if (!compiled) {
        std::fprintf(stderr, "compile failed: %s (%s)\n", test.name, test.source);
        return false;
    }
    if (!eml_sr::compiler::is_pure_eml_source(*compiled)) {
        std::fprintf(stderr, "compile output is not pure EML: %s -> %s\n", test.source, compiled->c_str());
        return false;
    }
    if (!eml_sr::compiler::verify_compiled_identity(
            test.source,
            *compiled,
            test.envs,
            1e-9,
            test.skip_nonreal_target)) {
        std::fprintf(stderr, "compiled identity failed: %s\n  source:   %s\n  compiled: %s\n",
                     test.name, test.source, compiled->c_str());
        return false;
    }
    return true;
}

}  // namespace

int main() {
    using eml_sr::witness::Env;

    const std::vector<Env> no_env{{}};
    const std::vector<Env> unary = eml_sr::witness::unary_envs();
    const std::vector<Env> binary = binary_envs();

    const Case cases[] = {
        {"0", "0", no_env},
        {"-1", "-1", no_env},
        {"2", "2", no_env},
        {"E", "E", no_env},
        {"Pi", "Pi", no_env},
        {"Exp", "Exp[x]", unary, true},
        {"Log", "Log[x]", unary, true},
        {"Minus", "Minus[x]", unary, true},
        {"Inv", "Inv[x]", unary, true},
        {"Half", "Half[x]", unary, true},
        {"Sqr", "Sqr[x]", unary, true},
        {"Sqrt", "Sqrt[x]", unary, true},
        {"LogisticSigmoid", "LogisticSigmoid[x]", unary, true},
        {"Cosh", "Cosh[x]", unary, true},
        {"Sinh", "Sinh[x]", unary, true},
        {"Tanh", "Tanh[x]", unary, true},
        {"Cos", "Cos[x]", unary, true},
        {"Sin", "Sin[x]", unary, true},
        {"Tan", "Tan[x]", unary, true},
        {"ArcSinh", "ArcSinh[x]", unary, true},
        {"ArcCosh", "ArcCosh[x]", unary, true},
        {"ArcCos", "ArcCos[x]", unary, true},
        {"ArcTanh", "ArcTanh[x]", unary, true},
        {"ArcSin", "ArcSin[x]", unary, true},
        {"ArcTan", "ArcTan[x]", unary, true},
        {"Subtract", "Subtract[x, y]", binary},
        {"Plus", "Plus[x, y]", binary},
        {"Times", "Times[x, y]", binary},
        {"Divide", "Divide[x, y]", binary},
        {"Avg", "Avg[x, y]", binary},
        {"Power", "Power[x, y]", binary},
        {"LogBase", "Log[x, y]", binary},
        {"Hypot", "Hypot[x, y]", binary},
    };

    for (const Case& test : cases) {
        if (!run_case(test)) {
            return 1;
        }
    }

    std::puts("OK eml_sr compiler");
    return 0;
}
