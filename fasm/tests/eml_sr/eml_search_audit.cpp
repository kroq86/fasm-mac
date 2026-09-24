#include "adam.hpp"
#include "arena_search.hpp"
#include "pool_search.hpp"
#include "search.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

std::vector<eml_sr::DataPoint> points_for(const std::string& name) {
    if (name == "exp") {
        return {
            {0.1, std::exp(0.1)},
            {0.5, std::exp(0.5)},
            {1.0, std::exp(1.0)},
        };
    }
    if (name == "ln") {
        return {
            {1.0, 0.0},
            {2.0, std::log(2.0)},
            {3.0, std::log(3.0)},
            {std::exp(1.0), 1.0},
        };
    }
    if (name == "poly") {
        return {
            {1.0, 2.0},
            {2.0, 5.0},
            {3.0, 10.0},
            {4.0, 17.0},
        };
    }
    throw std::runtime_error("bad case");
}

double goal_for(const std::string& name) {
    return name == "poly" ? 0.0 : 1e-9;
}

std::uint64_t rss_bytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    std::string memo = "per";
    std::string name = "poly";
    int depth = 4;
    int repeats = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--memo" && i + 1 < argc) {
            memo = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            name = argv[++i];
        } else if (arg == "--depth" && i + 1 < argc) {
            depth = std::stoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::stoi(argv[++i]);
        } else {
            std::cerr << "usage: eml_search_audit --memo per|global --case exp|ln|poly --depth N --repeats N\n";
            return 2;
        }
    }
    if (depth < 1 || repeats < 1) {
        return 2;
    }

    const std::vector<eml_sr::DataPoint> data = points_for(name);
    const double goal = goal_for(name);
    eml_sr::SearchOptions opts{};
    opts.profile = true;
    opts.method = eml_sr::SearchMethod::Enumerate;
    opts.memo_lifetime = memo == "global" ? eml_sr::MemoLifetime::SearchGlobal : eml_sr::MemoLifetime::PerCandidate;

    for (int rep = 0; rep < repeats; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        const eml_sr::SearchResult result = eml_sr::search_best(data, depth, opts, goal);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "memo=" << (opts.memo_lifetime == eml_sr::MemoLifetime::SearchGlobal ? "global" : "per")
                  << " case=" << name
                  << " depth=" << depth
                  << " rep=" << rep
                  << " ms=" << ms
                  << " candidates=" << result.stats.candidates_evaled
                  << " forms=" << result.stats.forms_seen
                  << " eml_calls=" << result.stats.eml_calls
                  << " hits=" << result.stats.cache_hits
                  << " misses=" << result.stats.cache_misses
                  << " max_memo=" << result.stats.max_memo_entries
                  << " rss=" << rss_bytes()
                  << " mse=" << result.mse
                  << " rpn=" << result.rpn
                  << '\n';
        std::cout.flush();
    }
    return 0;
}
