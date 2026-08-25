#pragma once

#include <cstdint>
#include <iostream>
#include <string>

namespace eml_sr {

struct SearchStats {
    std::uint64_t forms_seen{0};
    std::uint64_t candidates_evaled{0};
    std::uint64_t eml_calls{0};
    std::uint64_t pruned_by_partial_mse{0};
    std::uint64_t pruned_by_numeric{0};
    std::uint64_t pruned_by_quick_mse{0};
    std::uint64_t cache_hits{0};
    std::uint64_t cache_misses{0};
    std::uint64_t best_update_count{0};

    void merge(const SearchStats& other) {
        forms_seen += other.forms_seen;
        candidates_evaled += other.candidates_evaled;
        eml_calls += other.eml_calls;
        pruned_by_partial_mse += other.pruned_by_partial_mse;
        pruned_by_numeric += other.pruned_by_numeric;
        pruned_by_quick_mse += other.pruned_by_quick_mse;
        cache_hits += other.cache_hits;
        cache_misses += other.cache_misses;
        best_update_count += other.best_update_count;
    }

    [[nodiscard]] std::uint64_t pruned_total() const {
        return pruned_by_partial_mse + pruned_by_numeric + pruned_by_quick_mse;
    }

    void print(std::ostream& out, double best_mse) const {
        out << "forms=" << forms_seen << ' '
            << "candidates=" << candidates_evaled << ' '
            << "eml_calls=" << eml_calls << ' '
            << "pruned=" << pruned_total() << ' '
            << "cache_hit=" << cache_hits << ' '
            << "best_mse=" << best_mse << '\n';
    }
};

}  // namespace eml_sr
