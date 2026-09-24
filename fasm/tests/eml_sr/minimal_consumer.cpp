#include "adam.hpp"
#include "arena_search.hpp"
#include "pool_search.hpp"
#include "search.hpp"

#include <iostream>
#include <vector>

int main() {
    const std::vector<eml_sr::DataPoint> points = {
        {1.0, 2.0},
        {2.0, 5.0},
        {3.0, 10.0},
        {4.0, 17.0},
    };
    eml_sr::SearchOptions config{};
    const eml_sr::SearchResult result = eml_sr::search_best(points, 4, config, 0.0);
    std::cout << result.mse << '\n' << result.rpn << '\n';
    return result.rpn == "f f x eml eml x f eml eml" ? 0 : 1;
}
