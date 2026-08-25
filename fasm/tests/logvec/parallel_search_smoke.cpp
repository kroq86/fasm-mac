#include "../../apps/logvec/vector_index.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s FIXTURE.lv\n", argv[0]);
        return 1;
    }
    const std::filesystem::path index_path = argv[1];
    const logvec::VectorIndex index = logvec::VectorIndex::load(index_path);
    std::vector<float> query(index.dim(), 0.0f);
    query[0] = 1.0f;

    const auto serial = index.search(query, 2, 1);
    const auto t2 = index.search(query, 2, 2);
    const auto t4 = index.search(query, 2, 4);
    if (!logvec::sameHits(serial, t2) || !logvec::sameHits(serial, t4)) {
        std::fprintf(stderr, "FAIL parallel search mismatch\n");
        return 1;
    }
    std::printf("OK parallel_search_smoke count=%llu threads=2,4\n",
                static_cast<unsigned long long>(index.count()));
    return 0;
}
