#pragma once

#include "../logvec/vector_index.hpp"
#include "manifest.hpp"
#include "state.hpp"
#include "ollama_client.hpp"

#include <cstdio>
#include <iostream>
#include <optional>
#include <string>
#include <sys/utsname.h>
#include <vector>

namespace ragbox {

struct DoctorOptions {
    std::filesystem::path index_path;
    std::filesystem::path manifest_path;
    std::string ollama_url = "http://127.0.0.1:11434";
    std::string model = "nomic-embed-text";
    bool skip_ollama = false;
};

inline int runDoctor(const DoctorOptions& opts) {
    int failures = 0;
    auto fail = [&](const std::string& msg) {
        std::cerr << "FAIL " << msg << '\n';
        ++failures;
    };
    auto ok = [&](const std::string& msg) {
        std::cout << "OK " << msg << '\n';
    };

    utsname un{};
    if (uname(&un) == 0) {
        const std::string machine = un.machine;
        if (machine != "x86_64") {
            std::cout << "NOTE running on " << machine << " — ragbox binary is x86_64; use arch -x86_64\n";
        } else {
            ok("x86_64 host");
        }
    }

    if (!opts.skip_ollama) {
        if (ollamaPing(opts.ollama_url)) {
            ok("ollama reachable");
        } else {
            fail("ollama unreachable at " + opts.ollama_url);
        }
        if (failures == 0 && ollamaHasModel(opts.ollama_url, opts.model)) {
            ok("model " + opts.model);
        } else if (failures == 0) {
            fail("model not found: " + opts.model);
        }
    } else {
        ok("ollama checks skipped");
    }

    if (!opts.index_path.empty()) {
        try {
            const logvec::VectorIndex index = logvec::VectorIndex::load(opts.index_path);
            if (index.count() == 0) {
                fail("index empty");
            } else {
                ok("index valid count=" + std::to_string(index.count()) + " dim=" + std::to_string(index.dim()));
            }

            const std::filesystem::path delta_path = defaultDeltaPath(opts.index_path);
            const std::filesystem::path state_path = defaultStatePath(opts.index_path);
            if (std::filesystem::exists(delta_path)) {
                const logvec::VectorIndex delta = logvec::VectorIndex::load(delta_path);
                ok("delta count=" + std::to_string(delta.count()) + " dim=" + std::to_string(delta.dim()));
            }
            if (std::filesystem::exists(state_path)) {
                const IndexState state = loadState(state_path);
                ok("state files=" + std::to_string(state.files.size()) + " superseded=" +
                   std::to_string(state.superseded_doc_ids.size()));
            }

            if (!opts.manifest_path.empty()) {
                const Manifest manifest = loadManifest(opts.manifest_path);
                if (std::filesystem::exists(state_path)) {
                    const IndexState state = loadState(state_path);
                    std::optional<logvec::VectorIndex> delta_index;
                    const logvec::VectorIndex* delta_ptr = nullptr;
                    if (std::filesystem::exists(delta_path)) {
                        delta_index = logvec::VectorIndex::load(delta_path);
                        delta_ptr = &*delta_index;
                    }
                    validateManifestSearch(manifest, index, delta_ptr, supersededSet(state));
                } else {
                    validateManifestIndex(manifest, index.dim(), index.count());
                }
                ok("manifest matches index");
            }
        } catch (const std::exception& e) {
            fail(std::string("index/manifest: ") + e.what());
        }
    }

    return failures == 0 ? 0 : 1;
}

}  // namespace ragbox
