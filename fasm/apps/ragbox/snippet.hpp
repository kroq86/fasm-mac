#pragma once

#include "manifest.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace ragbox {

inline std::string truncateSnippet(const std::string& text, std::size_t max_len) {
    if (text.size() <= max_len) {
        return text;
    }
    return text.substr(0, max_len);
}

inline std::string loadSnippet(const Manifest& m, const ManifestRecord& rec, std::size_t max_len) {
    if (!rec.text.empty()) {
        return truncateSnippet(rec.text, max_len);
    }
    if (m.root.empty()) {
        return {};
    }
    const auto path = std::filesystem::path(m.root) / rec.path;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open snippet source: " + path.string());
    }
    in.seekg(static_cast<std::streamoff>(rec.offset));
    const std::size_t want = std::min(rec.length, max_len);
    std::string out(want, '\0');
    if (want > 0 && !in.read(out.data(), static_cast<std::streamsize>(want))) {
        throw std::runtime_error("failed to read snippet: " + path.string());
    }
    return out;
}

}  // namespace ragbox
