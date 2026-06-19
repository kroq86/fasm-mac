#pragma once

#include "chunker.hpp"
#include "manifest.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ragbox {

struct FileState {
    std::string hash;
    std::vector<std::uint64_t> doc_ids;
};

struct IndexState {
    static constexpr std::uint32_t kVersion = 1;

    std::uint32_t version{kVersion};
    std::uint64_t next_doc_id{};
    std::size_t chunk_size{};
    std::size_t overlap{};
    std::string model;
    std::string root;
    std::map<std::string, FileState> files;
    std::vector<std::uint64_t> superseded_doc_ids;
};

inline std::filesystem::path defaultDeltaPath(const std::filesystem::path& index_path) {
    return index_path.string() + ".delta";
}

inline std::filesystem::path defaultStatePath(const std::filesystem::path& index_path) {
    return index_path.string() + ".state.json";
}

inline std::string bytesToHex(const unsigned char* data, std::size_t len) {
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < len; ++i) {
        ss << std::setw(2) << static_cast<unsigned>(data[i]);
    }
    return ss.str();
}

inline std::string fileContentHash(const std::filesystem::path& path) {
    const std::string content = readTextFile(path, 16 * 1024 * 1024);
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(content.data(), content.size(), digest);
    return bytesToHex(digest, CC_SHA256_DIGEST_LENGTH);
}

inline std::map<std::string, std::string> collectFilesWithHashes(const std::filesystem::path& root) {
    const auto abs_root = std::filesystem::absolute(root);
    if (!std::filesystem::is_directory(abs_root)) {
        throw std::runtime_error("root is not a directory: " + abs_root.string());
    }
    std::vector<std::filesystem::path> files;
    collectFiles(abs_root, abs_root, files);
    std::sort(files.begin(), files.end());

    std::map<std::string, std::string> out;
    for (const auto& file_path : files) {
        const auto rel = std::filesystem::relative(file_path, abs_root).generic_string();
        out.emplace(rel, fileContentHash(file_path));
    }
    return out;
}

inline IndexState stateFromChunks(
    const std::vector<Chunk>& chunks,
    const std::filesystem::path& root,
    const std::string& model,
    std::size_t chunk_size,
    std::size_t overlap) {
    IndexState state{};
    state.next_doc_id = chunks.empty() ? 0 : chunks.back().doc_id + 1;
    state.chunk_size = chunk_size;
    state.overlap = overlap;
    state.model = model;
    state.root = std::filesystem::absolute(root).string();

    const auto abs_root = std::filesystem::absolute(root);
    std::map<std::string, std::vector<std::uint64_t>> ids_by_path;
    for (const auto& chunk : chunks) {
        ids_by_path[chunk.path].push_back(chunk.doc_id);
    }
    for (const auto& [path, doc_ids] : ids_by_path) {
        FileState fs{};
        fs.hash = fileContentHash(abs_root / path);
        fs.doc_ids = doc_ids;
        state.files.emplace(path, std::move(fs));
    }
    return state;
}

inline void writeStateAtomic(const std::filesystem::path& path, const IndexState& state) {
    const auto tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to write state: " + path.string());
        }
        out << "{\n";
        out << "  \"version\": " << state.version << ",\n";
        out << "  \"next_doc_id\": " << state.next_doc_id << ",\n";
        out << "  \"chunk_size\": " << state.chunk_size << ",\n";
        out << "  \"overlap\": " << state.overlap << ",\n";
        out << "  \"model\": \"" << jsonEscape(state.model) << "\",\n";
        out << "  \"root\": \"" << jsonEscape(state.root) << "\",\n";
        out << "  \"files\": {\n";
        std::size_t fi = 0;
        for (const auto& [path, file_state] : state.files) {
            out << "    \"" << jsonEscape(path) << "\": {\n";
            out << "      \"hash\": \"" << jsonEscape(file_state.hash) << "\",\n";
            out << "      \"doc_ids\": [";
            for (std::size_t i = 0; i < file_state.doc_ids.size(); ++i) {
                if (i > 0) {
                    out << ", ";
                }
                out << file_state.doc_ids[i];
            }
            out << "]\n";
            out << "    }";
            if (++fi < state.files.size()) {
                out << ',';
            }
            out << '\n';
        }
        out << "  },\n";
        out << "  \"superseded_doc_ids\": [";
        for (std::size_t i = 0; i < state.superseded_doc_ids.size(); ++i) {
            if (i > 0) {
                out << ", ";
            }
            out << state.superseded_doc_ids[i];
        }
        out << "]\n";
        out << "}\n";
    }
    std::filesystem::rename(tmp, path);
}

inline bool parseJsonUIntArray(const std::string& s, std::size_t& i, std::vector<std::uint64_t>& out) {
    out.clear();
    if (!expectChar(s, i, '[')) {
        return false;
    }
    skipWs(s, i);
    while (i < s.size() && s[i] != ']') {
        std::uint64_t v = 0;
        if (!parseJsonUInt(s, i, v)) {
            return false;
        }
        out.push_back(v);
        skipWs(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
        }
        skipWs(s, i);
    }
    return expectChar(s, i, ']');
}

inline IndexState loadState(const std::filesystem::path& path) {
    const std::string json = readWholeText(path);
    IndexState state{};

    std::size_t pos = 0;
    if (findKey(json, "version", 0, pos)) {
        pos += 10;
        std::uint64_t v = 0;
        if (!parseJsonUInt(json, pos, v)) {
            throw std::runtime_error("BadStateVersion");
        }
        state.version = static_cast<std::uint32_t>(v);
    }
    if (findKey(json, "next_doc_id", 0, pos)) {
        pos += 15;
        if (!parseJsonUInt(json, pos, state.next_doc_id)) {
            throw std::runtime_error("BadStateNextDocId");
        }
    }
    if (findKey(json, "chunk_size", 0, pos)) {
        pos += 13;
        std::uint64_t v = 0;
        if (!parseJsonUInt(json, pos, v)) {
            throw std::runtime_error("BadStateChunkSize");
        }
        state.chunk_size = static_cast<std::size_t>(v);
    }
    if (findKey(json, "overlap", 0, pos)) {
        pos += 10;
        std::uint64_t v = 0;
        if (!parseJsonUInt(json, pos, v)) {
            throw std::runtime_error("BadStateOverlap");
        }
        state.overlap = static_cast<std::size_t>(v);
    }
    if (findKey(json, "model", 0, pos)) {
        pos += 8;
        if (!parseJsonString(json, pos, state.model)) {
            throw std::runtime_error("BadStateModel");
        }
    }
    if (findKey(json, "root", 0, pos)) {
        pos += 7;
        if (!parseJsonString(json, pos, state.root)) {
            throw std::runtime_error("BadStateRoot");
        }
    }

    const std::string files_key = "\"files\"";
    std::size_t files_pos = json.find(files_key);
    if (files_pos != std::string::npos) {
        std::size_t i = files_pos + files_key.size();
        if (expectChar(json, i, ':') && expectChar(json, i, '{')) {
            skipWs(json, i);
            while (i < json.size() && json[i] != '}') {
                std::string path;
                if (!parseJsonString(json, i, path)) {
                    break;
                }
                if (!expectChar(json, i, ':') || !expectChar(json, i, '{')) {
                    throw std::runtime_error("BadStateFile");
                }
                FileState fs{};
                while (i < json.size() && json[i] != '}') {
                    std::string key;
                    if (!parseJsonString(json, i, key)) {
                        throw std::runtime_error("BadStateFileKey");
                    }
                    if (!expectChar(json, i, ':')) {
                        throw std::runtime_error("BadStateFileKey");
                    }
                    if (key == "hash") {
                        if (!parseJsonString(json, i, fs.hash)) {
                            throw std::runtime_error("BadStateHash");
                        }
                    } else if (key == "doc_ids") {
                        if (!parseJsonUIntArray(json, i, fs.doc_ids)) {
                            throw std::runtime_error("BadStateDocIds");
                        }
                    } else {
                        skipWs(json, i);
                        if (i < json.size() && json[i] == '"') {
                            std::string skip;
                            if (!parseJsonString(json, i, skip)) {
                                throw std::runtime_error("BadStateFile");
                            }
                        } else if (i < json.size() && json[i] == '[') {
                            std::vector<std::uint64_t> skip;
                            if (!parseJsonUIntArray(json, i, skip)) {
                                throw std::runtime_error("BadStateFile");
                            }
                        } else {
                            while (i < json.size() && json[i] != ',' && json[i] != '}') {
                                ++i;
                            }
                        }
                    }
                    skipWs(json, i);
                    if (i < json.size() && json[i] == ',') {
                        ++i;
                    }
                }
                expectChar(json, i, '}');
                state.files.emplace(std::move(path), std::move(fs));
                skipWs(json, i);
                if (i < json.size() && json[i] == ',') {
                    ++i;
                }
                skipWs(json, i);
            }
        }
    }

    if (findKey(json, "superseded_doc_ids", 0, pos)) {
        pos += 21;
        if (!parseJsonUIntArray(json, pos, state.superseded_doc_ids)) {
            throw std::runtime_error("BadStateSuperseded");
        }
    }

    if (state.version != IndexState::kVersion) {
        throw std::runtime_error("BadStateVersion");
    }
    return state;
}

inline std::set<std::uint64_t> supersededSet(const IndexState& state) {
    return {state.superseded_doc_ids.begin(), state.superseded_doc_ids.end()};
}

inline void supersedeDocIds(IndexState& state, const std::vector<std::uint64_t>& doc_ids) {
    for (const auto id : doc_ids) {
        if (std::find(state.superseded_doc_ids.begin(), state.superseded_doc_ids.end(), id) ==
            state.superseded_doc_ids.end()) {
            state.superseded_doc_ids.push_back(id);
        }
    }
}

inline void removeManifestPath(Manifest& manifest, const std::string& path) {
    manifest.records.erase(
        std::remove_if(
            manifest.records.begin(),
            manifest.records.end(),
            [&](const ManifestRecord& rec) { return rec.path == path; }),
        manifest.records.end());
}

struct RefreshPlan {
    std::vector<std::string> deleted;
    std::vector<std::string> changed;
    std::vector<std::string> added;
};

inline RefreshPlan planRefresh(const IndexState& state, const std::map<std::string, std::string>& current) {
    RefreshPlan plan{};
    for (const auto& [path, file_state] : state.files) {
        const auto it = current.find(path);
        if (it == current.end()) {
            plan.deleted.push_back(path);
        } else if (it->second != file_state.hash) {
            plan.changed.push_back(path);
        }
    }
    for (const auto& [path, hash] : current) {
        (void)hash;
        if (state.files.find(path) == state.files.end()) {
            plan.added.push_back(path);
        }
    }
    return plan;
}

}  // namespace ragbox
