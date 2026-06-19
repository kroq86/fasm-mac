#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ragbox {

struct Chunk {
    std::uint64_t doc_id{};
    std::string path;
    std::size_t offset{};
    std::string text;
};

inline bool isIncludedExtension(const std::filesystem::path& path) {
    static const char* kExts[] = {
        ".md",  ".txt",  ".jsonl", ".py",  ".go",   ".rs",  ".js",  ".ts",
        ".cpp", ".hpp",  ".c",     ".h",   ".zig",  ".asm", ".sh",
    };
    const auto ext = path.extension().string();
    for (const char* candidate : kExts) {
        if (ext == candidate) {
            return true;
        }
    }
    return false;
}

inline bool shouldSkipDir(const std::filesystem::path& name) {
    const auto s = name.filename().string();
    if (s.empty() || s[0] == '.') {
        return true;
    }
    return s == "node_modules" || s == "__pycache__";
}

inline std::string readTextFile(const std::filesystem::path& path, std::size_t limit) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(in.tellg());
    if (size > limit) {
        throw std::runtime_error("file too large: " + path.string());
    }
    in.seekg(0, std::ios::beg);
    std::string data(size, '\0');
    if (size > 0 && !in.read(data.data(), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("failed to read file: " + path.string());
    }
    return data;
}

inline void splitFile(
    const std::filesystem::path& root,
    const std::filesystem::path& file_path,
    std::size_t chunk_size,
    std::size_t overlap,
    std::uint64_t& next_doc_id,
    std::vector<Chunk>& out) {
    const auto rel = std::filesystem::relative(file_path, root).generic_string();
    const std::string content = readTextFile(file_path, 16 * 1024 * 1024);
    if (content.empty()) {
        return;
    }
    if (chunk_size == 0) {
        throw std::runtime_error("chunk_size must be > 0");
    }
    if (overlap >= chunk_size) {
        throw std::runtime_error("overlap must be < chunk_size");
    }
    const std::size_t step = chunk_size - overlap;
    for (std::size_t offset = 0; offset < content.size(); offset += step) {
        const std::size_t len = std::min(chunk_size, content.size() - offset);
        Chunk chunk{};
        chunk.doc_id = next_doc_id++;
        chunk.path = rel;
        chunk.offset = offset;
        chunk.text = content.substr(offset, len);
        out.push_back(std::move(chunk));
        if (offset + len >= content.size()) {
            break;
        }
    }
}

inline void collectFiles(
    const std::filesystem::path& dir,
    const std::filesystem::path& root,
    std::vector<std::filesystem::path>& files) {
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_directory()) {
            if (shouldSkipDir(entry.path())) {
                continue;
            }
            collectFiles(entry.path(), root, files);
            continue;
        }
        if (!entry.is_regular_file()) {
            continue;
        }
        if (!isIncludedExtension(entry.path())) {
            continue;
        }
        files.push_back(entry.path());
    }
}

inline std::vector<Chunk> chunkDirectory(
    const std::filesystem::path& root,
    std::size_t chunk_size,
    std::size_t overlap) {
    const auto abs_root = std::filesystem::absolute(root);
    if (!std::filesystem::is_directory(abs_root)) {
        throw std::runtime_error("root is not a directory: " + abs_root.string());
    }

    std::vector<std::filesystem::path> files;
    collectFiles(abs_root, abs_root, files);

    std::sort(files.begin(), files.end());
    std::vector<Chunk> chunks;
    std::uint64_t next_doc_id = 0;
    for (const auto& file_path : files) {
        splitFile(abs_root, file_path, chunk_size, overlap, next_doc_id, chunks);
    }
    return chunks;
}

}  // namespace ragbox
