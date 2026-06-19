#pragma once

#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace logvec {

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path, std::size_t limit) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("failed to open file: " + path.string());
        }
        struct stat st {};
        if (::fstat(fd_, &st) != 0) {
            closeFd();
            throw std::runtime_error("failed to stat file: " + path.string());
        }
        if (st.st_size < 0) {
            closeFd();
            throw std::runtime_error("invalid file size: " + path.string());
        }
        size_ = static_cast<std::size_t>(st.st_size);
        if (size_ > limit) {
            closeFd();
            throw std::runtime_error("file too large: " + path.string());
        }
        if (size_ == 0) {
            return;
        }
        base_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (base_ == MAP_FAILED) {
            base_ = nullptr;
            closeFd();
            throw std::runtime_error("mmap failed: " + path.string());
        }
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    ~MappedFile() {
        if (base_ != nullptr) {
            ::munmap(base_, size_);
        }
        closeFd();
    }

    [[nodiscard]] std::span<const std::uint8_t> bytes() const {
        if (size_ == 0) {
            return {};
        }
        return {static_cast<const std::uint8_t*>(base_), size_};
    }

private:
    void closeFd() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    int fd_{-1};
    void* base_{nullptr};
    std::size_t size_{};
};

}  // namespace logvec
