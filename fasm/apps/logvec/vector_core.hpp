#pragma once

#include <cstdint>

namespace logvec {

inline constexpr std::uint32_t kDimMin = 1;
inline constexpr std::uint32_t kDimMax = 4096;
inline constexpr std::uint64_t kDocIdAuto = UINT64_MAX;

extern "C" {

float lb_vec_dot_f32(const float* a, const float* b, std::uint64_t len);
float lb_vec_norm_f32(const float* v, std::uint64_t len);
int lb_vec_topk_cosine_exact(
    const float* query,
    const float* vectors,
    const float* norms,
    std::uint64_t count,
    std::uint64_t dim,
    std::uint64_t k,
    std::uint32_t* out_index,
    float* out_score);
int lb_vec_topk_cosine_lv(
    const float* query,
    const std::uint8_t* records,
    std::uint64_t count,
    std::uint64_t dim,
    std::uint64_t k,
    std::uint64_t record_stride,
    std::uint32_t* out_index,
    float* out_score);
int lb_logvec_payload_validate(
    const std::uint8_t* payload,
    std::uint64_t len,
    std::uint32_t* out_dim,
    std::uint64_t* out_doc_id,
    const float** out_vector);
std::uint32_t lb_crc32c(const std::uint8_t* ptr, std::uint64_t len);

}  // extern "C"

}  // namespace logvec
