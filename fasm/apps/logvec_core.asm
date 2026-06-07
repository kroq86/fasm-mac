; C ABI exports for logvec FASM math + payload validation.

format ELF64

section ".text" executable

public lb_vec_dot_f32
public lb_vec_norm_f32
public lb_vec_topk_cosine_exact
public lb_logvec_payload_validate

include "fasm/core/vec_f32.inc"
include "fasm/core/logvec_payload.inc"
