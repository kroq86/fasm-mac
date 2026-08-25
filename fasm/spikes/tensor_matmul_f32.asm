format ELF64
section '.text' executable
public tensor_matmul_forward_f32
public tensor_matmul_backward_f32
public tensor_matmul_validate
include 'fasm/core/tensor_matmul_f32.inc'
