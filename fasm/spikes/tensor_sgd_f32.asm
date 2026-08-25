format ELF64
section '.text' executable
public tensor_parameters_sgd_f32
public tensor_parameters_zero_grad_f32
extrn tensor_shape_validate_f32
include 'fasm/core/tensor_sgd_f32.inc'
