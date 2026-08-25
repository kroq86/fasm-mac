; Two-layer MLP: affine -> ReLU -> affine.

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/mlp_f32.inc"

entry start

start:
	lea	rdi, [nn_input]
	lea	rsi, [weights_1]
	lea	rdx, [bias_1]
	lea	rcx, [hidden]
	mov	r8, 2
	mov	r9, 2
	call	mlp_dense_f32
	test	eax, eax
	jnz	.failed

	lea	rdi, [hidden]
	mov	rsi, 2
	call	mlp_relu_f32_inplace
	test	eax, eax
	jnz	.failed

	lea	rdi, [hidden]
	lea	rsi, [weights_2]
	lea	rdx, [bias_2]
	lea	rcx, [result]
	mov	r8, 2
	mov	r9, 2
	call	mlp_dense_f32
	test	eax, eax
	jnz	.failed

	movss	xmm0, [result]
	mulss	xmm0, [thousand]
	cvttss2si rax, xmm0
	call	print_int_sp
	movss	xmm0, [result + 4]
	mulss	xmm0, [thousand]
	cvttss2si rax, xmm0
	call	print_int_nl
	exit	EXIT_SUCCESS

.failed:
	exit	EXIT_FAILURE

segment readable writeable

nn_input  dd 1.0, 2.0
weights_1 dd 1.0, -1.0
	      dd 0.5, 0.25
bias_1    dd 0.0, 0.5
weights_2 dd -1.0, 1.0
	      dd 1.0, 1.0
bias_2    dd 0.25, -0.25
thousand  dd 1000.0
hidden    rd 2
result    rd 2

include "fasm/core/runtime_bss.inc"
runtime_print_bss
