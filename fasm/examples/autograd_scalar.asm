; Scalar autograd demo: relu(a*b + c), including reverse-mode gradients.

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/autograd_f32.inc"

entry start

start:
	lea	rdi, [tape + 0 * AUTOGRAD_VALUE_SIZE]
	movss	xmm0, [value_a]
	call	autograd_leaf_f32
	lea	rdi, [tape + 1 * AUTOGRAD_VALUE_SIZE]
	movss	xmm0, [value_b]
	call	autograd_leaf_f32
	lea	rdi, [tape + 2 * AUTOGRAD_VALUE_SIZE]
	movss	xmm0, [value_c]
	call	autograd_leaf_f32
	lea	rdi, [tape + 3 * AUTOGRAD_VALUE_SIZE]
	lea	rsi, [tape + 0 * AUTOGRAD_VALUE_SIZE]
	lea	rdx, [tape + 1 * AUTOGRAD_VALUE_SIZE]
	call	autograd_mul_f32
	lea	rdi, [tape + 4 * AUTOGRAD_VALUE_SIZE]
	lea	rsi, [tape + 3 * AUTOGRAD_VALUE_SIZE]
	lea	rdx, [tape + 2 * AUTOGRAD_VALUE_SIZE]
	call	autograd_add_f32
	lea	rdi, [tape + 5 * AUTOGRAD_VALUE_SIZE]
	lea	rsi, [tape + 4 * AUTOGRAD_VALUE_SIZE]
	call	autograd_relu_f32

	lea	rdi, [tape]
	mov	rsi, 6
	lea	rdx, [tape + 5 * AUTOGRAD_VALUE_SIZE]
	call	autograd_backward_f32
	test	eax, eax
	jnz	.failed

	lea	rdi, [tape + 5 * AUTOGRAD_VALUE_SIZE + AUTOGRAD_VALUE_DATA]
	call	print_scaled
	lea	rdi, [tape + 0 * AUTOGRAD_VALUE_SIZE + AUTOGRAD_VALUE_GRAD]
	call	print_scaled
	lea	rdi, [tape + 1 * AUTOGRAD_VALUE_SIZE + AUTOGRAD_VALUE_GRAD]
	call	print_scaled
	lea	rdi, [tape + 2 * AUTOGRAD_VALUE_SIZE + AUTOGRAD_VALUE_GRAD]
	call	print_scaled_nl
	exit	EXIT_SUCCESS

.failed:
	exit	EXIT_FAILURE

; rdi = float pointer
print_scaled:
	movss	xmm0, dword [rdi]
	mulss	xmm0, [thousand]
	cvttss2si rax, xmm0
	jmp	print_int_sp

print_scaled_nl:
	movss	xmm0, dword [rdi]
	mulss	xmm0, [thousand]
	cvttss2si rax, xmm0
	jmp	print_int_nl

segment readable writeable

value_a  dd 2.0
value_b  dd -3.0
value_c  dd 10.0
thousand dd 1000.0
align 8
tape rb 6 * AUTOGRAD_VALUE_SIZE

include "fasm/core/runtime_bss.inc"
runtime_print_bss
