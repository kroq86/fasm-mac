; Experimental scalar reverse-mode automatic differentiation for f32 values.
;
; Nodes live in a caller-owned tape in forward/topological order. Operations
; store parent pointers; autograd_backward_f32 walks the tape in reverse.
; This is a small-expression correctness/reference spike, not the storage model
; for tensor training. See docs/autograd-spikes.md for measured constraints.
;
; AutogradValue layout (32 bytes):
;   +0  dd data
;   +4  dd grad
;   +8  dq lhs (0 for leaf)
;   +16 dq rhs (0 for unary/leaf)
;   +24 db op, 7 bytes reserved
;
; C ABI:
;   void autograd_leaf_f32(Value *out, float data)
;     rdi=out, xmm0=data
;   void autograd_add_f32(Value *out, Value *lhs, Value *rhs)
;     rdi=out, rsi=lhs, rdx=rhs
;   void autograd_mul_f32(Value *out, Value *lhs, Value *rhs)
;     rdi=out, rsi=lhs, rdx=rhs
;   void autograd_relu_f32(Value *out, Value *input)
;     rdi=out, rsi=input
;   int autograd_backward_f32(Value *tape, uint64_t count, Value *output)
;     rdi=tape, rsi=count, rdx=output
;     eax=0 on success, -1 for null pointers or zero count
;
; All routines preserve callee-saved registers. The backward pass accumulates
; gradients, so shared nodes and repeated operands work as expected.

AUTOGRAD_VALUE_DATA equ 0
AUTOGRAD_VALUE_GRAD equ 4
AUTOGRAD_VALUE_LHS  equ 8
AUTOGRAD_VALUE_RHS  equ 16
AUTOGRAD_VALUE_OP   equ 24
AUTOGRAD_VALUE_SIZE equ 32

AUTOGRAD_OP_LEAF equ 0
AUTOGRAD_OP_ADD  equ 1
AUTOGRAD_OP_MUL  equ 2
AUTOGRAD_OP_RELU equ 3

macro autograd_clear_node node
{
	mov	dword [node + AUTOGRAD_VALUE_GRAD], 0
	mov	qword [node + AUTOGRAD_VALUE_LHS], 0
	mov	qword [node + AUTOGRAD_VALUE_RHS], 0
	mov	qword [node + AUTOGRAD_VALUE_OP], 0
}

autograd_leaf_f32:
	movss	dword [rdi + AUTOGRAD_VALUE_DATA], xmm0
	autograd_clear_node rdi
	ret

autograd_add_f32:
	movss	xmm0, dword [rsi + AUTOGRAD_VALUE_DATA]
	addss	xmm0, dword [rdx + AUTOGRAD_VALUE_DATA]
	movss	dword [rdi + AUTOGRAD_VALUE_DATA], xmm0
	autograd_clear_node rdi
	mov	qword [rdi + AUTOGRAD_VALUE_LHS], rsi
	mov	qword [rdi + AUTOGRAD_VALUE_RHS], rdx
	mov	byte [rdi + AUTOGRAD_VALUE_OP], AUTOGRAD_OP_ADD
	ret

autograd_mul_f32:
	movss	xmm0, dword [rsi + AUTOGRAD_VALUE_DATA]
	mulss	xmm0, dword [rdx + AUTOGRAD_VALUE_DATA]
	movss	dword [rdi + AUTOGRAD_VALUE_DATA], xmm0
	autograd_clear_node rdi
	mov	qword [rdi + AUTOGRAD_VALUE_LHS], rsi
	mov	qword [rdi + AUTOGRAD_VALUE_RHS], rdx
	mov	byte [rdi + AUTOGRAD_VALUE_OP], AUTOGRAD_OP_MUL
	ret

autograd_relu_f32:
	movss	xmm0, dword [rsi + AUTOGRAD_VALUE_DATA]
	xorps	xmm1, xmm1
	maxss	xmm0, xmm1
	movss	dword [rdi + AUTOGRAD_VALUE_DATA], xmm0
	autograd_clear_node rdi
	mov	qword [rdi + AUTOGRAD_VALUE_LHS], rsi
	mov	byte [rdi + AUTOGRAD_VALUE_OP], AUTOGRAD_OP_RELU
	ret

autograd_backward_f32:
	test	rdi, rdi
	jz	.abf_bad
	test	rsi, rsi
	jz	.abf_bad
	test	rdx, rdx
	jz	.abf_bad
	push	rbx
	push	r12
	mov	r12, rdi
	mov	rbx, rsi
	xor	rax, rax
.abf_zero:
	mov	dword [r12 + rax + AUTOGRAD_VALUE_GRAD], 0
	add	rax, AUTOGRAD_VALUE_SIZE
	dec	rsi
	jnz	.abf_zero
	movss	xmm0, [_autograd_f32_one]
	movss	dword [rdx + AUTOGRAD_VALUE_GRAD], xmm0
	mov	rax, rbx
	shl	rax, 5
	lea	r10, [r12 + rax - AUTOGRAD_VALUE_SIZE]
.abf_node:
	movzx	eax, byte [r10 + AUTOGRAD_VALUE_OP]
	cmp	eax, AUTOGRAD_OP_ADD
	je	.abf_add
	cmp	eax, AUTOGRAD_OP_MUL
	je	.abf_mul
	cmp	eax, AUTOGRAD_OP_RELU
	je	.abf_relu
	jmp	.abf_next
.abf_add:
	movss	xmm0, dword [r10 + AUTOGRAD_VALUE_GRAD]
	mov	r8, qword [r10 + AUTOGRAD_VALUE_LHS]
	addss	xmm0, dword [r8 + AUTOGRAD_VALUE_GRAD]
	movss	dword [r8 + AUTOGRAD_VALUE_GRAD], xmm0
	movss	xmm0, dword [r10 + AUTOGRAD_VALUE_GRAD]
	mov	r8, qword [r10 + AUTOGRAD_VALUE_RHS]
	addss	xmm0, dword [r8 + AUTOGRAD_VALUE_GRAD]
	movss	dword [r8 + AUTOGRAD_VALUE_GRAD], xmm0
	jmp	.abf_next
.abf_mul:
	movss	xmm0, dword [r10 + AUTOGRAD_VALUE_GRAD]
	mov	r8, qword [r10 + AUTOGRAD_VALUE_RHS]
	mulss	xmm0, dword [r8 + AUTOGRAD_VALUE_DATA]
	mov	r9, qword [r10 + AUTOGRAD_VALUE_LHS]
	addss	xmm0, dword [r9 + AUTOGRAD_VALUE_GRAD]
	movss	dword [r9 + AUTOGRAD_VALUE_GRAD], xmm0
	movss	xmm0, dword [r10 + AUTOGRAD_VALUE_GRAD]
	mulss	xmm0, dword [r9 + AUTOGRAD_VALUE_DATA]
	addss	xmm0, dword [r8 + AUTOGRAD_VALUE_GRAD]
	movss	dword [r8 + AUTOGRAD_VALUE_GRAD], xmm0
	jmp	.abf_next
.abf_relu:
	mov	r8, qword [r10 + AUTOGRAD_VALUE_LHS]
	xorps	xmm1, xmm1
	movss	xmm0, dword [r8 + AUTOGRAD_VALUE_DATA]
	ucomiss	xmm0, xmm1
	jbe	.abf_next
	movss	xmm0, dword [r10 + AUTOGRAD_VALUE_GRAD]
	addss	xmm0, dword [r8 + AUTOGRAD_VALUE_GRAD]
	movss	dword [r8 + AUTOGRAD_VALUE_GRAD], xmm0
.abf_next:
	sub	r10, AUTOGRAD_VALUE_SIZE
	dec	rbx
	jnz	.abf_node
	pop	r12
	pop	rbx
	xor	eax, eax
	ret
.abf_bad:
	mov	eax, -1
	ret

align 4
_autograd_f32_one dd 1.0
