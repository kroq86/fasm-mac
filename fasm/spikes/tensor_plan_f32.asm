; Experimental compilation from semantic TapeNode graph to direct-call steps.

format ELF64
section '.text' executable

public tensor_plan_compile_f32
public tensor_plan_forward_f32
public tensor_plan_backward_f32

extrn tensor_matmul_forward_f32
extrn tensor_matmul_backward_f32
extrn tensor_relu_forward_f32
extrn tensor_relu_backward_f32
extrn tensor_mse_forward_f32
extrn tensor_mse_backward_f32
extrn tensor_bias_add_forward_f32
extrn tensor_bias_add_backward_f32

NODE_OP equ 0
NODE_LHS equ 4
NODE_RHS equ 8
NODE_TENSOR equ 16
NODE_SIZE equ 32
OP_LEAF equ 0
OP_MATMUL equ 1
OP_RELU equ 2
OP_MSE equ 3
OP_BIAS_ADD equ 4

PLAN_FORWARD equ 0
PLAN_BACKWARD equ 8
PLAN_LHS equ 16
PLAN_RHS equ 24
PLAN_OUT equ 32
PLAN_STEP_SIZE equ 40

; int tensor_plan_compile_f32(const Node *nodes, uint64_t node_count,
;                             PlanStep *steps, uint64_t step_capacity,
;                             uint64_t *out_step_count)
; Graph topology/lifetime must be validated before compilation.
tensor_plan_compile_f32:
	test	rdi, rdi
	jz	.tpc_bad
	test	rsi, rsi
	jz	.tpc_bad
	test	rdx, rdx
	jz	.tpc_bad
	test	r8, r8
	jz	.tpc_bad
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 16
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	[rsp], r8
	xor	rbx, rbx
	xor	r9, r9
.tpc_node:
	cmp	r9, r13
	jae	.tpc_done
	mov	rax, r9
	shl	rax, 5
	lea	r10, [r12 + rax]
	mov	eax, [r10 + NODE_OP]
	test	eax, eax
	jz	.tpc_next
	cmp	rbx, r15
	jae	.tpc_capacity
	cmp	eax, OP_MATMUL
	je	.tpc_matmul
	cmp	eax, OP_RELU
	je	.tpc_relu
	cmp	eax, OP_MSE
	je	.tpc_mse
	cmp	eax, OP_BIAS_ADD
	je	.tpc_bias
	jmp	.tpc_malformed
.tpc_matmul:
	lea	rax, [tensor_matmul_forward_f32]
	lea	rdx, [tensor_matmul_backward_f32]
	jmp	.tpc_emit
.tpc_relu:
	lea	rax, [tensor_plan_relu_forward_thunk]
	lea	rdx, [tensor_plan_relu_backward_thunk]
	jmp	.tpc_emit
.tpc_mse:
	lea	rax, [tensor_mse_forward_f32]
	lea	rdx, [tensor_mse_backward_f32]
	jmp	.tpc_emit
.tpc_bias:
	lea	rax, [tensor_bias_add_forward_f32]
	lea	rdx, [tensor_bias_add_backward_f32]
.tpc_emit:
	imul	r11, rbx, PLAN_STEP_SIZE
	add	r11, r14
	mov	[r11 + PLAN_FORWARD], rax
	mov	[r11 + PLAN_BACKWARD], rdx
	mov	edx, [r10 + NODE_LHS]
	shl	rdx, 5
	mov	rax, [r12 + rdx + NODE_TENSOR]
	mov	[r11 + PLAN_LHS], rax
	cmp	dword [r10 + NODE_OP], OP_RELU
	je	.tpc_no_rhs
	mov	edx, [r10 + NODE_RHS]
	shl	rdx, 5
	mov	rax, [r12 + rdx + NODE_TENSOR]
	mov	[r11 + PLAN_RHS], rax
	jmp	.tpc_out_ptr
.tpc_no_rhs:
	mov	qword [r11 + PLAN_RHS], 0
.tpc_out_ptr:
	mov	rax, [r10 + NODE_TENSOR]
	mov	[r11 + PLAN_OUT], rax
	inc	rbx
.tpc_next:
	inc	r9
	jmp	.tpc_node
.tpc_done:
	mov	rax, [rsp]
	mov	[rax], rbx
	xor	eax, eax
	jmp	.tpc_out
.tpc_capacity:
	mov	eax, -3
	jmp	.tpc_out
.tpc_malformed:
	mov	eax, -2
.tpc_out:
	add	rsp, 16
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
.tpc_bad:
	mov	eax, -1
	ret

; int tensor_plan_forward_f32(const PlanStep *steps, uint64_t count)
tensor_plan_forward_f32:
	test	rdi, rdi
	jz	.tpf_bad
	test	rsi, rsi
	jz	.tpf_bad
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	xor	rbx, rbx
.tpf_step:
	imul	rax, rbx, PLAN_STEP_SIZE
	add	rax, r12
	mov	rdi, [rax + PLAN_LHS]
	mov	rsi, [rax + PLAN_RHS]
	mov	rdx, [rax + PLAN_OUT]
	call	qword [rax + PLAN_FORWARD]
	test	eax, eax
	jnz	.tpf_out
	inc	rbx
	cmp	rbx, r13
	jb	.tpf_step
	xor	eax, eax
.tpf_out:
	pop	r13
	pop	r12
	pop	rbx
	ret
.tpf_bad:
	mov	eax, -1
	ret

; int tensor_plan_backward_f32(const PlanStep *steps, uint64_t count)
tensor_plan_backward_f32:
	test	rdi, rdi
	jz	.tpb_bad
	test	rsi, rsi
	jz	.tpb_bad
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rbx, r13
.tpb_step:
	dec	rbx
	imul	rax, rbx, PLAN_STEP_SIZE
	add	rax, r12
	mov	rdi, [rax + PLAN_LHS]
	mov	rsi, [rax + PLAN_RHS]
	mov	rdx, [rax + PLAN_OUT]
	call	qword [rax + PLAN_BACKWARD]
	test	eax, eax
	jnz	.tpb_out
	test	rbx, rbx
	jnz	.tpb_step
	xor	eax, eax
.tpb_out:
	pop	r13
	pop	r12
	pop	rbx
	ret
.tpb_bad:
	mov	eax, -1
	ret

; Normalize unary ReLU to the plan's (lhs, rhs, out) call shape.
tensor_plan_relu_forward_thunk:
	mov	rsi, rdx
	jmp	tensor_relu_forward_f32

tensor_plan_relu_backward_thunk:
	mov	rsi, rdx
	jmp	tensor_relu_backward_f32
