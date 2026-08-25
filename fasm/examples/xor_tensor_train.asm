; Train a 2 -> 4 -> 1 tensor MLP on XOR using the core tape ABI.

format ELF64
section '.text' executable

public main
extrn printf
extrn tensor_tape_forward_f32
extrn tensor_tape_backward_f32
extrn tensor_parameters_sgd_f32
extrn tensor_parameters_zero_grad_f32
extrn tensor_tape_validate_f32
extrn tensor_tape_lifetime_validate_f32
extrn tensor_plan_compile_f32
extrn tensor_plan_compile_fused_f32
extrn tensor_plan_forward_f32
extrn tensor_plan_backward_f32

include 'fasm/core/tensor_runtime_f32.inc'

main:
	push	rbp
	mov	rbp, rsp
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 8
	; Validate semantic graph/lifetimes once, then lower to direct-call steps.
	lea	rdi, [tape]
	mov	rsi, 12
	call	tensor_tape_validate_f32
	test	eax, eax
	jnz	.failed
	lea	rdi, [tape]
	mov	rsi, 12
	lea	rdx, [arenas]
	mov	rcx, 2
	call	tensor_tape_lifetime_validate_f32
	test	eax, eax
	jnz	.failed
	; Lower the unchanged semantic graph with automatic single-consumer fusion.
	; Arguments 7 and 8 are passed on the System V stack.
	sub	rsp, 16
	lea	rax, [plan_count]
	mov	[rsp], rax
	lea	rax, [context_count]
	mov	[rsp + 8], rax
	lea	rdi, [tape]
	mov	rsi, 12
	lea	rdx, [plan_steps]
	mov	rcx, 3
	lea	r8, [fusion_contexts]
	mov	r9, 2
	call	tensor_plan_compile_fused_f32
	add	rsp, 16
	test	eax, eax
	jnz	.compile_failed
	cmp	qword [plan_count], 3
	jne	.count_failed
	cmp	qword [context_count], 2
	jne	.context_failed
	mov	r12, 1500
.train:
	; Static liveness plan: clear the one shared 384-byte scratch arena before
	; forward. Saved forward values then remain intact until their backward use.
	lea	rdi, [scratch]
	mov	rcx, 48
	xor	eax, eax
	rep stosq
	lea	rdi, [x_grad]
	mov	rcx, 8
	rep stosd
	lea	rdi, [tape]
	mov	rsi, 12
	call	tensor_parameters_zero_grad_f32
	test	eax, eax
	jnz	.failed
	lea	rdi, [plan_steps]
	mov	rsi, [plan_count]
	call	tensor_plan_forward_f32
	test	eax, eax
	jnz	.forward_failed
	lea	rdi, [plan_steps]
	mov	rsi, [plan_count]
	call	tensor_plan_backward_f32
	test	eax, eax
	jnz	.backward_failed
	lea	rdi, [tape]
	mov	rsi, 12
	movss	xmm0, [learning_rate]
	call	tensor_parameters_sgd_f32
	test	eax, eax
	jnz	.sgd_failed
	dec	r12
	jnz	.train

	lea	rdi, [scratch]
	mov	rcx, 48
	xor	eax, eax
	rep stosq
	lea	rdi, [plan_steps]
	mov	rsi, [plan_count]
	call	tensor_plan_forward_f32
	test	eax, eax
	jnz	.final_forward_failed

	; Classification gate: [0,1,1,0] with a generous non-flaky margin.
	mov	dword [rsp], 0
	movss	xmm0, [pred_data]
	ucomiss	xmm0, [low_limit]
	jae	.bad_result
	movss	xmm0, [pred_data + 4]
	ucomiss	xmm0, [high_limit]
	jb	.bad_result
	movss	xmm0, [pred_data + 8]
	ucomiss	xmm0, [high_limit]
	jb	.bad_result
	movss	xmm0, [pred_data + 12]
	ucomiss	xmm0, [low_limit]
	jae	.bad_result
	movss	xmm0, [loss_data]
	ucomiss	xmm0, [loss_limit]
	jb	.print_result
.bad_result:
	mov	dword [rsp], 1
.print_result:
	movss	xmm0, [loss_data]
	mulss	xmm0, [thousand]
	cvttss2si rbx, xmm0
	movss	xmm0, [pred_data]
	mulss	xmm0, [thousand]
	cvttss2si r12, xmm0
	movss	xmm0, [pred_data + 4]
	mulss	xmm0, [thousand]
	cvttss2si r13, xmm0
	movss	xmm0, [pred_data + 8]
	mulss	xmm0, [thousand]
	cvttss2si r14, xmm0
	movss	xmm0, [pred_data + 12]
	mulss	xmm0, [thousand]
	cvttss2si r15, xmm0
	lea	rdi, [result_format]
	mov	rsi, rbx
	mov	rdx, r12
	mov	rcx, r13
	mov	r8, r14
	mov	r9, r15
	xor	eax, eax
	call	printf
	mov	eax, dword [rsp]
	jmp	.done
.failed:
	mov	eax, 2
	jmp	.done
.compile_failed:
	mov	eax, 3
	jmp	.done
.count_failed:
	mov	eax, 4
	jmp	.done
.context_failed:
	mov	eax, 5
	jmp	.done
.forward_failed:
	mov	eax, 6
	jmp	.done
.backward_failed:
	mov	eax, 7
	jmp	.done
.sgd_failed:
	mov	eax, 8
	jmp	.done
.final_forward_failed:
	mov	eax, 9
.done:
	add	rsp, 8
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	pop	rbp
	ret

section '.data' writeable align 64

learning_rate dd 0.1
low_limit dd 0.1
high_limit dd 0.9
loss_limit dd 0.01
thousand dd 1000.0
result_format db 'xor trained plan_steps=3 fused_contexts=2 scratch_bytes=384 loss_milli=%d predictions_milli=%d %d %d %d',10,0

x_data dd 0.0,0.0, 0.0,1.0, 1.0,0.0, 1.0,1.0
x_grad rd 8
w1_data dd 0.5,-0.7,0.3,0.8, -0.4,0.6,0.9,-0.2
w1_grad rd 8
b1_data dd 0.1,0.1,-0.1,0.0
b1_grad rd 4
w2_data dd 0.7,-0.5,0.6,-0.8
w2_grad rd 4
b2_data dd 0.0
b2_grad rd 1
target_data dd 0.0,1.0,1.0,0.0
loss_grad dd 1.0

align 64
scratch rb 384
z1_data = scratch + 0
z1_grad = scratch + 160
z1b_data = scratch + 64
z1b_grad = scratch + 224
h_data = scratch + 0
h_grad = scratch + 288
z2_data = scratch + 128
z2_grad = scratch + 352
pred_data = scratch + 144
pred_grad = scratch + 368
loss_data = scratch + 128

align 8
x_tensor dq x_data,x_grad,4,2
w1_tensor dq w1_data,w1_grad,2,4
b1_tensor dq b1_data,b1_grad,1,4
w2_tensor dq w2_data,w2_grad,4,1
b2_tensor dq b2_data,b2_grad,1,1
target_tensor dq target_data,0,4,1
z1_tensor dq z1_data,z1_grad,4,4
z1b_tensor dq z1b_data,z1b_grad,4,4
h_tensor dq h_data,h_grad,4,4
z2_tensor dq z2_data,z2_grad,4,1
pred_tensor dq pred_data,pred_grad,4,1
loss_tensor dq loss_data,loss_grad,1,1

; Only generation (+24) is consumed by the lifetime validator.
persistent_arena dq 0,0,0,1
scratch_arena dq 0,0,0,1
arenas dq persistent_arena,scratch_arena

align 8
plan_count dq 0
context_count dq 0
plan_steps rb 3 * 40
fusion_contexts rb 2 * 48

macro node op,lhs,rhs,flags,tensor,slot
{
	dd op,lhs,rhs,flags
	dq tensor
	dd slot,1
}

align 8
tape:
node TENSOR_OP_LEAF,TENSOR_INDEX_NONE,TENSOR_INDEX_NONE,TENSOR_FLAG_INPUT,x_tensor,0
node TENSOR_OP_LEAF,TENSOR_INDEX_NONE,TENSOR_INDEX_NONE,TENSOR_FLAG_PARAMETER,w1_tensor,0
node TENSOR_OP_LEAF,TENSOR_INDEX_NONE,TENSOR_INDEX_NONE,TENSOR_FLAG_PARAMETER,b1_tensor,0
node TENSOR_OP_LEAF,TENSOR_INDEX_NONE,TENSOR_INDEX_NONE,TENSOR_FLAG_PARAMETER,w2_tensor,0
node TENSOR_OP_LEAF,TENSOR_INDEX_NONE,TENSOR_INDEX_NONE,TENSOR_FLAG_PARAMETER,b2_tensor,0
node TENSOR_OP_LEAF,TENSOR_INDEX_NONE,TENSOR_INDEX_NONE,TENSOR_FLAG_CONSTANT,target_tensor,0
node TENSOR_OP_MATMUL,0,1,TENSOR_FLAG_TEMPORARY,z1_tensor,1
node TENSOR_OP_BIAS_ADD,6,2,TENSOR_FLAG_TEMPORARY,z1b_tensor,1
node TENSOR_OP_RELU,7,TENSOR_INDEX_NONE,TENSOR_FLAG_TEMPORARY,h_tensor,1
node TENSOR_OP_MATMUL,8,3,TENSOR_FLAG_TEMPORARY,z2_tensor,1
node TENSOR_OP_BIAS_ADD,9,4,TENSOR_FLAG_TEMPORARY,pred_tensor,1
node TENSOR_OP_MSE,10,5,TENSOR_FLAG_TEMPORARY,loss_tensor,1
