; Tensor shape, topology, and lifetime-token validation. Include in text.

TENSOR_DATA equ 0
TENSOR_GRAD equ 8
TENSOR_ROWS equ 16
TENSOR_COLS equ 24

TAPE_NODE_OP     equ 0
TAPE_NODE_LHS    equ 4
TAPE_NODE_RHS    equ 8
TAPE_NODE_FLAGS  equ 12
TAPE_NODE_TENSOR equ 16
TAPE_NODE_SIZE   equ 32

TAPE_INDEX_NONE equ 0ffffffffh
TAPE_OP_LEAF    equ 0
TAPE_OP_MATMUL  equ 1
TAPE_OP_RELU    equ 2
TAPE_OP_MSE     equ 3
TAPE_OP_BIAS_ADD equ 4

TAPE_FLAG_PARAMETER equ 1
TAPE_FLAG_CONSTANT  equ 2
TAPE_FLAG_INPUT     equ 4
TAPE_FLAG_TEMPORARY equ 8

; int tensor_shape_validate_f32(const Tensor *tensor, uint64_t *out_elements)
; Returns 0, -1 for invalid descriptor, -2 when element or byte count overflows.
; A null out_elements pointer is allowed. Gradient storage is optional.
tensor_shape_validate_f32:
	test	rdi, rdi
	jz	.tsv_bad
	cmp	qword [rdi + TENSOR_DATA], 0
	je	.tsv_bad
	mov	rax, [rdi + TENSOR_ROWS]
	test	rax, rax
	jz	.tsv_bad
	mov	rcx, [rdi + TENSOR_COLS]
	test	rcx, rcx
	jz	.tsv_bad
	mul	rcx
	test	rdx, rdx
	jnz	.tsv_overflow
	mov	rdx, 3fffffffffffffffh
	cmp	rax, rdx
	ja	.tsv_overflow
	test	rsi, rsi
	jz	.tsv_ok
	mov	[rsi], rax
.tsv_ok:
	xor	eax, eax
	ret
.tsv_bad:
	mov	eax, -1
	ret
.tsv_overflow:
	mov	eax, -2
	ret

; int tensor_tape_validate_f32(const TapeNode *nodes, uint64_t count)
; Nodes must be in topological order. Parents are uint32 indices, so copying or
; relocating the complete tape cannot invalidate edges.
; Returns 0, -1 for bad arguments, -2 for malformed nodes/edges.
tensor_tape_validate_f32:
	test	rdi, rdi
	jz	.ttv_bad
	test	rsi, rsi
	jz	.ttv_bad
	mov	eax, TAPE_INDEX_NONE
	cmp	rsi, rax
	jae	.ttv_bad
	xor	r8, r8
	mov	r9, rdi
.ttv_node:
	cmp	qword [r9 + TAPE_NODE_TENSOR], 0
	je	.ttv_malformed
	mov	eax, [r9 + TAPE_NODE_OP]
	cmp	eax, TAPE_OP_LEAF
	je	.ttv_leaf
	cmp	eax, TAPE_OP_MATMUL
	je	.ttv_binary
	cmp	eax, TAPE_OP_RELU
	je	.ttv_unary
	cmp	eax, TAPE_OP_MSE
	je	.ttv_binary
	cmp	eax, TAPE_OP_BIAS_ADD
	je	.ttv_binary
	jmp	.ttv_malformed
.ttv_leaf:
	mov	eax, [r9 + TAPE_NODE_FLAGS]
	test	eax, eax
	jz	.ttv_malformed
	test	eax, 0fffffff8h
	jnz	.ttv_malformed
	lea	edx, [rax - 1]
	test	eax, edx
	jnz	.ttv_malformed
	cmp	dword [r9 + TAPE_NODE_LHS], TAPE_INDEX_NONE
	jne	.ttv_malformed
	cmp	dword [r9 + TAPE_NODE_RHS], TAPE_INDEX_NONE
	jne	.ttv_malformed
	jmp	.ttv_next
.ttv_unary:
	cmp	dword [r9 + TAPE_NODE_FLAGS], TAPE_FLAG_TEMPORARY
	jne	.ttv_malformed
	mov	eax, [r9 + TAPE_NODE_LHS]
	cmp	rax, r8
	jae	.ttv_malformed
	cmp	dword [r9 + TAPE_NODE_RHS], TAPE_INDEX_NONE
	jne	.ttv_malformed
	jmp	.ttv_next
.ttv_binary:
	cmp	dword [r9 + TAPE_NODE_FLAGS], TAPE_FLAG_TEMPORARY
	jne	.ttv_malformed
	mov	eax, [r9 + TAPE_NODE_LHS]
	cmp	rax, r8
	jae	.ttv_malformed
	mov	eax, [r9 + TAPE_NODE_RHS]
	cmp	rax, r8
	jae	.ttv_malformed
.ttv_next:
	inc	r8
	add	r9, TAPE_NODE_SIZE
	cmp	r8, rsi
	jb	.ttv_node
	xor	eax, eax
	ret
.ttv_bad:
	mov	eax, -1
	ret
.ttv_malformed:
	mov	eax, -2
	ret

; int tensor_tape_lifetime_validate_f32(const TapeNode *nodes, uint64_t count,
;                                       Arena *const *arenas,
;                                       uint64_t arena_count)
; Node +24 is arena_slot:u32, +28 is generation:u32. Slot UINT32_MAX denotes
; an unmanaged external view and requires generation 0.
; Returns 0, -1 bad args, -4 stale/unknown lifetime token.
tensor_tape_lifetime_validate_f32:
	test	rdi, rdi
	jz	.ttlv_bad
	test	rsi, rsi
	jz	.ttlv_bad
	test	rdx, rdx
	jz	.ttlv_bad
	test	rcx, rcx
	jz	.ttlv_bad
	xor	r8, r8
	mov	r9, rdi
.ttlv_node:
	mov	eax, dword [r9 + 24]
	cmp	eax, TAPE_INDEX_NONE
	je	.ttlv_external
	cmp	rax, rcx
	jae	.ttlv_stale
	mov	r10, [rdx + rax * 8]
	test	r10, r10
	jz	.ttlv_stale
	mov	eax, dword [r9 + 28]
	cmp	eax, dword [r10 + 24]
	jne	.ttlv_stale
	jmp	.ttlv_next
.ttlv_external:
	cmp	dword [r9 + 28], 0
	jne	.ttlv_stale
.ttlv_next:
	inc	r8
	add	r9, TAPE_NODE_SIZE
	cmp	r8, rsi
	jb	.ttlv_node
	xor	eax, eax
	ret
.ttlv_bad:
	mov	eax, -1
	ret
.ttlv_stale:
	mov	eax, -4
	ret
