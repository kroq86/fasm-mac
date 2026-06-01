; LeetCode 339. Nested List Weight Sum
; https://leetcode.com/problems/nested-list-weight-sum/
;
; Each integer is multiplied by its nesting depth (root list depth = 1).
; Input: [[1,1],2,[1,1]]  =>  2*1 + 2*1 + 1*2 + 2*1 + 2*1 = 10

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/dynvec.inc"
include "fasm/core/print_value.inc"
include "fasm/core/leetcode.inc"

entry start

start:
	lea	rdi, [storage_heap]
	call	heap_init

	lea	rdi, [storage_heap]
	mov	rsi, 4
	call	vec_init
	mov	[storage_root_vec], rax

	; inner [1,1]
	lea	rdi, [storage_heap]
	mov	rsi, [storage_root_vec]
	mov	rdx, 2
	call	vec_append_array
	mov	[storage_inner1], rax
	lea	rdi, [storage_heap]
	mov	rsi, [storage_inner1]
	mov	rdx, 1
	call	vec_append_int
	lea	rdi, [storage_heap]
	mov	rsi, [storage_inner1]
	mov	rdx, 1
	call	vec_append_int

	; 2
	lea	rdi, [storage_heap]
	mov	rsi, [storage_root_vec]
	mov	rdx, 2
	call	vec_append_int

	; inner [1,1]
	lea	rdi, [storage_heap]
	mov	rsi, [storage_root_vec]
	mov	rdx, 2
	call	vec_append_array
	mov	[storage_inner2], rax
	lea	rdi, [storage_heap]
	mov	rsi, [storage_inner2]
	mov	rdx, 1
	call	vec_append_int
	lea	rdi, [storage_heap]
	mov	rsi, [storage_inner2]
	mov	rdx, 1
	call	vec_append_int

	mov	rdi, [storage_root_vec]
	mov	rsi, 1
	call	depth_weight_sum
	call	print_int_nl

	lea	rdi, [storage_heap]
	call	heap_destroy
	exit EXIT_SUCCESS

; rdi = VecHeader*, rsi = depth
; rax = weighted sum
depth_weight_sum:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	xor	r14, r14
	xor	rbx, rbx
.dws_loop:
	mov	rcx, [r12 + VEC_COUNT_OFF]
	cmp	r14, rcx
	jae	.dws_done
	mov	rdi, r12
	mov	rsi, r14
	call	vec_get
	mov	rdi, rax
	movzx	eax, byte [rdi + VALUE_TAG_OFF]
	cmp	al, VAL_INT
	jne	.dws_array
	call	value_as_int
	imul	rax, r13
	add	rbx, rax
	jmp	.dws_next
.dws_array:
	mov	rdi, [rdi + VALUE_DATA_OFF]
	mov	rsi, r13
	inc	rsi
	call	depth_weight_sum
	add	rbx, rax
.dws_next:
	inc	r14
	jmp	.dws_loop
.dws_done:
	mov	rax, rbx
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

segment readable writeable

storage_heap rb HEAP_SIZE
storage_root_vec dq ?
storage_inner1 dq ?
storage_inner2 dq ?

include "fasm/core/runtime_bss.inc"
runtime_value_bss
