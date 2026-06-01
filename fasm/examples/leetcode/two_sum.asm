; LeetCode 1. Two Sum
; https://leetcode.com/problems/two-sum/
;
; Given nums and target, return indices i j (i < j) with nums[i]+nums[j]=target.
; Build: nums=[2,7,11,15], target=9  =>  0 1

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
	mov	[storage_nums], rax

	lea	rdi, [storage_heap]
	mov	rsi, [storage_nums]
	mov	rdx, 2
	call	vec_append_int
	lea	rdi, [storage_heap]
	mov	rsi, [storage_nums]
	mov	rdx, 7
	call	vec_append_int
	lea	rdi, [storage_heap]
	mov	rsi, [storage_nums]
	mov	rdx, 11
	call	vec_append_int
	lea	rdi, [storage_heap]
	mov	rsi, [storage_nums]
	mov	rdx, 15
	call	vec_append_int

	mov	qword [storage_target], 9

	mov	rdi, [storage_nums]
	mov	rsi, [storage_target]
	call	two_sum

	mov	rax, [storage_ans_i]
	call	print_int_sp
	mov	rax, [storage_ans_j]
	call	print_int_nl

	lea	rdi, [storage_heap]
	call	heap_destroy
	exit EXIT_SUCCESS

; rdi = nums VecHeader*, rsi = target
; fills storage_ans_i, storage_ans_j
two_sum:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	rbx, [r12 + VEC_COUNT_OFF]
	xor	r14, r14
.outer:
	cmp	r14, rbx
	jae	.ts_fail
	mov	r15, r14
.inc_inner:
	inc	r15
	cmp	r15, rbx
	jae	.outer_next
	mov	rdi, r12
	mov	rsi, r14
	call	vec_get_int
	mov	r10, rax
	mov	rdi, r12
	mov	rsi, r15
	call	vec_get_int
	add	rax, r10
	cmp	rax, r13
	jne	.inc_inner
	mov	[storage_ans_i], r14
	mov	[storage_ans_j], r15
	jmp	.ts_done
.outer_next:
	inc	r14
	jmp	.outer
.ts_fail:
	mov	qword [storage_ans_i], -1
	mov	qword [storage_ans_j], -1
.ts_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

segment readable writeable

storage_heap rb HEAP_SIZE
storage_nums dq ?
storage_target dq ?
storage_ans_i dq ?
storage_ans_j dq ?

include "fasm/core/runtime_bss.inc"
runtime_value_bss
