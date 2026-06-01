; LeetCode 1. Two Sum (O(n) hash map)
; https://leetcode.com/problems/two-sum/
; nums=[2,7,11,15], target=9  =>  0 1

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/dynvec.inc"
include "fasm/core/hashmap.inc"
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
	lea	rdx, [test_nums]
	mov	rcx, test_nums_count
	call	vec_build_ints

	mov	qword [storage_target], 9

	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, HASHMAP_DEFAULT_BUCKETS
	call	hashmap_init

	mov	rdi, [storage_nums]
	mov	rsi, [storage_target]
	call	two_sum_hashmap

	mov	rax, [storage_ans_i]
	call	print_int_sp
	mov	rax, [storage_ans_j]
	call	print_int_nl

	lea	rdi, [storage_heap]
	call	heap_destroy
	exit EXIT_SUCCESS

; rdi = nums VecHeader*, rsi = target
two_sum_hashmap:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	rbx, [r12 + VEC_COUNT_OFF]
	xor	r14, r14
.loop:
	cmp	r14, rbx
	jae	.tsh_fail
	mov	rdi, r12
	mov	rsi, r14
	call	vec_get_int
	mov	r15, rax
	mov	rax, r13
	sub	rax, r15
	lea	rdi, [storage_map]
	mov	rsi, rax
	call	hashmap_get
	cmp	rax, HASHMAP_MISSING
	je	.tsh_insert
	mov	[storage_ans_i], rax
	mov	[storage_ans_j], r14
	jmp	.tsh_done
.tsh_insert:
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, r15
	mov	rcx, r14
	call	hashmap_put
	inc	r14
	jmp	.loop
.tsh_fail:
	mov	qword [storage_ans_i], -1
	mov	qword [storage_ans_j], -1
.tsh_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

segment readable writeable

storage_heap rb HEAP_SIZE
storage_nums dq ?
storage_map rb HASHMAP_SIZE
storage_target dq ?
storage_ans_i dq ?
storage_ans_j dq ?

test_nums dq 2, 7, 11, 15
test_nums_count = 4

include "fasm/core/runtime_bss.inc"
runtime_value_bss
