; LeetCode 912. Sort an Array
; https://leetcode.com/problems/sort-an-array/
; Input: [-1,0,1,2,3]  =>  -1 0 1 2 3

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/dynvec.inc"
include "fasm/core/sort.inc"
include "fasm/core/print_value.inc"
include "fasm/core/leetcode.inc"

entry start

start:
	lea	rdi, [storage_heap]
	call	heap_init

	lea	rdi, [storage_heap]
	mov	rsi, 5
	call	vec_init
	mov	[storage_nums], rax

	lea	rdi, [storage_heap]
	mov	rsi, [storage_nums]
	lea	rdx, [test_nums]
	mov	rcx, test_nums_count
	call	vec_build_ints

	mov	rdi, [storage_nums]
	call	vec_sort_int

	mov	r14, [storage_nums]
	mov	rbx, [r14 + VEC_COUNT_OFF]
	xor	r15, r15
.print_loop:
	cmp	r15, rbx
	jae	.print_done
	mov	rdi, r14
	mov	rsi, r15
	call	vec_get_int
	call	print_int_sp
	inc	r15
	jmp	.print_loop
.print_done:
	mov	al, 10
	call	print_char

	lea	rdi, [storage_heap]
	call	heap_destroy
	exit EXIT_SUCCESS

start.fail:
	lea	rdi, [storage_heap]
	call	heap_destroy
	exit EXIT_FAILURE

segment readable writeable

storage_heap rb HEAP_SIZE
storage_nums dq ?

test_nums dq -1, 0, 1, 2, 3
test_nums_count = 5

include "fasm/core/runtime_bss.inc"
runtime_value_bss
