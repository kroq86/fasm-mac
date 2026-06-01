; LeetCode 88. Merge Sorted Array
; https://leetcode.com/problems/merge-sorted-array/
; nums1=[1,2,3,0,0,0], nums2=[2,5,6]  =>  1 2 2 3 5 6

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"

entry start

start:
	mov	r12, m - 1
	mov	r13, n - 1
	mov	r14, m + n - 1
.loop:
	cmp	r13, -1
	je	.print
	cmp	r12, -1
	je	.take_nums2
	mov	rax, [nums1 + r12 * 8]
	mov	rcx, [nums2 + r13 * 8]
	cmp	rax, rcx
	jl	.take_nums2
	mov	[nums1 + r14 * 8], rax
	dec	r12
	jmp	.next
.take_nums2:
	mov	rcx, [nums2 + r13 * 8]
	mov	[nums1 + r14 * 8], rcx
	dec	r13
.next:
	dec	r14
	jmp	.loop
.print:
	lea	rdi, [nums1]
	mov	rsi, m + n
	call	print_i64_array
	exit EXIT_SUCCESS

segment readable writeable

nums1 dq 1, 2, 3, 0, 0, 0
nums2 dq 2, 5, 6
m = 3
n = 3

include "fasm/core/runtime_bss.inc"
runtime_print_bss
