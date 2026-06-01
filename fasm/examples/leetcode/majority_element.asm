; LeetCode 169. Majority Element
; https://leetcode.com/problems/majority-element/
; nums=[2,2,1,1,1,2,2]  =>  2

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"

entry start

start:
	xor	rbx, rbx
	xor	r12, r12
	xor	r13, r13
.loop:
	cmp	rbx, test_nums_count
	jae	.done
	mov	rax, [test_nums + rbx * 8]
	test	r13, r13
	jz	.reset
	cmp	rax, r12
	je	.same
	dec	r13
	jmp	.next
.reset:
	mov	r12, rax
	mov	r13, 1
	jmp	.next
.same:
	inc	r13
.next:
	inc	rbx
	jmp	.loop
.done:
	mov	rax, r12
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

test_nums dq 2, 2, 1, 1, 1, 2, 2
test_nums_count = 7

include "fasm/core/runtime_bss.inc"
runtime_print_bss
