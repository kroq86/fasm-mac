; LeetCode 387. First Unique Character in a String
; https://leetcode.com/problems/first-unique-character-in-a-string/
; s="leetcode"  =>  0

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"

entry start

start:
	xor	rbx, rbx
.count:
	mov	al, [s + rbx]
	test	al, al
	jz	.find
	sub	al, 'a'
	movzx	rax, al
	inc	qword [counts + rax * 8]
	inc	rbx
	jmp	.count
.find:
	xor	rbx, rbx
.find_loop:
	mov	al, [s + rbx]
	test	al, al
	jz	.missing
	sub	al, 'a'
	movzx	rax, al
	cmp	qword [counts + rax * 8], 1
	je	.found
	inc	rbx
	jmp	.find_loop
.found:
	mov	rax, rbx
	jmp	.print
.missing:
	mov	rax, -1
.print:
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

s db 'leetcode', 0
counts rq 26

include "fasm/core/runtime_bss.inc"
runtime_print_bss
