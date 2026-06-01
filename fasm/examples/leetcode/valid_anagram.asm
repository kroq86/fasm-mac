; LeetCode 242. Valid Anagram
; https://leetcode.com/problems/valid-anagram/
; s="anagram", t="nagaram"  =>  1

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"

entry start

start:
	call	is_anagram
	call	print_bool_nl
	exit EXIT_SUCCESS

is_anagram:
	lea	rdi, [s]
	call	count_add
	lea	rdi, [t]
	call	count_sub
	xor	rcx, rcx
.check:
	cmp	rcx, 26
	jae	.yes
	cmp	qword [counts + rcx * 8], 0
	jne	.no
	inc	rcx
	jmp	.check
.yes:
	mov	rax, 1
	ret
.no:
	xor	rax, rax
	ret

count_add:
	movzx	eax, byte [rdi]
	test	al, al
	jz	.done
	sub	al, 'a'
	movzx	rax, al
	inc	qword [counts + rax * 8]
	inc	rdi
	jmp	count_add
.done:
	ret

count_sub:
	movzx	eax, byte [rdi]
	test	al, al
	jz	.done
	sub	al, 'a'
	movzx	rax, al
	dec	qword [counts + rax * 8]
	inc	rdi
	jmp	count_sub
.done:
	ret

segment readable writeable

s db 'anagram', 0
t db 'nagaram', 0
counts rq 26

include "fasm/core/runtime_bss.inc"
runtime_print_bss
