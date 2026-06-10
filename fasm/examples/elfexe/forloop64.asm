; Nested for-loop demo using the for/endfor macros.
; Outer loop rbx = 0..4: prints the index.
; Inner loop r12 = 0..2: prints "hello" for each index.

format ELF64 executable 3
include "fasm/core/for.inc"

entry _start

segment readable executable

_start:
	for rbx, 0, 5

		mov	rdi, rbx
		call	print_u64

		for r12, 0, 3
			mov	eax, 1
			mov	edi, 1
			lea	rsi, [hello]
			mov	edx, hello_len
			syscall
		endfor r12

	endfor rbx

	mov	eax, 60
	xor	edi, edi
	syscall


; rdi = unsigned 64-bit integer — prints value followed by newline
print_u64:
	sub	rsp, 40

	mov	rax, rdi
	lea	rsi, [rsp + 39]
	mov	byte [rsi], 10
	mov	rcx, 1

	test	rax, rax
	jnz	.convert

	dec	rsi
	mov	byte [rsi], '0'
	inc	rcx
	jmp	.write

.convert:
	mov	r8, 10
.next_digit:
	xor	edx, edx
	div	r8
	add	dl, '0'
	dec	rsi
	mov	[rsi], dl
	inc	rcx
	test	rax, rax
	jnz	.next_digit

.write:
	mov	eax, 1
	mov	edi, 1
	mov	rdx, rcx
	syscall

	add	rsp, 40
	ret


segment readable writable

hello		db "hello", 10
hello_len	= $ - hello
