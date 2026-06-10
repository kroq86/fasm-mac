; Control flow macro demo: _if/_else/_endif, _while/_endw, _repeat/_until.
;
; Output (Linux ELF64):
;   even: 0
;   odd:  1
;   even: 2
;   odd:  3
;   even: 4
;   countdown: 3 2 1 done

format ELF64 executable 3
include "fasm/core/control.inc"
include "fasm/core/for.inc"

entry _start

segment readable executable

_start:
	; _while: iterate rbx 0..4, classify even/odd with _if
	xor rbx, rbx
	_while
		cmp rbx, 5
		jge __WE

		test rbx, 1
		_if je                      ; even
			mov	eax, 1
			mov	edi, 1
			lea	rsi, [str_even]
			mov	edx, str_even_len
			syscall
		_else                       ; odd
			mov	eax, 1
			mov	edi, 1
			lea	rsi, [str_odd]
			mov	edx, str_odd_len
			syscall
		_endif

		mov	rdi, rbx
		call	print_u64

		inc rbx
	_endw

	; _repeat/_until: countdown 3 → 1
	lea	rsi, [str_countdown]
	mov	eax, 1
	mov	edi, 1
	mov	edx, str_countdown_len
	syscall

	mov rbx, 3
	_repeat
		mov	rdi, rbx
		call	print_u64
		dec rbx
	_until rbx, jg, 0

	lea	rsi, [str_done]
	mov	eax, 1
	mov	edi, 1
	mov	edx, str_done_len
	syscall

	mov	eax, 60
	xor	edi, edi
	syscall


; rdi = unsigned 64-bit integer — writes decimal + newline to stdout
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
.digit:
	xor	edx, edx
	div	r8
	add	dl, '0'
	dec	rsi
	mov	[rsi], dl
	inc	rcx
	test	rax, rax
	jnz	.digit
.write:
	mov	eax, 1
	mov	edi, 1
	mov	rdx, rcx
	syscall
	add	rsp, 40
	ret


segment readable writable

str_even		db "even: "
str_even_len		= $ - str_even
str_odd			db "odd:  "
str_odd_len		= $ - str_odd
str_countdown		db "countdown: "
str_countdown_len	= $ - str_countdown
str_done		db "done", 10
str_done_len		= $ - str_done
