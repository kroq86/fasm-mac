; Hex formatting helpers (stdout via caller or io_write).
; Requires print_io.inc and runtime_print_bss.

HEX_LINE_BYTES equ 16

; al = nibble 0..15 -> al = ascii hex char
hex_nibble_to_char:
	cmp	al, 10
	jb	.hnc_digit
	add	al, 'a' - 10
	ret
.hnc_digit:
	add	al, '0'
	ret

; rdi = buf ptr, al = byte -> writes 2 hex chars, returns advanced rdi in rax
hex_write_byte:
	push	rbx
	mov	rbx, rdi
	movzx	ecx, al
	shr	al, 4
	call	hex_nibble_to_char
	mov	[rbx], al
	inc	rbx
	mov	al, cl
	and	al, 0Fh
	call	hex_nibble_to_char
	mov	[rbx], al
	inc	rbx
	mov	rax, rbx
	pop	rbx
	ret

; rdi = out buf, rcx = offset -> writes 8 hex offset digits, rax = advanced ptr
hex_write_offset8:
	push	rbx
	push	r12
	mov	r12, rdi
	mov	rbx, rcx
	lea	rdi, [r12 + 7]
	mov	rcx, 8
.hwo_loop:
	mov	rax, rbx
	and	rax, 0Fh
	mov	al, al
	call	hex_nibble_to_char
	mov	[rdi], al
	dec	rdi
	shr	rbx, 4
	loop	.hwo_loop
	lea	rax, [r12 + 8]
	pop	r12
	pop	rbx
	ret

; rdi = out buf, rsi = data, rdx = len (1..16), rcx = file offset
; writes one xxd-style line + newline to stdout
hex_write_line:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	rdi, r12
	mov	rcx, r15
	call	hex_write_offset8
	mov	rbx, rax
	mov	byte [rbx], ':'
	inc	rbx
	mov	byte [rbx], ' '
	inc	rbx
	mov	rcx, r14
	mov	rsi, r13
.hwl_hex:
	test	rcx, rcx
	jz	.hwl_hex_pad
	movzx	eax, byte [rsi]
	mov	rdi, rbx
	push	rsi
	push	rcx
	call	hex_write_byte
	pop	rcx
	pop	rsi
	mov	rbx, rax
	mov	byte [rbx], ' '
	inc	rbx
	inc	rsi
	dec	rcx
	jmp	.hwl_hex
.hwl_hex_pad:
	mov	rax, HEX_LINE_BYTES
	sub	rax, r14
	jz	.hwl_ascii_sep
.hwl_pad:
	mov	byte [rbx], ' '
	mov	byte [rbx + 1], ' '
	mov	byte [rbx + 2], ' '
	add	rbx, 3
	dec	rax
	jnz	.hwl_pad
.hwl_ascii_sep:
	mov	byte [rbx], ' '
	mov	byte [rbx + 1], ' '
	add	rbx, 2
	mov	rcx, r14
	mov	rsi, r13
.hwl_ascii:
	test	rcx, rcx
	jz	.hwl_nl
	movzx	eax, byte [rsi]
	cmp	al, 32
	jb	.hwl_dot
	cmp	al, 126
	ja	.hwl_dot
	mov	[rbx], al
	jmp	.hwl_adv
.hwl_dot:
	mov	byte [rbx], '.'
.hwl_adv:
	inc	rsi
	inc	rbx
	dec	rcx
	jmp	.hwl_ascii
.hwl_nl:
	mov	byte [rbx], 10
	inc	rbx
	mov	rdi, r12
	mov	rsi, rbx
	sub	rsi, r12
	call	io_write
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
