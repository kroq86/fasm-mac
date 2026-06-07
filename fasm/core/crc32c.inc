; Software CRC32C (Castagnoli) helpers.
;
; Public CRC value uses the usual init/final xor convention:
;   crc32c_compute("", 0) = 00000000h
;   crc32c_compute("123456789", 9) = E3069283h

CRC32C_POLY_REFLECTED equ 082F63B78h

; rdi = buffer, rsi = length
; eax = CRC32C
crc32c_compute:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	eax, 0FFFFFFFFh
.c32_loop:
	test	r13, r13
	jz	.c32_done
	movzx	ecx, byte [r12]
	xor	eax, ecx
	mov	bl, 8
.c32_bit:
	test	eax, 1
	jz	.c32_shift
	shr	eax, 1
	xor	eax, CRC32C_POLY_REFLECTED
	jmp	.c32_next_bit
.c32_shift:
	shr	eax, 1
.c32_next_bit:
	dec	bl
	jnz	.c32_bit
	inc	r12
	dec	r13
	jmp	.c32_loop
.c32_done:
	not	eax
	pop	r13
	pop	r12
	pop	rbx
	ret
