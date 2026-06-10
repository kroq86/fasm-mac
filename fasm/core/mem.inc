; Bulk memory operations.
; Algorithms adapted from HeavyThing (Jeff Marrison) and Agner Fog's asmlib (GPLv2+).
; https://2ton.com.au/  |  https://www.agner.org/optimize/
;
; memcpy  rdi=dst rsi=src rdx=n
; memset  rdi=dst esi=byte rdx=n
; memcmp  rdi=a   rsi=b   rdx=n  → rax (<0 / 0 / >0)
; memmove rdi=dst rsi=src rdx=n
; memxor  rdi=dst rsi=src rdx=n  (dst[i] ^= src[i])
;
; All follow System V AMD64 ABI. No return value except memcmp.

; ------------------------------------------------------------
; memcpy — copy n bytes from src to dst (no overlap assumed)
; rdi=dst, rsi=src, rdx=n
; ------------------------------------------------------------
memcpy:
	mov	r9, rdi
	cmp	rdx, 0x40
	jae	.large

	; small path (< 64): backward-index trick, processes 32/16/8/4/2/1
	add	rdi, rdx
	add	rsi, rdx
	neg	rdx

.do32:
	cmp	rdx, -32
	jg	.do16
	mov	rcx, [rsi+rdx]
	mov	r8,  [rsi+rdx+8]
	mov	[rdi+rdx],   rcx
	mov	[rdi+rdx+8], r8
	mov	rcx, [rsi+rdx+0x10]
	mov	r8,  [rsi+rdx+0x18]
	mov	[rdi+rdx+0x10], rcx
	mov	[rdi+rdx+0x18], r8
	add	rdx, 32
.do16:
	cmp	rdx, -16
	jg	.do8
	mov	rcx, [rsi+rdx]
	mov	r8,  [rsi+rdx+8]
	mov	[rdi+rdx],   rcx
	mov	[rdi+rdx+8], r8
	add	rdx, 16
.do8:
	cmp	rdx, -8
	jg	.do4
	mov	rcx, [rsi+rdx]
	mov	[rdi+rdx], rcx
	add	rdx, 8
.do4:
	cmp	rdx, -4
	jg	.do2
	mov	ecx, [rsi+rdx]
	mov	[rdi+rdx], ecx
	add	rdx, 4
	jz	.done_small
.do2:
	cmp	rdx, -2
	jg	.do1
	movzx	ecx, word [rsi+rdx]
	mov	[rdi+rdx], cx
	add	rdx, 2
	jz	.done_small
.do1:
	cmp	rdx, -1
	jg	.done_small
	movzx	ecx, byte [rsi+rdx]
	mov	[rdi+rdx], cl
.done_small:
	mov	rax, r9
	ret

.large:
	; align dst to 16-byte boundary, then use rep movsb for the rest
	; rep movsb with ERMSB is fast on modern CPUs (Ivy Bridge+)
	push	rbx
	mov	rbx, r9
	mov	rcx, rdx
	rep movsb
	mov	rax, rbx
	pop	rbx
	ret

; ------------------------------------------------------------
; memset — fill n bytes with byte value
; rdi=dst, esi=byte (low 8 bits), rdx=n
; ------------------------------------------------------------
memset:
	push	rbx
	mov	rbx, rdi
	; replicate byte to all 4 bytes of esi
	imul	esi, 0x01010101
	cmp	rdx, 16
	ja	.large_set

	; small dispatch (≤16): unrolled overlapping stores
	add	rdi, rdx
	neg	rdx
.ms4:
	cmp	rdx, -4
	jg	.ms2
	mov	[rdi+rdx], esi
	add	rdx, 4
	jz	.ms_done
.ms2:
	cmp	rdx, -2
	jg	.ms1
	mov	[rdi+rdx], si
	add	rdx, 2
	jz	.ms_done
.ms1:
	cmp	rdx, -1
	jg	.ms_done
	mov	[rdi+rdx], sil
.ms_done:
	mov	rax, rbx
	pop	rbx
	ret

.large_set:
	movzx	eax, sil
	mov	rcx, rdx
	rep stosb
	mov	rax, rbx
	pop	rbx
	ret

; ------------------------------------------------------------
; memcmp — compare n bytes
; rdi=a, rsi=b, rdx=n  →  rax (<0 / 0 / >0)
; Uses XOR + BSF to locate the first differing byte in each qword.
; Adapted from HeavyThing / Agner Fog.
; ------------------------------------------------------------
memcmp:
	add	rdi, rdx
	add	rsi, rdx
	neg	rdx

.mc32:
	cmp	rdx, -32
	jg	.mc16
	mov	rax, [rdi+rdx]
	xor	rax, [rsi+rdx]
	jnz	.inequal
	mov	rax, [rdi+rdx+8]
	xor	rax, [rsi+rdx+8]
	jnz	.inequal8
	mov	rax, [rdi+rdx+0x10]
	xor	rax, [rsi+rdx+0x10]
	jnz	.inequal16
	mov	rax, [rdi+rdx+0x18]
	xor	rax, [rsi+rdx+0x18]
	jnz	.inequal24
	add	rdx, 32
	jmp	.mc32
.mc16:
	cmp	rdx, -16
	jg	.mc8
	mov	rax, [rdi+rdx]
	xor	rax, [rsi+rdx]
	jnz	.inequal
	mov	rax, [rdi+rdx+8]
	xor	rax, [rsi+rdx+8]
	jnz	.inequal8
	add	rdx, 16
.mc8:
	cmp	rdx, -8
	jg	.mc4
	mov	rax, [rdi+rdx]
	xor	rax, [rsi+rdx]
	jnz	.inequal
	add	rdx, 8
.mc4:
	cmp	rdx, -4
	jg	.mc2
	mov	eax, [rdi+rdx]
	xor	eax, [rsi+rdx]
	jnz	.inequal
	add	rdx, 4
.mc2:
	cmp	rdx, -2
	jg	.mc1
	xor	eax, eax
	mov	ax, [rdi+rdx]
	xor	ax, [rsi+rdx]
	jnz	.inequal
	add	rdx, 2
.mc1:
	cmp	rdx, -1
	jg	.mc_equal
	xor	eax, eax
	mov	al, [rdi+rdx]
	xor	al, [rsi+rdx]
	jnz	.inequal
.mc_equal:
	xor	eax, eax
	ret

	; XOR found a difference in the qword at [rdi+rdx+offset].
	; BSF gives bit position of first differing bit; >>3 gives byte offset.
.inequal24:
	bsf	rax, rax
	shr	rax, 3
	add	rdx, rax
	movzx	eax, byte [rdi+rdx+0x18]
	movzx	edi, byte [rsi+rdx+0x18]
	sub	eax, edi
	ret
.inequal16:
	bsf	rax, rax
	shr	rax, 3
	add	rdx, rax
	movzx	eax, byte [rdi+rdx+0x10]
	movzx	edi, byte [rsi+rdx+0x10]
	sub	eax, edi
	ret
.inequal8:
	bsf	rax, rax
	shr	rax, 3
	add	rdx, rax
	movzx	eax, byte [rdi+rdx+8]
	movzx	edi, byte [rsi+rdx+8]
	sub	eax, edi
	ret
.inequal:
	bsf	rax, rax
	shr	rax, 3
	add	rdx, rax
	movzx	eax, byte [rdi+rdx]
	movzx	edi, byte [rsi+rdx]
	sub	eax, edi
	ret

; ------------------------------------------------------------
; memmove — copy n bytes, handles overlapping src/dst
; rdi=dst, rsi=src, rdx=n
; ------------------------------------------------------------
memmove:
	cmp	rdi, rsi
	je	.done
	jb	.forward		; dst < src: safe to copy forward
	mov	rax, rsi
	add	rax, rdx
	cmp	rdi, rax
	jae	.forward		; dst >= src+n: no overlap
	; backward copy
	push	rbx
	mov	rbx, rdi
	lea	rdi, [rdi+rdx-1]
	lea	rsi, [rsi+rdx-1]
	std
	mov	rcx, rdx
	rep movsb
	cld
	mov	rax, rbx
	pop	rbx
	ret
.forward:
	push	rbx
	mov	rbx, rdi
	mov	rcx, rdx
	rep movsb
	mov	rax, rbx
	pop	rbx
	ret
.done:
	mov	rax, rdi
	ret

; ------------------------------------------------------------
; memxor — dst[i] ^= src[i] for n bytes
; rdi=dst, rsi=src, rdx=n
; ------------------------------------------------------------
memxor:
	test	rdx, rdx
	jz	.done
	push	rbx
	xor	rbx, rbx
.loop:
	cmp	rbx, rdx
	jae	.out
	mov	al, [rsi+rbx]
	xor	[rdi+rbx], al
	inc	rbx
	jmp	.loop
.out:
	pop	rbx
.done:
	ret
