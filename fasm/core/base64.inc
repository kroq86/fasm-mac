; Base64 encode / decode for raw byte buffers.
; Encode algorithm adapted from HeavyThing base64_latin1.inc (Jeff Marrison, GPLv2+).
; https://2ton.com.au/HeavyThing/
;
; base64_encode  rdi=src  rsi=byte_count  rdx=dst  →  rax=chars_written
; base64_decode  rdi=src  rsi=char_count  rdx=dst  →  rax=bytes_written
;
; dst must be pre-allocated:
;   encode: ceil(n/3)*4 bytes
;   decode: roughly ceil(n/4)*3 bytes (actual count in rax)
;
; No line breaks inserted. Whitespace/CR/LF in decode input is skipped.
; Returns 0 for empty input.

; ------------------------------------------------------------
; base64_encode
; rdi=src, rsi=byte_count, rdx=dst  →  rax=chars_written
; Process 3 input bytes → 4 base64 chars.
; ------------------------------------------------------------
base64_encode:
	test	rsi, rsi
	jz	.nothing

	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15

	mov	r12, rdi		; src pointer
	mov	r13, rsi		; remaining bytes
	mov	r14, rdx		; dst pointer
	xor	r15d, r15d		; chars written

.loop:
	cmp	r13, 3
	jae	.three
	cmp	r13, 2
	je	.two

	; ---- one byte left ----
	movzx	ebx, byte [r12]
	mov	esi, ebx
	shr	esi, 2
	and	esi, 0x3f
	movzx	eax, byte [base64_enc_table + rsi]
	mov	[r14+r15], al
	shl	ebx, 4
	and	ebx, 0x3f
	movzx	eax, byte [base64_enc_table + rbx]
	mov	[r14+r15+1], al
	mov	byte [r14+r15+2], '='
	mov	byte [r14+r15+3], '='
	add	r15, 4
	jmp	.done

.two:
	; ---- two bytes left ----
	movzx	ebx, word [r12]		; load 2 bytes little-endian: byte0 in bl, byte1 in bh
	mov	esi, ebx
	shr	esi, 2
	and	esi, 0x3f
	movzx	eax, byte [base64_enc_table + rsi]
	mov	[r14+r15], al

	mov	esi, ebx
	mov	ecx, ebx
	shl	esi, 4
	shr	ecx, 12
	and	esi, 0x3f
	and	ecx, 0xf
	or	esi, ecx
	movzx	eax, byte [base64_enc_table + rsi]
	mov	[r14+r15+1], al

	shr	ebx, 8
	shl	ebx, 2
	and	ebx, 0x3f
	movzx	eax, byte [base64_enc_table + rbx]
	mov	[r14+r15+2], al
	mov	byte [r14+r15+3], '='
	add	r15, 4
	jmp	.done

.three:
	; ---- full 3-byte group → 4 chars ----
	mov	ebx, dword [r12]	; read 4 bytes (4th ignored)

	; char 0: bits 7..2 of byte 0
	mov	esi, ebx
	shr	esi, 2
	and	esi, 0x3f
	movzx	eax, byte [base64_enc_table + rsi]
	mov	[r14+r15], al

	; char 1: bits 1..0 of byte 0 | bits 7..4 of byte 1
	mov	esi, ebx
	mov	ecx, ebx
	shl	esi, 4
	shr	ecx, 12
	and	esi, 0x3f
	and	ecx, 0xf
	or	esi, ecx
	movzx	eax, byte [base64_enc_table + rsi]
	mov	[r14+r15+1], al

	; char 2: bits 3..0 of byte 1 | bits 7..6 of byte 2
	mov	esi, ebx
	mov	ecx, ebx
	shl	esi, 2
	shr	ecx, 8
	shr	ebx, 8			; advance to byte 2
	and	esi, 0x3f
	shr	ecx, 6
	and	ecx, 0x3
	or	esi, ecx
	movzx	eax, byte [base64_enc_table + rsi]
	mov	[r14+r15+2], al

	; char 3: bits 5..0 of byte 2
	and	ebx, 0x3f
	movzx	eax, byte [base64_enc_table + rbx]
	mov	[r14+r15+3], al

	add	r15, 4
	add	r12, 3
	sub	r13, 3
	jnz	.loop

.done:
	mov	rax, r15
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
.nothing:
	xor	eax, eax
	ret

; ------------------------------------------------------------
; base64_decode
; rdi=src, rsi=char_count, rdx=dst  →  rax=bytes_written
; Standard 4-groups-at-a-time decode:
;   byte0 = (c0<<2)|(c1>>4)
;   byte1 = (c1<<4)|(c2>>2)
;   byte2 = (c2<<6)| c3
; Skips whitespace. Stops on '=' padding or invalid char.
; ------------------------------------------------------------
base64_decode:
	test	rsi, rsi
	jz	.nothing

	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15

	mov	r12, rdi		; src pointer
	mov	r13, rsi		; remaining chars
	mov	r14, rdx		; dst start (saved for byte count)
	mov	r15, rdx		; dst write pointer

.group:
	; collect 4 decoded 6-bit values into r8b..r11b (or stop early)
	call	.getchar
	cmp	eax, -1
	je	.finish
	mov	r8d, eax		; c0

	call	.getchar
	cmp	eax, -1
	je	.finish
	mov	r9d, eax		; c1

	; byte 0 = (c0<<2)|(c1>>4)
	mov	ecx, r8d
	shl	ecx, 2
	mov	edx, r9d
	shr	edx, 4
	or	ecx, edx
	mov	[r15], cl
	inc	r15

	call	.getchar
	cmp	eax, -1
	je	.finish		; c2 missing → done after 1 byte
	mov	r10d, eax		; c2

	; byte 1 = (c1<<4)|(c2>>2)
	mov	ecx, r9d
	shl	ecx, 4
	mov	edx, r10d
	shr	edx, 2
	and	ecx, 0xf0
	and	edx, 0x0f
	or	ecx, edx
	mov	[r15], cl
	inc	r15

	call	.getchar
	cmp	eax, -1
	je	.finish		; c3 missing → done after 2 bytes
	mov	r11d, eax		; c3

	; byte 2 = (c2<<6)|c3
	mov	ecx, r10d
	shl	ecx, 6
	and	r11d, 0x3f
	or	ecx, r11d
	mov	[r15], cl
	inc	r15

	jmp	.group

.finish:
	mov	rax, r15
	sub	rax, r14		; bytes written
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
.nothing:
	xor	eax, eax
	ret

; internal: read next valid base64 char, skip whitespace
; returns decoded 6-bit value in eax, or -1 on padding/invalid/end
.getchar:
.gc_again:
	test	r13, r13
	jz	.gc_eof
	movzx	ecx, byte [r12]
	inc	r12
	dec	r13
	cmp	ecx, ' '
	je	.gc_again
	cmp	ecx, 0x0d
	je	.gc_again
	cmp	ecx, 0x0a
	je	.gc_again
	cmp	ecx, 0x09
	je	.gc_again
	cmp	ecx, '='
	je	.gc_eof
	cmp	ecx, 128
	jae	.gc_eof
	mov	eax, dword [base64_dec_table + rcx*4]
	ret
.gc_eof:
	mov	eax, -1
	ret

; ------------------------------------------------------------
; Tables (include in readable segment alongside functions)
; ------------------------------------------------------------
base64_enc_table  db  'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

; 128-entry decode table (dword per ASCII char, -1 = invalid)
base64_dec_table:
	dd  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1  ; 0x00-0x0f
	dd  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1  ; 0x10-0x1f
	dd  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63  ; 0x20-0x2f  (+/)
	dd  52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1  ; 0x30-0x3f  (0-9)
	dd  -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14 ; 0x40-0x4f  (A-O)
	dd  15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1  ; 0x50-0x5f  (P-Z)
	dd  -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40  ; 0x60-0x6f  (a-o)
	dd  41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1  ; 0x70-0x7f  (p-z)
