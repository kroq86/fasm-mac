; Streaming SHA-256 implementation.
; Context memory: SHA256_CTX_SIZE bytes. Digest output: 32 bytes.

SHA256_CTX_STATE equ 0
SHA256_CTX_BITLEN_LO equ 32
SHA256_CTX_BITLEN_HI equ 36
SHA256_CTX_BUFLEN equ 40
SHA256_CTX_BUF equ 48
SHA256_CTX_SIZE equ 112

; rdi = ctx
sha256_init:
	mov	dword [rdi + SHA256_CTX_STATE + 0], 06a09e667h
	mov	dword [rdi + SHA256_CTX_STATE + 4], 0bb67ae85h
	mov	dword [rdi + SHA256_CTX_STATE + 8], 03c6ef372h
	mov	dword [rdi + SHA256_CTX_STATE + 12], 0a54ff53ah
	mov	dword [rdi + SHA256_CTX_STATE + 16], 0510e527fh
	mov	dword [rdi + SHA256_CTX_STATE + 20], 09b05688ch
	mov	dword [rdi + SHA256_CTX_STATE + 24], 01f83d9abh
	mov	dword [rdi + SHA256_CTX_STATE + 28], 05be0cd19h
	mov	dword [rdi + SHA256_CTX_BITLEN_LO], 0
	mov	dword [rdi + SHA256_CTX_BITLEN_HI], 0
	mov	qword [rdi + SHA256_CTX_BUFLEN], 0
	ret

; rdi = ctx, rsi = data, rdx = len
sha256_update:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
.su_loop:
	test	r14, r14
	jz	.su_done
	mov	rbx, [r12 + SHA256_CTX_BUFLEN]
	mov	rax, 64
	sub	rax, rbx
	cmp	rax, r14
	jbe	.su_take_ok
	mov	rax, r14
.su_take_ok:
	mov	r15, rax
	lea	rdi, [r12 + SHA256_CTX_BUF + rbx]
	mov	rsi, r13
	mov	rcx, r15
	rep	movsb
	add	[r12 + SHA256_CTX_BUFLEN], r15
	add	r13, r15
	sub	r14, r15
	mov	rax, r15
	shl	eax, 3
	add	dword [r12 + SHA256_CTX_BITLEN_LO], eax
	adc	dword [r12 + SHA256_CTX_BITLEN_HI], 0
	cmp	qword [r12 + SHA256_CTX_BUFLEN], 64
	jne	.su_loop
	mov	rdi, r12
	lea	rsi, [r12 + SHA256_CTX_BUF]
	call	sha256_transform
	mov	qword [r12 + SHA256_CTX_BUFLEN], 0
	jmp	.su_loop
.su_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = ctx, rsi = digest out (32 bytes)
sha256_final:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rbx, [r12 + SHA256_CTX_BUFLEN]
	mov	byte [r12 + SHA256_CTX_BUF + rbx], 080h
	inc	rbx
	cmp	rbx, 56
	jbe	.sf_pad56
.sf_pad64:
	cmp	rbx, 64
	jae	.sf_block
	mov	byte [r12 + SHA256_CTX_BUF + rbx], 0
	inc	rbx
	jmp	.sf_pad64
.sf_block:
	mov	rdi, r12
	lea	rsi, [r12 + SHA256_CTX_BUF]
	call	sha256_transform
	xor	rbx, rbx
.sf_zero56:
	cmp	rbx, 56
	jae	.sf_len
	mov	byte [r12 + SHA256_CTX_BUF + rbx], 0
	inc	rbx
	jmp	.sf_zero56
.sf_pad56:
	cmp	rbx, 56
	jae	.sf_len
	mov	byte [r12 + SHA256_CTX_BUF + rbx], 0
	inc	rbx
	jmp	.sf_pad56
.sf_len:
	mov	eax, dword [r12 + SHA256_CTX_BITLEN_HI]
	bswap	eax
	mov	dword [r12 + SHA256_CTX_BUF + 56], eax
	mov	eax, dword [r12 + SHA256_CTX_BITLEN_LO]
	bswap	eax
	mov	dword [r12 + SHA256_CTX_BUF + 60], eax
	mov	rdi, r12
	lea	rsi, [r12 + SHA256_CTX_BUF]
	call	sha256_transform
	xor	rbx, rbx
.sf_out:
	cmp	rbx, 8
	jae	.sf_done
	mov	eax, dword [r12 + SHA256_CTX_STATE + rbx * 4]
	bswap	eax
	mov	dword [r13 + rbx * 4], eax
	inc	rbx
	jmp	.sf_out
.sf_done:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = ctx, rsi = 64-byte block
sha256_transform:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 320
	mov	r12, rdi
	mov	r13, rsi
	xor	rbx, rbx
.st_load:
	cmp	rbx, 16
	jae	.st_extend
	mov	eax, dword [r13 + rbx * 4]
	bswap	eax
	mov	dword [rsp + rbx * 4], eax
	inc	rbx
	jmp	.st_load
.st_extend:
	cmp	rbx, 64
	jae	.st_init_work
	mov	eax, dword [rsp + rbx * 4 - 60]
	mov	edx, eax
	ror	edx, 7
	mov	ecx, eax
	ror	ecx, 18
	xor	edx, ecx
	mov	ecx, eax
	shr	ecx, 3
	xor	edx, ecx
	mov	eax, dword [rsp + rbx * 4 - 8]
	mov	ecx, eax
	ror	ecx, 17
	mov	r8d, eax
	ror	r8d, 19
	xor	ecx, r8d
	mov	r8d, eax
	shr	r8d, 10
	xor	ecx, r8d
	add	edx, ecx
	add	edx, dword [rsp + rbx * 4 - 64]
	add	edx, dword [rsp + rbx * 4 - 28]
	mov	dword [rsp + rbx * 4], edx
	inc	rbx
	jmp	.st_extend
.st_init_work:
	xor	rbx, rbx
.st_state_copy:
	cmp	rbx, 8
	jae	.st_rounds
	mov	eax, dword [r12 + SHA256_CTX_STATE + rbx * 4]
	mov	dword [rsp + 256 + rbx * 4], eax
	inc	rbx
	jmp	.st_state_copy
.st_rounds:
	xor	rbx, rbx
.st_round:
	cmp	rbx, 64
	jae	.st_add_state
	mov	eax, dword [rsp + 256 + 16] ; e
	mov	edx, eax
	ror	edx, 6
	mov	ecx, eax
	ror	ecx, 11
	xor	edx, ecx
	mov	ecx, eax
	ror	ecx, 25
	xor	edx, ecx
	mov	eax, dword [rsp + 256 + 16] ; e
	mov	ecx, dword [rsp + 256 + 20] ; f
	and	ecx, eax
	mov	r8d, eax
	not	r8d
	and	r8d, dword [rsp + 256 + 24] ; g
	xor	ecx, r8d
	add	edx, ecx
	add	edx, dword [rsp + 256 + 28] ; h
	add	edx, dword [sha256_k + rbx * 4]
	add	edx, dword [rsp + rbx * 4]
	mov	dword [rsp + 288], edx ; t1
	mov	eax, dword [rsp + 256] ; a
	mov	edx, eax
	ror	edx, 2
	mov	ecx, eax
	ror	ecx, 13
	xor	edx, ecx
	mov	ecx, eax
	ror	ecx, 22
	xor	edx, ecx
	mov	eax, dword [rsp + 256] ; a
	mov	ecx, eax
	and	ecx, dword [rsp + 256 + 4] ; b
	mov	r8d, eax
	and	r8d, dword [rsp + 256 + 8] ; c
	xor	ecx, r8d
	mov	r8d, dword [rsp + 256 + 4]
	and	r8d, dword [rsp + 256 + 8]
	xor	ecx, r8d
	add	edx, ecx ; t2
	mov	eax, dword [rsp + 256 + 24]
	mov	dword [rsp + 256 + 28], eax
	mov	eax, dword [rsp + 256 + 20]
	mov	dword [rsp + 256 + 24], eax
	mov	eax, dword [rsp + 256 + 16]
	mov	dword [rsp + 256 + 20], eax
	mov	eax, dword [rsp + 256 + 12]
	add	eax, dword [rsp + 288]
	mov	dword [rsp + 256 + 16], eax
	mov	eax, dword [rsp + 256 + 8]
	mov	dword [rsp + 256 + 12], eax
	mov	eax, dword [rsp + 256 + 4]
	mov	dword [rsp + 256 + 8], eax
	mov	eax, dword [rsp + 256]
	mov	dword [rsp + 256 + 4], eax
	mov	eax, dword [rsp + 288]
	add	eax, edx
	mov	dword [rsp + 256], eax
	inc	rbx
	jmp	.st_round
.st_add_state:
	xor	rbx, rbx
.st_add_loop:
	cmp	rbx, 8
	jae	.st_done
	mov	eax, dword [rsp + 256 + rbx * 4]
	add	dword [r12 + SHA256_CTX_STATE + rbx * 4], eax
	inc	rbx
	jmp	.st_add_loop
.st_done:
	add	rsp, 320
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

sha256_k dd 0428a2f98h, 071374491h, 0b5c0fbcfh, 0e9b5dba5h
	 dd 03956c25bh, 059f111f1h, 0923f82a4h, 0ab1c5ed5h
	 dd 0d807aa98h, 012835b01h, 0243185beh, 0550c7dc3h
	 dd 072be5d74h, 080deb1feh, 09bdc06a7h, 0c19bf174h
	 dd 0e49b69c1h, 0efbe4786h, 00fc19dc6h, 0240ca1cch
	 dd 02de92c6fh, 04a7484aah, 05cb0a9dch, 076f988dah
	 dd 0983e5152h, 0a831c66dh, 0b00327c8h, 0bf597fc7h
	 dd 0c6e00bf3h, 0d5a79147h, 006ca6351h, 014292967h
	 dd 027b70a85h, 02e1b2138h, 04d2c6dfch, 053380d13h
	 dd 0650a7354h, 0766a0abbh, 081c2c92eh, 092722c85h
	 dd 0a2bfe8a1h, 0a81a664bh, 0c24b8b70h, 0c76c51a3h
	 dd 0d192e819h, 0d6990624h, 0f40e3585h, 0106aa070h
	 dd 019a4c116h, 01e376c08h, 02748774ch, 034b0bcb5h
	 dd 0391c0cb3h, 04ed8aa4ah, 05b9cca4fh, 0682e6ff3h
	 dd 0748f82eeh, 078a5636fh, 084c87814h, 08cc70208h
	 dd 090befffah, 0a4506cebh, 0bef9a3f7h, 0c67178f2h
