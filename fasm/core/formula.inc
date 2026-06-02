; Minimal Homebrew formula text helpers.
; Works with simple literal lines like: url "...", sha256 "...", version "...", bin.install "..."

FORMULA_OK equ 0
FORMULA_NOT_FOUND equ 1
FORMULA_OVERFLOW equ 2

; rdi = haystack ptr, rsi = haystack len, rdx = needle ptr, rcx = needle len
; r8 = output ptr, r9 = output capacity
; Copies bytes after needle until the next double quote, appends NUL.
; rax = FORMULA_OK / FORMULA_NOT_FOUND / FORMULA_OVERFLOW
formula_extract_quoted:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	xor	rbx, rbx
.feq_outer:
	mov	rax, r13
	sub	rax, r15
	cmp	rbx, rax
	ja	.feq_not_found
	xor	r10, r10
.feq_inner:
	cmp	r10, r15
	jae	.feq_found
	lea	r11, [r12 + rbx]
	mov	al, [r11 + r10]
	cmp	al, [r14 + r10]
	jne	.feq_next
	inc	r10
	jmp	.feq_inner
.feq_next:
	inc	rbx
	jmp	.feq_outer
.feq_found:
	add	rbx, r15
	xor	r10, r10
.feq_copy:
	cmp	rbx, r13
	jae	.feq_not_found
	mov	al, [r12 + rbx]
	cmp	al, '"'
	je	.feq_done
	mov	rcx, r9
	dec	rcx
	cmp	r10, rcx
	jae	.feq_overflow
	mov	[r8 + r10], al
	inc	r10
	inc	rbx
	jmp	.feq_copy
.feq_done:
	mov	byte [r8 + r10], 0
	mov	rax, FORMULA_OK
	jmp	.feq_out
.feq_not_found:
	mov	rax, FORMULA_NOT_FOUND
	jmp	.feq_out
.feq_overflow:
	mov	rax, FORMULA_OVERFLOW
.feq_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = path/string, returns rax = pointer to basename within string.
formula_basename_ptr:
	mov	rax, rdi
	xor	rcx, rcx
.fbp_loop:
	mov	dl, [rdi + rcx]
	test	dl, dl
	jz	.fbp_done
	cmp	dl, '/'
	jne	.fbp_next
	lea	rax, [rdi + rcx + 1]
.fbp_next:
	inc	rcx
	jmp	.fbp_loop
.fbp_done:
	ret

; rdi = out, rsi = app name, rdx = version
; Writes <name>-<version>-macos-x86_64.tar.gz, appends NUL. rax = out.
formula_build_pkg_name:
	push	rbx
	push	r12
	push	r13
	mov	rbx, rdi
	mov	r12, rsi
	mov	r13, rdx
	mov	rsi, r12
	call	.fbp_copy_cstr
	mov	byte [rdi], '-'
	inc	rdi
	mov	rsi, r13
	call	.fbp_copy_cstr
	lea	rsi, [formula_pkg_suffix]
	call	.fbp_copy_cstr
	mov	rax, rbx
	pop	r13
	pop	r12
	pop	rbx
	ret
.fbp_copy_cstr:
	mov	al, [rsi]
	mov	[rdi], al
	inc	rsi
	inc	rdi
	test	al, al
	jnz	.fbp_copy_cstr
	dec	rdi
	ret

formula_pkg_suffix db '-macos-x86_64.tar.gz', 0
