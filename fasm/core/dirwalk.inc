; Directory tree walk (macOS x86_64 v0).
; Requires platform.inc, str.inc, file.inc.
;
; App BSS: dirwalk_bss macro.
; dirwalk_foreach callback: rdi=path, rsi=type, rdx=size, rcx=user
; Types: DIRWALK_TYPE_FILE, DIRWALK_TYPE_DIR

DIRWALK_OK equ 0
DIRWALK_ERR equ -1
DIRWALK_ERR_NOTDIR equ -2

DIRWALK_TYPE_FILE equ 1
DIRWALK_TYPE_DIR equ 2

DIRWALK_PATH_MAX equ 4096
DIRWALK_DIRENT_BUF equ 8192
DIRWALK_SAVE_SIZE equ DIRWALK_DIRENT_BUF + 40

macro dirwalk_bss
{
	dirwalk_path rb DIRWALK_PATH_MAX
	align 16
	dirwalk_dirent rb DIRWALK_DIRENT_BUF
	align 16
	dirwalk_stat rb STAT64_SIZE
	dirwalk_base dq ?
	dirwalk_fd dq ?
	dirwalk_callback dq ?
	dirwalk_user dq ?
}

; rdi = path, rsi = stat buf (STAT64_SIZE bytes)
; rax = DIRWALK_OK / DIRWALK_ERR
dirwalk_lstat:
	mov	rax, SYS_stat64
	syscall
	jc	.dls_err
	mov	rax, DIRWALK_OK
	ret
.dls_err:
	mov	rax, DIRWALK_ERR
	ret

; rdi = dest scratch buffer (DIRWALK_PATH_MAX bytes), rsi = parent dir, rdx = name
; rax = dest ptr or 0 on overflow
dirwalk_path_join:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rdi, rsi
	call	str_len
	mov	rbx, rax
	mov	rcx, rbx
	mov	rdi, r12
	mov	rsi, r13
	rep	movsb
	mov	rax, DIRWALK_PATH_MAX
	sub	rax, 4
	cmp	rbx, rax
	jae	.dpj_fail
	mov	byte [rdi], '/'
	inc	rdi
	mov	rsi, rdx
.dpj_name:
	mov	al, [rsi]
	mov	[rdi], al
	test	al, al
	jz	.dpj_ok
	lea	rax, [r12 + DIRWALK_PATH_MAX - 1]
	cmp	rdi, rax
	jae	.dpj_fail
	inc	rsi
	inc	rdi
	jmp	.dpj_name
.dpj_ok:
	mov	rax, r12
	jmp	.dpj_out
.dpj_fail:
	xor	rax, rax
.dpj_out:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = dir path
dirwalk_descend:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, DIRWALK_PATH_MAX
	mov	r12, rsp
	mov	rsi, rdi
	mov	rdi, r12
.dd_copy_parent:
	mov	al, [rsi]
	mov	[rdi], al
	inc	rsi
	inc	rdi
	test	al, al
	jz	.dd_parent_ok
	lea	rax, [r12 + DIRWALK_PATH_MAX]
	cmp	rdi, rax
	jb	.dd_copy_parent
	jmp	.dd_fail_with_stack
.dd_parent_ok:
	mov	rdi, r12
	call	file_open_read
	cmp	rax, 0
	jl	.dd_fail_with_stack
	mov	[dirwalk_fd], rax
	mov	qword [dirwalk_base], 0
.dd_read:
	mov	rdi, [dirwalk_fd]
	lea	rsi, [dirwalk_dirent]
	mov	rdx, DIRWALK_DIRENT_BUF
	lea	r10, [dirwalk_base]
	mov	rax, SYS_getdirentries64
	syscall
	jc	.dd_close_fail
	test	rax, rax
	jz	.dd_close_ok
	mov	r14, rax
	xor	r15, r15
.dd_entry:
	cmp	r15, r14
	jae	.dd_read
	movzx	r11d, word [dirwalk_dirent + r15 + DIRENT_RECLEN_OFF]
	test	r11, r11
	jz	.dd_read
	lea	rdi, [dirwalk_dirent + r15 + DIRENT_NAME_OFF]
	cmp	byte [rdi], '.'
	jne	.dd_notdot
	cmp	byte [rdi + 1], 0
	je	.dd_next
	cmp	byte [rdi + 1], '.'
	jne	.dd_notdot
	cmp	byte [rdi + 2], 0
	je	.dd_next
.dd_notdot:
	lea	rdi, [dirwalk_path]
	mov	rsi, r12
	lea	rdx, [dirwalk_dirent + r15 + DIRENT_NAME_OFF]
	call	dirwalk_path_join
	test	rax, rax
	jz	.dd_close_fail
	movzx	eax, byte [dirwalk_dirent + r15 + DIRENT_TYPE_OFF]
	cmp	al, DT_DIR
	je	.dd_isdir
	cmp	al, DT_REG
	je	.dd_isfile
	test	al, al
	jnz	.dd_next
	push	r14
	push	r15
	lea	rsi, [dirwalk_stat]
	lea	rdi, [dirwalk_path]
	call	dirwalk_lstat
	pop	r15
	pop	r14
	cmp	rax, DIRWALK_OK
	jne	.dd_next
	mov	eax, dword [dirwalk_stat + STAT64_MODE_OFF]
	and	eax, S_IFMT
	cmp	eax, S_IFDIR
	je	.dd_isdir
	cmp	eax, S_IFREG
	jne	.dd_next
.dd_isfile:
	xor	rdx, rdx
	push	r14
	push	r15
	lea	rsi, [dirwalk_stat]
	lea	rdi, [dirwalk_path]
	call	dirwalk_lstat
	pop	r15
	pop	r14
	cmp	rax, DIRWALK_OK
	jne	.dd_cb_file
	mov	rdx, qword [dirwalk_stat + STAT64_SIZE_OFF]
.dd_cb_file:
	push	r14
	push	r15
	push	rdx
	lea	rdi, [dirwalk_path]
	mov	rsi, DIRWALK_TYPE_FILE
	pop	rdx
	mov	rcx, [dirwalk_user]
	call	qword [dirwalk_callback]
	pop	r15
	pop	r14
	jmp	.dd_next
.dd_isdir:
	push	r14
	push	r15
	lea	rdi, [dirwalk_path]
	mov	rsi, DIRWALK_TYPE_DIR
	xor	rdx, rdx
	mov	rcx, [dirwalk_user]
	call	qword [dirwalk_callback]
	pop	r15
	pop	r14
	movzx	eax, byte [dirwalk_dirent + r15 + DIRENT_TYPE_OFF]
	cmp	al, DT_LNK
	je	.dd_next
	sub	rsp, DIRWALK_SAVE_SIZE
	mov	[rsp], r14
	mov	[rsp + 8], r15
	mov	rax, [dirwalk_base]
	mov	[rsp + 16], rax
	mov	rax, [dirwalk_fd]
	mov	[rsp + 24], rax
	lea	rsi, [dirwalk_dirent]
	lea	rdi, [rsp + 40]
	mov	rcx, DIRWALK_DIRENT_BUF / 8
	rep	movsq
	lea	rdi, [dirwalk_path]
	call	dirwalk_descend
	mov	r11, rax
	mov	r14, [rsp]
	mov	r15, [rsp + 8]
	mov	rax, [rsp + 16]
	mov	[dirwalk_base], rax
	mov	rax, [rsp + 24]
	mov	[dirwalk_fd], rax
	lea	rsi, [rsp + 40]
	lea	rdi, [dirwalk_dirent]
	mov	rcx, DIRWALK_DIRENT_BUF / 8
	rep	movsq
	add	rsp, DIRWALK_SAVE_SIZE
	cmp	r11, DIRWALK_OK
	jne	.dd_close_fail
	jmp	.dd_next
.dd_next:
	movzx	r11d, word [dirwalk_dirent + r15 + DIRENT_RECLEN_OFF]
	add	r15, r11
	jmp	.dd_entry
.dd_close_ok:
	mov	rdi, [dirwalk_fd]
	call	file_close
	mov	rax, DIRWALK_OK
	jmp	.dd_out
.dd_close_fail:
	mov	rdi, [dirwalk_fd]
	call	file_close
.dd_fail:
	mov	rax, DIRWALK_ERR
.dd_out:
	add	rsp, DIRWALK_PATH_MAX
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
.dd_fail_with_stack:
	mov	rax, DIRWALK_ERR
	jmp	.dd_out

; rdi = root, rsi = callback, rdx = user
; rax = DIRWALK_OK / DIRWALK_ERR / DIRWALK_ERR_NOTDIR
dirwalk_foreach:
	push	rbx
	push	r12
	mov	[dirwalk_callback], rsi
	mov	[dirwalk_user], rdx
	mov	r12, rdi
	lea	rsi, [dirwalk_stat]
	call	dirwalk_lstat
	cmp	rax, DIRWALK_OK
	jne	.dfe_err
	mov	eax, dword [dirwalk_stat + STAT64_MODE_OFF]
	and	eax, S_IFMT
	cmp	eax, S_IFDIR
	jne	.dfe_notdir
	mov	rdi, r12
	call	dirwalk_descend
	jmp	.dfe_out
.dfe_notdir:
	mov	rax, DIRWALK_ERR_NOTDIR
	jmp	.dfe_out
.dfe_err:
	mov	rax, DIRWALK_ERR
.dfe_out:
	pop	r12
	pop	rbx
	ret
