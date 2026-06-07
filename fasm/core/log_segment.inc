; Durable append-only log segment helpers.
;
; Record file format: u32_le payload_len, then payload bytes.
; Index file format: u64_le record byte offset in the record file.

LOGSEG_OK equ 0
LOGSEG_EOF equ 1
LOGSEG_ERR equ -1
LOGSEG_TOO_LARGE equ -2

; rdi = log path, rsi = index path, rdx = payload ptr, rcx = payload len
; rax = message offset, or LOGSEG_ERR
log_segment_append:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	rdi, r13
	open_file rdi, O_RDWR or O_CREAT, 420
	jump_if_syscall_error .lsa_err
	mov	rbx, rax
	mov	rdi, rbx
	xor	rsi, rsi
	mov	rdx, SEEK_END
	mov	rax, SYS_lseek
	syscall
	jc	.lsa_close_idx_err
	mov	r11, rax
	shr	rax, 3
	push	rax
	mov	rdi, r12
	open_file rdi, O_WRONLY or O_CREAT or O_APPEND, 420
	jump_if_syscall_error .lsa_pop_close_idx_err
	mov	r10, rax
	mov	rdi, r10
	xor	rsi, rsi
	mov	rdx, SEEK_END
	mov	rax, SYS_lseek
	syscall
	jc	.lsa_close_both_err
	mov	[logseg_u64], rax
	mov	eax, r15d
	mov	[logseg_u32], eax
	mov	rdi, r10
	lea	rsi, [logseg_u32]
	mov	rdx, 4
	call	logseg_write_all_fd
	cmp	rax, 0
	jl	.lsa_close_both_err
	mov	rdi, r10
	mov	rsi, r14
	mov	rdx, r15
	call	logseg_write_all_fd
	cmp	rax, 0
	jl	.lsa_close_both_err
	mov	rdi, rbx
	lea	rsi, [logseg_u64]
	mov	rdx, 8
	call	logseg_write_all_fd
	cmp	rax, 0
	jl	.lsa_close_both_err
	mov	rdi, r10
	close_file rdi
	mov	rdi, rbx
	close_file rdi
	pop	rax
	jmp	.lsa_out
.lsa_close_both_err:
	mov	rdi, r10
	close_file rdi
.lsa_pop_close_idx_err:
	add	rsp, 8
.lsa_close_idx_err:
	mov	rdi, rbx
	close_file rdi
.lsa_err:
	mov	rax, LOGSEG_ERR
.lsa_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = log path, rsi = index path, rdx = message offset, rcx = payload buffer,
; r8 = payload cap, r9 = payload len out*
; rax = LOGSEG_OK / LOGSEG_EOF / LOGSEG_ERR / LOGSEG_TOO_LARGE
log_segment_read:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	[logseg_len_out], r9
	mov	[logseg_cap], r8
	mov	rdi, r13
	open_file rdi, O_RDONLY, 0
	jump_if_syscall_error .lsr_eof
	mov	rbx, rax
	mov	rdi, rbx
	mov	rsi, r14
	shl	rsi, 3
	mov	rdx, SEEK_SET
	mov	rax, SYS_lseek
	syscall
	jc	.lsr_close_idx_err
	mov	rdi, rbx
	lea	rsi, [logseg_u64]
	mov	rdx, 8
	call	logseg_read_exact_fd
	cmp	rax, 0
	je	.lsr_close_idx_eof
	jl	.lsr_close_idx_err
	mov	rdi, rbx
	close_file rdi
	mov	rdi, r12
	open_file rdi, O_RDONLY, 0
	jump_if_syscall_error .lsr_err
	mov	rbx, rax
	mov	rdi, rbx
	mov	rsi, [logseg_u64]
	mov	rdx, SEEK_SET
	mov	rax, SYS_lseek
	syscall
	jc	.lsr_close_log_err
	mov	rdi, rbx
	lea	rsi, [logseg_u32]
	mov	rdx, 4
	call	logseg_read_exact_fd
	cmp	rax, 1
	jne	.lsr_close_log_err
	mov	eax, [logseg_u32]
	mov	r10, [logseg_len_out]
	mov	[r10], rax
	cmp	rax, [logseg_cap]
	ja	.lsr_close_log_too_large
	mov	rdi, rbx
	mov	rsi, r15
	mov	rdx, rax
	call	logseg_read_exact_fd
	cmp	rax, 1
	jne	.lsr_close_log_err
	mov	rdi, rbx
	close_file rdi
	mov	rax, LOGSEG_OK
	jmp	.lsr_out
.lsr_close_idx_eof:
	mov	rdi, rbx
	close_file rdi
.lsr_eof:
	mov	rax, LOGSEG_EOF
	jmp	.lsr_out
.lsr_close_idx_err:
	mov	rdi, rbx
	close_file rdi
	jmp	.lsr_err
.lsr_close_log_too_large:
	mov	rdi, rbx
	close_file rdi
	mov	rax, LOGSEG_TOO_LARGE
	jmp	.lsr_out
.lsr_close_log_err:
	mov	rdi, rbx
	close_file rdi
.lsr_err:
	mov	rax, LOGSEG_ERR
.lsr_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = fd, rsi = buf, rdx = len; rax = 0 ok, -1 error
logseg_write_all_fd:
	push	rbx
	push	r12
	push	r13
	mov	rbx, rdi
	mov	r12, rsi
	mov	r13, rdx
.lwaf_loop:
	test	r13, r13
	jz	.lwaf_ok
	mov	rdi, rbx
	mov	rsi, r12
	mov	rdx, r13
	mov	rax, SYS_write
	syscall
	jc	.lwaf_err
	test	rax, rax
	jle	.lwaf_err
	add	r12, rax
	sub	r13, rax
	jmp	.lwaf_loop
.lwaf_ok:
	xor	rax, rax
	jmp	.lwaf_out
.lwaf_err:
	mov	rax, -1
.lwaf_out:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = fd, rsi = buf, rdx = len; rax = 1 ok, 0 EOF, -1 error
logseg_read_exact_fd:
	push	rbx
	push	r12
	push	r13
	mov	rbx, rdi
	mov	r12, rsi
	mov	r13, rdx
.lref_loop:
	test	r13, r13
	jz	.lref_ok
	mov	rdi, rbx
	mov	rsi, r12
	mov	rdx, r13
	mov	rax, SYS_read
	syscall
	jc	.lref_err
	test	rax, rax
	jz	.lref_eof
	add	r12, rax
	sub	r13, rax
	jmp	.lref_loop
.lref_ok:
	mov	rax, 1
	jmp	.lref_out
.lref_eof:
	xor	rax, rax
	jmp	.lref_out
.lref_err:
	mov	rax, -1
.lref_out:
	pop	r13
	pop	r12
	pop	rbx
	ret

macro log_segment_bss
{
	logseg_u64 dq ?
	logseg_u32 dd ?
	logseg_cap dq ?
	logseg_len_out dq ?
}
