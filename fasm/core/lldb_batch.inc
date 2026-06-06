; LLDB batch output helpers for macdbg.

MACDBG_OUT_CAP equ 131072

; rdi = haystack ptr, rsi = haystack len, rdx = needle ptr, rcx = needle len
; rax = pointer to first match, or 0
mem_find:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	test	rcx, rcx
	jz	.mf_no
	xor	rbx, rbx
.mf_outer:
	mov	rax, rbx
	add	rax, rcx
	cmp	rax, r13
	ja	.mf_no
	xor	r10, r10
.mf_inner:
	cmp	r10, rcx
	jae	.mf_yes
	lea	r11, [r12 + rbx]
	mov	al, [r11 + r10]
	cmp	al, [r14 + r10]
	jne	.mf_next
	inc	r10
	jmp	.mf_inner
.mf_next:
	inc	rbx
	jmp	.mf_outer
.mf_yes:
	lea	rax, [r12 + rbx]
	jmp	.mf_done
.mf_no:
	xor	rax, rax
.mf_done:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = output ptr, rsi = output len; rax = status cstr
lldb_detect_status:
	push	rdi
	push	rsi
	lea	rdx, [lldb_pat_exc_bad]
	mov	rcx, lldb_pat_exc_bad_len
	call	mem_find
	test	rax, rax
	jnz	.lds_crashed
	pop	rsi
	pop	rdi
	push	rdi
	push	rsi
	lea	rdx, [lldb_pat_stop_reason]
	mov	rcx, lldb_pat_stop_reason_len
	call	mem_find
	test	rax, rax
	jnz	.lds_crashed
	pop	rsi
	pop	rdi
	push	rdi
	push	rsi
	lea	rdx, [lldb_pat_exited]
	mov	rcx, lldb_pat_exited_len
	call	mem_find
	test	rax, rax
	jnz	.lds_exited
	pop	rsi
	pop	rdi
	push	rdi
	push	rsi
	lea	rdx, [lldb_pat_permission]
	mov	rcx, lldb_pat_permission_len
	call	mem_find
	test	rax, rax
	jnz	.lds_perm
	pop	rsi
	pop	rdi
	lea	rax, [status_lldb_error]
	ret
.lds_crashed:
	pop	rsi
	pop	rdi
	lea	rax, [status_crashed]
	ret
.lds_exited:
	pop	rsi
	pop	rdi
	lea	rax, [status_exited]
	ret
.lds_perm:
	pop	rsi
	pop	rdi
	lea	rax, [status_permission]
	ret

lldb_pat_exc_bad db 'EXC_BAD_ACCESS'
lldb_pat_exc_bad_len = $ - lldb_pat_exc_bad
lldb_pat_stop_reason db 'stop reason = EXC'
lldb_pat_stop_reason_len = $ - lldb_pat_stop_reason
lldb_pat_exited db 'exited with status ='
lldb_pat_exited_len = $ - lldb_pat_exited
lldb_pat_permission db 'not allowed to attach'
lldb_pat_permission_len = $ - lldb_pat_permission

status_exited db 'exited', 0
status_crashed db 'crashed', 0
status_permission db 'permission_unavailable', 0
status_lldb_error db 'lldb_error', 0
