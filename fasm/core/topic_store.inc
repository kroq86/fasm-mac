; Topic and consumer-offset path helpers for logbus.

TOPIC_STORE_OK equ 0
TOPIC_STORE_ERR equ -1
TOPIC_STORE_PATH_MAX equ 4096

; rdi = name cstr; rax = 1 valid, 0 invalid
topic_name_valid:
	push	rbx
	xor	rbx, rbx
.tnv_loop:
	mov	al, [rdi + rbx]
	test	al, al
	jz	.tnv_done
	cmp	al, '0'
	jb	.tnv_check_upper
	cmp	al, '9'
	jbe	.tnv_next
.tnv_check_upper:
	cmp	al, 'A'
	jb	.tnv_check_lower
	cmp	al, 'Z'
	jbe	.tnv_next
.tnv_check_lower:
	cmp	al, 'a'
	jb	.tnv_check_extra
	cmp	al, 'z'
	jbe	.tnv_next
.tnv_check_extra:
	cmp	al, '_'
	je	.tnv_next
	cmp	al, '-'
	je	.tnv_next
	cmp	al, '.'
	je	.tnv_next
	xor	rax, rax
	jmp	.tnv_out
.tnv_next:
	inc	rbx
	jmp	.tnv_loop
.tnv_done:
	test	rbx, rbx
	jz	.tnv_bad_empty
	mov	rax, 1
	jmp	.tnv_out
.tnv_bad_empty:
	xor	rax, rax
.tnv_out:
	pop	rbx
	ret

; rdi = root dir, rsi = scratch path
; rax = TOPIC_STORE_OK / TOPIC_STORE_ERR
topic_store_prepare:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rdi, r12
	call	topic_mkdir_ignore
	mov	rdi, r13
	mov	rsi, r12
	call	topic_copy_cstr
	cmp	rax, 0
	jl	.tsp_err
	mov	rdi, r13
	lea	rsi, [topic_topics_suffix]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tsp_err
	mov	rdi, r13
	call	topic_mkdir_ignore
	mov	rdi, r13
	mov	rsi, r12
	call	topic_copy_cstr
	cmp	rax, 0
	jl	.tsp_err
	mov	rdi, r13
	lea	rsi, [topic_offsets_suffix]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tsp_err
	mov	rdi, r13
	call	topic_mkdir_ignore
	mov	rax, TOPIC_STORE_OK
	jmp	.tsp_out
.tsp_err:
	mov	rax, TOPIC_STORE_ERR
.tsp_out:
	pop	r13
	pop	r12
	ret

; rdi=root, rsi=topic, rdx=log_out, rcx=idx_out, r8=dir_scratch
; rax = TOPIC_STORE_OK / TOPIC_STORE_ERR
topic_build_segment_paths:
	mov	r9, r8
	mov	r8, rcx
	mov	rcx, rdx
	xor	rdx, rdx
	jmp	topic_build_segment_paths_base

; rdi=root, rsi=topic, rdx=base offset, rcx=log_out, r8=idx_out, r9=dir_scratch
; rax = TOPIC_STORE_OK / TOPIC_STORE_ERR
topic_build_segment_paths_base:
	push	r12
	push	r13
	push	r14
	push	r15
	push	rbx
	mov	r12, rdi
	mov	r13, rsi
	mov	rbx, rdx
	mov	r14, rcx
	mov	r15, r8
	mov	[topic_scratch], r9
	mov	rdi, r13
	call	topic_name_valid
	test	rax, rax
	jz	.tbspb_err
	mov	rdi, [topic_scratch]
	mov	rsi, r12
	call	topic_copy_cstr
	cmp	rax, 0
	jl	.tbspb_err
	mov	rdi, [topic_scratch]
	lea	rsi, [topic_topics_suffix]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbspb_err
	mov	rdi, [topic_scratch]
	lea	rsi, [topic_slash]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbspb_err
	mov	rdi, [topic_scratch]
	mov	rsi, r13
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbspb_err
	mov	rdi, [topic_scratch]
	call	topic_mkdir_ignore
	mov	rdi, r14
	mov	rsi, [topic_scratch]
	call	topic_copy_cstr
	cmp	rax, 0
	jl	.tbspb_err
	mov	rdi, r14
	lea	rsi, [topic_slash]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbspb_err
	mov	rdi, r14
	mov	rsi, rbx
	call	topic_append_u20
	cmp	rax, 0
	jl	.tbspb_err
	mov	rdi, r14
	lea	rsi, [topic_log_ext]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbspb_err
	mov	rdi, r15
	mov	rsi, [topic_scratch]
	call	topic_copy_cstr
	cmp	rax, 0
	jl	.tbspb_err
	mov	rdi, r15
	lea	rsi, [topic_slash]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbspb_err
	mov	rdi, r15
	mov	rsi, rbx
	call	topic_append_u20
	cmp	rax, 0
	jl	.tbspb_err
	mov	rdi, r15
	lea	rsi, [topic_idx_ext]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbspb_err
	mov	rax, TOPIC_STORE_OK
	jmp	.tbspb_out
.tbspb_err:
	mov	rax, TOPIC_STORE_ERR
.tbspb_out:
	pop	rbx
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	ret

; rdi=root, rsi=topic, rdx=out, rcx=dir_scratch
; rax = TOPIC_STORE_OK / TOPIC_STORE_ERR
topic_build_global_idx_path:
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	[topic_scratch], rcx
	mov	rdi, r13
	call	topic_name_valid
	test	rax, rax
	jz	.tbgip_err
	mov	rdi, [topic_scratch]
	mov	rsi, r12
	call	topic_copy_cstr
	cmp	rax, 0
	jl	.tbgip_err
	mov	rdi, [topic_scratch]
	lea	rsi, [topic_topics_suffix]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbgip_err
	mov	rdi, [topic_scratch]
	lea	rsi, [topic_slash]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbgip_err
	mov	rdi, [topic_scratch]
	mov	rsi, r13
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbgip_err
	mov	rdi, [topic_scratch]
	call	topic_mkdir_ignore
	mov	rdi, r14
	mov	rsi, [topic_scratch]
	call	topic_copy_cstr
	cmp	rax, 0
	jl	.tbgip_err
	mov	rdi, r14
	lea	rsi, [topic_global_idx_suffix]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbgip_err
	mov	rax, TOPIC_STORE_OK
	jmp	.tbgip_out
.tbgip_err:
	mov	rax, TOPIC_STORE_ERR
.tbgip_out:
	pop	r14
	pop	r13
	pop	r12
	ret

; rdi=root, rsi=group, rdx=topic, rcx=out
; rax = TOPIC_STORE_OK / TOPIC_STORE_ERR
topic_build_offset_path:
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	rdi, r13
	call	topic_name_valid
	test	rax, rax
	jz	.tbop_err
	mov	rdi, r14
	call	topic_name_valid
	test	rax, rax
	jz	.tbop_err
	mov	rdi, r15
	mov	rsi, r12
	call	topic_copy_cstr
	cmp	rax, 0
	jl	.tbop_err
	mov	rdi, r15
	lea	rsi, [topic_offsets_suffix]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbop_err
	mov	rdi, r15
	lea	rsi, [topic_slash]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbop_err
	mov	rdi, r15
	mov	rsi, r13
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbop_err
	mov	rdi, r15
	lea	rsi, [topic_dot]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbop_err
	mov	rdi, r15
	mov	rsi, r14
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbop_err
	mov	rdi, r15
	lea	rsi, [topic_offset_suffix]
	call	topic_append_cstr
	cmp	rax, 0
	jl	.tbop_err
	mov	rax, TOPIC_STORE_OK
	jmp	.tbop_out
.tbop_err:
	mov	rax, TOPIC_STORE_ERR
.tbop_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	ret

topic_mkdir_ignore:
	mov	rsi, 493
	mov	rax, SYS_mkdir
	syscall
	xor	rax, rax
	ret

; rdi=dst, rsi=src; rax=0 ok, -1 too long
topic_copy_cstr:
	push	rbx
	xor	rbx, rbx
.tcc_loop:
	cmp	rbx, TOPIC_STORE_PATH_MAX - 1
	jae	.tcc_err
	mov	al, [rsi + rbx]
	mov	[rdi + rbx], al
	test	al, al
	jz	.tcc_ok
	inc	rbx
	jmp	.tcc_loop
.tcc_ok:
	xor	rax, rax
	jmp	.tcc_out
.tcc_err:
	mov	rax, -1
.tcc_out:
	pop	rbx
	ret

; rdi=dst cstr, rsi=suffix cstr; rax=0 ok, -1 too long
topic_append_cstr:
	push	rbx
	push	r12
	mov	r12, rdi
	xor	rbx, rbx
.tac_len:
	cmp	byte [r12 + rbx], 0
	je	.tac_copy
	inc	rbx
	cmp	rbx, TOPIC_STORE_PATH_MAX - 1
	jae	.tac_err
	jmp	.tac_len
.tac_copy:
	mov	al, [rsi]
	cmp	rbx, TOPIC_STORE_PATH_MAX - 1
	jae	.tac_err
	mov	[r12 + rbx], al
	test	al, al
	jz	.tac_ok
	inc	rbx
	inc	rsi
	jmp	.tac_copy
.tac_ok:
	xor	rax, rax
	jmp	.tac_out
.tac_err:
	mov	rax, -1
.tac_out:
	pop	r12
	pop	rbx
	ret

; rdi=dst cstr, rsi=uint; append 20 zero-padded decimal digits.
; rax=0 ok, -1 too long
topic_append_u20:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	xor	rbx, rbx
.tau_len:
	cmp	byte [r12 + rbx], 0
	je	.tau_have_len
	inc	rbx
	cmp	rbx, TOPIC_STORE_PATH_MAX - 21
	jae	.tau_err
	jmp	.tau_len
.tau_have_len:
	cmp	rbx, TOPIC_STORE_PATH_MAX - 21
	jae	.tau_err
	mov	rcx, 20
	lea	r14, [r12 + rbx + 19]
.tau_digits:
	mov	rax, r13
	xor	rdx, rdx
	mov	rsi, 10
	div	rsi
	mov	r13, rax
	add	dl, '0'
	mov	[r14], dl
	dec	r14
	dec	rcx
	jnz	.tau_digits
	mov	byte [r12 + rbx + 20], 0
	xor	rax, rax
	jmp	.tau_out
.tau_err:
	mov	rax, -1
.tau_out:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

topic_topics_suffix db '/topics', 0
topic_offsets_suffix db '/offsets', 0
topic_slash db '/', 0
topic_dot db '.', 0
topic_log_ext db '.log', 0
topic_idx_ext db '.idx', 0
topic_global_idx_suffix db '/global.idx', 0
topic_offset_suffix db '.offset', 0

macro topic_store_bss
{
	topic_scratch dq ?
}
