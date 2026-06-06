; Cooperative green-thread scheduler for macOS x86_64.
;
; Include from an executable segment. The including app must emit `coro_bss`
; from a writeable segment.
;
; v1 is intentionally stackful, fixed-size, and single-threaded.

match =CORO_MAX_TASKS, CORO_MAX_TASKS {
	CORO_MAX_TASKS equ 64
}

match =CORO_STACK_SIZE, CORO_STACK_SIZE {
	CORO_STACK_SIZE equ 16384
}

CORO_STATE_EMPTY equ 0
CORO_STATE_RUNNABLE equ 1
CORO_STATE_WAITING equ 2
CORO_STATE_DEAD equ 3

CORO_TASK_STATE_OFF equ 0
CORO_TASK_RSP_OFF equ 8
CORO_TASK_ENTRY_OFF equ 16
CORO_TASK_ARG_OFF equ 24
CORO_TASK_WAIT_FD_OFF equ 32
CORO_TASK_WAIT_FILTER_OFF equ 40
CORO_TASK_SIZE equ 48

macro coro_bss
{
	coro_tasks rb CORO_TASK_SIZE * CORO_MAX_TASKS
	coro_stacks rb CORO_STACK_SIZE * CORO_MAX_TASKS
	coro_task_count dq ?
	coro_current dq ?
	coro_next_scan dq ?
	coro_sched_rsp dq ?
	coro_kq dq ?
	coro_change rb KEVENT_SIZE
	coro_event rb KEVENT_SIZE
}

; rax = kqueue fd, or -1
coro_init:
	push	rbx
	mov	qword [coro_task_count], 0
	mov	qword [coro_current], -1
	mov	qword [coro_next_scan], 0
	xor	rbx, rbx
.ci_clear:
	cmp	rbx, CORO_MAX_TASKS
	jae	.ci_kq
	mov	rax, rbx
	imul	rax, CORO_TASK_SIZE
	mov	qword [coro_tasks + rax + CORO_TASK_STATE_OFF], CORO_STATE_EMPTY
	inc	rbx
	jmp	.ci_clear
.ci_kq:
	mov	rax, SYS_kqueue
	syscall
	jc	.ci_err
	mov	[coro_kq], rax
	pop	rbx
	ret
.ci_err:
	mov	rax, -1
	pop	rbx
	ret

; rdi = entry(task_arg), rsi = task_arg
; rax = task index, or -1
coro_spawn:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	xor	rbx, rbx
.cs_find:
	cmp	rbx, CORO_MAX_TASKS
	jae	.cs_fail
	mov	rax, rbx
	imul	rax, CORO_TASK_SIZE
	cmp	qword [coro_tasks + rax + CORO_TASK_STATE_OFF], CORO_STATE_EMPTY
	je	.cs_found
	cmp	qword [coro_tasks + rax + CORO_TASK_STATE_OFF], CORO_STATE_DEAD
	je	.cs_found
	inc	rbx
	jmp	.cs_find
.cs_found:
	mov	rax, rbx
	imul	rax, CORO_TASK_SIZE
	lea	rdx, [coro_tasks + rax]
	mov	qword [rdx + CORO_TASK_STATE_OFF], CORO_STATE_RUNNABLE
	mov	[rdx + CORO_TASK_ENTRY_OFF], r12
	mov	[rdx + CORO_TASK_ARG_OFF], r13
	mov	qword [rdx + CORO_TASK_WAIT_FD_OFF], -1
	mov	qword [rdx + CORO_TASK_WAIT_FILTER_OFF], 0

	mov	rax, rbx
	imul	rax, CORO_STACK_SIZE
	lea	rcx, [coro_stacks + rax + CORO_STACK_SIZE]
	and	rcx, -16
	sub	rcx, 56
	mov	qword [rcx + 0], 0
	mov	qword [rcx + 8], 0
	mov	qword [rcx + 16], 0
	mov	qword [rcx + 24], 0
	mov	qword [rcx + 32], 0
	mov	qword [rcx + 40], 0
	lea	rax, [coro_task_bootstrap]
	mov	[rcx + 48], rax
	mov	[rdx + CORO_TASK_RSP_OFF], rcx

	inc	qword [coro_task_count]
	mov	rax, rbx
	jmp	.cs_out
.cs_fail:
	mov	rax, -1
.cs_out:
	pop	r13
	pop	r12
	pop	rbx
	ret

coro_run:
.cr_loop:
	call	coro_pick_runnable
	cmp	rax, 0
	jl	.cr_no_runnable
	mov	rdi, rax
	call	coro_resume_task
	jmp	.cr_loop
.cr_no_runnable:
	call	coro_has_alive
	test	rax, rax
	jz	.cr_done
	call	coro_wait_events
	jmp	.cr_loop
.cr_done:
	ret

; rax = runnable task index, or -1
coro_pick_runnable:
	push	rbx
	push	r12
	mov	r12, [coro_next_scan]
	xor	rbx, rbx
.cpr_loop:
	cmp	rbx, CORO_MAX_TASKS
	jae	.cpr_none
	mov	rax, r12
	cmp	rax, CORO_MAX_TASKS
	jb	.cpr_idx_ok
	xor	r12, r12
	xor	rax, rax
.cpr_idx_ok:
	mov	rdx, rax
	imul	rdx, CORO_TASK_SIZE
	cmp	qword [coro_tasks + rdx + CORO_TASK_STATE_OFF], CORO_STATE_RUNNABLE
	je	.cpr_found
	inc	r12
	inc	rbx
	jmp	.cpr_loop
.cpr_found:
	inc	rax
	cmp	rax, CORO_MAX_TASKS
	jb	.cpr_store
	xor	rax, rax
.cpr_store:
	mov	[coro_next_scan], rax
	mov	rax, r12
	jmp	.cpr_out
.cpr_none:
	mov	rax, -1
.cpr_out:
	pop	r12
	pop	rbx
	ret

; rax = 1 if any task is runnable/waiting, else 0
coro_has_alive:
	push	rbx
	xor	rbx, rbx
.cha_loop:
	cmp	rbx, CORO_MAX_TASKS
	jae	.cha_no
	mov	rax, rbx
	imul	rax, CORO_TASK_SIZE
	cmp	qword [coro_tasks + rax + CORO_TASK_STATE_OFF], CORO_STATE_RUNNABLE
	je	.cha_yes
	cmp	qword [coro_tasks + rax + CORO_TASK_STATE_OFF], CORO_STATE_WAITING
	je	.cha_yes
	inc	rbx
	jmp	.cha_loop
.cha_yes:
	mov	rax, 1
	jmp	.cha_out
.cha_no:
	xor	rax, rax
.cha_out:
	pop	rbx
	ret

coro_wait_events:
	syscall6 SYS_kevent, [coro_kq], 0, 0, coro_event, 1, 0
	jc	.cwe_done
	cmp	rax, 1
	jb	.cwe_done
	mov	rax, qword [coro_event + KEVENT_UDATA_OFF]
	cmp	rax, CORO_MAX_TASKS
	jae	.cwe_done
	mov	rdx, rax
	imul	rdx, CORO_TASK_SIZE
	cmp	qword [coro_tasks + rdx + CORO_TASK_STATE_OFF], CORO_STATE_WAITING
	jne	.cwe_done
	mov	qword [coro_tasks + rdx + CORO_TASK_STATE_OFF], CORO_STATE_RUNNABLE
.cwe_done:
	ret

; rdi = task index
coro_resume_task:
	mov	[coro_current], rdi
	mov	rax, rdi
	imul	rax, CORO_TASK_SIZE
	lea	rdx, [coro_tasks + rax]
	mov	[coro_sched_rsp], rsp
	mov	rsp, [rdx + CORO_TASK_RSP_OFF]
	jmp	coro_restore_context

coro_restore_context:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	pop	rbp
	ret

coro_task_bootstrap:
	mov	rax, [coro_current]
	imul	rax, CORO_TASK_SIZE
	lea	rbx, [coro_tasks + rax]
	mov	rdi, [rbx + CORO_TASK_ARG_OFF]
	call	qword [rbx + CORO_TASK_ENTRY_OFF]
	call	coro_exit

coro_yield:
	push	rbp
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	rax, [coro_current]
	imul	rax, CORO_TASK_SIZE
	lea	rdx, [coro_tasks + rax]
	mov	[rdx + CORO_TASK_RSP_OFF], rsp
	cmp	qword [rdx + CORO_TASK_STATE_OFF], CORO_STATE_WAITING
	je	.cy_switch
	mov	qword [rdx + CORO_TASK_STATE_OFF], CORO_STATE_RUNNABLE
.cy_switch:
	mov	rsp, [coro_sched_rsp]
	ret

coro_exit:
	mov	rax, [coro_current]
	imul	rax, CORO_TASK_SIZE
	lea	rdx, [coro_tasks + rax]
	mov	qword [rdx + CORO_TASK_STATE_OFF], CORO_STATE_DEAD
	dec	qword [coro_task_count]
	mov	rsp, [coro_sched_rsp]
	ret

; rdi = fd, rsi = filter; rax = 0 after resume, -1 on registration error
coro_wait_fd:
	push	rbx
	push	r12
	mov	r12, rdi
	mov	rbx, rsi
	mov	rax, [coro_current]
	imul	rax, CORO_TASK_SIZE
	lea	rdx, [coro_tasks + rax]
	mov	qword [rdx + CORO_TASK_STATE_OFF], CORO_STATE_WAITING
	mov	[rdx + CORO_TASK_WAIT_FD_OFF], r12
	mov	[rdx + CORO_TASK_WAIT_FILTER_OFF], rbx

	mov	qword [coro_change + KEVENT_IDENT_OFF], r12
	mov	word [coro_change + KEVENT_FILTER_OFF], bx
	mov	word [coro_change + KEVENT_FLAGS_OFF], EV_ADD or EV_ONESHOT
	mov	dword [coro_change + KEVENT_FFLAGS_OFF], 0
	mov	qword [coro_change + KEVENT_DATA_OFF], 0
	mov	rax, [coro_current]
	mov	qword [coro_change + KEVENT_UDATA_OFF], rax
	syscall6 SYS_kevent, [coro_kq], coro_change, 1, 0, 0, 0
	jc	.cwf_err
	call	coro_yield
	xor	rax, rax
	jmp	.cwf_out
.cwf_err:
	mov	qword [rdx + CORO_TASK_STATE_OFF], CORO_STATE_RUNNABLE
	mov	rax, -1
.cwf_out:
	pop	r12
	pop	rbx
	ret

; rdi = fd
coro_wait_read:
	mov	rsi, EVFILT_READ
	jmp	coro_wait_fd

; rdi = fd
coro_wait_write:
	mov	rsi, EVFILT_WRITE
	jmp	coro_wait_fd
