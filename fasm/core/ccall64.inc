; Minimal x86_64 System V C-call helpers for Mach-O objects linked by clang.
;
; Contract:
; - Caller must have rsp 16-byte aligned before invoking these macros.
; - Arguments are integer/pointer-sized values in the standard registers.
; - C callees may clobber rax, rcx, rdx, rsi, rdi, r8-r11.

macro ccall0 proc
{
	call	proc
}

macro ccall1 proc, a1
{
	mov	rdi, a1
	call	proc
}

macro ccall2 proc, a1, a2
{
	mov	rdi, a1
	mov	rsi, a2
	call	proc
}

macro ccall3 proc, a1, a2, a3
{
	mov	rdi, a1
	mov	rsi, a2
	mov	rdx, a3
	call	proc
}

macro ccall4 proc, a1, a2, a3, a4
{
	mov	rdi, a1
	mov	rsi, a2
	mov	rdx, a3
	mov	rcx, a4
	call	proc
}

macro ccall5 proc, a1, a2, a3, a4, a5
{
	mov	rdi, a1
	mov	rsi, a2
	mov	rdx, a3
	mov	rcx, a4
	mov	r8, a5
	call	proc
}

macro ccall6 proc, a1, a2, a3, a4, a5, a6
{
	mov	rdi, a1
	mov	rsi, a2
	mov	rdx, a3
	mov	rcx, a4
	mov	r8, a5
	mov	r9, a6
	call	proc
}
