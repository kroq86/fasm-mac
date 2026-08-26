format ELF64
section '.text' executable
public tensor_transformer_steps_execute

; Experimental direct-call action ABI.
; int tensor_transformer_steps_execute(const ExecStep *steps, uint64_t count)
; ExecStep: run(void *context) at +0, context at +8, kind:u32 at +16,
; flags:u32 at +20, reserved:u64 at +24. Stops at the first non-zero result.
tensor_transformer_steps_execute:
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	xor	r14d, r14d
.next:
	cmp	r14, r13
	jae	.ok
	mov	rax, r14
	shl	rax, 5
	add	rax, r12
	mov	rdi, [rax + 8]
	call	qword [rax]
	test	eax, eax
	jnz	.out
	inc	r14
	jmp	.next
.ok:
	xor	eax, eax
.out:
	pop	r14
	pop	r13
	pop	r12
	ret
