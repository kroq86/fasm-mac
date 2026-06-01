; Small helpers shared by runnable LeetCode examples.

include "fasm/core/print_io.inc"

print_bool_nl:
	test	rax, rax
	jz	.pbn_zero
	mov	rax, 1
	jmp	print_int_nl
.pbn_zero:
	xor	rax, rax
	jmp	print_int_nl

; rdi = int64 array, rsi = count
print_i64_array:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	xor	rbx, rbx
.pia_loop:
	cmp	rbx, r13
	jae	.pia_done
	mov	rax, [r12 + rbx * 8]
	call	print_int_sp
	inc	rbx
	jmp	.pia_loop
.pia_done:
	mov	al, 10
	call	print_char
	pop	r13
	pop	r12
	pop	rbx
	ret
