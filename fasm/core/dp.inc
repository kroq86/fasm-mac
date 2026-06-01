; Small reusable int-array dynamic-programming / scan helpers.

; rdi = int64 array, rsi = count
; rax = max profit from one buy/sell transaction
dp_max_profit_one_trade:
	push	rbx
	push	r12
	push	r13
	test	rsi, rsi
	jz	.dmp_zero
	mov	r12, [rdi]
	xor	r13, r13
	mov	rbx, 1
.dmp_loop:
	cmp	rbx, rsi
	jae	.dmp_done
	mov	rax, [rdi + rbx * 8]
	cmp	rax, r12
	jge	.dmp_profit
	mov	r12, rax
	jmp	.dmp_next
.dmp_profit:
	sub	rax, r12
	cmp	rax, r13
	jle	.dmp_next
	mov	r13, rax
.dmp_next:
	inc	rbx
	jmp	.dmp_loop
.dmp_done:
	mov	rax, r13
	pop	r13
	pop	r12
	pop	rbx
	ret
.dmp_zero:
	xor	rax, rax
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = int64 array, rsi = count
; rax = maximum subarray sum
dp_max_subarray:
	push	rbx
	push	r12
	push	r13
	test	rsi, rsi
	jz	.dms_zero
	mov	r12, [rdi]
	mov	r13, r12
	mov	rbx, 1
.dms_loop:
	cmp	rbx, rsi
	jae	.dms_done
	mov	rax, [rdi + rbx * 8]
	mov	rcx, r12
	add	rcx, rax
	cmp	rcx, rax
	jl	.dms_use_current
	mov	r12, rcx
	jmp	.dms_update_best
.dms_use_current:
	mov	r12, rax
.dms_update_best:
	cmp	r12, r13
	jle	.dms_next
	mov	r13, r12
.dms_next:
	inc	rbx
	jmp	.dms_loop
.dms_done:
	mov	rax, r13
	pop	r13
	pop	r12
	pop	rbx
	ret
.dms_zero:
	xor	rax, rax
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = n
; rax = number of ways
dp_climbing_stairs:
	push	rbx
	push	r12
	push	r13
	cmp	rdi, 1
	ja	.dcs_many
	mov	rax, 1
	pop	r13
	pop	r12
	pop	rbx
	ret
.dcs_many:
	mov	r12, 1
	mov	r13, 1
	mov	rbx, 2
.dcs_loop:
	cmp	rbx, rdi
	ja	.dcs_done
	mov	rax, r12
	add	rax, r13
	mov	r12, r13
	mov	r13, rax
	inc	rbx
	jmp	.dcs_loop
.dcs_done:
	mov	rax, r13
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = int64 array, rsi = count
; rax = maximum non-adjacent sum
dp_house_robber:
	push	rbx
	push	r12
	push	r13
	xor	r12, r12
	xor	r13, r13
	xor	rbx, rbx
.dhr_loop:
	cmp	rbx, rsi
	jae	.dhr_done
	mov	rax, r12
	add	rax, [rdi + rbx * 8]
	cmp	rax, r13
	jge	.dhr_take
	mov	rax, r13
.dhr_take:
	mov	r12, r13
	mov	r13, rax
	inc	rbx
	jmp	.dhr_loop
.dhr_done:
	mov	rax, r13
	pop	r13
	pop	r12
	pop	rbx
	ret
