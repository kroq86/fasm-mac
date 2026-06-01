; Byte-grid graph helpers for LeetCode flood-fill style problems.

macro grid_bss
{
	grid_land_byte rb 1
	grid_water_byte rb 1
}

; rdi = row, rsi = col, rdx = column count
; rax = row * column_count + col
grid_index:
	mov	rax, rdi
	imul	rax, rdx
	add	rax, rsi
	ret

; rdi = grid bytes, rsi = rows, rdx = cols, rcx = land byte, r8b = water byte
; rax = number of 4-connected components. Mutates matching land cells to water.
grid_count_components4:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	[grid_land_byte], cl
	mov	[grid_water_byte], r8b
	xor	r15, r15
	xor	rbx, rbx
.gcc_row:
	cmp	rbx, r13
	jae	.gcc_done
	xor	r10, r10
.gcc_col:
	cmp	r10, r14
	jae	.gcc_next_row
	mov	rax, rbx
	imul	rax, r14
	add	rax, r10
	mov	cl, [grid_land_byte]
	cmp	byte [r12 + rax], cl
	jne	.gcc_next_col
	inc	r15
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r14
	mov	rcx, rbx
	mov	r8, r10
	call	grid_flood_fill4
.gcc_next_col:
	inc	r10
	jmp	.gcc_col
.gcc_next_row:
	inc	rbx
	jmp	.gcc_row
.gcc_done:
	mov	rax, r15
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = grid, rsi = rows, rdx = cols, rcx = row, r8 = col
; Uses grid_land_byte/grid_water_byte set by grid_count_components4.
grid_flood_fill4:
	test	rcx, rcx
	js	.gff_ret
	test	r8, r8
	js	.gff_ret
	cmp	rcx, rsi
	jae	.gff_ret
	cmp	r8, rdx
	jae	.gff_ret
	mov	rax, rcx
	imul	rax, rdx
	add	rax, r8
	mov	r9b, [grid_land_byte]
	cmp	byte [rdi + rax], r9b
	jne	.gff_ret
	mov	r9b, [grid_water_byte]
	mov	byte [rdi + rax], r9b
	push	rdi
	push	rsi
	push	rdx
	push	rcx
	push	r8
	dec	rcx
	call	grid_flood_fill4
	pop	r8
	pop	rcx
	pop	rdx
	pop	rsi
	pop	rdi
	push	rdi
	push	rsi
	push	rdx
	push	rcx
	push	r8
	inc	rcx
	call	grid_flood_fill4
	pop	r8
	pop	rcx
	pop	rdx
	pop	rsi
	pop	rdi
	push	rdi
	push	rsi
	push	rdx
	push	rcx
	push	r8
	dec	r8
	call	grid_flood_fill4
	pop	r8
	pop	rcx
	pop	rdx
	pop	rsi
	pop	rdi
	inc	r8
	call	grid_flood_fill4
.gff_ret:
	ret
