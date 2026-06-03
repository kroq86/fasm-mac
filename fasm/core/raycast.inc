; Tiny fixed-point raycast helpers.
;
; Positions are Q8 fixed point. Maps are row-major byte grids:
;   0 = open, 1 = wall, 2 = exit.

RAY_FP_SHIFT equ 8
RAY_FP_ONE equ 256
RAY_ANGLE_MASK equ 255
RAY_MAX_STEPS equ 160

; rdi = map, rsi = width, rdx = height, rcx = x_q8, r8 = y_q8
; al = map tile, or 1 for out-of-bounds
ray_map_at:
	push	rbx
	mov	rax, rcx
	sar	rax, RAY_FP_SHIFT
	mov	rbx, r8
	sar	rbx, RAY_FP_SHIFT
	test	rax, rax
	js	.rma_wall
	test	rbx, rbx
	js	.rma_wall
	cmp	rax, rsi
	jae	.rma_wall
	cmp	rbx, rdx
	jae	.rma_wall
	imul	rbx, rsi
	add	rbx, rax
	mov	al, [rdi + rbx]
	pop	rbx
	ret
.rma_wall:
	mov	al, 1
	pop	rbx
	ret

; rdi = map, rsi = width, rdx = height, rcx = pos_x_q8, r8 = pos_y_q8,
; r9 = angle index 0..63
; rax = projected column height in a 100px virtual viewport
; rdx = tile hit in low byte, side shade flag in bit 8
ray_cast_height:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, r9
	and	r15, RAY_ANGLE_MASK
	lea	rax, [ray_dir_x]
	movsx	r10, word [rax + r15 * 2]
	lea	rax, [ray_dir_y]
	movsx	r11, word [rax + r15 * 2]
	mov	rbx, rcx
	mov	r9, r8
	mov	r8, 1
.rch_loop:
	add	rbx, r10
	add	r9, r11
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r14
	mov	rcx, rbx
	push	r8
	mov	r8, r9
	call	ray_map_at
	pop	r8
	cmp	al, 0
	jne	.rch_hit
	inc	r8
	cmp	r8, RAY_MAX_STEPS
	jbe	.rch_loop
	mov	al, 1
.rch_hit:
	movzx	r10, al
	xor	r11, r11
	mov	rax, rbx
	and	rax, 255
	mov	rdx, 256
	sub	rdx, rax
	cmp	rax, rdx
	jbe	.rch_xdist
	mov	rax, rdx
.rch_xdist:
	mov	rcx, r9
	and	rcx, 255
	mov	rdx, 256
	sub	rdx, rcx
	cmp	rcx, rdx
	jbe	.rch_ydist
	mov	rcx, rdx
.rch_ydist:
	cmp	rax, rcx
	jae	.rch_side_ready
	mov	r11, 1
.rch_side_ready:
	mov	rax, 900
	xor	rdx, rdx
	mov	rcx, r8
	add	rcx, 1
	div	rcx
	cmp	rax, 4
	jae	.rch_min_ok
	mov	rax, 4
.rch_min_ok:
	cmp	rax, 100
	jbe	.rch_done
	mov	rax, 100
.rch_done:
	mov	rcx, r10
	shl	r11, 8
	or	rcx, r11
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	mov	rdx, rcx
	ret

ray_dir_x:
	dw 32, 32, 32, 32, 32, 32, 32, 32, 31, 31, 31, 31, 31, 30, 30, 30
	dw 30, 29, 29, 29, 28, 28, 27, 27, 27, 26, 26, 25, 25, 24, 24, 23
	dw 23, 22, 21, 21, 20, 20, 19, 18, 18, 17, 16, 16, 15, 14, 14, 13
	dw 12, 12, 11, 10, 9, 9, 8, 7, 6, 5, 5, 4, 3, 2, 2, 1
	dw 0, -1, -2, -2, -3, -4, -5, -5, -6, -7, -8, -9, -9, -10, -11, -12
	dw -12, -13, -14, -14, -15, -16, -16, -17, -18, -18, -19, -20, -20, -21, -21, -22
	dw -23, -23, -24, -24, -25, -25, -26, -26, -27, -27, -27, -28, -28, -29, -29, -29
	dw -30, -30, -30, -30, -31, -31, -31, -31, -31, -32, -32, -32, -32, -32, -32, -32
	dw -32, -32, -32, -32, -32, -32, -32, -32, -31, -31, -31, -31, -31, -30, -30, -30
	dw -30, -29, -29, -29, -28, -28, -27, -27, -27, -26, -26, -25, -25, -24, -24, -23
	dw -23, -22, -21, -21, -20, -20, -19, -18, -18, -17, -16, -16, -15, -14, -14, -13
	dw -12, -12, -11, -10, -9, -9, -8, -7, -6, -5, -5, -4, -3, -2, -2, -1
	dw 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11, 12
	dw 12, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 21, 22
	dw 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 27, 28, 28, 29, 29, 29
	dw 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32

ray_dir_y:
	dw 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11, 12
	dw 12, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 21, 22
	dw 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 27, 28, 28, 29, 29, 29
	dw 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32
	dw 32, 32, 32, 32, 32, 32, 32, 32, 31, 31, 31, 31, 31, 30, 30, 30
	dw 30, 29, 29, 29, 28, 28, 27, 27, 27, 26, 26, 25, 25, 24, 24, 23
	dw 23, 22, 21, 21, 20, 20, 19, 18, 18, 17, 16, 16, 15, 14, 14, 13
	dw 12, 12, 11, 10, 9, 9, 8, 7, 6, 5, 5, 4, 3, 2, 2, 1
	dw 0, -1, -2, -2, -3, -4, -5, -5, -6, -7, -8, -9, -9, -10, -11, -12
	dw -12, -13, -14, -14, -15, -16, -16, -17, -18, -18, -19, -20, -20, -21, -21, -22
	dw -23, -23, -24, -24, -25, -25, -26, -26, -27, -27, -27, -28, -28, -29, -29, -29
	dw -30, -30, -30, -30, -31, -31, -31, -31, -31, -32, -32, -32, -32, -32, -32, -32
	dw -32, -32, -32, -32, -32, -32, -32, -32, -31, -31, -31, -31, -31, -30, -30, -30
	dw -30, -29, -29, -29, -28, -28, -27, -27, -27, -26, -26, -25, -25, -24, -24, -23
	dw -23, -22, -21, -21, -20, -20, -19, -18, -18, -17, -16, -16, -15, -14, -14, -13
	dw -12, -12, -11, -10, -9, -9, -8, -7, -6, -5, -5, -4, -3, -2, -2, -1
