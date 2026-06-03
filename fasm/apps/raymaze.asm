; raymaze: original tiny raycaster game using raylib.
;
; Build:
;   fasm --emit=macho-obj fasm/apps/raymaze.asm /tmp/raymaze.o
;   clang -arch x86_64 /tmp/raymaze.o -lraylib ... -o raymaze

format ELF64
include "fasm/core/platform.inc"

SCREEN_W equ 800
SCREEN_H equ 450
VIEW_W equ 400
VIEW_H equ 100
COL_W equ 2
MAP_W equ 16
MAP_H equ 16
PLAYER_RADIUS equ 56

KEY_ESCAPE equ 256
KEY_RIGHT equ 262
KEY_LEFT equ 263
KEY_A equ 65
KEY_D equ 68
KEY_E equ 69
KEY_Q equ 81
KEY_S equ 83
KEY_W equ 87

COLOR_SKY equ 0FF2F2418h
COLOR_FLOOR equ 0FF202528h
COLOR_WALL equ 0FFB88758h
COLOR_WALL_SIDE equ 0FF946A42h
COLOR_WALL_FAR equ 0FF7C5A38h
COLOR_EXIT equ 0FF70C060h
COLOR_TEXT equ 0FFF2EFE6h

section ".text" executable

public main
extrn InitWindow
extrn SetTargetFPS
extrn WindowShouldClose
extrn IsKeyDown
extrn BeginDrawing
extrn ClearBackground
extrn DrawRectangle
extrn DrawText
extrn EndDrawing
extrn CloseWindow

include "fasm/core/ccall64.inc"
include "fasm/core/raycast.inc"

main:
	push	rbp
	mov	rbp, rsp
	mov	[argv_base], rsi
	cmp	rdi, 1
	je	.main_game
	cmp	rdi, 2
	je	.main_one_arg
	cmp	rdi, 3
	je	.main_snapshot_arg
	jmp	print_usage_error
.main_one_arg:
	mov	rax, [rsi + 8]
	mov	rdi, rax
	lea	rsi, [arg_help]
	call	str_eq
	cmp	rax, 1
	je	print_usage_ok
	jmp	print_usage_error
.main_snapshot_arg:
	mov	rsi, [argv_base]
	mov	rax, [rsi + 8]
	mov	rdi, rax
	lea	rsi, [arg_snapshot]
	call	str_eq
	cmp	rax, 1
	jne	print_usage_error
	mov	rsi, [argv_base]
	mov	rdi, [rsi + 16]
	call	write_snapshot
	pop	rbp
	ret
.main_game:
	call	init_state
	mov	rdi, SCREEN_W
	mov	rsi, SCREEN_H
	lea	rdx, [window_title]
	call	InitWindow
	ccall1	SetTargetFPS, 60
.main_loop:
	call	WindowShouldClose
	test	rax, rax
	jnz	.main_done
	cmp	qword [game_won], 0
	jne	.main_render
	call	handle_input
.main_render:
	call	render_frame
	jmp	.main_loop
.main_done:
	call	CloseWindow
	xor	rax, rax
	pop	rbp
	ret

print_usage_ok:
	mov	rax, SYS_write
	mov	rdi, STDOUT
	lea	rsi, [usage_msg]
	mov	rdx, usage_msg_len
	syscall
	xor	rax, rax
	pop	rbp
	ret

print_usage_error:
	mov	rax, SYS_write
	mov	rdi, STDERR
	lea	rsi, [usage_msg]
	mov	rdx, usage_msg_len
	syscall
	mov	rax, 2
	pop	rbp
	ret

init_state:
	mov	qword [player_x], 384
	mov	qword [player_y], 384
	mov	qword [player_angle], 0
	mov	qword [game_won], 0
	ret

; rdi = a, rsi = b; rax = 1 if equal, 0 otherwise
str_eq:
.se_loop:
	mov	al, [rdi]
	mov	cl, [rsi]
	cmp	al, cl
	jne	.se_no
	test	al, al
	jz	.se_yes
	inc	rdi
	inc	rsi
	jmp	.se_loop
.se_yes:
	mov	rax, 1
	ret
.se_no:
	xor	rax, rax
	ret

handle_input:
	push	rbp
	mov	rbp, rsp
	ccall1	IsKeyDown, KEY_LEFT
	test	rax, rax
	jz	.hi_q
	sub	qword [player_angle], 3
.hi_q:
	ccall1	IsKeyDown, KEY_Q
	test	rax, rax
	jz	.hi_right
	sub	qword [player_angle], 3
.hi_right:
	ccall1	IsKeyDown, KEY_RIGHT
	test	rax, rax
	jz	.hi_e
	add	qword [player_angle], 3
.hi_e:
	ccall1	IsKeyDown, KEY_E
	test	rax, rax
	jz	.hi_w
	add	qword [player_angle], 3
.hi_w:
	ccall1	IsKeyDown, KEY_W
	test	rax, rax
	jz	.hi_s
	mov	rdi, [player_angle]
	mov	rsi, 1
	call	move_by_angle
.hi_s:
	ccall1	IsKeyDown, KEY_S
	test	rax, rax
	jz	.hi_a
	mov	rdi, [player_angle]
	mov	rsi, -1
	call	move_by_angle
.hi_a:
	ccall1	IsKeyDown, KEY_A
	test	rax, rax
	jz	.hi_d
	mov	rdi, [player_angle]
	sub	rdi, 64
	mov	rsi, 1
	call	move_by_angle
.hi_d:
	ccall1	IsKeyDown, KEY_D
	test	rax, rax
	jz	.hi_done
	mov	rdi, [player_angle]
	add	rdi, 64
	mov	rsi, 1
	call	move_by_angle
.hi_done:
	and	qword [player_angle], RAY_ANGLE_MASK
	pop	rbp
	ret

; rdi = angle, rsi = direction sign (-1 or 1)
move_by_angle:
	push	rbx
	push	r12
	push	r13
	push	r14
	and	rdi, RAY_ANGLE_MASK
	lea	rax, [ray_dir_x]
	movsx	r12, word [rax + rdi * 2]
	lea	rax, [ray_dir_y]
	movsx	r13, word [rax + rdi * 2]
	sar	r12, 1
	sar	r13, 1
	cmp	rsi, 0
	jge	.mba_signed
	neg	r12
	neg	r13
.mba_signed:
	mov	rbx, [player_x]
	add	rbx, r12
	mov	r14, [player_y]
	add	r14, r13
	mov	rdi, rbx
	mov	rsi, r14
	call	can_stand_at
	cmp	al, 1
	je	.mba_done
	mov	[player_x], rbx
	mov	[player_y], r14
	cmp	al, 2
	jne	.mba_done
	mov	qword [game_won], 1
.mba_done:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = x_q8, rsi = y_q8
; al = center tile if all radius checks are passable, 1 if blocked
can_stand_at:
	push	rbx
	push	r12
	push	r13
	mov	rbx, rdi
	mov	r13, rsi
	lea	rdi, [maze_map]
	mov	rsi, MAP_W
	mov	rdx, MAP_H
	mov	rcx, rbx
	mov	r8, r13
	call	ray_map_at
	cmp	al, 1
	je	.csa_blocked
	mov	r12b, al
	lea	rdi, [maze_map]
	mov	rsi, MAP_W
	mov	rdx, MAP_H
	mov	rcx, rbx
	add	rcx, PLAYER_RADIUS
	mov	r8, r13
	call	ray_map_at
	cmp	al, 1
	je	.csa_blocked
	lea	rdi, [maze_map]
	mov	rsi, MAP_W
	mov	rdx, MAP_H
	mov	rcx, rbx
	sub	rcx, PLAYER_RADIUS
	mov	r8, r13
	call	ray_map_at
	cmp	al, 1
	je	.csa_blocked
	lea	rdi, [maze_map]
	mov	rsi, MAP_W
	mov	rdx, MAP_H
	mov	rcx, rbx
	mov	r8, r13
	add	r8, PLAYER_RADIUS
	call	ray_map_at
	cmp	al, 1
	je	.csa_blocked
	lea	rdi, [maze_map]
	mov	rsi, MAP_W
	mov	rdx, MAP_H
	mov	rcx, rbx
	mov	r8, r13
	sub	r8, PLAYER_RADIUS
	call	ray_map_at
	cmp	al, 1
	je	.csa_blocked
	mov	al, r12b
	jmp	.csa_done
.csa_blocked:
	mov	al, 1
.csa_done:
	pop	r13
	pop	r12
	pop	rbx
	ret

render_frame:
	push	rbp
	mov	rbp, rsp
	push	r12
	push	r13
	push	r14
	push	r15
	call	BeginDrawing
	ccall1	ClearBackground, COLOR_SKY
	ccall5	DrawRectangle, 0, 225, SCREEN_W, 225, COLOR_FLOOR
	xor	r12, r12
.rf_col:
	cmp	r12, VIEW_W
	jae	.rf_text
	mov	rax, r12
	xor	rdx, rdx
	mov	rcx, 6
	div	rcx
	mov	r9, [player_angle]
	add	r9, rax
	sub	r9, 33
	lea	rdi, [maze_map]
	mov	rsi, MAP_W
	mov	rdx, MAP_H
	mov	rcx, [player_x]
	mov	r8, [player_y]
	call	ray_cast_height
	mov	r10, rdx
	mov	r14, rax
	imul	r14, 4
	mov	r13, SCREEN_H
	sub	r13, r14
	sar	r13, 1
	mov	r15, COLOR_WALL
	cmp	r10b, 2
	je	.rf_exit_color
	cmp	rax, 34
	jb	.rf_far_color
	test	r10, 100h
	jz	.rf_draw
	mov	r15, COLOR_WALL_SIDE
	jmp	.rf_draw
.rf_far_color:
	mov	r15, COLOR_WALL_FAR
	jmp	.rf_draw
.rf_exit_color:
	mov	r15, COLOR_EXIT
.rf_draw:
	mov	rdi, r12
	imul	rdi, COL_W
	mov	rsi, r13
	mov	rdx, COL_W
	mov	rcx, r14
	mov	r8, r15
	call	DrawRectangle
	inc	r12
	jmp	.rf_col
.rf_text:
	lea	rdi, [hud_text]
	mov	rsi, 18
	mov	rdx, 16
	mov	rcx, 18
	mov	r8, COLOR_TEXT
	call	DrawText
	cmp	qword [game_won], 0
	je	.rf_done
	lea	rdi, [win_text]
	mov	rsi, 292
	mov	rdx, 208
	mov	rcx, 28
	mov	r8, COLOR_TEXT
	call	DrawText
.rf_done:
	call	EndDrawing
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbp
	ret

; rdi = output path
write_snapshot:
	push	rbp
	mov	rbp, rsp
	push	r12
	mov	r12, rdi
	call	init_state
	call	render_snapshot_pixels
	mov	rax, SYS_open
	mov	rdi, r12
	mov	rsi, O_WRONLY + O_CREAT + O_TRUNC
	mov	rdx, 420
	syscall
	jc	.ws_error
	mov	r12, rax
	mov	rdi, r12
	lea	rsi, [ppm_header]
	mov	rdx, ppm_header_len
	mov	rax, SYS_write
	syscall
	mov	rdi, r12
	lea	rsi, [ppm_pixels]
	mov	rdx, VIEW_W * VIEW_H * 3
	mov	rax, SYS_write
	syscall
	mov	rdi, r12
	mov	rax, SYS_close
	syscall
	xor	rax, rax
	jmp	.ws_done
.ws_error:
	mov	rax, SYS_write
	mov	rdi, STDERR
	lea	rsi, [snapshot_err]
	mov	rdx, snapshot_err_len
	syscall
	mov	rax, 2
.ws_done:
	pop	r12
	pop	rbp
	ret

render_snapshot_pixels:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	xor	r12, r12
.rsp_cols:
	cmp	r12, VIEW_W
	jae	.rsp_pixels
	mov	rax, r12
	xor	rdx, rdx
	mov	rcx, 6
	div	rcx
	mov	r9, [player_angle]
	add	r9, rax
	sub	r9, 33
	lea	rdi, [maze_map]
	mov	rsi, MAP_W
	mov	rdx, MAP_H
	mov	rcx, [player_x]
	mov	r8, [player_y]
	call	ray_cast_height
	mov	r10, rdx
	lea	rbx, [column_heights]
	mov	[rbx + r12], al
	lea	rbx, [column_tiles]
	mov	[rbx + r12], r10b
	shr	r10, 8
	lea	rbx, [column_sides]
	mov	[rbx + r12], r10b
	inc	r12
	jmp	.rsp_cols
.rsp_pixels:
	lea	r15, [ppm_pixels]
	xor	r13, r13
.rsp_y:
	cmp	r13, VIEW_H
	jae	.rsp_done
	xor	r12, r12
.rsp_x:
	cmp	r12, VIEW_W
	jae	.rsp_next_y
	lea	rbx, [column_heights]
	movzx	r14, byte [rbx + r12]
	mov	rax, VIEW_H
	sub	rax, r14
	sar	rax, 1
	mov	rbx, rax
	cmp	r13, rbx
	jb	.rsp_sky
	add	rbx, r14
	cmp	r13, rbx
	jae	.rsp_floor
	lea	rbx, [column_tiles]
	cmp	byte [rbx + r12], 2
	je	.rsp_exit
	cmp	r14, 34
	jb	.rsp_wall_far
	lea	rbx, [column_sides]
	cmp	byte [rbx + r12], 0
	jne	.rsp_wall_side
	mov	byte [r15], 88
	mov	byte [r15 + 1], 135
	mov	byte [r15 + 2], 184
	jmp	.rsp_advance
.rsp_wall_side:
	mov	byte [r15], 66
	mov	byte [r15 + 1], 106
	mov	byte [r15 + 2], 148
	jmp	.rsp_advance
.rsp_wall_far:
	mov	byte [r15], 56
	mov	byte [r15 + 1], 90
	mov	byte [r15 + 2], 124
	jmp	.rsp_advance
.rsp_exit:
	mov	byte [r15], 96
	mov	byte [r15 + 1], 192
	mov	byte [r15 + 2], 112
	jmp	.rsp_advance
.rsp_sky:
	mov	byte [r15], 24
	mov	byte [r15 + 1], 36
	mov	byte [r15 + 2], 47
	jmp	.rsp_advance
.rsp_floor:
	mov	byte [r15], 40
	mov	byte [r15 + 1], 37
	mov	byte [r15 + 2], 32
.rsp_advance:
	add	r15, 3
	inc	r12
	jmp	.rsp_x
.rsp_next_y:
	inc	r13
	jmp	.rsp_y
.rsp_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

section ".data" writeable

window_title db "raymaze", 0
arg_help db "--help", 0
arg_snapshot db "--snapshot", 0
usage_msg db "usage: raymaze [--snapshot file.ppm]", 10
usage_msg_len = $ - usage_msg
snapshot_err db "raymaze: cannot write snapshot", 10
snapshot_err_len = $ - snapshot_err
hud_text db "raymaze  W/S move  A/D strafe  Q/E turn  Esc quit", 0
win_text db "EXIT FOUND", 0
ppm_header db "P6", 10, "400 100", 10, "255", 10
ppm_header_len = $ - ppm_header

maze_map:
	db 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
	db 1,0,0,0,0,0,0,0,0,0,0,0,0,2,1,1
	db 1,0,1,1,1,1,0,1,1,1,1,1,1,0,1,1
	db 1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1
	db 1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1
	db 1,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1
	db 1,0,1,1,1,1,0,1,1,1,0,1,1,1,1,1
	db 1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1
	db 1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1
	db 1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1
	db 1,0,1,1,1,1,1,1,0,1,1,1,1,0,1,1
	db 1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1
	db 1,0,1,1,1,1,0,1,1,1,1,1,1,0,1,1
	db 1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1
	db 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
	db 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1

player_x dq ?
player_y dq ?
player_angle dq ?
game_won dq ?
argv_base dq ?

section ".bss" writeable

column_heights rb VIEW_W
column_tiles rb VIEW_W
column_sides rb VIEW_W
ppm_pixels rb VIEW_W * VIEW_H * 3
