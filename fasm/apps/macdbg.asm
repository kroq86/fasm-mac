; macdbg: AI-native LLDB snapshot debugger for macOS.
;
; Build:
;   fasm --emit=macho-obj fasm/apps/macdbg.asm /tmp/macdbg.o
;   clang -arch x86_64 /tmp/macdbg.o $(pkg-config --cflags --libs raylib) -o macdbg

format ELF64
include "fasm/core/platform.inc"

REPORT_CAP equ 262144
UI_W equ 1100
UI_H equ 720

KEY_J equ 74
KEY_R equ 82

COLOR_BG equ 0FF171411h
COLOR_PANEL equ 0FF26211Dh
COLOR_PANEL_DARK equ 0FF201B18h
COLOR_TEXT equ 0FFE8E0D6h
COLOR_MUTED equ 0FF9E9488h
COLOR_ACCENT equ 0FF69B7DCh
COLOR_EXITED equ 0FF6FC17Ah
COLOR_CRASHED equ 0FFDE665Ch

section ".text" executable

public main
extrn getenv
extrn InitWindow
extrn SetTargetFPS
extrn WindowShouldClose
extrn IsKeyPressed
extrn BeginDrawing
extrn ClearBackground
extrn DrawRectangle
extrn DrawText
extrn EndDrawing
extrn CloseWindow

include "fasm/core/ccall64.inc"
include "fasm/core/process.inc"
include "fasm/core/file.inc"
include "fasm/core/json_emit.inc"
include "fasm/core/lldb_batch.inc"

macro ui_text text, x, y, size, color
{
	lea	rdi, [text]
	mov	rsi, x
	mov	rdx, y
	mov	rcx, size
	mov	r8, color
	call	DrawText
}

main:
	push	rbp
	mov	rbp, rsp
	push	rbx
	sub	rsp, 8
	mov	[argc], rdi
	mov	[argv], rsi
	cmp	rdi, 2
	je	.maybe_help
	mov	rdi, [rsi + 8]
	lea	rsi, [opt_ui]
	call	str_eq
	cmp	rax, 1
	je	.parse_ui_mode
	mov	rsi, [argv]
	mov	rdi, [argc]
	cmp	rdi, 4
	jb	usage_error
	mov	rdi, [rsi + 8]
	lea	rsi, [opt_snapshot]
	call	str_eq
	cmp	rax, 1
	jne	usage_error
	mov	rax, [argv]
	mov	rdi, [rax + 16]
	lea	rsi, [opt_args]
	call	str_eq
	cmp	rax, 1
	je	.parse_args_mode
	mov	rax, [argv]
	cmp	qword [argc], 4
	jne	usage_error
	mov	rdi, [rax + 16]
	mov	[target_path], rdi
	mov	rsi, [argv]
	mov	rdi, [rsi + 24]
	mov	[report_path], rdi
	mov	qword [args_start], 0
	mov	qword [args_end], 0
	call	run_snapshot
	add	rsp, 8
	pop	rbx
	pop	rbp
	ret
.parse_ui_mode:
	cmp	qword [argc], 3
	je	.parse_ui_plain
	mov	rax, [argv]
	mov	rdi, [rax + 16]
	lea	rsi, [opt_args]
	call	str_eq
	cmp	rax, 1
	je	.parse_ui_args
	jmp	usage_error
.parse_ui_plain:
	mov	rax, [argv]
	mov	rdi, [rax + 16]
	mov	[target_path], rdi
	mov	qword [args_start], 0
	mov	qword [args_end], 0
	lea	rax, [ui_report_path]
	mov	[report_path], rax
	call	run_ui
	add	rsp, 8
	pop	rbx
	pop	rbp
	ret
.parse_ui_args:
	cmp	qword [argc], 5
	jb	usage_error
	mov	rax, [argv]
	mov	rdi, [rax + 24]
	mov	[target_path], rdi
	mov	qword [args_start], 4
	mov	rax, [argc]
	mov	[args_end], rax
	lea	rax, [ui_report_path]
	mov	[report_path], rax
	call	run_ui
	add	rsp, 8
	pop	rbx
	pop	rbp
	ret
.parse_args_mode:
	cmp	qword [argc], 6
	jb	usage_error
	mov	rax, [argv]
	mov	rdi, [rax + 24]
	mov	[target_path], rdi
	mov	qword [args_start], 4
	mov	qword [args_end], 0
	mov	rbx, 4
.find_dashdash:
	cmp	rbx, [argc]
	jae	usage_error
	mov	rax, [argv]
	mov	rdi, [rax + rbx * 8]
	lea	rsi, [opt_dashdash]
	call	str_eq
	cmp	rax, 1
	je	.found_dashdash
	inc	rbx
	jmp	.find_dashdash
.found_dashdash:
	mov	[args_end], rbx
	mov	rax, rbx
	inc	rax
	cmp	rax, [argc]
	jae	usage_error
	mov	rdx, [argv]
	mov	rdi, [rdx + rax * 8]
	mov	[report_path], rdi
	call	run_snapshot
	add	rsp, 8
	pop	rbx
	pop	rbp
	ret
.maybe_help:
	mov	rdi, [rsi + 8]
	lea	rsi, [opt_help]
	call	str_eq
	cmp	rax, 1
	je	usage_ok
	jmp	usage_error

usage_ok:
	mov	rax, SYS_write
	mov	rdi, STDOUT
	lea	rsi, [usage_msg]
	mov	rdx, usage_msg_len
	syscall
	xor	rax, rax
	add	rsp, 8
	pop	rbx
	pop	rbp
	ret

usage_error:
	mov	rax, SYS_write
	mov	rdi, STDERR
	lea	rsi, [usage_msg]
	mov	rdx, usage_msg_len
	syscall
	mov	rax, 2
	add	rsp, 8
	pop	rbx
	pop	rbp
	ret

run_snapshot:
	call	write_lldb_commands
	cmp	rax, 0
	jne	.rs_report_error
	lea	rdi, [lldb_shell_cmd]
	call	process_system
	mov	[system_status], rax
	lea	rdi, [lldb_out_path]
	lea	rsi, [lldb_out_buf]
	mov	rdx, MACDBG_OUT_CAP
	lea	rcx, [lldb_out_len]
	call	file_read_all_fixed
	cmp	rax, FILE_OK
	jne	.rs_report_error
	lea	rdi, [lldb_out_buf]
	mov	rsi, [lldb_out_len]
	call	lldb_detect_status
	mov	[status_ptr], rax
	call	write_report
	ret
.rs_report_error:
	call	write_error_report
	ret

run_ui:
	lea	rax, [status_lldb_error]
	mov	[status_ptr], rax
	mov	qword [lldb_out_len], 0
	mov	qword [json_out_len], 0
	mov	qword [ui_autoclose], 0
	lea	rdi, [env_ui_autoclose]
	call	getenv
	test	rax, rax
	jz	.rui_open
	mov	qword [ui_autoclose], 1
.rui_open:
	mov	rdi, UI_W
	mov	rsi, UI_H
	lea	rdx, [ui_title]
	call	InitWindow
	ccall1	SetTargetFPS, 60
	call	run_snapshot
.ui_loop:
	call	WindowShouldClose
	test	rax, rax
	jnz	.ui_done
	ccall1	IsKeyPressed, KEY_R
	test	rax, rax
	jz	.ui_check_raw
	call	run_snapshot
.ui_check_raw:
	ccall1	IsKeyPressed, KEY_J
	test	rax, rax
	jz	.ui_render
	xor	qword [ui_show_raw], 1
.ui_render:
	call	render_ui
	cmp	qword [ui_autoclose], 0
	jne	.ui_done
	jmp	.ui_loop
.ui_done:
	call	CloseWindow
	xor	rax, rax
	ret

render_ui:
	call	BeginDrawing
	ccall1	ClearBackground, COLOR_BG
	ui_text	ui_title, 18, 14, 20, COLOR_TEXT
	ui_text	ui_hint, 760, 18, 14, COLOR_MUTED
	ccall5	DrawRectangle, 16, 48, 1068, 48, COLOR_PANEL
	ui_text	ui_target_label, 28, 62, 14, COLOR_MUTED
	mov	rdi, [target_path]
	mov	rsi, 260
	call	copy_cstr_limited
	ui_text	ui_line_buf, 88, 62, 14, COLOR_TEXT
	call	draw_status_badge
	cmp	qword [ui_show_raw], 0
	je	.rui_dashboard
	call	render_raw_view
	jmp	.rui_done
.rui_dashboard:
	call	render_dashboard
.rui_done:
	call	EndDrawing
	ret

draw_status_badge:
	mov	rax, COLOR_ACCENT
	mov	rdi, [status_ptr]
	lea	rsi, [status_exited]
	call	str_eq
	cmp	rax, 1
	jne	.dsb_crash
	mov	rax, COLOR_EXITED
	jmp	.dsb_draw
.dsb_crash:
	mov	rdi, [status_ptr]
	lea	rsi, [status_crashed]
	call	str_eq
	cmp	rax, 1
	jne	.dsb_draw
	mov	rax, COLOR_CRASHED
	jmp	.dsb_draw
.dsb_default:
	mov	rax, COLOR_ACCENT
.dsb_draw:
	cmp	rax, 1
	jbe	.dsb_default
	mov	[ui_status_color], rax
	mov	r8, rax
	ccall5	DrawRectangle, 930, 58, 132, 26, r8
	mov	rdi, [status_ptr]
	mov	rsi, 40
	call	copy_cstr_limited
	ui_text	ui_line_buf, 942, 64, 13, COLOR_BG
	ret

render_dashboard:
	ccall5	DrawRectangle, 16, 112, 280, 430, COLOR_PANEL
	ccall5	DrawRectangle, 312, 112, 452, 266, COLOR_PANEL
	ccall5	DrawRectangle, 780, 112, 304, 266, COLOR_PANEL
	ccall5	DrawRectangle, 312, 394, 772, 294, COLOR_PANEL_DARK
	ui_text	ui_summary_title, 30, 126, 16, COLOR_TEXT
	call	draw_summary_panel
	ui_text	ui_backtrace_title, 328, 126, 16, COLOR_TEXT
	mov	qword [ui_line_cap], 58
	lea	rdi, [pat_thread]
	mov	rsi, pat_thread_len
	mov	rdx, 328
	mov	rcx, 154
	mov	r8, 8
	call	draw_lldb_lines_from_pattern
	ui_text	ui_registers_title, 796, 126, 16, COLOR_TEXT
	mov	qword [ui_line_cap], 34
	lea	rdi, [reg_pat_rip]
	mov	rsi, reg_pat_rip_len
	mov	rdx, 796
	mov	rcx, 154
	mov	r8, 9
	call	draw_lldb_lines_from_pattern
	ui_text	ui_disasm_title, 328, 410, 16, COLOR_TEXT
	mov	qword [ui_line_cap], 86
	lea	rdi, [pat_disasm_arrow]
	mov	rsi, pat_disasm_arrow_len
	mov	rdx, 328
	mov	rcx, 438
	mov	r8, 6
	call	draw_lldb_lines_from_pattern
	ui_text	ui_stack_title, 328, 580, 16, COLOR_TEXT
	mov	qword [ui_line_cap], 86
	lea	rdi, [reg_pat_rsp]
	mov	rsi, reg_pat_rsp_len
	mov	rdx, 328
	mov	rcx, 608
	mov	r8, 4
	call	draw_lldb_lines_from_pattern
	ret

draw_summary_panel:
	ui_text	ui_status_label, 30, 162, 14, COLOR_MUTED
	mov	rdi, [status_ptr]
	mov	rsi, 48
	call	copy_cstr_limited
	ui_text	ui_line_buf, 116, 162, 14, COLOR_TEXT
	ui_text	ui_signal_label, 30, 192, 14, COLOR_MUTED
	call	copy_signal_text
	ui_text	ui_line_buf, 116, 192, 14, COLOR_TEXT
	ui_text	ui_exit_label, 30, 222, 14, COLOR_MUTED
	call	copy_exit_text
	ui_text	ui_line_buf, 116, 222, 14, COLOR_TEXT
	ui_text	ui_macho_label, 30, 252, 14, COLOR_MUTED
	ui_text	ui_macho_text, 116, 252, 14, COLOR_TEXT
	ui_text	ui_summary_label, 30, 302, 14, COLOR_MUTED
	mov	rdi, [status_ptr]
	lea	rsi, [status_crashed]
	call	str_eq
	cmp	rax, 1
	je	.dsp_crash
	mov	rdi, [status_ptr]
	lea	rsi, [status_exited]
	call	str_eq
	cmp	rax, 1
	je	.dsp_exit
	ui_text	summary_error, 30, 330, 14, COLOR_TEXT
	ret
.dsp_crash:
	ui_text	ui_summary_crash_short, 30, 330, 14, COLOR_TEXT
	ret
.dsp_exit:
	ui_text	ui_summary_exit_short, 30, 330, 14, COLOR_TEXT
	ret

render_raw_view:
	ccall5	DrawRectangle, 16, 112, 1068, 576, COLOR_PANEL_DARK
	ui_text	ui_raw_title, 30, 126, 16, COLOR_TEXT
	cmp	qword [json_out_len], 0
	je	.rrv_lldb
	mov	qword [ui_line_cap], 130
	lea	rdi, [report_buf]
	mov	rsi, [json_out_len]
	mov	rdx, 30
	mov	rcx, 156
	mov	r8, 30
	call	draw_lldb_lines
	ret
.rrv_lldb:
	mov	qword [ui_line_cap], 130
	lea	rdi, [lldb_out_buf]
	mov	rsi, [lldb_out_len]
	mov	rdx, 30
	mov	rcx, 156
	mov	r8, 30
	call	draw_lldb_lines
	ret

; rdi = pattern, rsi = pattern len, rdx = x, rcx = y, r8 = max lines
draw_lldb_lines_from_pattern:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	[ui_draw_x], rdx
	mov	[ui_draw_y], rcx
	mov	r15, r8
	mov	r12, rdi
	mov	r13, rsi
	lea	rdi, [lldb_out_buf]
	mov	rsi, [lldb_out_len]
	mov	rdx, r12
	mov	rcx, r13
	call	mem_find
	test	rax, rax
	jz	.dll_missing
	mov	rdi, rax
	lea	rax, [lldb_out_buf]
	add	rax, [lldb_out_len]
	sub	rax, rdi
	mov	rsi, rax
	mov	rdx, [ui_draw_x]
	mov	rcx, [ui_draw_y]
	mov	r8, r15
	call	draw_lldb_lines
	jmp	.dll_done
.dll_missing:
	ui_text	ui_missing_text, [ui_draw_x], [ui_draw_y], 14, COLOR_MUTED
.dll_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = bytes, rsi = len, rdx = x, rcx = y, r8 = max lines
draw_lldb_lines:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rcx
	mov	r15, r8
	mov	[ui_draw_x], rdx
	xor	rbx, rbx
.dllines_loop:
	cmp	rbx, r15
	jae	.dllines_done
	cmp	r13, 0
	jbe	.dllines_done
	mov	rdi, r12
	mov	rsi, r13
	lea	rdx, [ui_line_buf]
	mov	rcx, [ui_line_cap]
	call	copy_line
	mov	[ui_line_consumed], rax
	ui_text	ui_line_buf, [ui_draw_x], r14, 13, COLOR_TEXT
	mov	rax, [ui_line_consumed]
	add	r12, rax
	sub	r13, rax
	add	r14, 18
	inc	rbx
	jmp	.dllines_loop
.dllines_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = src, rsi = len, rdx = dst, rcx = dst cap; rax = consumed bytes
copy_line:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdx
	mov	r14, rcx
	dec	r14
	xor	rbx, rbx
	xor	r13, r13
	xor	r10, r10
.cl_loop:
	cmp	rbx, rsi
	jae	.cl_finish
	mov	al, [rdi + rbx]
	cmp	al, 10
	je	.cl_newline
	cmp	r13, r14
	jae	.cl_truncated
	mov	[r12 + r13], al
	inc	r13
	inc	rbx
	jmp	.cl_loop
.cl_truncated:
	mov	r10, 1
	inc	rbx
	jmp	.cl_loop
.cl_newline:
	inc	rbx
.cl_store_zero:
	jmp	.cl_terminate
.cl_finish:
	jmp	.cl_terminate
.cl_terminate:
	cmp	r10, 0
	je	.cl_plain_zero
	cmp	r14, 4
	jb	.cl_plain_zero
	mov	r11, r14
	sub	r11, 3
	mov	byte [r12 + r11], '.'
	inc	r11
	mov	byte [r12 + r11], '.'
	inc	r11
	mov	byte [r12 + r11], '.'
	mov	byte [r12 + r14], 0
	jmp	.cl_done
.cl_plain_zero:
	mov	byte [r12 + r13], 0
.cl_done:
	mov	rax, rbx
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = cstr, rsi = cap
copy_cstr_limited:
	push	rbx
	mov	rbx, rsi
	dec	rbx
	lea	rdx, [ui_line_buf]
	xor	rcx, rcx
.ccl_loop:
	cmp	rcx, rbx
	jae	.ccl_done
	mov	al, [rdi + rcx]
	test	al, al
	jz	.ccl_done
	mov	[rdx + rcx], al
	inc	rcx
	jmp	.ccl_loop
.ccl_done:
	mov	byte [rdx + rcx], 0
	pop	rbx
	ret

copy_signal_text:
	lea	rdi, [lldb_out_buf]
	mov	rsi, [lldb_out_len]
	lea	rdx, [lldb_pat_exc_bad]
	mov	rcx, lldb_pat_exc_bad_len
	call	mem_find
	test	rax, rax
	jz	.cst_none
	lea	rdi, [sig_exc_bad]
	mov	rsi, 32
	jmp	copy_cstr_limited
.cst_none:
	lea	rdi, [ui_none_text]
	mov	rsi, 32
	jmp	copy_cstr_limited

copy_exit_text:
	lea	rdi, [lldb_out_buf]
	mov	rsi, [lldb_out_len]
	lea	rdx, [lldb_pat_exited]
	mov	rcx, lldb_pat_exited_len
	call	mem_find
	test	rax, rax
	jz	.cet_none
	add	rax, lldb_pat_exited_len
.cet_skip:
	cmp	byte [rax], 32
	jne	.cet_copy
	inc	rax
	jmp	.cet_skip
.cet_copy:
	mov	rdi, rax
	mov	rsi, 16
	jmp	copy_cstr_limited
.cet_none:
	lea	rdi, [ui_null_text]
	mov	rsi, 16
	jmp	copy_cstr_limited

write_lldb_commands:
	mov	rax, SYS_open
	lea	rdi, [lldb_cmd_path]
	mov	rsi, O_WRONLY + O_CREAT + O_TRUNC
	mov	rdx, 420
	syscall
	jc	.wlc_err
	mov	[cmd_fd], rax
	lea	rdi, [cmd_settings]
	call	cmd_write_cstr
	lea	rdi, [cmd_target]
	call	cmd_write_cstr
	mov	rdi, [target_path]
	call	cmd_write_cstr
	lea	rdi, [nl]
	call	cmd_write_cstr
	cmp	qword [args_start], 0
	je	.wlc_no_args
	lea	rdi, [cmd_run_args]
	call	cmd_write_cstr
	call	cmd_write_args
	lea	rdi, [nl]
	call	cmd_write_cstr
.wlc_no_args:
	lea	rdi, [cmd_run]
	call	cmd_write_cstr
	lea	rdi, [cmd_status]
	call	cmd_write_cstr
	lea	rdi, [cmd_registers]
	call	cmd_write_cstr
	lea	rdi, [cmd_bt]
	call	cmd_write_cstr
	lea	rdi, [cmd_disasm]
	call	cmd_write_cstr
	lea	rdi, [cmd_memory]
	call	cmd_write_cstr
	lea	rdi, [cmd_images]
	call	cmd_write_cstr
	mov	rdi, [cmd_fd]
	mov	rax, SYS_close
	syscall
	xor	rax, rax
	ret
.wlc_err:
	mov	rax, 1
	ret

; rdi = cstr
cmd_write_cstr:
	push	rdi
	call	str_len
	mov	rdx, rax
	pop	rsi
	mov	rdi, [cmd_fd]
	mov	rax, SYS_write
	syscall
	ret

cmd_write_args:
	push	rbx
	mov	rbx, [args_start]
.cwa_loop:
	cmp	rbx, [args_end]
	jae	.cwa_done
	cmp	rbx, [args_start]
	je	.cwa_arg
	lea	rdi, [space]
	call	cmd_write_cstr
.cwa_arg:
	mov	rax, [argv]
	mov	rdi, [rax + rbx * 8]
	call	cmd_write_cstr
	inc	rbx
	jmp	.cwa_loop
.cwa_done:
	pop	rbx
	ret

write_report:
	lea	rax, [report_buf]
	mov	[json_out_ptr], rax
	mov	qword [json_out_len], 0
	mov	qword [json_out_cap], REPORT_CAP
	lea	rdi, [json_open]
	call	json_append_cstr
	lea	rdi, [json_target_key]
	call	json_append_cstr
	mov	rdi, [target_path]
	call	str_len
	mov	rsi, rax
	mov	rdi, [target_path]
	call	json_append_escaped
	lea	rdi, [json_status_key]
	call	json_append_cstr
	mov	rdi, [status_ptr]
	call	str_len
	mov	rsi, rax
	mov	rdi, [status_ptr]
	call	json_append_escaped
	lea	rdi, [json_exit_code_key]
	call	json_append_cstr
	call	append_exit_code
	lea	rdi, [json_signal_key]
	call	json_append_cstr
	call	append_signal
	lea	rdi, [json_summary_key]
	call	json_append_cstr
	call	append_summary
	lea	rdi, [json_registers_open]
	call	json_append_cstr
	call	append_registers
	lea	rdi, [json_registers_close]
	call	json_append_cstr
	lea	rdi, [json_backtrace_key]
	call	json_append_cstr
	lea	rdi, [pat_bt_cmd]
	mov	rsi, pat_bt_cmd_len
	mov	rdx, 20
	call	append_section_lines
	lea	rdi, [json_disasm_key]
	call	json_append_cstr
	lea	rdi, [pat_disasm_cmd]
	mov	rsi, pat_disasm_cmd_len
	mov	rdx, 36
	call	append_section_lines
	lea	rdi, [json_stack_key]
	call	json_append_cstr
	lea	rdi, [pat_memory_cmd]
	mov	rsi, pat_memory_cmd_len
	mov	rdx, 40
	call	append_section_lines
	lea	rdi, [json_macho]
	call	json_append_cstr
	lea	rdi, [json_raw_key]
	call	json_append_cstr
	lea	rdi, [lldb_out_buf]
	mov	rsi, [lldb_out_len]
	call	json_append_escaped
	lea	rdi, [json_close]
	call	json_append_cstr
	jmp	write_report_file

write_error_report:
	lea	rax, [status_lldb_error]
	mov	[status_ptr], rax
	lea	rax, [report_buf]
	mov	[json_out_ptr], rax
	mov	qword [json_out_len], 0
	mov	qword [json_out_cap], REPORT_CAP
	lea	rdi, [json_open]
	call	json_append_cstr
	lea	rdi, [json_target_key]
	call	json_append_cstr
	mov	rdi, [target_path]
	call	str_len
	mov	rsi, rax
	mov	rdi, [target_path]
	call	json_append_escaped
	lea	rdi, [json_status_key]
	call	json_append_cstr
	lea	rdi, [status_lldb_error]
	mov	rsi, status_lldb_error_len
	call	json_append_escaped
	lea	rdi, [json_exit_code_null]
	call	json_append_cstr
	lea	rdi, [json_signal_null]
	call	json_append_cstr
	lea	rdi, [json_summary_error]
	call	json_append_cstr
	lea	rdi, [json_minimal_tail]
	call	json_append_cstr
	lea	rdi, [json_close]
	call	json_append_cstr
	call	write_report_file
	mov	rax, 3
	ret

write_report_file:
	mov	rax, SYS_open
	mov	rdi, [report_path]
	mov	rsi, O_WRONLY + O_CREAT + O_TRUNC
	mov	rdx, 420
	syscall
	jc	.wrf_err
	mov	r8, rax
	mov	rdi, r8
	lea	rsi, [report_buf]
	mov	rdx, [json_out_len]
	mov	rax, SYS_write
	syscall
	mov	rdi, r8
	mov	rax, SYS_close
	syscall
	mov	rdi, [status_ptr]
	lea	rsi, [status_permission]
	call	str_eq
	cmp	rax, 1
	je	.wrf_perm
	mov	rdi, [status_ptr]
	lea	rsi, [status_lldb_error]
	call	str_eq
	cmp	rax, 1
	je	.wrf_perm
	xor	rax, rax
	ret
.wrf_perm:
	mov	rax, 3
	ret
.wrf_err:
	mov	rax, 2
	ret

append_signal:
	lea	rdi, [lldb_out_buf]
	mov	rsi, [lldb_out_len]
	lea	rdx, [lldb_pat_exc_bad]
	mov	rcx, lldb_pat_exc_bad_len
	call	mem_find
	test	rax, rax
	jz	.as_null
	lea	rdi, [sig_exc_bad]
	mov	rsi, sig_exc_bad_len
	jmp	json_append_escaped
.as_null:
	lea	rdi, [json_null]
	jmp	json_append_cstr

append_exit_code:
	push	rbx
	push	r12
	lea	rdi, [lldb_out_buf]
	mov	rsi, [lldb_out_len]
	lea	rdx, [lldb_pat_exited]
	mov	rcx, lldb_pat_exited_len
	call	mem_find
	test	rax, rax
	jz	.aec_null
	add	rax, lldb_pat_exited_len
.aec_skip:
	cmp	byte [rax], 32
	jne	.aec_start
	inc	rax
	jmp	.aec_skip
.aec_start:
	mov	r12, rax
	xor	rbx, rbx
.aec_len:
	cmp	rbx, 3
	jae	.aec_emit
	mov	al, [r12 + rbx]
	cmp	al, '0'
	jb	.aec_emit
	cmp	al, '9'
	ja	.aec_emit
	inc	rbx
	jmp	.aec_len
.aec_emit:
	test	rbx, rbx
	jz	.aec_null
	mov	rdi, r12
	mov	rsi, rbx
	call	json_append
	jmp	.aec_done
.aec_null:
	lea	rdi, [json_null]
	call	json_append_cstr
.aec_done:
	pop	r12
	pop	rbx
	ret

append_summary:
	mov	rdi, [status_ptr]
	lea	rsi, [status_crashed]
	call	str_eq
	cmp	rax, 1
	je	.sum_crash
	mov	rdi, [status_ptr]
	lea	rsi, [status_exited]
	call	str_eq
	cmp	rax, 1
	je	.sum_exit
	lea	rdi, [summary_error]
	mov	rsi, summary_error_len
	jmp	json_append_escaped
.sum_crash:
	lea	rdi, [summary_crash]
	mov	rsi, summary_crash_len
	jmp	json_append_escaped
.sum_exit:
	lea	rdi, [summary_exit]
	mov	rsi, summary_exit_len
	jmp	json_append_escaped

append_registers:
	lea	rdi, [reg_key_rip]
	call	json_append_cstr
	lea	rdi, [reg_pat_rip]
	mov	rsi, reg_pat_rip_len
	call	append_reg_value
	lea	rdi, [reg_key_rsp]
	call	json_append_cstr
	lea	rdi, [reg_pat_rsp]
	mov	rsi, reg_pat_rsp_len
	call	append_reg_value
	lea	rdi, [reg_key_rbp]
	call	json_append_cstr
	lea	rdi, [reg_pat_rbp]
	mov	rsi, reg_pat_rbp_len
	call	append_reg_value
	lea	rdi, [reg_key_rax]
	call	json_append_cstr
	lea	rdi, [reg_pat_rax]
	mov	rsi, reg_pat_rax_len
	call	append_reg_value
	lea	rdi, [reg_key_rbx]
	call	json_append_cstr
	lea	rdi, [reg_pat_rbx]
	mov	rsi, reg_pat_rbx_len
	call	append_reg_value
	lea	rdi, [reg_key_rcx]
	call	json_append_cstr
	lea	rdi, [reg_pat_rcx]
	mov	rsi, reg_pat_rcx_len
	call	append_reg_value
	lea	rdi, [reg_key_rdx]
	call	json_append_cstr
	lea	rdi, [reg_pat_rdx]
	mov	rsi, reg_pat_rdx_len
	jmp	append_reg_value

; rdi = pattern ptr, rsi = pattern len
append_reg_value:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	lea	rdi, [lldb_out_buf]
	mov	rsi, [lldb_out_len]
	mov	rdx, r12
	mov	rcx, r13
	call	mem_find
	test	rax, rax
	jz	.arv_empty
	add	rax, r13
	mov	r12, rax
	xor	rbx, rbx
.arv_len:
	cmp	rbx, 32
	jae	.arv_emit
	mov	al, [r12 + rbx]
	cmp	al, 32
	jbe	.arv_emit
	inc	rbx
	jmp	.arv_len
.arv_emit:
	mov	rdi, r12
	mov	rsi, rbx
	call	json_append_escaped
	jmp	.arv_done
.arv_empty:
	lea	rdi, [empty_string]
	mov	rsi, 0
	call	json_append_escaped
.arv_done:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = command pattern, rsi = pattern len, rdx = max lines
append_section_lines:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	lea	rdi, [json_array_open]
	call	json_append_cstr
	lea	rdi, [lldb_out_buf]
	mov	rsi, [lldb_out_len]
	mov	rdx, r12
	mov	rcx, r13
	call	mem_find
	test	rax, rax
	jz	.asl_close
	add	rax, r13
	mov	r12, rax
	lea	rax, [lldb_out_buf]
	add	rax, [lldb_out_len]
	sub	rax, r12
	mov	r13, rax
	xor	rbx, rbx
	xor	r15, r15
.asl_loop:
	cmp	rbx, r14
	jae	.asl_close
	cmp	r13, 0
	jbe	.asl_close
	cmp	r13, 6
	jb	.asl_copy
	cmp	byte [r12], '('
	jne	.asl_copy
	cmp	byte [r12 + 1], 'l'
	jne	.asl_copy
	cmp	byte [r12 + 2], 'l'
	jne	.asl_copy
	cmp	byte [r12 + 3], 'd'
	jne	.asl_copy
	cmp	byte [r12 + 4], 'b'
	jne	.asl_copy
	cmp	byte [r12 + 5], ')'
	je	.asl_close
.asl_copy:
	mov	rdi, r12
	mov	rsi, r13
	lea	rdx, [ui_line_buf]
	mov	rcx, 512
	call	copy_line
	mov	[ui_line_consumed], rax
	cmp	byte [ui_line_buf], 0
	je	.asl_next
	cmp	r15, 0
	je	.asl_emit
	lea	rdi, [json_comma_space]
	call	json_append_cstr
.asl_emit:
	lea	rdi, [ui_line_buf]
	call	str_len
	mov	rsi, rax
	lea	rdi, [ui_line_buf]
	call	json_append_escaped
	mov	r15, 1
	inc	rbx
.asl_next:
	mov	rax, [ui_line_consumed]
	add	r12, rax
	sub	r13, rax
	jmp	.asl_loop
.asl_close:
	lea	rdi, [json_array_close]
	call	json_append_cstr
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = null-terminated string
; rax = length
str_len:
	xor	rax, rax
.sl_loop:
	cmp	byte [rdi + rax], 0
	je	.sl_done
	inc	rax
	jmp	.sl_loop
.sl_done:
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

section ".data" writeable

opt_help db '--help', 0
opt_snapshot db '--snapshot', 0
opt_ui db '--ui', 0
opt_args db '--args', 0
opt_dashdash db '--', 0
usage_msg db 'usage: macdbg --snapshot <program> <report.json>', 10
	db '       macdbg --snapshot --args <program> [args...] -- <report.json>', 10
	db '       macdbg --ui <program>', 10
	db '       macdbg --ui --args <program> [args...]', 10
usage_msg_len = $ - usage_msg

lldb_cmd_path db '/tmp/macdbg-lldb.cmds', 0
lldb_out_path db '/tmp/macdbg-lldb.out', 0
lldb_shell_cmd db '/usr/bin/lldb --batch -s /tmp/macdbg-lldb.cmds > /tmp/macdbg-lldb.out 2>&1', 0
ui_report_path db '/tmp/macdbg-ui-report.json', 0
env_ui_autoclose db 'MACDBG_UI_AUTOCLOSE', 0

cmd_settings db 'settings set target.x86-disassembly-flavor intel', 10
	db 'settings set stop-disassembly-count 32', 10, 0
cmd_target db 'target create ', 0
cmd_run_args db 'settings set -- target.run-args ', 0
cmd_run db 'run', 10, 0
cmd_status db 'process status', 10, 0
cmd_registers db 'register read --format hex rip rsp rbp rax rbx rcx rdx', 10, 0
cmd_bt db 'thread backtrace --count 16', 10, 0
cmd_disasm db 'disassemble --pc --count 32', 10, 0
cmd_memory db 'memory read --format x --size 8 --count 32 $rsp', 10, 0
cmd_images db 'image list', 10, 0
nl db 10, 0
space db ' ', 0

json_open db '{', 10, '  "tool": "macdbg",', 10, '  "mode": "snapshot",', 10, 0
json_target_key db '  "target": ', 0
json_status_key db ',', 10, '  "status": ', 0
json_exit_code_key db ',', 10, '  "exit_code": ', 0
json_exit_code_null db ',', 10, '  "exit_code": null', 0
json_signal_key db ',', 10, '  "signal": ', 0
json_summary_key db ',', 10, '  "summary": ', 0
json_registers_open db ',', 10, '  "registers": {', 0
json_registers_close db '}', 0
json_backtrace_key db ',', 10, '  "backtrace": ', 0
json_disasm_key db ',', 10, '  "disasm": ', 0
json_stack_key db ',', 10, '  "stack_memory": ', 0
json_macho db ',', 10, '  "macho": {"format": "Mach-O 64-bit", "arch": "x86_64"}', 0
json_raw_key db ',', 10, '  "raw_tail": [', 0
json_close db ']', 10, '}', 10, 0
json_null db 'null', 0
json_array_open db '[', 0
json_array_close db ']', 0
json_comma_space db ', ', 0
json_signal_null db ',', 10, '  "signal": null', 0
json_summary_error db ',', 10, '  "summary": "Unable to run LLDB snapshot"', 0
json_minimal_tail db ',', 10, '  "registers": {},', 10, '  "backtrace": [],', 10, '  "disasm": [],', 10, '  "stack_memory": [],', 10, '  "macho": {},', 10, '  "raw_tail": ["macdbg could not capture LLDB output"]', 0

sig_exc_bad db 'EXC_BAD_ACCESS'
sig_exc_bad_len = $ - sig_exc_bad
summary_crash db 'Stopped on invalid memory access or exception'
summary_crash_len = $ - summary_crash
summary_exit db 'Target exited under LLDB'
summary_exit_len = $ - summary_exit
summary_error db 'LLDB did not produce a recognized stopped or exited state'
summary_error_len = $ - summary_error
status_lldb_error_len = 10
empty_string db 0
reg_key_rip db '"rip": ', 0
reg_key_rsp db ', "rsp": ', 0
reg_key_rbp db ', "rbp": ', 0
reg_key_rax db ', "rax": ', 0
reg_key_rbx db ', "rbx": ', 0
reg_key_rcx db ', "rcx": ', 0
reg_key_rdx db ', "rdx": ', 0
reg_pat_rip db 'rip = '
reg_pat_rip_len = $ - reg_pat_rip
reg_pat_rsp db 'rsp = '
reg_pat_rsp_len = $ - reg_pat_rsp
reg_pat_rbp db 'rbp = '
reg_pat_rbp_len = $ - reg_pat_rbp
reg_pat_rax db 'rax = '
reg_pat_rax_len = $ - reg_pat_rax
reg_pat_rbx db 'rbx = '
reg_pat_rbx_len = $ - reg_pat_rbx
reg_pat_rcx db 'rcx = '
reg_pat_rcx_len = $ - reg_pat_rcx
reg_pat_rdx db 'rdx = '
reg_pat_rdx_len = $ - reg_pat_rdx
json_one_byte db 0
json_esc_quote db 92, 34
json_esc_slash db 92, 92
json_esc_nl db 92, 'n'
json_esc_tab db 92, 't'

ui_title db 'macdbg snapshot viewer', 0
ui_hint db 'R rerun snapshot   J raw JSON/LLDB   Esc quit', 0
ui_target_label db 'target', 0
ui_summary_title db 'Summary', 0
ui_backtrace_title db 'Backtrace', 0
ui_registers_title db 'Registers', 0
ui_disasm_title db 'Disassembly near PC', 0
ui_stack_title db 'Stack / Memory', 0
ui_raw_title db 'Raw snapshot JSON / LLDB tail', 0
ui_status_label db 'status', 0
ui_signal_label db 'signal', 0
ui_exit_label db 'exit', 0
ui_macho_label db 'macho', 0
ui_summary_label db 'summary', 0
ui_macho_text db 'Mach-O 64-bit / x86_64', 0
ui_missing_text db '(not available in LLDB output)', 0
ui_none_text db 'none', 0
ui_null_text db 'null', 0
ui_summary_crash_short db 'Invalid memory access', 0
ui_summary_exit_short db 'Target exited under LLDB', 0
pat_thread db 'thread #'
pat_thread_len = $ - pat_thread
pat_disasm_arrow db '->'
pat_disasm_arrow_len = $ - pat_disasm_arrow
pat_bt_cmd db 'thread backtrace --count 16', 10
pat_bt_cmd_len = $ - pat_bt_cmd
pat_disasm_cmd db 'disassemble --pc --count 32', 10
pat_disasm_cmd_len = $ - pat_disasm_cmd
pat_memory_cmd db 'memory read --format x --size 8 --count 32 $rsp', 10
pat_memory_cmd_len = $ - pat_memory_cmd

section ".bss" writeable

argc dq ?
argv dq ?
target_path dq ?
report_path dq ?
args_start dq ?
args_end dq ?
cmd_fd dq ?
system_status dq ?
status_ptr dq ?
lldb_out_len dq ?
json_out_ptr dq ?
json_out_len dq ?
json_out_cap dq ?
ui_show_raw dq ?
ui_autoclose dq ?
ui_status_color dq ?
ui_draw_x dq ?
ui_draw_y dq ?
ui_line_cap dq ?
ui_line_consumed dq ?

report_buf rb REPORT_CAP
lldb_out_buf rb MACDBG_OUT_CAP
ui_line_buf rb 512
