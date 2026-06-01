; machodoctor: standalone Mach-O binary inspector.

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/macho.inc"

FILE_BUF_SIZE equ 4194304

MODE_HUMAN equ 0
MODE_JSON equ 1
MODE_DEPS equ 2
MODE_CHECK equ 3

entry start

start:
	mov	rbx, rsp
	mov	rax, [rbx]
	mov	[argc], rax
	cmp	rax, 2
	jb	usage
	mov	qword [mode], MODE_HUMAN
	mov	qword [arg_index], 1
	call	parse_args
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	usage
	mov	rax, [rbx + 8 + rax * 8]
	mov	[path_ptr], rax
	mov	rdi, rax
	call	read_target
	cmp	rax, 2
	je	exit_error
	call	inspect_macho
	cmp	rax, 2
	je	exit_error
	cmp	qword [mode], MODE_CHECK
	jne	exit_ok
	cmp	qword [check_issue_count], 0
	jne	exit_check_failed
exit_ok:
	exit EXIT_SUCCESS
exit_check_failed:
	exit EXIT_FAILURE
exit_error:
	exit 2

usage:
	lea	rdi, [usage_msg]
	mov	rsi, usage_msg_len
	call	write_stderr
	exit 2

parse_args:
.pa_loop:
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	usage
	mov	rdi, [rbx + 8 + rax * 8]
	cmp	byte [rdi], '-'
	jne	.pa_done
	cmp	byte [rdi + 1], '-'
	jne	usage
	lea	rsi, [opt_json]
	call	str_eq
	test	rax, rax
	jnz	.pa_json
	mov	rax, [arg_index]
	mov	rdi, [rbx + 8 + rax * 8]
	lea	rsi, [opt_deps]
	call	str_eq
	test	rax, rax
	jnz	.pa_deps
	mov	rax, [arg_index]
	mov	rdi, [rbx + 8 + rax * 8]
	lea	rsi, [opt_check]
	call	str_eq
	test	rax, rax
	jnz	.pa_check
	jmp	usage
.pa_json:
	mov	qword [mode], MODE_JSON
	jmp	.pa_next
.pa_deps:
	mov	qword [mode], MODE_DEPS
	jmp	.pa_next
.pa_check:
	mov	qword [mode], MODE_CHECK
.pa_next:
	inc	qword [arg_index]
	jmp	.pa_loop
.pa_done:
	ret

; rdi = path
; rax = 0 success, 2 error
read_target:
	mov	[read_path], rdi
	open_file rdi, O_RDONLY, 0
	jump_if_syscall_error .rt_open_error
	mov	[file_fd], rax
	mov	qword [file_size], 0
.rt_loop:
	mov	rax, FILE_BUF_SIZE
	sub	rax, [file_size]
	jz	.rt_too_large
	mov	rdi, [file_fd]
	lea	rsi, [file_buf]
	add	rsi, [file_size]
	mov	rdx, rax
	mov	rax, SYS_read
	syscall
	jc	.rt_read_error
	test	rax, rax
	jz	.rt_done
	add	[file_size], rax
	jmp	.rt_loop
.rt_done:
	close_file [file_fd]
	xor	rax, rax
	ret
.rt_too_large:
	close_file [file_fd]
	lea	rdi, [too_large_msg]
	mov	rsi, too_large_msg_len
	call	write_stderr
	mov	rax, 2
	ret
.rt_open_error:
	lea	rdi, [open_err_msg]
	mov	rsi, open_err_msg_len
	call	write_stderr
	mov	rdi, [read_path]
	call	write_cstr_stderr
	mov	al, 10
	call	write_char_stderr
	mov	rax, 2
	ret
.rt_read_error:
	close_file [file_fd]
	lea	rdi, [read_err_msg]
	mov	rsi, read_err_msg_len
	call	write_stderr
	mov	rax, 2
	ret

; rax = 0 success, 2 parse error
inspect_macho:
	call	reset_state
	cmp	qword [file_size], MACH_HEADER_64_SIZE
	jb	parse_error
	mov	eax, dword [file_buf + MH_MAGIC_OFF]
	cmp	eax, FAT_CIGAM
	je	.im_fat
	cmp	eax, FAT_MAGIC
	je	.im_fat
	cmp	eax, MH_MAGIC_64
	jne	parse_error
.im_thin:
	mov	eax, dword [file_buf + MH_CPUTYPE_OFF]
	mov	[cpu_type], rax
	mov	eax, dword [file_buf + MH_FILETYPE_OFF]
	mov	[file_type], rax
	mov	eax, dword [file_buf + MH_NCMDS_OFF]
	mov	[ncmds], rax
	mov	eax, dword [file_buf + MH_SIZEOFCMDS_OFF]
	mov	[sizeofcmds], rax
	mov	rax, [sizeofcmds]
	add	rax, MACH_HEADER_64_SIZE
	cmp	rax, [file_size]
	ja	parse_error
	call	walk_load_commands
	cmp	rax, 2
	je	parse_error
	cmp	qword [mode], MODE_JSON
	je	print_json
	cmp	qword [mode], MODE_DEPS
	je	print_deps
	cmp	qword [mode], MODE_CHECK
	je	print_check
	call	print_human
	xor	rax, rax
	ret
.im_fat:
	call	select_fat_x86_64_slice
	cmp	rax, 2
	je	parse_error
	mov	eax, dword [file_buf + MH_MAGIC_OFF]
	cmp	eax, MH_MAGIC_64
	jne	parse_error
	jmp	.im_thin

parse_error:
	lea	rdi, [parse_err_msg]
	mov	rsi, parse_err_msg_len
	call	write_stderr
	mov	rax, 2
	ret

select_fat_x86_64_slice:
	cmp	qword [file_size], 8
	jb	.sfx_err
	mov	rdi, 4
	call	read_be32
	mov	r12, rax
	mov	r13, 8
.sfx_loop:
	test	r12, r12
	jz	.sfx_err
	mov	rax, r13
	add	rax, 20
	cmp	rax, [file_size]
	ja	.sfx_err
	mov	rdi, r13
	call	read_be32
	cmp	rax, CPU_TYPE_X86_64
	je	.sfx_found
	add	r13, 20
	dec	r12
	jmp	.sfx_loop
.sfx_found:
	mov	rdi, r13
	add	rdi, 8
	call	read_be32
	mov	r14, rax
	mov	rdi, r13
	add	rdi, 12
	call	read_be32
	mov	r15, rax
	mov	rax, r14
	add	rax, r15
	cmp	rax, [file_size]
	ja	.sfx_err
	cmp	r15, FILE_BUF_SIZE
	ja	.sfx_err
	xor	r10, r10
.sfx_copy:
	cmp	r10, r15
	jae	.sfx_done
	mov	al, [file_buf + r14 + r10]
	mov	[file_buf + r10], al
	inc	r10
	jmp	.sfx_copy
.sfx_done:
	mov	[file_size], r15
	xor	rax, rax
	ret
.sfx_err:
	mov	rax, 2
	ret

; rdi = offset in file_buf; rax = big-endian u32
read_be32:
	xor	rax, rax
	movzx	rcx, byte [file_buf + rdi]
	shl	rcx, 24
	or	rax, rcx
	movzx	rcx, byte [file_buf + rdi + 1]
	shl	rcx, 16
	or	rax, rcx
	movzx	rcx, byte [file_buf + rdi + 2]
	shl	rcx, 8
	or	rax, rcx
	movzx	rcx, byte [file_buf + rdi + 3]
	or	rax, rcx
	ret

reset_state:
	mov	qword [dylib_count], 0
	mov	qword [rpath_count], 0
	mov	qword [segment_count], 0
	mov	qword [has_code_signature], 0
	mov	qword [has_build_version], 0
	mov	qword [has_entry], 0
	mov	qword [entry_value], 0
	mov	qword [minos_value], 0
	mov	qword [sdk_value], 0
	mov	qword [check_issue_count], 0
	ret

walk_load_commands:
	mov	r12, MACH_HEADER_64_SIZE
	xor	r13, r13
.wlc_loop:
	cmp	r13, [ncmds]
	jae	.wlc_ok
	mov	rax, r12
	add	rax, 8
	cmp	rax, [file_size]
	ja	.wlc_err
	mov	eax, dword [file_buf + r12 + LC_CMD_OFF]
	mov	[cmd_value], rax
	mov	eax, dword [file_buf + r12 + LC_CMDSIZE_OFF]
	mov	[cmd_size], rax
	cmp	qword [cmd_size], 8
	jb	.wlc_err
	mov	rax, r12
	add	rax, [cmd_size]
	cmp	rax, [file_size]
	ja	.wlc_err
	mov	eax, dword [cmd_value]
	cmp	eax, LC_SEGMENT_64
	je	.wlc_segment
	cmp	eax, LC_LOAD_DYLIB
	je	.wlc_dylib
	cmp	eax, LC_LOAD_WEAK_DYLIB
	je	.wlc_dylib
	cmp	eax, LC_ID_DYLIB
	je	.wlc_dylib
	cmp	eax, LC_RPATH
	je	.wlc_rpath
	cmp	eax, LC_CODE_SIGNATURE
	je	.wlc_codesig
	cmp	eax, LC_BUILD_VERSION
	je	.wlc_build
	cmp	eax, LC_MAIN
	je	.wlc_main
	cmp	eax, LC_UNIXTHREAD
	je	.wlc_unixthread
	jmp	.wlc_next
.wlc_segment:
	inc	qword [segment_count]
	jmp	.wlc_next
.wlc_dylib:
	call	store_dylib_command
	jmp	.wlc_next
.wlc_rpath:
	call	store_rpath_command
	jmp	.wlc_next
.wlc_codesig:
	mov	qword [has_code_signature], 1
	jmp	.wlc_next
.wlc_build:
	cmp	qword [cmd_size], 24
	jb	.wlc_err
	mov	qword [has_build_version], 1
	mov	eax, dword [file_buf + r12 + BUILD_MINOS_OFF]
	mov	[minos_value], rax
	mov	eax, dword [file_buf + r12 + BUILD_SDK_OFF]
	mov	[sdk_value], rax
	jmp	.wlc_next
.wlc_main:
	cmp	qword [cmd_size], 24
	jb	.wlc_err
	mov	qword [has_entry], 1
	mov	rax, qword [file_buf + r12 + MAIN_ENTRYOFF_OFF]
	mov	[entry_value], rax
	jmp	.wlc_next
.wlc_unixthread:
	cmp	qword [cmd_size], UNIXTHREAD_RIP_OFF + 8
	jb	.wlc_err
	mov	qword [has_entry], 1
	mov	rax, qword [file_buf + r12 + UNIXTHREAD_RIP_OFF]
	mov	[entry_value], rax
.wlc_next:
	add	r12, [cmd_size]
	inc	r13
	jmp	.wlc_loop
.wlc_ok:
	xor	rax, rax
	ret
.wlc_err:
	mov	rax, 2
	ret

store_dylib_command:
	cmp	qword [dylib_count], MAX_ITEMS
	jae	.sdc_done
	cmp	qword [cmd_size], 24
	jb	.sdc_done
	mov	eax, dword [file_buf + r12 + DYLIB_NAME_OFF_OFF]
	mov	r10, rax
	cmp	r10, [cmd_size]
	jae	.sdc_done
	mov	rax, r12
	add	rax, r10
	lea	rax, [file_buf + rax]
	mov	rcx, [dylib_count]
	mov	[dylib_ptrs + rcx * 8], rax
	inc	qword [dylib_count]
.sdc_done:
	ret

store_rpath_command:
	cmp	qword [rpath_count], MAX_ITEMS
	jae	.src_done
	cmp	qword [cmd_size], 12
	jb	.src_done
	mov	eax, dword [file_buf + r12 + RPATH_PATH_OFF_OFF]
	mov	r10, rax
	cmp	r10, [cmd_size]
	jae	.src_done
	mov	rax, r12
	add	rax, r10
	lea	rax, [file_buf + rax]
	mov	rcx, [rpath_count]
	mov	[rpath_ptrs + rcx * 8], rax
	inc	qword [rpath_count]
.src_done:
	ret

print_human:
	lea	rdi, [path_label]
	call	print_cstr
	mov	rdi, [path_ptr]
	call	print_cstr
	call	print_nl
	lea	rdi, [format_label]
	call	print_cstr
	lea	rdi, [format_macho64]
	call	print_cstr
	call	call_print_arch_type
	lea	rdi, [minos_label]
	call	print_cstr
	cmp	qword [has_build_version], 0
	je	.ph_unknown_minos
	mov	rax, [minos_value]
	call	print_version
	jmp	.ph_after_minos
.ph_unknown_minos:
	lea	rdi, [unknown_msg]
	call	print_cstr
.ph_after_minos:
	call	print_nl
	lea	rdi, [entry_label]
	call	print_cstr
	cmp	qword [has_entry], 0
	je	.ph_unknown_entry
	mov	rax, [entry_value]
	call	print_hex64
	jmp	.ph_after_entry
.ph_unknown_entry:
	lea	rdi, [unknown_msg]
	call	print_cstr
.ph_after_entry:
	call	print_nl
	lea	rdi, [dylibs_label]
	call	print_cstr
	mov	rax, [dylib_count]
	call	print_none_or_count
	lea	rdi, [rpaths_label]
	call	print_cstr
	mov	rax, [rpath_count]
	call	print_none_or_count
	lea	rdi, [codesig_label]
	call	print_cstr
	call	print_codesig_word
	call	print_nl
	ret

call_print_arch_type:
	call	print_nl
	lea	rdi, [arch_label]
	call	print_cstr
	call	print_cpu_name
	call	print_nl
	lea	rdi, [type_label]
	call	print_cstr
	call	print_filetype_name
	call	print_nl
	ret

print_deps:
	lea	rdi, [dylibs_header]
	call	print_cstr
	mov	rdi, dylib_ptrs
	mov	rsi, [dylib_count]
	call	print_string_list
	lea	rdi, [rpaths_header]
	call	print_cstr
	mov	rdi, rpath_ptrs
	mov	rsi, [rpath_count]
	call	print_string_list
	xor	rax, rax
	ret

print_check:
	cmp	qword [has_code_signature], 0
	jne	.pc_codesig_ok
	lea	rdi, [warn_unsigned]
	call	print_cstr
	inc	qword [check_issue_count]
.pc_codesig_ok:
	cmp	qword [has_entry], 0
	jne	.pc_entry_ok
	lea	rdi, [warn_no_entry]
	call	print_cstr
	inc	qword [check_issue_count]
.pc_entry_ok:
	cmp	qword [check_issue_count], 0
	jne	.pc_done
	lea	rdi, [check_ok_msg]
	call	print_cstr
.pc_done:
	xor	rax, rax
	ret

print_json:
	lea	rdi, [json_open]
	call	print_cstr
	lea	rdi, [json_path]
	call	print_cstr
	mov	rdi, [path_ptr]
	call	print_cstr
	lea	rdi, [json_arch]
	call	print_cstr
	call	print_cpu_name
	lea	rdi, [json_type]
	call	print_cstr
	call	print_filetype_name
	lea	rdi, [json_minos]
	call	print_cstr
	cmp	qword [has_build_version], 0
	je	.pj_null_minos
	mov	al, '"'
	call	print_char
	mov	rax, [minos_value]
	call	print_version
	mov	al, '"'
	call	print_char
	jmp	.pj_after_minos
.pj_null_minos:
	lea	rdi, [json_null]
	call	print_cstr
.pj_after_minos:
	lea	rdi, [json_entry]
	call	print_cstr
	cmp	qword [has_entry], 0
	je	.pj_null_entry
	mov	al, '"'
	call	print_char
	mov	rax, [entry_value]
	call	print_hex64
	mov	al, '"'
	call	print_char
	jmp	.pj_after_entry
.pj_null_entry:
	lea	rdi, [json_null]
	call	print_cstr
.pj_after_entry:
	lea	rdi, [json_dylibs]
	call	print_cstr
	mov	rax, [dylib_count]
	call	print_int64
	lea	rdi, [json_rpaths]
	call	print_cstr
	mov	rax, [rpath_count]
	call	print_int64
	lea	rdi, [json_codesig]
	call	print_cstr
	cmp	qword [has_code_signature], 0
	je	.pj_false
	lea	rdi, [json_true]
	call	print_cstr
	jmp	.pj_close
.pj_false:
	lea	rdi, [json_false]
	call	print_cstr
.pj_close:
	lea	rdi, [json_close]
	call	print_cstr
	xor	rax, rax
	ret

print_none_or_count:
	test	rax, rax
	jz	.pnoc_none
	call	print_int_nl
	ret
.pnoc_none:
	lea	rdi, [none_msg]
	call	print_cstr
	call	print_nl
	ret

print_string_list:
	push	rbx
	push	r12
	mov	r12, rdi
	mov	rbx, rsi
	test	rbx, rbx
	jz	.psl_none
	xor	r10, r10
.psl_loop:
	cmp	r10, rbx
	jae	.psl_done
	lea	rdi, [list_prefix]
	call	print_cstr
	mov	rdi, [r12 + r10 * 8]
	call	print_cstr
	call	print_nl
	inc	r10
	jmp	.psl_loop
.psl_none:
	lea	rdi, [list_none]
	call	print_cstr
.psl_done:
	pop	r12
	pop	rbx
	ret

print_cpu_name:
	mov	rax, [cpu_type]
	cmp	rax, CPU_TYPE_X86_64
	je	.pcn_x86
	cmp	rax, CPU_TYPE_ARM64
	je	.pcn_arm
	lea	rdi, [unknown_msg]
	jmp	print_cstr
.pcn_x86:
	lea	rdi, [x86_msg]
	jmp	print_cstr
.pcn_arm:
	lea	rdi, [arm_msg]
	jmp	print_cstr

print_filetype_name:
	mov	rax, [file_type]
	cmp	rax, MH_EXECUTE
	je	.pft_exec
	cmp	rax, MH_DYLIB
	je	.pft_dylib
	cmp	rax, MH_BUNDLE
	je	.pft_bundle
	cmp	rax, MH_OBJECT
	je	.pft_object
	lea	rdi, [unknown_msg]
	jmp	print_cstr
.pft_exec:
	lea	rdi, [execute_msg]
	jmp	print_cstr
.pft_dylib:
	lea	rdi, [dylib_msg]
	jmp	print_cstr
.pft_bundle:
	lea	rdi, [bundle_msg]
	jmp	print_cstr
.pft_object:
	lea	rdi, [object_msg]
	jmp	print_cstr

print_codesig_word:
	cmp	qword [has_code_signature], 0
	je	.pcw_absent
	lea	rdi, [present_msg]
	jmp	print_cstr
.pcw_absent:
	lea	rdi, [absent_msg]
	jmp	print_cstr

; rax = packed version A.B.C as nibbles 16.8.8
print_version:
	push	rax
	shr	rax, 16
	call	print_int64
	mov	al, '.'
	call	print_char
	pop	rax
	push	rax
	shr	rax, 8
	and	rax, 0ffh
	call	print_int64
	pop	rax
	and	rax, 0ffh
	test	rax, rax
	jz	.pv_done
	push	rax
	mov	al, '.'
	call	print_char
	pop	rax
	call	print_int64
.pv_done:
	ret

; rax = uint64
print_hex64:
	push	rbx
	push	r12
	mov	r12, rax
	lea	rdi, [hex_prefix]
	call	print_cstr
	mov	rbx, 60
.ph_loop:
	mov	rax, r12
	mov	rcx, rbx
	shr	rax, cl
	and	rax, 0fh
	mov	al, [hex_digits + rax]
	call	print_char
	sub	rbx, 4
	jns	.ph_loop
	pop	r12
	pop	rbx
	ret

print_nl:
	mov	al, 10
	jmp	print_char

write_stderr:
	mov	rdx, rsi
	mov	rsi, rdi
	mov	rdi, STDERR
	mov	rax, SYS_write
	syscall
	ret

write_char_stderr:
	mov	[stderr_char], al
	lea	rdi, [stderr_char]
	mov	rsi, 1
	jmp	write_stderr

write_cstr_stderr:
	push	rdi
	call	str_len
	mov	rsi, rax
	pop	rdi
	jmp	write_stderr

segment readable writeable

MAX_ITEMS = 64

argc dq ?
arg_index dq ?
mode dq ?
path_ptr dq ?
read_path dq ?
file_fd dq ?
file_size dq ?

cpu_type dq ?
file_type dq ?
ncmds dq ?
sizeofcmds dq ?
cmd_value dq ?
cmd_size dq ?

dylib_count dq ?
rpath_count dq ?
segment_count dq ?
has_code_signature dq ?
has_build_version dq ?
has_entry dq ?
entry_value dq ?
minos_value dq ?
sdk_value dq ?
check_issue_count dq ?

dylib_ptrs rq MAX_ITEMS
rpath_ptrs rq MAX_ITEMS

opt_json db '--json', 0
opt_deps db '--deps', 0
opt_check db '--check', 0

usage_msg db 'usage: machodoctor [--json|--deps|--check] <mach-o>', 10
usage_msg_len = $ - usage_msg
open_err_msg db 'machodoctor: cannot open: '
open_err_msg_len = $ - open_err_msg
read_err_msg db 'machodoctor: cannot read file', 10
read_err_msg_len = $ - read_err_msg
parse_err_msg db 'machodoctor: not a supported Mach-O 64-bit file', 10
parse_err_msg_len = $ - parse_err_msg
too_large_msg db 'machodoctor: file too large for v1 buffer', 10
too_large_msg_len = $ - too_large_msg

path_label db 'path: ', 0
format_label db 'format: ', 0
arch_label db 'arch: ', 0
type_label db 'type: ', 0
minos_label db 'min macOS: ', 0
entry_label db 'entry: ', 0
dylibs_label db 'dylibs: ', 0
rpaths_label db 'rpaths: ', 0
codesig_label db 'code signature: ', 0

format_macho64 db 'Mach-O 64-bit', 0
x86_msg db 'x86_64', 0
arm_msg db 'arm64', 0
execute_msg db 'executable', 0
dylib_msg db 'dylib', 0
bundle_msg db 'bundle', 0
object_msg db 'object', 0
unknown_msg db 'unknown', 0
none_msg db 'none', 0
present_msg db 'present', 0
absent_msg db 'absent', 0

dylibs_header db 'dylibs:', 10, 0
rpaths_header db 'rpaths:', 10, 0
list_prefix db '  ', 0
list_none db '  none', 10, 0
warn_unsigned db 'WARN unsigned executable', 10, 0
warn_no_entry db 'FAIL no entrypoint load command', 10, 0
check_ok_msg db 'OK no obvious Mach-O load-command issues', 10, 0

json_open db '{', 10, 0
json_path db '  "path": "', 0
json_arch db '",', 10, '  "format": "Mach-O 64-bit",', 10, '  "arch": "', 0
json_type db '",', 10, '  "type": "', 0
json_minos db '",', 10, '  "min_macos": ', 0
json_entry db ',', 10, '  "entry": ', 0
json_dylibs db ',', 10, '  "dylib_count": ', 0
json_rpaths db ',', 10, '  "rpath_count": ', 0
json_codesig db ',', 10, '  "code_signature": ', 0
json_close db 10, '}', 10, 0
json_null db 'null', 0
json_true db 'true', 0
json_false db 'false', 0

hex_prefix db '0x', 0
hex_digits db '0123456789abcdef'
stderr_char rb 1

file_buf rb FILE_BUF_SIZE

include "fasm/core/runtime_bss.inc"
runtime_print_bss
