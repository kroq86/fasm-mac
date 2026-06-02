; shipcheck: local release artifact QA.
;
; Usage: shipcheck Formula/foo.rb dist/foo-0.1.0-macos-x86_64.tar.gz path/to/foo

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/file.inc"
include "fasm/core/formula.inc"
include "fasm/core/sha256.inc"
include "fasm/core/macho.inc"

FORMULA_BUF_SIZE equ 65536
BINARY_BUF_SIZE equ 4194304
READ_BUF_SIZE equ 65536
FIELD_SIZE equ 512
SHA_HEX_SIZE equ 65

entry start

start:
	mov	rbx, rsp
	cmp	qword [rbx], 4
	jne	usage
	mov	rax, [rbx + 16]
	mov	[formula_path], rax
	mov	rax, [rbx + 24]
	mov	[pkg_path], rax
	mov	rax, [rbx + 32]
	mov	[binary_path], rax
	call	run_checks
	cmp	rax, 2
	je	exit_error
	cmp	qword [issue_count], 0
	jne	exit_failed
	exit	EXIT_SUCCESS
exit_failed:
	exit	EXIT_FAILURE
exit_error:
	exit	2

usage:
	lea	rdi, [usage_msg]
	mov	rsi, usage_msg_len
	call	write_stderr
	exit	2

run_checks:
	mov	qword [issue_count], 0
	call	read_formula
	cmp	rax, 2
	je	.rc_err
	call	parse_formula
	cmp	rax, 2
	je	.rc_err
	call	compute_pkg_sha
	cmp	rax, 2
	je	.rc_err
	call	read_binary
	cmp	rax, 2
	je	.rc_err
	call	derive_names
	call	check_formula_values
	call	check_macho
	call	print_report
	xor	rax, rax
	ret
.rc_err:
	mov	rax, 2
	ret

read_formula:
	mov	rdi, [formula_path]
	lea	rsi, [formula_buf]
	mov	rdx, FORMULA_BUF_SIZE - 1
	lea	rcx, [formula_size]
	call	file_read_all_fixed
	cmp	rax, FILE_OK
	jne	.rf_err
	mov	rax, [formula_size]
	mov	byte [formula_buf + rax], 0
	xor	rax, rax
	ret
.rf_err:
	lea	rdi, [err_formula]
	call	print_cstr_stderr
	mov	rdi, [formula_path]
	call	print_cstr_stderr
	call	print_nl_stderr
	mov	rax, 2
	ret

parse_formula:
	lea	rdi, [formula_buf]
	mov	rsi, [formula_size]
	lea	rdx, [url_key]
	mov	rcx, url_key_len
	lea	r8, [formula_url]
	mov	r9, FIELD_SIZE
	call	formula_extract_quoted
	test	rax, rax
	jnz	.pf_bad_url
	lea	rdi, [formula_buf]
	mov	rsi, [formula_size]
	lea	rdx, [sha_key]
	mov	rcx, sha_key_len
	lea	r8, [formula_sha]
	mov	r9, SHA_HEX_SIZE
	call	formula_extract_quoted
	test	rax, rax
	jnz	.pf_bad_sha
	lea	rdi, [formula_buf]
	mov	rsi, [formula_size]
	lea	rdx, [version_key]
	mov	rcx, version_key_len
	lea	r8, [formula_version]
	mov	r9, FIELD_SIZE
	call	formula_extract_quoted
	test	rax, rax
	jnz	.pf_bad_version
	lea	rdi, [formula_buf]
	mov	rsi, [formula_size]
	lea	rdx, [install_key]
	mov	rcx, install_key_len
	lea	r8, [install_name]
	mov	r9, FIELD_SIZE
	call	formula_extract_quoted
	test	rax, rax
	jnz	.pf_bad_install
	xor	rax, rax
	ret
.pf_bad_url:
	lea	rdi, [err_url]
	jmp	.pf_err
.pf_bad_sha:
	lea	rdi, [err_sha]
	jmp	.pf_err
.pf_bad_version:
	lea	rdi, [err_version]
	jmp	.pf_err
.pf_bad_install:
	lea	rdi, [err_install]
.pf_err:
	call	print_cstr_stderr
	mov	rax, 2
	ret

compute_pkg_sha:
	lea	rdi, [sha_ctx]
	call	sha256_init
	mov	rdi, [pkg_path]
	call	file_open_read
	cmp	rax, 0
	jl	.cps_err
	mov	[pkg_fd], rax
.cps_loop:
	mov	rdi, [pkg_fd]
	lea	rsi, [read_buf]
	mov	rdx, READ_BUF_SIZE
	call	file_read_chunk
	cmp	rax, 0
	jl	.cps_read_err
	je	.cps_done
	lea	rdi, [sha_ctx]
	lea	rsi, [read_buf]
	mov	rdx, rax
	call	sha256_update
	jmp	.cps_loop
.cps_done:
	mov	rdi, [pkg_fd]
	call	file_close
	lea	rdi, [sha_ctx]
	lea	rsi, [sha_digest]
	call	sha256_final
	lea	rdi, [actual_sha]
	lea	rsi, [sha_digest]
	call	bytes32_to_hex
	xor	rax, rax
	ret
.cps_read_err:
	mov	rdi, [pkg_fd]
	call	file_close
.cps_err:
	lea	rdi, [err_package]
	call	print_cstr_stderr
	mov	rdi, [pkg_path]
	call	print_cstr_stderr
	call	print_nl_stderr
	mov	rax, 2
	ret

read_binary:
	mov	rdi, [binary_path]
	lea	rsi, [binary_buf]
	mov	rdx, BINARY_BUF_SIZE
	lea	rcx, [binary_size]
	call	file_read_all_fixed
	cmp	rax, FILE_OK
	jne	.rb_err
	xor	rax, rax
	ret
.rb_err:
	lea	rdi, [err_binary]
	call	print_cstr_stderr
	mov	rdi, [binary_path]
	call	print_cstr_stderr
	call	print_nl_stderr
	mov	rax, 2
	ret

derive_names:
	mov	rdi, [pkg_path]
	call	formula_basename_ptr
	mov	[pkg_basename], rax
	lea	rdi, [formula_url]
	call	formula_basename_ptr
	mov	[url_basename], rax
	mov	rdi, [binary_path]
	call	formula_basename_ptr
	mov	[binary_basename], rax
	lea	rdi, [expected_pkg_name]
	lea	rsi, [install_name]
	lea	rdx, [formula_version]
	call	formula_build_pkg_name
	ret

check_formula_values:
	lea	rdi, [formula_sha]
	lea	rsi, [actual_sha]
	call	str_eq
	test	rax, rax
	jnz	.cfv_sha_ok
	lea	rdi, [fail_sha]
	call	add_issue
.cfv_sha_ok:
	mov	rdi, [pkg_basename]
	lea	rsi, [expected_pkg_name]
	call	str_eq
	test	rax, rax
	jnz	.cfv_pkg_ok
	lea	rdi, [fail_pkg_name]
	call	add_issue
.cfv_pkg_ok:
	mov	rdi, [url_basename]
	mov	rsi, [pkg_basename]
	call	str_eq
	test	rax, rax
	jnz	.cfv_url_ok
	lea	rdi, [fail_url_name]
	call	add_issue
.cfv_url_ok:
	mov	rdi, [binary_basename]
	lea	rsi, [install_name]
	call	str_eq
	test	rax, rax
	jnz	.cfv_bin_ok
	lea	rdi, [fail_bin_name]
	call	add_issue
.cfv_bin_ok:
	ret

check_macho:
	cmp	qword [binary_size], MACH_HEADER_64_SIZE
	jb	.cm_bad
	mov	eax, dword [binary_buf + MH_MAGIC_OFF]
	cmp	eax, MH_MAGIC_64
	jne	.cm_bad
	mov	eax, dword [binary_buf + MH_CPUTYPE_OFF]
	cmp	eax, CPU_TYPE_X86_64
	jne	.cm_bad_arch
	mov	eax, dword [binary_buf + MH_FILETYPE_OFF]
	cmp	eax, MH_EXECUTE
	jne	.cm_bad_type
	mov	qword [macho_ok], 1
	call	scan_macho_loads
	ret
.cm_bad:
	lea	rdi, [fail_macho]
	call	add_issue
	ret
.cm_bad_arch:
	lea	rdi, [fail_arch]
	call	add_issue
	ret
.cm_bad_type:
	lea	rdi, [fail_type]
	call	add_issue
	ret

scan_macho_loads:
	mov	qword [has_minos], 0
	mov	qword [has_entry], 0
	mov	eax, dword [binary_buf + MH_NCMDS_OFF]
	mov	[load_count], rax
	lea	rbx, [binary_buf + MACH_HEADER_64_SIZE]
	xor	r12, r12
.sml_loop:
	cmp	r12, [load_count]
	jae	.sml_done
	mov	rax, rbx
	sub	rax, binary_buf
	add	rax, LC_CMDSIZE_OFF + 4
	cmp	rax, [binary_size]
	ja	.sml_done
	mov	eax, dword [rbx + LC_CMD_OFF]
	mov	edx, dword [rbx + LC_CMDSIZE_OFF]
	cmp	edx, 8
	jb	.sml_done
	cmp	eax, LC_BUILD_VERSION
	je	.sml_build
	cmp	eax, LC_MAIN
	je	.sml_main
	jmp	.sml_next
.sml_build:
	mov	eax, dword [rbx + BUILD_MINOS_OFF]
	mov	[minos_raw], rax
	mov	qword [has_minos], 1
	jmp	.sml_next
.sml_main:
	mov	qword [has_entry], 1
	jmp	.sml_next
.sml_next:
	add	rbx, rdx
	inc	r12
	jmp	.sml_loop
.sml_done:
	ret

print_report:
	lea	rdi, [label_formula]
	call	print_cstr
	mov	rdi, [formula_path]
	call	print_cstr
	call	print_nl
	lea	rdi, [label_package]
	call	print_cstr
	mov	rdi, [pkg_path]
	call	print_cstr
	call	print_nl
	lea	rdi, [label_binary]
	call	print_cstr
	mov	rdi, [binary_path]
	call	print_cstr
	call	print_nl
	lea	rdi, [label_version]
	call	print_cstr
	lea	rdi, [formula_version]
	call	print_cstr
	call	print_nl
	lea	rdi, [label_urlbase]
	call	print_cstr
	mov	rdi, [url_basename]
	call	print_cstr
	call	print_nl
	lea	rdi, [label_pkgbase]
	call	print_cstr
	mov	rdi, [pkg_basename]
	call	print_cstr
	call	print_nl
	lea	rdi, [label_sha_exp]
	call	print_cstr
	lea	rdi, [formula_sha]
	call	print_cstr
	call	print_nl
	lea	rdi, [label_sha_act]
	call	print_cstr
	lea	rdi, [actual_sha]
	call	print_cstr
	call	print_nl
	lea	rdi, [label_macho]
	call	print_cstr
	cmp	qword [macho_ok], 1
	jne	.pr_bad_macho
	lea	rdi, [macho_ok_msg]
	call	print_cstr
	cmp	qword [has_minos], 1
	jne	.pr_macho_nl
	lea	rdi, [minos_label]
	call	print_cstr
	mov	eax, dword [minos_raw]
	shr	eax, 16
	call	print_int64
	mov	al, '.'
	call	print_char
	mov	eax, dword [minos_raw]
	shr	eax, 8
	and	eax, 0ffh
	call	print_int64
.pr_macho_nl:
	call	print_nl
	jmp	.pr_status
.pr_bad_macho:
	lea	rdi, [macho_bad_msg]
	call	print_cstr
	call	print_nl
.pr_status:
	lea	rdi, [label_status]
	call	print_cstr
	cmp	qword [issue_count], 0
	jne	.pr_fail
	lea	rdi, [status_pass]
	call	print_cstr
	call	print_nl
	ret
.pr_fail:
	lea	rdi, [status_fail]
	call	print_cstr
	mov	rax, [issue_count]
	call	print_int_nl
	ret

add_issue:
	inc	qword [issue_count]
	lea	rsi, [issue_prefix]
	push	rdi
	mov	rdi, rsi
	call	print_cstr_stderr
	pop	rdi
	call	print_cstr_stderr
	call	print_nl_stderr
	ret

; rdi = out hex buffer, rsi = 32-byte digest
bytes32_to_hex:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	xor	rbx, rbx
.bth_loop:
	cmp	rbx, 32
	jae	.bth_done
	movzx	eax, byte [r13 + rbx]
	mov	cl, al
	shr	al, 4
	call	nibble_to_hex
	mov	[r12 + rbx * 2], al
	mov	al, cl
	and	al, 0fh
	call	nibble_to_hex
	mov	[r12 + rbx * 2 + 1], al
	inc	rbx
	jmp	.bth_loop
.bth_done:
	mov	byte [r12 + 64], 0
	pop	r13
	pop	r12
	pop	rbx
	ret

nibble_to_hex:
	cmp	al, 10
	jb	.nth_digit
	add	al, 'a' - 10
	ret
.nth_digit:
	add	al, '0'
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

print_cstr_stderr:
	push	rdi
	xor	rax, rax
.pcs_len:
	cmp	byte [rdi + rax], 0
	je	.pcs_out
	inc	rax
	jmp	.pcs_len
.pcs_out:
	mov	rsi, rax
	pop	rdi
	jmp	write_stderr

print_nl_stderr:
	lea	rdi, [nl_byte]
	mov	rsi, 1
	jmp	write_stderr

usage_msg db 'usage: shipcheck Formula/foo.rb dist/foo-0.1.0-macos-x86_64.tar.gz path/to/foo', 10
usage_msg_len = $ - usage_msg

url_key db 'url "'
url_key_len = $ - url_key
sha_key db 'sha256 "'
sha_key_len = $ - sha_key
version_key db 'version "'
version_key_len = $ - version_key
install_key db 'bin.install "'
install_key_len = $ - install_key

label_formula db 'formula: ', 0
label_package db 'package: ', 0
label_binary db 'binary: ', 0
label_version db 'version: ', 0
label_urlbase db 'url basename: ', 0
label_pkgbase db 'package basename: ', 0
label_sha_exp db 'sha256 expected: ', 0
label_sha_act db 'sha256 actual: ', 0
label_macho db 'mach-o: ', 0
label_status db 'status: ', 0
minos_label db ' min macOS ', 0
macho_ok_msg db 'x86_64 executable', 0
macho_bad_msg db 'invalid', 0
status_pass db 'pass', 0
status_fail db 'fail issues=', 0
issue_prefix db 'FAIL ', 0
fail_sha db 'sha256 mismatch', 0
fail_pkg_name db 'package basename does not match <name>-<version>-macos-x86_64.tar.gz', 0
fail_url_name db 'formula url basename does not match package basename', 0
fail_bin_name db 'binary basename does not match bin.install target', 0
fail_macho db 'binary is not a supported Mach-O 64-bit executable', 0
fail_arch db 'binary is not x86_64', 0
fail_type db 'binary is not executable filetype', 0
err_formula db 'shipcheck: cannot read formula: ', 0
err_package db 'shipcheck: cannot read package: ', 0
err_binary db 'shipcheck: cannot read binary: ', 0
err_url db 'shipcheck: formula url not found', 10, 0
err_sha db 'shipcheck: formula sha256 not found', 10, 0
err_version db 'shipcheck: formula version not found', 10, 0
err_install db 'shipcheck: formula bin.install not found', 10, 0
nl_byte db 10

segment readable writeable

formula_path dq ?
pkg_path dq ?
binary_path dq ?
formula_size dq ?
binary_size dq ?
pkg_fd dq ?
issue_count dq ?
pkg_basename dq ?
url_basename dq ?
binary_basename dq ?
macho_ok dq ?
has_minos dq ?
has_entry dq ?
minos_raw dq ?
load_count dq ?

formula_url rb FIELD_SIZE
formula_sha rb SHA_HEX_SIZE
formula_version rb FIELD_SIZE
install_name rb FIELD_SIZE
expected_pkg_name rb FIELD_SIZE
actual_sha rb SHA_HEX_SIZE
sha_digest rb 32
sha_ctx rb SHA256_CTX_SIZE
read_buf rb READ_BUF_SIZE
formula_buf rb FORMULA_BUF_SIZE
binary_buf rb BINARY_BUF_SIZE

include "fasm/core/runtime_bss.inc"
runtime_print_bss
