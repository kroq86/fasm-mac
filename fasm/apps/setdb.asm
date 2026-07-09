; setdb: tiny pure set-theoretic database CLI.
;
; Usage:
;   setdb new universe.db
;   setdb add universe.db set atom...
;   setdb remove universe.db set atom
;   setdb relation universe.db rel left right
;   setdb unrelation universe.db rel left right
;   setdb members universe.db set
;   setdb member universe.db set atom
;   setdb union|intersect|diff universe.db a b
;   setdb subset universe.db a b
;   setdb select universe.db rel first|second atom
;   setdb join universe.db a b
;   setdb domain universe.db rel
;   setdb range universe.db rel
;   setdb inverse universe.db rel
;   setdb rdiff universe.db a b
;   setdb runion universe.db a b
;   setdb rintersect universe.db a b
;   setdb transitive-closure universe.db rel

format ELF64 executable 3
include "fasm/core/platform.inc"

SETDB_PATH_MAX equ 4096
SETDB_PAYLOAD_MAX equ 1024
SETDB_TEMP_ARENA_SIZE equ 65536
LOAD_READ_BUF_SIZE equ 8192
LOAD_LINE_BUF_SIZE equ 1024

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/mem.inc"
include "fasm/core/arena.inc"
include "fasm/core/log_segment.inc"
include "fasm/core/file.inc"
include "fasm/core/scanner.inc"
include "fasm/core/setdb.inc"

entry start

start:
	lea	rdi, [setdb_temp_arena]
	lea	rsi, [setdb_temp_storage]
	mov	rdx, SETDB_TEMP_ARENA_SIZE
	call	arena_init
	lea	rax, [setdb_temp_arena]
	mov	[setdb_tmp_arena], rax
	mov	[argv_base], rsp
	mov	rbx, rsp
	mov	rax, [rbx]
	mov	[argc], rax
	cmp	rax, 1
	je	usage
	cmp	rax, 2
	jne	.have_command
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_help]
	call	str_eq
	test	rax, rax
	jnz	help_run
	mov	rdi, [rbx + 16]
	lea	rsi, [opt_help]
	call	str_eq
	test	rax, rax
	jnz	help_run
	mov	rdi, [rbx + 16]
	lea	rsi, [opt_h]
	call	str_eq
	test	rax, rax
	jnz	help_run
	jmp	usage
.have_command:
	cmp	rax, 3
	jb	usage
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_help]
	call	str_eq
	test	rax, rax
	jnz	help_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_new]
	call	str_eq
	test	rax, rax
	jnz	cmd_new_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_sets]
	call	str_eq
	test	rax, rax
	jnz	cmd_sets_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_relations]
	call	str_eq
	test	rax, rax
	jnz	cmd_relations_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_dump]
	call	str_eq
	test	rax, rax
	jnz	cmd_dump_run
	cmp	qword [argc], 4
	jb	usage
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_add]
	call	str_eq
	test	rax, rax
	jnz	cmd_add_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_remove]
	call	str_eq
	test	rax, rax
	jnz	cmd_remove_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_relation]
	call	str_eq
	test	rax, rax
	jnz	cmd_relation_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_unrelation]
	call	str_eq
	test	rax, rax
	jnz	cmd_unrelation_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_members]
	call	str_eq
	test	rax, rax
	jnz	cmd_members_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_member]
	call	str_eq
	test	rax, rax
	jnz	cmd_member_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_union]
	call	str_eq
	test	rax, rax
	jnz	cmd_union_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_intersect]
	call	str_eq
	test	rax, rax
	jnz	cmd_intersect_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_diff]
	call	str_eq
	test	rax, rax
	jnz	cmd_diff_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_subset]
	call	str_eq
	test	rax, rax
	jnz	cmd_subset_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_select]
	call	str_eq
	test	rax, rax
	jnz	cmd_select_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_join]
	call	str_eq
	test	rax, rax
	jnz	cmd_join_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_domain]
	call	str_eq
	test	rax, rax
	jnz	cmd_domain_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_range]
	call	str_eq
	test	rax, rax
	jnz	cmd_range_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_inverse]
	call	str_eq
	test	rax, rax
	jnz	cmd_inverse_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_rdiff]
	call	str_eq
	test	rax, rax
	jnz	cmd_rdiff_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_runion]
	call	str_eq
	test	rax, rax
	jnz	cmd_runion_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_rintersect]
	call	str_eq
	test	rax, rax
	jnz	cmd_rintersect_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_transitive_closure]
	call	str_eq
	test	rax, rax
	jnz	cmd_transitive_closure_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_contains]
	call	str_eq
	test	rax, rax
	jnz	cmd_contains_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_pairs]
	call	str_eq
	test	rax, rax
	jnz	cmd_pairs_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_tag]
	call	str_eq
	test	rax, rax
	jnz	cmd_tag_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_files]
	call	str_eq
	test	rax, rax
	jnz	cmd_files_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_tags]
	call	str_eq
	test	rax, rax
	jnz	cmd_tags_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_store_domain]
	call	str_eq
	test	rax, rax
	jnz	cmd_store_domain_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_store_range]
	call	str_eq
	test	rax, rax
	jnz	cmd_store_range_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_store_inverse]
	call	str_eq
	test	rax, rax
	jnz	cmd_store_inverse_run
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_load]
	call	str_eq
	test	rax, rax
	jnz	cmd_load_run
	jmp	usage

help_run:
	lea	rdi, [help_msg]
	mov	rsi, help_msg_len
	call	write_stdout
	exit	EXIT_SUCCESS

cmd_new_run:
	cmp	qword [argc], 3
	jne	usage
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 24]
	call	db_new
	cmp	rax, 0
	jl	io_error
	exit	EXIT_SUCCESS

cmd_add_run:
	cmp	qword [argc], 5
	jb	usage
	call	load_db_from_argv
	mov	qword [arg_index], 4
.loop:
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	.done
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	mov	rsi, [rbx + 8 + rax * 8]
	call	setdb_set_add
	cmp	rax, 0
	jl	usage
	lea	rdi, [op_sadd]
	mov	rsi, [rbx + 32]
	mov	rax, [arg_index]
	mov	rdx, [rbx + 8 + rax * 8]
	xor	rcx, rcx
	call	append3
	inc	qword [arg_index]
	jmp	.loop
.done:
	exit	EXIT_SUCCESS

cmd_remove_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	mov	rsi, [rbx + 40]
	call	setdb_set_remove
	cmp	rax, 0
	jl	usage
	lea	rdi, [op_srem]
	mov	rsi, [rbx + 32]
	mov	rdx, [rbx + 40]
	xor	rcx, rcx
	call	append3
	exit	EXIT_SUCCESS

cmd_relation_run:
	cmp	qword [argc], 6
	jne	usage
	call	load_db_from_argv
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	mov	rsi, [rbx + 40]
	mov	rdx, [rbx + 48]
	call	setdb_rel_add
	cmp	rax, 0
	jl	usage
	lea	rdi, [op_radd]
	mov	rsi, [rbx + 32]
	mov	rdx, [rbx + 40]
	mov	rcx, [rbx + 48]
	call	append4
	exit	EXIT_SUCCESS

cmd_unrelation_run:
	cmp	qword [argc], 6
	jne	usage
	call	load_db_from_argv
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	mov	rsi, [rbx + 40]
	mov	rdx, [rbx + 48]
	call	setdb_rel_remove
	cmp	rax, 0
	jl	usage
	lea	rdi, [op_rrem]
	mov	rsi, [rbx + 32]
	mov	rdx, [rbx + 40]
	mov	rcx, [rbx + 48]
	call	append4
	exit	EXIT_SUCCESS

cmd_members_run:
	cmp	qword [argc], 4
	jne	usage
	call	load_db_from_argv
	mov	rbx, [argv_base]
	call	setdb_tmp_clear
	mov	rdi, [rbx + 32]
	call	add_set_to_tmp
	call	print_tmp_atoms_sorted
	exit	EXIT_SUCCESS

cmd_member_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	call	setdb_intern
	test	rax, rax
	jz	usage
	mov	rdi, rax
	call	setdb_find_set
	cmp	rax, SETDB_MISSING
	je	.member_no
	mov	r12, rax
	mov	rdi, [rbx + 40]
	call	setdb_intern
	test	rax, rax
	jz	usage
	mov	rdi, r12
	mov	rsi, rax
	call	setdb_set_find_member
	cmp	rax, SETDB_MISSING
	je	.member_no
	lea	rdi, [true_msg]
	call	print_cstr
	exit	EXIT_SUCCESS
.member_no:
	lea	rdi, [false_msg]
	call	print_cstr
	exit	EXIT_SUCCESS

cmd_union_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	call	add_set_to_tmp
	mov	rdi, [rbx + 40]
	call	add_set_to_tmp
	call	print_tmp_atoms_sorted
	exit	EXIT_SUCCESS

cmd_intersect_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	mov	rsi, [rbx + 40]
	call	build_intersection
	call	print_tmp_atoms_sorted
	exit	EXIT_SUCCESS

cmd_diff_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	mov	rsi, [rbx + 40]
	call	build_diff
	call	print_tmp_atoms_sorted
	exit	EXIT_SUCCESS

cmd_subset_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	mov	rsi, [rbx + 40]
	call	is_subset
	test	rax, rax
	jz	.sub_no
	lea	rdi, [true_msg]
	call	print_cstr
	exit	EXIT_SUCCESS
.sub_no:
	lea	rdi, [false_msg]
	call	print_cstr
	exit	EXIT_SUCCESS

cmd_select_run:
	cmp	qword [argc], 6
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 40]
	lea	rsi, [kw_first]
	call	str_eq
	test	rax, rax
	jnz	.select_ok
	mov	rdi, [rbx + 40]
	lea	rsi, [kw_second]
	call	str_eq
	test	rax, rax
	jz	usage
.select_ok:
	mov	rdi, [rbx + 32]
	mov	rsi, [rbx + 40]
	mov	rdx, [rbx + 48]
	call	build_select
	call	print_tmp_atoms_sorted
	exit	EXIT_SUCCESS

cmd_join_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	mov	rsi, [rbx + 40]
	call	build_join
	call	print_tmp_pairs_sorted
	exit	EXIT_SUCCESS

cmd_domain_run:
	cmp	qword [argc], 4
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	call	build_domain
	call	print_tmp_atoms_sorted
	exit	EXIT_SUCCESS

cmd_range_run:
	cmp	qword [argc], 4
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	call	build_range
	call	print_tmp_atoms_sorted
	exit	EXIT_SUCCESS

cmd_inverse_run:
	cmp	qword [argc], 4
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	call	build_inverse
	call	print_tmp_pairs_sorted
	exit	EXIT_SUCCESS

cmd_rdiff_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	mov	rsi, [rbx + 40]
	call	build_rdiff
	call	print_tmp_pairs_sorted
	exit	EXIT_SUCCESS

cmd_runion_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	mov	rsi, [rbx + 40]
	call	build_runion
	call	print_tmp_pairs_sorted
	exit	EXIT_SUCCESS

cmd_rintersect_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	mov	rsi, [rbx + 40]
	call	build_rintersect
	call	print_tmp_pairs_sorted
	exit	EXIT_SUCCESS

cmd_transitive_closure_run:
	cmp	qword [argc], 4
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	call	build_transitive_closure
	cmp	rax, 0
	jl	io_error
	call	print_tmp_pairs_sorted
	exit	EXIT_SUCCESS

cmd_sets_run:
	cmp	qword [argc], 3
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	xor	rbx, rbx
.loop:
	cmp	rbx, [setdb_set_count]
	jae	.done
	mov	rdi, [setdb_set_names + rbx * 8]
	call	setdb_tmp_add_atom
	cmp	rax, 0
	jl	io_error
	inc	rbx
	jmp	.loop
.done:
	call	print_tmp_atoms_sorted
	exit	EXIT_SUCCESS

cmd_relations_run:
	cmp	qword [argc], 3
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	xor	rbx, rbx
.loop:
	cmp	rbx, [setdb_rel_count]
	jae	.done
	mov	rdi, [setdb_rel_names + rbx * 8]
	call	setdb_tmp_add_atom
	cmp	rax, 0
	jl	io_error
	inc	rbx
	jmp	.loop
.done:
	call	print_tmp_atoms_sorted
	exit	EXIT_SUCCESS

cmd_dump_run:
	cmp	qword [argc], 3
	jne	usage
	call	load_db_from_argv
	xor	rbx, rbx
.set_loop:
	cmp	rbx, [setdb_set_count]
	jae	.rel_loop_init
	mov	r12, rbx
	imul	r12, SETDB_MAX_SET_MEMBERS
	xor	r13, r13
.member_loop:
	cmp	r13, [setdb_set_counts + rbx * 8]
	jae	.set_next
	lea	rdi, [op_sadd]
	call	print_cstr
	mov	al, ' '
	call	print_char
	mov	rdi, [setdb_set_names + rbx * 8]
	call	print_cstr
	mov	al, ' '
	call	print_char
	mov	rax, r12
	add	rax, r13
	mov	rdi, [setdb_set_members + rax * 8]
	call	print_cstr
	mov	al, 10
	call	print_char
	inc	r13
	jmp	.member_loop
.set_next:
	inc	rbx
	jmp	.set_loop
.rel_loop_init:
	xor	rbx, rbx
.rel_loop:
	cmp	rbx, [setdb_rel_count]
	jae	.done
	mov	r12, rbx
	imul	r12, SETDB_MAX_REL_PAIRS
	xor	r13, r13
.pair_loop:
	cmp	r13, [setdb_rel_counts + rbx * 8]
	jae	.rel_next
	lea	rdi, [op_radd]
	call	print_cstr
	mov	al, ' '
	call	print_char
	mov	rdi, [setdb_rel_names + rbx * 8]
	call	print_cstr
	mov	al, ' '
	call	print_char
	mov	rax, r12
	add	rax, r13
	mov	rdi, [setdb_rel_left + rax * 8]
	call	print_cstr
	mov	al, ' '
	call	print_char
	mov	rax, r12
	add	rax, r13
	mov	rdi, [setdb_rel_right + rax * 8]
	call	print_cstr
	mov	al, 10
	call	print_char
	inc	r13
	jmp	.pair_loop
.rel_next:
	inc	rbx
	jmp	.rel_loop
.done:
	exit	EXIT_SUCCESS

cmd_contains_run:
	cmp	qword [argc], 4
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	call	setdb_intern
	test	rax, rax
	jz	.done
	mov	r13, rax
	xor	r12, r12
.loop:
	cmp	r12, [setdb_set_count]
	jae	.done
	mov	rdi, r12
	mov	rsi, r13
	call	setdb_set_find_member
	cmp	rax, SETDB_MISSING
	je	.next
	mov	rdi, [setdb_set_names + r12 * 8]
	call	setdb_tmp_add_atom
	cmp	rax, 0
	jl	io_error
.next:
	inc	r12
	jmp	.loop
.done:
	call	print_tmp_atoms_sorted
	exit	EXIT_SUCCESS

cmd_pairs_run:
	cmp	qword [argc], 4
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	call	add_rel_to_tmp
	call	print_tmp_pairs_sorted
	exit	EXIT_SUCCESS

cmd_tag_run:
	cmp	qword [argc], 5
	jb	usage
	call	load_db_from_argv
	mov	rbx, [argv_base]
	lea	rdi, [tag_set_files]
	mov	rsi, [rbx + 32]
	call	setdb_set_add
	cmp	rax, 0
	jl	usage
	lea	rdi, [op_sadd]
	lea	rsi, [tag_set_files]
	mov	rdx, [rbx + 32]
	xor	rcx, rcx
	call	append3
	mov	qword [arg_index], 4
.loop:
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	.done
	mov	rbx, [argv_base]
	mov	r12, [rbx + 8 + rax * 8]
	lea	rdi, [tag_set_tags]
	mov	rsi, r12
	call	setdb_set_add
	cmp	rax, 0
	jl	usage
	lea	rdi, [op_sadd]
	lea	rsi, [tag_set_tags]
	mov	rdx, r12
	xor	rcx, rcx
	call	append3
	mov	rbx, [argv_base]
	lea	rdi, [tag_rel_has_tag]
	mov	rsi, [rbx + 32]
	mov	rdx, r12
	call	setdb_rel_add
	cmp	rax, 0
	jl	usage
	lea	rdi, [op_radd]
	lea	rsi, [tag_rel_has_tag]
	mov	rbx, [argv_base]
	mov	rdx, [rbx + 32]
	mov	rcx, r12
	call	append4
	inc	qword [arg_index]
	jmp	.loop
.done:
	exit	EXIT_SUCCESS

cmd_files_run:
	cmp	qword [argc], 4
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	lea	rdi, [tag_rel_has_tag]
	lea	rsi, [kw_second]
	mov	rdx, [rbx + 32]
	call	build_select
	call	print_tmp_atoms_sorted
	exit	EXIT_SUCCESS

cmd_tags_run:
	cmp	qword [argc], 4
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	lea	rdi, [tag_rel_has_tag]
	lea	rsi, [kw_first]
	mov	rdx, [rbx + 32]
	call	build_select
	call	print_tmp_atoms_sorted
	exit	EXIT_SUCCESS

cmd_store_domain_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	call	build_domain
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 40]
	call	store_tmp_atoms_into_set
	exit	EXIT_SUCCESS

cmd_store_range_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	call	build_range
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 40]
	call	store_tmp_atoms_into_set
	exit	EXIT_SUCCESS

cmd_store_inverse_run:
	cmp	qword [argc], 5
	jne	usage
	call	load_db_from_argv
	call	setdb_tmp_clear
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	call	build_inverse
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 40]
	call	store_tmp_pairs_into_rel
	exit	EXIT_SUCCESS

; rdi = target set name
store_tmp_atoms_into_set:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, [setdb_tmp_head]
.loop:
	test	r13, r13
	jz	.done
	mov	rdi, r12
	mov	rsi, [r13 + SETDB_TMP_LEFT_OFF]
	call	setdb_set_add
	cmp	rax, 0
	jl	usage
	lea	rdi, [op_sadd]
	mov	rsi, r12
	mov	rdx, [r13 + SETDB_TMP_LEFT_OFF]
	xor	rcx, rcx
	call	append3
	mov	r13, [r13 + SETDB_TMP_NEXT_OFF]
	jmp	.loop
.done:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = target relation name
store_tmp_pairs_into_rel:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, [setdb_tmp_head]
.loop:
	test	r13, r13
	jz	.done
	mov	rdi, r12
	mov	rsi, [r13 + SETDB_TMP_LEFT_OFF]
	mov	rdx, [r13 + SETDB_TMP_RIGHT_OFF]
	call	setdb_rel_add
	cmp	rax, 0
	jl	usage
	lea	rdi, [op_radd]
	mov	rsi, r12
	mov	rdx, [r13 + SETDB_TMP_LEFT_OFF]
	mov	rcx, [r13 + SETDB_TMP_RIGHT_OFF]
	call	append4
	mov	r13, [r13 + SETDB_TMP_NEXT_OFF]
	jmp	.loop
.done:
	pop	r13
	pop	r12
	pop	rbx
	ret

; Facts file: one op per line, same wire format as an ops.log payload
; (SADD/SREM/RADD/RREM), '#'-prefixed and blank lines ignored. Each line is
; applied to memory first (apply_payload, reusing the exact replay parser)
; and only appended to ops.log if that succeeds, matching every other
; mutating command's validate-then-append order.
cmd_load_run:
	cmp	qword [argc], 4
	jne	usage
	call	load_db_from_argv
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 32]
	lea	rsi, [load_read_buf]
	mov	rdx, LOAD_READ_BUF_SIZE
	lea	rcx, [load_line_buf]
	mov	r8, LOAD_LINE_BUF_SIZE
	lea	r9, [load_line_cb]
	call	scanner_scan_file
	cmp	rax, SCANNER_OK
	jne	io_error
	exit	EXIT_SUCCESS

; rdi = line ptr (scanner's buffer, not null-terminated), rsi = line len,
; rdx = line number (unused)
; scanner_emit_line calls [scanner_callback] without saving r12/r13/r14 --
; scanner_scan_file's own byte loop keeps live state in those registers
; across the call. This callback must not touch them; stash rdi/rsi into
; memory immediately instead of caching them in registers.
load_line_cb:
	mov	[load_cb_line_ptr], rdi
	mov	[load_cb_line_len], rsi
	test	rsi, rsi
	jz	.done
	mov	al, [rdi]
	cmp	al, '#'
	je	.done
	mov	byte [rdi + rsi], 0
	lea	rdi, [load_line_copy]
	mov	rsi, [load_cb_line_ptr]
	mov	rdx, [load_cb_line_len]
	inc	rdx
	call	memcpy
	lea	rdi, [load_line_copy]
	call	apply_payload
	cmp	rax, 0
	jl	usage
	lea	rdi, [ops_log_path]
	lea	rsi, [ops_idx_path]
	mov	rdx, [load_cb_line_ptr]
	mov	rcx, [load_cb_line_len]
	call	log_segment_append
	cmp	rax, LOGSEG_ERR
	je	io_error
.done:
	ret

load_db_from_argv:
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 24]
	mov	[db_path], rdi
	call	build_db_paths
	cmp	rax, 0
	jl	io_error
	call	setdb_init
	call	replay_log
	cmp	rax, 0
	jl	io_error
	ret

db_new:
	push	r12
	mov	r12, rdi
	mov	[db_path], r12
	mov	rax, SYS_mkdir
	mov	rdi, r12
	mov	rsi, 493
	syscall
	call	build_db_paths
	cmp	rax, 0
	jl	.err
	lea	rdi, [ops_log_path]
	call	create_empty
	cmp	rax, 0
	jl	.err
	lea	rdi, [ops_idx_path]
	call	create_empty
	cmp	rax, 0
	jl	.err
	xor	rax, rax
	jmp	.out
.err:
	mov	rax, -1
.out:
	pop	r12
	ret

create_empty:
	open_file rdi, O_WRONLY or O_CREAT or O_TRUNC, 420
	jump_if_syscall_error .err
	mov	rdi, rax
	close_file rdi
	xor	rax, rax
	ret
.err:
	mov	rax, -1
	ret

build_db_paths:
	lea	rdi, [ops_log_path]
	mov	rsi, [db_path]
	call	path_copy
	cmp	rax, 0
	jl	.err
	lea	rdi, [ops_log_path]
	lea	rsi, [ops_log_suffix]
	call	path_append
	cmp	rax, 0
	jl	.err
	lea	rdi, [ops_idx_path]
	mov	rsi, [db_path]
	call	path_copy
	cmp	rax, 0
	jl	.err
	lea	rdi, [ops_idx_path]
	lea	rsi, [ops_idx_suffix]
	call	path_append
	ret
.err:
	mov	rax, -1
	ret

path_copy:
	push	rdi
	mov	rdi, rsi
	call	str_len
	cmp	rax, SETDB_PATH_MAX - 1
	ja	.err
	mov	rdx, rax
	inc	rdx
	pop	rdi
	call	memcpy
	xor	rax, rax
	ret
.err:
	pop	rdi
	mov	rax, -1
	ret

path_append:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	call	str_len
	mov	rbx, rax
	mov	rdi, r13
	call	str_len
	lea	rcx, [rbx + rax]
	cmp	rcx, SETDB_PATH_MAX - 1
	ja	.err
	lea	rdi, [r12 + rbx]
	mov	rsi, r13
	mov	rdx, rax
	inc	rdx
	call	memcpy
	xor	rax, rax
	jmp	.out
.err:
	mov	rax, -1
.out:
	pop	r13
	pop	r12
	pop	rbx
	ret

append3:
	mov	[payload_arg0], rdi
	mov	[payload_arg1], rsi
	mov	[payload_arg2], rdx
	mov	qword [payload_arg3], 0
	call	build_payload3
	jmp	append_payload

append4:
	mov	[payload_arg0], rdi
	mov	[payload_arg1], rsi
	mov	[payload_arg2], rdx
	mov	[payload_arg3], rcx
	call	build_payload4
	jmp	append_payload

append_payload:
	lea	rdi, [ops_log_path]
	lea	rsi, [ops_idx_path]
	lea	rdx, [payload_buf]
	mov	rcx, [payload_len]
	call	log_segment_append
	cmp	rax, LOGSEG_ERR
	je	io_error
	ret

build_payload3:
	call	payload_reset
	mov	rdi, [payload_arg0]
	call	payload_token
	mov	rdi, [payload_arg1]
	call	payload_token
	mov	rdi, [payload_arg2]
	call	payload_token_nl
	ret

build_payload4:
	call	payload_reset
	mov	rdi, [payload_arg0]
	call	payload_token
	mov	rdi, [payload_arg1]
	call	payload_token
	mov	rdi, [payload_arg2]
	call	payload_token
	mov	rdi, [payload_arg3]
	call	payload_token_nl
	ret

payload_reset:
	mov	qword [payload_len], 0
	ret

payload_token:
	push	rdi
	call	payload_cstr
	lea	rdi, [space_ch]
	call	payload_byte
	pop	rdi
	ret

payload_token_nl:
	call	payload_cstr
	lea	rdi, [nl_ch]
	call	payload_byte
	ret

payload_cstr:
	push	rbx
	push	r12
	mov	r12, rdi
	call	str_len
	mov	rbx, rax
	mov	rax, [payload_len]
	lea	rcx, [rax + rbx]
	cmp	rcx, SETDB_PAYLOAD_MAX
	jae	usage
	lea	rdi, [payload_buf + rax]
	mov	rsi, r12
	mov	rdx, rbx
	call	memcpy
	add	[payload_len], rbx
	pop	r12
	pop	rbx
	ret

payload_byte:
	mov	rax, [payload_len]
	cmp	rax, SETDB_PAYLOAD_MAX
	jae	usage
	mov	cl, [rdi]
	mov	[payload_buf + rax], cl
	inc	qword [payload_len]
	ret

replay_log:
	xor	r12, r12
.loop:
	lea	rdi, [ops_log_path]
	lea	rsi, [ops_idx_path]
	mov	rdx, r12
	lea	rcx, [replay_buf]
	mov	r8, SETDB_PAYLOAD_MAX - 1
	lea	r9, [replay_len]
	call	log_segment_read
	cmp	rax, LOGSEG_EOF
	je	.ok
	cmp	rax, LOGSEG_OK
	jne	.err
	mov	rax, [replay_len]
	mov	byte [replay_buf + rax], 0
	lea	rdi, [replay_buf]
	call	apply_payload
	cmp	rax, 0
	jl	.err
	inc	r12
	jmp	.loop
.ok:
	xor	rax, rax
	ret
.err:
	mov	rax, -1
	ret

apply_payload:
	mov	[parse_ptr], rdi
	call	next_token
	mov	[tok0], rax
	call	next_token
	mov	[tok1], rax
	call	next_token
	mov	[tok2], rax
	call	next_token
	mov	[tok3], rax
	mov	rdi, [tok0]
	lea	rsi, [op_sadd]
	call	str_eq
	test	rax, rax
	jnz	.ap_sadd
	mov	rdi, [tok0]
	lea	rsi, [op_srem]
	call	str_eq
	test	rax, rax
	jnz	.ap_srem
	mov	rdi, [tok0]
	lea	rsi, [op_radd]
	call	str_eq
	test	rax, rax
	jnz	.ap_radd
	mov	rdi, [tok0]
	lea	rsi, [op_rrem]
	call	str_eq
	test	rax, rax
	jnz	.ap_rrem
	jmp	.ap_err
.ap_sadd:
	cmp	qword [tok3], 0
	jne	.ap_err
	mov	rdi, [tok1]
	mov	rsi, [tok2]
	jmp	setdb_set_add
.ap_srem:
	cmp	qword [tok3], 0
	jne	.ap_err
	mov	rdi, [tok1]
	mov	rsi, [tok2]
	jmp	setdb_set_remove
.ap_radd:
	cmp	qword [tok3], 0
	je	.ap_err
	mov	rdi, [tok1]
	mov	rsi, [tok2]
	mov	rdx, [tok3]
	jmp	setdb_rel_add
.ap_rrem:
	cmp	qword [tok3], 0
	je	.ap_err
	mov	rdi, [tok1]
	mov	rsi, [tok2]
	mov	rdx, [tok3]
	jmp	setdb_rel_remove
.ap_err:
	mov	rax, -1
	ret

next_token:
	mov	rdi, [parse_ptr]
.skip:
	mov	al, [rdi]
	cmp	al, ' '
	je	.skip_next
	cmp	al, 10
	je	.none
	test	al, al
	jz	.none
	jmp	.start
.skip_next:
	inc	rdi
	jmp	.skip
.start:
	mov	rax, rdi
.scan:
	mov	cl, [rdi]
	cmp	cl, ' '
	je	.term
	cmp	cl, 10
	je	.term
	test	cl, cl
	jz	.done
	inc	rdi
	jmp	.scan
.term:
	mov	byte [rdi], 0
	inc	rdi
.done:
	mov	[parse_ptr], rdi
	ret
.none:
	mov	[parse_ptr], rdi
	xor	rax, rax
	ret

add_set_to_tmp:
	push	rbx
	push	r12
	push	r13
	call	setdb_intern
	test	rax, rax
	jz	.done
	mov	rdi, rax
	call	setdb_find_set
	cmp	rax, SETDB_MISSING
	je	.done
	mov	r12, rax
	xor	rbx, rbx
	mov	r13, r12
	imul	r13, SETDB_MAX_SET_MEMBERS
.loop:
	cmp	rbx, [setdb_set_counts + r12 * 8]
	jae	.done
	mov	rax, r13
	add	rax, rbx
	mov	rdi, [setdb_set_members + rax * 8]
	call	setdb_tmp_add_atom
	inc	rbx
	jmp	.loop
.done:
	pop	r13
	pop	r12
	pop	rbx
	ret

build_intersection:
	push	rbx
	push	r12
	push	r13
	push	r14
	call	find_two_sets
	cmp	rax, SETDB_MISSING
	je	.done
	mov	r12, rax
	mov	r13, rdx
	cmp	r13, SETDB_MISSING
	je	.done
	xor	rbx, rbx
	mov	r14, r12
	imul	r14, SETDB_MAX_SET_MEMBERS
.loop:
	cmp	rbx, [setdb_set_counts + r12 * 8]
	jae	.done
	mov	rax, r14
	add	rax, rbx
	mov	rsi, [setdb_set_members + rax * 8]
	mov	rdi, r13
	call	setdb_set_find_member
	cmp	rax, SETDB_MISSING
	je	.next
	mov	rax, r14
	add	rax, rbx
	mov	rdi, [setdb_set_members + rax * 8]
	call	setdb_tmp_add_atom
.next:
	inc	rbx
	jmp	.loop
.done:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

build_diff:
	push	rbx
	push	r12
	push	r13
	push	r14
	call	find_two_sets
	cmp	rax, SETDB_MISSING
	je	.done
	mov	r12, rax
	mov	r13, rdx
	xor	rbx, rbx
	mov	r14, r12
	imul	r14, SETDB_MAX_SET_MEMBERS
.loop:
	cmp	rbx, [setdb_set_counts + r12 * 8]
	jae	.done
	cmp	r13, SETDB_MISSING
	je	.add
	mov	rax, r14
	add	rax, rbx
	mov	rsi, [setdb_set_members + rax * 8]
	mov	rdi, r13
	call	setdb_set_find_member
	cmp	rax, SETDB_MISSING
	jne	.next
.add:
	mov	rax, r14
	add	rax, rbx
	mov	rdi, [setdb_set_members + rax * 8]
	call	setdb_tmp_add_atom
.next:
	inc	rbx
	jmp	.loop
.done:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

find_two_sets:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	call	setdb_intern
	test	rax, rax
	jz	.missing
	mov	rdi, rax
	call	setdb_find_set
	cmp	rax, SETDB_MISSING
	je	.missing
	mov	rdx, rax
	mov	rdi, r13
	call	setdb_intern
	test	rax, rax
	jz	.missing
	mov	rdi, rax
	call	setdb_find_set
	cmp	rax, SETDB_MISSING
	je	.second_missing
	xchg	rax, rdx
	jmp	.out
.second_missing:
	mov	rax, rdx
	mov	rdx, SETDB_MISSING
	jmp	.out
.missing:
	mov	rax, SETDB_MISSING
	mov	rdx, SETDB_MISSING
.out:
	pop	r13
	pop	r12
	ret

is_subset:
	push	rbx
	push	r12
	push	r13
	push	r14
	call	find_two_sets
	cmp	rax, SETDB_MISSING
	je	.true
	cmp	rdx, SETDB_MISSING
	je	.false
	mov	r12, rax
	mov	r13, rdx
	xor	rbx, rbx
	mov	r14, r12
	imul	r14, SETDB_MAX_SET_MEMBERS
.loop:
	cmp	rbx, [setdb_set_counts + r12 * 8]
	jae	.true
	mov	rax, r14
	add	rax, rbx
	mov	rsi, [setdb_set_members + rax * 8]
	mov	rdi, r13
	call	setdb_set_find_member
	cmp	rax, SETDB_MISSING
	je	.false
	inc	rbx
	jmp	.loop
.true:
	mov	rax, 1
	jmp	.out
.false:
	xor	rax, rax
.out:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

build_select:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rsi
	mov	r13, rdx
	call	setdb_intern
	test	rax, rax
	jz	.done
	mov	rdi, rax
	call	setdb_find_rel
	cmp	rax, SETDB_MISSING
	je	.done
	mov	r14, rax
	mov	rdi, r13
	call	setdb_intern
	test	rax, rax
	jz	.done
	mov	r15, rax
	xor	rbx, rbx
	mov	r13, r14
	imul	r13, SETDB_MAX_REL_PAIRS
.loop:
	cmp	rbx, [setdb_rel_counts + r14 * 8]
	jae	.done
	mov	rdi, r12
	lea	rsi, [kw_first]
	call	str_eq
	test	rax, rax
	jz	.second
	mov	rax, r13
	add	rax, rbx
	mov	rdi, [setdb_rel_left + rax * 8]
	mov	rsi, r15
	call	str_eq
	test	rax, rax
	jz	.next
	mov	rax, r13
	add	rax, rbx
	mov	rdi, [setdb_rel_right + rax * 8]
	call	setdb_tmp_add_atom
	jmp	.next
.second:
	mov	rax, r13
	add	rax, rbx
	mov	rdi, [setdb_rel_right + rax * 8]
	mov	rsi, r15
	call	str_eq
	test	rax, rax
	jz	.next
	mov	rax, r13
	add	rax, rbx
	mov	rdi, [setdb_rel_left + rax * 8]
	call	setdb_tmp_add_atom
.next:
	inc	rbx
	jmp	.loop
.done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

build_join:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rsi
	call	setdb_intern
	test	rax, rax
	jz	.done
	mov	rdi, rax
	call	setdb_find_rel
	cmp	rax, SETDB_MISSING
	je	.done
	mov	r13, rax
	mov	rdi, r12
	call	setdb_intern
	test	rax, rax
	jz	.done
	mov	rdi, rax
	call	setdb_find_rel
	cmp	rax, SETDB_MISSING
	je	.done
	mov	r14, rax
	xor	rbx, rbx
.outer:
	cmp	rbx, [setdb_rel_counts + r13 * 8]
	jae	.done
	xor	r15, r15
.inner:
	cmp	r15, [setdb_rel_counts + r14 * 8]
	jae	.outer_next
	mov	rax, r13
	imul	rax, SETDB_MAX_REL_PAIRS
	add	rax, rbx
	mov	rdi, [setdb_rel_right + rax * 8]
	mov	rax, r14
	imul	rax, SETDB_MAX_REL_PAIRS
	add	rax, r15
	mov	rsi, [setdb_rel_left + rax * 8]
	call	str_eq
	test	rax, rax
	jz	.inner_next
	mov	rax, r13
	imul	rax, SETDB_MAX_REL_PAIRS
	add	rax, rbx
	mov	rdi, [setdb_rel_left + rax * 8]
	mov	rax, r14
	imul	rax, SETDB_MAX_REL_PAIRS
	add	rax, r15
	mov	rsi, [setdb_rel_right + rax * 8]
	call	setdb_tmp_add_pair
.inner_next:
	inc	r15
	jmp	.inner
.outer_next:
	inc	rbx
	jmp	.outer
.done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

find_rel_by_name:
	call	setdb_intern
	test	rax, rax
	jz	.missing
	mov	rdi, rax
	call	setdb_find_rel
	ret
.missing:
	mov	rax, SETDB_MISSING
	ret

add_rel_to_tmp:
	push	rbx
	push	r12
	push	r13
	call	find_rel_by_name
	cmp	rax, SETDB_MISSING
	je	.done
	mov	r12, rax
	xor	rbx, rbx
	mov	r13, r12
	imul	r13, SETDB_MAX_REL_PAIRS
.loop:
	cmp	rbx, [setdb_rel_counts + r12 * 8]
	jae	.done
	mov	rax, r13
	add	rax, rbx
	mov	rdi, [setdb_rel_left + rax * 8]
	mov	rsi, [setdb_rel_right + rax * 8]
	call	setdb_tmp_add_pair
	cmp	rax, 0
	jl	io_error
	inc	rbx
	jmp	.loop
.done:
	pop	r13
	pop	r12
	pop	rbx
	ret

build_domain:
	push	rbx
	push	r12
	push	r13
	call	find_rel_by_name
	cmp	rax, SETDB_MISSING
	je	.done
	mov	r12, rax
	xor	rbx, rbx
	mov	r13, r12
	imul	r13, SETDB_MAX_REL_PAIRS
.loop:
	cmp	rbx, [setdb_rel_counts + r12 * 8]
	jae	.done
	mov	rax, r13
	add	rax, rbx
	mov	rdi, [setdb_rel_left + rax * 8]
	call	setdb_tmp_add_atom
	cmp	rax, 0
	jl	io_error
	inc	rbx
	jmp	.loop
.done:
	pop	r13
	pop	r12
	pop	rbx
	ret

build_range:
	push	rbx
	push	r12
	push	r13
	call	find_rel_by_name
	cmp	rax, SETDB_MISSING
	je	.done
	mov	r12, rax
	xor	rbx, rbx
	mov	r13, r12
	imul	r13, SETDB_MAX_REL_PAIRS
.loop:
	cmp	rbx, [setdb_rel_counts + r12 * 8]
	jae	.done
	mov	rax, r13
	add	rax, rbx
	mov	rdi, [setdb_rel_right + rax * 8]
	call	setdb_tmp_add_atom
	cmp	rax, 0
	jl	io_error
	inc	rbx
	jmp	.loop
.done:
	pop	r13
	pop	r12
	pop	rbx
	ret

build_inverse:
	push	rbx
	push	r12
	push	r13
	call	find_rel_by_name
	cmp	rax, SETDB_MISSING
	je	.done
	mov	r12, rax
	xor	rbx, rbx
	mov	r13, r12
	imul	r13, SETDB_MAX_REL_PAIRS
.loop:
	cmp	rbx, [setdb_rel_counts + r12 * 8]
	jae	.done
	mov	rax, r13
	add	rax, rbx
	mov	rdi, [setdb_rel_right + rax * 8]
	mov	rsi, [setdb_rel_left + rax * 8]
	call	setdb_tmp_add_pair
	cmp	rax, 0
	jl	io_error
	inc	rbx
	jmp	.loop
.done:
	pop	r13
	pop	r12
	pop	rbx
	ret

find_two_rels:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	call	find_rel_by_name
	mov	rdx, rax
	mov	rdi, r13
	call	find_rel_by_name
	xchg	rax, rdx
	pop	r13
	pop	r12
	ret

build_rdiff:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	call	find_two_rels
	cmp	rax, SETDB_MISSING
	je	.done
	mov	r12, rax
	mov	r13, rdx
	xor	rbx, rbx
	mov	r14, r12
	imul	r14, SETDB_MAX_REL_PAIRS
.loop:
	cmp	rbx, [setdb_rel_counts + r12 * 8]
	jae	.done
	mov	rax, r14
	add	rax, rbx
	mov	r15, rax
	cmp	r13, SETDB_MISSING
	je	.add
	mov	rdi, r13
	mov	rsi, [setdb_rel_left + r15 * 8]
	mov	rdx, [setdb_rel_right + r15 * 8]
	call	setdb_rel_find_pair
	cmp	rax, SETDB_MISSING
	jne	.next
.add:
	mov	rdi, [setdb_rel_left + r15 * 8]
	mov	rsi, [setdb_rel_right + r15 * 8]
	call	setdb_tmp_add_pair
	cmp	rax, 0
	jl	io_error
.next:
	inc	rbx
	jmp	.loop
.done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

build_runion:
	push	r12
	mov	r12, rsi
	call	add_rel_to_tmp
	mov	rdi, r12
	call	add_rel_to_tmp
	pop	r12
	ret

build_rintersect:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	call	find_two_rels
	cmp	rax, SETDB_MISSING
	je	.done
	cmp	rdx, SETDB_MISSING
	je	.done
	mov	r12, rax
	mov	r13, rdx
	xor	rbx, rbx
	mov	r14, r12
	imul	r14, SETDB_MAX_REL_PAIRS
.loop:
	cmp	rbx, [setdb_rel_counts + r12 * 8]
	jae	.done
	mov	rax, r14
	add	rax, rbx
	mov	r15, rax
	mov	rdi, r13
	mov	rsi, [setdb_rel_left + r15 * 8]
	mov	rdx, [setdb_rel_right + r15 * 8]
	call	setdb_rel_find_pair
	cmp	rax, SETDB_MISSING
	je	.next
	mov	rdi, [setdb_rel_left + r15 * 8]
	mov	rsi, [setdb_rel_right + r15 * 8]
	call	setdb_tmp_add_pair
	cmp	rax, 0
	jl	io_error
.next:
	inc	rbx
	jmp	.loop
.done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

build_transitive_closure:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	call	find_rel_by_name
	cmp	rax, SETDB_MISSING
	je	.ok
	mov	r12, rax
	mov	rdi, [argv_base]
	mov	rdi, [rdi + 32]
	call	add_rel_to_tmp
.again:
	xor	r15, r15
	mov	rbx, [setdb_tmp_head]
.outer:
	test	rbx, rbx
	jz	.check
	cmp	qword [rbx + SETDB_TMP_RIGHT_OFF], 0
	je	.outer_next
	xor	r13, r13
	mov	r14, r12
	imul	r14, SETDB_MAX_REL_PAIRS
.inner:
	cmp	r13, [setdb_rel_counts + r12 * 8]
	jae	.outer_next
	mov	rax, r14
	add	rax, r13
	mov	rdi, [rbx + SETDB_TMP_RIGHT_OFF]
	mov	rsi, [setdb_rel_left + rax * 8]
	call	str_eq
	test	rax, rax
	jz	.inner_next
	mov	rax, r14
	add	rax, r13
	mov	rdi, [rbx + SETDB_TMP_LEFT_OFF]
	mov	rsi, [setdb_rel_right + rax * 8]
	call	setdb_tmp_contains_pair
	test	rax, rax
	jnz	.inner_next
	mov	rax, r14
	add	rax, r13
	mov	rdi, [rbx + SETDB_TMP_LEFT_OFF]
	mov	rsi, [setdb_rel_right + rax * 8]
	call	setdb_tmp_add_pair
	cmp	rax, 0
	jl	.err
	mov	r15, 1
.inner_next:
	inc	r13
	jmp	.inner
.outer_next:
	mov	rbx, [rbx + SETDB_TMP_NEXT_OFF]
	jmp	.outer
.check:
	test	r15, r15
	jnz	.again
.ok:
	xor	rax, rax
	jmp	.out
.err:
	mov	rax, -1
.out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

print_tmp_atoms_sorted:
	call	clear_printed
.loop:
	call	find_min_atom
	test	rax, rax
	je	.done
	mov	byte [rax + SETDB_TMP_PRINTED_OFF], 1
	mov	rdi, [rax + SETDB_TMP_LEFT_OFF]
	call	print_cstr
	call	print_nl
	jmp	.loop
.done:
	ret

print_tmp_pairs_sorted:
	push	rbx
	call	clear_printed
.loop:
	call	find_min_pair
	test	rax, rax
	je	.done
	mov	rbx, rax
	mov	byte [rbx + SETDB_TMP_PRINTED_OFF], 1
	lea	rdi, [lparen]
	call	print_cstr
	mov	rdi, [rbx + SETDB_TMP_LEFT_OFF]
	call	print_cstr
	lea	rdi, [comma]
	call	print_cstr
	mov	rdi, [rbx + SETDB_TMP_RIGHT_OFF]
	call	print_cstr
	lea	rdi, [rparen_nl]
	call	print_cstr
	jmp	.loop
.done:
	pop	rbx
	ret

clear_printed:
	mov	rbx, [setdb_tmp_head]
.loop:
	test	rbx, rbx
	jz	.done
	mov	byte [rbx + SETDB_TMP_PRINTED_OFF], 0
	mov	rbx, [rbx + SETDB_TMP_NEXT_OFF]
	jmp	.loop
.done:
	ret

find_min_atom:
	push	rbx
	push	r12
	xor	r12, r12
	mov	rbx, [setdb_tmp_head]
.loop:
	test	rbx, rbx
	jz	.done
	cmp	byte [rbx + SETDB_TMP_PRINTED_OFF], 0
	jne	.next
	cmp	qword [rbx + SETDB_TMP_RIGHT_OFF], 0
	jne	.next
	test	r12, r12
	je	.take
	mov	rdi, [rbx + SETDB_TMP_LEFT_OFF]
	mov	rsi, [r12 + SETDB_TMP_LEFT_OFF]
	call	setdb_str_cmp
	cmp	rax, 0
	jge	.next
.take:
	mov	r12, rbx
.next:
	mov	rbx, [rbx + SETDB_TMP_NEXT_OFF]
	jmp	.loop
.done:
	mov	rax, r12
	pop	r12
	pop	rbx
	ret

find_min_pair:
	push	rbx
	push	r12
	xor	r12, r12
	mov	rbx, [setdb_tmp_head]
.loop:
	test	rbx, rbx
	jz	.done
	cmp	byte [rbx + SETDB_TMP_PRINTED_OFF], 0
	jne	.next
	cmp	qword [rbx + SETDB_TMP_RIGHT_OFF], 0
	je	.next
	test	r12, r12
	je	.take
	mov	rdi, [rbx + SETDB_TMP_LEFT_OFF]
	mov	rsi, [r12 + SETDB_TMP_LEFT_OFF]
	call	setdb_str_cmp
	cmp	rax, 0
	jl	.take
	jg	.next
	mov	rdi, [rbx + SETDB_TMP_RIGHT_OFF]
	mov	rsi, [r12 + SETDB_TMP_RIGHT_OFF]
	call	setdb_str_cmp
	cmp	rax, 0
	jge	.next
.take:
	mov	r12, rbx
.next:
	mov	rbx, [rbx + SETDB_TMP_NEXT_OFF]
	jmp	.loop
.done:
	mov	rax, r12
	pop	r12
	pop	rbx
	ret

print_nl:
	lea	rdi, [nl_ch]
	mov	rsi, 1
	jmp	write_stdout

write_stdout:
	mov	rdx, rsi
	mov	rsi, rdi
	mov	rdi, STDOUT
	mov	rax, SYS_write
	syscall
	ret

usage:
	lea	rdi, [help_msg]
	mov	rsi, help_msg_len
	call	write_stderr
	exit	2

io_error:
	lea	rdi, [io_err_msg]
	mov	rsi, io_err_msg_len
	call	write_stderr
	exit	1

write_stderr:
	mov	rdx, rsi
	mov	rsi, rdi
	mov	rdi, STDERR
	mov	rax, SYS_write
	syscall
	ret

cmd_new db 'new', 0
cmd_add db 'add', 0
cmd_remove db 'remove', 0
cmd_relation db 'relation', 0
cmd_unrelation db 'unrelation', 0
cmd_members db 'members', 0
cmd_member db 'member', 0
cmd_union db 'union', 0
cmd_intersect db 'intersect', 0
cmd_diff db 'diff', 0
cmd_subset db 'subset', 0
cmd_select db 'select', 0
cmd_join db 'join', 0
cmd_domain db 'domain', 0
cmd_range db 'range', 0
cmd_inverse db 'inverse', 0
cmd_rdiff db 'rdiff', 0
cmd_runion db 'runion', 0
cmd_rintersect db 'rintersect', 0
cmd_transitive_closure db 'transitive-closure', 0
cmd_sets db 'sets', 0
cmd_relations db 'relations', 0
cmd_contains db 'contains', 0
cmd_pairs db 'pairs', 0
cmd_tag db 'tag', 0
cmd_files db 'files', 0
cmd_tags db 'tags', 0
cmd_store_domain db 'store-domain', 0
cmd_store_range db 'store-range', 0
cmd_store_inverse db 'store-inverse', 0
cmd_load db 'load', 0
cmd_dump db 'dump', 0
cmd_help db 'help', 0
opt_help db '--help', 0
opt_h db '-h', 0

op_sadd db 'SADD', 0
op_srem db 'SREM', 0
op_radd db 'RADD', 0
op_rrem db 'RREM', 0
kw_first db 'first', 0
kw_second db 'second', 0
tag_set_files db 'files', 0
tag_set_tags db 'tags', 0
tag_rel_has_tag db 'has_tag', 0

ops_log_suffix db '/ops.log', 0
ops_idx_suffix db '/ops.idx', 0
space_ch db ' '
nl_ch db 10
true_msg db 'true', 10, 0
false_msg db 'false', 10, 0
lparen db '(', 0
comma db ',', 0
rparen_nl db ')', 10, 0
help_msg db 'setdb - pure set-theoretic database CLI', 10
	db 10
	db 'Usage:', 10
	db '  setdb help', 10
	db '  setdb new DB', 10
	db '  setdb add DB SET ATOM...', 10
	db '  setdb remove DB SET ATOM', 10
	db '  setdb relation DB REL LEFT RIGHT', 10
	db '  setdb unrelation DB REL LEFT RIGHT', 10
	db '  setdb members DB SET', 10
	db '  setdb member DB SET ATOM', 10
	db '  setdb union DB SET_A SET_B', 10
	db '  setdb intersect DB SET_A SET_B', 10
	db '  setdb diff DB SET_A SET_B', 10
	db '  setdb subset DB SET_A SET_B', 10
	db '  setdb select DB REL first ATOM', 10
	db '  setdb select DB REL second ATOM', 10
	db '  setdb join DB REL_A REL_B', 10
	db '  setdb domain DB REL', 10
	db '  setdb range DB REL', 10
	db '  setdb inverse DB REL', 10
	db '  setdb rdiff DB REL_A REL_B', 10
	db '  setdb runion DB REL_A REL_B', 10
	db '  setdb rintersect DB REL_A REL_B', 10
	db '  setdb transitive-closure DB REL', 10
	db '  setdb sets DB', 10
	db '  setdb relations DB', 10
	db '  setdb contains DB ATOM', 10
	db '  setdb pairs DB REL', 10
	db '  setdb tag DB FILE TAG...', 10
	db '  setdb files DB TAG', 10
	db '  setdb tags DB FILE', 10
	db '  setdb store-domain DB REL NAME', 10
	db '  setdb store-range DB REL NAME', 10
	db '  setdb store-inverse DB REL NAME', 10
	db '  setdb load DB FACTS_FILE', 10
	db '  setdb dump DB', 10
	db 10
	db 'Model:', 10
	db '  DB is a directory bundle with ops.log and ops.idx.', 10
	db '  Sets contain unique atoms. Relations contain unique ordered pairs.', 10
	db '  There is no SQL, no NULL, no duplicate rows, and no multiset semantics.', 10
	db '  Names may contain letters, digits, underscore, dash, dot, and slash.', 10
	db 10
	db 'Commands:', 10
	db '  new         Create DB directory and empty append-only operation log.', 10
	db '  add         Add one or more atoms to SET; duplicate atoms are ignored.', 10
	db '  remove      Remove ATOM from SET; missing atoms are ignored.', 10
	db '  relation    Add pair (LEFT,RIGHT) to REL; duplicates are ignored.', 10
	db '  unrelation  Remove pair (LEFT,RIGHT) from REL; missing pairs are ignored.', 10
	db '  members     Print SET members, one atom per line, sorted.', 10
	db '  member      Print true if ATOM is in SET, otherwise false.', 10
	db '  union       Print SET_A union SET_B, sorted.', 10
	db '  intersect   Print SET_A intersection SET_B, sorted.', 10
	db '  diff        Print SET_A minus SET_B, sorted.', 10
	db '  subset      Print true if SET_A is a subset of SET_B, otherwise false.', 10
	db '  select      For first ATOM, print y where (ATOM,y) is in REL.', 10
	db '              For second ATOM, print x where (x,ATOM) is in REL.', 10
	db '  join        Relation composition: (x,y) in REL_A and (y,z) in REL_B -> (x,z).', 10
	db '  domain      Print all x where (x,y) is in REL.', 10
	db '  range       Print all y where (x,y) is in REL.', 10
	db '  inverse     Print (y,x) for every (x,y) in REL.', 10
	db '  rdiff       Print pairs in REL_A but not REL_B.', 10
	db '  runion      Print relation union, duplicate-free.', 10
	db '  rintersect  Print relation intersection.', 10
	db '  transitive-closure', 10
	db '              Print reachable pairs over one or more REL steps.', 10
	db '  sets        Print all set names in DB, sorted.', 10
	db '  relations   Print all relation names in DB, sorted.', 10
	db '  contains    Print all sets that contain ATOM, sorted.', 10
	db '  pairs       Print every pair in REL, sorted.', 10
	db '  tag         Tag FILE with one or more TAGs (files/tags/has_tag sugar).', 10
	db '  files       Print all files tagged TAG, sorted.', 10
	db '  tags        Print all tags on FILE, sorted.', 10
	db '  store-domain', 10
	db '              Compute domain(REL), then SADD each atom into set NAME.', 10
	db '  store-range', 10
	db '              Compute range(REL), then SADD each atom into set NAME.', 10
	db '  store-inverse', 10
	db '              Compute inverse(REL), then RADD each pair into relation NAME.', 10
	db '  load        Apply each op line (SADD/SREM/RADD/RREM) from FACTS_FILE.', 10
	db '              # and blank lines are ignored. One bad line stops the load.', 10
	db '  dump        Print every set/relation as SADD/RADD lines; load-able.', 10
	db 10
	db 'Examples:', 10
	db '  setdb new universe.db', 10
	db '  setdb add universe.db users alice bob carol', 10
	db '  setdb add universe.db admins alice', 10
	db '  setdb relation universe.db follows alice bob', 10
	db '  setdb relation universe.db follows bob carol', 10
	db '  setdb diff universe.db users admins', 10
	db '  setdb select universe.db follows first alice', 10
	db '  setdb join universe.db follows follows', 10
	db '  setdb transitive-closure universe.db follows', 10
	db 10
	db 'Output:', 10
	db '  Sets print one atom per line. Relations print one (left,right) pair per line.', 10
	db '  Empty results print nothing and exit successfully.', 10
	db 10
	db 'Exit codes:', 10
	db '  0  success', 10
	db '  1  I/O error or corrupt log', 10
	db '  2  invalid command, arity, or name', 10
help_msg_len = $ - help_msg
io_err_msg db 'setdb: io or corrupt log error', 10
io_err_msg_len = $ - io_err_msg

segment readable writeable

argc dq ?
argv_base dq ?
arg_index dq ?
db_path dq ?
ops_log_path rb SETDB_PATH_MAX
ops_idx_path rb SETDB_PATH_MAX
payload_buf rb SETDB_PAYLOAD_MAX
payload_len dq ?
payload_arg0 dq ?
payload_arg1 dq ?
payload_arg2 dq ?
payload_arg3 dq ?
replay_buf rb SETDB_PAYLOAD_MAX
replay_len dq ?
parse_ptr dq ?
tok0 dq ?
tok1 dq ?
tok2 dq ?
tok3 dq ?
setdb_temp_arena rb ARENA_SIZE
setdb_temp_storage rb SETDB_TEMP_ARENA_SIZE
load_read_buf rb LOAD_READ_BUF_SIZE
load_line_buf rb LOAD_LINE_BUF_SIZE + 1
load_line_copy rb LOAD_LINE_BUF_SIZE + 1
load_cb_line_ptr dq ?
load_cb_line_len dq ?

include "fasm/core/runtime_bss.inc"
runtime_print_bss
log_segment_bss
setdb_bss
scanner_bss
