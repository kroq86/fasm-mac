; Small pure set-theoretic universe helpers.
;
; Contracts use System V AMD64 ABI. Strings are null-terminated and may not
; contain spaces in v1. Data is bounded on purpose so product smoke tests stay
; tiny and predictable.

SETDB_OK equ 0
SETDB_ERR equ -1
SETDB_MISSING equ 0FFFFFFFFFFFFFFFFh

SETDB_MAX_STRINGS equ 512
SETDB_STRING_POOL_SIZE equ 65536
SETDB_MAX_SETS equ 64
SETDB_MAX_SET_MEMBERS equ 256
SETDB_MAX_RELS equ 64
SETDB_MAX_REL_PAIRS equ 512
SETDB_TMP_MAX equ 1024
SETDB_TMP_LEFT_OFF equ 0
SETDB_TMP_RIGHT_OFF equ 8
SETDB_TMP_PRINTED_OFF equ 16
SETDB_TMP_NEXT_OFF equ 24
SETDB_TMP_ENTRY_SIZE equ 32

macro setdb_bss
{
	setdb_string_count dq ?
	setdb_string_pool_used dq ?
	setdb_strings rq SETDB_MAX_STRINGS
	setdb_string_pool rb SETDB_STRING_POOL_SIZE

	setdb_set_count dq ?
	setdb_set_names rq SETDB_MAX_SETS
	setdb_set_counts rq SETDB_MAX_SETS
	setdb_set_members rq SETDB_MAX_SETS * SETDB_MAX_SET_MEMBERS

	setdb_rel_count dq ?
	setdb_rel_names rq SETDB_MAX_RELS
	setdb_rel_counts rq SETDB_MAX_RELS
	setdb_rel_left rq SETDB_MAX_RELS * SETDB_MAX_REL_PAIRS
	setdb_rel_right rq SETDB_MAX_RELS * SETDB_MAX_REL_PAIRS

	setdb_tmp_arena dq ?
	setdb_tmp_head dq ?
	setdb_tmp_count dq ?
}

; rdi = cstr
; rax = 1 valid, 0 invalid
setdb_name_valid:
	push	rbx
	xor	rbx, rbx
.loop:
	mov	al, [rdi + rbx]
	test	al, al
	jz	.done
	cmp	al, '0'
	jb	.check_upper
	cmp	al, '9'
	jbe	.next
.check_upper:
	cmp	al, 'A'
	jb	.check_lower
	cmp	al, 'Z'
	jbe	.next
.check_lower:
	cmp	al, 'a'
	jb	.check_extra
	cmp	al, 'z'
	jbe	.next
.check_extra:
	cmp	al, '_'
	je	.next
	cmp	al, '-'
	je	.next
	cmp	al, '.'
	je	.next
	xor	rax, rax
	jmp	.out
.next:
	inc	rbx
	jmp	.loop
.done:
	test	rbx, rbx
	setnz	al
	movzx	rax, al
.out:
	pop	rbx
	ret

; no args
setdb_init:
	mov	qword [setdb_string_count], 0
	mov	qword [setdb_string_pool_used], 0
	mov	qword [setdb_set_count], 0
	mov	qword [setdb_rel_count], 0
	mov	qword [setdb_tmp_count], 0
	ret

; rdi=a, rsi=b
; rax = -1/0/1
setdb_str_cmp:
	xor	rax, rax
.loop:
	movzx	ecx, byte [rdi]
	movzx	edx, byte [rsi]
	cmp	cl, dl
	jne	.diff
	test	cl, cl
	jz	.eq
	inc	rdi
	inc	rsi
	jmp	.loop
.diff:
	mov	rax, -1
	cmp	cl, dl
	jb	.out
	mov	rax, 1
	jmp	.out
.eq:
	xor	rax, rax
.out:
	ret

; rdi = cstr
; rax = canonical ptr, or 0 on overflow/invalid
setdb_intern:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	call	setdb_name_valid
	test	rax, rax
	jz	.fail
	xor	rbx, rbx
.find_loop:
	cmp	rbx, [setdb_string_count]
	jae	.new
	mov	rdi, [setdb_strings + rbx * 8]
	mov	rsi, r12
	call	str_eq
	test	rax, rax
	jnz	.found
	inc	rbx
	jmp	.find_loop
.found:
	mov	rax, [setdb_strings + rbx * 8]
	jmp	.out
.new:
	cmp	qword [setdb_string_count], SETDB_MAX_STRINGS
	jae	.fail
	mov	rdi, r12
	call	str_len
	mov	r13, rax
	mov	rax, [setdb_string_pool_used]
	lea	rcx, [rax + r13 + 1]
	cmp	rcx, SETDB_STRING_POOL_SIZE
	ja	.fail
	lea	rdi, [setdb_string_pool + rax]
	mov	rsi, r12
	mov	rdx, r13
	inc	rdx
	call	memcpy
	mov	rbx, [setdb_string_count]
	lea	rax, [setdb_string_pool]
	add	rax, [setdb_string_pool_used]
	mov	[setdb_strings + rbx * 8], rax
	inc	qword [setdb_string_count]
	mov	rcx, r13
	inc	rcx
	add	[setdb_string_pool_used], rcx
	jmp	.out
.fail:
	xor	rax, rax
.out:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = set name ptr
; rax = set index, or SETDB_MISSING
setdb_find_set:
	push	rbx
	push	r12
	mov	r12, rdi
	xor	rbx, rbx
.loop:
	cmp	rbx, [setdb_set_count]
	jae	.missing
	mov	rdi, [setdb_set_names + rbx * 8]
	mov	rsi, r12
	call	str_eq
	test	rax, rax
	jnz	.found
	inc	rbx
	jmp	.loop
.found:
	mov	rax, rbx
	jmp	.out
.missing:
	mov	rax, SETDB_MISSING
.out:
	pop	r12
	pop	rbx
	ret

; rdi = set name cstr
; rax = set index, or SETDB_MISSING
setdb_get_or_create_set:
	push	rbx
	push	r12
	mov	r12, rdi
	call	setdb_intern
	test	rax, rax
	jz	.fail
	mov	rdi, rax
	call	setdb_find_set
	cmp	rax, SETDB_MISSING
	jne	.out
	cmp	qword [setdb_set_count], SETDB_MAX_SETS
	jae	.fail
	mov	rbx, [setdb_set_count]
	mov	rdi, r12
	call	setdb_intern
	mov	[setdb_set_names + rbx * 8], rax
	mov	qword [setdb_set_counts + rbx * 8], 0
	inc	qword [setdb_set_count]
	mov	rax, rbx
	jmp	.out
.fail:
	mov	rax, SETDB_MISSING
.out:
	pop	r12
	pop	rbx
	ret

; rdi = set index, rsi = atom canonical ptr
; rax = member offset or SETDB_MISSING
setdb_set_find_member:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rax, r12
	imul	rax, SETDB_MAX_SET_MEMBERS
	xor	rbx, rbx
.loop:
	cmp	rbx, [setdb_set_counts + r12 * 8]
	jae	.missing
	mov	rdx, rax
	add	rdx, rbx
	mov	rdi, [setdb_set_members + rdx * 8]
	mov	rsi, r13
	call	str_eq
	test	rax, rax
	jnz	.found
	mov	rax, r12
	imul	rax, SETDB_MAX_SET_MEMBERS
	inc	rbx
	jmp	.loop
.found:
	mov	rax, rbx
	jmp	.out
.missing:
	mov	rax, SETDB_MISSING
.out:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = set name, rsi = atom
; rax = SETDB_OK / SETDB_ERR
setdb_set_add:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	call	setdb_get_or_create_set
	cmp	rax, SETDB_MISSING
	je	.err
	mov	rbx, rax
	mov	rdi, r13
	call	setdb_intern
	test	rax, rax
	jz	.err
	mov	r13, rax
	mov	rdi, rbx
	mov	rsi, r13
	call	setdb_set_find_member
	cmp	rax, SETDB_MISSING
	jne	.ok
	mov	rax, [setdb_set_counts + rbx * 8]
	cmp	rax, SETDB_MAX_SET_MEMBERS
	jae	.err
	mov	rcx, rbx
	imul	rcx, SETDB_MAX_SET_MEMBERS
	add	rcx, rax
	mov	[setdb_set_members + rcx * 8], r13
	inc	qword [setdb_set_counts + rbx * 8]
.ok:
	xor	rax, rax
	jmp	.out
.err:
	mov	rax, SETDB_ERR
.out:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = set name, rsi = atom
setdb_set_remove:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	call	setdb_intern
	test	rax, rax
	jz	.err
	mov	rdi, rax
	call	setdb_find_set
	cmp	rax, SETDB_MISSING
	je	.ok
	mov	rbx, rax
	mov	rdi, r13
	call	setdb_intern
	test	rax, rax
	jz	.err
	mov	r13, rax
	mov	rdi, rbx
	mov	rsi, r13
	call	setdb_set_find_member
	cmp	rax, SETDB_MISSING
	je	.ok
	mov	r14, rax
	mov	rax, [setdb_set_counts + rbx * 8]
	dec	rax
	mov	[setdb_set_counts + rbx * 8], rax
	mov	rcx, rbx
	imul	rcx, SETDB_MAX_SET_MEMBERS
	add	rcx, r14
	mov	rdx, rbx
	imul	rdx, SETDB_MAX_SET_MEMBERS
	add	rdx, rax
	mov	rax, [setdb_set_members + rdx * 8]
	mov	[setdb_set_members + rcx * 8], rax
.ok:
	xor	rax, rax
	jmp	.out
.err:
	mov	rax, SETDB_ERR
.out:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = rel canonical name ptr
; rax = rel index or SETDB_MISSING
setdb_find_rel:
	push	rbx
	push	r12
	mov	r12, rdi
	xor	rbx, rbx
.loop:
	cmp	rbx, [setdb_rel_count]
	jae	.missing
	mov	rdi, [setdb_rel_names + rbx * 8]
	mov	rsi, r12
	call	str_eq
	test	rax, rax
	jnz	.found
	inc	rbx
	jmp	.loop
.found:
	mov	rax, rbx
	jmp	.out
.missing:
	mov	rax, SETDB_MISSING
.out:
	pop	r12
	pop	rbx
	ret

; rdi = rel name cstr
; rax = rel index or SETDB_MISSING
setdb_get_or_create_rel:
	push	rbx
	push	r12
	mov	r12, rdi
	call	setdb_intern
	test	rax, rax
	jz	.fail
	mov	rdi, rax
	call	setdb_find_rel
	cmp	rax, SETDB_MISSING
	jne	.out
	cmp	qword [setdb_rel_count], SETDB_MAX_RELS
	jae	.fail
	mov	rbx, [setdb_rel_count]
	mov	rdi, r12
	call	setdb_intern
	mov	[setdb_rel_names + rbx * 8], rax
	mov	qword [setdb_rel_counts + rbx * 8], 0
	inc	qword [setdb_rel_count]
	mov	rax, rbx
	jmp	.out
.fail:
	mov	rax, SETDB_MISSING
.out:
	pop	r12
	pop	rbx
	ret

; rdi = rel index, rsi = left ptr, rdx = right ptr
; rax = pair offset or SETDB_MISSING
setdb_rel_find_pair:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	rax, r12
	imul	rax, SETDB_MAX_REL_PAIRS
	xor	rbx, rbx
.loop:
	cmp	rbx, [setdb_rel_counts + r12 * 8]
	jae	.missing
	mov	rdx, rax
	add	rdx, rbx
	mov	rdi, [setdb_rel_left + rdx * 8]
	mov	rsi, r13
	call	str_eq
	test	rax, rax
	jz	.next
	mov	rax, r12
	imul	rax, SETDB_MAX_REL_PAIRS
	mov	rdx, rax
	add	rdx, rbx
	mov	rdi, [setdb_rel_right + rdx * 8]
	mov	rsi, r14
	call	str_eq
	test	rax, rax
	jnz	.found
.next:
	mov	rax, r12
	imul	rax, SETDB_MAX_REL_PAIRS
	inc	rbx
	jmp	.loop
.found:
	mov	rax, rbx
	jmp	.out
.missing:
	mov	rax, SETDB_MISSING
.out:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = rel name, rsi = left atom, rdx = right atom
setdb_rel_add:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	call	setdb_get_or_create_rel
	cmp	rax, SETDB_MISSING
	je	.err
	mov	rbx, rax
	mov	rdi, r13
	call	setdb_intern
	test	rax, rax
	jz	.err
	mov	r13, rax
	mov	rdi, r14
	call	setdb_intern
	test	rax, rax
	jz	.err
	mov	r14, rax
	mov	rdi, rbx
	mov	rsi, r13
	mov	rdx, r14
	call	setdb_rel_find_pair
	cmp	rax, SETDB_MISSING
	jne	.ok
	mov	rax, [setdb_rel_counts + rbx * 8]
	cmp	rax, SETDB_MAX_REL_PAIRS
	jae	.err
	mov	rcx, rbx
	imul	rcx, SETDB_MAX_REL_PAIRS
	add	rcx, rax
	mov	[setdb_rel_left + rcx * 8], r13
	mov	[setdb_rel_right + rcx * 8], r14
	inc	qword [setdb_rel_counts + rbx * 8]
.ok:
	xor	rax, rax
	jmp	.out
.err:
	mov	rax, SETDB_ERR
.out:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = rel name, rsi = left atom, rdx = right atom
setdb_rel_remove:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	call	setdb_intern
	test	rax, rax
	jz	.err
	mov	rdi, rax
	call	setdb_find_rel
	cmp	rax, SETDB_MISSING
	je	.ok
	mov	rbx, rax
	mov	rdi, r13
	call	setdb_intern
	test	rax, rax
	jz	.err
	mov	r13, rax
	mov	rdi, r14
	call	setdb_intern
	test	rax, rax
	jz	.err
	mov	r14, rax
	mov	rdi, rbx
	mov	rsi, r13
	mov	rdx, r14
	call	setdb_rel_find_pair
	cmp	rax, SETDB_MISSING
	je	.ok
	mov	r15, rax
	mov	rax, [setdb_rel_counts + rbx * 8]
	dec	rax
	mov	[setdb_rel_counts + rbx * 8], rax
	mov	rcx, rbx
	imul	rcx, SETDB_MAX_REL_PAIRS
	add	rcx, r15
	mov	rdx, rbx
	imul	rdx, SETDB_MAX_REL_PAIRS
	add	rdx, rax
	mov	rax, [setdb_rel_left + rdx * 8]
	mov	[setdb_rel_left + rcx * 8], rax
	mov	rax, [setdb_rel_right + rdx * 8]
	mov	[setdb_rel_right + rcx * 8], rax
.ok:
	xor	rax, rax
	jmp	.out
.err:
	mov	rax, SETDB_ERR
.out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = string ptr
; rax = 1 if already in tmp atom entries, 0 otherwise
setdb_tmp_contains_atom:
	push	rbx
	push	r12
	mov	r12, rdi
	mov	rbx, [setdb_tmp_head]
.loop:
	test	rbx, rbx
	jz	.no
	cmp	qword [rbx + SETDB_TMP_RIGHT_OFF], 0
	jne	.next
	mov	rdi, [rbx + SETDB_TMP_LEFT_OFF]
	mov	rsi, r12
	call	str_eq
	test	rax, rax
	jnz	.yes
.next:
	mov	rbx, [rbx + SETDB_TMP_NEXT_OFF]
	jmp	.loop
.yes:
	mov	rax, 1
	jmp	.out
.no:
	xor	rax, rax
.out:
	pop	r12
	pop	rbx
	ret

; rdi = atom ptr
setdb_tmp_add_atom:
	push	rbx
	push	r12
	mov	r12, rdi
	call	setdb_tmp_contains_atom
	test	rax, rax
	jnz	.ok
	mov	rbx, [setdb_tmp_count]
	cmp	rbx, SETDB_TMP_MAX
	jae	.err
	mov	rdi, [setdb_tmp_arena]
	mov	rsi, SETDB_TMP_ENTRY_SIZE
	call	arena_alloc
	test	rax, rax
	jz	.err
	mov	[rax + SETDB_TMP_LEFT_OFF], r12
	mov	qword [rax + SETDB_TMP_RIGHT_OFF], 0
	mov	byte [rax + SETDB_TMP_PRINTED_OFF], 0
	mov	rcx, [setdb_tmp_head]
	mov	[rax + SETDB_TMP_NEXT_OFF], rcx
	mov	[setdb_tmp_head], rax
	inc	qword [setdb_tmp_count]
.ok:
	xor	rax, rax
	jmp	.out
.err:
	mov	rax, SETDB_ERR
.out:
	pop	r12
	pop	rbx
	ret

; rdi=left ptr, rsi=right ptr
; rax = 1 if already in tmp pairs
setdb_tmp_contains_pair:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rbx, [setdb_tmp_head]
.loop:
	test	rbx, rbx
	jz	.no
	mov	rdi, [rbx + SETDB_TMP_LEFT_OFF]
	mov	rsi, r12
	call	str_eq
	test	rax, rax
	jz	.next
	mov	rdi, [rbx + SETDB_TMP_RIGHT_OFF]
	test	rdi, rdi
	jz	.next
	mov	rsi, r13
	call	str_eq
	test	rax, rax
	jnz	.yes
.next:
	mov	rbx, [rbx + SETDB_TMP_NEXT_OFF]
	jmp	.loop
.yes:
	mov	rax, 1
	jmp	.out
.no:
	xor	rax, rax
.out:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi=left ptr, rsi=right ptr
setdb_tmp_add_pair:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	call	setdb_tmp_contains_pair
	test	rax, rax
	jnz	.ok
	mov	rbx, [setdb_tmp_count]
	cmp	rbx, SETDB_TMP_MAX
	jae	.err
	mov	rdi, [setdb_tmp_arena]
	mov	rsi, SETDB_TMP_ENTRY_SIZE
	call	arena_alloc
	test	rax, rax
	jz	.err
	mov	[rax + SETDB_TMP_LEFT_OFF], r12
	mov	[rax + SETDB_TMP_RIGHT_OFF], r13
	mov	byte [rax + SETDB_TMP_PRINTED_OFF], 0
	mov	rcx, [setdb_tmp_head]
	mov	[rax + SETDB_TMP_NEXT_OFF], rcx
	mov	[setdb_tmp_head], rax
	inc	qword [setdb_tmp_count]
.ok:
	xor	rax, rax
	jmp	.out
.err:
	mov	rax, SETDB_ERR
.out:
	pop	r13
	pop	r12
	pop	rbx
	ret

setdb_tmp_clear:
	mov	qword [setdb_tmp_count], 0
	mov	qword [setdb_tmp_head], 0
	ret
