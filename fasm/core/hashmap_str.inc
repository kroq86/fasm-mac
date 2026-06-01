; Chaining hash map: null-terminated string key -> tagged value (int or string).

HASHMAP_STR_MISSING equ -1
HASHMAP_STR_WRONGTYPE equ -2
HASHMAP_DEFAULT_BUCKETS equ 256
HASHMAP_STR_LOAD_NUM equ 4
HASHMAP_STR_LOAD_DEN equ 3

HM_VAL_INT equ 0
HM_VAL_STR equ 1

struc HashEntryStr
{
	.key dq ?
	.type dq ?
	.val dq ?
	.next dq ?
}

struc HashMapStr
{
	.bucket_count dq ?
	.buckets dq ?
	.size dq ?
}

virtual at 0
	_HashEntryStr HashEntryStr
	HASH_ENTRY_STR_SIZE = $
	_HashMapStr HashMapStr
	HASHMAP_STR_SIZE = $
end virtual

HASH_ENTRY_STR_KEY_OFF equ 0
HASH_ENTRY_STR_TYPE_OFF equ 8
HASH_ENTRY_STR_VAL_OFF equ 16
HASH_ENTRY_STR_NEXT_OFF equ 24

HASHMAP_STR_BUCKET_COUNT_OFF equ 0
HASHMAP_STR_BUCKETS_OFF equ 8
HASHMAP_STR_SIZE_OFF equ 16

; rdi = key ptr, rsi = bucket_count (power of 2)
; rax = bucket index
hash_index_str:
	push	rsi
	call	str_hash
	pop	rsi
	dec	rsi
	and	rax, rsi
	ret

; rdi = Heap*, rsi = null-terminated source
; rax = heap copy (null-terminated) or 0 on failure
str_copy_to_heap:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rdi, rsi
	call	str_len
	mov	rbx, rax
	lea	rsi, [rax + 1]
	mov	rdi, r12
	call	heap_alloc
	test	rax, rax
	jz	.sch_fail
	mov	rcx, rbx
	inc	rcx
	mov	rdi, rax
	mov	rsi, r13
	rep	movsb
.sch_fail:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = Heap*, rsi = HashMapStr*, rdx = bucket_count
hashmap_str_init:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	rbx, rsi
	mov	r13, rdx
	mov	[rbx + HASHMAP_STR_BUCKET_COUNT_OFF], r13
	mov	qword [rbx + HASHMAP_STR_SIZE_OFF], 0
	mov	rdi, r12
	mov	rsi, r13
	shl	rsi, 3
	call	heap_alloc
	mov	[rbx + HASHMAP_STR_BUCKETS_OFF], rax
	mov	rcx, r13
	xor	rax, rax
	mov	rdi, [rbx + HASHMAP_STR_BUCKETS_OFF]
.hsi_zero:
	mov	[rdi], rax
	add	rdi, 8
	dec	rcx
	jnz	.hsi_zero
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = HashMapStr*
hashmap_str_clear:
	push	r12
	mov	r12, rdi
	mov	rcx, [r12 + HASHMAP_STR_BUCKET_COUNT_OFF]
	mov	rdi, [r12 + HASHMAP_STR_BUCKETS_OFF]
	xor	rax, rax
.hsc_loop:
	test	rcx, rcx
	jz	.hsc_done
	mov	[rdi], rax
	add	rdi, 8
	dec	rcx
	jmp	.hsc_loop
.hsc_done:
	mov	qword [r12 + HASHMAP_STR_SIZE_OFF], 0
	pop	r12
	ret

; rdi = HashMapStr*, rsi = key ptr
; rax = bucket slot address (&head pointer)
hashmap_str_bucket_slot:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rdi, r13
	mov	rax, [r12 + HASHMAP_STR_BUCKET_COUNT_OFF]
	mov	rsi, rax
	call	hash_index_str
	mov	rdi, [r12 + HASHMAP_STR_BUCKETS_OFF]
	lea	rax, [rdi + rax * 8]
	pop	r13
	pop	r12
	ret

; rdi = HashMapStr*, rsi = key ptr
; rax = entry ptr or 0
hashmap_str_find_entry:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	call	hashmap_str_bucket_slot
	mov	rbx, [rax]
.hfe_loop:
	test	rbx, rbx
	jz	.hfe_missing
	mov	rdi, [rbx + HASH_ENTRY_STR_KEY_OFF]
	mov	rsi, r13
	call	str_eq
	test	rax, rax
	jnz	.hfe_found
	mov	rbx, [rbx + HASH_ENTRY_STR_NEXT_OFF]
	jmp	.hfe_loop
.hfe_found:
	mov	rax, rbx
	jmp	.hfe_done
.hfe_missing:
	xor	rax, rax
.hfe_done:
	pop	r13
	pop	r12
	pop	rbx
	ret

hashmap_str_get_entry:
	jmp	hashmap_str_find_entry

; rdi = HashMapStr*, rsi = key ptr
hashmap_str_get_int:
	push	rbx
	call	hashmap_str_find_entry
	test	rax, rax
	jz	.hsgi_missing
	mov	rbx, rax
	cmp	qword [rbx + HASH_ENTRY_STR_TYPE_OFF], HM_VAL_INT
	jne	.hsgi_wrong
	mov	rax, [rbx + HASH_ENTRY_STR_VAL_OFF]
	jmp	.hsgi_done
.hsgi_missing:
	mov	rax, HASHMAP_STR_MISSING
	jmp	.hsgi_done
.hsgi_wrong:
	mov	rax, HASHMAP_STR_WRONGTYPE
.hsgi_done:
	pop	rbx
	ret

hashmap_str_contains:
	call	hashmap_str_find_entry
	test	rax, rax
	setnz	al
	movzx	rax, al
	ret

; rdi=Heap*, rsi=Map*, rdx=key, rcx=type, r8=val
hashmap_str_put_entry:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	rbx, r8
	mov	rdi, r13
	mov	rsi, r14
	call	hashmap_str_bucket_slot
	push	rax
	mov	rax, [rsp]
	mov	rax, [rax]
.hspe_loop:
	test	rax, rax
	jz	.hspe_new
	mov	rdi, [rax + HASH_ENTRY_STR_KEY_OFF]
	mov	rsi, r14
	call	str_eq
	test	rax, rax
	jnz	.hspe_update
	mov	rax, [rax + HASH_ENTRY_STR_NEXT_OFF]
	jmp	.hspe_loop
.hspe_update:
	mov	[rax + HASH_ENTRY_STR_TYPE_OFF], r15
	mov	[rax + HASH_ENTRY_STR_VAL_OFF], rbx
	xor	r14, r14
	jmp	.hspe_resize
.hspe_new:
	mov	rdi, r12
	mov	rsi, r14
	call	str_copy_to_heap
	test	rax, rax
	jz	.hspe_done_pop
	mov	r14, rax
	mov	rdi, r12
	mov	rsi, HASH_ENTRY_STR_SIZE
	call	heap_alloc
	mov	[rax + HASH_ENTRY_STR_KEY_OFF], r14
	mov	[rax + HASH_ENTRY_STR_TYPE_OFF], r15
	mov	[rax + HASH_ENTRY_STR_VAL_OFF], rbx
	mov	r10, [rsp]
	mov	rcx, [r10]
	mov	[rax + HASH_ENTRY_STR_NEXT_OFF], rcx
	mov	[r10], rax
	inc	qword [r13 + HASHMAP_STR_SIZE_OFF]
	mov	r14, 1
.hspe_resize:
	test	r14, r14
	jz	.hspe_done_pop
	mov	rdi, r12
	mov	rsi, r13
	call	hashmap_str_maybe_resize
.hspe_done_pop:
	pop	r10
.hspe_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

hashmap_str_put_int:
	mov	r8, rcx
	mov	rcx, HM_VAL_INT
	jmp	hashmap_str_put_entry

hashmap_str_put_str:
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	rdi, r12
	mov	rsi, rcx
	call	str_copy_to_heap
	test	rax, rax
	jz	.hsps_done
	mov	r8, rax
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r14
	mov	rcx, HM_VAL_STR
	call	hashmap_str_put_entry
.hsps_done:
	pop	r14
	pop	r13
	pop	r12
	ret

hashmap_str_maybe_resize:
	push	rbx
	mov	rbx, rsi
	mov	rax, [rbx + HASHMAP_STR_SIZE_OFF]
	imul	rax, HASHMAP_STR_LOAD_NUM
	mov	rcx, [rbx + HASHMAP_STR_BUCKET_COUNT_OFF]
	imul	rcx, HASHMAP_STR_LOAD_DEN
	cmp	rax, rcx
	jbe	.hsmr_done
	call	hashmap_str_resize
.hsmr_done:
	pop	rbx
	ret

hashmap_str_resize:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	rax, [r13 + HASHMAP_STR_BUCKET_COUNT_OFF]
	shl	rax, 1
	mov	r14, rax
	mov	rdi, r12
	mov	rsi, r14
	shl	rsi, 3
	call	heap_alloc
	mov	r15, rax
	mov	rcx, r14
	xor	rax, rax
	mov	rdi, r15
.hsr_zero:
	mov	[rdi], rax
	add	rdi, 8
	dec	rcx
	jnz	.hsr_zero
	mov	rbx, [r13 + HASHMAP_STR_BUCKET_COUNT_OFF]
	xor	r10, r10
.hsr_bucket:
	cmp	r10, rbx
	jae	.hsr_finish
	mov	rdi, [r13 + HASHMAP_STR_BUCKETS_OFF]
	mov	r11, [rdi + r10 * 8]
.hsr_chain:
	test	r11, r11
	jz	.hsr_next_bucket
	mov	rax, r11
	mov	r11, [r11 + HASH_ENTRY_STR_NEXT_OFF]
	mov	rdi, [rax + HASH_ENTRY_STR_KEY_OFF]
	mov	rsi, r14
	call	hash_index_str
	mov	rdi, r15
	lea	rdi, [rdi + rax * 8]
	mov	rcx, [rdi]
	mov	[rax + HASH_ENTRY_STR_NEXT_OFF], rcx
	mov	[rdi], rax
	jmp	.hsr_chain
.hsr_next_bucket:
	inc	r10
	jmp	.hsr_bucket
.hsr_finish:
	mov	[r13 + HASHMAP_STR_BUCKETS_OFF], r15
	mov	[r13 + HASHMAP_STR_BUCKET_COUNT_OFF], r14
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

hashmap_str_del:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	call	hashmap_str_bucket_slot
	mov	r14, rax
	mov	rbx, [r14]
.hsd_loop:
	test	rbx, rbx
	jz	.hsd_missing
	mov	rdi, [rbx + HASH_ENTRY_STR_KEY_OFF]
	mov	rsi, r13
	call	str_eq
	test	rax, rax
	jnz	.hsd_unlink
	lea	r14, [rbx + HASH_ENTRY_STR_NEXT_OFF]
	mov	rbx, [r14]
	jmp	.hsd_loop
.hsd_unlink:
	mov	rax, [rbx + HASH_ENTRY_STR_NEXT_OFF]
	mov	[r14], rax
	dec	qword [r12 + HASHMAP_STR_SIZE_OFF]
	mov	rax, 1
	jmp	.hsd_done
.hsd_missing:
	xor	rax, rax
.hsd_done:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi=Map*, rsi=callback, rdx=user data
hashmap_str_foreach:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	push	rcx
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	rcx, [r12 + HASHMAP_STR_BUCKET_COUNT_OFF]
	xor	r15, r15
.hsf_bucket:
	cmp	r15, rcx
	jae	.hsf_done
	mov	rax, [r12 + HASHMAP_STR_BUCKETS_OFF]
	mov	rbx, [rax + r15 * 8]
.hsf_chain:
	test	rbx, rbx
	jz	.hsf_next
	mov	rdi, [rbx + HASH_ENTRY_STR_KEY_OFF]
	mov	rsi, rbx
	mov	rdx, r14
	call	r13
	mov	rbx, [rbx + HASH_ENTRY_STR_NEXT_OFF]
	jmp	.hsf_chain
.hsf_next:
	inc	r15
	jmp	.hsf_bucket
.hsf_done:
	pop	rcx
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

gc_mark_hashmap_str:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	rax, [r12 + HASHMAP_STR_BUCKETS_OFF]
	mark_block rax
	mov	rcx, [r12 + HASHMAP_STR_BUCKET_COUNT_OFF]
	xor	r13, r13
.gmhs_bucket:
	cmp	r13, rcx
	jae	.gmhs_done
	mov	rax, [r12 + HASHMAP_STR_BUCKETS_OFF]
	mov	rbx, [rax + r13 * 8]
.gmhs_chain:
	test	rbx, rbx
	jz	.gmhs_next
	mark_block rbx
	mov	rdi, [rbx + HASH_ENTRY_STR_KEY_OFF]
	mark_block rdi
	cmp	qword [rbx + HASH_ENTRY_STR_TYPE_OFF], HM_VAL_STR
	jne	.gmhs_next_entry
	mov	rdi, [rbx + HASH_ENTRY_STR_VAL_OFF]
	mark_block rdi
.gmhs_next_entry:
	mov	rbx, [rbx + HASH_ENTRY_STR_NEXT_OFF]
	jmp	.gmhs_chain
.gmhs_next:
	inc	r13
	jmp	.gmhs_bucket
.gmhs_done:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi=Heap*, rsi=HashMapStr*
gc_collect_hashmap_str:
	push	rbx
	push	r12
	push	r14
	mov	r12, rdi
	mov	rcx, [r12 + HEAP_BLOCKS_OFF]
.gchs_clear:
	test	rcx, rcx
	jz	.gchs_mark
	mov	byte [rcx + BLOCK_MARKED_OFF], 0
	mov	rcx, [rcx + BLOCK_NEXT_OFF]
	jmp	.gchs_clear
.gchs_mark:
	mov	rdi, rsi
	call	gc_mark_hashmap_str
	lea	r14, [r12 + HEAP_BLOCKS_OFF]
	mov	rcx, [r12 + HEAP_BLOCKS_OFF]
.gchs_sweep:
	test	rcx, rcx
	jz	.gchs_done
	mov	rbx, [rcx + BLOCK_NEXT_OFF]
	cmp	byte [rcx + BLOCK_MARKED_OFF], 0
	jne	.gchs_keep
	mov	rdi, rcx
	mov	rsi, [rcx + BLOCK_SIZE_OFF]
	mov	rax, SYS_munmap
	syscall
	sub	qword [r12 + HEAP_TOTAL_OFF], rsi
	mov	[r14], rbx
	mov	rcx, rbx
	jmp	.gchs_sweep
.gchs_keep:
	mov	byte [rcx + BLOCK_MARKED_OFF], 0
	lea	r14, [rcx + BLOCK_NEXT_OFF]
	mov	rcx, rbx
	jmp	.gchs_sweep
.gchs_done:
	pop	r14
	pop	r12
	pop	rbx
	ret
