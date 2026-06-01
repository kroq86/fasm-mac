; Chaining hash map: int64 key -> int64 value.

HASH_MUL equ 2654435761
HASHMAP_DEFAULT_BUCKETS equ 256
HASHMAP_MISSING equ -1

struc HashEntry
{
	.key dq ?
	.value dq ?
	.next dq ?
}

struc HashMapInt
{
	.bucket_count dq ?
	.buckets dq ?
}

virtual at 0
	_HashEntry HashEntry
	HASH_ENTRY_SIZE = $
	_HashMap HashMapInt
	HASHMAP_SIZE = $
end virtual

HASH_ENTRY_KEY_OFF equ 0
HASH_ENTRY_VALUE_OFF equ 8
HASH_ENTRY_NEXT_OFF equ 16

HASHMAP_BUCKET_COUNT_OFF equ 0
HASHMAP_BUCKETS_OFF equ 8

; rdi = key, rsi = bucket_count (power of 2)
; rax = bucket index
hash_index:
	mov	rax, rdi
	mov	rcx, rax
	shl	rcx, 13
	xor	rax, rcx
	mov	rcx, rax
	shr	rcx, 7
	xor	rax, rcx
	mov	rcx, rsi
	dec	rcx
	and	rax, rcx
	ret

; rdi = Heap*, rsi = HashMapInt*, rdx = bucket_count
hashmap_init:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	rbx, rsi
	mov	r13, rdx
	mov	[rbx + HASHMAP_BUCKET_COUNT_OFF], r13
	mov	rdi, r12
	mov	rsi, r13
	shl	rsi, 3
	call	heap_alloc
	mov	[rbx + HASHMAP_BUCKETS_OFF], rax
	mov	rcx, r13
	xor	rax, rax
	mov	rdi, [rbx + HASHMAP_BUCKETS_OFF]
.zero_loop:
	mov	[rdi], rax
	add	rdi, 8
	dec	rcx
	jnz	.zero_loop
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = HashMapInt*, rsi = key
; rax = bucket slot address (&head pointer)
hashmap_bucket_slot:
	push	rsi
	mov	rax, rsi
	mov	rsi, [rdi + HASHMAP_BUCKET_COUNT_OFF]
	call	hash_index
	mov	rsi, [rdi + HASHMAP_BUCKETS_OFF]
	lea	rax, [rsi + rax * 8]
	pop	rsi
	ret

; rdi = HashMapInt*, rsi = key
; rax = value or HASHMAP_MISSING
hashmap_get:
	push	rbx
	push	r12
	mov	r12, rdi
	mov	rbx, rsi
	call	hashmap_bucket_slot
	mov	rax, [rax]
.find_loop:
	test	rax, rax
	jz	.get_missing
	cmp	[rax + HASH_ENTRY_KEY_OFF], rbx
	je	.get_found
	mov	rax, [rax + HASH_ENTRY_NEXT_OFF]
	jmp	.find_loop
.get_found:
	mov	rax, [rax + HASH_ENTRY_VALUE_OFF]
	jmp	.get_done
.get_missing:
	mov	rax, HASHMAP_MISSING
.get_done:
	pop	r12
	pop	rbx
	ret

; rdi = HashMapInt*, rsi = key
; rax = 1 if present, 0 otherwise
hashmap_contains:
	call	hashmap_get
	cmp	rax, HASHMAP_MISSING
	je	.hc_no
	mov	rax, 1
	ret
.hc_no:
	xor	rax, rax
	ret

; rdi = Heap*, rsi = HashMapInt*, rdx = key, rcx = value
hashmap_put:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	rdi, r13
	mov	rsi, r14
	call	hashmap_bucket_slot
	mov	rbx, rax
	mov	rax, [rbx]
.update_loop:
	test	rax, rax
	jz	.put_new
	cmp	[rax + HASH_ENTRY_KEY_OFF], r14
	je	.put_update
	mov	rax, [rax + HASH_ENTRY_NEXT_OFF]
	jmp	.update_loop
.put_update:
	mov	[rax + HASH_ENTRY_VALUE_OFF], r15
	jmp	.put_done
.put_new:
	mov	rdi, r12
	mov	rsi, HASH_ENTRY_SIZE
	call	heap_alloc
	mov	[rax + HASH_ENTRY_KEY_OFF], r14
	mov	[rax + HASH_ENTRY_VALUE_OFF], r15
	mov	rcx, [rbx]
	mov	[rax + HASH_ENTRY_NEXT_OFF], rcx
	mov	[rbx], rax
.put_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = HashMapInt*
gc_mark_hashmap:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	rax, [r12 + HASHMAP_BUCKETS_OFF]
	mark_block rax
	mov	rcx, [r12 + HASHMAP_BUCKET_COUNT_OFF]
	xor	r13, r13
.bucket_loop:
	cmp	r13, rcx
	jae	.gmh_done
	mov	rax, [r12 + HASHMAP_BUCKETS_OFF]
	mov	rbx, [rax + r13 * 8]
.chain_loop:
	test	rbx, rbx
	jz	.bucket_next
	mark_block rbx
	mov	rbx, [rbx + HASH_ENTRY_NEXT_OFF]
	jmp	.chain_loop
.bucket_next:
	inc	r13
	jmp	.bucket_loop
.gmh_done:
	pop	r13
	pop	r12
	pop	rbx
	ret
