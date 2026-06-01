; Tagged Value, VecHeader, StrBuf layouts and sizes.

VAL_INT equ 0
VAL_STR equ 1
VAL_ARRAY equ 2

GC_MAX_DEPTH equ 16

struc BlockHeader
{
	.next dq ?
	.size dq ?
	.marked db ?
	align 8
}

struc Heap
{
	.blocks dq ?
	.total dq ?
}

struc Value tag, data
{
	.tag db tag
	align 8
	.data dq data
}

struc VecHeader
{
	.data dq ?
	.count dq ?
	.cap dq ?
}

struc StrBuf
{
	.len dq ?
}

virtual at 0
	_Block BlockHeader
	BLOCK_HEADER_SIZE = $
	_Heap Heap
	HEAP_SIZE = $
	_Value Value 0, 0
	VALUE_SIZE = $
	_Vec VecHeader
	VEC_HEADER_SIZE = $
	_StrBuf StrBuf
	STRBUF_HEADER_SIZE = $
end virtual

BLOCK_NEXT_OFF equ 0
BLOCK_SIZE_OFF equ 8
BLOCK_MARKED_OFF equ 16

HEAP_BLOCKS_OFF equ 0
HEAP_TOTAL_OFF equ 8

VALUE_TAG_OFF equ 0
VALUE_DATA_OFF equ 8

VEC_DATA_OFF equ 0
VEC_COUNT_OFF equ 8
VEC_CAP_OFF equ 16

STRBUF_LEN_OFF equ 0

macro payload_to_header ptr_reg, header_reg
{
	mov header_reg, ptr_reg
	sub header_reg, BLOCK_HEADER_SIZE
}

macro mark_block ptr_reg
{
	test ptr_reg, ptr_reg
	jz @f
	payload_to_header ptr_reg, rax
	mov byte [rax + BLOCK_MARKED_OFF], 1
@@:
}
