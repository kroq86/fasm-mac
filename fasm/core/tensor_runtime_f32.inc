; Minimal dense f32 tensor runtime ABI for x86_64 System V / C callers.
;
; TensorView (32 bytes): data*, grad*, rows:u64, cols:u64.
; TapeNode (32 bytes): op/lhs/rhs/flags:u32, TensorView*, arena slot/gen:u32.
; Parent indices precede consumers; UINT32_MAX means no parent. Backward kernels
; accumulate gradients. The caller seeds the final output gradient.
;
; Return codes: 0 success, -1 invalid argument, -2 malformed/overflow,
; -4 stale lifetime token.

TENSOR_F32_DATA equ 0
TENSOR_F32_GRAD equ 8
TENSOR_F32_ROWS equ 16
TENSOR_F32_COLS equ 24
TENSOR_F32_SIZE equ 32

TENSOR_NODE_OP equ 0
TENSOR_NODE_LHS equ 4
TENSOR_NODE_RHS equ 8
TENSOR_NODE_FLAGS equ 12
TENSOR_NODE_TENSOR equ 16
TENSOR_NODE_ARENA_SLOT equ 24
TENSOR_NODE_GENERATION equ 28
TENSOR_NODE_SIZE equ 32

TENSOR_INDEX_NONE equ 0ffffffffh
TENSOR_OP_LEAF equ 0
TENSOR_OP_MATMUL equ 1
TENSOR_OP_RELU equ 2
TENSOR_OP_MSE equ 3
TENSOR_OP_BIAS_ADD equ 4

TENSOR_FLAG_PARAMETER equ 1
TENSOR_FLAG_CONSTANT equ 2
TENSOR_FLAG_INPUT equ 4
TENSOR_FLAG_TEMPORARY equ 8
