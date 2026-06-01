; LeetCode 121. Best Time to Buy and Sell Stock
; https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
; prices=[7,1,5,3,6,4]  =>  5

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"
include "fasm/core/dp.inc"

entry start

start:
	lea	rdi, [prices]
	mov	rsi, prices_count
	call	dp_max_profit_one_trade
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

prices dq 7, 1, 5, 3, 6, 4
prices_count = 6

include "fasm/core/runtime_bss.inc"
runtime_print_bss
