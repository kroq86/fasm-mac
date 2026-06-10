; for / endfor — register-based counted loop.
;
;   for rbx, 0, 5      ; rbx = 0; rbx < 5; rbx++
;       mov rdi, [arr + rbx*8]
;       ...
;   endfor rbx
;
; Nesting works as long as each level uses a distinct register.
; The counter register is live inside the body and usable as an index.

macro for reg, low, upper {
	mov	reg, low
.#reg#_again:
	cmp	reg, upper
	jge	.#reg#_over
}

macro endfor reg {
	inc	reg
	jmp	.#reg#_again
.#reg#_over:
}
