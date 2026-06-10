; Structured control flow macros for fasm1.
;
; Inspired by fasmg packages/x86/include/macro/if.inc (Tomasz Grysztar).
; Adapted for fasm1: uses \{ \} inside macro bodies, no calminstruction/else match.
;
; ------------------------------------------------------------
; _if jXX / _else / _endif
;
;   cmp rax, rbx
;   _if jl              ; if rax < rbx (signed)
;       ...
;   _else               ; optional
;       ...
;   _endif
;
; jXX is the jump that would take the TRUE branch (je, jne, jl, jg,
; jle, jge, jb, ja, jbe, jae).  The macro emits the inverted jump.
; ------------------------------------------------------------

macro _if jXX {
	local _if_else, _if_end
	__IF_ELSE equ _if_else
	__IF_END  equ _if_end
	match =je,  jXX \{ jne _if_else \}
	match =jne, jXX \{ je  _if_else \}
	match =jl,  jXX \{ jge _if_else \}
	match =jg,  jXX \{ jle _if_else \}
	match =jle, jXX \{ jg  _if_else \}
	match =jge, jXX \{ jl  _if_else \}
	match =jb,  jXX \{ jae _if_else \}
	match =ja,  jXX \{ jbe _if_else \}
	match =jbe, jXX \{ ja  _if_else \}
	match =jae, jXX \{ jb  _if_else \}
}

macro _else {
	jmp __IF_END
	__IF_ELSE:
	restore __IF_ELSE
	local _if_else2
	__IF_ELSE equ _if_else2
}

macro _endif {
	__IF_ELSE:
	__IF_END:
	restore __IF_ELSE
	restore __IF_END
}

; ------------------------------------------------------------
; _while / _endw
;
;   _while                  ; opens loop; __WS and __WE are live
;       cmp rax, 0
;       jle __WE            ; user writes the exit condition
;       dec rax
;   _endw
;
; __WS = loop-start label, __WE = loop-end label.
; Nesting works: each _while/_endw pair pushes/pops its own labels.
; ------------------------------------------------------------

macro _while {
	local ws, we
	__WS equ ws
	__WE equ we
	ws:
}

macro _endw {
	jmp __WS
	__WE:
	restore __WS
	restore __WE
}

; ------------------------------------------------------------
; _repeat / _until reg, jXX, val
;
;   _repeat
;       dec rcx
;   _until rcx, jg, 0       ; continue while rcx > 0
;
; jXX is the "keep looping" condition (jg = loop again if greater).
; The macro emits  cmp reg, val  then the loop-back jump.
; ------------------------------------------------------------

macro _repeat {
	local rs
	__RS equ rs
	rs:
}

macro _until reg, jXX, val {
	cmp reg, val
	match =jl,  jXX \{ jl  __RS \}
	match =jg,  jXX \{ jg  __RS \}
	match =je,  jXX \{ je  __RS \}
	match =jne, jXX \{ jne __RS \}
	match =jle, jXX \{ jle __RS \}
	match =jge, jXX \{ jge __RS \}
	match =jb,  jXX \{ jb  __RS \}
	match =ja,  jXX \{ ja  __RS \}
	match =jbe, jXX \{ jbe __RS \}
	match =jae, jXX \{ jae __RS \}
	restore __RS
}
