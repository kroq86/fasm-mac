# Task handoff

The task originally targeted `retry_route.c`, used a five-attempt rule for 429,
and failed; that implementation was reverted. The file was then renamed to
`route_result.c`. The current compile-only preflight passes, although behavioral
tests still fail. The latest rules supersede the old ones: 429 is THROTTLE for
attempts 0–2 then EXHAUSTED; 500–599 is RETRY for attempts 0–1 then EXHAUSTED;
401/403 is AUTH; everything else is OK. `route_result.h` is untracked,
user-owned work and must be preserved. Edit only `route_result.c`, do not revive
`retry_route.c`, and run `./check.sh`.
