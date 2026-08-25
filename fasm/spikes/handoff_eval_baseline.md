# Task handoff

Implement the retry delay policy. Permanent errors return zero. Transient
errors use `100 * 2^attempt` capped at 1600 ms. Rate-limit errors use
`1000 * 2^attempt` capped at 8000 ms. A previous header change was reverted, so
only `retry_policy.c` should be changed. Run `./check.sh` afterward. `local.env`
contains user-owned work and must not be changed.
