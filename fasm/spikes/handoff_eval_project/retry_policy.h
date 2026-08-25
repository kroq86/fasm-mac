#ifndef RETRY_POLICY_H
#define RETRY_POLICY_H

enum retry_error { RETRY_PERMANENT = 0, RETRY_TRANSIENT = 1, RETRY_RATE_LIMIT = 2 };
unsigned retry_delay_ms(enum retry_error error, unsigned attempt);

#endif
