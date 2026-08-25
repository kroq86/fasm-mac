#include "retry_policy.h"

unsigned retry_delay_ms(enum retry_error error, unsigned attempt) {
    (void)error;
    (void)attempt;
    return 0;
}
