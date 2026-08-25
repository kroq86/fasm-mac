#include "retry_policy.h"
int main(void) {
    int failed = 0;
    failed |= retry_delay_ms(RETRY_PERMANENT, 0) != 0;
    failed |= retry_delay_ms(RETRY_TRANSIENT, 0) != 100;
    failed |= retry_delay_ms(RETRY_TRANSIENT, 2) != 400;
    failed |= retry_delay_ms(RETRY_RATE_LIMIT, 0) != 1000;
    failed |= retry_delay_ms(RETRY_RATE_LIMIT, 2) != 4000;
    return failed;
}
