#include "retry_policy.h"
#include <assert.h>

int main(void) {
    assert(retry_delay_ms(RETRY_PERMANENT, 99) == 0);
    assert(retry_delay_ms(RETRY_TRANSIENT, 4) == 1600);
    assert(retry_delay_ms(RETRY_TRANSIENT, 99) == 1600);
    assert(retry_delay_ms(RETRY_RATE_LIMIT, 3) == 8000);
    assert(retry_delay_ms(RETRY_RATE_LIMIT, 99) == 8000);
    return 0;
}
