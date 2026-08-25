#include "route_result.h"
#include <assert.h>
int main(void) {
    assert(route_result(429, 2) == ROUTE_THROTTLE);
    assert(route_result(429, 3) == ROUTE_EXHAUSTED);
    assert(route_result(499, 99) == ROUTE_OK);
    assert(route_result(500, 1) == ROUTE_RETRY);
    assert(route_result(599, 2) == ROUTE_EXHAUSTED);
    assert(route_result(600, 0) == ROUTE_OK);
    return 0;
}
