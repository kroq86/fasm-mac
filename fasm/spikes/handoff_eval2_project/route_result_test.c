#include "route_result.h"
int main(void) {
    int failed = 0;
    failed |= route_result(200, 0) != ROUTE_OK;
    failed |= route_result(401, 0) != ROUTE_AUTH;
    failed |= route_result(403, 9) != ROUTE_AUTH;
    failed |= route_result(429, 0) != ROUTE_THROTTLE;
    failed |= route_result(500, 0) != ROUTE_RETRY;
    return failed;
}
