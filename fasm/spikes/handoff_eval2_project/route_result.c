#include "route_result.h"
enum route_outcome route_result(int status, unsigned attempt) {
    (void)status;
    (void)attempt;
    return ROUTE_OK;
}
