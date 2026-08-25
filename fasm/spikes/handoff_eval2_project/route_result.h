#ifndef ROUTE_RESULT_H
#define ROUTE_RESULT_H
enum route_outcome { ROUTE_OK, ROUTE_AUTH, ROUTE_THROTTLE, ROUTE_RETRY, ROUTE_EXHAUSTED };
enum route_outcome route_result(int status, unsigned attempt);
#endif
