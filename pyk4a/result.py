from enum import IntEnum


# k4a_wait_result_t
class Result(IntEnum):
    Success = 0
    Failed = 1
    Timeout = 2
