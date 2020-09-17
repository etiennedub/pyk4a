from enum import IntEnum


# k4a_wait_result_t
class Result(IntEnum):
    Success = 0
    Failed = 1
    Timeout = 2


# k4a_buffer_result_t
class BufferResult(IntEnum):
    Success = 0
    Failed = 1
    TooSmall = 2


# k4a_stream_result_t
class StreamResult(IntEnum):
    Success = 0
    Failed = 1
    EOF = 2
