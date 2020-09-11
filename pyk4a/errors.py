from .results import Result


class K4AException(Exception):
    pass


class K4ATimeoutException(K4AException):
    pass


def _verify_error(res: int):
    """
    Validate k4a_module result
    """
    res = Result(res)
    if res == Result.Failed:
        raise K4AException()
    elif res == Result.Timeout:
        raise K4ATimeoutException()
