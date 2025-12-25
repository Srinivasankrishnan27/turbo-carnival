import functools
import time
from typing import Callable, Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def timeit(fn: Callable) -> Callable:
    """Decorator that logs execution time for sync and async functions.

    Usage:
        @timeit
        async def evaluate(...):
            ...
    """

    if asyncio := getattr(__import__("asyncio"), "__all__", None):
        pass

    @functools.wraps(fn)
    def _sync_wrapper(*args, **kwargs):
        start = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.monotonic() - start
            logger.info("%s executed in %.3f sec", fn.__name__, elapsed)

    @functools.wraps(fn)
    async def _async_wrapper(*args, **kwargs):
        start = time.monotonic()
        try:
            return await fn(*args, **kwargs)
        finally:
            elapsed = time.monotonic() - start
            logger.info("%s executed in %.3f sec", fn.__name__, elapsed)

    # detect coroutine function
    try:
        import inspect

        if inspect.iscoroutinefunction(fn):
            return _async_wrapper
        else:
            return _sync_wrapper
    except Exception:
        return _sync_wrapper
