import time
from functools import wraps


def timeit(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        evaluator_name = "Unknown"
        if args and hasattr(args[0], "method_name"):
            evaluator_name = args[0].method_name

        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Time taken by {evaluator_name}.{func.__name__}: {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper
