import os
from contextlib import ExitStack, contextmanager


@contextmanager
def maybe_ray():
    with ExitStack() as stack:
        if "RAY_ADDRESS" in os.environ:
            import joblib
            from ray.util.joblib import register_ray

            register_ray()
            stack.enter_context(joblib.parallel_backend("ray"))
        yield
