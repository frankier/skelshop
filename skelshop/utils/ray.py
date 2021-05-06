import logging
import os
from contextlib import ExitStack, contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def maybe_ray():
    with ExitStack() as stack:
        if "RAY_ADDRESS" in os.environ:
            import joblib
            from ray.util.joblib import register_ray

            logger.debug(
                "Using RAY_ADDRESS=%s as joblib backend", os.environ["RAY_ADDRESS"]
            )

            register_ray()
            stack.enter_context(joblib.parallel_backend("ray"))
        yield
