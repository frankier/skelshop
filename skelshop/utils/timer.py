import logging
from time import perf_counter_ns

logger = logging.getLogger(__name__)


class Timer:
    def __init__(self, name="task", logger=logger):
        self.name = name
        self.logger = logger

    def __enter__(self):
        if self.logger.isEnabledFor(logging.INFO):
            self.start = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger.isEnabledFor(logging.INFO):
            end = perf_counter_ns()
            self.logger.info(
                "[Time] {} consumes {:.4f} s".format(
                    self.name, (end - self.start) * (10 ** -9)
                )
            )
