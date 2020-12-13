from functools import wraps
from pathlib import Path

import click


class PathPath(click.Path):
    """
    A Click path argument that returns a pathlib Path.
    """

    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


def save_options(options, cb=lambda args, kwargs, inner: inner(*args, **kwargs)):
    def inner(wrapped):
        @wraps(wrapped)
        def wrapper(*args, **kwargs):
            cb(args, kwargs, wrapped)

        for option in options:
            wrapper = option(wrapper)
        return wrapper

    return inner
