from pathlib import Path

import click


class PathPath(click.Path):
    """
    A Click path argument that returns a pathlib Path.
    """

    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))
