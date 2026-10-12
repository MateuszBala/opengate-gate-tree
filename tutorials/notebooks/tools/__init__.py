"""Helpers a tutorial would otherwise repeat in its first cell.

The notebooks import these by name, because a notebook runs with its own
directory on the path::

    from tools import environment, paths, plotting

Each module answers one question a tutorial always asks: where the files are,
what version of the package is installed, and what the plots should look like.
Anything that turns out to be repeated across notebooks belongs here too; a
helper that only one notebook needs belongs in that notebook, where it can be
read beside the work it does.
"""

from tools import environment, paths, plotting

__all__ = ["environment", "paths", "plotting"]
