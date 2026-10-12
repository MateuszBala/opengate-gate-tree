"""What a tutorial was run against.

The header of a notebook states the version of the package it was written for.
That line is written by hand and ages the way hand-written lines do, so a
notebook prints what is actually installed beside it: a reader can then see at
once whether the two agree.

Public functions:

describe() -> str
    A few lines naming the package version, Python, and today's date.
package_version() -> str
    The version of the package the notebook is running against.
"""

import sys
from datetime import date
from importlib.metadata import PackageNotFoundError, version

PACKAGE = "opengate-gate-tree"


def package_version() -> str:
    """Return the version of the package this notebook is running against.

    Returns
    -------
    str
        The installed version, or ``"not installed"`` when the package is not
        in the environment - which is worth reading as an answer rather than as
        an exception, since it names what is wrong.
    """
    try:
        return version(PACKAGE)
    except PackageNotFoundError:
        return "not installed"


def describe() -> str:
    """Return a few lines naming what this notebook is running against.

    Returns
    -------
    str
        The package version, the Python version and today's date, one per line.

    Examples
    --------
    >>> describe().splitlines()[1].startswith("python")
    True
    """
    interpreter = ".".join(str(number) for number in sys.version_info[:3])
    return "\n".join(
        [
            f"{PACKAGE}: {package_version()}",
            f"python: {interpreter}",
            f"run on: {date.today().isoformat()}",
        ]
    )
