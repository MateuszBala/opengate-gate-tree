r"""Reading an angle into the range a question is asked in.

An angle is only defined up to whole turns, so the same direction can be
written as :math:`\pi/2` or as :math:`-3\pi/2`, and a calculation that answers
one of them is answering the other as well. Which of the two a plot, a
histogram or a comparison wants depends on the question, so the range is
something to be chosen rather than something to be lived with.

The general form reduces an angle by whole widths of the range it is being read
into:

.. math::

    \boxed{
    \theta' = \theta_{\min} +
              \left(\theta - \theta_{\min}\right) \bmod
              \left(\theta_{\max} - \theta_{\min}\right)
    }

so a range a full turn wide - ``[0, 2*pi)`` or ``[-pi, pi)`` - is the everyday
case of reading the same direction differently, and a narrower one identifies
angles that differ by its width: ``[0, pi)`` reads a direction and its reverse
as one, which is what an undirected line means.

The range is half-open, ``[angle_min, angle_max)``. A histogram is what these
are usually for, and an angle that could land in the first bin or the last one
depending on rounding is the artefact worth avoiding.

Public functions:

transform_to_angle_range(values, angle_min, angle_max) -> numpy.ndarray
    Read angles into any range.
wrap_to_two_pi(values) -> numpy.ndarray
    Read angles into ``[0, 2*pi)``, the range an azimuth is binned in.
wrap_to_signed_pi(values) -> numpy.ndarray
    Read angles into ``[-pi, pi)``, the range ``arctan2`` answers in.
"""

import math
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

# A full turn, which is what an angle short of one is measured against.
TWO_PI: Final[float] = 2.0 * np.pi

# Half a turn, the other end of the two everyday ranges.
PI: Final[float] = float(np.pi)


def transform_to_angle_range(
    values: ArrayLike,
    angle_min: float,
    angle_max: float,
) -> np.ndarray:
    r"""Read angles into the range ``[angle_min, angle_max)``.

    Every angle is reduced by whole widths of the range until it lies in it,
    which leaves it describing the same direction whenever the width is a whole
    turn, and identifies angles that differ by the width when it is narrower.

    Parameters
    ----------
    values : array_like
        Angles in radians, of any size and of any sign.
    angle_min : float
        Lower end of the range, which belongs to it.
    angle_max : float
        Upper end, which does not.

    Returns
    -------
    numpy.ndarray
        The same angles, in ``[angle_min, angle_max)``.

    Raises
    ------
    ValueError
        If either end is not a finite number, or the range is empty or reversed
        - a range that holds nothing has nothing to read an angle into.

    Examples
    --------
    >>> transform_to_angle_range([3.0 * np.pi], 0.0, 2.0 * np.pi) / np.pi
    array([1.])
    >>> transform_to_angle_range([1.5 * np.pi], -np.pi, np.pi) / np.pi
    array([-0.5])
    """
    low = _finite(angle_min, "angle_min")
    high = _finite(angle_max, "angle_max")
    if high <= low:
        raise ValueError(
            f"An angle is read into a range that holds something: got "
            f"[{low}, {high}), which is empty."
        )
    width = high - low
    angles = np.asarray(values, dtype=np.float64)
    turned = low + np.mod(angles - low, width)
    # An angle a hair below the lower end, reduced by a width, rounds to the
    # upper end - the one value the range excludes.
    reduced: np.ndarray = np.where(turned >= high, low, turned)
    return reduced


def wrap_to_two_pi(angles: ArrayLike) -> np.ndarray:
    """Read angles into ``[0, 2*pi)``.

    The range an azimuth is binned in. ``arctan2`` answers in ``(-pi, pi]``,
    which splits a half turn either side of zero and makes a histogram of
    azimuths read as two halves.

    Parameters
    ----------
    angles : array_like
        Angles in radians.

    Returns
    -------
    numpy.ndarray
        The same angles, in ``[0, 2*pi)``.
    """
    return transform_to_angle_range(angles, 0.0, TWO_PI)


def wrap_to_signed_pi(angles: ArrayLike) -> np.ndarray:
    """Read angles into ``[-pi, pi)``.

    The range ``arctan2`` answers in, and the one a difference of two angles
    is read in: a turn of ``3*pi/2`` one way is a turn of ``-pi/2`` the other,
    and this says the second.

    Parameters
    ----------
    angles : array_like
        Angles in radians.

    Returns
    -------
    numpy.ndarray
        The same angles, in ``[-pi, pi)``.
    """
    return transform_to_angle_range(angles, -PI, PI)


def _finite(value: float, name: str) -> float:
    """Return an end of a range, refusing one that describes no angle."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"'{name}' takes a finite angle in radians, got {number}.")
    return number
