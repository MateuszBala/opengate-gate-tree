"""Filters and selectors for the data of a "Hits" tree.

The functions here work on the pandas view of extracted data. They come in
pairs, and the pair is the same everywhere:

- ``is_<something>`` answers with a boolean column of the same length and the
  same index as its input, which is what combines with other conditions
  (``&``, ``|``) and what indexes other columns;
- the other name of the pair answers with the rows themselves, which is what
  chains.

Only what pandas has no name for is added. Comparing, ``isin`` and combining
masks already work on a column read from a GATE file, so the package does not
restate them; a filter earns its place by naming something the data means -
a closed range, a shape in space, the identity of an event, the meaning of a
code.

Public functions:

is_in_range(values, low, high, inclusive) -> pandas.Series
    Which values fall in a range.
in_range(values, low, high, inclusive) -> pandas.Series
    The values that fall in a range.
"""

from typing import Literal

import pandas as pd

# Which ends of a range belong to it, in the vocabulary of ``pandas.Series.between``.
InclusiveSide = Literal["both", "neither", "left", "right"]


def is_in_range(
    values: pd.Series,
    low: float,
    high: float,
    inclusive: InclusiveSide = "both",
) -> pd.Series:
    """Return which values fall in a range.

    Parameters
    ----------
    values : pandas.Series
        Column to test.
    low, high : float
        Ends of the range.
    inclusive : {"both", "neither", "left", "right"}
        Which ends belong to the range. The vocabulary is the one of
        :meth:`pandas.Series.between`, so a reader of pandas needs no second
        convention, and a value that is not a number falls outside the range
        the same way it does there.

    Returns
    -------
    pandas.Series
        Boolean column of the same length and index as ``values``.
    """
    return values.between(low, high, inclusive=inclusive)


def in_range(
    values: pd.Series,
    low: float,
    high: float,
    inclusive: InclusiveSide = "both",
) -> pd.Series:
    """Return the values that fall in a range.

    Parameters
    ----------
    values : pandas.Series
        Column to select from.
    low, high : float
        Ends of the range.
    inclusive : {"both", "neither", "left", "right"}
        Which ends belong to the range.

    Returns
    -------
    pandas.Series
        The values in the range, with the index they had.
    """
    return values[is_in_range(values, low, high, inclusive)]
