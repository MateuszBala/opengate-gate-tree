r"""Vectorised operations on whole columns of vectors.

Every function here works on an array of shape ``(N, 3)``, where ``N`` is the
number of vectors and the last axis holds the components ``(x, y, z)``. Rows
are computed together, with no Python loop, which is what makes a question
about half a million hits answerable.

The shape is the contract, and it is checked: a single vector of shape ``(3,)``
would otherwise be read as three vectors of one component each, and the answer
would look reasonable.

A vector of zero length has no direction, so a question about its direction has
no answer. Such a row comes back as ``nan`` and the package reports how many
rows it did that for, rather than refusing the whole column - the same rule
:func:`~opengate_gate_tree.tree.hits.positronium.decode_positronium_column`
follows, and for the same reason: one row of half a million must not end an
analysis.

Public functions:

as_vectors(values) -> numpy.ndarray
    Read anything shaped like vectors as an ``(N, 3)`` array of ``float64``.
ensure_vectors(array) -> None
    Refuse an array that does not hold vectors.
norm(vectors) -> numpy.ndarray
    Length of every vector.
normalize(vectors) -> numpy.ndarray
    The same vectors, of unit length.
dot(left, right) -> numpy.ndarray
    Scalar product, row by row.
cross(left, right) -> numpy.ndarray
    Vector product, row by row.
clip_cosine(values) -> numpy.ndarray
    Cosines pulled back into the domain of ``arccos``.
wrap_to_two_pi(angles) -> numpy.ndarray
    Angles moved into ``[0, 2*pi)``.
"""

from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from opengate_gate_tree.logger import log

# Below this length a vector counts as zero: it has no direction to speak of.
# The value is the one the reference implementation of these calculations uses.
MIN_NORM: Final[float] = 1e-12

# How many components a vector has here. Positions and momentum directions in
# a GATE tree are three-dimensional, and so is everything built from them.
VECTOR_DIMENSION: Final[int] = 3

# The domain of arccos. A scalar product of two unit vectors lands just outside
# it often enough that every angle has to be pulled back in first.
COS_MIN: Final[float] = -1.0
COS_MAX: Final[float] = 1.0

# A full turn, which is what an angle short of one is measured against.
TWO_PI: Final[float] = 2.0 * np.pi


def as_vectors(values: ArrayLike) -> np.ndarray:
    """Read anything shaped like vectors as an ``(N, 3)`` array of ``float64``.

    Takes what a frame, a list or an array offers and answers with the one
    shape the rest of the module works on. The columns of a GATE file are
    ``float32``; the computation is done in ``float64``, because a scalar
    product of two nearly parallel unit vectors loses about three significant
    digits before it reaches ``arccos``.

    Parameters
    ----------
    values : array_like
        Vectors, as an array, a data frame of three columns, or a sequence of
        triples.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 3)`` and dtype ``float64``.

    Raises
    ------
    ValueError
        If what was given does not hold three components per vector.

    Examples
    --------
    >>> as_vectors([[3.0, 4.0, 0.0]])
    array([[3., 4., 0.]])
    """
    array = np.asarray(values, dtype=np.float64)
    ensure_vectors(array)
    return array


def ensure_vectors(array: np.ndarray) -> None:
    """Refuse an array that does not hold vectors.

    Parameters
    ----------
    array : numpy.ndarray
        Array to check.

    Raises
    ------
    ValueError
        If the array is not two-dimensional, or its last axis does not hold
        three components. A single vector of shape ``(3,)`` is refused here
        rather than read as three vectors of one component, which is what the
        arithmetic below would otherwise do.
    """
    if array.ndim != 2 or array.shape[1] != VECTOR_DIMENSION:
        raise ValueError(
            f"Vectors are given as an array of shape (N, {VECTOR_DIMENSION}), "
            f"got shape {array.shape}."
        )


def norm(vectors: ArrayLike) -> np.ndarray:
    """Return the length of every vector.

    Parameters
    ----------
    vectors : array_like
        Vectors of shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Lengths, of shape ``(N,)``.

    Raises
    ------
    ValueError
        If the input does not hold vectors.
    """
    lengths: np.ndarray = np.linalg.norm(as_vectors(vectors), axis=-1)
    return lengths


def normalize(vectors: ArrayLike, name: str = "vector") -> np.ndarray:
    """Return the same vectors, of unit length.

    Parameters
    ----------
    vectors : array_like
        Vectors of shape ``(N, 3)``.
    name : str
        What the vectors are, used in the report about the rows that have no
        direction.

    Returns
    -------
    numpy.ndarray
        Vectors of shape ``(N, 3)`` and unit length, with ``nan`` in every row
        whose length was below :data:`MIN_NORM`.

    Raises
    ------
    ValueError
        If the input does not hold vectors.

    Examples
    --------
    >>> normalize([[3.0, 4.0, 0.0]])
    array([[0.6, 0.8, 0. ]])
    """
    array = as_vectors(vectors)
    lengths = np.linalg.norm(array, axis=-1, keepdims=True)
    undirected = lengths < MIN_NORM
    _report_undirected(undirected, name)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(undirected, np.nan, array / lengths)
    return unit


def dot(left: ArrayLike, right: ArrayLike) -> np.ndarray:
    """Return the scalar product of two columns of vectors, row by row.

    Parameters
    ----------
    left, right : array_like
        Vectors of shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Scalar products, of shape ``(N,)``.

    Raises
    ------
    ValueError
        If either input does not hold vectors, or the two hold different
        numbers of them.
    """
    first, second = _paired(left, right)
    products: np.ndarray = np.sum(first * second, axis=-1)
    return products


def cross(left: ArrayLike, right: ArrayLike) -> np.ndarray:
    """Return the vector product of two columns of vectors, row by row.

    Parameters
    ----------
    left, right : array_like
        Vectors of shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Vector products, of shape ``(N, 3)``. The result is zero where the two
        vectors are parallel, which is what makes a plane through them
        undefined.

    Raises
    ------
    ValueError
        If either input does not hold vectors, or the two hold different
        numbers of them.
    """
    first, second = _paired(left, right)
    products: np.ndarray = np.cross(first, second)
    return products


def clip_cosine(values: ArrayLike) -> np.ndarray:
    """Return cosines pulled back into the domain of ``arccos``.

    The scalar product of two unit vectors is a cosine in exact arithmetic and
    something like ``1.0000000000000002`` in this one. Left alone, that answers
    ``nan`` for the one angle everybody checks first: the angle between a
    vector and itself.

    Parameters
    ----------
    values : array_like
        Cosines of angles.

    Returns
    -------
    numpy.ndarray
        The same values, held within ``[-1, 1]``.
    """
    clipped: np.ndarray = np.clip(np.asarray(values, dtype=np.float64), COS_MIN, COS_MAX)
    return clipped


def wrap_to_two_pi(angles: ArrayLike) -> np.ndarray:
    """Return angles moved into ``[0, 2*pi)``.

    ``arctan2`` answers in ``(-pi, pi]``, which splits a half turn either side
    of zero and makes a histogram of azimuths read as two halves. Adding a full
    turn to the negative ones puts them in the range an analysis expects.

    Parameters
    ----------
    angles : array_like
        Angles in radians, usually straight from ``arctan2``.

    Returns
    -------
    numpy.ndarray
        The same angles, in ``[0, 2*pi)``.
    """
    values = np.asarray(angles, dtype=np.float64)
    wrapped: np.ndarray = np.where(values < 0.0, values + TWO_PI, values)
    return wrapped


def _paired(left: ArrayLike, right: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return two columns of vectors, refusing a pair that cannot be zipped."""
    first = as_vectors(left)
    second = as_vectors(right)
    if len(first) != len(second):
        raise ValueError(
            f"Two columns of vectors are compared row by row, so they hold as many rows "
            f"each: got {len(first)} and {len(second)}."
        )
    return first, second


def _report_undirected(undirected: np.ndarray, name: str) -> None:
    """Report the rows that hold no direction, without refusing the column."""
    count = int(np.count_nonzero(undirected))
    if count:
        log().warning(
            "%d of %d %s values have no length, so they have no direction. "
            "Those rows are read as nothing at all.",
            count,
            len(undirected),
            name,
        )
