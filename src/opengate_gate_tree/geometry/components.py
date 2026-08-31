r"""Reading a vector in the terms a question is asked in.

Two ways of taking a vector apart, both of them everyday work on hit data:

- along a direction and across it. A vector splits into the part that lies
  along an axis and the part that is left, :math:`v = v_\parallel + v_\perp`,
  which is how a component along a polarization, a beam or the axis of a
  scanner is read off.
- in spherical components. A direction is a radius, a polar angle from the
  ``z`` axis and an azimuth around it, which is the form an angular
  distribution is binned in.

The conventions are the physical ones and are stated here rather than left to
be inferred: the polar angle is measured from ``z`` and lies in ``[0, pi]``,
the azimuth is ``atan2(y, x)`` moved into ``[0, 2*pi)``. The other convention
in circulation swaps the names of the two angles.

Public functions:

parallel_component(vectors, axis) -> numpy.ndarray
    The part of every vector that lies along an axis.
perpendicular_component(vectors, axis) -> numpy.ndarray
    The part that is left.
spherical_components(vectors) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    Radius, polar angle and azimuth of every vector.
"""

import numpy as np
from numpy.typing import ArrayLike

from opengate_gate_tree.geometry.vectors import (
    MIN_NORM,
    VECTOR_DIMENSION,
    as_vectors,
    clip_cosine,
    ensure_vectors,
    normalize,
    wrap_to_two_pi,
)


def parallel_component(vectors: ArrayLike, axis: ArrayLike) -> np.ndarray:
    r"""Return the part of every vector that lies along an axis.

    :math:`v_\parallel = (\hat{a} \cdot v)\,\hat{a}`, so only the direction of
    the axis matters: an axis of any length gives the same answer.

    Parameters
    ----------
    vectors : array_like
        Vectors of shape ``(N, 3)``.
    axis : array_like
        The direction to read along: one vector of shape ``(3,)`` for the whole
        column, or ``(N, 3)`` for one per row.

    Returns
    -------
    numpy.ndarray
        Vectors of shape ``(N, 3)``, with ``nan`` in every row whose axis has
        no direction.

    Raises
    ------
    ValueError
        If the vectors or the axis are not shaped as vectors, or the axis
        gives neither one direction nor one per row.

    Examples
    --------
    >>> parallel_component([[1.0, 2.0, 3.0]], [0.0, 0.0, 1.0])
    array([[0., 0., 3.]])
    """
    array = as_vectors(vectors)
    direction = normalize(_as_axis(axis, len(array)), name="axis")
    along = np.sum(array * direction, axis=-1, keepdims=True)
    projection: np.ndarray = along * direction
    return projection


def perpendicular_component(vectors: ArrayLike, axis: ArrayLike) -> np.ndarray:
    r"""Return the part of every vector that is left across an axis.

    :math:`v_\perp = v - v_\parallel`, so the two parts add back up to the
    vector they came from.

    Parameters
    ----------
    vectors : array_like
        Vectors of shape ``(N, 3)``.
    axis : array_like
        The direction to read across: one vector of shape ``(3,)`` for the
        whole column, or ``(N, 3)`` for one per row.

    Returns
    -------
    numpy.ndarray
        Vectors of shape ``(N, 3)``, with ``nan`` in every row whose axis has
        no direction.

    Raises
    ------
    ValueError
        If the vectors or the axis are not shaped as vectors, or the axis
        gives neither one direction nor one per row.

    Examples
    --------
    >>> perpendicular_component([[1.0, 2.0, 3.0]], [0.0, 0.0, 1.0])
    array([[1., 2., 0.]])
    """
    array = as_vectors(vectors)
    across: np.ndarray = array - parallel_component(array, axis)
    return across


def spherical_components(vectors: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return the radius, polar angle and azimuth of every vector.

    :math:`r = \lVert v \rVert`, :math:`\theta = \arccos(v_z / r)` measured
    from the ``z`` axis and lying in ``[0, \pi]``, and
    :math:`\varphi = \mathrm{atan2}(v_y, v_x)` moved into ``[0, 2\pi)``.

    The three are answered together because all three come from the same
    radius, and asking for one of them alone would compute it twice.

    Parameters
    ----------
    vectors : array_like
        Vectors of shape ``(N, 3)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Radius, polar angle and azimuth, each of shape ``(N,)``. A vector of no
        length has a radius of zero and no angles, so both angles are ``nan``
        there. On the ``z`` axis itself the azimuth is zero, which is what
        ``atan2`` answers rather than a statement about the data.

    Raises
    ------
    ValueError
        If the input does not hold vectors.
    """
    array = as_vectors(vectors)
    radius = np.linalg.norm(array, axis=-1)
    undirected = radius < MIN_NORM
    with np.errstate(invalid="ignore", divide="ignore"):
        polar = np.where(undirected, np.nan, np.arccos(clip_cosine(array[:, 2] / radius)))
    azimuth = np.where(undirected, np.nan, wrap_to_two_pi(np.arctan2(array[:, 1], array[:, 0])))
    return radius, polar, azimuth


def _as_axis(axis: ArrayLike, count: int) -> np.ndarray:
    """Return an axis as one direction per row, however it was given.

    One direction for a whole column is the usual case - the axis of a
    scanner, the direction of a beam - so a single vector is spread over the
    rows rather than refused.
    """
    array = np.asarray(axis, dtype=np.float64)
    if array.shape == (VECTOR_DIMENSION,):
        return np.broadcast_to(array, (count, VECTOR_DIMENSION))
    ensure_vectors(array)
    if len(array) != count:
        raise ValueError(
            f"An axis is one direction for the whole column or one per row: "
            f"got {len(array)} for {count} row(s)."
        )
    return array
