r"""Angles between directions and between the planes two directions span.

The scattering plane of a photon is spanned by where it was going and where it
went, and its orientation is carried by the normal
:math:`\hat{n} = \widehat{\hat{k_0} \times \hat{k}}`. The angle between two
such planes, which is the quantity a coincidence measurement is about, is the
angle between their normals.

Every angle here is in radians and lies in ``[0, pi]``, as ``arccos`` answers;
:func:`~opengate_gate_tree.units.rad_to_deg` reads it in degrees. Lengths do
not matter - each input is normalised first - so directions read straight from
a tree can be handed over as they are.

Two directions that are parallel span no plane, and there is no angle between
a plane and something that is not one: those rows come back as ``nan`` and are
reported, following the rule of
:func:`~opengate_gate_tree.geometry.vectors.normalize`.

Public functions:

angle_between(left, right) -> numpy.ndarray
    Angle between two directions.
plane_normal(before, after) -> numpy.ndarray
    Normal of the plane two directions span.
angle_between_normals(left, right) -> numpy.ndarray
    Angle between two planes given by their normals.
angle_between_planes(before_a, after_a, before_b, after_b) -> numpy.ndarray
    Angle between two planes given by the directions that span them.
"""

import numpy as np
from numpy.typing import ArrayLike

from opengate_gate_tree.geometry.vectors import cross, dot, normalize


def angle_between(left: ArrayLike, right: ArrayLike) -> np.ndarray:
    r"""Return the angle between two directions, row by row.

    :math:`\theta = \arccos(\hat{u} \cdot \hat{v})`, in ``[0, pi]``. This is
    the scattering angle when the two are the directions before and after an
    interaction.

    It is computed as :math:`\operatorname{atan2}
    (\lVert \hat{u} \times \hat{v} \rVert, \hat{u} \cdot \hat{v})`, which is
    the same angle and keeps its accuracy where the cosine loses it: near zero
    and near :math:`\pi`.

    Parameters
    ----------
    left, right : array_like
        Directions of shape ``(N, 3)``, of any length.

    Returns
    -------
    numpy.ndarray
        Angles in radians, of shape ``(N,)``, with ``nan`` where either
        direction has no length.

    Raises
    ------
    ValueError
        If either input does not hold vectors, or the two hold different
        numbers of them.

    Examples
    --------
    >>> angle_between([[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    array([1.57079633])
    """
    first = normalize(left)
    second = normalize(right)
    # arctan2 of the two products rather than arccos of one of them. The two
    # agree exactly in arithmetic, and differ where it matters: a cosine
    # carries a small angle in its last bits, so arccos answers 0 for an angle
    # of 1e-9 rad, while this form is accurate at both ends of the range.
    angles: np.ndarray = np.arctan2(
        np.linalg.norm(cross(first, second), axis=-1), dot(first, second)
    )
    return angles


def plane_normal(before: ArrayLike, after: ArrayLike) -> np.ndarray:
    r"""Return the normal of the plane two directions span.

    :math:`\hat{n} = \widehat{\hat{k_0} \times \hat{k}}`. The plane is what the
    two directions lie in; the normal is how its orientation is carried around,
    since a plane through the origin is exactly the directions perpendicular to
    one vector.

    Parameters
    ----------
    before, after : array_like
        Directions of shape ``(N, 3)``, of any length - typically where a
        particle was going and where it went.

    Returns
    -------
    numpy.ndarray
        Unit normals of shape ``(N, 3)``, with ``nan`` in every row where the
        two directions are parallel and span no plane.

    Raises
    ------
    ValueError
        If either input does not hold vectors, or the two hold different
        numbers of them.
    """
    return normalize(cross(normalize(before), normalize(after)), name="plane normal")


def angle_between_normals(left: ArrayLike, right: ArrayLike) -> np.ndarray:
    r"""Return the angle between two planes given by their normals.

    :math:`\phi = \arccos(\hat{n}_a \cdot \hat{n}_b)`, in ``[0, pi]``.

    Parameters
    ----------
    left, right : array_like
        Normals of shape ``(N, 3)``, of any length.

    Returns
    -------
    numpy.ndarray
        Angles in radians, of shape ``(N,)``.

    Raises
    ------
    ValueError
        If either input does not hold vectors, or the two hold different
        numbers of them.
    """
    return angle_between(left, right)


def angle_between_planes(
    before_a: ArrayLike,
    after_a: ArrayLike,
    before_b: ArrayLike,
    after_b: ArrayLike,
) -> np.ndarray:
    r"""Return the angle between the planes two pairs of directions span.

    The planes are built with :func:`plane_normal` and compared with
    :func:`angle_between_normals`, which is the quantity a measurement of two
    correlated scatterings is about.

    A normal points to one side of its plane, and which side follows from the
    order of the two directions that spanned it: reading a pair backwards
    answers ``pi`` minus the angle. Give both pairs in the same order - before,
    then after - and the answer is the one the physics asks for.

    Parameters
    ----------
    before_a, after_a : array_like
        Directions before and after the first interaction, shape ``(N, 3)``.
    before_b, after_b : array_like
        The same for the second one.

    Returns
    -------
    numpy.ndarray
        Angles in radians, of shape ``(N,)``, with ``nan`` where either pair
        spans no plane.

    Raises
    ------
    ValueError
        If an input does not hold vectors, or the four hold different numbers
        of them.
    """
    return angle_between_normals(
        plane_normal(before_a, after_a),
        plane_normal(before_b, after_b),
    )
