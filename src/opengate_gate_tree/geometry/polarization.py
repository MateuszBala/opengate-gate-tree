r"""Reconstruction of the polarization direction of a scattered photon.

Compton scattering happens most readily perpendicular to the polarization of
the incoming photon, so the plane a photon scattered in carries what can be
known about that polarization: the estimate is the normal of the scattering
plane,

.. math::

    \hat{\varepsilon} = \frac{\hat{k_0} \times \hat{k}}
                             {\lVert \hat{k_0} \times \hat{k} \rVert}

with :math:`\hat{k_0}` where the photon was going and :math:`\hat{k}` where it
went. This is the same computation as
:func:`~opengate_gate_tree.geometry.angles.plane_normal`, under the name of the
quantity it estimates.

It is an estimate per photon, not a measurement: a single scattering fixes the
plane, and the polarization is only distributed around its normal. What the
estimate is good for is a distribution over many photons.

The estimate is defined on momentum directions and takes nothing else. Where
those directions come from is a separate question: a "Hits" tree carries the
direction after an interaction in ``momDir``, and the direction a photon
arrived with is rebuilt from positions by
:func:`~opengate_gate_tree.geometry.momentum.momentum_direction_from_positions`.

Public functions:

polarization_direction(before, after) -> numpy.ndarray
    Polarization estimated from the directions before and after a scattering.
"""

import numpy as np
from numpy.typing import ArrayLike

from opengate_gate_tree.geometry.angles import plane_normal


def polarization_direction(before: ArrayLike, after: ArrayLike) -> np.ndarray:
    r"""Return the polarization estimated from a scattering.

    :math:`\hat{\varepsilon} = \widehat{\hat{k_0} \times \hat{k}}`, the normal
    of the scattering plane.

    The two directions may be given in either length; only their directions
    reach the answer. Giving them the other way round reverses the sense of the
    result, which is why the sense carries nothing: the estimate is a line in
    space, not an arrow along it.

    Parameters
    ----------
    before : array_like
        Directions before the scattering, of shape ``(N, 3)``.
    after : array_like
        Directions after it, of shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Unit vectors of shape ``(N, 3)``, with ``nan`` in every row where the
        photon carried straight on and scattered in no particular plane.

    Raises
    ------
    ValueError
        If either input does not hold vectors, or the two hold different
        numbers of them.

    Examples
    --------
    >>> polarization_direction([[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    array([[0., 0., 1.]])
    """
    return plane_normal(before, after)
