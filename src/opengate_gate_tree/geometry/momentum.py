r"""Momentum directions rebuilt from the places a particle was.

A detector records where something happened, not where the particle was going.
The direction of flight between two of those places is the step from one to
the other,

.. math::

    \hat{k} = \frac{p_{\mathrm{end}} - p_{\mathrm{start}}}
                   {\lVert p_{\mathrm{end}} - p_{\mathrm{start}} \rVert}

which is how a direction is reconstructed from data throughout detector
analysis: from the source to the first interaction, from one interaction to the
next, from a decay point to a detected hit.

A "Hits" tree does carry ``momDirX``, ``momDirY`` and ``momDirZ``, but they hold
the direction **after** the interaction at that hit. The direction a particle
arrived with is not in the file and is rebuilt here.

Positions are read in ``float64`` before the step is taken. A GATE file writes
them as ``float32``, and subtracting two coordinates around 200 mm cancels away
several of the digits a ``float32`` was carrying.

Public functions:

momentum_direction_from_positions(start, end) -> numpy.ndarray
    Direction of flight from one place to the next.
"""

import numpy as np
from numpy.typing import ArrayLike

from opengate_gate_tree.geometry.vectors import as_vectors, normalize


def momentum_direction_from_positions(start: ArrayLike, end: ArrayLike) -> np.ndarray:
    r"""Return the direction of flight from one place to the next.

    Parameters
    ----------
    start : array_like
        Where the particle set off, of shape ``(N, 3)`` - a source position, or
        the position of the interaction it is leaving.
    end : array_like
        Where it arrived, of shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Unit vectors of shape ``(N, 3)``, with ``nan`` in every row where the
        two places are the same and there is no direction between them.

    Raises
    ------
    ValueError
        If either input does not hold vectors, or the two hold different
        numbers of them.

    Examples
    --------
    >>> momentum_direction_from_positions([[0.0, 0.0, 0.0]], [[0.0, 5.0, 0.0]])
    array([[0., 1., 0.]])
    """
    departure = as_vectors(start)
    arrival = as_vectors(end)
    if len(departure) != len(arrival):
        raise ValueError(
            f"A direction runs from one place to another, so both are given for as many "
            f"rows each: got {len(departure)} and {len(arrival)}."
        )
    return normalize(arrival - departure, name="momentum direction")
