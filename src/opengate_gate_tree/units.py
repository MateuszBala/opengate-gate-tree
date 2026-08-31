"""Conversions between the units GATE writes and the ones an analysis uses.

GATE writes energy in MeV, length in mm and time in s. An analysis rarely
wants all three: energies are read in keV, a detector is described in cm or m,
and the resolution of a scanner is quoted in ns. The functions here convert
between those units and nothing else - they multiply or divide by one number,
and that number is written down once.

Names read as the conversion they make, ``<from>_to_<to>``. The symbol of a
unit keeps its capital letters, because they carry meaning: ``MeV`` and ``meV``
are different units, so ``mev_to_kev`` would name a different conversion.

Every conversion answers with the type it was given - a number for a number, an
array for an array, a column for a column, keeping its index and its name - and
computes in ``float64``. A GATE file holds ``float32`` columns, and a time of
one minute read in nanoseconds does not fit the 24 bits of a ``float32``
mantissa: the values would round to the nearest few microseconds.

Public functions:

MeV_to_keV(values), keV_to_MeV(values)
    Energy, between the unit GATE writes and the one spectra are read in.
mm_to_cm(values), cm_to_mm(values), mm_to_m(values), m_to_mm(values)
    Length, between the unit GATE writes and the ones geometry is described in.
cm_to_m(values), m_to_cm(values)
    Length, the pair that does not involve the unit GATE writes.
s_to_ms(values), ms_to_s(values), s_to_ns(values), ns_to_s(values)
    Time, between the unit GATE writes and the ones resolutions are quoted in.
ms_to_ns(values), ns_to_ms(values)
    Time, the pair that does not involve the unit GATE writes.
rad_to_deg(values), deg_to_rad(values)
    Angle, thin wrappers over :func:`numpy.rad2deg` and :func:`numpy.deg2rad`.

Public attributes:

GATE_UNITS : Mapping[str, str]
    The unit GATE writes each kind of quantity in.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast

import numpy as np
import pandas as pd

# How many of the smaller unit go into the larger one. Both directions of a
# pair use the same number, so the two cannot drift apart.
KEV_PER_MEV: Final[float] = 1000.0
MM_PER_CM: Final[float] = 10.0
MM_PER_M: Final[float] = 1000.0
CM_PER_M: Final[float] = 100.0
MS_PER_S: Final[float] = 1000.0
NS_PER_S: Final[float] = 1_000_000_000.0
NS_PER_MS: Final[float] = 1_000_000.0

# The unit GATE writes each kind of quantity in, which is where every
# conversion of a branch starts.
GATE_UNITS: Final[Mapping[str, str]] = MappingProxyType(
    {"energy": "MeV", "length": "mm", "time": "s"}
)


def MeV_to_keV[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert energy from megaelectronvolts to kiloelectronvolts.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Energies in MeV, the unit GATE writes.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same energies in keV.

    Examples
    --------
    >>> MeV_to_keV(0.511)
    511.0
    """
    return _scaled(values, KEV_PER_MEV)


def keV_to_MeV[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert energy from kiloelectronvolts to megaelectronvolts.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Energies in keV.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same energies in MeV, the unit GATE writes.
    """
    return _divided(values, KEV_PER_MEV)


def mm_to_cm[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert length from millimetres to centimetres.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Lengths in mm, the unit GATE writes.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same lengths in cm.
    """
    return _divided(values, MM_PER_CM)


def cm_to_mm[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert length from centimetres to millimetres.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Lengths in cm.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same lengths in mm, the unit GATE writes.
    """
    return _scaled(values, MM_PER_CM)


def mm_to_m[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert length from millimetres to metres.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Lengths in mm, the unit GATE writes.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same lengths in m.
    """
    return _divided(values, MM_PER_M)


def m_to_mm[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert length from metres to millimetres.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Lengths in m, as a GATE macro often gives them.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same lengths in mm, the unit GATE writes.
    """
    return _scaled(values, MM_PER_M)


def cm_to_m[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert length from centimetres to metres.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Lengths in cm.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same lengths in m.
    """
    return _divided(values, CM_PER_M)


def m_to_cm[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert length from metres to centimetres.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Lengths in m.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same lengths in cm.
    """
    return _scaled(values, CM_PER_M)


def s_to_ms[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert time from seconds to milliseconds.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Times in s, the unit GATE writes.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same times in ms.
    """
    return _scaled(values, MS_PER_S)


def ms_to_s[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert time from milliseconds to seconds.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Times in ms.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same times in s, the unit GATE writes.
    """
    return _divided(values, MS_PER_S)


def s_to_ns[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert time from seconds to nanoseconds.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Times in s, the unit GATE writes.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same times in ns, the unit a time resolution is quoted in.
    """
    return _scaled(values, NS_PER_S)


def ns_to_s[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert time from nanoseconds to seconds.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Times in ns.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same times in s, the unit GATE writes.
    """
    return _divided(values, NS_PER_S)


def ms_to_ns[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert time from milliseconds to nanoseconds.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Times in ms.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same times in ns.
    """
    return _scaled(values, NS_PER_MS)


def ns_to_ms[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert time from nanoseconds to milliseconds.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Times in ns.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same times in ms.
    """
    return _divided(values, NS_PER_MS)


def rad_to_deg[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert angle from radians to degrees.

    A thin wrapper over :func:`numpy.rad2deg`, here so that the four families
    of conversion are found in one place. Every angle this package computes is
    in radians, as in NumPy, and this is what reads it in degrees.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Angles in radians.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same angles in degrees.
    """
    return _by_ufunc(values, np.rad2deg)


def deg_to_rad[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Convert angle from degrees to radians.

    A thin wrapper over :func:`numpy.deg2rad`, the counterpart of
    :func:`rad_to_deg`.

    Parameters
    ----------
    values : float | numpy.ndarray | pandas.Series
        Angles in degrees, as they are usually written down.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        The same angles in radians, which is what every function here takes.
    """
    return _by_ufunc(values, np.deg2rad)


def _scaled[Quantity: (float, np.ndarray, pd.Series)](values: Quantity, factor: float) -> Quantity:
    """Return values multiplied by a factor, computed in ``float64``."""
    return cast(Quantity, _as_float64(values) * factor)


def _divided[Quantity: (float, np.ndarray, pd.Series)](values: Quantity, factor: float) -> Quantity:
    """Return values divided by a factor, computed in ``float64``.

    Dividing rather than multiplying by the reciprocal: one over ten has no
    exact binary form, so the reciprocal would add an error that dividing by
    the factor itself does not.
    """
    return cast(Quantity, _as_float64(values) / factor)


def _by_ufunc[Quantity: (float, np.ndarray, pd.Series)](
    values: Quantity, conversion: np.ufunc
) -> Quantity:
    """Return what a NumPy conversion answers, in the kind it was given.

    NumPy answers a plain number with a ``numpy.float64``, and the module
    promises the kind it was handed back, so a number stays a number.
    """
    converted = conversion(_as_float64(values))
    if isinstance(values, pd.Series | np.ndarray):
        return cast(Quantity, converted)
    return cast(Quantity, float(converted))


def _as_float64[Quantity: (float, np.ndarray, pd.Series)](values: Quantity) -> Quantity:
    """Return values as ``float64``, keeping what carries an index or a shape."""
    if isinstance(values, pd.Series | np.ndarray):
        return cast(Quantity, values.astype(np.float64))
    return cast(Quantity, float(values))
