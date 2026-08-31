"""What the gammas of a PositroniumSource were, as read from the "Hits" tree.

A ``PositroniumSource`` writes four branches saying where each gamma came
from: which source model emitted it, through which decay channel, what kind of
gamma it is, and which channel of the configured mixture it belongs to. GATE
stores all four as integers, and their meaning is defined by the enums of
``GateEmittedGammaInformation.hh``:

============ ============================================ =====================
Branch       Meaning of the values                        Enum in GATE
============ ============================================ =====================
`sourceType` which model emitted the gamma                ``SourceKind``
`decayType`  which decay channel it came through          ``DecayModel``
`gammaType`  what kind of gamma it is                     ``GammaKind``
`decayIndex` channel of the configured mixture, or -1     none
============ ============================================ =====================

The classes here are named after the branches rather than after the enums of
GATE, because a branch name is what the reader of a file works with. Each
class names its counterpart, so the way back to the source of the values stays
short.

They derive from :class:`enum.IntEnum`, which means their members **are** the
integers GATE wrote. A column read from a file can be compared against them
with nothing in between::

    prompt = data["gammaType"] == GammaType.PROMPT

A plain :class:`enum.Enum` would not work here, and would not say so: comparing
a column against one of its members yields a mask of ``False`` rather than an
error.

Public objects:

SourceType
    Model that emitted a gamma.
DecayType
    Decay channel a gamma came through.
GammaType
    Kind of gamma.
POSITRONIUM_BRANCHES
    Branches whose values these classes describe.
DECAY_INDEX_BRANCH, NOT_A_POSITRONIUM_SOURCE
    Branch holding the channel of the mixture, and the value it holds for a
    gamma written by another source.
positronium_enum(branch) -> type[IntEnum] | None
    Class describing the values of a branch, when one describes them.
is_positronium_source(decay_index) -> numpy.ndarray
    Which rows were written by a PositroniumSource.
"""

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt


class SourceType(IntEnum):
    """Model that emitted a gamma.

    Counterpart of ``GateEmittedGammaInformation::SourceKind``.
    """

    NOT_DEFINED = 0
    SINGLE_GAMMA_EMITTER = 1
    PARA_POSITRONIUM = 2
    ORTHO_POSITRONIUM = 3
    DIRECT_ANNIHILATION = 4


class DecayType(IntEnum):
    """Decay channel a gamma came through.

    Counterpart of ``GateEmittedGammaInformation::DecayModel``.
    """

    NONE = 0
    STANDARD = 1
    DEEXCITATION = 2


class GammaType(IntEnum):
    """Kind of gamma.

    Counterpart of ``GateEmittedGammaInformation::GammaKind``.
    """

    UNKNOWN = 0
    SINGLE = 1
    ANNIHILATION = 2
    PROMPT = 3


# Branches whose values one of the classes above describes.
POSITRONIUM_BRANCHES: Final[Mapping[str, type[IntEnum]]] = MappingProxyType(
    {
        "sourceType": SourceType,
        "decayType": DecayType,
        "gammaType": GammaType,
    }
)

# Branch holding which channel of a configured mixture a gamma came from.
DECAY_INDEX_BRANCH: Final[str] = "decayIndex"

# Value that branch holds for a gamma written by a source that is not a
# PositroniumSource. The channel numbers themselves depend on the order the
# fractions were configured in, so only this value has a fixed meaning.
NOT_A_POSITRONIUM_SOURCE: Final[int] = -1


def positronium_enum(branch: str) -> type[IntEnum] | None:
    """Return the class describing the values of a branch.

    Parameters
    ----------
    branch : str
        Branch name.

    Returns
    -------
    type[IntEnum] | None
        Class describing the values of the branch, or ``None`` when the
        package describes none. ``decayIndex`` has no class of its own: its
        values are channel numbers, and which channel a number means depends
        on how the source was configured.
    """
    return POSITRONIUM_BRANCHES.get(branch)


def is_positronium_source(decay_index: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    """Return which rows were written by a PositroniumSource.

    Parameters
    ----------
    decay_index : numpy.ndarray
        Column of the ``decayIndex`` branch.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` where the row carries a channel number rather
        than the value standing for another kind of source.
    """
    return np.asarray(decay_index != NOT_A_POSITRONIUM_SOURCE, dtype=np.bool_)
