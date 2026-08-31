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
`decayIndex` component of the sampled decay, or -1       none
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
decode_positronium_value(branch, value) -> IntEnum
    What one value of a branch means.
decode_positronium_column(branch, column) -> numpy.ndarray
    What every value of a column means.
is_positronium_source(decay_index) -> numpy.ndarray
    Which rows carry the decay metadata of a PositroniumSource.
"""

from collections import Counter
from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from opengate_gate_tree.logger import log


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

# Value that branch holds when a row carries no decay metadata: a gamma from
# another kind of source, or one the metadata never reached. The component
# numbers themselves depend on how the source was configured, so this is the
# only value of the branch with a fixed meaning.
NOT_A_POSITRONIUM_SOURCE: Final[int] = -1

# NumPy data type kinds a branch of whole numbers can be read from.
WHOLE_NUMBER_DTYPE_KINDS: Final[frozenset[str]] = frozenset({"i", "u", "b"})


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
    """Return which rows carry the decay metadata of a PositroniumSource.

    What the mask answers is narrower than it may look. It says the row was
    given the metadata of a sampled decay component, which GATE writes for
    every gamma of a ``PositroniumSource`` — including the ones from a direct
    annihilation component, since the source numbers those like the rest. A
    row without it comes from another kind of source, or from a particle the
    metadata never reached. What a gamma itself was is said by ``sourceType``.

    Parameters
    ----------
    decay_index : numpy.ndarray
        Column of the ``decayIndex`` branch.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` where the row carries a component number rather
        than the value standing for no metadata.

    Raises
    ------
    ValueError
        If the column is not one-dimensional or does not hold whole numbers.
        A comparison against a value that is not a column answers with a
        single truth value instead of one per row, which selects everything
        without saying so.
    """
    column = _whole_number_column(DECAY_INDEX_BRANCH, decay_index)
    return np.asarray(column != NOT_A_POSITRONIUM_SOURCE, dtype=np.bool_)


def decode_positronium_value(branch: str, value: int) -> IntEnum:
    """Return what one value of a branch means.

    Parameters
    ----------
    branch : str
        Branch name.
    value : int
        Value written by GATE.

    Returns
    -------
    IntEnum
        Member of the class describing the branch.

    Raises
    ------
    ValueError
        If the package describes no values for the branch, or the value is
        not one of them. A question about a single value has one answer or
        none: reading it as the value standing for "not defined" would report
        something the file does not say.
    """
    enum_class = _described_enum(branch)
    try:
        return enum_class(value)
    except ValueError as err:
        written = ", ".join(f"{int(member)} ({member.name})" for member in enum_class)
        raise ValueError(
            f"Branch '{branch}' has no meaning for the value {value}. "
            f"Values GATE writes there: {written}."
        ) from err


def decode_positronium_column(branch: str, column: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return what every value of a column means.

    A value the package does not know becomes ``None`` rather than stopping
    the reading: a GATE build can write one, and an analysis of the rows that
    are understood is still worth having. The values that were not understood
    are reported in the log, with how often each of them occurs.

    Parameters
    ----------
    branch : str
        Branch name.
    column : numpy.ndarray
        Column of that branch.

    Returns
    -------
    numpy.ndarray
        Array of members of the class describing the branch, with ``None``
        wherever the value is not one of them.

    Raises
    ------
    ValueError
        If the package describes no values for the branch, or the column is
        not a one-dimensional column of whole numbers. A column of another
        type would be read by truncating its values, which is the silent
        substitution this function exists to avoid.
    """
    enum_class = _described_enum(branch)
    known: dict[int, IntEnum] = {int(member): member for member in enum_class}
    values = [int(value) for value in _whole_number_column(branch, column)]

    unknown = Counter(value for value in values if value not in known)
    if unknown:
        reported = ", ".join(
            f"{value} ({count} row(s))" for value, count in sorted(unknown.items())
        )
        log().warning(
            "Branch '%s' holds %d value(s) this version does not describe: %s. "
            "They are read as nothing at all.",
            branch,
            len(unknown),
            reported,
        )

    decoded = np.empty(len(values), dtype=object)
    decoded[:] = [known.get(value) for value in values]
    return decoded


def _whole_number_column(branch: str, column: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return a column of whole numbers, refusing anything else.

    Values are read from a column, one per row. Anything that is not such a
    column either compares as a single value, which selects every row without
    saying so, or is read by truncation, which reads one thing as another.
    """
    values = np.asarray(column)
    if values.ndim != 1 or values.dtype.kind not in WHOLE_NUMBER_DTYPE_KINDS:
        raise ValueError(
            f"Branch '{branch}' has to be read from a one-dimensional column of whole numbers, "
            f"got a {values.ndim}-dimensional column of {values.dtype}."
        )
    return values


def _described_enum(branch: str) -> type[IntEnum]:
    """Return the class describing a branch, refusing one that has none."""
    enum_class = positronium_enum(branch)
    if enum_class is None:
        raise ValueError(
            f"Branch '{branch}' does not hold values this package describes. "
            f"Described branches: {list(POSITRONIUM_BRANCHES)}."
        )
    return enum_class
