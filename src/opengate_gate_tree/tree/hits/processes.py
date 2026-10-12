"""Names of the processes a "Hits" tree records.

The ``processName`` branch holds the name of the process of every hit, as text.
:class:`Process` names the ones the reference outputs hold. It derives from
:class:`enum.StrEnum`, so its members **are** those strings and compare
against a column of them with nothing in between::

    compton = data["processName"] == Process.COMPTON

The names come from the physics list a simulation runs with, not from GATE
itself, so another list can write names the class does not know. Everything
in the package that takes a process takes a plain string as well, and a name
the class does not list is not an error.

Public objects:

Process
    Process of an interaction, as the "Hits" tree names it.
"""

from enum import StrEnum


class Process(StrEnum):
    """Process of an interaction, as the "Hits" tree names it."""

    COMPTON = "Compton"
    PHOTOELECTRIC = "PhotoElectric"
    RAYLEIGH = "RayleighScattering"
