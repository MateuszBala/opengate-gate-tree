"""Shared parts of the output writers.

The module defines the interface every writer implements and the helpers they
share, mainly the normalization of text columns.

Text columns need a different representation per backend. Both accepted
representations start from a list of strings, so the conversion lives here:
uproot rejects NumPy unicode arrays in a tree, and h5py rejects them when a
string data type is requested.

Public objects:

TreeWriter
    Interface implemented by every output writer.
is_text_column(column) -> bool
    Report whether a column holds text.
as_text_list(name, column) -> list[str]
    Return a text column as a list of strings.
prepare_output_directory(path) -> None
    Create the directory the output file will be written to.
reject_branch_names(data, characters, file_format, reason) -> None
    Refuse branch names a backend cannot store faithfully.
"""

from pathlib import Path
from typing import Any, Protocol

import numpy.typing as npt

from opengate_gate_tree.errors import ExportError
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.tree.treedata import TreeData

# NumPy data type kinds that hold text. Columns read from a ROOT file arrive as
# arrays of Python objects ("O"), while data built in Python is more often a
# fixed-width unicode ("U") or byte string ("S") array.
TEXT_DTYPE_KINDS = frozenset({"O", "U", "S"})


class TreeWriter(Protocol):
    """Interface implemented by every output writer."""

    file_format: OutputFileFormat

    def write(self, data: TreeData, path: Path) -> Path:
        """Write the data to ``path`` and return the path written to."""
        ...


def reject_branch_names(
    data: TreeData,
    characters: str,
    file_format: OutputFileFormat,
    reason: str,
) -> None:
    """Refuse branch names a backend cannot store faithfully.

    Both backends accept such a name and then write something other than what
    was asked for, without reporting anything. Refusing the file is the only
    way the caller learns about it before reading the result.

    Parameters
    ----------
    data : TreeData
        Data about to be written.
    characters : str
        Characters the backend cannot carry in a branch name.
    file_format : OutputFileFormat
        Format being written, named in the message.
    reason : str
        What the backend would do with such a name.

    Raises
    ------
    ExportError
        If any branch name holds one of the characters.
    """
    offenders = [
        name for name in data.branch_names if any(character in name for character in characters)
    ]
    if offenders:
        raise ExportError(
            f"Branches cannot be written to a '{file_format.value}' file because their names "
            f'hold one of "{characters}": {offenders}. {reason}'
        )


def is_text_column(column: npt.NDArray[Any]) -> bool:
    """Report whether a column holds text rather than numbers."""
    return column.dtype.kind in TEXT_DTYPE_KINDS


def as_text_list(name: str, column: npt.NDArray[Any]) -> list[str]:
    """Return a text column as a list of strings.

    Parameters
    ----------
    name : str
        Branch name, used in the error message.
    column : numpy.ndarray
        Column to convert.

    Returns
    -------
    list[str]
        Column values as strings.

    Raises
    ------
    ExportError
        If the column holds values that are not text. An array of Python
        objects can hold anything, and converting such values with ``str``
        would write something other than the data the caller provided.
    """
    values: list[str] = []
    for value in column:
        if isinstance(value, bytes):
            values.append(value.decode("utf-8"))
        elif isinstance(value, str):
            values.append(value)
        else:
            raise ExportError(
                f"Branch '{name}' holds a value that is not text: "
                f"{value!r} of type {type(value).__name__}."
            )
    return values


def prepare_output_directory(path: Path) -> None:
    """Create the directory the output file will be written to.

    Parameters
    ----------
    path : Path
        Path of the output file.

    Raises
    ------
    ExportError
        If the directory cannot be created.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise ExportError(f"Output directory could not be created: {path.parent}") from err
