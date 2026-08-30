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
as_text_list(column) -> list[str]
    Return a text column as a list of strings.
prepare_output_directory(path) -> None
    Create the directory the output file will be written to.
"""

from pathlib import Path
from typing import Any, Protocol

import numpy.typing as npt

from opengate_gate_tree.errors import ExportError
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.tree.treedata import TreeData

# NumPy data type kind used for text columns loaded from a ROOT file.
TEXT_DTYPE_KIND = "O"


class TreeWriter(Protocol):
    """Interface implemented by every output writer."""

    file_format: OutputFileFormat

    def write(self, data: TreeData, path: Path) -> Path:
        """Write the data to ``path`` and return the path written to."""
        ...


def is_text_column(column: npt.NDArray[Any]) -> bool:
    """Report whether a column holds text rather than numbers."""
    return column.dtype.kind == TEXT_DTYPE_KIND


def as_text_list(column: npt.NDArray[Any]) -> list[str]:
    """Return a text column as a list of strings."""
    return [str(value) for value in column]


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
