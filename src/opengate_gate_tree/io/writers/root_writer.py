"""Export of a GATE tree to ROOT.

The tree is written as a ``TTree``, the container the analysis code reading
these files expects. Assigning a mapping to a key of an uproot file writes a
``ROOT::RNTuple`` instead, which round-trips through uproot but cannot be read
by code using ``TTree::Draw``, ``SetBranchAddress`` or ``RDataFrame``. The
writer therefore declares the branches with ``mktree`` and appends the data
with ``extend``.

Data is appended only when the tree holds entries: ``extend`` raises on an
empty batch that contains a text branch, while ``mktree`` on its own already
produces a valid tree with every branch declared.

Public objects:

RootTreeWriter
    Writer producing a ROOT file holding a TTree.
"""

from pathlib import Path
from typing import Any, Final

import uproot

from opengate_gate_tree.errors import ExportError
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.io.writers.base import (
    as_text_list,
    is_text_column,
    prepare_output_directory,
    reject_branch_names,
)
from opengate_gate_tree.tree.treedata import ARRAY_BRANCH_NDIM, TreeData

# Characters uproot reads as an array dimension rather than as part of a name.
FORBIDDEN_NAME_CHARACTERS: Final[str] = "[]"

# What uproot does with such a name, reported to the caller.
FORBIDDEN_REASON: Final[str] = (
    "uproot reads brackets in a branch name as an array dimension, which writes a file whose "
    "branches cannot be read back. The GateToTree layout, which splits volumeID into "
    "volumeID[0] and up, can be written as CSV or HDF5 instead."
)


class RootTreeWriter:
    """Writer producing a ROOT file holding a TTree."""

    file_format = OutputFileFormat.ROOT

    def write(self, data: TreeData, path: Path) -> Path:
        """Write the data as a ROOT TTree.

        Parameters
        ----------
        data : TreeData
            Data to write.
        path : Path
            Path of the output file. An existing file is overwritten.

        Returns
        -------
        Path
            Path that was written to.

        Raises
        ------
        ExportError
            If the data holds no branches, a branch name holds a bracket, or
            the file cannot be written.
        """
        if not data.branch_names:
            raise ExportError("A ROOT tree cannot be written without branches.")

        reject_branch_names(data, FORBIDDEN_NAME_CHARACTERS, self.file_format, FORBIDDEN_REASON)
        prepare_output_directory(path)
        branch_types, branch_data = _branch_specification(data)
        tree_name = data.tree.value

        try:
            with uproot.recreate(path) as output_file:
                output_file.mktree(tree_name, branch_types)
                if data.entry_count:
                    output_file[tree_name].extend(branch_data)
        except (OSError, ValueError, TypeError) as err:
            raise ExportError(f"ROOT file could not be written: {path}") from err
        return path


def _branch_specification(data: TreeData) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the data into uproot branch types and the matching values."""
    branch_types: dict[str, Any] = {}
    branch_data: dict[str, Any] = {}

    for name, column in data.columns.items():
        if is_text_column(column):
            branch_types[name] = str
            branch_data[name] = as_text_list(name, column)
        elif column.ndim == ARRAY_BRANCH_NDIM:
            branch_types[name] = (column.dtype, column.shape[1:])
            branch_data[name] = column
        else:
            branch_types[name] = column.dtype
            branch_data[name] = column

    return branch_types, branch_data
