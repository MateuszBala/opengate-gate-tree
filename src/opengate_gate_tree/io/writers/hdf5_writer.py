"""Export of a GATE tree to HDF5.

The tree becomes one group holding one dataset per branch. Fixed-width array
branches keep their shape as two-dimensional datasets, so ``volumeID`` is
written as ``(entries, width)`` rather than split into separate columns.

Text branches are written with a variable-length UTF-8 string type and read
back with ``h5py`` through ``asstr()``.

The group tracks insertion order, because HDF5 lists links alphabetically by
default and the branch order carries meaning for the analysis reading the file.

Public objects:

Hdf5TreeWriter
    Writer producing an HDF5 file.
"""

from pathlib import Path

import h5py

from opengate_gate_tree import __version__
from opengate_gate_tree.errors import ExportError
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.io.writers.base import (
    as_text_list,
    is_text_column,
    prepare_output_directory,
)
from opengate_gate_tree.tree.treedata import TreeData


class Hdf5TreeWriter:
    """Writer producing an HDF5 file."""

    file_format = OutputFileFormat.HDF5

    def write(self, data: TreeData, path: Path) -> Path:
        """Write the data as HDF5.

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
            If the file cannot be written.
        """
        prepare_output_directory(path)
        try:
            with h5py.File(path, "w") as output_file:
                group = output_file.create_group(data.tree.value, track_order=True)
                for name, column in data.columns.items():
                    if is_text_column(column):
                        group.create_dataset(
                            name,
                            data=as_text_list(name, column),
                            dtype=h5py.string_dtype(encoding="utf-8"),
                        )
                    else:
                        group.create_dataset(name, data=column)
                group.attrs["gate_tree"] = data.tree.value
                group.attrs["entries"] = data.entry_count
                group.attrs["branches"] = list(data.branch_names)
                group.attrs["package_version"] = __version__
        except (OSError, TypeError, ValueError) as err:
            raise ExportError(f"HDF5 file could not be written: {path}") from err
        return path
