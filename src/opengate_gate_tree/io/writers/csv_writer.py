"""Export of a GATE tree to CSV.

A CSV file holds scalar cells, so fixed-width array branches are written as one
column per component, named ``<branch>_<index>``. The file therefore does not
carry the branch back as an array when read again; ROOT and HDF5 do.

Public objects:

CsvTreeWriter
    Writer producing a CSV file.
"""

from pathlib import Path

from opengate_gate_tree.errors import ExportError
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.io.writers.base import prepare_output_directory
from opengate_gate_tree.tree.treedata import TreeData


class CsvTreeWriter:
    """Writer producing a CSV file."""

    file_format = OutputFileFormat.CSV

    def write(self, data: TreeData, path: Path) -> Path:
        """Write the data as CSV.

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
            data.to_dataframe().to_csv(path, index=False)
        except OSError as err:
            raise ExportError(f"CSV file could not be written: {path}") from err
        return path
