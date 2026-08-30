"""Export of extracted GATE trees to the supported output formats.

The subpackage holds one writer per output format and the lookup used to pick
one. Output files carry the extracted tree only: the histograms stored next to
the trees in a GATE file are not copied over.

Public functions
----------------
get_writer(file_format: OutputFileFormat) -> TreeWriter
    Return the writer producing the requested format.
write_tree(data: TreeData, path: Path, file_format: OutputFileFormat) -> Path
    Write the data in the requested format.
"""

from pathlib import Path
from typing import Final

from opengate_gate_tree.errors import ExportError
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.io.writers.base import TreeWriter
from opengate_gate_tree.io.writers.csv_writer import CsvTreeWriter
from opengate_gate_tree.io.writers.hdf5_writer import Hdf5TreeWriter
from opengate_gate_tree.io.writers.root_writer import RootTreeWriter
from opengate_gate_tree.tree.treedata import TreeData

# Writer used for each supported output format.
WRITERS: Final[dict[OutputFileFormat, TreeWriter]] = {
    OutputFileFormat.CSV: CsvTreeWriter(),
    OutputFileFormat.HDF5: Hdf5TreeWriter(),
    OutputFileFormat.ROOT: RootTreeWriter(),
}


def get_writer(file_format: OutputFileFormat) -> TreeWriter:
    """Return the writer producing the requested format.

    Parameters
    ----------
    file_format : OutputFileFormat
        Requested output format.

    Returns
    -------
    TreeWriter
        Writer for the requested format.

    Raises
    ------
    ExportError
        If no writer produces the requested format.
    """
    writer = WRITERS.get(file_format)
    if writer is None:
        raise ExportError(
            f"No writer is available for the '{file_format.value}' format. "
            f"Formats available: {[value.value for value in WRITERS]}."
        )
    return writer


def write_tree(data: TreeData, path: Path, file_format: OutputFileFormat) -> Path:
    """Write the data in the requested format.

    Parameters
    ----------
    data : TreeData
        Data to write.
    path : Path
        Path of the output file. An existing file is overwritten.
    file_format : OutputFileFormat
        Format of the output file.

    Returns
    -------
    Path
        Path that was written to.

    Raises
    ------
    ExportError
        If no writer produces the requested format, or the file cannot be
        written.
    """
    return get_writer(file_format).write(data, path)
