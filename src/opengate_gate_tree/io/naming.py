"""Names of the files a run writes.

An output file is named after the title the caller gave it, the tree the data
came from, and the format it is written in: ``patient_01.hits.csv``. The tree
is part of the name because one input file holds several trees, and a run
extracting the hits and a run extracting the singles of the same simulation
should not land on the same name.

The report of a run, when one is asked for, goes next to the data under the
same title: ``patient_01.hits.stats.json``.

Naming lives here rather than in the command line, so that code using the
package as a library lands on the same names without repeating the rule.

Public functions:

build_output_file_name(title, tree, file_format) -> str
    Name of the file holding an extracted tree.
build_output_file_path(directory, title, tree, file_format) -> Path
    Path of the file holding an extracted tree.
build_statistics_file_path(directory, title, tree) -> Path
    Path of the report describing an extracted tree.
"""

from pathlib import Path
from typing import Final

from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.tree.gatetree import GateTree

# Ending of the file holding the report of a run.
STATISTICS_SUFFIX: Final[str] = "stats.json"


def build_output_file_name(
    title: str,
    tree: GateTree,
    file_format: OutputFileFormat,
) -> str:
    """Return the name of the file holding an extracted tree.

    Parameters
    ----------
    title : str
        Title of the run, used as the first part of the name.
    tree : GateTree
        Tree the data was extracted from.
    file_format : OutputFileFormat
        Format the data is written in.

    Returns
    -------
    str
        Name in the form ``<title>.<tree>.<format>``.
    """
    return f"{title}.{tree.value.lower()}.{file_format.value}"


def build_output_file_path(
    directory: Path,
    title: str,
    tree: GateTree,
    file_format: OutputFileFormat,
) -> Path:
    """Return the path of the file holding an extracted tree.

    Parameters
    ----------
    directory : Path
        Directory the file is written to.
    title : str
        Title of the run.
    tree : GateTree
        Tree the data was extracted from.
    file_format : OutputFileFormat
        Format the data is written in.

    Returns
    -------
    Path
        Path of the output file.
    """
    return directory / build_output_file_name(title, tree, file_format)


def build_statistics_file_path(directory: Path, title: str, tree: GateTree) -> Path:
    """Return the path of the report describing an extracted tree.

    Parameters
    ----------
    directory : Path
        Directory the report is written to.
    title : str
        Title of the run, the same one the data carries.
    tree : GateTree
        Tree the data was extracted from.

    Returns
    -------
    Path
        Path of the report file.
    """
    return directory / f"{title}.{tree.value.lower()}.{STATISTICS_SUFFIX}"
