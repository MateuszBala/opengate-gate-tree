"""Writing a summary of extracted data to a file.

The report is written as JSON, because it is meant to be read by whatever
comes next in the analysis as much as by a person: a run compared against
another one, a table of process counts, a check that a file holds what a
simulation was supposed to produce. The rendering meant for reading is
:func:`~opengate_gate_tree.tree.statistics.format_statistics`.

Public functions:

write_statistics(statistics, path) -> Path
    Write a summary as a JSON file.
"""

import json
from pathlib import Path

from opengate_gate_tree.errors import ExportError
from opengate_gate_tree.io.writers.base import prepare_output_directory
from opengate_gate_tree.tree.statistics import TreeStatistics, statistics_to_dict

# Indentation of the written report.
REPORT_INDENT: int = 2


def write_statistics(statistics: TreeStatistics, path: Path) -> Path:
    """Write a summary as a JSON file.

    Parameters
    ----------
    statistics : TreeStatistics
        Summary to write.
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
        with path.open("w", encoding="utf-8") as report_file:
            json.dump(
                statistics_to_dict(statistics),
                report_file,
                indent=REPORT_INDENT,
                allow_nan=False,
            )
            report_file.write("\n")
    except (OSError, TypeError, ValueError) as err:
        raise ExportError(f"Statistics file could not be written: {path}") from err
    return path
