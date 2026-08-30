"""Tool runtime configuration.

The module defines :class:`RunConfig`, which stores parsed runtime arguments:
input/output paths, optional output title, and operation mode flags.

Public objects:

RunConfig
    Immutable set of parameters for a single tool run.
"""

from dataclasses import dataclass, field
from pathlib import Path

from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.tree.gatetree import GateTree


@dataclass(frozen=True)
class RunConfig:
    """Parameters for a single tool run.

    Attributes
    ----------
    input_gate_root_file : Path
        Path to the gate ``*.root`` file to process.
    output_dir : Path
        Path to the directory where the output files will be saved.
    output_file_title : str | None
        Title of the output file (without extension).
    gate_tree : GateTree
        The gate tree structure loaded from the input file.
    output_file_format : OutputFileFormat | None
        Format of the output file.
    branches_to_extract : list[str]
        List of branch names to extract from the gate tree.
    """

    input_gate_root_file: Path | None
    output_dir: Path | None
    output_file_title: str | None
    gate_tree: GateTree | None
    output_file_format: OutputFileFormat | None
    branches_to_extract: list[str] = field(default_factory=list)
