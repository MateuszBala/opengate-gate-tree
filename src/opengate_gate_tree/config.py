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
    input_tree_name : str | None
        Name of the tree in the input file, when it differs from the standard
        one or when the file holds several trees of hits.
    merge_hits_trees : bool
        Whether to read every tree of hits in the file as a single dataset.
    write_statistics : bool
        Whether to write a report describing the extracted data.
    skip_hits_validation : bool
        Whether to extract the branches without recognising and checking the
        structure of the "Hits" tree.
    """

    input_gate_root_file: Path | None
    output_dir: Path | None
    output_file_title: str | None
    gate_tree: GateTree | None
    output_file_format: OutputFileFormat | None
    branches_to_extract: list[str] = field(default_factory=list)
    input_tree_name: str | None = None
    merge_hits_trees: bool = False
    write_statistics: bool = False
    skip_hits_validation: bool = False
