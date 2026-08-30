"""OutputFileFormat enum and parser.

This module defines the OutputFileFormat enum and provides a function to parse
strings into OutputFileFormat enum members.
"""

from enum import Enum


class OutputFileFormat(Enum):
    """Enum representing the output file format."""

    ROOT = "root"
    HDF5 = "hdf5"
    CSV = "csv"


def parse_output_file_format(name: str) -> OutputFileFormat:
    """Parse a string into an OutputFileFormat enum member.

    Parameters
    ----------
    name : str
        Name of the output file format member.

    Returns
    -------
    OutputFileFormat
        Corresponding OutputFileFormat enum member.

    Raises
    ------
    ValueError
        If the name does not correspond to any OutputFileFormat member.
    """
    try:
        return OutputFileFormat[name.upper()]
    except KeyError as err:
        raise ValueError(f"Unknown OutputFileFormat member: {name}") from err
