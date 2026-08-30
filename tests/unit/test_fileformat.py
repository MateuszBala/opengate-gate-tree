"""Unit tests for output file format parsing."""

import pytest

from opengate_gate_tree.io.fileformat import OutputFileFormat, parse_output_file_format


def test_parse_output_file_format_is_case_insensitive() -> None:
    """Known format names should parse regardless of case."""
    assert parse_output_file_format("root") == OutputFileFormat.ROOT
    assert parse_output_file_format("HDF5") == OutputFileFormat.HDF5
    assert parse_output_file_format("Csv") == OutputFileFormat.CSV


def test_parse_output_file_format_raises_for_unknown_value() -> None:
    """Unknown format names should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown OutputFileFormat member"):
        parse_output_file_format("json")
