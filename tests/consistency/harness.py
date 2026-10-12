"""Comparing an analysis written with the package against one made outside it.

An analysis consistency test takes an input file, the output of an analysis
made independently of this package - by other code, in another language - and
the same analysis written with the package, and checks that the two agree.
Each case is a directory under ``cases/``:

``input.root``
    The GATE output both analyses read; 500 kB or less, as every file the
    repository carries.
``reference.csv``
    The output of the analysis made outside the package: one row per event,
    ``runID`` and ``eventID`` first, the other columns named as
    :meth:`EventTrackHitAnalysis.extract` names them (``t1h2_edep``).
``analysis.py``
    The same analysis written with the package: ``ANALYSIS``, an
    :class:`EventTrackHitAnalysis`, and ``VARIABLES``, the list of variables
    of the table. ``RTOL`` may give the relative tolerance of floating point
    columns; it is ``1e-6`` otherwise, since GATE writes single precision and
    the other analysis may compute in double.
``README.md``
    Where the reference comes from: the tool, its version, the commit of its
    code, what the analysis asks, and why the tolerance is what it is.

Two tables agree when they hold the same events, the same columns, equal
whole numbers and text, and floating point values within the tolerance.

Public objects:

Case
    One case: its directory, and what it holds.
discover_cases(directory) -> list[Case]
    Every case under a directory.
load_case(directory) -> Case
    One case.
compare_tables(table, reference, rtol) -> list[str]
    Every difference between a table and its reference.
"""

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from opengate_gate_tree import EventTrackHitAnalysis

# Columns that name an event in both tables.
EVENT_KEY: Final[list[str]] = ["runID", "eventID"]

# Tolerance of floating point columns when a case gives none.
DEFAULT_RTOL: Final[float] = 1e-6

# Files every case has to hold.
CASE_FILES: Final[tuple[str, ...]] = ("input.root", "reference.csv", "analysis.py", "README.md")

# How many events or values a difference lists before it only counts them.
SHOWN: Final[int] = 10


@dataclass(frozen=True)
class Case:
    """One analysis consistency case.

    Attributes
    ----------
    directory : Path
        Directory of the case.
    analysis : EventTrackHitAnalysis
        The analysis written with the package.
    variables : Sequence[Any]
        The variables of its table.
    rtol : float
        Relative tolerance of floating point columns.
    """

    directory: Path
    analysis: EventTrackHitAnalysis
    variables: Sequence[Any]
    rtol: float

    @property
    def name(self) -> str:
        """Name of the case, the name of its directory."""
        return self.directory.name

    @property
    def input_path(self) -> Path:
        """The GATE output the case reads."""
        return self.directory / "input.root"

    def reference(self) -> pd.DataFrame:
        """Return the output of the analysis made outside the package."""
        return pd.read_csv(self.directory / "reference.csv")


def discover_cases(directory: Path) -> list[Case]:
    """Return every case under a directory, in the order of their names."""
    return [
        load_case(path)
        for path in sorted(directory.iterdir())
        if path.is_dir() and not path.name.startswith((".", "_"))
    ]


def load_case(directory: Path) -> Case:
    """Return one case.

    Raises
    ------
    ValueError
        If the case lacks a file, or its analysis does not define what it has to.
    """
    missing = [name for name in CASE_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Case {directory.name} lacks {missing}; a case holds {list(CASE_FILES)}.")
    spec = importlib.util.spec_from_file_location(
        f"consistency_{directory.name}", directory / "analysis.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    analysis = getattr(module, "ANALYSIS", None)
    variables = getattr(module, "VARIABLES", None)
    if not isinstance(analysis, EventTrackHitAnalysis) or not isinstance(variables, list | tuple):
        raise ValueError(
            f"The analysis.py of case {directory.name} has to define ANALYSIS, an "
            "EventTrackHitAnalysis, and VARIABLES, a list of variables."
        )
    return Case(directory, analysis, variables, float(getattr(module, "RTOL", DEFAULT_RTOL)))


def compare_tables(table: pd.DataFrame, reference: pd.DataFrame, rtol: float) -> list[str]:
    """Return every difference between a table and its reference.

    Parameters
    ----------
    table : pandas.DataFrame
        The table the package extracted.
    reference : pandas.DataFrame
        The table the other analysis wrote.
    rtol : float
        Relative tolerance of floating point columns.

    Returns
    -------
    list[str]
        One line per difference; empty when the tables agree.
    """
    for name, frame in (("table", table), ("reference", reference)):
        lacking = [column for column in EVENT_KEY if column not in frame.columns]
        if lacking:
            return [f"The {name} lacks the event key column(s) {lacking}."]

    differences = []
    only_table = sorted(set(table.columns) - set(reference.columns))
    only_reference = sorted(set(reference.columns) - set(table.columns))
    if only_table:
        differences.append(f"Columns the reference does not hold: {only_table}.")
    if only_reference:
        differences.append(f"Columns the package does not give: {only_reference}.")

    ours = table.set_index(EVENT_KEY)
    theirs = reference.set_index(EVENT_KEY)
    for name, frame in (("table", ours), ("reference", theirs)):
        if not frame.index.is_unique:
            differences.append(f"The {name} names an event more than once.")
    if differences:
        return differences

    missing = theirs.index.difference(ours.index)
    extra = ours.index.difference(theirs.index)
    if len(missing):
        differences.append(
            f"{len(missing)} event(s) of the reference the package did not keep: "
            f"{list(missing[:SHOWN])}."
        )
    if len(extra):
        differences.append(
            f"{len(extra)} event(s) the package kept that the reference does not: "
            f"{list(extra[:SHOWN])}."
        )

    shared = ours.index.intersection(theirs.index)
    for column in sorted(set(ours.columns) & set(theirs.columns)):
        mine = ours.loc[shared, column].to_numpy()
        other = theirs.loc[shared, column].to_numpy()
        if np.issubdtype(mine.dtype, np.floating) or np.issubdtype(other.dtype, np.floating):
            agree = np.isclose(mine.astype(float), other.astype(float), rtol=rtol, atol=0.0)
        else:
            agree = mine.astype(str) == other.astype(str)
        if not agree.all():
            events = list(shared[~agree][:SHOWN])
            differences.append(
                f"Column {column} differs in {int((~agree).sum())} event(s), first {events}."
            )
    return differences
