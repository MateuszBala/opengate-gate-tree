"""Shared test fixtures.

Two sources of test data are used together:

- a real GATE 9.x file, trimmed to 2000 entries, which guarantees that the
  readers work against branch names, types and file layout produced by GATE
- files generated with ``uproot``, which cover cases the real file does not
  contain, such as branches of varying length or a missing tree

The files in ``fixtures/hits-variants`` add one output per structure the "Hits"
tree can have. Their provenance and the reason one of them is not trimmed are
described in the README next to them.

The files in ``fixtures/positronium`` add one output per way a
``PositroniumSource`` can be configured, for the branches that describe what a
gamma came from.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import numpy.typing as npt
import pytest
import uproot
from uproot.interpretation.strings import AsStrings

# Directory holding the binary test fixtures.
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Real GATE 9.x output file, trimmed to 2000 entries of the "Hits" tree.
GATE_HITS_FIXTURE = FIXTURES_DIR / "data-only-hits-tree-small-size.root"

# One simulation output per structure the "Hits" tree can have.
HITS_VARIANTS_DIR = FIXTURES_DIR / "hits-variants"

# One simulation output per way a PositroniumSource can be configured.
POSITRONIUM_DIR = FIXTURES_DIR / "positronium"

TreeColumns = Mapping[str, npt.NDArray[Any]]


@dataclass(frozen=True)
class HitsVariantLayout:
    """Measured properties of one "Hits" tree variant fixture.

    Attributes
    ----------
    key : str
        Short name the tests refer to the fixture by.
    file_name : str
        Name of the file in ``fixtures/hits-variants``.
    reference : str
        Label the reference material gives the structure, such as ``"A1"``.
    tree_names : tuple[str, ...]
        Trees stored in the file, in file order.
    entries : int
        Number of entries of every tree in the file.
    branch_count : int
        Number of branches of every tree in the file.
    """

    key: str
    file_name: str
    reference: str
    tree_names: tuple[str, ...]
    entries: int
    branch_count: int

    @property
    def path(self) -> Path:
        """Path of the fixture file."""
        return HITS_VARIANTS_DIR / self.file_name


# Properties of the variant fixtures, measured on the files they were cut from.
# "b1-tree-common-output.root" keeps every entry of the simulation, because the
# GateToTree layout cannot be rewritten by uproot; see the README next to it.
HITS_VARIANT_LAYOUTS: tuple[HitsVariantLayout, ...] = (
    HitsVariantLayout("a1", "a1-no-system.root", "A1", ("Hits",), 500, 40),
    HitsVariantLayout("a2", "a2-system.root", "A2", ("Hits",), 500, 46),
    HitsVariantLayout("a3", "a3-system-septal.root", "A3", ("Hits",), 500, 47),
    HitsVariantLayout("a4", "a4-no-system-cc.root", "A4", ("Hits",), 500, 30),
    HitsVariantLayout("a5", "a5-system-cc.root", "A5", ("Hits",), 500, 36),
    HitsVariantLayout("b1", "b1-tree-common-output.root", "B1", ("tree",), 4441, 54),
    HitsVariantLayout(
        "multi-run",
        "name-multi-run.root",
        "A1",
        ("Hits", "Hits_run1", "Hits_run2"),
        500,
        40,
    ),
    HitsVariantLayout(
        "multi-sd",
        "name-multi-sd.root",
        "A1",
        ("Hits_DET_INNER", "Hits_DET_OUTER"),
        500,
        40,
    ),
)


# Branches GATE writes as double precision.
HITS_FLOAT64_BRANCHES = frozenset({"trackLocalTime", "time"})

# Branches GATE writes as single precision.
HITS_FLOAT32_BRANCHES = frozenset(
    {
        "edep",
        "stepLength",
        "trackLength",
        "posX",
        "posY",
        "posZ",
        "localPosX",
        "localPosY",
        "localPosZ",
        "sourcePosX",
        "sourcePosY",
        "sourcePosZ",
        "momDirX",
        "momDirY",
        "momDirZ",
        "axialPos",
        "rotationAngle",
        "sourceEnergy",
        "energyFinal",
        "energyIniT",
    }
)

# Branches GATE writes as text.
HITS_TEXT_BRANCHES = frozenset(
    {"processName", "comptVolName", "RayleighVolName", "postStepProcess"}
)

# Branches GATE writes as a fixed-width array, mapped to their type.
HITS_ARRAY_BRANCHES = {"volumeID": "int32[10]"}

# Values used for text branches of generated trees.
HITS_TEXT_VALUES = ("Compton", "PhotoElectric", "Transportation")


def expected_hits_branch_type(name: str) -> str:
    """Return the type GATE writes a branch of the given name with.

    The rule is the one measured on the reference files: the type follows the
    name of the branch. It is kept on the test side on purpose, so that the
    schemas of the package are compared against something derived from the
    files rather than from themselves.

    Parameters
    ----------
    name : str
        Branch name.

    Returns
    -------
    str
        Short type name, as :func:`branch_type_name` reports it.
    """
    if name in HITS_ARRAY_BRANCHES:
        return HITS_ARRAY_BRANCHES[name]
    if name in HITS_TEXT_BRANCHES:
        return "text"
    if name in HITS_FLOAT64_BRANCHES:
        return "float64"
    if name in HITS_FLOAT32_BRANCHES:
        return "float32"
    return "int32"


@dataclass(frozen=True)
class PositroniumLayout:
    """Measured properties of one PositroniumSource fixture.

    Attributes
    ----------
    key : str
        Short name the tests refer to the fixture by.
    file_name : str
        Name of the file in ``fixtures/positronium``.
    scene : str
        How the source was configured.
    source_types, decay_types, gamma_types, decay_indices : tuple[int, ...]
        Distinct values the file holds in each of the four branches, sorted.
    """

    key: str
    file_name: str
    scene: str
    source_types: tuple[int, ...]
    decay_types: tuple[int, ...]
    gamma_types: tuple[int, ...]
    decay_indices: tuple[int, ...]

    @property
    def path(self) -> Path:
        """Path of the fixture file."""
        return POSITRONIUM_DIR / self.file_name


# Values of the four branches of every PositroniumSource fixture, measured on
# the simulation outputs they were cut from. Cutting to 500 entries kept the
# whole set of values in every file; see the README next to them.
POSITRONIUM_LAYOUTS: tuple[PositroniumLayout, ...] = (
    PositroniumLayout("pps", "pps.root", "pPs, two gammas", (2,), (1,), (2,), (0,)),
    PositroniumLayout("ops", "ops.root", "oPs, three gammas", (3,), (1,), (2,), (0,)),
    PositroniumLayout(
        "pps-prompt", "pps-prompt.root", "pPs with a prompt gamma", (2,), (2,), (2, 3), (0,)
    ),
    PositroniumLayout(
        "ops-prompt", "ops-prompt.root", "oPs with a prompt gamma", (3,), (2,), (2, 3), (0,)
    ),
    PositroniumLayout(
        "pps-direct", "pps-direct.root", "half pPs, half direct", (2, 4), (1,), (2,), (0, 1)
    ),
    PositroniumLayout(
        "ops-direct", "ops-direct.root", "half oPs, half direct", (3, 4), (1,), (2,), (0, 1)
    ),
    PositroniumLayout(
        "back-to-back", "back-to-back.root", "back-to-back, no positronium", (0,), (0,), (0,), (-1,)
    ),
    PositroniumLayout(
        "all-variants",
        "all-variants.root",
        "all seven channels",
        (0, 2, 3, 4),
        (0, 1, 2),
        (0, 2, 3),
        (-1, 0, 1),
    ),
)

# Branches a PositroniumSource fills, mapped to the layout field holding their
# measured values.
POSITRONIUM_BRANCH_FIELDS: Mapping[str, str] = {
    "sourceType": "source_types",
    "decayType": "decay_types",
    "gammaType": "gamma_types",
    "decayIndex": "decay_indices",
}


def branch_type_name(interpretation: Any) -> str:
    """Return a short name for the type an uproot interpretation reads.

    Parameters
    ----------
    interpretation : Any
        Interpretation of a branch, as reported by uproot.

    Returns
    -------
    str
        One of ``"int32"``, ``"float32"``, ``"float64"``, ``"text"`` or
        ``"<type>[<width>]"`` for a fixed-width array branch.
    """
    if isinstance(interpretation, AsStrings):
        return "text"
    dtype = interpretation.to_dtype
    scalar = dtype.base.newbyteorder("=")
    if dtype.shape:
        return f"{scalar.name}[{dtype.shape[0]}]"
    return str(scalar.name)


@dataclass(frozen=True)
class GateHitsLayout:
    """Known properties of the "Hits" tree in the real GATE fixture file."""

    entries: int
    branch_count: int
    volume_id_width: int
    tree_names: tuple[str, ...]


@pytest.fixture(scope="session")
def gate_hits_file() -> Path:
    """Return the path of the real GATE fixture file."""
    return GATE_HITS_FIXTURE


@pytest.fixture(scope="session")
def gate_hits_layout() -> GateHitsLayout:
    """Return the properties the real GATE fixture file is expected to have."""
    return GateHitsLayout(
        entries=2000,
        branch_count=40,
        volume_id_width=10,
        tree_names=("pet_data", "Hits", "OpticalData"),
    )


@pytest.fixture(scope="session")
def hits_variant_layouts() -> Mapping[str, HitsVariantLayout]:
    """Return the variant fixtures, keyed by their short name."""
    return {layout.key: layout for layout in HITS_VARIANT_LAYOUTS}


@pytest.fixture(scope="session")
def hits_variant_files() -> Mapping[str, Path]:
    """Return the paths of the variant fixtures, keyed by their short name."""
    return {layout.key: layout.path for layout in HITS_VARIANT_LAYOUTS}


@pytest.fixture(scope="session")
def positronium_layouts() -> Mapping[str, PositroniumLayout]:
    """Return the PositroniumSource fixtures, keyed by their short name."""
    return {layout.key: layout for layout in POSITRONIUM_LAYOUTS}


@pytest.fixture(scope="session")
def positronium_files() -> Mapping[str, Path]:
    """Return the paths of the PositroniumSource fixtures, keyed by short name."""
    return {layout.key: layout.path for layout in POSITRONIUM_LAYOUTS}


@pytest.fixture
def make_gate_root_file(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory writing a ROOT file with the given trees.

    Columns may be one-dimensional arrays, two-dimensional arrays of a fixed
    width, or object arrays of strings. A tree without entries is written with
    its branches declared but no data appended, because uproot cannot extend a
    tree that holds a text branch with an empty batch.
    """

    def write_root_file(
        trees: Mapping[str, TreeColumns],
        file_name: str = "generated.root",
    ) -> Path:
        path = tmp_path / file_name
        with uproot.recreate(path) as out:
            for tree_name, columns in trees.items():
                branch_types, branch_data = _branch_specification(columns)
                out.mktree(tree_name, branch_types)
                if _entry_count(columns):
                    out[tree_name].extend(branch_data)
        return path

    return write_root_file


@pytest.fixture
def make_hits_root_file(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory writing a ROOT file with trees of the given branches.

    Columns are generated from the branch names, following the rule GATE
    follows, so a tree is described by its branch names alone. Names listed in
    ``jagged`` are written as branches of varying length, which the package
    cannot load but can look at.

    Names holding brackets are refused: the uproot writer reads them as an
    array dimension, which silently produces a file that cannot be read back.
    Tests needing the layout that uses such names work on the B1 fixture.
    """

    def write_hits_file(
        trees: Mapping[str, Sequence[str]],
        entries: int = 5,
        file_name: str = "hits.root",
        jagged: Sequence[str] = (),
    ) -> Path:
        bracketed = sorted({name for names in trees.values() for name in names if "[" in name})
        if bracketed:
            raise ValueError(
                f"uproot writes brackets in a branch name as an array dimension, so {bracketed} "
                f"cannot be generated; use the B1 fixture for that layout."
            )
        path = tmp_path / file_name
        with uproot.recreate(path) as out:
            for tree_name, names in trees.items():
                columns = {name: _hits_column(name, entries) for name in names}
                branch_types, branch_data = _branch_specification(columns)
                for name in jagged:
                    branch_types[name] = "var * float64"
                    branch_data[name] = ak.Array([[float(index)] for index in range(entries)])
                out.mktree(tree_name, branch_types)
                if entries:
                    out[tree_name].extend(branch_data)
        return path

    return write_hits_file


@pytest.fixture
def make_jagged_root_file(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory writing a ROOT file with a branch of varying length."""

    def write_jagged_file(
        tree_name: str = "Hits",
        file_name: str = "jagged.root",
    ) -> Path:
        path = tmp_path / file_name
        with uproot.recreate(path) as out:
            out.mktree(tree_name, {"eventID": np.int32, "hitTimes": "var * float64"})
            out[tree_name].extend(
                {
                    "eventID": np.arange(3, dtype=np.int32),
                    "hitTimes": ak.Array([[1.0], [2.0, 3.0], []]),
                }
            )
        return path

    return write_jagged_file


def _hits_column(name: str, entries: int) -> npt.NDArray[Any]:
    """Return a generated column for a branch of the given name."""
    if name in HITS_ARRAY_BRANCHES:
        return np.arange(entries * 10, dtype=np.int32).reshape(entries, 10)
    if name in HITS_TEXT_BRANCHES:
        values = [HITS_TEXT_VALUES[index % len(HITS_TEXT_VALUES)] for index in range(entries)]
        return np.array(values, dtype=object)
    if name in HITS_FLOAT64_BRANCHES:
        return np.arange(entries, dtype=np.float64) / 8
    if name in HITS_FLOAT32_BRANCHES:
        return np.arange(entries, dtype=np.float32) / 4
    return np.arange(entries, dtype=np.int32)


def _branch_specification(
    columns: TreeColumns,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split columns into uproot branch types and the matching data."""
    branch_types: dict[str, Any] = {}
    branch_data: dict[str, Any] = {}

    for name, column in columns.items():
        if column.dtype == object:
            branch_types[name] = str
            branch_data[name] = [str(value) for value in column]
        elif column.ndim == 2:
            branch_types[name] = (column.dtype, column.shape[1:])
            branch_data[name] = column
        else:
            branch_types[name] = column.dtype
            branch_data[name] = column

    return branch_types, branch_data


def _entry_count(columns: TreeColumns) -> int:
    """Return the number of entries described by the columns."""
    for column in columns.values():
        return int(column.shape[0])
    return 0
