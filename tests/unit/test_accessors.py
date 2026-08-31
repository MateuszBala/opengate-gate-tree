"""Unit tests for the ``gate`` namespace on a pandas column and frame."""

import inspect
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytest

from opengate_gate_tree.io.reader import read_hits_trees, read_tree
from opengate_gate_tree.tree import filters
from opengate_gate_tree.tree.accessors import ACCESSOR_NAME
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import DecayType, GammaType, SourceType

# Every filter of a single column, with arguments chosen so that the answer is
# a proper subset of the scene: a filter that selected everything, or nothing,
# would compare equal to a broken counterpart. The name is the name of both the
# function and the method.
COLUMN_CALLS: list[tuple[str, str, tuple[object, ...]]] = [
    ("is_in_range", "edep", (0.2, 0.4)),
    ("in_range", "edep", (0.2, 0.4)),
    ("is_source_type", "sourceType", (SourceType.ORTHO_POSITRONIUM,)),
    ("select_by_source_type", "sourceType", (SourceType.ORTHO_POSITRONIUM,)),
    ("is_decay_type", "decayType", (DecayType.DEEXCITATION,)),
    ("select_by_decay_type", "decayType", (DecayType.DEEXCITATION,)),
    ("is_gamma_type", "gammaType", (GammaType.PROMPT, GammaType.UNKNOWN)),
    ("select_by_gamma_type", "gammaType", (GammaType.PROMPT,)),
    ("is_process", "processName", ("Compton",)),
    ("select_by_process", "processName", ("Compton",)),
]

# Every filter reading several columns at once, with the scene it is asked
# about: the identity filters need a file of several runs, the rest a file
# where a source wrote its metadata for some of the rows and not for others.
FRAME_CALLS: list[tuple[str, str, tuple[object, ...]]] = [
    ("is_in_box", "positronium", ((100, 0, 0), 200)),
    ("in_box", "positronium", ((100, 0, 0), 200)),
    ("is_in_sphere", "positronium", ((0, 0, 0), 215.0)),
    ("in_sphere", "positronium", ((0, 0, 0), 215.0)),
    ("is_in_cylinder", "positronium", ((0, 0), 250.0, (-100.0, 100.0), 100.0)),
    ("in_cylinder", "positronium", ((0, 0), 250.0, (-100.0, 100.0), 100.0)),
    ("is_from_run", "runs", (1,)),
    ("by_run", "runs", (1,)),
    ("is_from_event", "runs", (1, 5)),
    ("by_event", "runs", (1, 5)),
    ("has_decay_metadata", "positronium", ()),
    ("with_decay_metadata", "positronium", ()),
]


@pytest.fixture(scope="module")
def hits(positronium_files: Mapping[str, Path]) -> pd.DataFrame:
    """Return a scene holding every branch the filters read.

    This one was written by a source configured with every component at once,
    so its rows differ in what a gamma was and in whether it carries decay
    metadata at all - which is what lets a filter answer part of the scene.
    """
    return read_tree(positronium_files["all-variants"], GateTree.HITS).to_dataframe()


@pytest.fixture(scope="module")
def scenes(hits: pd.DataFrame, hits_variant_files: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    """Return the scenes the frame filters are asked about, by name."""
    return {
        "positronium": hits,
        "runs": read_hits_trees(hits_variant_files["multi-run"]).to_dataframe(),
    }


def _selected(answer: pd.Series | pd.DataFrame) -> int:
    """Return how many rows an answer covers, whichever of the two it is."""
    if isinstance(answer, pd.DataFrame):
        return len(answer)
    if answer.dtype == bool:
        return int(answer.sum())
    return len(answer)


def public_filters() -> list[str]:
    """Return the names of the filters the package offers."""
    return [
        name
        for name, value in vars(filters).items()
        if not name.startswith("_")
        and inspect.isfunction(value)
        and value.__module__ == filters.__name__
    ]


def test_the_namespace_is_there_after_importing_the_package(hits: pd.DataFrame) -> None:
    """Importing the package is what puts the filters within reach."""
    # ARRANGE
    # No additional setup required.

    # ACT
    on_column = hasattr(hits["edep"], ACCESSOR_NAME)
    on_frame = hasattr(hits, ACCESSOR_NAME)

    # ASSERT
    assert on_column
    assert on_frame


@pytest.mark.parametrize(
    ("name", "column", "arguments"),
    COLUMN_CALLS,
    ids=[name for name, _, _ in COLUMN_CALLS],
)
def test_a_column_method_answers_what_its_function_answers(
    name: str,
    column: str,
    arguments: tuple[object, ...],
    hits: pd.DataFrame,
) -> None:
    """The namespace is a way of writing, so it must add nothing of its own."""
    # ARRANGE
    values = hits[column]

    # ACT
    from_method = getattr(values.gate, name)(*arguments)
    from_function = getattr(filters, name)(values, *arguments)

    # ASSERT
    assert from_method.equals(from_function)
    assert 0 < _selected(from_method) < len(values)


@pytest.mark.parametrize(
    ("name", "scene", "arguments"),
    FRAME_CALLS,
    ids=[name for name, _, _ in FRAME_CALLS],
)
def test_a_frame_method_answers_what_its_function_answers(
    name: str,
    scene: str,
    arguments: tuple[object, ...],
    scenes: dict[str, pd.DataFrame],
) -> None:
    """The same, for the filters that read several columns.

    A mask carries the index of the frame it was built from whatever it holds,
    so comparing the two answers by their index alone would pass for a method
    answering the opposite of its function. The values are what is compared.
    """
    # ARRANGE
    frame = scenes[scene]

    # ACT
    from_method = getattr(frame.gate, name)(*arguments)
    from_function = getattr(filters, name)(frame, *arguments)

    # ASSERT
    assert from_method.equals(from_function)
    assert 0 < _selected(from_method) < len(frame)


def test_every_filter_is_reachable_through_the_namespace() -> None:
    """A filter added without its method would be reachable one way only."""
    # ARRANGE
    from opengate_gate_tree.tree.accessors import GateFrameAccessor, GateSeriesAccessor

    reachable = {
        name
        for accessor in (GateSeriesAccessor, GateFrameAccessor)
        for name in vars(accessor)
        if not name.startswith("_")
    }

    # ACT
    offered = set(public_filters())

    # ASSERT
    assert offered == reachable


def test_columns_chain(hits: pd.DataFrame) -> None:
    """A column in, a column out, which is the shape the API promises."""
    # ARRANGE
    energies = hits["edep"]

    # ACT
    narrowed = energies.gate.in_range(0.0, 1.0).gate.in_range(0.2, 0.4)

    # ASSERT
    assert isinstance(narrowed, pd.Series)
    assert list(narrowed) == list(energies.gate.in_range(0.2, 0.4))


def test_frames_chain(hits: pd.DataFrame) -> None:
    """A frame in, a frame out, so shapes and identities compose.

    Every stage removes rows, so a stage that stopped filtering would be seen:
    the sphere leaves out the hits furthest from the centre, and the metadata
    the gammas of the component that carries none.
    """
    # ARRANGE
    # No additional setup required.

    # ACT
    selected = hits.gate.in_sphere((0, 0, 0), 215.0).gate.by_run(0).gate.with_decay_metadata()

    # ASSERT
    assert isinstance(selected, pd.DataFrame)
    by_hand = hits[
        filters.is_in_sphere(hits, (0, 0, 0), 215.0)
        & filters.is_from_run(hits, 0)
        & filters.has_decay_metadata(hits)
    ]
    assert selected.equals(by_hand)
    assert 0 < len(selected) < len(hits)


def test_a_column_of_a_selected_frame_carries_the_namespace(hits: pd.DataFrame) -> None:
    """The two accessors meet where a frame filter is followed by a column one."""
    # ARRANGE
    # No additional setup required.

    # ACT
    energies = hits.gate.with_decay_metadata()["edep"].gate.in_range(0.2, 0.4)

    # ASSERT
    assert isinstance(energies, pd.Series)
    assert 0 < len(energies) < len(hits)


def test_importing_the_package_registers_the_namespace_quietly() -> None:
    """Registering a name in pandas is shared state, and it has to be uneventful."""
    # ARRANGE
    program = (
        "import warnings;"
        "warnings.simplefilter('error');"
        "import pandas as pd, opengate_gate_tree;"
        "print(hasattr(pd.Series(dtype=float), 'gate'), hasattr(pd.DataFrame(), 'gate'))"
    )

    # ACT
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )

    # ASSERT
    assert result.stdout.strip() == "True True"
    assert result.stderr == ""


def test_the_namespace_answers_to_the_name_the_package_declares() -> None:
    """The name is a constant, so nothing has to guess it."""
    # ARRANGE
    values = pd.Series([1.0, 2.0])

    # ACT
    accessor = getattr(values, ACCESSOR_NAME)

    # ASSERT
    assert ACCESSOR_NAME == "gate"
    assert isinstance(accessor.in_range(1.0, 2.0), pd.Series)


def test_the_arguments_of_a_method_are_the_arguments_of_its_function() -> None:
    """A method taking other arguments than its function would be a second API.

    Names, order, kinds and defaults are all compared: a method whose default
    differed from its function's would answer a different question when asked
    the short way.
    """
    # ARRANGE
    from opengate_gate_tree.tree.accessors import GateFrameAccessor, GateSeriesAccessor

    mismatched: list[str] = []

    # ACT
    for accessor, first in ((GateSeriesAccessor, "values"), (GateFrameAccessor, "frame")):
        for name in vars(accessor):
            if name.startswith("_"):
                continue
            method = list(inspect.signature(getattr(accessor, name)).parameters.values())[1:]
            function = list(inspect.signature(getattr(filters, name)).parameters.values())[1:]
            if method != function:
                mismatched.append(f"{first}.{name}: {method} against {function}")

    # ASSERT
    assert mismatched == []


def test_a_sequence_of_columns_is_still_accepted(hits: pd.DataFrame) -> None:
    """A shape takes the columns it should read, through the namespace too."""
    # ARRANGE
    local: Sequence[str] = ("localPosX", "localPosY", "localPosZ")

    # ACT
    from_method = hits.gate.is_in_box((0, 0, 0), 1000, local)
    from_function = filters.is_in_box(hits, (0, 0, 0), 1000, local)

    # ASSERT
    assert list(from_method) == list(from_function)
