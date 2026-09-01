"""Unit tests for the conversions between units."""

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.units import (
    GATE_UNITS,
    MeV_to_keV,
    cm_to_m,
    cm_to_mm,
    deg_to_rad,
    keV_to_MeV,
    m_to_cm,
    m_to_mm,
    mm_to_cm,
    mm_to_m,
    ms_to_ns,
    ms_to_s,
    ns_to_ms,
    ns_to_s,
    rad_to_deg,
    s_to_ms,
    s_to_ns,
)

# The scenes the fixtures come from, in the units their macros are written in.
# A conversion has to turn one into the other, so these are the answers known
# before any file is read.
A1_SHELL_INNER_RADIUS_MM = 200.0  # a1-no-system/detector.mac: setRmin 0.2 m
A1_SHELL_INNER_RADIUS_M = 0.2
A2_BORE_RADIUS_MM = 409.0  # a2-system/detector.mac: setRmin 409.0 mm
A2_BORE_RADIUS_CM = 40.9
RUN_LENGTH_S = 0.02  # application.mac: setTimeSlice 0.02 s
RUN_LENGTH_MS = 20.0
RUN_LENGTH_NS = 2e7

# The energy of an annihilation gamma, which is what the fixtures deposit.
ANNIHILATION_ENERGY_MEV = 0.511
ANNIHILATION_ENERGY_KEV = 511.0

# Every conversion with a value it is pinned by. A round trip cannot pin one:
# both directions read the same constant, so a wrong constant survives it.
KNOWN_VALUES: list[tuple[Callable[[float], float], float, float]] = [
    (MeV_to_keV, ANNIHILATION_ENERGY_MEV, ANNIHILATION_ENERGY_KEV),
    (keV_to_MeV, ANNIHILATION_ENERGY_KEV, ANNIHILATION_ENERGY_MEV),
    (mm_to_cm, A2_BORE_RADIUS_MM, A2_BORE_RADIUS_CM),
    (cm_to_mm, A2_BORE_RADIUS_CM, A2_BORE_RADIUS_MM),
    (mm_to_m, A1_SHELL_INNER_RADIUS_MM, A1_SHELL_INNER_RADIUS_M),
    (m_to_mm, A1_SHELL_INNER_RADIUS_M, A1_SHELL_INNER_RADIUS_MM),
    (cm_to_m, A2_BORE_RADIUS_CM, 0.409),
    (m_to_cm, A1_SHELL_INNER_RADIUS_M, 20.0),
    (s_to_ms, RUN_LENGTH_S, RUN_LENGTH_MS),
    (ms_to_s, RUN_LENGTH_MS, RUN_LENGTH_S),
    (s_to_ns, RUN_LENGTH_S, RUN_LENGTH_NS),
    (ns_to_s, RUN_LENGTH_NS, RUN_LENGTH_S),
    (ms_to_ns, RUN_LENGTH_MS, RUN_LENGTH_NS),
    (ns_to_ms, RUN_LENGTH_NS, RUN_LENGTH_MS),
    (rad_to_deg, np.pi, 180.0),
    (deg_to_rad, 180.0, np.pi),
]

VALUE_IDS = [f"{conversion.__name__}" for conversion, _, _ in KNOWN_VALUES]

# Every conversion with the one that undoes it.
INVERSE_PAIRS: list[tuple[Callable[[float], float], Callable[[float], float]]] = [
    (MeV_to_keV, keV_to_MeV),
    (mm_to_cm, cm_to_mm),
    (mm_to_m, m_to_mm),
    (cm_to_m, m_to_cm),
    (s_to_ms, ms_to_s),
    (s_to_ns, ns_to_s),
    (ms_to_ns, ns_to_ms),
    (rad_to_deg, deg_to_rad),
]

PAIR_IDS = [f"{forward.__name__}" for forward, _ in INVERSE_PAIRS]


@pytest.fixture(scope="module")
def hits(hits_variant_files: Mapping[str, Path]) -> pd.DataFrame:
    """Return a scene whose geometry and timing the macros state."""
    return read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()


def test_energy_is_read_in_the_unit_a_spectrum_is_drawn_in() -> None:
    """The annihilation line is 511 keV, and that is the number it has to give."""
    # ARRANGE
    # No additional setup required.

    # ACT
    in_kev = MeV_to_keV(ANNIHILATION_ENERGY_MEV)
    in_mev = keV_to_MeV(ANNIHILATION_ENERGY_KEV)

    # ASSERT
    assert in_kev == pytest.approx(ANNIHILATION_ENERGY_KEV)
    assert in_mev == pytest.approx(ANNIHILATION_ENERGY_MEV)


def test_length_is_read_in_the_unit_the_macro_was_written_in() -> None:
    """The shell of the a1 scene is 0.2 m in its macro and 200 mm in the file."""
    # ARRANGE
    # No additional setup required.

    # ACT
    in_metres = mm_to_m(A1_SHELL_INNER_RADIUS_MM)
    in_centimetres = mm_to_cm(A2_BORE_RADIUS_MM)

    # ASSERT
    assert in_metres == pytest.approx(A1_SHELL_INNER_RADIUS_M)
    assert in_centimetres == pytest.approx(A2_BORE_RADIUS_CM)
    assert m_to_mm(A1_SHELL_INNER_RADIUS_M) == pytest.approx(A1_SHELL_INNER_RADIUS_MM)


def test_time_is_read_in_the_units_a_run_and_a_resolution_are_quoted_in() -> None:
    """A run is quoted in seconds, a time resolution in nanoseconds."""
    # ARRANGE
    # No additional setup required.

    # ACT
    in_milliseconds = s_to_ms(RUN_LENGTH_S)
    in_nanoseconds = s_to_ns(RUN_LENGTH_S)

    # ASSERT
    assert in_milliseconds == pytest.approx(RUN_LENGTH_MS)
    assert in_nanoseconds == pytest.approx(RUN_LENGTH_NS)
    assert ms_to_ns(RUN_LENGTH_MS) == pytest.approx(RUN_LENGTH_NS)


def test_the_shell_of_the_a1_scene_measures_what_its_macro_says(hits: pd.DataFrame) -> None:
    """The conversion is checked against a file, not only against a constant.

    Every hit of this scene happened in a lead shell whose macro gives the
    inner radius as 0.2 m. Read in metres, no hit lies inside that radius.
    """
    # ARRANGE
    radius_mm = np.sqrt(hits["posX"] ** 2 + hits["posY"] ** 2 + hits["posZ"] ** 2)

    # ACT
    radius_m = mm_to_m(radius_mm)

    # ASSERT
    assert radius_m.min() >= A1_SHELL_INNER_RADIUS_M
    assert radius_m.min() == pytest.approx(A1_SHELL_INNER_RADIUS_M, abs=1e-4)
    assert mm_to_cm(radius_mm).max() == pytest.approx(radius_m.max() * 100.0)


@pytest.mark.parametrize(("forward", "backward"), INVERSE_PAIRS, ids=PAIR_IDS)
def test_a_conversion_is_undone_by_its_counterpart(
    forward: Callable[[float], float],
    backward: Callable[[float], float],
) -> None:
    """Every unit of a family reaches every other one, and comes back."""
    # ARRANGE
    values = 1.234

    # ACT
    there_and_back = backward(forward(values))

    # ASSERT
    assert there_and_back == pytest.approx(values)


@pytest.mark.parametrize(("forward", "backward"), INVERSE_PAIRS, ids=PAIR_IDS)
def test_a_conversion_answers_with_the_kind_it_was_given(
    forward: Callable[[float], float],
    backward: Callable[[float], float],
) -> None:
    """A number in, a number out - not a NumPy scalar wearing its clothes."""
    # ARRANGE
    # No additional setup required.

    # ACT
    converted = forward(2.0)

    # ASSERT
    assert type(converted) is float
    assert type(backward(2.0)) is float


def test_a_column_keeps_its_index_and_its_name(hits: pd.DataFrame) -> None:
    """A conversion is one step of a chain, so nothing about the column is lost."""
    # ARRANGE
    energies = hits["edep"].iloc[[3, 1, 2]]

    # ACT
    in_kev = MeV_to_keV(energies)

    # ASSERT
    assert isinstance(in_kev, pd.Series)
    assert list(in_kev.index) == [3, 1, 2]
    assert in_kev.name == "edep"
    assert list(in_kev) == pytest.approx([value * 1000.0 for value in energies])


def test_an_array_stays_an_array_of_the_same_shape() -> None:
    """Positions are read as arrays, and an array of them converts in one call."""
    # ARRANGE
    positions_mm = np.array([[200.0, 0.0, 0.0], [0.0, 100.0, 0.0]], dtype=np.float32)

    # ACT
    positions_cm = mm_to_cm(positions_mm)

    # ASSERT
    assert isinstance(positions_cm, np.ndarray)
    assert positions_cm.shape == positions_mm.shape
    assert positions_cm.tolist() == [[20.0, 0.0, 0.0], [0.0, 10.0, 0.0]]


def test_a_conversion_computes_in_float64(hits: pd.DataFrame) -> None:
    """A GATE file holds float32 columns, and a time in ns does not fit one.

    Positions and energies are written as float32; times are float64 already.
    Sixty seconds is 6e10 ns, and a float32 mantissa carries 24 bits, so that
    value would land on a grid four microseconds wide - the last assertion is
    what that costs. Every conversion answers in float64 whatever it was given.
    """
    # ARRANGE
    energies = hits["edep"]
    one_minute = np.float32(60.0)

    # ACT
    in_kev = MeV_to_keV(energies)

    # ASSERT
    assert energies.dtype == np.float32
    assert in_kev.dtype == np.float64
    assert s_to_ns(hits["time"]).dtype == np.float64
    assert s_to_ns(float(one_minute)) == pytest.approx(6e10)
    assert float(np.float32(6e10)) != 6e10


def test_the_shortest_time_in_the_file_is_read_in_nanoseconds(hits: pd.DataFrame) -> None:
    """The other end of the scale: microseconds of flight, read as nanoseconds."""
    # ARRANGE
    earliest = float(hits["time"].min())

    # ACT
    in_nanoseconds = s_to_ns(earliest)

    # ASSERT
    assert in_nanoseconds == pytest.approx(earliest * 1e9)
    assert 1000.0 < in_nanoseconds < 10000.0


def test_a_missing_value_stays_missing() -> None:
    """A conversion says nothing about a value the file did not hold."""
    # ARRANGE
    values = pd.Series([0.511, np.nan])

    # ACT
    converted = MeV_to_keV(values)

    # ASSERT
    assert converted[0] == pytest.approx(511.0)
    assert np.isnan(converted[1])
    assert np.isnan(MeV_to_keV(float("nan")))


@pytest.mark.parametrize(
    ("ours", "theirs"), [(rad_to_deg, np.rad2deg), (deg_to_rad, np.deg2rad)], ids=["rad", "deg"]
)
def test_an_angle_is_converted_by_numpy(
    ours: Callable[[np.ndarray], np.ndarray],
    theirs: Callable[[np.ndarray], np.ndarray],
) -> None:
    """The angle pair is a wrapper, so it must answer what NumPy answers."""
    # ARRANGE
    angles = np.array([0.0, 0.5, 1.0, np.pi, 180.0])

    # ACT
    converted = ours(angles)

    # ASSERT
    assert np.array_equal(converted, theirs(angles))


def test_a_half_turn_is_a_hundred_and_eighty_degrees() -> None:
    """The one angle everybody checks first."""
    # ARRANGE
    # No additional setup required.

    # ACT
    degrees = rad_to_deg(np.pi)
    radians = deg_to_rad(180.0)

    # ASSERT
    assert degrees == pytest.approx(180.0)
    assert radians == pytest.approx(np.pi)


def test_the_units_gate_writes_are_stated_once() -> None:
    """Every conversion of a branch starts from what GATE wrote it in."""
    # ARRANGE
    # No additional setup required.

    # ACT
    units = dict(GATE_UNITS)

    # ASSERT
    assert units == {"energy": "MeV", "length": "mm", "time": "s"}
    with pytest.raises(TypeError):
        GATE_UNITS["energy"] = "keV"  # type: ignore[index]


@pytest.mark.parametrize(("conversion", "given", "expected"), KNOWN_VALUES, ids=VALUE_IDS)
def test_a_conversion_turns_a_known_value_into_the_known_answer(
    conversion: Callable[[float], float],
    given: float,
    expected: float,
) -> None:
    """The factor is the whole of the behaviour, so every one is pinned here.

    A round trip is blind to it: both directions of a pair read the same
    constant, so a wrong constant undoes itself. The numbers are the ones the
    macros of the fixtures give, apart from the two that no macro states -
    40.9 cm in metres and 0.2 m in centimetres - and half a turn.
    """
    # ARRANGE
    # No additional setup required.

    # ACT
    converted = conversion(given)

    # ASSERT
    assert converted == pytest.approx(expected)
