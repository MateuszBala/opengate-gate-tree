"""End-to-end use of the filters, on the files a simulation wrote.

A filter is worth having only inside a chain that answers a question about a
run. These read a file and ask such questions, in the form the guide shows.
"""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from opengate_gate_tree import GammaType, GateTree, SourceType, read_hits_trees, read_tree

# The scanner of the a2 scene, from its detector.mac: a ring between 409 and
# 500 mm, 1060 mm long, built of crystals of 3.2 by 20 by 3.2 mm.
SCANNER_INNER_RADIUS = 409.0
SCANNER_OUTER_RADIUS = 500.0
SCANNER_HALF_LENGTH = 530.0

# The prompt gammas of the ops-prompt scene start at 1.3 MeV, its annihilation
# gammas at 511 keV (source.mac).
ANNIHILATION_ENERGY = 0.511


def test_a_question_about_one_event_reads_as_one_chain(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Which hits of one event landed in the scanner, and how much they left there."""
    # ARRANGE
    frame = read_hits_trees(hits_variant_files["multi-run"]).to_dataframe()

    # ACT
    energies = (
        frame.gate.by_event(1, 5)
        .gate.in_sphere((0, 0, 0), 500.0)["edep"]
        .gate.in_range(0.0, ANNIHILATION_ENERGY)
    )

    # ASSERT
    assert isinstance(energies, pd.Series)
    assert len(energies) > 0
    assert (energies <= ANNIHILATION_ENERGY).all()


def test_a_question_about_the_detector_reads_as_one_chain(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Which hits of the first run landed in the ring of the scanner."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a2"], GateTree.HITS).to_dataframe()

    # ACT
    in_the_ring = frame.gate.in_cylinder(
        (0, 0),
        SCANNER_OUTER_RADIUS,
        z_range=(-SCANNER_HALF_LENGTH, SCANNER_HALF_LENGTH),
        inner_radius=SCANNER_INNER_RADIUS,
    ).gate.by_run(0)

    # ASSERT
    assert len(in_the_ring) == len(frame)


def test_a_question_about_the_gammas_of_a_source_reads_as_one_chain(
    positronium_files: Mapping[str, Path],
) -> None:
    """How much energy the prompt gammas of a positronium source left behind.

    The macro of this scene gives a prompt gamma 1.3 MeV, so the answer has to
    hold deposits an annihilation gamma could not make.
    """
    # ARRANGE
    frame = read_tree(positronium_files["ops-prompt"], GateTree.HITS).to_dataframe()

    # ACT
    prompt = frame.gate.with_decay_metadata()
    energies = prompt["edep"][prompt["gammaType"].gate.is_gamma_type(GammaType.PROMPT)]

    # ASSERT
    assert (energies > ANNIHILATION_ENERGY).any()
    assert energies.max() <= 1.3


def test_conditions_combine_before_the_rows_are_cut(
    positronium_files: Mapping[str, Path],
) -> None:
    """Masks are what a question of several parts is built from, then cut once."""
    # ARRANGE
    frame = read_tree(positronium_files["pps-direct"], GateTree.HITS).to_dataframe()

    # ACT
    from_positronium = frame["sourceType"].gate.is_source_type(SourceType.PARA_POSITRONIUM)
    energetic = frame["edep"].gate.is_in_range(0.2, ANNIHILATION_ENERGY)
    selected = frame[from_positronium & energetic]

    # ASSERT
    assert 0 < len(selected) < len(frame)
    assert set(selected["decayIndex"]) == {0}
    assert selected["edep"].between(0.2, ANNIHILATION_ENERGY).all()
