"""End-to-end use of the converters and the vectors, on simulated files.

A conversion or an angle is worth having inside a chain that answers a question
about a run. These read a file and ask such questions in the form the guide
shows them.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree import (
    GateTree,
    angle_between,
    momentum_direction_from_positions,
    polarization_direction,
    rad_to_deg,
    read_tree,
)

# The scanner of the a2 scene, from its detector.mac: a ring between 409 and
# 500 mm, which is 40.9 to 50 cm.
SCANNER_BORE_RADIUS_MM = 409.0
SCANNER_BORE_RADIUS_CM = 40.9

# The annihilation line, in both units.
ANNIHILATION_ENERGY_MEV = 0.511
ANNIHILATION_ENERGY_KEV = 511.0

# What the identifiers of one track are.
TRACK_KEY = ["runID", "eventID", "trackID"]
POSITION = ["posX", "posY", "posZ"]
SOURCE = ["sourcePosX", "sourcePosY", "sourcePosZ"]


def test_a_spectrum_is_read_in_the_unit_it_is_drawn_in(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Energies deposited in the scanner, in keV, for the rows that carry them."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a2"], GateTree.HITS).to_dataframe()

    # ACT
    energies = (
        frame.gate.in_cylinder((0, 0), 500.0, inner_radius=SCANNER_BORE_RADIUS_MM)["edep"]
        .gate.in_range(0.0, ANNIHILATION_ENERGY_MEV)
        .gate.MeV_to_keV()
    )

    # ASSERT
    assert isinstance(energies, pd.Series)
    assert len(energies) > 0
    assert energies.max() <= ANNIHILATION_ENERGY_KEV


def test_a_detector_measured_in_centimetres(hits_variant_files: Mapping[str, Path]) -> None:
    """The macro gives the bore in millimetres; an analysis may want centimetres."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a2"], GateTree.HITS).to_dataframe()

    # ACT
    radius_cm = frame.gate.position().norm().gate.mm_to_cm()

    # ASSERT
    assert radius_cm.min() >= SCANNER_BORE_RADIUS_CM
    assert radius_cm.index.equals(frame.index)


def test_the_angular_distribution_of_a_scene(hits_variant_files: Mapping[str, Path]) -> None:
    """Where the hits sit, read as angles rather than as coordinates."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()

    # ACT
    spherical = frame.gate.position().spherical()

    # ASSERT
    assert list(spherical.columns) == ["radius", "polar", "azimuth"]
    assert spherical["polar"].gate.rad_to_deg().between(0.0, 180.0).all()
    assert spherical["azimuth"].gate.rad_to_deg().max() > 180.0


def test_the_angle_between_where_a_hit_is_and_where_it_went(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """A frame in, an angle in degrees out, in one chain."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()
    selected = frame.gate.by_run(0)

    # ACT
    angles = selected.gate.position().angle_to(selected.gate.momentum_direction())

    # ASSERT
    assert isinstance(angles, pd.Series)
    assert angles.index.equals(selected.index)
    assert angles.gate.rad_to_deg().between(0.0, 180.0).all()


def test_a_scattering_read_from_one_file_end_to_end(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The whole of this version in one chain: rows, vectors, angle, degrees.

    The first two hits of a track are the two ends of one scattering. The
    directions are rebuilt from positions, the polarization is estimated from
    them, and the scattering angle is read in degrees.
    """
    # ARRANGE
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()
    order = frame.groupby(TRACK_KEY, sort=False).cumcount()
    first = frame[order == 0].set_index(TRACK_KEY)
    second = frame[order == 1].set_index(TRACK_KEY)
    paired = first.index.intersection(second.index)
    first, second = first.loc[paired], second.loc[paired]

    # ACT
    incoming = momentum_direction_from_positions(
        first[SOURCE].to_numpy(), first[POSITION].to_numpy()
    )
    outgoing = momentum_direction_from_positions(
        first[POSITION].to_numpy(), second[POSITION].to_numpy()
    )
    polarization = polarization_direction(incoming, outgoing)
    scattering = rad_to_deg(angle_between(incoming, outgoing))

    # ASSERT
    assert len(scattering) == 151
    assert scattering.min() > 0.0
    assert scattering.max() < 180.0
    assert not np.isnan(polarization).any()
    assert rad_to_deg(angle_between(polarization, incoming)) == pytest.approx(
        np.full(len(polarization), 90.0), abs=1e-2
    )


def test_a_selection_survives_into_the_vectors(hits_variant_files: Mapping[str, Path]) -> None:
    """The filters of 0.5.0 and the vectors of 0.6.0 have to compose."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a2"], GateTree.HITS).to_dataframe()

    # ACT
    selected = frame.gate.by_run(0).gate.in_cylinder(
        (0, 0), 500.0, inner_radius=SCANNER_BORE_RADIUS_MM
    )
    directions = selected.gate.momentum_direction()

    # ASSERT
    assert len(directions) == len(selected)
    assert directions.index.equals(selected.index)
    assert directions.norm().to_numpy() == pytest.approx(np.ones(len(selected)))
