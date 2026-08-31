"""Unit tests for the shape filters."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.filters import (
    in_box,
    in_cylinder,
    in_sphere,
    is_in_box,
    is_in_cylinder,
    is_in_sphere,
)
from opengate_gate_tree.tree.gatetree import GateTree

# Geometry of the simulations the fixtures come from, taken from their
# detector.mac rather than measured off the data. Shapes asked for in these
# numbers have an answer known in advance: the detector either holds every hit
# or none of them.
#
# a1-no-system: a spherical shell of lead.
A1_CAVITY_RADIUS = 200.0  # setRmin 0.2 m
A1_SHELL_RADIUS = 500.0  # setRmax 0.5 m
#
# a2-system: a cylindricalPET scanner, and the crystal a hit happens in.
A2_BORE_RADIUS = 409.0  # setRmin 409.0 mm
A2_SCANNER_RADIUS = 500.0  # setRmax 500.0 mm
A2_SCANNER_LENGTH = 1060.0  # setHeight 1060 mm
A2_CRYSTAL_SIDES = (3.2, 20.0, 3.2)  # crystal setXLength / setYLength / setZLength

# Points chosen so that each one answers a different question: the origin, a
# point on the face of the box below, one on the sphere of radius 10, one off
# the axis of the cylinder, and one far along the axis.
POINTS = pd.DataFrame(
    {
        "posX": [0.0, 1.0, 10.0, 5.0, 0.0],
        "posY": [0.0, 0.0, 0.0, 0.0, 0.0],
        "posZ": [0.0, 0.0, 0.0, 0.0, 100.0],
    }
)


def test_a_point_on_a_face_belongs_to_the_box() -> None:
    """A hit sitting exactly on a boundary is inside, and a simulation puts hits there."""
    # ARRANGE
    # The second point lies on the face x = 1.

    # ACT
    inside = is_in_box(POINTS, (0, 0, 0), 2)

    # ASSERT
    assert list(inside) == [True, True, False, False, False]


def test_a_point_outside_the_box_is_left_out() -> None:
    """The box is the intersection of three ranges, one per column."""
    # ARRANGE
    # No additional setup required.

    # ACT
    selected = in_box(POINTS, (0, 0, 0), 2)

    # ASSERT
    assert list(selected.index) == [0, 1]


def test_a_point_on_the_surface_belongs_to_the_sphere() -> None:
    """The surface counts, for the reason the faces of a box count."""
    # ARRANGE
    # The third point lies at distance 10 from the origin, the last one at 100.

    # ACT
    selected = in_sphere(POINTS, (0, 0, 0), 10.0)

    # ASSERT
    assert list(selected.index) == [0, 1, 2, 3]


def test_a_sphere_is_not_a_box() -> None:
    """The corner of a box is further from the centre than its faces."""
    # ARRANGE
    corner = pd.DataFrame({"posX": [10.0], "posY": [10.0], "posZ": [10.0]})

    # ACT
    in_the_box = is_in_box(corner, (0, 0, 0), 20)
    in_the_sphere = is_in_sphere(corner, (0, 0, 0), 10.0)

    # ASSERT
    assert list(in_the_box) == [True]
    assert list(in_the_sphere) == [False]


def test_a_cylinder_reaches_along_its_axis() -> None:
    """Without a range along the axis the cylinder is unbounded there."""
    # ARRANGE
    # The last point sits on the axis, a hundred units away.

    # ACT
    selected = in_cylinder(POINTS, (0, 0), 6.0)

    # ASSERT
    assert list(selected.index) == [0, 1, 3, 4]


def test_a_range_along_the_axis_cuts_the_cylinder_short() -> None:
    """A window along the axis is how an axial field of view is asked for."""
    # ARRANGE
    # No additional setup required.

    # ACT
    selected = in_cylinder(POINTS, (0, 0), 6.0, z_range=(-1.0, 1.0))

    # ASSERT
    assert list(selected.index) == [0, 1, 3]


def test_an_inner_radius_makes_a_ring_that_drops_the_middle() -> None:
    """A ring is how a layer of a detector is selected."""
    # ARRANGE
    # No additional setup required.

    # ACT
    selected = in_cylinder(POINTS, (0, 0), 6.0, inner_radius=4.0)

    # ASSERT
    assert list(selected.index) == [3]


def test_the_axis_of_a_cylinder_follows_the_order_of_the_columns() -> None:
    """Another axis is a matter of naming the columns in another order."""
    # ARRANGE
    # The last point is far along z, which now lies across the cylinder.

    # ACT
    along_z = is_in_cylinder(POINTS, (0, 0), 6.0)
    along_y = is_in_cylinder(POINTS, (0, 0), 6.0, columns=("posX", "posZ", "posY"))

    # ASSERT
    assert list(along_z) == [True, True, False, True, True]
    assert list(along_y) == [True, True, False, True, False]


def test_another_triple_of_columns_answers_about_another_place(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """A tree carries three positions per hit, and a shape has to be told which.

    The box here is the crystal itself, 3.2 by 20 by 3.2 mm according to the
    detector macro of this simulation. Every hit happened inside one, so in
    local coordinates the box holds all of them; in global ones it holds none,
    because the crystals sit over four hundred millimetres from the centre.
    """
    # ARRANGE
    frame = read_tree(hits_variant_files["a2"], GateTree.HITS).to_dataframe()

    # ACT
    global_hits = is_in_box(frame, (0, 0, 0), A2_CRYSTAL_SIDES)
    local_hits = is_in_box(
        frame, (0, 0, 0), A2_CRYSTAL_SIDES, columns=("localPosX", "localPosY", "localPosZ")
    )

    # ASSERT
    assert not global_hits.any()
    assert local_hits.all()


def test_a_shape_counts_the_rows_a_computation_by_hand_counts(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The filter has to agree with the arithmetic it stands for."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a2"], GateTree.HITS).to_dataframe()
    radius = np.sqrt(frame["posX"] ** 2 + frame["posY"] ** 2)

    # ACT
    # The middle fifth of the scanner, along its axis.
    selected = in_cylinder(frame, (0, 0), A2_SCANNER_RADIUS, z_range=(-100.0, 100.0))

    # ASSERT
    expected = frame[(radius <= A2_SCANNER_RADIUS) & frame["posZ"].between(-100.0, 100.0)]
    assert list(selected.index) == list(expected.index)
    assert 0 < len(selected) < len(frame)


def test_a_missing_column_is_reported_by_its_name() -> None:
    """A frame without the coordinates cannot answer a question about space."""
    # ARRANGE
    frame = pd.DataFrame({"edep": [1.0, 2.0]})

    # ACT / ASSERT
    with pytest.raises(KeyError, match="posX"):
        is_in_box(frame, (0, 0, 0), 1)


@pytest.mark.parametrize(
    ("centre", "sides"),
    [((0, 0), 2), ((0, 0, 0), (2, 2))],
    ids=["short-centre", "short-sides"],
)
def test_a_box_that_does_not_fit_the_columns_is_refused(
    centre: tuple[float, ...],
    sides: tuple[float, ...] | float,
) -> None:
    """A box is as many ranges as there are columns, and both arguments say which."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="one value per column"):
        is_in_box(POINTS, centre, sides)


def test_a_position_of_another_number_of_columns_is_refused() -> None:
    """Three columns make a position; two or four make a mistake."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="3 columns"):
        is_in_sphere(POINTS, (0, 0), 1.0, columns=("posX", "posY"))


@pytest.mark.parametrize("radius", [-1.0, -0.001], ids=["minus-one", "just-below-zero"])
def test_a_negative_radius_is_refused(radius: float) -> None:
    """A radius is a distance, and a distance is not negative."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="cannot be negative"):
        is_in_sphere(POINTS, (0, 0, 0), radius)


def test_a_ring_wider_inside_than_outside_is_refused() -> None:
    """Such a ring could hold nothing, and reads like two arguments swapped."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="inner radius"):
        is_in_cylinder(POINTS, (0, 0), 5.0, inner_radius=10.0)


def test_an_empty_frame_answers_empty() -> None:
    """A selection that removed every row is still a frame to ask questions of."""
    # ARRANGE
    frame = POINTS.iloc[:0]

    # ACT
    mask = is_in_sphere(frame, (0, 0, 0), 1.0)
    selected = in_sphere(frame, (0, 0, 0), 1.0)

    # ASSERT
    assert len(mask) == 0
    assert len(selected) == 0


def test_one_length_makes_a_cube() -> None:
    """Asking for a cube should not mean writing the same number three times."""
    # ARRANGE
    # No additional setup required.

    # ACT
    cube = is_in_box(POINTS, (0, 0, 0), 2)
    written_out = is_in_box(POINTS, (0, 0, 0), (2, 2, 2))

    # ASSERT
    assert list(cube) == list(written_out)


def test_a_side_reaches_half_its_length_either_way() -> None:
    """A box is given by its size, so the sides say how long, not how far."""
    # ARRANGE
    points = pd.DataFrame({"posX": [0.0, 5.0, 5.1], "posY": [0.0, 0.0, 0.0], "posZ": [0.0] * 3})

    # ACT
    selected = in_box(points, (0, 0, 0), 10)

    # ASSERT
    assert list(selected.index) == [0, 1]


def test_a_box_can_be_longer_in_one_direction() -> None:
    """A detector is rarely a cube, so each side is given on its own."""
    # ARRANGE
    # Points 1 and 3 sit on the x axis, one and five units out.

    # ACT
    cube = in_box(POINTS, (0, 0, 0), 2)
    stretched = in_box(POINTS, (0, 0, 0), (12, 2, 2))

    # ASSERT
    assert list(cube.index) == [0, 1]
    assert list(stretched.index) == [0, 1, 3]


def test_a_box_sits_where_its_centre_is() -> None:
    """The centre is what moves the box, as it moves a sphere."""
    # ARRANGE
    # No additional setup required.

    # ACT
    selected = in_box(POINTS, (10, 0, 0), 2)

    # ASSERT
    assert list(selected.index) == [2]


@pytest.mark.parametrize("sides", [-1, (2, -1, 2)], ids=["cube", "one-side"])
def test_a_negative_side_is_refused(sides: tuple[float, ...] | float) -> None:
    """A side is a length, and a length is not negative."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="side of a box cannot be negative"):
        is_in_box(POINTS, (0, 0, 0), sides)


def test_the_scanner_holds_every_hit_and_its_bore_holds_none(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The detector macro says where the hits can be, and the filter has to agree.

    The scanner of this file is a ring between 409 and 500 mm, 1060 mm long.
    Every hit happened in it, and none of them in the bore it surrounds - so
    both answers are known before the file is read.
    """
    # ARRANGE
    frame = read_tree(hits_variant_files["a2"], GateTree.HITS).to_dataframe()
    half_length = A2_SCANNER_LENGTH / 2

    # ACT
    in_the_scanner = is_in_cylinder(
        frame,
        (0, 0),
        A2_SCANNER_RADIUS,
        z_range=(-half_length, half_length),
        inner_radius=A2_BORE_RADIUS,
    )
    in_the_bore = is_in_cylinder(frame, (0, 0), A2_BORE_RADIUS)

    # ASSERT
    assert in_the_scanner.all()
    assert not in_the_bore.any()


def test_the_shell_holds_every_hit_and_its_cavity_holds_none(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The detector of this file is a spherical shell, so a sphere answers exactly.

    Lead between 200 and 500 mm from the centre, according to the detector
    macro: a sphere of the outer radius holds every hit, one of the inner
    radius holds none.
    """
    # ARRANGE
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()

    # ACT
    in_the_shell = is_in_sphere(frame, (0, 0, 0), A1_SHELL_RADIUS)
    in_the_cavity = is_in_sphere(frame, (0, 0, 0), A1_CAVITY_RADIUS)

    # ASSERT
    assert in_the_shell.all()
    assert not in_the_cavity.any()
