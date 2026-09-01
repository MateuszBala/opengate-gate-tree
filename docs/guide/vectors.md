# Vectors And Angles

A "Hits" tree carries vectors as three columns side by side: `posX, posY, posZ`
for where a hit happened, `momDirX, momDirY, momDirZ` for where the particle
went afterwards. The package reads such a triple as the one thing it is, so
that an analysis reads like the physics it means.

```python
from pathlib import Path

from opengate_gate_tree import GateTree, read_tree

frame = read_tree(Path("simulation.root"), GateTree.HITS).to_dataframe()

position = frame.gate.position()
direction = frame.gate.momentum_direction()
angles = position.angle_to(direction).gate.rad_to_deg()
```

Every formula on this page is the one the code computes. Where a convention
could go two ways — which angle is the polar one, which side a normal points to
— the formula settles it.

## Reading Vectors Out Of A Frame

Four triples of a "Hits" tree have a method of their own, and any other three
columns are read by naming them:

| Method | Columns | What it holds |
| --- | --- | --- |
| `frame.gate.position()` | `posX`, `posY`, `posZ` | where the hit happened |
| `frame.gate.local_position()` | `localPosX`… | the same, inside the volume |
| `frame.gate.source_position()` | `sourcePosX`… | where the gamma was born |
| `frame.gate.momentum_direction()` | `momDirX`… | where the particle went **after** the hit |
| `frame.gate.vector("aX", "aY", "aZ")` | any three | anything else |

The result is a `VectorView`: the values as an `(N, 3)` array of `float64` and
the index of the rows they came from. It owns nothing and copies nothing back
into the frame.

## What A View Answers With

The shape of the answer follows the shape of the question:

| Question | Answer |
| --- | --- |
| a number per row — `norm`, `dot`, `angle_to` | `pandas.Series`, with the index of the rows |
| a vector per row — `unit`, `cross`, `parallel_to` | another `VectorView` |
| three numbers per row — `spherical` | `pandas.DataFrame` |
| the components — `.x`, `.y`, `.z` | `pandas.Series` |

That is what lets a selection, a computation and another selection stand in one
chain, and what puts a result back beside the data it came from.

### Warning

Two views combine only when they describe the same rows. Vectors of different
rows would be zipped by position, which is the mistake pandas alignment exists
to prevent, so it raises `ValueError` instead. Select the rows first, then read
the vectors.

## Lengths And Directions

The length of a vector is its norm, and its direction is what is left when the
length is divided out:

$$
\boxed{
\hat{v} = \frac{v}{\lVert v \rVert}
}
$$

```python
lengths = position.norm()          # a column, in the unit the file holds
directions = position.unit()       # another view, of unit length
```

## Products

The scalar product measures agreement, the vector product leaves the plane of
its arguments:

$$
\boxed{
u \cdot v = \sum_i u_i v_i
\qquad
(u \times v) \perp u,\ (u \times v) \perp v
}
$$

```python
projection = position.dot(direction)     # a column
normal = position.cross(direction)       # another view
```

## The Angle Between Two Directions

$$
\boxed{
\theta = \arccos\left(\hat{u}\cdot\hat{v}\right),
\qquad \theta \in [0, \pi]
}
$$

```python
angles = position.angle_to(direction)            # radians
degrees = angles.gate.rad_to_deg()               # degrees
```

Lengths do not reach the answer — both vectors are normalised first — so
directions read straight from a tree can be handed over as they are. When the
two are the directions before and after an interaction, this is the scattering
angle.

### Note

The cosine is pulled back into `[-1, 1]` before `arccos` sees it. The scalar
product of a unit vector with itself is `1.0000000000000002` often enough that
the first thing anybody checks — the angle between a vector and itself — would
otherwise answer `nan`.

## Along An Axis And Across It

A vector splits into the part that lies along a direction and the part that is
left:

$$
\boxed{
v_\parallel = \left(\hat{a}\cdot v\right)\hat{a}
\qquad
v_\perp = v - v_\parallel
\qquad
v = v_\parallel + v_\perp
}
$$

where $\hat{a}$ is the direction being read along, of unit length:

$$
\boxed{
\hat{a} = \frac{a}{\lVert a \rVert}
}
$$

which is why the length of the axis never reaches the answer — only where it
points does.

```python
along = position.parallel_to([0.0, 0.0, 1.0])       # the axis of the scanner
across = position.perpendicular_to([0.0, 0.0, 1.0])
```

The axis is $a$: whatever direction the question is about — the axis of a
scanner, a beam, a reconstructed polarization. It is given as one direction for
the whole column, or as one per row, which is another view.

### Note

The package never picks an axis for you. Where an axis has to be invented
because the question has none of its own — as when an azimuth needs a
reference direction to be measured from — the reference implementation of these
calculations chooses it from the direction it is working with:

$$
\hat{a} =
\begin{cases}
(0, 0, 1), & \lvert \hat{k_0}_z \rvert < 0.9,\\
(1, 0, 0), & \lvert \hat{k_0}_z \rvert \ge 0.9.
\end{cases}
$$

The switch avoids a cross product of two nearly parallel vectors, which has no
direction to speak of. That azimuth is not part of this version; the rule is
here because it is what an axis chosen rather than given has to look after.

## Spherical Components

$$
\boxed{
r = \lVert v \rVert
\qquad
\theta = \arccos\left(\frac{v_z}{r}\right)
\qquad
\varphi = \operatorname{atan2}(v_y, v_x)
}
$$

with the azimuth moved into a whole turn:

$$
\boxed{
\varphi =
\begin{cases}
\varphi, & \varphi \ge 0,\\
\varphi + 2\pi, & \varphi < 0.
\end{cases}
}
$$

```python
spherical = position.spherical()
spherical.columns     # Index(['radius', 'polar', 'azimuth'])
```

| Column | Meaning | Range |
| --- | --- | --- |
| `radius` | length of the vector | `[0, ∞)` |
| `polar` | angle from the `z` axis | `[0, π]` |
| `azimuth` | angle around `z`, from `x` towards `y` | `[0, 2π)` |

### Warning

The other convention in circulation calls the azimuth θ and the polar angle φ.
The columns are named `polar` and `azimuth` rather than `theta` and `phi` so
that nothing has to be guessed. `atan2` answers in `(-π, π]`, which would split
a half turn either side of zero and make a histogram of azimuths read as two
halves; the wrap above is what prevents that.

## Planes And The Angle Between Them

Two directions span a plane, and the plane is carried around by its normal:

$$
\boxed{
\hat{n} = \frac{u \times v}
               {\lVert u \times v \rVert}
}
$$

where $u$ and $v$ are any two vectors that are not parallel.

The angle between two planes is the angle between their normals:

$$
\boxed{
\phi = \arccos\left(\hat{n}_a \cdot \hat{n}_b\right),
\qquad \phi \in [0, \pi]
}
$$

```python
from opengate_gate_tree import angle_between_planes, plane_normal

normal = plane_normal(before, after)
between = angle_between_planes(before_a, after_a, before_b, after_b)
```

### Warning

A normal points to one side of its plane, and which side follows from the order
of the two directions that spanned it. Reading one pair backwards answers `π`
minus the angle. Give both pairs in the same order — before, then after.

## Directions Rebuilt From Positions

A detector records where something happened, not where the particle was going.
The direction of flight between two of those places is the step from one to the
other:

$$
\boxed{
\hat{k} = \frac{p_{\text{end}} - p_{\text{start}}}
               {\lVert p_{\text{end}} - p_{\text{start}} \rVert}
}
$$

```python
from opengate_gate_tree import momentum_direction_from_positions

incoming = momentum_direction_from_positions(source_position, first_hit)
outgoing = momentum_direction_from_positions(first_hit, second_hit)
```

### Note

`momDirX`, `momDirY` and `momDirZ` hold the direction **after** the interaction
at that hit. Measured on the reference files: for a track with a second hit,
that branch points exactly at the next hit, and it differs from the direction
the photon arrived with by the scattering angle. So the incoming direction is
not in the file, and is rebuilt from positions as above.

## Polarization

Compton scattering happens most readily perpendicular to the polarization of
the incoming photon, so the plane a photon scattered in carries what can be
known about it: the estimate is the normal of that plane.

$$
\boxed{
\hat{\varepsilon} = \frac{\hat{k_0} \times \hat{k}}
                         {\lVert \hat{k_0} \times \hat{k} \rVert}
}
$$

where $\hat{k_0}$ and $\hat{k}$ are the directions before and after the
scattering.

```python
from opengate_gate_tree import polarization_direction

polarization = polarization_direction(incoming, outgoing)
```

The estimate is defined on momentum directions and takes nothing else; where
those come from is the question above.

### Warning

This is an estimate per photon, not a measurement. A single scattering fixes
the plane, and the polarization is only distributed around its normal — what
the estimate is good for is a distribution over many photons. Reading the
scattering backwards reverses the sense of the answer, so it describes a line
in space rather than an arrow along it: summing these vectors would find them
cancelling.

## Rows That Answer Nothing

A vector of no length has no direction, and two parallel directions span no
plane. Those rows come back as `nan`, and the package reports how many:

```text
WARNING  3 of 12000 plane normal values have no length, so they have no
         direction. Those rows are read as nothing at all.
```

Refusing the whole column would be the other choice, and it is the wrong one
here: one degenerate row out of half a million must not end an analysis. The
same rule governs
[reading positronium values](positronium.md#reading-values-as-names).

## Precision

Vectors are read in `float64` whatever the file holds. GATE writes positions
and momentum directions as `float32`, and two things go wrong at that width:
the scalar product of two nearly parallel unit vectors loses about three
significant digits before `arccos` sees it, and a step between two hits is a
difference of coordinates around 200 mm, where the cancellation costs several
more.

## In A Chain

```python
selected = frame.gate.by_run(0).gate.in_cylinder((0, 0), 500.0, inner_radius=409.0)

angles = (
    selected.gate.position()
    .angle_to(selected.gate.momentum_direction())
    .gate.rad_to_deg()
)
```

A view goes back into a frame when the result is one:

```python
selected.join(selected.gate.position().unit().to_frame("dir"))
```

See [Filtering And Selecting](filtering.md) for the first half of that chain,
and [Units](units.md) for the last step.
