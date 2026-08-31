"""Processing of GATE output file trees.

The package can be used as a command line tool and as a library. Everything
the command line does is reachable from here.

Reading a tree and writing it in another format:

.. code-block:: python

    from pathlib import Path

    from opengate_gate_tree import GateTree, OutputFileFormat, read_tree, write_tree

    data = read_tree(Path("simulation.root"), GateTree.HITS, ["eventID", "edep"])
    frame = data.to_dataframe()
    write_tree(data, Path("out/hits.hdf5"), OutputFileFormat.HDF5)

Use :class:`RootFile` directly when several trees are read from one file, so
that the file is opened once.

The structure of the "Hits" tree depends on the simulation that wrote it.
:class:`HitsTreeVariant` names the structures the package supports,
:func:`detect_hits_variant` recognises the one a tree has, and
:func:`expected_branches` states which branches each of them holds. Reading a
"Hits" tree recognises and checks its structure unless ``validate=False`` says
otherwise. Hits stored under another name, one tree per run or per sensitive
detector, are found by their structure, and the tree to read can be named.
:func:`read_hits_trees` reads them all as one dataset, recording which tree
each row came from.

:func:`compute_statistics` summarises what was extracted, and
:func:`write_statistics` saves that summary next to the data.

Filters and selectors work on the pandas view of the data, both as functions
and through the ``gate`` namespace this package registers on a column and on a
frame::

    prompt = frame["gammaType"].gate.select_by_gamma_type(GammaType.PROMPT)
    in_the_ring = frame.gate.in_cylinder((0, 0), radius=500.0, inner_radius=409.0)

The branches a ``PositroniumSource`` writes carry integers whose meaning is
described by :class:`SourceType`, :class:`DecayType` and :class:`GammaType`.
Their members are those integers, so a column compares against them as it was
read. :func:`decode_positronium_value` and :func:`decode_positronium_column`
read those values as names, :func:`positronium_enum` says which class describes
a branch, and :func:`has_positronium_metadata` reads the ``decayIndex`` branch to
tell rows carrying the decay metadata of such a source from the rest.

Failures while reading or writing files are reported through a subclass of
:class:`GateTreeError`, so a single ``except`` clause covers them. Malformed
arguments, such as an empty branch name or inconsistent columns, raise
``ValueError`` instead, and are not caught by that clause.

Three columns of a frame are a column of vectors, and
:meth:`VectorView.from_frame` reads them as one, keeping the rows they came
from::

    direction = frame.gate.position() - frame.gate.source_position()
    forward = direction.unit().dot(frame.gate.momentum_direction())

:func:`as_vectors` does the same for an array.

A vector is taken apart along a direction with :func:`parallel_component` and
:func:`perpendicular_component`, and read in spherical components with
:func:`spherical_components`: a radius, a polar angle from ``z`` in
``[0, pi]``, and an azimuth in ``[0, 2*pi)``. :func:`normalize`, :func:`dot` and
:func:`cross` work on whole columns at once; a vector of no length has no
direction, so such a row is answered with ``nan`` and reported, rather than
refusing the column it sits in.

Quantities are read in the units GATE writes them in - MeV, mm and s. The
conversions to the units an analysis uses are named after what they do::

    energies = MeV_to_keV(frame["edep"])
    resolution = s_to_ns(frame["time"])

The package does not configure logging on import. Applications decide that for
themselves; :func:`opengate_gate_tree.logging_setup.configure_logging` is
available for the ones that want the defaults used by the command line.

Public attributes:

__version__ : str
    Package version consistent with ``pyproject.toml``.
"""

# The version is defined before the re-exports below: the HDF5 writer records
# it in the files it writes, and importing it from a package that is still
# being initialised only works once the name exists.
__version__ = "0.5.0"

import logging  # noqa: E402

from opengate_gate_tree.errors import (  # noqa: E402
    AmbiguousTreeError,
    BranchNotFoundError,
    ExportError,
    GateTreeError,
    HitsTreeValidationError,
    RootFileError,
    TreeMergeError,
    TreeNotFoundError,
    UnknownHitsVariantError,
    UnsupportedBranchTypeError,
)
from opengate_gate_tree.geometry.components import (  # noqa: E402
    parallel_component,
    perpendicular_component,
    spherical_components,
)
from opengate_gate_tree.geometry.vectors import (  # noqa: E402
    as_vectors,
    clip_cosine,
    cross,
    dot,
    ensure_vectors,
    norm,
    normalize,
    wrap_to_two_pi,
)
from opengate_gate_tree.geometry.vectorview import VectorView  # noqa: E402
from opengate_gate_tree.io.fileformat import (  # noqa: E402
    OutputFileFormat,
    parse_output_file_format,
)
from opengate_gate_tree.io.naming import (  # noqa: E402
    build_output_file_name,
    build_output_file_path,
    build_statistics_file_path,
)
from opengate_gate_tree.io.reader import read_hits_trees, read_tree  # noqa: E402
from opengate_gate_tree.io.rootfile import RootFile  # noqa: E402
from opengate_gate_tree.io.statistics import write_statistics  # noqa: E402
from opengate_gate_tree.io.writers import write_tree  # noqa: E402
from opengate_gate_tree.logging_setup import LOGGER_NAME  # noqa: E402
from opengate_gate_tree.tree.accessors import ACCESSOR_NAME  # noqa: E402
from opengate_gate_tree.tree.filters import (  # noqa: E402
    EVENT_COLUMN,
    POSITION_COLUMNS,
    RUN_COLUMN,
    InclusiveSide,
    by_event,
    by_run,
    has_decay_metadata,
    in_box,
    in_cylinder,
    in_range,
    in_sphere,
    is_decay_type,
    is_from_event,
    is_from_run,
    is_gamma_type,
    is_in_box,
    is_in_cylinder,
    is_in_range,
    is_in_sphere,
    is_process,
    is_source_type,
    select_by_decay_type,
    select_by_gamma_type,
    select_by_process,
    select_by_source_type,
    with_decay_metadata,
)
from opengate_gate_tree.tree.gatetree import GateTree, parse_gate_tree  # noqa: E402
from opengate_gate_tree.tree.hits.detection import (  # noqa: E402
    HitsTreeDetection,
    describe_hits_tree,
    detect_hits_variant,
    summarise_hits_tree,
)
from opengate_gate_tree.tree.hits.positronium import (  # noqa: E402
    DECAY_INDEX_BRANCH,
    NO_POSITRONIUM_METADATA,
    POSITRONIUM_BRANCHES,
    DecayType,
    GammaType,
    SourceType,
    decode_positronium_column,
    decode_positronium_value,
    has_positronium_metadata,
    positronium_enum,
)
from opengate_gate_tree.tree.hits.schema import (  # noqa: E402
    BranchKind,
    BranchSpec,
    expected_branches,
    supported_variants,
    variant_reference,
)
from opengate_gate_tree.tree.hits.validation import validate_hits_tree  # noqa: E402
from opengate_gate_tree.tree.hits.variant import (  # noqa: E402
    GateSystemType,
    HitsTreeVariant,
)
from opengate_gate_tree.tree.merge import SOURCE_TREE_BRANCH, merge_tree_data  # noqa: E402
from opengate_gate_tree.tree.statistics import (  # noqa: E402
    BranchStatistics,
    HitsSummary,
    TreeStatistics,
    compute_statistics,
    format_statistics,
    statistics_to_dict,
)
from opengate_gate_tree.tree.treedata import TreeData  # noqa: E402
from opengate_gate_tree.units import (  # noqa: E402
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

# Keep the package quiet when the application using it has not configured
# logging. Without a handler, logging falls back to writing warnings straight
# to stderr, ignoring the configuration of the surrounding application.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())

__all__ = [
    "ACCESSOR_NAME",
    "AmbiguousTreeError",
    "BranchKind",
    "BranchNotFoundError",
    "BranchSpec",
    "BranchStatistics",
    "DECAY_INDEX_BRANCH",
    "DecayType",
    "EVENT_COLUMN",
    "ExportError",
    "GATE_UNITS",
    "GammaType",
    "GateSystemType",
    "GateTree",
    "GateTreeError",
    "HitsSummary",
    "HitsTreeDetection",
    "HitsTreeValidationError",
    "HitsTreeVariant",
    "InclusiveSide",
    "MeV_to_keV",
    "NO_POSITRONIUM_METADATA",
    "OutputFileFormat",
    "POSITION_COLUMNS",
    "POSITRONIUM_BRANCHES",
    "RUN_COLUMN",
    "RootFile",
    "RootFileError",
    "SOURCE_TREE_BRANCH",
    "SourceType",
    "TreeData",
    "TreeMergeError",
    "TreeNotFoundError",
    "TreeStatistics",
    "UnknownHitsVariantError",
    "UnsupportedBranchTypeError",
    "VectorView",
    "__version__",
    "as_vectors",
    "build_output_file_name",
    "build_output_file_path",
    "build_statistics_file_path",
    "by_event",
    "by_run",
    "clip_cosine",
    "cm_to_m",
    "cm_to_mm",
    "compute_statistics",
    "cross",
    "decode_positronium_column",
    "decode_positronium_value",
    "deg_to_rad",
    "describe_hits_tree",
    "detect_hits_variant",
    "dot",
    "ensure_vectors",
    "expected_branches",
    "format_statistics",
    "has_decay_metadata",
    "has_positronium_metadata",
    "in_box",
    "in_cylinder",
    "in_range",
    "in_sphere",
    "is_decay_type",
    "is_from_event",
    "is_from_run",
    "is_gamma_type",
    "is_in_box",
    "is_in_cylinder",
    "is_in_range",
    "is_in_sphere",
    "is_process",
    "is_source_type",
    "keV_to_MeV",
    "m_to_cm",
    "m_to_mm",
    "merge_tree_data",
    "mm_to_cm",
    "mm_to_m",
    "ms_to_ns",
    "ms_to_s",
    "norm",
    "normalize",
    "ns_to_ms",
    "ns_to_s",
    "parallel_component",
    "parse_gate_tree",
    "parse_output_file_format",
    "perpendicular_component",
    "positronium_enum",
    "rad_to_deg",
    "read_hits_trees",
    "read_tree",
    "s_to_ms",
    "s_to_ns",
    "select_by_decay_type",
    "select_by_gamma_type",
    "select_by_process",
    "select_by_source_type",
    "spherical_components",
    "statistics_to_dict",
    "summarise_hits_tree",
    "supported_variants",
    "validate_hits_tree",
    "variant_reference",
    "with_decay_metadata",
    "wrap_to_two_pi",
    "write_statistics",
    "write_tree",
]
