"""Reading GATE ROOT files.

The module defines :class:`RootFile`, a thin wrapper over ``uproot`` that
exposes the parts of a GATE output file the package works with.

Only trees are considered. GATE files also store histograms, such as
``latest_event_ID`` and ``total_nb_primaries``; they are ignored when the file
contents are inspected and are never carried over to an output file.

Reading the "Hits" tree recognises which structure it has and checks the tree
against it before any data is loaded, so a file that is not what it is taken
for is reported rather than half read. Passing ``validate=False`` extracts the
branches without asking what structure they form, which is what a file from a
GATE build the package does not know needs.

Public objects:

RootFile
    Reader for a single GATE ROOT file.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import numpy.typing as npt
import uproot

from opengate_gate_tree.errors import RootFileError
from opengate_gate_tree.io.validation import (
    branch_type_name,
    find_tree_name,
    resolve_tree_name,
    validate_branch_interpretations,
    validate_branches_present,
    validate_root_file_path,
)
from opengate_gate_tree.logger import log
from opengate_gate_tree.tree.branch import normalize_branch_selection
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.detection import HitsTreeDetection, detect_hits_variant
from opengate_gate_tree.tree.hits.validation import validate_hits_tree
from opengate_gate_tree.tree.treedata import TreeData

# Class name of the ROOT objects the package reads.
TREE_CLASS_NAME: Final[str] = "TTree"

# Separator between a ROOT key name and its cycle number.
CYCLE_SEPARATOR: Final[str] = ";"


class RootFile:
    """Reader for a single GATE ROOT file.

    The file is opened on construction and should be closed when no longer
    needed, either through :meth:`close` or by using the instance as a context
    manager.
    """

    def __init__(self, path: Path) -> None:
        """Open a GATE ROOT file.

        Parameters
        ----------
        path : Path
            Path to the ROOT file.

        Raises
        ------
        RootFileError
            If the path is not a readable ROOT file.
        """
        validate_root_file_path(path)
        self._path = path
        try:
            self._file = uproot.open(path)
        except (OSError, ValueError) as err:
            raise RootFileError(f"File could not be read as a ROOT file: {path}") from err

    def __enter__(self) -> "RootFile":
        """Return the reader itself so it can be used as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the underlying file."""
        self.close()

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()

    @property
    def path(self) -> Path:
        """Path of the opened file."""
        return self._path

    @property
    def tree_names(self) -> tuple[str, ...]:
        """Names of the trees stored in the file, without cycle numbers.

        Objects that are not trees, such as the histograms written by GATE,
        are not reported.
        """
        names = [
            _strip_cycle(str(key))
            for key, class_name in self._file.classnames().items()
            if class_name == TREE_CLASS_NAME
        ]
        return tuple(dict.fromkeys(names))

    def has_tree(self, tree: GateTree) -> bool:
        """Return whether the requested tree is present in the file."""
        return find_tree_name(self.tree_names, tree) is not None

    def resolve_tree_name(self, tree: GateTree) -> str:
        """Return the key under which the requested tree is stored.

        Names are compared exactly first, then without regard to case.

        Parameters
        ----------
        tree : GateTree
            Requested tree.

        Returns
        -------
        str
            Key of the tree in the file.

        Raises
        ------
        TreeNotFoundError
            If the tree is not present in the file.
        """
        return resolve_tree_name(self.tree_names, tree, self._path)

    def branch_names(self, tree: GateTree) -> tuple[str, ...]:
        """Return the branch names of the requested tree, in file order.

        Raises
        ------
        TreeNotFoundError
            If the tree is not present in the file.
        """
        return tuple(str(name) for name in self._file[self.resolve_tree_name(tree)].keys())

    def detect_hits_tree(self) -> HitsTreeDetection:
        """Recognise the structure of the "Hits" tree stored in the file.

        Returns
        -------
        HitsTreeDetection
            Structure of the tree.

        Raises
        ------
        TreeNotFoundError
            If the file holds no "Hits" tree.
        UnknownHitsVariantError
            If the structure of the tree is not a supported one.
        """
        tree_key, tree_object = self._open_tree(GateTree.HITS)
        return detect_hits_variant(tuple(str(name) for name in tree_object.keys()), tree_key)

    def read(
        self,
        tree: GateTree,
        branches: Sequence[str] | None = None,
        validate: bool = True,
    ) -> TreeData:
        """Read a tree into the package representation.

        Parameters
        ----------
        tree : GateTree
            Tree to read.
        branches : Sequence[str] | None
            Branches to read. When omitted or empty, every branch is read.
            Repeated names are read once, at the position of their first
            occurrence.
        validate : bool
            Whether to recognise the structure of the "Hits" tree and check
            the tree against it. The check covers the whole tree, not only the
            branches being read, and runs before any data is loaded. Other
            trees are read the same way either way.

        Returns
        -------
        TreeData
            Columns of the requested branches.

        Raises
        ------
        TreeNotFoundError
            If the tree is not present in the file.
        UnknownHitsVariantError
            If the structure of the "Hits" tree is not a supported one.
        HitsTreeValidationError
            If the "Hits" tree does not match the structure it was recognised
            as.
        BranchNotFoundError
            If any requested branch is not present in the tree.
        UnsupportedBranchTypeError
            If any requested branch uses an unsupported type.
        ValueError
            If any requested branch name is empty.
        """
        tree_key, tree_object = self._open_tree(tree)
        available = tuple(str(name) for name in tree_object.keys())

        if validate and tree is GateTree.HITS:
            detection = detect_hits_variant(available, tree_key)
            dtypes = {
                name: branch_type_name(tree_object[name].interpretation) for name in available
            }
            validate_hits_tree(available, dtypes, detection)

        selected = normalize_branch_selection(branches or [], available)
        validate_branches_present(available, selected)
        validate_branch_interpretations(
            {name: tree_object[name].interpretation for name in selected}
        )

        columns: Mapping[str, npt.NDArray[Any]] = {
            name: tree_object[name].array(library="np") for name in selected
        }
        data = TreeData(tree, columns)

        if data.entry_count == 0:
            log().warning("Tree '%s' in file %s has no entries.", tree_key, self._path)

        return data

    def _open_tree(self, tree: GateTree) -> tuple[str, Any]:
        """Return the key of a tree and the tree itself."""
        tree_key = self.resolve_tree_name(tree)
        return tree_key, self._file[tree_key]


def _strip_cycle(key: str) -> str:
    """Return a ROOT key without its cycle number."""
    return key.split(CYCLE_SEPARATOR)[0]
