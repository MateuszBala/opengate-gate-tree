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

Hits are not always stored under the name "Hits". The ``GateToTree`` output
calls its tree ``tree``, and a file can hold one tree per run or one per
sensitive detector, named after it. A tree asked for by name is used as given;
otherwise the name is matched, and for hits the trees are examined so that one
holding them is found whatever it is called.

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

from opengate_gate_tree.errors import AmbiguousTreeError, RootFileError
from opengate_gate_tree.io.validation import (
    branch_type_name,
    find_hits_tree_names,
    find_tree_name,
    resolve_requested_tree_name,
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
from opengate_gate_tree.tree.merge import merge_tree_data
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
        """Return whether the requested tree is present in the file.

        Hits are looked for the way they are read: by name first, then by
        structure, so a file whose hits sit in a tree called something else
        answers ``True`` here and can be read. A file holding hits in several
        trees answers ``True`` as well, although reading it needs one of them
        to be named.
        """
        if find_tree_name(self.tree_names, tree) is not None:
            return True
        return tree is GateTree.HITS and bool(self.hits_tree_names())

    def hits_tree_names(self) -> tuple[str, ...]:
        """Return the names of the trees holding hits, whatever they are called.

        A tree holds hits when its branches form one of the supported
        structures. A file usually holds one such tree, but the output can be
        split into one tree per run or per sensitive detector.

        Returns
        -------
        tuple[str, ...]
            Names of the trees holding hits, in file order.
        """
        branches_by_tree = {name: self._branch_names(name) for name in self.tree_names}
        return find_hits_tree_names(branches_by_tree)

    def resolve_tree_name(self, tree: GateTree, name: str | None = None) -> str:
        """Return the key under which the requested tree is stored.

        A name given by the caller is used as it is. Otherwise names are
        compared exactly, then without regard to case, and for hits the trees
        are examined as a last step, so that a tree holding them is found even
        when it is called something else.

        Parameters
        ----------
        tree : GateTree
            Requested tree.
        name : str | None
            Name of the tree in the file, when the caller knows it.

        Returns
        -------
        str
            Key of the tree in the file.

        Raises
        ------
        TreeNotFoundError
            If the tree is not present in the file.
        AmbiguousTreeError
            If several trees hold hits and none of them was named.
        """
        if name is not None:
            return resolve_requested_tree_name(self.tree_names, name, self._path)

        matched = find_tree_name(self.tree_names, tree)
        if matched is not None:
            if tree is GateTree.HITS:
                self._report_other_hits_trees(matched)
            return matched

        if tree is GateTree.HITS:
            return self._resolve_hits_tree_by_structure()

        return resolve_tree_name(self.tree_names, tree, self._path)

    def branch_names(self, tree: GateTree, name: str | None = None) -> tuple[str, ...]:
        """Return the branch names of the requested tree, in file order.

        Raises
        ------
        TreeNotFoundError
            If the tree is not present in the file.
        AmbiguousTreeError
            If several trees hold hits and none of them was named.
        """
        return self._branch_names(self.resolve_tree_name(tree, name))

    def detect_hits_tree(self, tree_name: str | None = None) -> HitsTreeDetection:
        """Recognise the structure of the "Hits" tree stored in the file.

        Parameters
        ----------
        tree_name : str | None
            Name of the tree in the file, when the caller knows it.

        Returns
        -------
        HitsTreeDetection
            Structure of the tree.

        Raises
        ------
        TreeNotFoundError
            If the file holds no "Hits" tree.
        AmbiguousTreeError
            If several trees hold hits and none of them was named.
        UnknownHitsVariantError
            If the structure of the tree is not a supported one.
        """
        tree_key, tree_object = self._open_tree(GateTree.HITS, tree_name)
        return detect_hits_variant(tuple(str(name) for name in tree_object.keys()), tree_key)

    def read(
        self,
        tree: GateTree,
        branches: Sequence[str] | None = None,
        tree_name: str | None = None,
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
        tree_name : str | None
            Name of the tree in the file, when it differs from the standard
            one or when a file holds several trees of hits.
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
        AmbiguousTreeError
            If several trees hold hits and none of them was named.
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
        tree_key, tree_object = self._open_tree(tree, tree_name)
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

    def read_hits(
        self,
        branches: Sequence[str] | None = None,
        tree_names: Sequence[str] | None = None,
        validate: bool = True,
        add_source_branch: bool = True,
    ) -> TreeData:
        """Read the hits of the file as a single dataset.

        A file splitting its hits into one tree per run, or one per sensitive
        detector, is read as one dataset with the trees placed one after
        another. Every tree is held in memory before they are joined, so this
        needs room for the whole file.

        Parameters
        ----------
        branches : Sequence[str] | None
            Branches to read from every tree. When omitted or empty, every
            branch is read.
        tree_names : Sequence[str] | None
            Trees to read, in the order their rows should follow. When
            omitted, every tree of the file holding hits is read, in file
            order.
        validate : bool
            Whether to recognise the structure of each tree and check it.
        add_source_branch : bool
            Whether to record which tree each row came from.

        Returns
        -------
        TreeData
            Rows of every tree that was read, one tree after another.

        Raises
        ------
        TreeNotFoundError
            If the file holds no hits, or a named tree is not present.
        UnknownHitsVariantError
            If the structure of a tree is not a supported one.
        HitsTreeValidationError
            If a tree does not match the structure it was recognised as.
        TreeMergeError
            If the trees do not hold the same structure.
        BranchNotFoundError
            If any requested branch is not present in a tree.
        """
        names = tuple(tree_names) if tree_names is not None else self.hits_tree_names()
        if not names:
            # No tree holds a whole structure. Fall back to resolving the name,
            # so that a file whose "Hits" tree the package does not recognise
            # is reported by what is wrong with that tree, and can still be
            # read with the check turned off. A file holding no hits at all
            # raises here, naming the trees it does hold.
            names = (self.resolve_tree_name(GateTree.HITS),)

        parts = [
            self.read(GateTree.HITS, branches, tree_name=name, validate=validate) for name in names
        ]
        return merge_tree_data(parts, names, add_source_branch)

    def _open_tree(self, tree: GateTree, name: str | None = None) -> tuple[str, Any]:
        """Return the key of a tree and the tree itself."""
        tree_key = self.resolve_tree_name(tree, name)
        return tree_key, self._file[tree_key]

    def _branch_names(self, tree_key: str) -> tuple[str, ...]:
        """Return the branch names of the tree stored under a key."""
        return tuple(str(name) for name in self._file[tree_key].keys())

    def _resolve_hits_tree_by_structure(self) -> str:
        """Return the name of the tree holding hits, found by its branches."""
        candidates = self.hits_tree_names()
        if len(candidates) == 1:
            log().info(
                "No tree named '%s' in file %s; reading '%s', whose branches hold hits.",
                GateTree.HITS.value,
                self._path,
                candidates[0],
            )
            return candidates[0]
        if len(candidates) > 1:
            raise AmbiguousTreeError(_ambiguous_message(candidates, self._path))
        return resolve_tree_name(self.tree_names, GateTree.HITS, self._path)

    def _report_other_hits_trees(self, matched: str) -> None:
        """Warn when the file holds hits beyond the tree that was matched."""
        others = [name for name in self.hits_tree_names() if name != matched]
        if others:
            log().warning(
                "File %s holds hits in %d more tree(s) besides '%s': %s. Only '%s' is read. "
                "Name one of them to read it instead, or read them all together.",
                self._path,
                len(others),
                matched,
                others,
                matched,
            )


def _strip_cycle(key: str) -> str:
    """Return a ROOT key without its cycle number."""
    return key.split(CYCLE_SEPARATOR)[0]


def _ambiguous_message(candidates: Sequence[str], source: Path) -> str:
    """Return the message reported when several trees hold hits."""
    return (
        f"File {source} holds hits in {len(candidates)} trees: {list(candidates)}. "
        f"None of them is named '{GateTree.HITS.value}', so the one to read cannot be chosen. "
        f"Name the tree to read, or read them all together as one dataset."
    )
