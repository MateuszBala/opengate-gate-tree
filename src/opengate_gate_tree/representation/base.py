"""What every representation of a GATE tree has in common.

A tree is a table: one row per entry, one column per branch. An analysis asks
about something else - an event and the tracks in it, a single, a coincidence -
and a representation is the tree read as those elements. It is a way of
reading the data, not a copy of it: the columns stay where they are, and the
representation keeps what it needs to find the rows belonging to each element.

Every representation is built from the same two sources, the columns a reader
returns and the pandas view of them, possibly after a selection. The rows a
frame was left with keep their labels, so an element can always be traced back
to the rows it was read from.

A representation is written as a flat tree again: the rows it holds, in its
own order, with columns saying where each row stands. That keeps every output
format of the package open to it, and keeps the file readable by tools that
know nothing of this package.

Public objects:

DataRepresentation
    Base class of every representation of a tree.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

from opengate_gate_tree.errors import RepresentationError
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.treedata import TreeData


class DataRepresentation(ABC):
    """Data of one GATE tree read as the elements of an analysis.

    Subclasses name the tree they read in :attr:`TREE` and say how many
    elements of their topmost kind the data holds.

    Parameters
    ----------
    source : TreeData | pandas.DataFrame
        Columns of the tree, as a reader returns them or as a frame. The
        columns are referenced, not copied, wherever pandas allows it.

    Raises
    ------
    RepresentationError
        If the columns were read from another tree than the one the
        representation reads.
    """

    #: Tree whose data the representation reads.
    TREE: ClassVar[GateTree]

    def __init__(self, source: TreeData | pd.DataFrame) -> None:
        if isinstance(source, pd.DataFrame):
            data = TreeData.from_dataframe(self.TREE, source)
            row_labels = source.index
        else:
            data = source
            row_labels = pd.RangeIndex(len(source))
        if data.tree is not self.TREE:
            raise RepresentationError(
                f"{type(self).__name__} reads the {self.TREE.value!r} tree, "
                f"but the data was read from the {data.tree.value!r} tree."
            )
        self._data = data
        self._row_labels = row_labels

    @property
    def data(self) -> TreeData:
        """Columns the representation reads."""
        return self._data

    @property
    def row_labels(self) -> pd.Index:
        """Label of every row of the data, in the order of the rows.

        The index of the frame the representation was built from, or the
        position of each entry when it was built from columns.
        """
        return self._row_labels

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of elements of the topmost kind."""

    @abstractmethod
    def to_tree_data(self) -> TreeData:
        """Return the representation as a flat tree, the way it is written."""
