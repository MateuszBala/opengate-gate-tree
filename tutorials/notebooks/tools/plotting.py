"""One look for the plots of every tutorial, and the plots themselves.

A reader moving from one notebook to the next should not have to re-read the
axes because the style changed, and a notebook should not spend ten lines on a
histogram. The style is applied once by :func:`use_tutorial_style`, and the
functions below draw the shapes a tutorial keeps asking for.

Every function takes an ``ax`` and returns one, so a plot can be built up over
several calls or placed in a grid of its own making. Every function drops the
values that are not finite before drawing, and says how many it dropped: a
column of angles from this package holds ``nan`` wherever a row had no answer,
and a plot that quietly leaves them out is a plot that lies about how much data
there was.

Public functions:

use_tutorial_style() -> None
    Apply the settings to matplotlib, for the rest of the notebook.
plot_hist_1d(x, ...) -> matplotlib.axes.Axes
    One histogram.
plot_hists_1d(histograms, ...) -> matplotlib.axes.Axes
    Several histograms over one binning, with a legend.
plot_hist_2d(x, y, ...) -> matplotlib.axes.Axes
    A two-dimensional histogram, with a colour bar.
plot_graph(x, y, ...) -> matplotlib.axes.Axes
    One series of points, with error bars where there are any.
plot_graphs(graphs, ...) -> matplotlib.axes.Axes
    Several series, with a legend.
plot_pie_chart(values, ...) -> matplotlib.axes.Axes
    Parts of a whole.

Public classes:

HistogramParams
    One histogram of :func:`plot_hists_1d`, with how it is drawn.
GraphParams
    One series of :func:`plot_graphs`, with how it is drawn.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

# Kept in one place so that a change reaches every tutorial at once.
TUTORIAL_STYLE: Final[dict[str, Any]] = {
    "figure.figsize": (8.0, 4.5),
    "figure.dpi": 110,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True,
}

# What a histogram counts, unless a notebook says otherwise.
COUNTS_LABEL: Final[str] = "counts"

# How many bins a histogram has when nobody chooses.
DEFAULT_BINS: Final[int] = 50


@dataclass(frozen=True)
class HistogramParams:
    """One histogram of :func:`plot_hists_1d`, with how it is drawn.

    Attributes
    ----------
    values : array_like
        What to count.
    label : str | None
        What the legend calls it.
    color : str | None
        Its colour; matplotlib chooses when this is ``None``.
    alpha : float
        How solid the fill is. The default lets the histograms underneath show
        through, which is the point of drawing them together.
    linewidth : float
        Width of the outline.
    linestyle : str
        Style of the outline, as matplotlib names them.
    fill : bool
        Whether the bars are filled or only outlined.
    """

    values: ArrayLike
    label: str | None = None
    color: str | None = None
    alpha: float = 0.5
    linewidth: float = 1.5
    linestyle: str = "-"
    fill: bool = True


@dataclass(frozen=True)
class GraphParams:
    """One series of :func:`plot_graphs`, with how it is drawn.

    Attributes
    ----------
    x, y : array_like
        The points.
    label : str | None
        What the legend calls it.
    color : str | None
        Its colour; matplotlib chooses when this is ``None``.
    marker : str
        Marker for a point. An empty string draws none.
    linestyle : str
        Style of the line through the points. An empty string draws none,
        which is what makes it a scatter plot.
    markersize : float
        Size of a marker.
    alpha : float
        How solid the series is.
    yerr, xerr : array_like | None
        Error bars, where the series has them.
    """

    x: ArrayLike
    y: ArrayLike
    label: str | None = None
    color: str | None = None
    marker: str = "o"
    linestyle: str = ""
    markersize: float = 5.0
    alpha: float = 1.0
    yerr: ArrayLike | None = None
    xerr: ArrayLike | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def use_tutorial_style() -> None:
    """Apply the tutorial style to matplotlib.

    Called once near the top of a notebook, before anything is drawn.

    Examples
    --------
    >>> use_tutorial_style()
    >>> mpl.rcParams["axes.grid"]
    True
    """
    mpl.rcParams.update(TUTORIAL_STYLE)


def plot_hist_1d(
    x: ArrayLike,
    xrange: tuple[float, float] | None = None,
    xbins: int = DEFAULT_BINS,
    xtitle: str = "",
    ytitle: str = COUNTS_LABEL,
    title: str = "",
    ax: Axes | None = None,
    **style: Any,
) -> Axes:
    """Draw one histogram.

    Parameters
    ----------
    x : array_like
        What to count.
    xrange : tuple[float, float] | None
        Ends of the axis. Taken from the data when omitted.
    xbins : int
        How many bins.
    xtitle, ytitle, title : str
        The labels.
    ax : matplotlib.axes.Axes | None
        Where to draw. A new figure is made when omitted.
    **style
        Passed on to :meth:`matplotlib.axes.Axes.hist`, for anything this
        signature does not name.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on, so that a notebook can add to them.
    """
    axes = _axes(ax)
    values = _finite(x, "x")
    axes.hist(values, bins=xbins, range=xrange, **style)
    _label(axes, xtitle, ytitle, title)
    return axes


def plot_hists_1d(
    histograms: Sequence[HistogramParams],
    xrange: tuple[float, float] | None = None,
    xbins: int = DEFAULT_BINS,
    xtitle: str = "",
    ytitle: str = COUNTS_LABEL,
    title: str = "",
    ax: Axes | None = None,
    legend: bool = True,
) -> Axes:
    """Draw several histograms over one another.

    All of them share one binning, computed from ``xrange`` and ``xbins`` or,
    when no range is given, from the values of every histogram together. That
    is the point of the function: histograms drawn one after another with
    binnings of their own cannot be compared, and the difference is invisible
    on the picture.

    Parameters
    ----------
    histograms : Sequence[HistogramParams]
        What to draw, and how to draw each one.
    xrange : tuple[float, float] | None
        Ends of the axis. Taken from all the values together when omitted.
    xbins : int
        How many bins, shared by every histogram.
    xtitle, ytitle, title : str
        The labels.
    ax : matplotlib.axes.Axes | None
        Where to draw. A new figure is made when omitted.
    legend : bool
        Whether to draw the legend. It says which histogram is which, so it is
        drawn unless a notebook has a reason otherwise.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Raises
    ------
    ValueError
        If no histogram was given.
    """
    if not histograms:
        raise ValueError("Drawing histograms together needs at least one of them.")
    axes = _axes(ax)
    values = [_finite(histogram.values, histogram.label or "histogram") for histogram in histograms]
    edges = np.histogram_bin_edges(np.concatenate(values), bins=xbins, range=xrange)
    for histogram, counted in zip(histograms, values, strict=True):
        axes.hist(
            counted,
            bins=edges,
            label=histogram.label,
            color=histogram.color,
            alpha=histogram.alpha,
            linewidth=histogram.linewidth,
            linestyle=histogram.linestyle,
            histtype="stepfilled" if histogram.fill else "step",
            edgecolor=histogram.color,
        )
    _label(axes, xtitle, ytitle, title)
    if legend and any(histogram.label for histogram in histograms):
        axes.legend()
    return axes


def plot_hist_2d(
    x: ArrayLike,
    y: ArrayLike,
    xrange: tuple[float, float] | None = None,
    xbins: int = DEFAULT_BINS,
    yrange: tuple[float, float] | None = None,
    ybins: int = DEFAULT_BINS,
    xtitle: str = "",
    ytitle: str = "",
    title: str = "",
    colorbar: bool = True,
    colorbar_title: str = COUNTS_LABEL,
    ax: Axes | None = None,
    **style: Any,
) -> Axes:
    """Draw a two-dimensional histogram.

    Parameters
    ----------
    x, y : array_like
        The two coordinates. A row is counted only when both are finite, since
        half a point is not a point.
    xrange, yrange : tuple[float, float] | None
        Ends of the axes. Taken from the data when omitted.
    xbins, ybins : int
        How many bins along each axis.
    xtitle, ytitle, title : str
        The labels.
    colorbar : bool
        Whether to draw the colour bar. Without it the colours mean nothing.
    colorbar_title : str
        What the colours count.
    ax : matplotlib.axes.Axes | None
        Where to draw. A new figure is made when omitted.
    **style
        Passed on to :meth:`matplotlib.axes.Axes.hist2d`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Raises
    ------
    ValueError
        If the two coordinates do not hold as many values each.
    """
    axes = _axes(ax)
    first, second = _finite_pair(x, y)
    limits = None if xrange is None or yrange is None else [list(xrange), list(yrange)]
    *_, image = axes.hist2d(first, second, bins=[xbins, ybins], range=limits, **style)
    if colorbar:
        axes.get_figure().colorbar(image, ax=axes, label=colorbar_title)
    _label(axes, xtitle, ytitle, title)
    return axes


def plot_graph(
    x: ArrayLike,
    y: ArrayLike,
    xtitle: str = "",
    ytitle: str = "",
    title: str = "",
    ax: Axes | None = None,
    **params: Any,
) -> Axes:
    """Draw one series of points.

    Parameters
    ----------
    x, y : array_like
        The points.
    xtitle, ytitle, title : str
        The labels.
    ax : matplotlib.axes.Axes | None
        Where to draw. A new figure is made when omitted.
    **params
        Anything :class:`GraphParams` holds - ``label``, ``color``, ``marker``,
        ``linestyle``, ``yerr`` and the rest.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    return plot_graphs(
        [GraphParams(x=x, y=y, **params)],
        xtitle=xtitle,
        ytitle=ytitle,
        title=title,
        ax=ax,
    )


def plot_graphs(
    graphs: Sequence[GraphParams],
    xtitle: str = "",
    ytitle: str = "",
    title: str = "",
    ax: Axes | None = None,
    legend: bool = True,
) -> Axes:
    """Draw several series of points on one pair of axes.

    Parameters
    ----------
    graphs : Sequence[GraphParams]
        What to draw, and how to draw each one.
    xtitle, ytitle, title : str
        The labels.
    ax : matplotlib.axes.Axes | None
        Where to draw. A new figure is made when omitted.
    legend : bool
        Whether to draw the legend.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Raises
    ------
    ValueError
        If no series was given, or one of them has as many x as y.
    """
    if not graphs:
        raise ValueError("Drawing series together needs at least one of them.")
    axes = _axes(ax)
    for graph in graphs:
        first = np.asarray(graph.x, dtype=np.float64)
        second = np.asarray(graph.y, dtype=np.float64)
        if len(first) != len(second):
            raise ValueError(
                f"A series is drawn point by point, so it holds as many x as y: "
                f"got {len(first)} and {len(second)}."
            )
        axes.errorbar(
            first,
            second,
            yerr=graph.yerr,
            xerr=graph.xerr,
            label=graph.label,
            color=graph.color,
            marker=graph.marker,
            linestyle=graph.linestyle,
            markersize=graph.markersize,
            alpha=graph.alpha,
            capsize=3,
            **graph.extra,
        )
    _label(axes, xtitle, ytitle, title)
    if legend and any(graph.label for graph in graphs):
        axes.legend()
    return axes


def plot_pie_chart(
    values: ArrayLike,
    labels: Sequence[str] | None = None,
    title: str = "",
    ax: Axes | None = None,
    autopct: str = "%1.1f%%",
    colors: Sequence[str] | None = None,
) -> Axes:
    """Draw parts of a whole.

    Parameters
    ----------
    values : array_like
        The parts. They are shown as shares of their sum, so they do not have
        to add up to anything in particular.
    labels : Sequence[str] | None
        What each part is called.
    title : str
        The label above.
    ax : matplotlib.axes.Axes | None
        Where to draw. A new figure is made when omitted.
    autopct : str
        How the share is written on each slice.
    colors : Sequence[str] | None
        The colours; matplotlib chooses when this is ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Raises
    ------
    ValueError
        If there are no parts, if any of them is negative - a pie chart of a
        negative share means nothing - or if the labels do not match them.
    """
    parts = np.asarray(values, dtype=np.float64)
    if parts.size == 0:
        raise ValueError("A pie chart of nothing has nothing to show.")
    if (parts < 0).any():
        raise ValueError(f"A part of a whole is not negative, got {parts.tolist()}.")
    if labels is not None and len(labels) != parts.size:
        raise ValueError(
            f"Every part is named or none is: got {len(labels)} label(s) for {parts.size} part(s)."
        )
    axes = _axes(ax)
    axes.pie(parts, labels=labels, autopct=autopct, colors=colors)
    axes.set_aspect("equal")
    if title:
        axes.set_title(title)
    return axes


def _axes(ax: Axes | None) -> Axes:
    """Return the axes to draw on, making a figure when none was given."""
    if ax is not None:
        return ax
    _, axes = plt.subplots()
    return axes


def _label(axes: Axes, xtitle: str, ytitle: str, title: str) -> None:
    """Put the labels on, leaving alone the ones nobody gave."""
    if xtitle:
        axes.set_xlabel(xtitle)
    if ytitle:
        axes.set_ylabel(ytitle)
    if title:
        axes.set_title(title)


def _finite(values: ArrayLike, name: str) -> np.ndarray:
    """Return the values that can be drawn, and say how many could not."""
    array = np.asarray(values, dtype=np.float64).ravel()
    keep = np.isfinite(array)
    dropped = int(array.size - np.count_nonzero(keep))
    if dropped:
        warnings.warn(
            f"{dropped} of {array.size} values of '{name}' are not finite and are left out "
            f"of the plot.",
            stacklevel=3,
        )
    return array[keep]


def _finite_pair(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return the pairs where both coordinates can be drawn."""
    first = np.asarray(x, dtype=np.float64).ravel()
    second = np.asarray(y, dtype=np.float64).ravel()
    if first.size != second.size:
        raise ValueError(
            f"A point has two coordinates, so both hold as many values: "
            f"got {first.size} and {second.size}."
        )
    keep = np.isfinite(first) & np.isfinite(second)
    dropped = int(first.size - np.count_nonzero(keep))
    if dropped:
        warnings.warn(
            f"{dropped} of {first.size} points have a coordinate that is not finite and are "
            f"left out of the plot.",
            stacklevel=3,
        )
    return first[keep], second[keep]
