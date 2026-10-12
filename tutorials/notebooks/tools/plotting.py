"""One look for the plots of every tutorial, and the plots themselves.

A reader moving from one notebook to the next should not have to re-read the
axes because the style changed, and a notebook should not spend ten lines on a
histogram. The style is applied once by :func:`use_tutorial_style`, and the
functions below draw the shapes a tutorial keeps asking for.

Every function takes an ``ax`` and returns one, so a plot can be built up over
several calls or placed in a grid of its own making. Every function that draws
values drops the ones that are not finite, and says how many it dropped - a pie
chart refuses them instead, since a share of an unknown whole is no share: a
column of angles from this package holds ``nan`` wherever a row had no answer,
and a plot that quietly leaves them out is a plot that lies about how much data
there was.

Public functions:

use_tutorial_style() -> None
    Apply the settings to matplotlib, for the rest of the notebook.
plot_hist_1d(x, ...) -> matplotlib.axes.Axes
    One histogram, binned or counted value by value.
plot_hists_1d(histograms, ...) -> matplotlib.axes.Axes
    Several histograms over one binning, with a legend.
plot_hist_2d(x, y, ...) -> matplotlib.axes.Axes
    A two-dimensional histogram, binned or counted value by value, with a
    colour bar.
plot_graph(x, y, ...) -> matplotlib.axes.Axes
    One series of points, with error bars where there are any.
plot_graphs(graphs, ...) -> matplotlib.axes.Axes
    Several series, with a legend.
plot_bar(x, heights, ...) -> matplotlib.axes.Axes
    One row of bars, each centred on the value it belongs to.
plot_bars(bars, ...) -> matplotlib.axes.Axes
    Several rows of bars, standing side by side above each value.
plot_pie_chart(values, ...) -> matplotlib.axes.Axes
    Parts of a whole.
count_values(values) -> tuple[numpy.ndarray, numpy.ndarray]
    How often each distinct value occurs.

Public classes:

HistogramParams
    One histogram of :func:`plot_hists_1d`, with how it is drawn.
GraphParams
    One series of :func:`plot_graphs`, with how it is drawn.
BarParams
    One row of bars of :func:`plot_bars`, with how it is drawn.
"""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, cast

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

# How much of the space between two values a group of bars takes. Less than
# one, so that the groups read as separate answers.
DEFAULT_BAR_WIDTH: Final[float] = 0.8

# Above this many values, the ticks are left to matplotlib: one tick per value
# is what makes a discrete axis readable, and what makes a crowded one
# unreadable.
TICK_LIMIT: Final[int] = 30


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
        Error bars, where the series has them. A point left out because it is
        not finite takes its error bar with it.
    extra : dict[str, Any]
        Anything else :meth:`matplotlib.axes.Axes.errorbar` takes.
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


@dataclass(frozen=True)
class BarParams:
    """One row of bars of :func:`plot_bars`, with how it is drawn.

    Attributes
    ----------
    x : array_like
        The values the bars stand above. They need not be the same for every
        row: a value one row does not have is drawn as no bar at all rather
        than as a bar of nothing.
    heights : array_like
        How tall each bar is, one per value.
    label : str | None
        What the legend calls this row.
    color : str | None
        Its colour; matplotlib chooses when this is ``None``.
    alpha : float
        How solid the bars are.
    edgecolor : str | None
        Colour of the outline, if any.
    linewidth : float
        Width of the outline.
    """

    x: ArrayLike
    heights: ArrayLike
    label: str | None = None
    color: str | None = None
    alpha: float = 0.85
    edgecolor: str | None = None
    linewidth: float = 0.0


def use_tutorial_style() -> None:
    """Apply the tutorial style to matplotlib.

    Called once near the top of a notebook, before anything is drawn.

    Examples
    --------
    >>> use_tutorial_style()
    >>> mpl.rcParams["axes.grid"]
    True
    """
    mpl.rcParams.update(cast(Any, TUTORIAL_STYLE))


def plot_hist_1d(
    x: ArrayLike,
    xrange: tuple[float, float] | None = None,
    xbins: int = DEFAULT_BINS,
    xtitle: str = "",
    ytitle: str = COUNTS_LABEL,
    title: str = "",
    ax: Axes | None = None,
    discrete: bool = False,
    **style: Any,
) -> Axes:
    """Draw one histogram.

    Parameters
    ----------
    x : array_like
        What to count.
    xrange : tuple[float, float] | None
        Ends of the axis. Taken from the data when omitted. Ignored when
        ``discrete`` is set, since there are no bins to place.
    xbins : int
        How many bins. Ignored when ``discrete`` is set.
    xtitle, ytitle, title : str
        The labels.
    ax : matplotlib.axes.Axes | None
        Where to draw. A new figure is made when omitted.
    discrete : bool
        Count each distinct value on its own and centre its bar over it,
        instead of dividing a range into bins. This is what a column of
        identifiers or of small counts asks for: with bins, the bar for a
        ``trackID`` of 1 runs from 1 to 2 and reads as though it belonged to
        both.
    **style
        Passed on to :meth:`matplotlib.axes.Axes.hist`, or to
        :func:`plot_bar` when ``discrete`` is set.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on, so that a notebook can add to them.
    """
    if discrete:
        values, counts = count_values(x, name="x")
        return plot_bar(values, counts, xtitle=xtitle, ytitle=ytitle, title=title, ax=ax, **style)
    axes = _axes(ax)
    axes.hist(_finite(x, "x"), bins=xbins, range=xrange, **style)
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
    discrete: bool = False,
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
        Ignored when ``discrete`` is set.
    xbins : int
        How many bins, shared by every histogram. Ignored when ``discrete`` is
        set.
    xtitle, ytitle, title : str
        The labels.
    ax : matplotlib.axes.Axes | None
        Where to draw. A new figure is made when omitted.
    legend : bool
        Whether to draw the legend. It says which histogram is which, so it is
        drawn unless a notebook has a reason otherwise.
    discrete : bool
        Count each distinct value on its own, and stand the histograms side by
        side above it rather than over one another. For values that are
        identifiers or small counts this is the readable form: three columns
        of ``trackID`` become three bars above 1, three above 2, and so on.

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
    if discrete:
        return plot_bars(
            [
                BarParams(
                    *count_values(histogram.values, name=histogram.label or "histogram"),
                    label=histogram.label,
                    color=histogram.color,
                    alpha=histogram.alpha,
                    linewidth=histogram.linewidth,
                )
                for histogram in histograms
            ],
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            ax=ax,
            legend=legend,
        )
    axes = _axes(ax)
    values = [_finite(histogram.values, histogram.label or "histogram") for histogram in histograms]
    edges = np.histogram_bin_edges(np.concatenate(values), bins=xbins, range=xrange)
    for histogram, counted in zip(histograms, values, strict=True):
        axes.hist(
            counted,
            bins=edges.tolist(),
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
    discrete: bool | tuple[bool, bool] = False,
    **style: Any,
) -> Axes:
    """Draw a two-dimensional histogram.

    Parameters
    ----------
    x, y : array_like
        The two coordinates. A row is counted only when both are finite, since
        half a point is not a point.
    xrange, yrange : tuple[float, float] | None
        Ends of the axes, each taken from the data when omitted, and ignored
        for an axis that is drawn discretely.
    xbins, ybins : int
        How many bins along each axis. Ignored for an axis that is drawn
        discretely, which has one cell per value.
    xtitle, ytitle, title : str
        The labels.
    colorbar : bool
        Whether to draw the colour bar. Without it the colours mean nothing.
    colorbar_title : str
        What the colours count.
    ax : matplotlib.axes.Axes | None
        Where to draw. A new figure is made when omitted.
    discrete : bool | tuple[bool, bool]
        Give each distinct value a cell of its own, centred on it, instead of
        dividing a range into bins. ``True`` does it for both axes; a pair does
        it for the ones it names, which is what a column of identifiers
        against a continuous quantity asks for - ``discrete=(True, False)``
        for ``trackID`` against energy.
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
    discrete_x, discrete_y = discrete if isinstance(discrete, tuple) else (discrete, discrete)

    horizontal = _discrete_edges(first) if discrete_x else xbins
    vertical = _discrete_edges(second) if discrete_y else ybins
    limits = None
    if (xrange is not None or yrange is not None) and first.size and not (discrete_x or discrete_y):
        limits = [
            list(xrange) if xrange is not None else [first.min(), first.max()],
            list(yrange) if yrange is not None else [second.min(), second.max()],
        ]

    *_, image = axes.hist2d(
        first, second, bins=cast(Any, [horizontal, vertical]), range=limits, **style
    )
    if discrete_x:
        _discrete_ticks(axes.set_xticks, first)
    if discrete_y:
        _discrete_ticks(axes.set_yticks, second)
    figure = axes.get_figure()
    if colorbar and figure is not None:
        figure.colorbar(image, ax=axes, label=colorbar_title)
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
        If no series was given, or one of them does not hold as many x as y.

    Warns
    -----
    UserWarning
        When points are left out because a coordinate is not finite.
    """
    if not graphs:
        raise ValueError("Drawing series together needs at least one of them.")
    axes = _axes(ax)
    for graph in graphs:
        first = np.asarray(graph.x, dtype=np.float64).ravel()
        second = np.asarray(graph.y, dtype=np.float64).ravel()
        if first.size != second.size:
            raise ValueError(
                f"A series is drawn point by point, so it holds as many x as y: "
                f"got {first.size} and {second.size}."
            )
        keep = _finite_mask(first, second, "points have a coordinate that is", stacklevel=3)
        axes.errorbar(
            first[keep],
            second[keep],
            yerr=_kept_errors(graph.yerr, keep),
            xerr=_kept_errors(graph.xerr, keep),
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


def plot_bar(
    x: ArrayLike,
    heights: ArrayLike,
    width: float = DEFAULT_BAR_WIDTH,
    xtitle: str = "",
    ytitle: str = COUNTS_LABEL,
    title: str = "",
    ax: Axes | None = None,
    **params: Any,
) -> Axes:
    """Draw one row of bars, each centred on the value it belongs to.

    Parameters
    ----------
    x : array_like
        The values the bars stand above.
    heights : array_like
        How tall each bar is, one per value.
    width : float
        How much of the space between neighbouring values a bar takes, from 0
        to 1. The default leaves a gap, so that the bars read as separate
        answers rather than as a filled range.
    xtitle, ytitle, title : str
        The labels.
    ax : matplotlib.axes.Axes | None
        Where to draw. A new figure is made when omitted.
    **params
        Anything :class:`BarParams` holds - ``label``, ``color``, ``alpha``,
        ``edgecolor``, ``linewidth``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    return plot_bars(
        [BarParams(x=x, heights=heights, **params)],
        width=width,
        xtitle=xtitle,
        ytitle=ytitle,
        title=title,
        ax=ax,
    )


def plot_bars(
    bars: Sequence[BarParams],
    width: float = DEFAULT_BAR_WIDTH,
    xtitle: str = "",
    ytitle: str = COUNTS_LABEL,
    title: str = "",
    ax: Axes | None = None,
    legend: bool = True,
) -> Axes:
    """Draw several rows of bars, standing side by side above each value.

    Every row is drawn over the same values - the ones any of them mentions -
    so that the bars above one value can be compared by eye. A value one row
    does not have is left empty there rather than drawn as zero, which are two
    different statements.

    The bars of one value share the space between it and its neighbours: the
    group is centred on the value, and the rows are placed within it in the
    order they were given.

    Parameters
    ----------
    bars : Sequence[BarParams]
        What to draw, and how to draw each row.
    width : float
        How much of the space between neighbouring values the whole group
        takes, from 0 to 1.
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
        If no row was given, if a row does not hold as many values as heights,
        or if the width is not between 0 and 1.

    Warns
    -----
    UserWarning
        When bars are left out because their value or height is not finite.
    """
    if not bars:
        raise ValueError("Drawing bars needs at least one row of them.")
    if not 0.0 < width <= 1.0:
        raise ValueError(f"A bar takes between 0 and 1 of the space it has, got {width}.")

    rows = [_bar_row(row) for row in bars]
    values = np.unique(np.concatenate([row[0] for row in rows]))
    span = _smallest_gap(values) * width
    thickness = span / len(rows)

    axes = _axes(ax)
    for number, (row_values, row_heights) in enumerate(rows):
        offset = (number - (len(rows) - 1) / 2) * thickness
        params = bars[number]
        axes.bar(
            row_values + offset,
            row_heights,
            width=thickness,
            label=params.label,
            color=params.color,
            alpha=params.alpha,
            edgecolor=params.edgecolor,
            linewidth=params.linewidth,
        )
    if len(values) <= TICK_LIMIT:
        axes.set_xticks(values)
    _label(axes, xtitle, ytitle, title)
    if legend and any(row.label for row in bars):
        axes.legend()
    return axes


def count_values(values: ArrayLike, name: str = "values") -> tuple[np.ndarray, np.ndarray]:
    """Return how often each distinct value occurs.

    Parameters
    ----------
    values : array_like
        What to count. Values that are not finite are left out and counted in
        a warning, as everywhere else here.
    name : str
        What the values are, for that warning.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The distinct values in order, and how often each occurs.

    Examples
    --------
    >>> count_values([1, 1, 2, 3, 3, 3])
    (array([1., 2., 3.]), array([2, 1, 3]))
    """
    distinct, counts = np.unique(_finite(values, name), return_counts=True)
    return distinct, counts


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
        If there are no parts, if any of them is negative or not finite - a
        pie chart of such a share means nothing - or if the labels do not
        match them.
    """
    parts = np.asarray(values, dtype=np.float64)
    if parts.size == 0:
        raise ValueError("A pie chart of nothing has nothing to show.")
    if not np.isfinite(parts).all():
        raise ValueError(f"A part of a whole is a finite number, got {parts.tolist()}.")
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


def _discrete_edges(values: np.ndarray) -> np.ndarray:
    """Return bin edges that put one cell around each distinct value.

    The edges lie halfway between neighbouring values, and half a step beyond
    the ends, so that a value sits in the middle of its cell rather than at
    the corner of two.

    Values spaced evenly - identifiers, counts, anything on a grid - land in
    the middle of their cell exactly. Values spaced unevenly land inside it,
    with each boundary halfway to the neighbour on that side: cells that tile
    the plane cannot also be centred on values that are not evenly spaced, and
    tiling is what a two-dimensional histogram is.
    """
    distinct = np.unique(values)
    if distinct.size == 1:
        return np.array([distinct[0] - 0.5, distinct[0] + 0.5])
    middles = (distinct[:-1] + distinct[1:]) / 2
    first = distinct[0] - (middles[0] - distinct[0])
    last = distinct[-1] + (distinct[-1] - middles[-1])
    return np.concatenate([[first], middles, [last]])


def _discrete_ticks(place: Callable[[np.ndarray], object], values: np.ndarray) -> None:
    """Put one tick under each value, while there are few enough to read."""
    distinct = np.unique(values)
    if distinct.size <= TICK_LIMIT:
        place(distinct)


def _bar_row(row: BarParams) -> tuple[np.ndarray, np.ndarray]:
    """Return one row of bars as values and heights, refusing a mismatch.

    A bar whose value or height is not finite is left out, and counted in a
    warning: one such value would otherwise make every bar of the plot as wide
    as ``nan``, and the plot empty.
    """
    values = np.asarray(row.x, dtype=np.float64).ravel()
    heights = np.asarray(row.heights, dtype=np.float64).ravel()
    if values.size != heights.size:
        raise ValueError(
            f"A bar stands above one value, so a row holds as many values as heights: "
            f"got {values.size} and {heights.size}."
        )
    keep = _finite_mask(values, heights, "bars have a value or a height that is")
    return values[keep], heights[keep]


def _smallest_gap(values: np.ndarray) -> float:
    """Return the space one value has to itself, along the axis."""
    if values.size < 2:
        return 1.0
    gaps = np.diff(values)
    return float(np.min(gaps))


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
    keep = _finite_mask(first, second, "points have a coordinate that is", stacklevel=4)
    return first[keep], second[keep]


def _finite_mask(
    first: np.ndarray, second: np.ndarray, what: str, stacklevel: int = 4
) -> np.ndarray:
    """Return where both values can be drawn, and say how many pairs could not."""
    keep = np.isfinite(first) & np.isfinite(second)
    dropped = int(first.size - np.count_nonzero(keep))
    if dropped:
        warnings.warn(
            f"{dropped} of {first.size} {what} not finite and are left out of the plot.",
            stacklevel=stacklevel,
        )
    return keep


def _kept_errors(errors: ArrayLike | None, keep: np.ndarray) -> ArrayLike | None:
    """Return the error bars of the points that are drawn.

    A single number is one error for every point and stays as it is; an array
    holds one error, or a pair of them, per point along its last axis.
    """
    if errors is None or np.ndim(errors) == 0:
        return errors
    return np.asarray(errors, dtype=np.float64)[..., keep]
