# Tools

What a tutorial would otherwise repeat in its first cell. Imported by name,
since a notebook runs with its own directory on the path:

```python
from tools import environment, paths, plotting
```

## `paths.py`

Where the files a tutorial reads and writes belong. Paths are worked out from
the position of the module, so a notebook finds the same directories whether it
was started from JupyterLab, from `make verify`, or from an editor somewhere
else.

- `output_dir(notebook)` - the one place a notebook may write, made on first
  use: `notebooks/.tmp/<notebook>/`.
- `input_example(name)`, `image(name)`, `simulation(name)` - a file or folder
  carried in the repository, by name. Each one raises `FileNotFoundError`
  listing what is there, so a notebook that names a file nobody added says so
  in the cell that asked for it rather than three cells later.
- `TUTORIALS_DIR`, `NOTEBOOKS_DIR`, `DATA_DIR`, `INPUT_EXAMPLES_DIR`,
  `IMAGES_DIR`, `SIMULATIONS_DIR`, `TEMPORARY_DIR` - the directories themselves.

## `environment.py`

What a tutorial was run against. The header of a notebook states the version of
the package it was written for; that line is written by hand and ages the way
hand-written lines do, so the notebook prints what is installed beside it.

- `describe()` - the package version, the Python version and today's date.
- `package_version()` - the installed version, or `"not installed"`, which is
  an answer rather than an exception.

## `plotting.py`

One look for the plots of every tutorial, so that a reader moving between
notebooks does not have to re-read the axes — and the plots a tutorial keeps
asking for, so that a histogram is one line rather than ten.

- `use_tutorial_style()` - applied once near the top of a notebook.
- `TUTORIAL_STYLE` - the settings themselves, in one place.

| Function | Draws |
| --- | --- |
| `plot_hist_1d(x, xrange, xbins, xtitle, ytitle, title, discrete)` | one histogram |
| `plot_hists_1d(histograms, ..., discrete)` | several histograms over one binning, with a legend |
| `plot_hist_2d(x, y, xrange, xbins, yrange, ybins, ...)` | a two-dimensional histogram, with a colour bar |
| `plot_graph(x, y, ...)` | one series of points, with error bars where there are any |
| `plot_graphs(graphs, ...)` | several series, with a legend |
| `plot_bar(x, heights, width, ...)` | one row of bars, each centred on its value |
| `plot_bars(bars, ...)` | several rows, standing side by side above each value |
| `plot_pie_chart(values, labels, ...)` | parts of a whole |
| `count_values(values)` | how often each distinct value occurs |

Every one of them takes an `ax` and returns one, so a plot can be built up over
several calls or placed into a grid the notebook made itself:

```python
from tools.plotting import GraphParams, HistogramParams, plot_graphs, plot_hists_1d

plot_hists_1d(
    [
        HistogramParams(prompt["edep"], label="prompt"),
        HistogramParams(annihilation["edep"], label="annihilation", fill=False),
    ],
    xrange=(0.0, 1.4),
    xbins=70,
    xtitle="energy deposited [MeV]",
)

plot_graphs(
    [
        GraphParams(energies, measured, yerr=errors, label="measured"),
        GraphParams(energies, predicted, label="theory", marker="", linestyle="--"),
    ],
    xtitle="energy [keV]",
    ytitle="angle [deg]",
)
```

`HistogramParams` and `GraphParams` hold one series each and how it is drawn -
colour, transparency, line style, markers, error bars - so that a notebook can
say what it wants without a wall of keyword arguments.

### Counting Discrete Values

A column of identifiers or of small counts does not want bins. With them, the
bar for a `trackID` of 1 runs from 1 to 2 and reads as though it belonged to
both; `discrete=True` counts each distinct value on its own and centres its bar
over it:

```python
plot_hist_1d(frame["trackID"], discrete=True, xtitle="trackID")
```

```text
binned:    bar for 1 runs from 1.00 to 1.75
discrete:  bar for 1 runs from 0.60 to 1.40, centred on 1
```

Several columns of discrete values stand side by side above each value rather
than over one another, which is how three runs are compared value by value:

```python
plot_hists_1d(
    [HistogramParams(run["trackID"], label=f"run {number}") for number, run in enumerate(runs)],
    discrete=True,
    xtitle="trackID",
)
```

The same shape is available from heights computed elsewhere, through
`plot_bar` and `plot_bars`, and `count_values` is the step between the two:

```python
values, counts = count_values(frame["trackID"])
plot_bar(values, counts, xtitle="trackID", ytitle="hits")
```

### Three Things These Do That A Bare Matplotlib Call Does Not

- **`plot_hists_1d` bins every histogram the same way**, from `xrange` and
  `xbins` or, when no range is given, from all the values together. Histograms
  drawn one after another with binnings of their own cannot be compared, and
  nothing on the picture says so.
- **A bar of a discrete histogram is centred on the value it counts**, so the
  picture reads as a distribution over values rather than over intervals. This
  matters most where it is easiest to misread: a narrow range, or a column of
  identifiers.
- **Values that are not finite are left out, and counted in a warning.** A
  column of angles from this package holds `nan` wherever a row had no answer,
  and a plot that quietly drops them tells the reader there was less data than
  there was.

## Adding A Tool

A helper belongs here when a second notebook needs it. Until then it belongs in
the notebook that uses it, where it can be read beside the work it does. Every
module here gets a section above, and every public function a docstring: this
folder is read by people writing tutorials, not only by the notebooks.
