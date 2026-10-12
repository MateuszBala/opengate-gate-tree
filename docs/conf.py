"""Sphinx configuration for the user documentation.

Usage:
  make docs        build the HTML documentation
  make docs-check  build with warnings treated as errors
"""

from opengate_gate_tree import __version__

project = "opengate-gate-tree"
author = "Mateusz Jakub Bała"
project_copyright = "2026, Mateusz Jakub Bała"
version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# The repository root holds files that are not part of the documentation.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Docstrings in this project follow the NumPy style.
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# Headings up to this depth get an anchor, so cross-file links keep working.
myst_heading_anchors = 3

# The geometry pages carry the formulas the code implements, written between
# dollar signs. MyST renders none of that unless it is asked to: without
# "dollarmath" the formulas would reach the page as literal text, and a build
# with -W would not complain, because nothing went wrong. "amsmath" is for the
# environments a case distinction needs.
myst_enable_extensions = ["dollarmath", "amsmath"]

html_theme = "furo"
html_title = f"{project} {release}"
