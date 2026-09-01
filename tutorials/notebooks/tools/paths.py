"""Where the files a tutorial reads and writes belong.

Paths are worked out from the position of this file, so a notebook finds the
same directories however it was started - from JupyterLab, from ``make
verify``, or from an editor that opened it somewhere else entirely.

A notebook writes to one place only: its own directory under ``.tmp/``, which
:func:`output_dir` makes on first use. Nothing there is carried in the
repository.

Public functions:

output_dir(notebook) -> pathlib.Path
    The directory this notebook writes to, made if it is not there yet.
input_example(name) -> pathlib.Path
    A ROOT file carried in the repository, by file name.
image(name) -> pathlib.Path
    A picture carried in the repository, by file name.
simulation(name) -> pathlib.Path
    The folder of one simulation, by its ``sim-NN-<name>`` name.

Public attributes:

TUTORIALS_DIR, NOTEBOOKS_DIR, DATA_DIR, INPUT_EXAMPLES_DIR, IMAGES_DIR,
SIMULATIONS_DIR, TEMPORARY_DIR : pathlib.Path
    The directories of the tutorials folder.
"""

from pathlib import Path
from typing import Final

# tools/ sits inside notebooks/, which sits inside tutorials/.
NOTEBOOKS_DIR: Final[Path] = Path(__file__).resolve().parent.parent
TUTORIALS_DIR: Final[Path] = NOTEBOOKS_DIR.parent

DATA_DIR: Final[Path] = TUTORIALS_DIR / "data"
INPUT_EXAMPLES_DIR: Final[Path] = DATA_DIR / "input-examples"
IMAGES_DIR: Final[Path] = DATA_DIR / "imgs"
SIMULATIONS_DIR: Final[Path] = TUTORIALS_DIR / "simulations"
TEMPORARY_DIR: Final[Path] = NOTEBOOKS_DIR / ".tmp"


def output_dir(notebook: str) -> Path:
    """Return the directory a notebook writes to, making it if needed.

    Parameters
    ----------
    notebook : str
        The name of the notebook, without its extension, as in
        ``"tutorial-01-reading-a-tree"``.

    Returns
    -------
    pathlib.Path
        ``notebooks/.tmp/<notebook>/``, which exists by the time it is
        returned.

    Examples
    --------
    >>> output = output_dir("tutorial-00-example")
    >>> output.is_dir()
    True
    """
    directory = TEMPORARY_DIR / notebook
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def input_example(name: str) -> Path:
    """Return a ROOT file carried in the repository.

    Parameters
    ----------
    name : str
        File name, as it appears in ``data/README.md``.

    Returns
    -------
    pathlib.Path
        The file.

    Raises
    ------
    FileNotFoundError
        If no such file is there, listing what is - a tutorial that names a
        file nobody added should say so at once, not three cells later.
    """
    return _existing(INPUT_EXAMPLES_DIR / name, INPUT_EXAMPLES_DIR, "input example")


def image(name: str) -> Path:
    """Return a picture carried in the repository.

    Parameters
    ----------
    name : str
        File name, as it appears in ``data/README.md``.

    Returns
    -------
    pathlib.Path
        The file.

    Raises
    ------
    FileNotFoundError
        If no such picture is there.
    """
    return _existing(IMAGES_DIR / name, IMAGES_DIR, "image")


def simulation(name: str) -> Path:
    """Return the folder of one simulation.

    Parameters
    ----------
    name : str
        The folder name, as in ``"sim-01-two-gammas"``.

    Returns
    -------
    pathlib.Path
        The folder, holding the macros and the README of that simulation.

    Raises
    ------
    FileNotFoundError
        If there is no such simulation.
    """
    return _existing(SIMULATIONS_DIR / name, SIMULATIONS_DIR, "simulation")


def _existing(path: Path, directory: Path, what: str) -> Path:
    """Return a path that is there, or say what is."""
    if not path.exists():
        available = sorted(item.name for item in directory.glob("*") if item.name != ".gitkeep")
        raise FileNotFoundError(
            f"No {what} named '{path.name}' in {directory}. There is: {available}."
        )
    return path
