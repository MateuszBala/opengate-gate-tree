"""Entry point for ``python -m opengate_gate_tree``.

Delegates execution to :func:`opengate_gate_tree.cli.main`.
"""

import sys

from opengate_gate_tree.cli import main

if __name__ == "__main__":
    sys.exit(main())
