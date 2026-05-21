"""headroom.py — per-cap convenience wrapper around capabilities/lib/headroom.py.

Use lib/headroom.py directly when possible; this template exists so a cap
can `python3 headroom.py` from inside its own dir without importing the
shared lib (e.g. in environments where the lib hasn't been promoted yet).

In round-3 practice, prefer:
  python3 ../../lib/headroom.py --eval-summary /tmp/<cap>-eval.json --print
"""

import sys
from pathlib import Path

REPO_LIB = Path(__file__).resolve().parents[2] / "lib"
sys.path.insert(0, str(REPO_LIB))

from headroom import main  # noqa: E402

if __name__ == "__main__":
    main()
