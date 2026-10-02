"""Assemble train/val/test/null tables on a laptop (no GPU).

    python data/scripts/build_tables.py --processed data/processed --out data/processed
"""

from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

_impl = Path(__file__).with_name("build_v2_tables.py")
if not _impl.exists():
    raise SystemExit(f"Table builder missing: {_impl}")
_mod = SourceFileLoader("plmlof_build_tables", str(_impl)).load_module()
main = _mod.main

if __name__ == "__main__":
    main()
