"""Project-level configuration for data locations and SQLite database path.

The data directory can be overridden via environment variables:
- YSWY_DATA_DIR: root data dir (defaults to <project>/data)
- YSWY_DB_PATH: full path to the SQLite file (defaults to <data>/immersion.sqlite3)
"""

import os
from pathlib import Path
from typing import Final


def _project_root() -> Path:
	"""Return an approximation of the project root (parent of the package)."""
	return Path(__file__).resolve().parents[1]


DATA_DIR: Final[Path] = Path(os.getenv("YSWY_DATA_DIR", _project_root() / "data"))
SUBTITLES_DIR: Final[Path] = DATA_DIR / "subtitles"
DB_PATH: Final[Path] = Path(os.getenv("YSWY_DB_PATH", DATA_DIR / "immersion.sqlite3"))


def ensure_data_dirs() -> None:
	"""Create the base data directories if they do not already exist."""
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)
