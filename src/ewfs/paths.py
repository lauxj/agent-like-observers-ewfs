"""Shared filesystem paths for the repository."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
PLOTS_ROOT = RESULTS_ROOT / "plots"

