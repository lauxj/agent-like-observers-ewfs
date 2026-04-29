"""Generate scheduler timing and time-ordering plots for a hardware run."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewfs.analysis.time_ordering_hardware import main


if __name__ == "__main__":
    main()

