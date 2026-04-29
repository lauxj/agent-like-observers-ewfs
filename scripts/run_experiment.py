"""Run the EWFS experiment pipeline from the repository root."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewfs.experiments.run import main


if __name__ == "__main__":
    main()

