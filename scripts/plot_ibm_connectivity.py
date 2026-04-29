"""Generate IBM backend connectivity plots used in the thesis figures."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ewfs.analysis.plot_ibm_connectivity import main


if __name__ == "__main__":
    main()
