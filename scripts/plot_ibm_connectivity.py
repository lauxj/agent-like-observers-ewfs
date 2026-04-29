"""Generate IBM backend connectivity plots used in the thesis figures."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewfs.analysis.plot_ibm_connectivity import plot_marrakesh_agent_connectivity


if __name__ == "__main__":
    plot_marrakesh_agent_connectivity()

