"""Generate EWFS agent-evaluation plots and summaries."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewfs.analysis.agent_evaluation import main


if __name__ == "__main__":
    main()

