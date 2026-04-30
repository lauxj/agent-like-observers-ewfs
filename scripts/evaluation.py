"""
Do evaluation of the data obtained from EWFS experiments:

Choose which data to consider for the evaluation and then run this script

includes: - LF violation plots for all agents and all runs
          - Agent performance evaluation (win rates, etc.) for all agents and all runs
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EWFS_DIR = PROJECT_ROOT / "ewfs"
if str(EWFS_DIR) not in sys.path:
    sys.path.insert(0, str(EWFS_DIR))

from analysis import agent_evaluation

#------------------------------------------------------------------------------
# Evaluation settings:

# choose from "paperdata" to reproduce the thesis results or "latest-runs" to use new runs
DATA_SOURCE = "paperdata"
#DATA_SOURCE = "latest-runs"

# number of runs to consider for the evaluation (paperdata has 10 runs)
LAST_N = 10

#------------------------------------------------------------------------------


def main():
    agent_evaluation.evaluate_with_settings(
        data_source=DATA_SOURCE,
        last_n=LAST_N,
    )


if __name__ == "__main__":
    main()
