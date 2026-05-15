"""
Running the EWFS experiments:

Change the settings below, then run this file. The detailed experiment behavior
is located in ewfs/experiments/run.py, which is called by the main() function here.

The data from the runs is saved in data/*

NOTE: TO use real hardware, install IBM API key first (see readme.md)
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EWFS_DIR = PROJECT_ROOT / "ewfs"
if str(EWFS_DIR) not in sys.path:
    sys.path.insert(0, str(EWFS_DIR))

from experiments import run


#------------------------------------------------------------------------------
# Experiment settings:

# choose which experiments to run (all together is possible)
RUN_NOISELESS = True
RUN_FAKE_HARDWARE = True
RUN_REAL_HARDWARE = True

# choose the backend used for transpilation, fake hardware, and real hardware
# options: "ibm_marrakesh", "ibm_fez", "ibm_kingston"
BACKEND_NAME = "ibm_marrakesh"

# choose shot counts
NOISELESS_SHOTS = 100_000
FAKE_HARDWARE_SHOTS = 10_000
REAL_HARDWARE_SHOTS = 1_000

# choose if relaxed LF accuracy tests should be included
INCLUDE_ACCURACY_TESTS = True

#------------------------------------------------------------------------------


def main():
    run.run_with_settings(
        run_noiseless=RUN_NOISELESS,
        run_fake_hardware=RUN_FAKE_HARDWARE,
        run_real_hardware=RUN_REAL_HARDWARE,
        backend_name=BACKEND_NAME,
        noiseless_shots=NOISELESS_SHOTS,
        fake_hardware_shots=FAKE_HARDWARE_SHOTS,
        real_hardware_shots=REAL_HARDWARE_SHOTS,
        include_accuracy_tests=INCLUDE_ACCURACY_TESTS,
    )


if __name__ == "__main__":
    main()
