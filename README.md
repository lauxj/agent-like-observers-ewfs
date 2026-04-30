# Local Friendliness Violations with Agents on Quantum Computers

This repository contains the code and data for my master's thesis project.

## Project Structure

- `ewfs/`: main Python code.
- `data/paperdata/`: frozen data used for the thesis plots and results.
- `data/data_noiseless_simulation/`: local output folder for new noiseless runs.
- `data/data_fake_hardware/`: local output folder for new fake-hardware runs.
- `data/data_real_hardware/`: local output folder for new real-hardware runs.
- `data/IBM_calibrations/`: IBM calibration data used by the project.
- `notebooks/`: notebooks used while developing and inspecting plots.
- `results/`: generated plots and output files. This folder is not tracked.
- `requirements.txt`: Python packages needed to run the project.

## Main Files

- `ewfs/run.py`: main file for starting new experimental runs. The run settings
  are chosen inside this file.
- `ewfs/agent_evaluation.py`: creates the thesis plots from saved data. By
  default it uses `data/paperdata/`.
- `ewfs/agents.py`: defines the agent quantum circuits.
- `ewfs/accuracy_test_circuits.py`: defines the memory accuracy-test circuits.
- `ewfs/noiseless_simulation.py`: noiseless simulator runs.
- `ewfs/fake_hardware.py`: noisy fake-hardware simulator runs.
- `ewfs/real_hardware.py`: real IBM hardware runs.
- `ewfs/ibm_transpilation.py`: transpilation for IBM backends.
- `ewfs/lf_violations.py`: LF correlator and violation calculations.
- `ewfs/plot_ibm_connectivity.py`: IBM connectivity/layout plots.
- `ewfs/time_ordering_hardware.py`: hardware scheduler timing plots.

## Installation

Create a fresh Python environment and install the dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Another recent Python 3 version may also work, as long as it is supported by the
Qiskit packages in `requirements.txt`.

## Usage

To make a new experimental run, open `ewfs/run.py`, choose the settings in the
configuration section, and run that file.

New runs are saved in the normal data folders:

- `data/data_noiseless_simulation/`
- `data/data_fake_hardware/`
- `data/data_real_hardware/`

To reproduce the thesis plots, use `ewfs/agent_evaluation.py`. By default, it
loads the frozen runs in `data/paperdata/`. The evaluation file can also be
switched to use newly generated runs from the normal data folders.

Real IBM hardware runs require an IBM Quantum API token saved locally. The
noiseless simulation, fake-hardware simulation, and evaluation from saved data
do not require submitting new hardware jobs.
