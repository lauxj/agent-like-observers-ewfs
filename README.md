# Local Friendliness violations with Agents on Quantum Computers [IN PROGRESS..]

This repository contains the code and data for my master's thesis project on "Local Friendliness violations with Agents on Quantum Computers".

The main goal of the project is to run an extended Wigner's friend scenario on a quantum computer and use quantum agents to represent a friend in the experiment.
This repo contains the files for the pipeline of the experiment:

1. Circuit construction
2. Noiseless (ideal) simulation
3. Transpilation of circuits for IBM backends
4. Noise simulation
5. IBM hardware runs
6. Evaluation of agents

## Project Structure

- `ewfs/`: all Python scripts for circuits, experiments, and plotting
- `data/`: saved thesis data and local output for new runs
- `notebooks/`: project notebooks [IN PROGRESS]
- `results/`: generated plots and output files

## Files in `ewfs/`

- `ewfs/run.py`: main file for starting new experimental runs. The run settings
  are chosen inside this file.
- `ewfs/agent_evaluation.py`: creates the thesis plots from saved data. By
  default it uses `data/paperdata/`, but can be used for new runs by changing its settings
- `ewfs/agents.py`: defines the agent quantum circuits.
- `ewfs/accuracy_test_circuits.py`: defines the relaxed LF accuracy-test circuits.
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

## IBM Quantum API Token

Real hardware runs need access to IBM Quantum.

1. Create an IBM Quantum account at `https://quantum.cloud.ibm.com/`.
2. Log in to the IBM Quantum Platform.
3. Create or copy an IBM Cloud API key from your account/API key settings.
4. Save the API key locally with Qiskit:

```bash
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='YOUR_IBM_API_KEY', set_as_default=True, overwrite=True)"
```

Replace `YOUR_IBM_API_KEY` with the API key from IBM Quantum. This only needs to
be done once on a machine. After that, `ewfs/real_hardware.py` and `ewfs/run.py`
can access IBM Quantum through Qiskit.
