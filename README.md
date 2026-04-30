# Local Friendliness violations with Agents on Quantum Computers

This repository contains the code and data for my master's thesis project on "Local Friendliness violations with Agents on Quantum Computers".

The main goal of the project is to run an extended Wigner's friend scenario (ewfs) on a quantum computer and use quantum agents to represent a friend in the experiment.
This repo contains the files for the pipeline of the experiment:

1. Circuit construction
2. Noiseless (ideal) simulation
3. Transpilation of circuits for IBM backends
4. Noise simulation
5. IBM hardware runs
6. Evaluation of agents

## Project Structure

- `ewfs/` -> all Python scripts for circuits, experiments, and analysis
- `scripts/` -> simple entry-point scripts with the main settings to change
- `data/` -> saved thesis data and local output for new runs
- `notebooks/` -> project notebooks
- `results/` -> folder for generated plots and output files (gets created when running evaluation)

## Files in `scripts/`

- `scripts/run_experiment.py` -> main file to run the experiment, choose what to run
- `scripts/evaluation.py` -> main file for evaluation, choose which data to consider for evaluation


## Files in `ewfs/`

- `ewfs/experiments/run.py` -> main experiment runner
- `ewfs/analysis/agent_evaluation.py` -> creates the thesis plots from saved data
- `ewfs/circuits/agents.py` -> defines the agent quantum circuit
- `ewfs/circuits/accuracy_test_circuits.py` -> defines the relaxed LF accuracy-test circuits
- `ewfs/experiments/noiseless_simulation.py` -> noiseless simulator runs.
- `ewfs/experiments/fake_hardware.py` -> noise-simulation runs
- `ewfs/experiments/real_hardware.py` -> real IBM hardware runs
- `ewfs/experiments/ibm_transpilation.py` -> transpilation for IBM backends
- `ewfs/analysis/lf_violations.py` -> LF correlator and violation calculations
- `ewfs/analysis/plot_ibm_connectivity.py` -> IBM connectivity/layout plots
- `ewfs/analysis/time_ordering_hardware.py` -> hardware scheduler timing plots



## Installation

Python 3.10 is required. Newer Python versions may have issues.
Clone the repository and install the requirements in a project-specific environment.

```bash
git clone https://github.com/lauxj/masters_thesis_project.git
cd masters_thesis_project
python3.10 -m venv .venv
source .venv/bin/activate
python --version  # should show Python 3.10.x
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

To open the demo notebook, run:

```bash
python -m notebook notebooks/project_demo.ipynb
```


## Usage

To make a new experimental run, open `scripts/run_experiment.py`, choose the
settings near the top of the file, and run that file. The detailed experiment
settings stay in `ewfs/experiments/run.py`.

New runs are saved in the normal data folders:

- `data/data_noiseless_simulation/`
- `data/data_fake_hardware/`
- `data/data_real_hardware/`

To reproduce the thesis plots, open `scripts/evaluation.py` and use the runs in `data/paperdata/`.
It can also be switched to use newly generated runs from the normal data folders.

Real IBM hardware runs require an IBM Quantum API token saved locally. The
noiseless simulation, fake-hardware simulation, and evaluation from saved data
do not require submitting new hardware jobs.

Transpilation and fake-hardware simulation do not submit real hardware jobs, but
they still load an IBM backend through Qiskit. Those sections therefore also
need the IBM Quantum API token below. The noiseless simulation and saved-data
evaluation can be run without it.

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
be done once on a machine. After that, `scripts/run_experiment.py` and
`ewfs/experiments/real_hardware.py`
can access IBM Quantum through Qiskit.
