# Local Friendliness violations with Agents on Quantum Computers

This repository contains the code and data for my master's thesis project on "Local Friendliness violations with Agents on Quantum Computers".

[Insert link to thesis here]

The main goal of the project is to run an extended Wigner's friend scenario on a quantum computer and use quantum agents to represent a friend in the experiment.
This repo contains the files for the pipeline of the experiment:

1. Circuit construction
2. Noiseless (ideal) simulation
3. Transpilation of circuits for IBM backends
4. Noise simulation
5. IBM hardware runs
6. Evaluation of agents

## Project Structure

- `ewfs/`: all Python scripts for circuits, experiments, and analysis
- `scripts/`: simple entry-point scripts with the main settings to change
- `data/`: saved thesis data and local output for new runs
- `notebooks/`: project notebooks
- `results/`: folder for generated plots and output files

## Files in `scripts/`

- `scripts/run_experiment.py`: main file to run the experiment, choose what to run
- `scripts/evaluation.py`: main file for evaluation, choose which data to consider for evaluation


## Files in `ewfs/`

- `ewfs/experiments/run.py`: main experiment runner
- `ewfs/analysis/agent_evaluation.py`: creates the thesis plots from saved data
- `ewfs/circuits/agents.py`: defines the agent quantum circuit
- `ewfs/circuits/accuracy_test_circuits.py`: defines the relaxed LF accuracy-test circuits
- `ewfs/experiments/noiseless_simulation.py`: noiseless simulator runs.
- `ewfs/experiments/fake_hardware.py`: noise-simulation runs
- `ewfs/experiments/real_hardware.py`: real IBM hardware runs
- `ewfs/experiments/ibm_transpilation.py`: transpilation for IBM backends
- `ewfs/analysis/lf_violations.py`: LF correlator and violation calculations
- `ewfs/analysis/plot_ibm_connectivity.py`: IBM connectivity/layout plots
- `ewfs/analysis/time_ordering_hardware.py`: hardware scheduler timing plots



## Installation

The project should be run inside its own Python environment. This keeps the
packages for this project separate from other Python projects on the computer.

First, install Python if it is not already installed. Python 3.10 is recommended.

Then open a terminal and move into the project folder. For example:

```bash
cd path/to/masters_thesis_project
```

Create a new environment called `.venv`:

```bash
python3.10 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

After activation, the terminal usually shows `(.venv)` at the beginning of the
line. This means the project environment is active.

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The installation only needs to be done once. When returning to the project later,
open a terminal in the project folder and activate the environment again:

```bash
source .venv/bin/activate
```

On Windows, the activation command is usually:

```bash
.venv\Scripts\activate
```

Another recent Python 3 version may also work, but Python 3.10 is the safest
choice for this repository.

## Running the Project in VS Code

The easiest way to work with this repository is to open the project folder in
VS Code and use one project-specific Python environment.

Recommended VS Code workflow:

1. Open this repository folder in VS Code.
2. Install the VS Code Python and Jupyter extensions if VS Code asks for them.
3. Open the VS Code terminal with `Terminal > New Terminal`.
4. Create a virtual environment:

```bash
python3.10 -m venv .venv
```

If `python3.10` is not available, use another recent Python 3 version:

```bash
python -m venv .venv
```

5. Activate the environment:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

6. Install the requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

7. Tell VS Code to use this environment:

- For Python scripts, use `Python: Select Interpreter` and choose `.venv`.
- For notebooks, use the kernel selector in the top-right of the notebook and
  choose the `.venv` Python environment.

After this setup, Python files such as `scripts/run_experiment.py` and
`scripts/evaluation.py` can be run from VS Code. The detailed run settings are
near the top of those files.

The easiest way to inspect the full pipeline interactively is the demo notebook:

```text
notebooks/project_demo.ipynb
```

Open it in VS Code, select the `.venv` kernel, and run the cells from top to
bottom.


## Usage

To make a new experimental run, open `scripts/run_experiment.py`, choose the
settings near the top of the file, and run that file. The detailed experiment
settings stay in `ewfs/experiments/run.py`.

New runs are saved in the normal data folders:

- `data/data_noiseless_simulation/`
- `data/data_fake_hardware/`
- `data/data_real_hardware/`

To reproduce the thesis plots, open `scripts/evaluation.py`, by using the runs in `data/paperdata/`. 
It can also be switched to use newly generated runs from the normal data folders.

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
be done once on a machine. After that, `scripts/run_experiment.py` and
`ewfs/experiments/real_hardware.py`
can access IBM Quantum through Qiskit.
