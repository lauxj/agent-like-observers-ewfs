# Local Friendliness violations with Agents on Quantum Computers

This repository contains the code and data for my master's thesis project,
"Local Friendliness violations with Agents on Quantum Computers".

The project runs an extended Wigner's friend scenario (EWFS) on quantum
computers. Quantum agents are used to represent the friend  Charlie in the experiment.
The main pipeline of the project is:

1. Circuit construction
2. Noiseless (ideal) simulation
3. Transpilation of circuits for IBM backends
4. Noise simulation
5. IBM hardware runs
6. Evaluation of agents

## Project Structure

- `ewfs/` -> Python scripts for circuits, experiments, and analysis
- `scripts/` -> small entry-point scripts with the main settings
- `data/` -> saved thesis data and outputs from new runs
- `notebooks/` -> project notebooks
- `results/` -> generated plots and output files 

## Files in `scripts/` and `ewfs/`
```bash
.
├── scripts/
│   ├── run_experiment.py        # main file for running experiments
│   └── evaluation.py           # main file for making evaluation
└── ewfs/
   ├── experiments/
   │   ├── run.py              # main experiment runner
   │   ├── noiseless_simulation.py   # noiseless simulator script
   │   ├── fake_hardware.py    # noise-simulation script
   │   ├── real_hardware.py    # real IBM hardware runs
   │   └── ibm_transpilation.py # transpilation for IBM backends
   ├── analysis/
   │   ├── agent_evaluation.py # creates thesis plots from data
   │   ├── lf_violations.py    # LF correlator & violation calculations
   │   ├── plot_ibm_connectivity.py # IBM connectivity/layout plots
   │   └── time_ordering_hardware.py # hardware scheduler timing plots
   └── circuits/
       ├── agents.py           # builds agent quantum circuits
       └── accuracy_test_circuits.py # relaxed LF accuracy-test circuits
```


## Files in `scripts/`

- `scripts/run_experiment.py` -> main file for running experiments
- `scripts/evaluation.py` -> main file for making evaluation

## Files in `ewfs/`

- `ewfs/experiments/run.py` -> main experiment runner
- `ewfs/analysis/agent_evaluation.py` -> creates the thesis plots from saved data or new data
- `ewfs/circuits/agents.py` -> builds the agent quantum circuits
- `ewfs/circuits/accuracy_test_circuits.py` -> builds the relaxed LF accuracy-test circuits
- `ewfs/experiments/noiseless_simulation.py` -> noiseless simulator script
- `ewfs/experiments/fake_hardware.py` -> noise-simulation script
- `ewfs/experiments/real_hardware.py` -> real IBM hardware runs
- `ewfs/experiments/ibm_transpilation.py` -> transpilation for IBM backends
- `ewfs/analysis/lf_violations.py` -> LF correlator and LF violation calculations
- `ewfs/analysis/plot_ibm_connectivity.py` -> IBM connectivity/layout plots
- `ewfs/analysis/time_ordering_hardware.py` -> hardware scheduler timing plots

## Installation

Use Python 3.10. Newer Python versions may give dependency problems.
Clone the repository and install the requirements in a virtual environment.

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

Using VS Code: A simple installation with VS Code would be to download the repository as a ZIP file, store it locally and then open it in VS Code. In the Terminal in VS Code, run the bash command above from the 3rd line.
To run the notebook, choose the installed .venv kernel i VS Code. 

## Usage

To open the demo notebook via Terminal:

```bash
python -m notebook notebooks/project_demo.ipynb
```

To make a new experiment run, open `scripts/run_experiment.py`, change the
settings near the top, and run the file. The more detailed settings are in
`ewfs/experiments/run.py`.

New runs are saved in the normal data folders:

- `data/data_noiseless_simulation/`
- `data/data_fake_hardware/`
- `data/data_real_hardware/`

To reproduce the thesis plots, open `scripts/evaluation.py` and use the runs in
`data/paperdata/`. You can also change it to use new runs from the normal data
folders.

Real IBM hardware runs need an IBM Quantum API token saved on your machine.
Noiseless simulation and evaluation from saved data do not need an IBM account.

Transpilation and fake-hardware simulation do not submit real hardware jobs, but
they still load an IBM backend through Qiskit. For those parts, you also need
the IBM token below.

## IBM Quantum API Token

For real hardware runs, and for loading IBM backends, you need access to IBM
Quantum.

1. Create an IBM Quantum account at `https://quantum.cloud.ibm.com/`.
2. Log in to the IBM Quantum Platform.
3. Create or copy an IBM Cloud API key from your account/API key settings.
4. Save the API key locally with Qiskit:

```bash
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='YOUR_IBM_API_KEY', set_as_default=True, overwrite=True)"
```

Replace `YOUR_IBM_API_KEY` with your IBM Quantum API key. You only need to do
this once on a machine. After that, Qiskit can load IBM backends from this
project.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
