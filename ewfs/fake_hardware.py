"""
Run fake-hardware simulations using calibration data from an IBM backend.

The script transpiles each agent circuit for a selected backend, builds a noisy
Aer simulator from that backend, and saves the resulting counts as JSON.
"""

import json
from datetime import datetime
from pathlib import Path
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from ibm_transpilation import PLOT_DIR as TRANSPILATION_PLOT_DIR
from ibm_transpilation import transpile_all_agents

# define directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_FAKE = PROJECT_ROOT / "data" / "data_fake_hardware"
DEFAULT_RESULT_FILENAME = "fake_hardware_noise_sim.json"

# prepare for fake hardware simulation by transpiling all circuits
def prepare_fake_hardware_run(
    backend,
    save_plots=True,
    folder_ts=None,
    agent_builders=None,
    plots_subdir="transpiled_agents",
):
    """Transpile all selected agent circuits before the noisy simulation."""
    folder_ts, run_folder_name = make_run_folder_name(backend, folder_ts)
    plots_dir = None
    if save_plots:
        plots_dir = TRANSPILATION_PLOT_DIR / "fake_hardware" / run_folder_name / plots_subdir

    transpiled_by_agent = transpile_all_agents(
        backend,
        save_plots=save_plots,
        plots_dir=plots_dir,
        agent_builders=agent_builders,
    )
    return transpiled_by_agent, folder_ts

# run noise simulation using the backend Noise model and save results
def run_fake_hardware_for_backend(
    backend,
    transpiled_by_agent,
    shots=10_000,
    folder_ts=None,
    result_filename=DEFAULT_RESULT_FILENAME,
):
    """Run noisy simulations for all transpiled agent circuits and save counts."""
    print("\n=== Fake hardware simulation ===")
    print(f"Backend: {backend.name}")
    print(f"Shots: {shots}")

    # Noise model from backend calibration data:
    noise_model = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=noise_model)

    run_data = {
        "agents": {},
        "backend": backend.name,
        "kind": "fake_hardware_noise_sim",
        "shots": int(shots),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for agent_name, tqc in transpiled_by_agent.items():
        counts = simulate_circuit(tqc, sim, shots)
        print(f"  {agent_name}: done")
        run_data["agents"][agent_name] = {"counts": counts}

    folder_ts, run_folder_name = make_run_folder_name(backend, folder_ts)
    out_dir = DATA_DIR_FAKE / run_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / result_filename
    save_json(out_path, run_data)
    print(f"Saved data to: {out_path.resolve()}")
    return out_path


def simulate_circuit(tqc, simulator, shots):
    """Run one transpiled circuit on the noisy simulator."""
    counts = simulator.run(tqc, shots=shots).result().get_counts()
    return counts_to_jsonable(counts)


def make_run_folder_name(backend, folder_ts=None):
    """Create the timestamp and shared run-folder name for data and plots."""
    if folder_ts is None:
        folder_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return folder_ts, f"{backend.name}_{folder_ts}"


def save_json(path: Path, obj):
    """Write one formatted JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def counts_to_jsonable(counts):
    """Convert Qiskit counts to plain JSON keys and integer values."""
    return {str(k): int(v) for k, v in counts.items()}


if __name__ == "__main__":
    # can be run here to test but usually gets called from the main run.py script
    backend = QiskitRuntimeService().backend("ibm_marrakesh")
    transpiled, folder_ts = prepare_fake_hardware_run(backend, save_plots=True)
    run_fake_hardware_for_backend(backend, transpiled, shots=10_000, folder_ts=folder_ts)
