"""
fake_hardware.py
runs fake hardware noise simulations for all agents
"""

from pathlib import Path
import json
from datetime import datetime
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
try:
    from .ibm_transpilation import transpile_all_agents, PLOT_DIR as IBM_TRANSPILATION_PLOT_DIR
except ImportError:
    from ibm_transpilation import transpile_all_agents, PLOT_DIR as IBM_TRANSPILATION_PLOT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_FAKE = PROJECT_ROOT / "data" / "data_fake_hardware"
DATA_DIR_FAKE.mkdir(parents=True, exist_ok=True)

FAKE_NOISE_SEED = None
BACKEND_NAME = "ibm_torino"


def make_run_folder_name(backend, folder_ts=None):
    """Create the shared run-folder name used for data and plots."""
    if folder_ts is None:
        folder_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return folder_ts, f"{backend.name}_{folder_ts}"


def save_json(path: Path, obj):
    """Save JSON file."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def counts_to_jsonable(counts):
    """Convert Qiskit counts dict to JSON-serializable format."""
    return {str(k): int(v) for k, v in counts.items()}


def simulate_with_backend_noise(tqc, backend, shots, sim=None):
    """Run one fake-hardware noisy simulation."""
    if sim is None:
        noise_model = NoiseModel.from_backend(backend)
        sim = AerSimulator(noise_model=noise_model, seed_simulator=FAKE_NOISE_SEED)

    counts = sim.run(tqc, shots=shots).result().get_counts()
    return counts_to_jsonable(counts)


def prepare_fake_hardware_run(
    backend,
    save_plots=True,
    folder_ts=None,
    agent_builders=None,
    plots_subdir="transpiled_agents",
):
    """Prepare transpiled circuits and matching plot folder for one fake-hardware run."""
    folder_ts, run_folder_name = make_run_folder_name(backend, folder_ts)
    plots_dir = IBM_TRANSPILATION_PLOT_DIR / "fake_hardware" / run_folder_name / plots_subdir
    transpiled_by_agent = transpile_all_agents(
        backend,
        save_plots=save_plots,
        plots_dir=plots_dir,
        agent_builders=agent_builders,
    )
    return transpiled_by_agent, folder_ts


def run_fake_hardware_for_backend(
    backend,
    transpiled_by_agent,
    shots=10_000,
    folder_ts=None,
    result_filename="fake_hardware_noise_sim.json",
):
    """Run calibrated-noise simulations for all agents on one backend and save raw data."""
    print("\n=== Fake hardware simulation ===")
    print(f"Backend: {backend.name}")
    print(f"Shots: {shots}")

    noise_model = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=noise_model, seed_simulator=FAKE_NOISE_SEED)

    run_data = {
        "agents": {},
        "backend": backend.name,
        "kind": "fake_hardware_noise_sim",
        "shots": int(shots),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for agent_name, tqc in transpiled_by_agent.items():
        counts = simulate_with_backend_noise(tqc, backend, shots=shots, sim=sim)
        print(f"  {agent_name}: done")
        run_data["agents"][agent_name] = {"counts": counts}

    folder_ts, run_folder_name = make_run_folder_name(backend, folder_ts)
    out_dir = DATA_DIR_FAKE / run_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / result_filename
    save_json(out_path, run_data)
    print(f"Saved data → {out_path.resolve()}")


if __name__ == "__main__":
    backend = QiskitRuntimeService().backend(BACKEND_NAME)
    transpiled, folder_ts = prepare_fake_hardware_run(backend, save_plots=True)
    run_fake_hardware_for_backend(backend, transpiled, shots=10_000, folder_ts=folder_ts)
