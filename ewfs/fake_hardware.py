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
from ibm_transpilation import transpile_all_agents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_FAKE = PROJECT_ROOT / "data" / "data_fake_hardware"
DATA_DIR_FAKE.mkdir(parents=True, exist_ok=True)

FAKE_NOISE_SEED = None


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


def run_fake_hardware_for_backend(backend, transpiled_by_agent, shots=10_000):
    """Run calibrated-noise simulations for all agents on one backend and save raw data."""
    print("\n=== Fake hardware simulation ===")
    print(f"Backend: {backend.name}")
    print(f"Shots: {shots}")

    noise_model = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=noise_model, seed_simulator=FAKE_NOISE_SEED)

    run_data = {
        "agents": {},
        "kind": "fake_hardware_noise_sim",
        "shots": int(shots),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for agent_name, tqc in transpiled_by_agent.items():
        counts = simulate_with_backend_noise(tqc, backend, shots=shots, sim=sim)
        print(f"  {agent_name}: done")
        run_data["agents"][agent_name] = {"counts": counts}

    folder_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = DATA_DIR_FAKE / f"{backend.name}_{folder_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(out_dir / "fake_hardware_noise_sim.json", run_data)
    print(f"Saved data → {out_dir.resolve()}")


if __name__ == "__main__":
    backend = QiskitRuntimeService().backend("ibm_torino")
    transpiled = transpile_all_agents(backend, save_plots=False)
    run_fake_hardware_for_backend(backend, transpiled, shots=10_000)
