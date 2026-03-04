import warnings
from pathlib import Path
import json
from datetime import datetime
import pickle

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.visualization import circuit_drawer
from ewfs.agents.agents import build_circuit_reflex, build_circuit_guessing, build_circuit_betting
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler  # IBM Quantum Platform

import matplotlib.pyplot as plt

# Silence a common Qiskit warning
warnings.filterwarnings(
    "ignore",
    message="Trying to add QuantumRegister to a QuantumCircuit having a layout",
)

# -----------------------------------------------------------------------------------
# PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Plots: root/results/plots_ibm_transpilation
PLOT_DIR = PROJECT_ROOT / "results" / "plots_ibm_transpilation"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Data: root/data/data_fake_hardware and root/data/data_real_hardware
DATA_DIR_FAKE = PROJECT_ROOT / "data" / "data_fake_hardware"
DATA_DIR_FAKE.mkdir(parents=True, exist_ok=True)

DATA_DIR_REAL = PROJECT_ROOT / "data" / "data_real_hardware"
DATA_DIR_REAL.mkdir(parents=True, exist_ok=True)

#-----------------------------------------------------------------------------------
# CONFIGURATIONS:

# Agents:
AGENTS = [
    ("Betting", build_circuit_betting)
]
#AGENTS = [
#    ("Reflex", build_circuit_reflex),
#    ("Guessing", build_circuit_guessing),
#    ("Betting", build_circuit_betting),
#]


# Manual qubit placement:
# for this check IBM Quantum platform live calibration data
MANUAL_LAYOUTS_BY_SIZE = {
    7: [28, 29, 30, 31, 32, 33, 34],
    8: [28, 29, 30, 31, 32, 33, 34, 35],
    9: [54,61,60,59,62,58,63,14,129],
}

# Transpilation
OPT_LEVEL = 0  # 0 leaves circuit the way it is

# Run Simulation with fake hardware noise (backend calibrations):
DO_FAKE_HARDWARE_NOISE_SIM = True
NOISE_SHOTS = 1000

# Run on  real hardware:
DO_REAL_HARDWARE_RUN = False
HARDWARE_SHOTS = 100


# Backends to use
REAL_BACKENDS = {
    "ibm_torino": True,
}

#-----------------------------------------------------------------------------------


def save_json(path: Path, obj):
    """Small helper function to save JSON files."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def counts_to_jsonable(counts):
    """Convert Qiskit counts dict to a JSON-serializable dict."""
    return {str(k): int(v) for k, v in counts.items()}


def simulate_with_backend_noise(tqc, backend, shots):
    """Noise simulation using a backend-derived Aer NoiseModel.

    Returns counts_json.
    """
    noise_model = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=noise_model)
    counts = sim.run(tqc, shots=shots).result().get_counts()
    return counts_to_jsonable(counts)


def transpile_agent_circuit(agent_name, build_fn, backend):
    """Build and transpile ONE circuit for one agent."""
    print(f"\nAgent: {agent_name}")

    qc = build_fn()

    try:
        initial_layout = MANUAL_LAYOUTS_BY_SIZE[qc.num_qubits]
    except KeyError as exc:
        raise ValueError(f"No manual layout defined for n={qc.num_qubits} qubits.") from exc

    tqc = transpile(
        qc,
        backend=backend,
        optimization_level=OPT_LEVEL,
        initial_layout=initial_layout,
    )

    ops = dict(tqc.count_ops())
    cz_n = ops.get("cz", 0)
    print(f"  depth={tqc.depth()}  cz={cz_n}")

    # Save transpiled circuit plot
    tqc.name = f"{agent_name}"
    agent_dir = PLOT_DIR / "transpiled" / agent_name.replace(" ", "_")
    agent_dir.mkdir(parents=True, exist_ok=True)
    plot_path = agent_dir / f"circuit_depth{tqc.depth()}_cz{cz_n}.png"
    fig = circuit_drawer(tqc, output="mpl", fold=-1)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return tqc


def transpile_all_agents(backend):
    out = {}
    for agent_name, build_fn in AGENTS:
        out[agent_name] = transpile_agent_circuit(agent_name, build_fn, backend)
    return out


def run_noise_sim_for_backend(backend):
    """Run calibrated-noise simulations for all agents on one backend and save raw data."""
    print("--- Noise sim (backend calibrations) ---")

    transpiled_by_agent = transpile_all_agents(backend)

    run_data = {
        "kind": "fake_hardware_noise_sim",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "backend": backend.name,
        "shots": int(NOISE_SHOTS),
        "agents": {},
    }

    for agent_name, tqc in transpiled_by_agent.items():
        counts = simulate_with_backend_noise(tqc, backend, shots=NOISE_SHOTS)
        print(f"  -> {agent_name}: done")
        run_data["agents"][agent_name] = {"counts": counts}

    out_dir = DATA_DIR_FAKE / f"{backend.name}_{run_data['timestamp']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "fake_hardware_noise_sim.json", run_data)
    print(f"Saved fake-hardware noise-sim data to: {out_dir.resolve()}")


def submit_hardware_job(transpiled_by_agent, backend):
    """Submit one job to IBM real hardware containing one circuit per agent."""
    sampler = Sampler(mode=backend)

    all_circuits = []
    meta_info = []  # agent_name

    for agent_name, tqc in transpiled_by_agent.items():
        all_circuits.append(tqc)
        meta_info.append(agent_name)

    job = sampler.run(all_circuits, shots=HARDWARE_SHOTS)
    results = job.result()
    return job, results, meta_info


def save_hardware_results(job, results, meta_info, backend):
    """Save hardware result counts for one backend run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = DATA_DIR_REAL / f"{backend.name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        job_id = job.job_id()
    except Exception:
        job_id = None

    save_json(
        results_dir / "job_info.json",
        {
            "backend": backend.name,
            "job_id": job_id,
            "shots": HARDWARE_SHOTS,
            "timestamp": timestamp,
        },
    )

    raw_counts_all = {}

    for agent_name, pub_res in zip(meta_info, results):
        raw_counts_all[agent_name] = get_counts_from_sampler_result(pub_res)

    save_json(results_dir / "raw_counts.json", raw_counts_all)

    processed = {
        agent: {"counts_by_register": raw_counts_all[agent], "shots": HARDWARE_SHOTS}
        for agent in raw_counts_all
    }
    save_json(results_dir / "processed_results.json", processed)

    with open(results_dir / "raw_sampler_result.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved to: {results_dir.resolve()}")


def run_hardware_for_backend(backend):
    """Main function to run one real-hardware job for all agents on one backend."""
    print("\n--- Hardware run (SamplerV2) ---")

    transpiled_by_agent = transpile_all_agents(backend)
    job, results, meta_info = submit_hardware_job(transpiled_by_agent, backend)
    save_hardware_results(job, results, meta_info, backend)


def run_all_agents():
    """Main function to run all specified agents on the specified hardware (Simulation or Hardware)."""
    if not (DO_FAKE_HARDWARE_NOISE_SIM or DO_REAL_HARDWARE_RUN):
        return

    print("\n=== IBM Quantum Platform backend ===")
    service = get_runtime_service()

    for backend_name, enabled in REAL_BACKENDS.items():
        if not enabled:
            continue

        backend_real = get_real_backend(service, backend_name)
        print(f"\nUsing backend: {backend_real.name}")

        if DO_FAKE_HARDWARE_NOISE_SIM:
            run_noise_sim_for_backend(backend_real)

        if DO_REAL_HARDWARE_RUN:
            run_hardware_for_backend(backend_real)


# For IBM Quantum Platform:
def get_runtime_service():
    """Connect to IBM Quantum Platform. Credentials (API) code must be saved locally."""
    return QiskitRuntimeService()

def get_real_backend(service: QiskitRuntimeService, backend_name: str):
    """Get a real backend handle (live calibrations)."""
    return service.backend(backend_name)

#-----------------------------------------------------------------------------------------


def get_counts_from_sampler_result(pub_res):
    """Extract counts for all classical registers from a SamplerV2 result item.

    Returns a dict: {<creg_name>: {bitstring: count, ...}, ...}
    """
    data = pub_res.data

    # SamplerV2 exposes one entry per classical register.
    if hasattr(data, "keys"):
        reg_names = list(data.keys())
        out = {}
        for reg in reg_names:
            out[str(reg)] = counts_to_jsonable(data[reg].get_counts())
        return out

    # Fallback for attribute-style containers
    out = {}
    reg_names = [k for k in data.__dict__.keys() if not k.startswith("_")]
    for reg in reg_names:
        datum = getattr(data, reg)
        if hasattr(datum, "get_counts"):
            out[str(reg)] = counts_to_jsonable(datum.get_counts())
    return out


if __name__ == "__main__":
    run_all_agents()
