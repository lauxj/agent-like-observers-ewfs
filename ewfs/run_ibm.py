import warnings
from pathlib import Path
import json
from datetime import datetime
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import reflex_agent
import guessing_agent
import betting_agent
import noiseless_simulation as noiseless
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler # Offline IBM hardware snapshot (no IBM account needed)

# Silence a common Qiskit warning
warnings.filterwarnings(
    "ignore",
    message="Trying to add QuantumRegister to a QuantumCircuit having a layout",
)

#-----------------------------------------------------------------------------------
# CONFIGURATIONS:

# Agents:
AGENTS = [
    ("Reflex", reflex_agent.build_measurement),
    ("Guessing", guessing_agent.build_measurement),
    ("Betting", betting_agent.build_measurement),
]

# Manual qubit placement:
# for this check IBM Quantum platform live calibration data
MANUAL_LAYOUTS_BY_SIZE = {
    4: [28, 29, 30, 31], # Reflex Agent
    5: [29, 30, 31, 32, 18], # Guessing agent
    6: [54, 61, 62, 60, 63, 59], # Betting Agent
}

# Transpilation
OPT_LEVEL = 0  # 0 leaves circuit the way it is

# Run Simulation with real backend noise:
DO_REAL_BACKEND_NOISE_SIM = True
NOISE_SHOTS = 10_000

# Run on  real hardware:
DO_REAL_HARDWARE_RUN = False
HARDWARE_SHOTS = 1000


# Backends to use
REAL_BACKENDS = {
    "ibm_torino": True,
}

# SB settings
SETTINGS = [
    ("A1B1", 1, 1),
    ("A1B2", 1, 2),
    ("A2B1", 2, 1),
    ("A2B2", 2, 2),
]

#-----------------------------------------------------------------------------------


def save_json(path: Path, obj):
    """Small helper function to save JSON files."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def S_from_E(E):
    """Compute S_SB from correlators E[(A,B)]."""
    return -E[(1, 1)] + E[(1, 2)] - E[(2, 1)] - E[(2, 2)] - 2


def S_error_from_E(E, shots_by_setting):
    """Return 1-sigma statistical uncertainty for S_SB from finite-shot sampling."""
    def n_for(setting):
        if isinstance(shots_by_setting, dict):
            return int(shots_by_setting[setting])
        return int(shots_by_setting)

    var = 0.0
    for setting in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        n = n_for(setting)
        e = float(E[setting])
        var += (1.0 - e * e) / n
    return var ** 0.5


def simulate_with_backend_noise(transpiled_by_setting, backend, shots):
    """Noise simulation using a backend-derived Aer NoiseModel."""
    noise_model = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=noise_model)

    E = {}
    for (A, B), tqc in transpiled_by_setting.items():
        counts = sim.run(tqc, shots=shots).result().get_counts()
        EAB = noiseless.exp_values_from_counts(counts, shots)
        E[(A, B)] = EAB

    return S_from_E(E)


def transpile_agent_circuits(agent_name, build_fn, alpha, beta1, beta2, backend):
    """Build and transpile the 4 SB-setting circuits for one agent. """

    print(f"\nAgent: {agent_name}")

    out = {}
    for label, A, B in SETTINGS:
        qc = build_fn(A, B, alpha, beta1, beta2)
        try:
            initial_layout = MANUAL_LAYOUTS_BY_SIZE[qc.num_qubits]
        except KeyError as exc:
            raise ValueError(f"No manual layout defined for n={qc.num_qubits} qubits.") from exc
        # Create transpiled circuit:
        tqc = transpile(
            qc,
            backend=backend,
            optimization_level=OPT_LEVEL,
            initial_layout=initial_layout,
        )

        ops = dict(tqc.count_ops())
        cz_n = ops.get("cz", 0)
        # Print information about the transpiled circuit
        print(f"  {label}: depth={tqc.depth()}  cz={cz_n}")

        out[(A, B)] = tqc

    return out


def get_counts_from_sampler_result(pub_res):
    """Extract counts dict from a SamplerV2 result item."""
    data = pub_res.data

    # Pick the first classical register found.
    if hasattr(data, "keys"):
        reg_names = list(data.keys())
        reg = reg_names[0]
        datum = data[reg]
    else:
        # Fallback for attribute-style containers
        reg_names = [k for k in data.__dict__.keys() if not k.startswith("_")]
        reg = reg_names[0]
        datum = getattr(data, reg)

    return datum.get_counts()


def transpile_all_agents(alpha, beta1, beta2, backend):
    """Transpile the 4 SB circuits for each agent and return a dict by agent name."""
    transpiled_by_agent = {}
    for agent_name, build_fn in AGENTS:
        transpiled_by_agent[agent_name] = transpile_agent_circuits(
            agent_name, build_fn, alpha, beta1, beta2, backend
        )
    return transpiled_by_agent


def run_noise_sim_for_backend(alpha, beta1, beta2, backend):
    """Run calibrated-noise simulations for all agents on one backend."""
    print("--- Noise sim (backend calibrations) ---")
    transpiled_by_agent = transpile_all_agents(alpha, beta1, beta2, backend)

    for agent_name in transpiled_by_agent:
        S_val = simulate_with_backend_noise(transpiled_by_agent[agent_name], backend, shots=NOISE_SHOTS)
        # Var(E) <= 1/N since E^2>=0 and Var(E)=(1-E^2)/N
        # Therefore we get a bound of Var(S) <= 4/N since S sums over four E
        S_err = (4.0 / NOISE_SHOTS) ** 0.5
        verdict = "VIOLATION" if S_val > 0 else "no violation"
        print(f"  -> {agent_name}: S_SB ≈ {S_val:.3f} ± {S_err:.3f} (1σ, shot noise) ({verdict})")


def submit_hardware_job(transpiled_by_agent, backend):
    """Submit one job to IBM real hardware containing all circuits across all agents."""
    sampler = Sampler(mode=backend)

    all_circuits = []
    meta_info = []  # (agent_name, A, B)

    for agent_name, circuits in transpiled_by_agent.items():
        for (A, B), tqc in circuits.items():
            all_circuits.append(tqc)
            meta_info.append((agent_name, A, B))

    job = sampler.run(all_circuits, shots=HARDWARE_SHOTS)
    results = job.result()
    return job, results, meta_info


def save_hardware_results(job, results, meta_info, backend):
    """Save hardware result counts and processed S_SB results for one backend run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"hardware_results/{backend.name}_{timestamp}")
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

    agent_E = {}

    for (agent_name, A, B), pub_res in zip(meta_info, results):
        counts = get_counts_from_sampler_result(pub_res)

        # Save raw counts
        fname = results_dir / f"{agent_name.replace(' ', '_')}_A{A}B{B}.json"
        save_json(fname, counts)

        # Expectation value
        EAB = noiseless.exp_values_from_counts(counts, HARDWARE_SHOTS)
        agent_E.setdefault(agent_name, {})[(A, B)] = EAB

    processed = {}
    for agent_name, E in agent_E.items():
        S_val = S_from_E(E)
        S_err = S_error_from_E(E, HARDWARE_SHOTS)
        processed[agent_name] = {
            "E_A1B1": float(E[(1, 1)]),
            "E_A1B2": float(E[(1, 2)]),
            "E_A2B1": float(E[(2, 1)]),
            "E_A2B2": float(E[(2, 2)]),
            "S_SB": float(S_val),
            "S_SB_err_1sigma": float(S_err),
            "shots": HARDWARE_SHOTS,
        }

    save_json(results_dir / "processed_results.json", processed)

    for agent_name, E in agent_E.items():
        S_val = S_from_E(E)
        S_err = S_error_from_E(E, HARDWARE_SHOTS)
        verdict = "VIOLATION" if S_val > 0 else "no violation"
        print(f"  -> {agent_name}: S_SB = {S_val:.3f} ± {S_err:.3f} (1σ) ({verdict})")

    print(f"\nSaved to: {results_dir.resolve()}")


def run_hardware_for_backend(alpha, beta1, beta2, backend):
    """Main function to run one real-hardware job for all agents on one backend."""
    print("\n--- Hardware run (SamplerV2) ---")

    transpiled_by_agent = transpile_all_agents(alpha, beta1, beta2, backend)
    job, results, meta_info = submit_hardware_job(transpiled_by_agent, backend)
    save_hardware_results(job, results, meta_info, backend)


def run_all_agents(alpha, beta1, beta2):
    """Main function to run all specified agents on the specified hardware (Simulation or Hardware)."""
    if not (DO_REAL_BACKEND_NOISE_SIM or DO_REAL_HARDWARE_RUN):
        return

    print("\n=== IBM Quantum Platform backend ===")
    service = get_runtime_service()

    for backend_name, enabled in REAL_BACKENDS.items():
        if not enabled:
            continue

        backend_real = get_real_backend(service, backend_name)
        print(f"\nUsing backend: {backend_real.name}")

        if DO_REAL_BACKEND_NOISE_SIM:
            run_noise_sim_for_backend(alpha, beta1, beta2, backend_real)

        if DO_REAL_HARDWARE_RUN:
            run_hardware_for_backend(alpha, beta1, beta2, backend_real)


# For IBM Quantum Platform:
def get_runtime_service():
    """Connect to IBM Quantum Platform. Credentials (API) code must be saved locally."""
    return QiskitRuntimeService()

def get_real_backend(service: QiskitRuntimeService, backend_name: str):
    """Get a real backend handle (live calibrations)."""
    return service.backend(backend_name)

#-----------------------------------------------------------------------------------------


if __name__ == "__main__":
    alpha, beta1, beta2 = noiseless.analytic_optimal_angles()
    run_all_agents(alpha, beta1, beta2)
