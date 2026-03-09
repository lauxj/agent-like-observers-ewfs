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
    ("Reflex Agent", build_circuit_reflex),
    ("Guessing Agent", build_circuit_guessing),
    ("Betting Agent", build_circuit_betting),
]


# Manual qubit placement:
# for this check IBM Quantum platform live calibration data
MANUAL_LAYOUTS_BY_SIZE = {
    6: [28, 29, 30, 31, 14,129], # Reflex Agent
    7: [29, 30, 31, 32, 18,14,129], # Guessing agent
    8: [54, 61, 62, 60, 63, 59,14,129], # Betting Agent
}

#9: [54,61,60,59,62,58,63,14,129]
#9: [18,12,11,10,13,9,14,1,132]

# Transpilation
OPT_LEVEL = 0  # 0 leaves circuit the way it is

# Run Simulation with fake hardware noise (backend calibrations):
DO_FAKE_HARDWARE_NOISE_SIM = True
NOISE_SHOTS = 10_000

# --- DEBUG (safe to delete later) ---
# If True, write a small LF-relevant summary JSON and print comparison to the previous run.
DEBUG_COMPARE_FAKE_RUNS = False
# If set to an int, Aer uses deterministic Monte-Carlo sampling; if None, each run differs stochastically.
FAKE_NOISE_SEED = None
# --- end DEBUG ---

# Run on  real hardware:
DO_REAL_HARDWARE_RUN = True
HARDWARE_SHOTS = 300


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


# --- DEBUG/ANALYSIS HELPERS (safe to delete) ---
def _as_4bit_strings(counts: dict) -> dict:
    """Ensure keys are 4-bit strings. Works if keys are already strings."""
    out = {}
    for k, v in counts.items():
        s = str(k)
        out[s] = int(v)
    return out


def lf_branch_denominators(counts: dict) -> dict:
    """Compute denominators for conditioning on the last two bits (c1,c0).

    Matches your lf_violations convention: c3,c2,c1,c0 = s[0],s[1],s[2],s[3].
    Returns dict with keys '00','01','10','11' for (c1,c0).
    """
    counts = _as_4bit_strings(counts)
    dens = {"00": 0, "01": 0, "10": 0, "11": 0}
    for s, n in counts.items():
        if len(s) < 4:
            continue
        c1, c0 = s[2], s[3]
        dens[c1 + c0] = dens.get(c1 + c0, 0) + int(n)
    return dens


def counts_sha256(counts: dict) -> str:
    """Hash counts dict deterministically (order-independent)."""
    items = sorted((str(k), int(v)) for k, v in counts.items())
    blob = "\n".join(f"{k}:{v}" for k, v in items)
    return __import__("hashlib").sha256(blob.encode("utf-8")).hexdigest()


def compare_counts(prev: dict, curr: dict) -> dict:
    """Simple distribution-difference metrics."""
    prev = _as_4bit_strings(prev)
    curr = _as_4bit_strings(curr)
    n_prev = sum(prev.values()) or 1
    n_curr = sum(curr.values()) or 1

    keys = set(prev) | set(curr)
    l1 = 0.0
    max_abs = 0.0
    max_key = None
    for k in keys:
        p = prev.get(k, 0) / n_prev
        q = curr.get(k, 0) / n_curr
        d = abs(p - q)
        l1 += d
        if d > max_abs:
            max_abs = d
            max_key = k

    return {
        "n_prev": int(n_prev),
        "n_curr": int(n_curr),
        "l1_distance": float(l1),
        "max_abs_diff": float(max_abs),
        "max_abs_diff_key": max_key,
        "den_prev": lf_branch_denominators(prev),
        "den_curr": lf_branch_denominators(curr),
        "sha_prev": counts_sha256(prev),
        "sha_curr": counts_sha256(curr),
    }


def find_latest_fake_file(backend_name: str) -> Path | None:
    """Return the most recent fake_hardware_noise_sim.json for this backend, if any."""
    candidates = sorted(DATA_DIR_FAKE.glob(f"{backend_name}_*/fake_hardware_noise_sim.json"))
    if not candidates:
        return None
    # choose newest by folder timestamp (lexicographic works for YYYYMMDD_HHMMSS)
    return candidates[-1]
# --- END DEBUG HELPERS ---


def simulate_with_backend_noise(tqc, backend, shots, sim=None):
    """Noise simulation using a backend-derived Aer NoiseModel.

    If `sim` is provided, it is reused (recommended for reproducibility/debugging).
    Returns counts_json.
    """
    if sim is None:
        noise_model = NoiseModel.from_backend(backend)
        sim = AerSimulator(noise_model=noise_model, seed_simulator=FAKE_NOISE_SEED)

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


def run_noise_sim_for_backend(backend, transpiled_by_agent):
    """Run calibrated-noise simulations for all agents on one backend and save raw data."""
    print("--- Noise sim (backend calibrations) ---")

    # Reuse one simulator so all agents share the same frozen noise model in this run
    noise_model = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=noise_model, seed_simulator=FAKE_NOISE_SEED)

    # DEBUG: load the previous fake-hardware file (if any) before writing the new one
    prev_path = find_latest_fake_file(backend.name) if DEBUG_COMPARE_FAKE_RUNS else None
    prev_data = None
    if prev_path is not None and prev_path.exists():
        try:
            with open(prev_path, "r") as f:
                prev_data = json.load(f)
        except Exception:
            prev_data = None

    run_data = {
        "agents": {},
        "kind": "fake_hardware_noise_sim",
        "shots": int(NOISE_SHOTS),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for agent_name, tqc in transpiled_by_agent.items():
        counts = simulate_with_backend_noise(tqc, backend, shots=NOISE_SHOTS, sim=sim)
        print(f"  -> {agent_name}: done")
        run_data["agents"][agent_name] = {"counts": counts}

    folder_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = DATA_DIR_FAKE / f"{backend.name}_{folder_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "fake_hardware_noise_sim.json"
    save_json(out_json, run_data)

    # DEBUG: write small LF-relevant summary + compare to previous run
    if DEBUG_COMPARE_FAKE_RUNS:
        summary = {
            "backend": backend.name,
            "shots": int(NOISE_SHOTS),
            "seed_simulator": FAKE_NOISE_SEED,
            "timestamp": run_data["timestamp"],
            "agents": {},
            "previous_file": str(prev_path) if prev_path is not None else None,
        }

        for agent_name in run_data["agents"].keys():
            c = run_data["agents"][agent_name]["counts"]
            summary["agents"][agent_name] = {
                "counts_sha256": counts_sha256(c),
                "den_c1c0": lf_branch_denominators(c),
            }

        save_json(out_dir / "debug_summary.json", summary)

        if prev_data is not None and isinstance(prev_data, dict) and "agents" in prev_data:
            print("\n[DEBUG] Comparing to previous fake-noise file:")
            print("   prev:", prev_path)
            print("   curr:", out_json)
            for agent_name in run_data["agents"].keys():
                if agent_name in prev_data.get("agents", {}) and "counts" in prev_data["agents"][agent_name]:
                    prev_counts = prev_data["agents"][agent_name]["counts"]
                    curr_counts = run_data["agents"][agent_name]["counts"]
                    cmp = compare_counts(prev_counts, curr_counts)
                    print(f"   {agent_name}: L1={cmp['l1_distance']:.4f}  max|Δp|={cmp['max_abs_diff']:.4f} @ {cmp['max_abs_diff_key']}  den_prev={cmp['den_prev']}  den_curr={cmp['den_curr']}")
                else:
                    print(f"   {agent_name}: no matching agent in previous file")

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

    run_data = {
        "agents": {},
        "kind": "real_hardware_run",
        "shots": int(HARDWARE_SHOTS),
        # same second-level time as folder naming, but ISO-like
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for agent_name, pub_res in zip(meta_info, results):
        counts = get_counts_from_sampler_result(pub_res)
        run_data["agents"][agent_name] = {"counts": counts}

    # Save a single unified result file (same schema as noiseless runs)
    save_json(results_dir / "real_hardware_run.json", run_data)

    with open(results_dir / "raw_sampler_result.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved to: {results_dir.resolve()}")


def run_hardware_for_backend(backend, transpiled_by_agent):
    """Main function to run one real-hardware job for all agents on one backend."""
    print("\n--- Hardware run (SamplerV2) ---")

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

        # Transpile ONCE per backend and reuse for both fake-noise sim and real hardware
        transpiled_by_agent = transpile_all_agents(backend_real)

        if DO_FAKE_HARDWARE_NOISE_SIM:
            run_noise_sim_for_backend(backend_real, transpiled_by_agent)

        if DO_REAL_HARDWARE_RUN:
            run_hardware_for_backend(backend_real, transpiled_by_agent)


# For IBM Quantum Platform:
def get_runtime_service():
    """Connect to IBM Quantum Platform. Credentials (API) code must be saved locally."""
    return QiskitRuntimeService()

def get_real_backend(service: QiskitRuntimeService, backend_name: str):
    """Get a real backend handle (live calibrations)."""
    return service.backend(backend_name)

#-----------------------------------------------------------------------------------------


def get_counts_from_sampler_result(pub_res):
    """Extract *single-register* counts from a SamplerV2 result item.

    Project convention: each circuit uses exactly ONE classical register.
    Returns: {bitstring: count, ...}
    """
    data = pub_res.data

    # Mapping-like container: one entry per classical register
    if hasattr(data, "keys"):
        reg_names = list(data.keys())
        if len(reg_names) != 1:
            raise ValueError(
                f"Expected exactly 1 classical register, found {len(reg_names)}: {reg_names}"
            )
        reg = reg_names[0]
        return counts_to_jsonable(data[reg].get_counts())

    # Attribute-style container fallback
    reg_names = [k for k in data.__dict__.keys() if not k.startswith("_")]
    regs_with_counts = []
    for reg in reg_names:
        datum = getattr(data, reg)
        if hasattr(datum, "get_counts"):
            regs_with_counts.append(reg)

    if len(regs_with_counts) != 1:
        raise ValueError(
            f"Expected exactly 1 classical register with counts, found {len(regs_with_counts)}: {regs_with_counts}"
        )

    reg = regs_with_counts[0]
    datum = getattr(data, reg)
    return counts_to_jsonable(datum.get_counts())


if __name__ == "__main__":
    run_all_agents()
