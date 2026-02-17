import warnings
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import reflex_agent
import guessing_agent
import betting_agent
import noiseless_simulation as noiseless
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler # Offline IBM hardware snapshot (no IBM account needed)

# Silence a common Qiskit warning that does not affect correctness
warnings.filterwarnings(
    "ignore",
    message="Trying to add QuantumRegister to a QuantumCircuit having a layout",
)

# Manual qubit placement
MANUAL_LAYOUTS_BY_SIZE = {
    4: [28,29,30,31],
    5: [29,30,31,32,18],
    6: [54,61,62,60,63,59]
}
# Transpiler settings
OPT_LEVEL = 0

# Verification: compare original vs transpiled distributions (ideal simulator)
DO_TVD_CHECK = False
TVD_SHOTS = 10_000
DO_TVD_CHECK_ONLINE = False

# Cache one simulator per backend:
NOISE_SHOTS = 10_000

# Simluation using real backend noise:
DO_REAL_BACKEND_NOISE_SIM = True

# Real hardware run:
DO_REAL_HARDWARE_RUN = False
HARDWARE_SHOTS = 1000

 # IBM QPU:
REAL_BACKENDS = {
    "ibm_torino": True
}

# SB settings:
SETTINGS = [
    ("A1B1", 1, 1),
    ("A1B2", 1, 2),
    ("A2B1", 2, 1),
    ("A2B2", 2, 2),
]

# Agents:
AGENTS = [
    ("Reflex Agent", reflex_agent.build_measurement),
    ("Guessing Agent", guessing_agent.build_measurement),
    ("Betting Agent", betting_agent.build_measurement),
]

# Plotting: "none", "show", or "save"
PLOT_MODE = "none"
PLOT_DIR = Path("../plots")


def _normalize_counts(counts, shots):
    return {k: v / shots for k, v in counts.items()}


def _tvd(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def tvd_original_vs_transpiled(qc, tqc, shots):
    """Total Variation Distance (TVD) between output distributions of qc and tqc.

    We run both circuits on an *ideal* (noise-free) simulator.
    TVD \u2248 0 means the transpiled circuit preserves the original measurement statistics.
    """
    sim = AerSimulator()  # ideal simulator (no noise)

    c1 = sim.run(qc, shots=shots).result().get_counts()
    c2 = sim.run(tqc, shots=shots).result().get_counts()

    p1 = _normalize_counts(c1, shots)
    p2 = _normalize_counts(c2, shots)
    return _tvd(p1, p2)



# IBM Quantum Platform:
def get_runtime_service():
    """Connect to IBM Quantum Platform. Credentials (API) code must be saved locally."""
    return QiskitRuntimeService()

def get_real_backend(service: QiskitRuntimeService, backend_name: str):
    """Get a real backend handle (live calibrations)."""
    return service.backend(backend_name)

def simulate_with_backend_noise(transpiled_by_setting, backend, shots, sim=None):
    """Compute S_SB using an Aer noise model derived from *this* backend's calibrations.

    If `sim` is provided, it is reused (recommended for speed).
    """
    if sim is None:
        noise_model = NoiseModel.from_backend(backend)
        sim = AerSimulator(noise_model=noise_model)

    E = {}
    for (A, B), tqc in transpiled_by_setting.items():
        counts = sim.run(tqc, shots=shots).result().get_counts()
        EAB = noiseless.exp_values_from_counts(counts, shots)
        E[(A, B)] = EAB

    return -E[(1, 1)] + E[(1, 2)] - E[(2, 1)] - E[(2, 2)] - 2

def get_initial_layout_for_circuit(qc: QuantumCircuit, backend, manual_layout=None):
    """Return an initial_layout list.

    STRICT mode: a manual layout MUST be provided and MUST be valid.
    If anything is wrong (missing size, wrong length, invalid indices, duplicates),
    we raise ValueError to abort the run instead of silently falling back to auto layout.
    """
    if manual_layout is None:
        raise ValueError(
            "MANUAL_LAYOUTS_BY_SIZE is None, but strict manual layout is required. "
            "Set MANUAL_LAYOUTS_BY_SIZE to a dict/list with valid physical qubit indices."
        )

    # Dict-by-size mode
    if isinstance(manual_layout, dict):
        if qc.num_qubits not in manual_layout:
            raise ValueError(
                f"No manual layout provided for circuit size n={qc.num_qubits}. "
                f"Available sizes: {sorted(manual_layout.keys())}."
            )
        layout = list(manual_layout[qc.num_qubits])
    else:
        # Single list mode
        layout = list(manual_layout)

    if len(layout) != qc.num_qubits:
        raise ValueError(
            f"Manual layout length {len(layout)} does not match circuit num_qubits={qc.num_qubits}."
        )

    bad = [q for q in layout if not (0 <= int(q) < backend.num_qubits)]
    if bad:
        raise ValueError(
            f"Manual layout contains invalid physical qubit indices {bad}. "
            f"Backend has num_qubits={backend.num_qubits}."
        )

    if len(set(layout)) != len(layout):
        raise ValueError("Manual layout contains duplicates (a physical qubit used twice).")

    return layout


def plot_circuit(title: str, qc: QuantumCircuit, file_stem: str):
    """Plot a circuit depending on PLOT_MODE.

    - PLOT_MODE = "none": do nothing
    - PLOT_MODE = "show": show a window
    - PLOT_MODE = "save": save to PLOT_DIR/<file_stem>.png
    """
    if PLOT_MODE == "none":
        return

    fig = qc.draw("mpl", fold=-1, idle_wires=False)
    fig.suptitle(title, fontsize=12)

    if PLOT_MODE == "save":
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOT_DIR / f"{file_stem}.png", dpi=250, bbox_inches="tight")

    if PLOT_MODE == "show":
        plt.show()

    plt.close(fig)


def transpile_agent_circuits(agent_name: str, build_fn, alpha: float, beta1: float, beta2: float, backend):
    """Build and transpile the 4 SB-setting circuits for one agent.

    Returns: dict[(A,B)] -> transpiled circuit.
    """

    print(f"\nAgent: {agent_name}")

    out = {}
    for label, A, B in SETTINGS:
        qc = build_fn(A, B, alpha, beta1, beta2)
        initial_layout = get_initial_layout_for_circuit(qc, backend, MANUAL_LAYOUTS_BY_SIZE)

        tqc = transpile(
            qc,
            backend=backend,
            optimization_level=OPT_LEVEL,
            initial_layout=initial_layout,
        )

        # Optional TVD check (ideal simulator)
        do_check = DO_TVD_CHECK if (DO_REAL_BACKEND_NOISE_SIM is False and DO_REAL_HARDWARE_RUN is False) else DO_TVD_CHECK_ONLINE
        tvd = tvd_original_vs_transpiled(qc, tqc, shots=TVD_SHOTS) if do_check else None

        ops = dict(tqc.count_ops())
        tvd_str = f"tvd={tvd:.4f}" if tvd is not None else "tvd=NA"

        cz_n = ops.get("cz", 0)
        twoq_n = cz_n

        print(f"  {label}: {tvd_str}  depth={tqc.depth()}  2q={twoq_n}  (cz={cz_n})")

        # Optional plots
        plot_circuit(f"{agent_name} — Original {label}", qc, f"{agent_name.replace(' ', '_')}_{label}_original")
        plot_circuit(
            f"{agent_name} — Transpiled {label} (opt={OPT_LEVEL})",
            tqc,
            f"{agent_name.replace(' ', '_')}_{label}_transpiled_opt{OPT_LEVEL}",
        )

        out[(A, B)] = tqc

    return out


def get_counts_from_sampler_result(pub_res):
    """Return a counts dict from a SamplerV2 result item.

    This keeps the main hardware loop easy to read.
    """
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


def run_all_agents(alpha: float, beta1: float, beta2: float):
    # --------------------------
    # IBM Quantum Platform backends
    # --------------------------
    if DO_REAL_BACKEND_NOISE_SIM or DO_REAL_HARDWARE_RUN:
        print("\n=== IBM Quantum Platform backend ===")
        service = get_runtime_service()
        for backend_name, enabled in REAL_BACKENDS.items():
            if not enabled:
                continue
            backend_real = get_real_backend(service, backend_name)
            print(f"\nUsing backend: {backend_real.name}")

            if DO_REAL_BACKEND_NOISE_SIM:
                print("--- Calibrated noise simulation (Aer NoiseModel from live calibrations) ---")
                for agent_name, build_fn in AGENTS:
                    transpiled = transpile_agent_circuits(agent_name, build_fn, alpha, beta1, beta2, backend_real)
                    S_val = simulate_with_backend_noise(transpiled, backend_real, shots=NOISE_SHOTS)
                    verdict = "VIOLATION" if S_val > 0 else "no violation"
                    print(f"  -> {agent_name}: S_SB ≈ {S_val:.3f} ({verdict})")

            if DO_REAL_HARDWARE_RUN:
                print("\n--- Real hardware run (SamplerV2, single job) ---")

                sampler = Sampler(mode=backend_real)

                # Collect ALL circuits across all agents
                all_circuits = []
                meta_info = []  # (agent_name, A, B)

                for agent_name, build_fn in AGENTS:
                    transpiled = transpile_agent_circuits(
                        agent_name, build_fn, alpha, beta1, beta2, backend_real
                    )

                    for (A, B), tqc in transpiled.items():
                        all_circuits.append(tqc)
                        meta_info.append((agent_name, A, B))

                # Submit ONE job with all 12 circuits
                job = sampler.run(all_circuits, shots=HARDWARE_SHOTS)
                results = job.result()

                # Prepare result storage
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_dir = Path(f"hardware_results/{backend_real.name}_{timestamp}")
                results_dir.mkdir(parents=True, exist_ok=True)

                # Save basic job information (minimal reproducibility)
                try:
                    job_id = job.job_id()
                except Exception:
                    job_id = None

                with open(results_dir / "job_info.json", "w") as f:
                    json.dump(
                        {
                            "backend": backend_real.name,
                            "job_id": job_id,
                            "shots": HARDWARE_SHOTS,
                            "timestamp": timestamp,
                            # Calibration snapshot timestamp reported by the backend
                            "calibration_last_update": getattr(getattr(backend_real, "properties", lambda: None)(), "last_update_date", None),
                        },
                        f,
                        indent=2,
                    )

                # Save backend properties (calibrations, gate errors, T1, T2, etc.)
                try:
                    props = backend_real.properties()
                    def _json_dump(path, data):
                        with open(path, "w") as f:
                            json.dump(data, f, indent=2)
                    _json_dump(results_dir / "backend_properties.json", props.to_dict() if hasattr(props, "to_dict") else props)

                    # Save backend configuration (connectivity, basis gates, etc.) for the same run
                    try:
                        cfg = backend_real.configuration()
                        _json_dump(
                            results_dir / "backend_configuration.json",
                            cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
                        )
                    except Exception:
                        pass

                except Exception:
                    pass

                # Organize results per agent
                agent_E = {}
                agent_counts = {}

                for (agent_name, A, B), pub_res in zip(meta_info, results):
                    counts = get_counts_from_sampler_result(pub_res)

                    # Save raw counts immediately
                    fname = results_dir / f"{agent_name.replace(' ', '_')}_A{A}B{B}.json"
                    with open(fname, "w") as f:
                        json.dump(counts, f, indent=2)

                    # Compute expectation value
                    EAB = noiseless.exp_values_from_counts(counts, HARDWARE_SHOTS)

                    if agent_name not in agent_E:
                        agent_E[agent_name] = {}
                        agent_counts[agent_name] = {}

                    agent_E[agent_name][(A, B)] = EAB
                    agent_counts[agent_name][(A, B)] = counts

                # Save processed expectation values and S values
                processed = {}
                for agent_name in agent_E:
                    E = agent_E[agent_name]
                    S_val = -E[(1, 1)] + E[(1, 2)] - E[(2, 1)] - E[(2, 2)] - 2
                    processed[agent_name] = {
                        "E_A1B1": float(E[(1, 1)]),
                        "E_A1B2": float(E[(1, 2)]),
                        "E_A2B1": float(E[(2, 1)]),
                        "E_A2B2": float(E[(2, 2)]),
                        "S_SB": float(S_val),
                        "shots": HARDWARE_SHOTS,
                    }

                with open(results_dir / "processed_results.json", "w") as f:
                    json.dump(processed, f, indent=2)

                # Compute S per agent
                for agent_name in agent_E:
                    E = agent_E[agent_name]
                    S_val = -E[(1, 1)] + E[(1, 2)] - E[(2, 1)] - E[(2, 2)] - 2
                    verdict = "VIOLATION" if S_val > 0 else "no violation"
                    print(f"  -> {agent_name}: S_SB ≈ {S_val:.3f} ({verdict})")

                print(f"\nRaw hardware data saved in: {results_dir.resolve()}")


if __name__ == "__main__":
    alpha, beta1, beta2 = noiseless.analytic_optimal_angles()
    run_all_agents(alpha, beta1, beta2)
