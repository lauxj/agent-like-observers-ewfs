import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import reflex_agent
import guessing_agent
import betting_agent
import noiseless_simulation as noiseless

# Offline IBM hardware snapshot (no IBM account needed)
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

# Silence a common Qiskit warning that does not affect correctness
warnings.filterwarnings(
    "ignore",
    message="Trying to add QuantumRegister to a QuantumCircuit having a layout",
)

# ==========================
# Settings (edit these)
# ==========================

# Manual qubit placement: logical qubit i -> physical qubit MANUAL_LAYOUTS_BY_SIZE[n][i]
# Use this to force the circuit into a good region of the device.
# Set MANUAL_LAYOUTS_BY_SIZE = None to let Qiskit choose automatically.
MANUAL_LAYOUTS_BY_SIZE = {
    4: [59, 60, 61, 62],           # Reflex Agent (4 qubits)
    5: [62, 63, 64, 65, 54],       # Guessing Agent (5 qubits)
    6: [54, 64, 65, 63, 66, 62],   # Betting Agent (6 qubits)
}


# Transpiler settings
OPT_LEVEL = 0

# Verification: compare original vs transpiled distributions (ideal simulator)
DO_TVD_CHECK = True
TVD_SHOTS = 10_000

# Building NoiseModel.from_backend(...) can be slow; cache one simulator per backend.
CACHE_NOISE_MODEL = True

NOISE_SHOTS = 10_000

# ==========================
# Real hardware (IBM Quantum Platform)
# ==========================


# Plotting (can be heavy). Choose: "none", "show", or "save"
PLOT_MODE = "none"
PLOT_DIR = Path("../plots")


def _normalize_counts(counts: dict, shots: int) -> dict[str, float]:
    return {k: v / shots for k, v in counts.items()}


def _tvd(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def tvd_original_vs_transpiled(qc: QuantumCircuit, tqc: QuantumCircuit, shots: int) -> float:
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


def simulate_with_backend_noise(transpiled_by_setting: dict, backend, shots: int, sim: AerSimulator | None = None) -> float:
    """Compute S_SB using an Aer noise model derived from *this* backend's calibrations.

    If `sim` is provided, it is reused (recommended for speed).
    """
    if sim is None:
        noise_model = NoiseModel.from_backend(backend)
        sim = AerSimulator(noise_model=noise_model)

    E = {}
    for (A, B), tqc in transpiled_by_setting.items():
        counts = sim.run(tqc, shots=shots).result().get_counts()
        _, _, EAB = noiseless.exp_values_from_counts(counts, shots)
        E[(A, B)] = EAB

    return -E[(1, 1)] + E[(1, 2)] - E[(2, 1)] - E[(2, 2)] - 2


def get_initial_layout_for_circuit(qc: QuantumCircuit, backend, manual_layout=None):
    """Return an initial_layout list or None.

    manual_layout can be:
      - None: auto layout
      - list[int]: use this for ALL circuits (must match qc.num_qubits)
      - dict[int, list[int]]: pick by qc.num_qubits

    If a manual layout is not available for this circuit size, we fall back to auto layout.
    """
    if manual_layout is None:
        return None

    # Dict-by-size mode
    if isinstance(manual_layout, dict):
        if qc.num_qubits not in manual_layout:
            return None
        layout = list(manual_layout[qc.num_qubits])
    else:
        # Single list mode
        layout = list(manual_layout)

    if len(layout) != qc.num_qubits:
        return None

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
    settings = [
        ("A1B1", 1, 1),
        ("A1B2", 1, 2),
        ("A2B1", 2, 1),
        ("A2B2", 2, 2),
    ]

    print(f"\nAgent: {agent_name}")

    out = {}
    for label, A, B in settings:
        qc = build_fn(A, B, alpha, beta1, beta2)
        initial_layout = get_initial_layout_for_circuit(qc, backend, MANUAL_LAYOUTS_BY_SIZE)

        tqc = transpile(
            qc,
            backend=backend,
            optimization_level=OPT_LEVEL,
            initial_layout=initial_layout,
        )

        # Optional TVD check (ideal simulator)
        tvd = tvd_original_vs_transpiled(qc, tqc, shots=TVD_SHOTS) if DO_TVD_CHECK else None

        ops = dict(tqc.count_ops())
        tvd_str = f"tvd={tvd:.4f}" if tvd is not None else "tvd=NA"

        cx_n = ops.get("cx", 0)
        ecr_n = ops.get("ecr", 0)
        cz_n = ops.get("cz", 0)
        twoq_n = cx_n + ecr_n + cz_n

        print(f"  {label}: {tvd_str}  depth={tqc.depth()}  2q={twoq_n}  (cx={cx_n}, ecr={ecr_n}, cz={cz_n})")

        # Optional plots
        plot_circuit(f"{agent_name} — Original {label}", qc, f"{agent_name.replace(' ', '_')}_{label}_original")
        plot_circuit(
            f"{agent_name} — Transpiled {label} (opt={OPT_LEVEL})",
            tqc,
            f"{agent_name.replace(' ', '_')}_{label}_transpiled_opt{OPT_LEVEL}",
        )

        out[(A, B)] = tqc

    return out


def run_all_agents(alpha: float, beta1: float, beta2: float):
    # --------------------------
    # 1) Offline: FakeBrisbane
    # --------------------------
    backend_fake = FakeBrisbane()

    if DO_TVD_CHECK:
        print("\nTVD = Total Variation Distance between output distributions")
        print("      (original vs transpiled, ideal simulator). TVD ≈ 0 means they match.\n")

    sim_fake = None
    if CACHE_NOISE_MODEL:
        sim_fake = AerSimulator(noise_model=NoiseModel.from_backend(backend_fake))

    agents = [
        ("Reflex Agent", reflex_agent.build_measurement),
        ("Guessing Agent", guessing_agent.build_measurement),
        ("Betting Agent", betting_agent.build_measurement),
    ]

    print("\n=== Offline noise simulation (FakeBrisbane) ===")
    for agent_name, build_fn in agents:
        transpiled = transpile_agent_circuits(agent_name, build_fn, alpha, beta1, beta2, backend_fake)
        S_val = simulate_with_backend_noise(transpiled, backend_fake, shots=NOISE_SHOTS, sim=sim_fake)
        verdict = "VIOLATION" if S_val > 0 else "no violation"
        print(f"  -> {agent_name}: S_SB ≈ {S_val:.3f} ({verdict})")


if __name__ == "__main__":
    alpha, beta1, beta2 = noiseless.analytic_optimal_angles()
    run_all_agents(alpha, beta1, beta2)
