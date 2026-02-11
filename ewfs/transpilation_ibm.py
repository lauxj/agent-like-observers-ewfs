import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import reflex_agent
import guessing_agent
import betting_agent


# Offline IBM hardware snapshots (Fake backends)
# Requires: pip install qiskit-ibm-runtime
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

# ==========================
#   Manual initial_layout (optional; supports multiple circuit sizes)
# ==========================
# Qiskit maps logical qubit i -> physical qubit initial_layout[i].
#
# In this project, different agents may produce circuits with DIFFERENT numbers of qubits.
# Therefore we support:
#   - MANUAL_LAYOUTS_BY_SIZE: dict mapping {num_qubits: [physical indices...]}
#   - or set MANUAL_LAYOUTS_BY_SIZE = None to let Qiskit choose automatically.
#
# Example (edit to your chosen connected regions):
MANUAL_LAYOUTS_BY_SIZE = {
    # Reflex Agent (4 qubits):
    4: [59, 60, 61, 62],
    # Guessing Agent (5 qubits):
    #5: [54, 64, 65, 63, 66],
    5: [62, 63, 64, 65, 54],
    # Betting Agent (6 qubits):
    6: [54, 64, 65, 63, 66, 62],
}
# Set to None to disable all manual layouts (auto layout for everything)
# MANUAL_LAYOUTS_BY_SIZE = None

# ==========================
#   Verification (recommended)
# ==========================
# If True, compare original vs transpiled circuits on an IDEAL simulator
# to ensure transpilation preserved the logic (up to global phase).
VERIFY_EQUIVALENCE = True
VERIFY_SHOTS = 20_000

# Verification helpers
def _normalize_counts(counts: dict, shots: int) -> dict[str, float]:
    """Convert counts to probabilities."""
    return {k: v / shots for k, v in counts.items()}


def _tvd(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance between two discrete distributions."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def verify_transpile_equivalence(
    qc: QuantumCircuit,
    tqc: QuantumCircuit,
    *,
    shots: int = VERIFY_SHOTS,
    label: str = "",
) -> None:
    """Check that qc and tqc implement the same logic.

    1) Compare measurement distributions on an *ideal* simulator.
    2) Try a stronger check: compare statevectors (after removing measurements) up to global phase.

    Notes:
    - This catches classical-bit ordering surprises and accidental semantic changes.
    - For the distribution check we use shot sampling; expect small nonzero TVD.
    """
    print("\n[check] Verifying equivalence" + (f" for {label}" if label else "") + "...")

    # --- (A) Ideal measurement distribution check ---
    ideal_sim = AerSimulator()  # no noise

    res1 = ideal_sim.run(qc, shots=shots).result()
    res2 = ideal_sim.run(tqc, shots=shots).result()

    c1 = res1.get_counts()
    c2 = res2.get_counts()

    p1 = _normalize_counts(c1, shots)
    p2 = _normalize_counts(c2, shots)
    tvd = _tvd(p1, p2)

    # Heuristic threshold: sampling noise ~ O(1/sqrt(shots)); for shots=20k TVD should be small.
    # If it's large (e.g., > 0.02), likely a genuine mismatch or bit-order issue.
    print(f"[check] Ideal-sim TVD(counts) = {tvd:.4f} (shots={shots})")

    # --- (B) Statevector check (strong, when available) ---
    try:
        qc_u = qc.remove_final_measurements(inplace=False)
        tqc_u = tqc.remove_final_measurements(inplace=False)

        psi = Statevector.from_instruction(qc_u)
        phi = Statevector.from_instruction(tqc_u)

        # Fidelity is invariant under global phase.
        fid = float(abs((psi.data.conj() @ phi.data)) ** 2)
        print(f"[check] Statevector fidelity = {fid:.12f}")
    except Exception as exc:
        print(f"[check] Statevector check skipped ({exc}).")

    # Quick verdict
    if tvd > 0.02:
        print("[check][WARN] TVD is large. This can indicate a mismatch or classical bit-order differences.")
    else:
        print("[check] OK (within sampling error).")


def get_transpile_backend(preferred_name: str = "ibm_brisbane"):
    """Return an offline Fake backend for transpilation.

    This avoids any IBM token/runtime setup. We always use FakeBrisbane.
    The `preferred_name` argument is ignored (kept only so other code doesn't need changing).
    """
    return FakeBrisbane()


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
            print(
                f"[info] No manual layout provided for num_qubits={qc.num_qubits}; "
                "falling back to auto layout."
            )
            return None
        layout = list(manual_layout[qc.num_qubits])
    else:
        # Single list mode
        layout = list(manual_layout)

    if len(layout) != qc.num_qubits:
        print(
            f"[warn] Manual layout length {len(layout)} does not match circuit num_qubits {qc.num_qubits}; "
            "falling back to auto layout."
        )
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


def transpile_and_summarize(qc: QuantumCircuit, backend, optimization_level: int = 0, initial_layout=None):
    """
    Transpile the circuit for a specific backend and print a short summary
    (mapping/layout, depth, and gate counts). Returns the transpiled circuit.
    """
    tqc = transpile(
        qc,
        backend=backend,
        optimization_level=optimization_level,
        initial_layout=initial_layout,
    )
    print("\n=== Transpilation Summary ===")
    print(f"Backend: {getattr(backend, 'name', lambda: str(backend))() if callable(getattr(backend, 'name', None)) else str(backend)}")
    print(f"Depth: {tqc.depth()}")
    print(f"Gate counts: {tqc.count_ops()}")
    return tqc



def draw_circuit(title: str, qc: QuantumCircuit, show: bool = True, save_path: str | None = None):
    """Consistent plotting defaults and ALWAYS free memory.

    PyCharm often reuses the same Python process between runs; if figures are not closed,
    memory accumulates and you can hit an out-of-memory error on the second run.
    """
    fig = qc.draw("mpl", fold=-1, idle_wires=False)
    fig.suptitle(title, fontsize=14)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    # Always close to free memory (crucial when drawing many circuits)
    plt.close(fig)


def transpile_agent_circuits(
    agent_name: str,
    build_fn,
    alpha: float,
    beta1: float,
    beta2: float,
    backend,
    optimization_level: int = 0,
    show_plots: bool = True,
    manual_layout=None,
):
    """Transpile the four SB setting circuits for one agent and print summaries."""

    settings = [
        ("A1B1", 1, 1),
        ("A1B2", 1, 2),
        ("A2B1", 2, 1),
        ("A2B2", 2, 2),
    ]

    print(f"\n==============================")
    print(f"Agent: {agent_name}")
    print(f"Backend: {getattr(backend, 'name', lambda: str(backend))() if callable(getattr(backend, 'name', None)) else str(backend)}")
    print(f"Optimization level: {optimization_level}")

    for setting_label, A, B in settings:
        qc = build_fn(A, B, alpha, beta1, beta2)

        initial_layout = get_initial_layout_for_circuit(qc, backend, manual_layout)

        # Original (logical) circuit
        if show_plots:
            draw_circuit(f"{agent_name} — Original {setting_label}", qc, show=True)

        # Backend transpilation (routing + native decomposition)
        tqc = transpile(
            qc,
            backend=backend,
            optimization_level=optimization_level,
            initial_layout=initial_layout,
        )
        if VERIFY_EQUIVALENCE:
            verify_transpile_equivalence(qc, tqc, shots=VERIFY_SHOTS, label=f"{agent_name} {setting_label}")
        print(f"\n--- {agent_name} {setting_label} (backend transpile) ---")
        print(f"Depth: {tqc.depth()}")
        print(f"Gate counts: {tqc.count_ops()}")
        if show_plots:
            draw_circuit(
                f"{agent_name} — Transpiled ({setting_label}, opt={optimization_level})",
                tqc,
                show=True,
            )



def transpile_all_agents(
    alpha: float,
    beta1: float,
    beta2: float,
    optimization_level: int = 0,
    show_plots: bool = True,
):
    """Transpile SB circuits for all agents (reflex, guessing, betting) on an IBM backend."""

    backend = get_transpile_backend()

    agents = [
        ("Reflex Agent", reflex_agent.build_measurement),
        ("Guessing Agent", guessing_agent.build_measurement),
        ("Betting Agent", betting_agent.build_measurement),
    ]

    for agent_name, build_fn in agents:
        transpile_agent_circuits(
            agent_name=agent_name,
            build_fn=build_fn,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            backend=backend,
            optimization_level=optimization_level,
            show_plots=show_plots,
            manual_layout=MANUAL_LAYOUTS_BY_SIZE,
        )


#
# ==========================
#   Helper functions for Aer noise simulation
# ==========================

def exp_from_counts(counts: dict, shots: int) -> tuple[float, float, float]:
    """Return (⟨A⟩, ⟨B⟩, ⟨AB⟩) from 2-bit counts.

    Convention used across your project:
      - bitstring s like "01"
      - B is s[0], A is s[1]
      - "0" -> +1, "1" -> -1

    IMPORTANT: This must match how your circuits measure into classical bits.
    """
    EA = 0.0
    EB = 0.0
    EAB = 0.0

    for s, c in counts.items():
        B = +1 if s[0] == "0" else -1
        A = +1 if s[1] == "0" else -1
        p = c / shots
        EA += p * A
        EB += p * B
        EAB += p * A * B

    return EA, EB, EAB


def S_SB_with_fake_backend_noise(build_fn, alpha, beta1, beta2, backend, shots: int = 10_000, optimization_level: int = 0, manual_layout=None) -> float:
    """Estimate S_SB using an Aer noise model derived from FakeBrisbane.

    Workflow:
      1) Build the logical circuit.
      2) Transpile to the backend (routing + basis decomposition).
      3) Simulate with AerSimulator using NoiseModel.from_backend(backend).

    This gives a more IBM-like prediction than uniform depolarizing noise.
    """
    noise_model = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=noise_model)

    settings = [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]

    E = {}
    for A_setting, B_setting in settings:
        qc = build_fn(A_setting, B_setting, alpha, beta1, beta2)
        initial_layout = get_initial_layout_for_circuit(qc, backend, manual_layout)
        tqc = transpile(
            qc,
            backend=backend,
            optimization_level=optimization_level,
            initial_layout=initial_layout,
        )
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts()
        _, _, EAB = exp_from_counts(counts, shots)
        E[(A_setting, B_setting)] = EAB

    # Semi-Brukner expression
    return -E[(1, 1)] + E[(1, 2)] - E[(2, 1)] - E[(2, 2)] - 2

if __name__ == "__main__":
    # Angles used across your project (Bell-plus optimal)
    alpha = 3.0 * np.pi / 2.0
    beta1 = 3.0 * np.pi / 4.0
    beta2 = 1.0 * np.pi / 4.0

    # Change backend_name to any IBM backend you have access to
    transpile_all_agents(
        alpha,
        beta1,
        beta2,
        optimization_level=0,
        show_plots=False,
    )

    # ---- Print FakeBrisbane backend properties (noise snapshot) ----
    backend = get_transpile_backend()
    props = backend.properties()

    print("\n==============================")
    print("FakeBrisbane backend properties summary (min/median/max)")
    print("==============================")

    ro_errs = []
    sx_errs = []
    x_errs = []
    ecr_errs = []

    for q in range(backend.num_qubits):
        try:
            ro_errs.append(props.readout_error(q))
        except Exception:
            pass
        for gate, bucket in [("sx", sx_errs), ("x", x_errs)]:
            try:
                bucket.append(props.gate_error(gate, [q]))
            except Exception:
                pass

    for g in props.gates:
        if g.gate == "ecr":
            try:
                ecr_errs.append(g.parameters[0].value)
            except Exception:
                pass

    def _summ(name, arr):
        if not arr:
            print(f"{name}: (no data)")
            return
        arr = np.array(arr, dtype=float)
        print(f"{name}: min={np.min(arr):.4e}, median={np.median(arr):.4e}, max={np.max(arr):.4e} (n={len(arr)})")

    _summ("Readout error", ro_errs)
    _summ("1Q sx error", sx_errs)
    _summ("1Q x error", x_errs)
    _summ("2Q ecr error", ecr_errs)


# ---- Predict S_SB using an IBM-like noise model from FakeBrisbane ----
    print("\n==============================")
    print("S_SB with FakeBrisbane-derived noise model")
    print("==============================")

    agents = [
        ("Reflex Agent", reflex_agent.build_measurement),
        ("Guessing Agent", guessing_agent.build_measurement),
        ("Betting Agent", betting_agent.build_measurement),
    ]

    # Keep shots moderate; this runs 4 circuits per agent.
    shots = 10_000
    for agent_name, build_fn in agents:
        S_val = S_SB_with_fake_backend_noise(
            build_fn,
            alpha,
            beta1,
            beta2,
            backend=backend,
            shots=shots,
            optimization_level=0,
            manual_layout=MANUAL_LAYOUTS_BY_SIZE,
        )
        verdict = "VIOLATION" if S_val > 0 else "no violation"
        print(f"{agent_name}: S_SB ≈ {S_val:.3f}  ->  {verdict}")
