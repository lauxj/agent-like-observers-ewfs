"""
ibm_transpilation.py
Transpile all circuits for one IBM backend
Set Manual qubit layout here
"""

import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.visualization import circuit_drawer
from qiskit_ibm_runtime import QiskitRuntimeService

try:
    from ewfs.agents import (
        build_circuit_reflex,
        build_circuit_guessing,
        build_circuit_betting,
        build_circuit_always_large,
    )
except ModuleNotFoundError:
    from agents import (
        build_circuit_reflex,
        build_circuit_guessing,
        build_circuit_betting,
        build_circuit_always_large,
    )

warnings.filterwarnings(
    "ignore",
    message="Trying to add QuantumRegister to a QuantumCircuit having a layout",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLOT_DIR = PROJECT_ROOT / "results" / "plots" / "plots_ibm_transpilation"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BACKEND_NAME = "ibm_torino"

AGENTS = [
    ("Reflex Agent", build_circuit_reflex),
    ("Guessing Agent", build_circuit_guessing),
    ("Betting Agent", build_circuit_betting),
    ("Always 3/4 Agent", build_circuit_always_large),
]

MANUAL_LAYOUTS_BY_BACKEND = {
    "ibm_torino": {
        # 6: [28, 29, 30, 31, 14, 129],  # Reflex Agent
        6: [60, 61, 62, 63, 14, 129],  # Reflex Agent
        7: [54, 61, 62, 60, 63, 14, 129],  # Guessing Agent
        8: [54, 61, 62, 60, 63, 59, 14, 129],  # Betting Agent / Always 3/4 Agent
    },
    "ibm_kingston": {
        6: [37, 45, 46, 47, 10, 142],  # Reflex Agent
        7: [37, 45, 46, 44, 47, 10, 142],  # Guessing Agent
        8: [37, 45, 46, 44, 47, 43, 10, 142],  # Betting Agent / Always 3/4 Agent
    },
    "ibm_fez": {
        # Fill these physical qubit indices manually before running on Fez.
        6: None,  # Reflex Agent
        7: None,  # Guessing Agent
        8: None,  # Betting Agent / Always 3/4 Agent
    },
    "ibm_marrakesh": {
        # Fill these physical qubit indices manually before running on Marrakesh.
        6: None,  # Reflex Agent
        7: None,  # Guessing Agent
        8: None,  # Betting Agent / Always 3/4 Agent
    },
}

OPT_LEVEL = 0


def safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_")


def get_manual_layout(backend_name, circuit_qubit_count):
    """Return the manual layout for one backend and circuit size."""
    try:
        layouts_by_size = MANUAL_LAYOUTS_BY_BACKEND[backend_name]
    except KeyError as exc:
        known_backends = ", ".join(sorted(MANUAL_LAYOUTS_BY_BACKEND))
        raise ValueError(
            f"No manual-layout configuration defined for backend '{backend_name}'. "
            f"Known backends: {known_backends}."
        ) from exc

    try:
        initial_layout = layouts_by_size[circuit_qubit_count]
    except KeyError as exc:
        raise ValueError(
            f"No manual layout defined for backend '{backend_name}' "
            f"and n={circuit_qubit_count} qubits."
        ) from exc

    if initial_layout is None:
        raise ValueError(
            f"Manual layout for backend '{backend_name}' and n={circuit_qubit_count} "
            "qubits is not filled in yet."
        )

    if len(initial_layout) != circuit_qubit_count:
        raise ValueError(
            f"Manual layout for backend '{backend_name}' and n={circuit_qubit_count} "
            f"must contain exactly {circuit_qubit_count} physical qubits."
        )

    return initial_layout


def transpile_agent_circuit(agent_name, build_fn, backend, save_plots=True, plots_dir=None):
    """Build and transpile one circuit for one agent."""
    print(f"  {agent_name}: transpiling")

    qc = build_fn()
    initial_layout = get_manual_layout(backend.name, qc.num_qubits)

    tqc = transpile(
        qc,
        backend=backend,
        optimization_level=OPT_LEVEL,
        initial_layout=initial_layout,
    )

    ops = dict(tqc.count_ops())
    cz_n = ops.get("cz", 0)
    print(f"    depth={tqc.depth()}, cz={cz_n}")

    if save_plots:
        tqc.name = agent_name
        base_plot_dir = plots_dir if plots_dir is not None else PLOT_DIR
        safe_name = safe_label(agent_name)
        agent_dir = base_plot_dir / safe_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{safe_name}_circuit_depth{tqc.depth()}_cz{cz_n}.png"
        plot_path = agent_dir / filename

        fig = circuit_drawer(tqc, output="mpl", fold=-1)
        fig.suptitle(f"{agent_name} – Transpiled Circuit", fontsize=14)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return tqc


def transpile_all_agents(backend, save_plots=True, plots_dir=None):
    """Transpile all agent circuits for one backend."""
    print("\n=== Transpilation ===")
    print(f"Backend: {backend.name}")
    out = {}
    for agent_name, build_fn in AGENTS:
        out[agent_name] = transpile_agent_circuit(
            agent_name=agent_name,
            build_fn=build_fn,
            backend=backend,
            save_plots=save_plots,
            plots_dir=plots_dir,
        )
    return out


if __name__ == "__main__":
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    transpile_all_agents(backend)
