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

from ewfs.agents.agents import (
    build_circuit_reflex,
    build_circuit_guessing,
    build_circuit_betting,
)

warnings.filterwarnings(
    "ignore",
    message="Trying to add QuantumRegister to a QuantumCircuit having a layout",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLOT_DIR = PROJECT_ROOT / "results" / "plots_ibm_transpilation"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = [
    ("Reflex Agent", build_circuit_reflex),
    ("Guessing Agent", build_circuit_guessing),
    ("Betting Agent", build_circuit_betting),
]

MANUAL_LAYOUTS_BY_SIZE = {
    6: [28, 29, 30, 31, 14, 129],          # Reflex Agent
    7: [54,61,62,60,63, 14, 129],      # Guessing Agent
    8: [54, 61, 62, 60, 63, 59, 14, 129],  # Betting Agent
}

OPT_LEVEL = 0


def transpile_agent_circuit(agent_name, build_fn, backend, save_plots=True):
    """Build and transpile one circuit for one agent."""
    print(f"  {agent_name}: transpiling")

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
    print(f"    depth={tqc.depth()}, cz={cz_n}")

    if save_plots:
        tqc.name = agent_name
        agent_dir = PLOT_DIR / "transpiled" / agent_name.replace(" ", "_")
        agent_dir.mkdir(parents=True, exist_ok=True)
        plot_path = agent_dir / f"circuit_depth{tqc.depth()}_cz{cz_n}.png"

        fig = circuit_drawer(tqc, output="mpl", fold=-1)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return tqc


def transpile_all_agents(backend, save_plots=True):
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
        )
    return out


if __name__ == "__main__":
    service = QiskitRuntimeService()
    backend = service.backend("ibm_torino")
    transpile_all_agents(backend)