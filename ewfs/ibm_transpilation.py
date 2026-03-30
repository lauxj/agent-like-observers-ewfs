"""
ibm_transpilation.py
Transpile all circuits for one IBM backend
Automatically choose the best qubit layout when calibration CSV data is available.
"""

import warnings
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.visualization import circuit_drawer
from qiskit_ibm_runtime import QiskitRuntimeService

try:
    from ewfs.agents import AGENTS
    from ewfs.find_best_agent_layouts import (
        DEFAULT_CHOICE_DISTANCE_WEIGHT,
        DEFAULT_COHERENCE_WEIGHT,
        DEFAULT_CZ_WEIGHT,
        DEFAULT_M_PRIORITY_FACTOR,
        DEFAULT_READOUT_WEIGHT,
        find_optimal_layout_for_circuit,
    )
except ModuleNotFoundError:
    from agents import AGENTS
    from find_best_agent_layouts import (
        DEFAULT_CHOICE_DISTANCE_WEIGHT,
        DEFAULT_COHERENCE_WEIGHT,
        DEFAULT_CZ_WEIGHT,
        DEFAULT_M_PRIORITY_FACTOR,
        DEFAULT_READOUT_WEIGHT,
        find_optimal_layout_for_circuit,
    )

warnings.filterwarnings(
    "ignore",
    message="Trying to add QuantumRegister to a QuantumCircuit having a layout",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLOT_DIR = PROJECT_ROOT / "results" / "plots" / "plots_ibm_transpilation"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BACKEND_NAME = "ibm_torino"
USE_AUTO_LAYOUT = True
AUTO_LAYOUT_READOUT_WEIGHT = DEFAULT_READOUT_WEIGHT
AUTO_LAYOUT_CZ_WEIGHT = DEFAULT_CZ_WEIGHT
AUTO_LAYOUT_COHERENCE_WEIGHT = DEFAULT_COHERENCE_WEIGHT
AUTO_LAYOUT_M_PRIORITY_FACTOR = DEFAULT_M_PRIORITY_FACTOR
AUTO_LAYOUT_CHOICE_DISTANCE_WEIGHT = DEFAULT_CHOICE_DISTANCE_WEIGHT

MANUAL_LAYOUTS_BY_BACKEND = {
    "ibm_torino": {
        # 6: [28, 29, 30, 31, 14, 129],  # Reflex Agent
        #6: [11,12,13,14, 1, 129],  # Reflex Agent
        #6: [60, 61, 62, 63, 14, 129],  # Reflex Agent
        6: [54, 61, 62, 63, 14, 129],  # Reflex Agent
        7: [54, 61, 62, 60, 63, 14, 129],  # Guessing Agent
        8: [54, 61, 62, 60, 63, 59, 14, 129],  # Betting Agent / Always 3/4 Agent
    },
    "ibm_kingston": {
        6: [37, 45, 46, 47, 10, 142],  # Reflex Agent
        7: [37, 45, 46, 44, 47, 10, 142],  # Guessing Agent
        8: [38,49,50,48,51,47, 10, 142],  # Betting Agent / Always 3/4 Agent
    },
    "ibm_fez": {
        # Fill these physical qubit indices manually before running on Fez.
        6: [37, 45, 46, 47, 10, 142],  # Reflex Agent
        7: [37, 45, 46, 44, 47, 10, 142],  # Guessing Agent
        8: [37, 45, 46, 44, 47, 43, 10, 142],  # Betting Agent / Always 3/4 Agent
    },
    "ibm_marrakesh": {
        # Fill these physical qubit indices manually before running on Marrakesh.
        6: [37, 45, 46, 47, 10, 142],  # Reflex Agent
        7: [37, 45, 46, 44, 47, 10, 142],  # Guessing Agent
        8: [37, 45, 46, 44, 47, 43, 10, 142],  # Betting Agent / Always 3/4 Agent
    },
}

OPT_LEVEL = 0
ACCURACY_TEST_INFIX = "_accuracy_test_"
SHARED_LAYOUT_REFERENCE = {
    "Always 3/4 Agent": "Betting Agent",
}


def safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_")


def make_run_folder_name(backend, folder_ts=None):
    """Create the shared run-folder name used for transpilation plot output."""
    if folder_ts is None:
        folder_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return folder_ts, f"{backend.name}_{folder_ts}"


def resolve_plots_dir(backend, plots_dir=None, folder_ts=None, category="transpilation_only"):
    """Resolve a run-specific transpilation plot directory."""
    if plots_dir is not None:
        return Path(plots_dir)

    _, run_folder_name = make_run_folder_name(backend, folder_ts)
    return PLOT_DIR / category / run_folder_name


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


def resolve_layout_reference(agent_name, qc):
    """Use the parent agent circuit as the layout reference for accuracy-test circuits."""
    layout_agent_name = agent_name
    layout_qc = qc

    if ACCURACY_TEST_INFIX in agent_name:
        parent_agent_name = agent_name.split(ACCURACY_TEST_INFIX, 1)[0]
        for known_agent_name, build_fn in AGENTS:
            if known_agent_name == parent_agent_name:
                layout_agent_name = parent_agent_name
                layout_qc = build_fn()
                break

    shared_layout_agent_name = SHARED_LAYOUT_REFERENCE.get(layout_agent_name)
    if shared_layout_agent_name is not None:
        for known_agent_name, build_fn in AGENTS:
            if known_agent_name == shared_layout_agent_name:
                return shared_layout_agent_name, build_fn()

    return layout_agent_name, layout_qc


def get_initial_layout(agent_name, qc, backend):
    """Return the best known initial layout for one circuit on one backend."""
    layout_agent_name, layout_qc = resolve_layout_reference(agent_name, qc)

    if USE_AUTO_LAYOUT:
        try:
            return find_optimal_layout_for_circuit(
                agent_name=layout_agent_name,
                circuit=layout_qc,
                backend_name=backend.name,
                readout_weight=AUTO_LAYOUT_READOUT_WEIGHT,
                cz_weight=AUTO_LAYOUT_CZ_WEIGHT,
                coherence_weight=AUTO_LAYOUT_COHERENCE_WEIGHT,
                m_priority_factor=AUTO_LAYOUT_M_PRIORITY_FACTOR,
                choice_distance_weight=AUTO_LAYOUT_CHOICE_DISTANCE_WEIGHT,
            )
        except ValueError as exc:
            if "No calibration CSV found for backend" not in str(exc):
                raise
            print(f"    auto-layout unavailable ({exc}); falling back to manual layout")

    return get_manual_layout(backend.name, qc.num_qubits)


def transpile_agent_circuit(agent_name, build_fn, backend, save_plots=True, plots_dir=None):
    """Build and transpile one circuit for one agent."""
    print(f"  {agent_name}: transpiling")

    qc = build_fn()
    initial_layout = get_initial_layout(agent_name=agent_name, qc=qc, backend=backend)
    print(f"    initial_layout={initial_layout}")

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


def transpile_all_agents(
    backend,
    save_plots=True,
    plots_dir=None,
    folder_ts=None,
    plot_category="transpilation_only",
    agent_builders=None,
):
    """Transpile all agent circuits for one backend."""
    print("\n=== Transpilation ===")
    print(f"Backend: {backend.name}")
    resolved_plots_dir = None
    if save_plots:
        resolved_plots_dir = resolve_plots_dir(
            backend,
            plots_dir=plots_dir,
            folder_ts=folder_ts,
            category=plot_category,
        )
    out = {}
    selected_agents = list(agent_builders) if agent_builders is not None else AGENTS
    for agent_name, build_fn in selected_agents:
        out[agent_name] = transpile_agent_circuit(
            agent_name=agent_name,
            build_fn=build_fn,
            backend=backend,
            save_plots=save_plots,
            plots_dir=resolved_plots_dir,
        )
    return out


if __name__ == "__main__":
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    transpile_all_agents(backend)
