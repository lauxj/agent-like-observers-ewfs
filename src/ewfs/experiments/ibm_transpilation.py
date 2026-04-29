"""
ibm_transpilation.py
Transpile the agent circuits for one IBM backend.
"""

import re
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.visualization import circuit_drawer
from qiskit_ibm_runtime import QiskitRuntimeService
from ewfs.circuits.agents import AGENTS
from ewfs.paths import PROJECT_ROOT

# define directories
PLOT_DIR = PROJECT_ROOT / "results" / "plots" / "plots_ibm_transpilation"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# optimization level must be 0 in order to preserve the circuit exactly as defined
OPT_LEVEL = 0

# plot settings:
TRANSPILED_PLOT_FOLD = 22
TRANSPILED_PLOT_FONTSIZE = 16
TRANSPILED_PLOT_SUBFONTSIZE = 12

MANUAL_LAYOUTS_BY_BACKEND = {
    "ibm_torino": {
        6: [54, 61, 62, 63, 14, 129],  # Reflex Agent
        7: [54, 61, 62, 60, 63, 14, 129],  # Guessing Agent
        8: [54, 61, 62, 60, 63, 59, 14, 129],  # Betting Agent / Always 3/4 Agent
    },
    "ibm_kingston": {
        6: [66, 67, 57, 47, 8, 149],  # Reflex Agent
        7: [66, 67, 57, 68, 47, 8, 149],  # Guessing Agent
        8: [66, 67, 57, 68, 47, 69, 8, 149],  # Betting Agent / Always 3/4 Agent
    },
    "ibm_fez": {
        # Fill these physical qubit indices manually before running on Fez.
        6: [10, 11, 12, 13, 5, 134],  # Reflex Agent
        7: [10, 11, 12, 18, 13, 5, 134],  # Guessing Agent
        8: [18, 11, 12, 10, 13, 9, 5, 134],  # Betting Agent / Always 3/4 Agent
    },
    "ibm_marrakesh": {
        # Fill these physical qubit indices manually before running on Marrakesh.
        6: [10, 11, 12, 13, 0, 155],  # Reflex Agent
        7: [10, 11, 12, 18, 13, 0, 155],  # Guessing Agent
        8: [18, 11, 12, 10, 13, 9, 0, 155],  # Betting Agent / Always 3/4 Agent
    },
}

BACKEND_LAYOUT_ALIASES = {
    "fake_torino": "ibm_torino",
    "fake_fez": "ibm_fez",
    "fake_marrakesh": "ibm_marrakesh",
}

ACCURACY_TEST_INFIX = "_accuracy_test_"
SHARED_LAYOUT_REFERENCE = {
    "Always 3/4 Agent": "Betting Agent",
}


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
    metrics = []
    selected_agents = list(agent_builders) if agent_builders is not None else AGENTS
    for agent_name, build_fn in selected_agents:
        tqc = transpile_agent_circuit(
            agent_name=agent_name,
            build_fn=build_fn,
            backend=backend,
            save_plots=save_plots,
            plots_dir=resolved_plots_dir,
        )
        out[agent_name] = tqc
        metrics.append(tqc.metadata.get("ewfs_metrics", transpiled_circuit_metrics(agent_name, tqc)))

    if save_plots and resolved_plots_dir is not None:
        save_json(
            resolved_plots_dir / "transpiled_circuit_metrics.json",
            {
                "backend": backend.name,
                "optimization_level": int(OPT_LEVEL),
                "layout_source": "manual",
                "agents": metrics,
            },
        )

    return out


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

    tqc.metadata = dict(tqc.metadata or {})
    tqc.metadata["ewfs_metrics"] = transpiled_circuit_metrics(agent_name, tqc)

    if save_plots:
        save_transpiled_plot(qc, tqc, agent_name, plots_dir, cz_n)

    return tqc


def get_initial_layout(agent_name, qc, backend):
    """Return the manual initial layout for one circuit on one backend."""
    layout_qc = get_layout_circuit(agent_name, qc)
    return get_manual_layout(backend.name, layout_qc.num_qubits)


def get_layout_circuit(agent_name, qc):
    """Choose which circuit determines the manual layout."""
    base_agent_name = agent_name
    layout_qc = qc

    if ACCURACY_TEST_INFIX in agent_name:
        base_agent_name = agent_name.split(ACCURACY_TEST_INFIX, 1)[0]
        for known_agent_name, build_fn in AGENTS:
            if known_agent_name == base_agent_name:
                layout_qc = build_fn()
                break

    shared_agent_name = SHARED_LAYOUT_REFERENCE.get(base_agent_name)
    if shared_agent_name is not None:
        for known_agent_name, build_fn in AGENTS:
            if known_agent_name == shared_agent_name:
                return build_fn()

    return layout_qc


def get_manual_layout(backend_name, circuit_qubit_count):
    """Return the manual layout for one backend and circuit size."""
    backend_name = BACKEND_LAYOUT_ALIASES.get(backend_name, backend_name)

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


def transpiled_circuit_metrics(agent_name, tqc):
    """Collect a few useful numbers for the transpiled circuit."""
    ops = dict(tqc.count_ops())
    return {
        "agent_name": agent_name,
        "depth": int(tqc.depth()),
        "cz_count": int(ops.get("cz", 0)),
        "size": int(tqc.size()),
        "num_qubits": int(tqc.num_qubits),
        "num_clbits": int(tqc.num_clbits),
        "operations": {name: int(count) for name, count in ops.items()},
    }


def save_transpiled_plot(original_qc, transpiled_qc, agent_name, plots_dir, cz_count):
    """Save the transpiled circuit as PNG and PDF."""
    plot_qc = transpiled_qc.copy()
    plot_qc.name = agent_name
    plot_qc.global_phase = 0
    base_plot_dir = plots_dir if plots_dir is not None else PLOT_DIR
    safe_name = safe_label(agent_name)
    agent_dir = base_plot_dir / safe_name
    agent_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{safe_name}_circuit_depth{transpiled_qc.depth()}_cz{cz_count}.png"
    plot_path = agent_dir / filename
    wire_order = build_plot_wire_order(original_qc, transpiled_qc)

    try:
        fig = circuit_drawer(
            plot_qc,
            output="mpl",
            fold=TRANSPILED_PLOT_FOLD,
            scale=1.2,
            style={
                "fontsize": TRANSPILED_PLOT_FONTSIZE,
                "subfontsize": TRANSPILED_PLOT_SUBFONTSIZE,
            },
            wire_order=wire_order,
        )
    except MissingOptionalLibraryError as exc:
        print(f"    skipped circuit plot ({exc})")
        return

    clean_transpiled_plot_labels(fig)
    fig.suptitle(f"{agent_name} - Transpiled Circuit", fontsize=20)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    if fig._suptitle is not None:
        fig._suptitle.remove()
    fig.savefig(plot_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_plot_wire_order(original_qc, transpiled_qc):
    """Preserve the original logical qubit order when drawing transpiled circuits."""
    transpile_layout = getattr(transpiled_qc, "layout", None)
    initial_layout = getattr(transpile_layout, "initial_layout", None)
    if initial_layout is None:
        return None

    physical_by_virtual = initial_layout.get_virtual_bits()
    ordered_physical = []
    seen_physical = set()

    for qubit in original_qc.qubits:
        physical = physical_by_virtual.get(qubit)
        if physical is None or physical in seen_physical:
            continue
        ordered_physical.append(physical)
        seen_physical.add(physical)

    for qubit in transpiled_qc.qubits:
        physical = transpiled_qc.find_bit(qubit).index
        if physical in seen_physical:
            continue
        ordered_physical.append(physical)
        seen_physical.add(physical)

    return ordered_physical


def clean_transpiled_plot_labels(fig):
    """Make Qiskit's plot labels easier to read in the thesis."""
    for ax in fig.axes:
        for text in ax.texts:
            label = text.get_text()

            # Register names sometimes appear as SB_0, SA_0, etc.
            label = re.sub(r"_\{0\}", "", label)
            label = re.sub(r"_0(?=\}?\\?$|\$|$)", "", label)

            # Qiskit writes conditions as c_0=0x1; use a simpler form.
            if label.startswith("c_") and "=0x" in label:
                bit_name, value = label.split("=0x")
                bit_index = bit_name.replace("c_", "")
                label = f"c[{bit_index}] = {int(value, 16)}"
                text.set_fontsize(20)
            elif label.isdigit():
                text.set_fontsize(20)

            text.set_text(label)


def safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_")


def make_run_folder_name(backend, folder_ts=None):
    """Create the shared run-folder name used for transpilation plot output."""
    if folder_ts is None:
        folder_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return folder_ts, f"{backend.name}_{folder_ts}"


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def resolve_plots_dir(backend, plots_dir=None, folder_ts=None, category="transpilation_only"):
    """Resolve a run-specific transpilation plot directory."""
    if plots_dir is not None:
        return Path(plots_dir)

    _, run_folder_name = make_run_folder_name(backend, folder_ts)
    return PLOT_DIR / category / run_folder_name


if __name__ == "__main__":
    #can be run here, but usually gets called from the main run.py script
    service = QiskitRuntimeService()
    backend = service.backend("ibm_marrakesh")
    transpile_all_agents(backend)
