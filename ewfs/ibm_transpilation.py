"""
ibm_transpilation.py
Transpile all circuits for one IBM backend
Automatically choose the best qubit layout when calibration CSV data is available.
"""

import warnings
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.exceptions import MissingOptionalLibraryError
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
# Toggle between searched layouts and the hard-coded manual layouts below.
USE_AUTO_LAYOUT = False
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

OPT_LEVEL = 0
ACCURACY_TEST_INFIX = "_accuracy_test_"
SHARED_LAYOUT_REFERENCE = {
    "Always 3/4 Agent": "Betting Agent",
}
LF_RELEVANT_QUBITS_BY_AGENT = {
    "Reflex Agent": ["Achoice", "Bchoice", "M", "SA", "SB"],
    "Guessing Agent": ["Achoice", "Bchoice", "M1", "SA", "SB"],
    "Betting Agent": ["Achoice", "Bchoice", "M1", "SA", "SB"],
    "Always 3/4 Agent": ["Achoice", "Bchoice", "M1", "SA", "SB"],
}
NON_GATE_OPS = {"barrier", "delay"}


def safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_")


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

    tqc.metadata = dict(tqc.metadata or {})
    tqc.metadata["ewfs_metrics"] = transpiled_circuit_metrics(agent_name, tqc, qc)

    if save_plots:
        tqc.name = agent_name
        base_plot_dir = plots_dir if plots_dir is not None else PLOT_DIR
        safe_name = safe_label(agent_name)
        agent_dir = base_plot_dir / safe_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{safe_name}_circuit_depth{tqc.depth()}_cz{cz_n}.png"
        plot_path = agent_dir / filename
        wire_order = build_plot_wire_order(qc, tqc)

        try:
            fig = circuit_drawer(tqc, output="mpl", fold=-1, wire_order=wire_order)
        except MissingOptionalLibraryError as exc:
            print(f"    skipped circuit plot ({exc})")
        else:
            fig.suptitle(f"{agent_name} – Transpiled Circuit", fontsize=14)
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            if fig._suptitle is not None:
                fig._suptitle.remove()
            fig.savefig(plot_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
            plt.close(fig)

    return tqc


def qubit_register_label(qubit, circuit=None):
    """Return a stable logical label for one original-circuit qubit."""
    if circuit is not None:
        bit_locations = circuit.find_bit(qubit).registers
        if bit_locations:
            register, index = bit_locations[0]
            return register.name if len(register) == 1 else f"{register.name}[{index}]"

    register = getattr(qubit, "_register", None)
    index = getattr(qubit, "_index", None)
    if register is not None and index is not None:
        return register.name if len(register) == 1 else f"{register.name}[{index}]"

    return str(qubit)


def build_logical_physical_qubit_map(original_qc, transpiled_qc):
    """Map original logical qubit labels to physical qubit indices after layout."""
    transpile_layout = getattr(transpiled_qc, "layout", None)
    initial_layout = getattr(transpile_layout, "initial_layout", None)
    if initial_layout is None:
        return {}

    physical_by_virtual = initial_layout.get_virtual_bits()
    mapping = {}
    for qubit in original_qc.qubits:
        physical = physical_by_virtual.get(qubit)
        if physical is None:
            continue
        mapping[qubit_register_label(qubit, original_qc)] = int(physical)
    return mapping


def count_ops_on_physical_qubits(circuit, physical_qubits, parent_qubit_map=None):
    """
    Count operations touching each selected physical qubit.

    Control-flow blocks are expanded so conditional gates contribute to the
    relevant qubits instead of being hidden behind a single if_else operation.
    """
    physical_qubits = set(physical_qubits)
    per_physical = {
        int(physical): {
            "quantum_gate_count": 0,
            "measurement_count": 0,
            "op_counts": {},
        }
        for physical in sorted(physical_qubits)
    }
    unique_touching_op_count = 0

    def local_to_physical(qubit):
        local_index = circuit.find_bit(qubit).index
        if parent_qubit_map is None:
            return local_index
        return parent_qubit_map.get(local_index)

    for instruction in circuit.data:
        operation = instruction.operation
        op_name = operation.name
        operation_qubits = [local_to_physical(qubit) for qubit in instruction.qubits]
        operation_qubits = [qubit for qubit in operation_qubits if qubit is not None]

        if getattr(operation, "blocks", None):
            for block in operation.blocks:
                block_parent_map = {
                    block_index: operation_qubits[block_index]
                    for block_index in range(min(len(block.qubits), len(operation_qubits)))
                }
                nested_counts, nested_unique = count_ops_on_physical_qubits(
                    block,
                    physical_qubits,
                    parent_qubit_map=block_parent_map,
                )
                unique_touching_op_count += nested_unique
                for physical, metrics in nested_counts.items():
                    for nested_op_name, nested_count in metrics["op_counts"].items():
                        per_physical[physical]["op_counts"][nested_op_name] = (
                            per_physical[physical]["op_counts"].get(nested_op_name, 0)
                            + nested_count
                        )
                    per_physical[physical]["quantum_gate_count"] += metrics["quantum_gate_count"]
                    per_physical[physical]["measurement_count"] += metrics["measurement_count"]
            continue

        if op_name in NON_GATE_OPS:
            continue

        touched_physical = sorted(set(operation_qubits) & physical_qubits)
        if not touched_physical:
            continue

        unique_touching_op_count += 1
        for physical in touched_physical:
            per_physical[physical]["op_counts"][op_name] = (
                per_physical[physical]["op_counts"].get(op_name, 0) + 1
            )
            if op_name == "measure":
                per_physical[physical]["measurement_count"] += 1
            else:
                per_physical[physical]["quantum_gate_count"] += 1

    return per_physical, unique_touching_op_count


def lf_relevant_qubit_metrics(agent_name, original_qc, transpiled_qc):
    """Return per-qubit transpiled gate counts for the bits used in LF violations."""
    relevant_labels = LF_RELEVANT_QUBITS_BY_AGENT.get(agent_name, [])
    logical_to_physical = build_logical_physical_qubit_map(original_qc, transpiled_qc)
    relevant_physical = [
        logical_to_physical[label]
        for label in relevant_labels
        if label in logical_to_physical
    ]
    per_physical, unique_touching_op_count = count_ops_on_physical_qubits(
        transpiled_qc,
        relevant_physical,
    )

    metrics = []
    for label in relevant_labels:
        physical = logical_to_physical.get(label)
        if physical is None:
            continue
        qubit_counts = per_physical.get(physical, {})
        metrics.append(
            {
                "logical_qubit": label,
                "physical_qubit": int(physical),
                "quantum_gate_count": int(qubit_counts.get("quantum_gate_count", 0)),
                "measurement_count": int(qubit_counts.get("measurement_count", 0)),
                "op_counts": {
                    name: int(count)
                    for name, count in sorted(qubit_counts.get("op_counts", {}).items())
                },
            }
        )

    quantum_gate_touches = sum(item["quantum_gate_count"] for item in metrics)
    measurement_touches = sum(item["measurement_count"] for item in metrics)

    return {
        "labels": relevant_labels,
        "unique_operations_touching_lf_qubits": int(unique_touching_op_count),
        "quantum_gate_touches": int(quantum_gate_touches),
        "measurement_touches": int(measurement_touches),
        "per_qubit": metrics,
    }


def transpiled_circuit_metrics(agent_name, tqc, original_qc=None):
    ops = dict(tqc.count_ops())
    metrics = {
        "agent_name": agent_name,
        "depth": int(tqc.depth()),
        "cz_count": int(ops.get("cz", 0)),
        "size": int(tqc.size()),
        "num_qubits": int(tqc.num_qubits),
        "num_clbits": int(tqc.num_clbits),
        "operations": {name: int(count) for name, count in ops.items()},
    }
    if original_qc is not None:
        metrics["lf_relevant_qubits"] = lf_relevant_qubit_metrics(agent_name, original_qc, tqc)
    return metrics


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
                "use_auto_layout": bool(USE_AUTO_LAYOUT),
                "agents": metrics,
            },
        )

    return out


if __name__ == "__main__":
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    transpile_all_agents(backend)
