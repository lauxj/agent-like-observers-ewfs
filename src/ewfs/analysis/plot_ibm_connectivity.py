"""
Plot the Marrakesh layouts used for the EWFS agent circuits.

The script creates one figure with four panels: Reflex, Guessing, Betting, and
Always-3/4. Each panel shows the selected physical qubits on ibm_marrakesh and
highlights the two-qubit gates used by that agent circuit.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from ewfs.circuits.agents import (
    build_circuit_always_large,
    build_circuit_betting,
    build_circuit_guessing,
    build_circuit_reflex,
)
from ewfs.paths import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "results" / "plots" / "plots_backend_connectivity"
OUTPUT_BASENAME = "ibm_marrakesh_agent_connectivity"

# These are the manual layouts used for the Marrakesh real-hardware runs.
# The list order follows the quantum-register order in agents.py.
AGENT_SPECS = [
    ("Reflex Agent", build_circuit_reflex, [10, 11, 12, 13, 0, 155]),
    ("Guessing Agent", build_circuit_guessing, [10, 11, 12, 18, 13, 0, 155]),
    ("Betting Agent", build_circuit_betting, [18, 11, 12, 10, 13, 9, 0, 155]),
    ("Always 3/4 Agent", build_circuit_always_large, [18, 11, 12, 10, 13, 9, 0, 155]),
]

ROLE_COLORS = {
    "core": "#2CA02C",    # system qubits SB and SA
    "action": "#D62828",  # memory/action/wallet qubits
    "choice": "#F4D35E",  # random setting choice qubits AC and BC
}


def plot_marrakesh_agent_connectivity() -> Path:
    """Create and save the Marrakesh agent-connectivity figure."""
    backend_graph = build_marrakesh_graph()

    fig, axes = plt.subplots(1, len(AGENT_SPECS), figsize=(15, 4.2))
    fig.suptitle(
        "ibm_marrakesh agent layouts",
        fontsize=16,
        y=0.96,
    )

    for ax, (agent_name, build_circuit, layout) in zip(axes, AGENT_SPECS):
        circuit = build_circuit()
        logical_names = logical_qubit_names(circuit)
        circuit_edges = physical_circuit_edges(circuit, layout)
        background_edges = backend_edges_between_selected_qubits(backend_graph, layout)
        positions = compact_agent_positions(layout, logical_names)

        draw_agent_panel(
            ax=ax,
            graph=backend_graph,
            positions=positions,
            agent_name=agent_name,
            layout=layout,
            logical_names=logical_names,
            background_edges=background_edges,
            circuit_edges=circuit_edges,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"

    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.0)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Marrakesh connectivity plot to: {png_path}")
    print(f"Saved PDF copy to: {pdf_path}")
    return png_path


def draw_agent_panel(
    ax,
    graph,
    positions,
    agent_name,
    layout,
    logical_names,
    background_edges,
    circuit_edges,
):
    """Draw one agent on its selected Marrakesh qubits."""
    missing_edges = sorted(set(circuit_edges) - set(background_edges))
    if missing_edges:
        raise ValueError(f"{agent_name} uses non-neighbor Marrakesh qubits: {missing_edges}")

    # Thin gray edges show all ibm_marrakesh couplings among these selected qubits.
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=background_edges,
        edge_color="#7A7A7A",
        width=2.0,
        ax=ax,
    )

    # Thick colored edges show the two-qubit gates that the circuit actually uses.
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=[edge for edge in circuit_edges if not is_core_edge(edge, layout, logical_names)],
        edge_color=ROLE_COLORS["action"],
        width=4.0,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=[edge for edge in circuit_edges if is_core_edge(edge, layout, logical_names)],
        edge_color=ROLE_COLORS["core"],
        width=4.5,
        ax=ax,
    )

    for role, color in ROLE_COLORS.items():
        nodes = [
            physical
            for physical, logical_name in zip(layout, logical_names)
            if qubit_role(logical_name) == role
        ]
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=nodes,
            node_color=color,
            node_size=850,
            edgecolors="black",
            linewidths=1.5,
            ax=ax,
        )

    labels = {
        physical: f"{physical}\n{logical_name}"
        for physical, logical_name in zip(layout, logical_names)
    }
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=7,
        font_weight="bold",
        ax=ax,
    )

    ax.set_title(agent_name, fontsize=12, pad=10)
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-2.65, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")


def build_marrakesh_graph() -> nx.Graph:
    """Build the undirected ibm_marrakesh coupling graph."""
    backend = FakeMarrakesh()
    graph = nx.Graph()
    graph.add_nodes_from(range(backend.num_qubits))
    graph.add_edges_from(tuple(sorted(edge)) for edge in backend.coupling_map.get_edges())
    return graph


def logical_qubit_names(circuit) -> list[str]:
    """Return logical qubit names in the same order used by the layout list."""
    names = []
    for register in circuit.qregs:
        if len(register) == 1:
            names.append(register.name)
        else:
            names.extend(f"{register.name}[{index}]" for index in range(len(register)))
    return names


def physical_circuit_edges(circuit, layout: list[int]) -> list[tuple[int, int]]:
    """Map each circuit two-qubit gate to physical Marrakesh qubits."""
    edges = set()

    for instruction in circuit.data:
        qargs = instruction.qubits
        if len(qargs) != 2:
            continue

        left = circuit.find_bit(qargs[0]).index
        right = circuit.find_bit(qargs[1]).index
        edges.add(tuple(sorted((layout[left], layout[right]))))

    return sorted(edges)


def backend_edges_between_selected_qubits(graph: nx.Graph, layout: list[int]) -> list[tuple[int, int]]:
    """Return Marrakesh coupling edges among the selected physical qubits."""
    return sorted(tuple(sorted(edge)) for edge in graph.subgraph(layout).edges())


def compact_agent_positions(layout: list[int], logical_names: list[str]) -> dict[int, tuple[float, float]]:
    """Place the same logical roles in the same visual positions in every panel."""
    positions_by_name = {
        "SB": (0.0, 1.05),
        "SA": (0.0, 0.05),
        "M": (-0.95, -0.85),
        "M1": (-0.95, -0.85),
        "M2": (0.95, -0.85),
        "R": (-1.55, -1.55),
        "G": (-1.55, -1.55),
        "W0": (-1.55, -1.55),
        "W1": (1.55, -1.55),
        "AC": (-1.45, 0.65),
        "BC": (1.45, 0.65),
    }

    return {
        physical: positions_by_name[logical_name]
        for physical, logical_name in zip(layout, logical_names)
    }


def is_core_edge(edge: tuple[int, int], layout: list[int], logical_names: list[str]) -> bool:
    """Return True for the SB-SA entangling edge."""
    core_nodes = {
        physical
        for physical, logical_name in zip(layout, logical_names)
        if logical_name in {"SB", "SA"}
    }
    return set(edge) == core_nodes


def qubit_role(logical_name: str) -> str:
    """Group logical qubits into the three colors used in the plot."""
    if logical_name in {"AC", "BC"}:
        return "choice"
    if logical_name in {"SB", "SA"}:
        return "core"
    return "action"


if __name__ == "__main__":
    plot_marrakesh_agent_connectivity()
