"""
Simple IBM Torino coupling-map plot without calibration data.

This version uses a hardcoded layout, draws the Torino grid in black,
and highlights the shared Betting Agent / Always 3/4 layout.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeTorino

if not hasattr(np, "alltrue"):
    np.alltrue = np.all


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SVG_GEOMETRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "IBM_coupling_map"
    / "ibm_torino_readout_error_cz_calibrations_2026-03-30T03_26_45Z.svg"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots" / "plots_backend_connectivity"
OUTPUT_PREFIX = OUTPUT_DIR / "ibm_torino_betting_layout_20260330_simple"

BETTING_LAYOUT = [68, 67, 66, 74, 65, 86, 18, 131]
BETTING_EDGES = [(65, 66), (66, 67), (67, 68), (67, 74), (74, 86)]
SD_SC_EDGE = [(67, 68)]
CHOICE_QUBITS = [18, 131]
SD_SC_QUBITS = [68, 67]
OTHER_LAYOUT_QUBITS = [qubit for qubit in BETTING_LAYOUT if qubit not in CHOICE_QUBITS + SD_SC_QUBITS]


def load_svg_qubit_coordinates(svg_path: Path) -> dict[int, tuple[float, float]]:
    """Recover the fixed Torino qubit coordinates from the saved SVG."""
    svg_text = svg_path.read_text()
    matches = re.findall(
        r'<g transform="translate\(([-\d.]+), ([-\d.]+)\)" '
        r'class="cursor-pointer transition-opacity" opacity="1"><circle r="8"',
        svg_text,
    )
    if len(matches) != 133:
        raise ValueError(f"Expected 133 Torino coordinates, found {len(matches)}.")

    return {
        index: (float(x), -0.9 * float(y))
        for index, (x, y) in enumerate(matches)
    }


def build_torino_graph() -> nx.Graph:
    """Build the undirected Torino coupling graph."""
    backend = FakeTorino()
    graph = nx.Graph()
    graph.add_edges_from({tuple(sorted(edge)) for edge in backend.coupling_map.get_edges()})
    return graph


def output_prefix_with_suffix(suffix: str) -> Path:
    return OUTPUT_PREFIX.with_suffix(f".{suffix}")


def plot_torino_betting_layout_simple() -> None:
    """Render a plain Torino grid with the betting layout highlighted."""
    positions = load_svg_qubit_coordinates(SVG_GEOMETRY_PATH)
    graph = build_torino_graph()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(30, 22))

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=list(graph.edges()),
        width=10,
        edge_color="black",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=BETTING_EDGES,
        width=10,
        edge_color="#d62828",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=SD_SC_EDGE,
        width=12,
        edge_color="#2ca02c",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color="black",
        node_size=4200,
        edgecolors="black",
        linewidths=4,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=OTHER_LAYOUT_QUBITS,
        node_color="#d62828",
        node_size=4200,
        edgecolors="black",
        linewidths=4,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=SD_SC_QUBITS,
        node_color="#2ca02c",
        node_size=4200,
        edgecolors="black",
        linewidths=4,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=CHOICE_QUBITS,
        node_color="#f4d35e",
        node_size=4200,
        edgecolors="black",
        linewidths=4,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, positions, font_size=30, font_color="white", ax=ax)

    ax.set_title(
        "ibm_torino Coupling Map (March 30, 2026)\n"
        "Betting Agent / Always 3/4 layout without calibration data",
        fontsize=32,
    )
    ax.axis("off")
    plt.tight_layout()

    plt.savefig(output_prefix_with_suffix("png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_torino_betting_layout_simple()
