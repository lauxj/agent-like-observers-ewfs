"""
Simple IBM Torino coupling-map plot in the same style as the uploaded notebook.

The plot uses the March 30, 2026 calibration CSV, validates the 10 starred
real-hardware Torino runs from that date, and highlights the shared
Betting Agent / Always 3/4 layout in red.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeTorino

if not hasattr(np, "alltrue"):
    np.alltrue = np.all


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_HARDWARE_DIR = PROJECT_ROOT / "data" / "data_real_hardware"
CALIBRATION_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "IBM_coupling_map"
    / "ibm_torino_calibrations_2026-03-30T03_26_45Z.csv"
)
SVG_GEOMETRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "IBM_coupling_map"
    / "ibm_torino_readout_error_cz_calibrations_2026-03-30T03_26_45Z.svg"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots" / "plots_backend_connectivity"
OUTPUT_PREFIX = OUTPUT_DIR / "ibm_torino_betting_layout_20260330_notebook_style"

EXPECTED_SELECTED_RUN_COUNT = 10
BETTING_LAYOUT = [68, 67, 66, 74, 65, 86, 18, 131]
BETTING_EDGES = [(65, 66), (66, 67), (67, 68), (67, 74), (74, 86)]
SD_QUBIT = 68
SC_QUBIT = 67
CHOICE_QUBIT_A = 18
CHOICE_QUBIT_B = 131
BETTING_CORE_QUBITS = [66, 74, 65, 86]

HIGHLIGHT_COLORS = {
    "core": "#d62828",
    "sd": "#1d4ed8",
    "sc": "#2ca02c",
    "choice_a": "#f4d35e",
    "choice_b": "#f4d35e",
}


def find_selected_march_30_runs(base_dir: Path) -> list[Path]:
    """Return the March 30 Torino real-hardware run folders marked with '*'."""
    return sorted(
        run_dir
        for run_dir in base_dir.glob("ibm_torino_20260330_*")
        if run_dir.name.endswith("*")
    )


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

    # Keep the geometry a bit more compact so the figure is less vertically stretched.
    return {
        index: (float(x), -0.9 * float(y))
        for index, (x, y) in enumerate(matches)
    }


def load_readout_errors(csv_path: Path) -> dict[int, float]:
    """Load per-qubit readout assignment errors from the calibration CSV."""
    readout_errors: dict[int, float] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            readout_errors[int(row["Qubit"])] = float(row["Readout assignment error"])
    return readout_errors


def load_cz_errors(csv_path: Path) -> dict[tuple[int, int], float]:
    """Load undirected CZ errors by averaging the values reported on each endpoint."""
    edge_values: dict[tuple[int, int], list[float]] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qubit = int(row["Qubit"])
            for item in row["CZ error"].split(";"):
                if not item:
                    continue
                neighbor_text, error_text = item.split(":")
                neighbor = int(neighbor_text)
                edge = tuple(sorted((qubit, neighbor)))
                edge_values.setdefault(edge, []).append(float(error_text))

    return {edge: sum(values) / len(values) for edge, values in edge_values.items()}


def build_torino_graph() -> nx.Graph:
    """Build the undirected Torino coupling graph."""
    backend = FakeTorino()
    graph = nx.Graph()
    graph.add_edges_from({tuple(sorted(edge)) for edge in backend.coupling_map.get_edges()})
    return graph


def plot_torino_betting_layout() -> None:
    """Render the notebook-style Torino calibration map with the betting layout highlighted."""
    selected_runs = find_selected_march_30_runs(REAL_HARDWARE_DIR)
    if len(selected_runs) != EXPECTED_SELECTED_RUN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SELECTED_RUN_COUNT} starred March 30 Torino runs, "
            f"found {len(selected_runs)}."
        )

    positions = load_svg_qubit_coordinates(SVG_GEOMETRY_PATH)
    readout_errors = load_readout_errors(CALIBRATION_CSV_PATH)
    cz_errors = load_cz_errors(CALIBRATION_CSV_PATH)
    graph = build_torino_graph()

    # Same purple style as the notebook.
    node_color_stops = ["#4B0082", "#b9a0ff"]
    node_cmap = mcolors.LinearSegmentedColormap.from_list("custom_purple", node_color_stops, N=100)
    node_vmin = min(readout_errors.values())
    node_vmax = max(readout_errors.values())
    node_norm = plt.Normalize(vmin=node_vmin, vmax=node_vmax)
    node_colors = [node_cmap(node_norm(readout_errors.get(node, 0.0))) for node in graph.nodes()]

    edge_color_stops = ["#4B0082", "#c5afff"]
    edge_cmap = mcolors.LinearSegmentedColormap.from_list("custom_purple_edges", edge_color_stops, N=100)
    edge_list = list(graph.edges())
    edge_values = [cz_errors.get(tuple(sorted(edge)), 0.0) for edge in edge_list]
    edge_vmin = min(edge_values)
    edge_vmax = max(edge_values)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(34, 28))

    # Red glow behind the betting edges so the layout is still obvious.
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=BETTING_EDGES,
        width=16,
        edge_color="red",
        alpha=0.22,
        ax=ax,
    )
    edge_norm = plt.Normalize(vmin=edge_vmin, vmax=edge_vmax)
    edge_colors = [edge_cmap(edge_norm(value)) for value in edge_values]
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=edge_list,
        edge_color=edge_colors,
        width=8,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=5000,
        edgecolors="black",
        linewidths=5,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=BETTING_CORE_QUBITS,
        node_color="none",
        node_size=5000,
        edgecolors=HIGHLIGHT_COLORS["core"],
        linewidths=10,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[SD_QUBIT],
        node_color="none",
        node_size=5000,
        edgecolors=HIGHLIGHT_COLORS["sd"],
        linewidths=10,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[SC_QUBIT],
        node_color="none",
        node_size=5000,
        edgecolors=HIGHLIGHT_COLORS["sc"],
        linewidths=10,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[CHOICE_QUBIT_A],
        node_color="none",
        node_size=5000,
        edgecolors=HIGHLIGHT_COLORS["choice_a"],
        linewidths=10,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[CHOICE_QUBIT_B],
        node_color="none",
        node_size=5000,
        edgecolors=HIGHLIGHT_COLORS["choice_b"],
        linewidths=10,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, positions, font_size=36, font_color="white", ax=ax)

    node_sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    node_sm.set_array([])

    edge_sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
    edge_sm.set_array([])

    ax.set_title(
        "ibm_torino Coupling Map (March 30, 2026 Calibrations)\n"
        "Betting Agent / Always 3/4 layout highlighted by role",
        fontsize=36,
    )
    ax.axis("off")
    fig.subplots_adjust(left=0.13, right=0.87, top=0.93, bottom=0.04)

    ax_position = ax.get_position()
    colorbar_height = ax_position.height * 0.78
    colorbar_y = ax_position.y0 + (ax_position.height - colorbar_height) / 2
    colorbar_width = 0.016
    left_colorbar_x = ax_position.x0 - 0.045
    right_colorbar_x = ax_position.x1 + 0.029

    node_cax = fig.add_axes([left_colorbar_x, colorbar_y, colorbar_width, colorbar_height])
    edge_cax = fig.add_axes([right_colorbar_x, colorbar_y, colorbar_width, colorbar_height])

    node_colorbar = fig.colorbar(node_sm, cax=node_cax)
    node_colorbar.ax.tick_params(labelsize=20)
    node_colorbar.ax.set_ylabel("Readout Assignment Error", fontsize=34)
    node_colorbar.ax.yaxis.set_label_position("left")
    node_colorbar.ax.yaxis.tick_left()

    edge_colorbar = fig.colorbar(edge_sm, cax=edge_cax)
    edge_colorbar.ax.tick_params(labelsize=20)
    edge_colorbar.ax.set_ylabel("Two-Qubit CZ Error", fontsize=34)

    for suffix in ("png", "pdf", "svg"):
        plt.savefig(output_prefix_with_suffix(suffix), bbox_inches="tight")
    plt.close()

    print("Saved notebook-style Torino plot to:")
    for suffix in ("png", "pdf", "svg"):
        print(f"  {output_prefix_with_suffix(suffix)}")
    print("Using starred March 30 run folders:")
    for run_dir in selected_runs:
        print(f"  {run_dir}")


def output_prefix_with_suffix(suffix: str) -> Path:
    return OUTPUT_PREFIX.with_suffix(f".{suffix}")


if __name__ == "__main__":
    plot_torino_betting_layout()
