#!/usr/bin/env python3
"""
Grid-like coupling map plot for IBM Torino with live readout assignment errors.

Usage:
  python plot_torino_coupling_map.py

Notes:
- Assumes you have saved your IBM account already, OR you set IBM_QUANTUM_TOKEN.
- Produces: ibm_torino_coupling_map_live.png (and .pdf)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx


from qiskit_ibm_runtime import QiskitRuntimeService


# Helper to try new channel names for QiskitRuntimeService
def make_service(token: str | None) -> QiskitRuntimeService:
    """Create a QiskitRuntimeService using the currently supported channel names.

    Newer qiskit-ibm-runtime versions accept channels:
      - 'ibm_quantum_platform'
      - 'ibm_cloud'
    We try both for robustness.
    """
    last_err: Exception | None = None
    for channel in ("ibm_quantum_platform", "ibm_cloud"):
        try:
            if token:
                return QiskitRuntimeService(channel=channel, token=token)
            return QiskitRuntimeService(channel=channel)
        except Exception as e:  # noqa: BLE001
            last_err = e

    # If we get here, nothing worked.
    assert last_err is not None
    raise last_err


def get_readout_assignment_errors(properties) -> dict[int, float]:
    """
    Extract per-qubit readout assignment error from BackendProperties.
    Tries common field names used across Qiskit versions.
    """
    errors = {}
    for qi, qparams in enumerate(properties.qubits):
        val = None
        for p in qparams:
            name = getattr(p, "name", "")
            # Common names that appear in IBM backend properties across versions
            if name in ("readout_error", "prob_meas0_prep1", "prob_meas1_prep0", "readout_assignment_error"):
                # If it's directly readout_error, use it.
                if name == "readout_error":
                    val = float(p.value)
                    break

        # If readout_error wasn't directly present, try to reconstruct it from confusion terms if available:
        if val is None:
            p01 = None  # P(meas=1 | prep=0)
            p10 = None  # P(meas=0 | prep=1)
            for p in qparams:
                name = getattr(p, "name", "")
                if name == "prob_meas1_prep0":
                    p01 = float(p.value)
                elif name == "prob_meas0_prep1":
                    p10 = float(p.value)
            if p01 is not None and p10 is not None:
                # average assignment error (common convention)
                val = 0.5 * (p01 + p10)

        errors[qi] = 0.0 if val is None else float(val)
    return errors


def build_graph_from_coupling_map(coupling_map: list[list[int]]) -> nx.Graph:
    G = nx.Graph()
    for a, b in coupling_map:
        G.add_edge(int(a), int(b))
    return G


def torino_grid_positions(spacing_factor_x=2.0, spacing_factor_y=3.0) -> dict[int, tuple[float, float]]:
    """
    Explicit Torino positions matching the grid-like example from your notebook.
    (Qubits 0..132)
    """
    sx, sy = spacing_factor_x, spacing_factor_y

    # This is exactly the explicit layout you used in plots.ipynb (the "ibm_torino coupling map" section).
    pos = {
        0: (0 * sx, 0),
        1: (1 * sx, 0),
        2: (2 * sx, 0),
        3: (3 * sx, 0),
        4: (4 * sx, 0),
        5: (5 * sx, 0),
        6: (6 * sx, 0),
        7: (7 * sx, 0),
        8: (8 * sx, 0),
        9: (9 * sx, 0),
        10: (10 * sx, 0),
        11: (11 * sx, 0),
        12: (12 * sx, 0),
        13: (13 * sx, 0),
        14: (14 * sx, 0),
        15: (0 * sx, -1 * sy),
        16: (4 * sx, -1 * sy),
        17: (8 * sx, -1 * sy),
        18: (12 * sx, -1 * sy),
        19: (0 * sx, -2 * sy),
        20: (1 * sx, -2 * sy),
        21: (2 * sx, -2 * sy),
        22: (3 * sx, -2 * sy),
        23: (4 * sx, -2 * sy),
        24: (5 * sx, -2 * sy),
        25: (6 * sx, -2 * sy),
        26: (7 * sx, -2 * sy),
        27: (8 * sx, -2 * sy),
        28: (9 * sx, -2 * sy),
        29: (10 * sx, -2 * sy),
        30: (11 * sx, -2 * sy),
        31: (12 * sx, -2 * sy),
        32: (13 * sx, -2 * sy),
        33: (14 * sx, -2 * sy),
        34: (2 * sx, -3 * sy),
        35: (6 * sx, -3 * sy),
        36: (10 * sx, -3 * sy),
        37: (14 * sx, -3 * sy),
        38: (0 * sx, -4 * sy),
        39: (1 * sx, -4 * sy),
        40: (2 * sx, -4 * sy),
        41: (3 * sx, -4 * sy),
        42: (4 * sx, -4 * sy),
        43: (5 * sx, -4 * sy),
        44: (6 * sx, -4 * sy),
        45: (7 * sx, -4 * sy),
        46: (8 * sx, -4 * sy),
        47: (9 * sx, -4 * sy),
        48: (10 * sx, -4 * sy),
        49: (11 * sx, -4 * sy),
        50: (12 * sx, -4 * sy),
        51: (13 * sx, -4 * sy),
        52: (14 * sx, -4 * sy),
        53: (0 * sx, -5 * sy),
        54: (4 * sx, -5 * sy),
        55: (8 * sx, -5 * sy),
        56: (12 * sx, -5 * sy),
        57: (0 * sx, -6 * sy),
        58: (1 * sx, -6 * sy),
        59: (2 * sx, -6 * sy),
        60: (3 * sx, -6 * sy),
        61: (4 * sx, -6 * sy),
        62: (5 * sx, -6 * sy),
        63: (6 * sx, -6 * sy),
        64: (7 * sx, -6 * sy),
        65: (8 * sx, -6 * sy),
        66: (9 * sx, -6 * sy),
        67: (10 * sx, -6 * sy),
        68: (11 * sx, -6 * sy),
        69: (12 * sx, -6 * sy),
        70: (13 * sx, -6 * sy),
        71: (14 * sx, -6 * sy),
        72: (2 * sx, -7 * sy),
        73: (6 * sx, -7 * sy),
        74: (10 * sx, -7 * sy),
        75: (14 * sx, -7 * sy),
        76: (0 * sx, -8 * sy),
        77: (1 * sx, -8 * sy),
        78: (2 * sx, -8 * sy),
        79: (3 * sx, -8 * sy),
        80: (4 * sx, -8 * sy),
        81: (5 * sx, -8 * sy),
        82: (6 * sx, -8 * sy),
        83: (7 * sx, -8 * sy),
        84: (8 * sx, -8 * sy),
        85: (9 * sx, -8 * sy),
        86: (10 * sx, -8 * sy),
        87: (11 * sx, -8 * sy),
        88: (12 * sx, -8 * sy),
        89: (13 * sx, -8 * sy),
        90: (14 * sx, -8 * sy),
        91: (0 * sx, -9 * sy),
        92: (4 * sx, -9 * sy),
        93: (8 * sx, -9 * sy),
        94: (12 * sx, -9 * sy),
        95: (0 * sx, -10 * sy),
        96: (1 * sx, -10 * sy),
        97: (2 * sx, -10 * sy),
        98: (3 * sx, -10 * sy),
        99: (4 * sx, -10 * sy),
        100: (5 * sx, -10 * sy),
        101: (6 * sx, -10 * sy),
        102: (7 * sx, -10 * sy),
        103: (8 * sx, -10 * sy),
        104: (9 * sx, -10 * sy),
        105: (10 * sx, -10 * sy),
        106: (11 * sx, -10 * sy),
        107: (12 * sx, -10 * sy),
        108: (13 * sx, -10 * sy),
        109: (14 * sx, -10 * sy),
        110: (2 * sx, -11 * sy),
        111: (6 * sx, -11 * sy),
        112: (10 * sx, -11 * sy),
        113: (14 * sx, -11 * sy),
        114: (0 * sx, -12 * sy),
        115: (1 * sx, -12 * sy),
        116: (2 * sx, -12 * sy),
        117: (3 * sx, -12 * sy),
        118: (4 * sx, -12 * sy),
        119: (5 * sx, -12 * sy),
        120: (6 * sx, -12 * sy),
        121: (7 * sx, -12 * sy),
        122: (8 * sx, -12 * sy),
        123: (9 * sx, -12 * sy),
        124: (10 * sx, -12 * sy),
        125: (11 * sx, -12 * sy),
        126: (12 * sx, -12 * sy),
        127: (13 * sx, -12 * sy),
        128: (14 * sx, -12 * sy),
        129: (0 * sx, -13 * sy),
        130: (4 * sx, -13 * sy),
        131: (8 * sx, -13 * sy),
        132: (12 * sx, -13 * sy),
    }
    return pos


def main():
    backend_name = "ibm_torino"

    # --- IBM login ---
    # Option A: you have previously done QiskitRuntimeService.save_account(...)
    # Option B: set env var IBM_QUANTUM_TOKEN (recommended for scripts)
    token = os.environ.get("IBM_QUANTUM_TOKEN", None)
    service = make_service(token)

    backend = service.backend(backend_name)

    # --- Get coupling + properties ---
    coupling_map = backend.configuration().coupling_map
    properties = backend.properties()
    readout_err = get_readout_assignment_errors(properties)

    G = build_graph_from_coupling_map(coupling_map)
    pos = torino_grid_positions(spacing_factor_x=2.0, spacing_factor_y=3.0)

    # Ensure we have a color value for every node in the graph
    values = np.array([readout_err.get(int(q), 0.0) for q in G.nodes()], dtype=float)
    vmin, vmax = float(values.min()), float(values.max())

    # Purple colormap (dark -> light), like your example
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_purple", ["#4B0082", "#9370DB"], N=256)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    node_colors = [cmap(norm(readout_err.get(int(q), 0.0))) for q in G.nodes()]

    # --- Draw ---
    plt.figure(figsize=(35, 50))
    ax = plt.gca()

    nx.draw_networkx_edges(G, pos, width=5, ax=ax)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=5000,
        edgecolors="black",
        linewidths=5,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=36, font_color="white", ax=ax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, label="Readout Assignment Error", ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel("Readout Assignment Error", fontsize=40)

    plt.title(f"{backend_name} Coupling Map (Live Calibrations)", fontsize=44, pad=30)
    plt.axis("off")
    plt.tight_layout()

    out_png = f"{backend_name}_coupling_map_live.png"
    out_pdf = f"{backend_name}_coupling_map_live.pdf"
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()