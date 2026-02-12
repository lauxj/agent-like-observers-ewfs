#!/usr/bin/env python3
"""IBM Torino calibration map.

This script visualizes device calibrations on a fixed, grid-style qubit layout.
Node colour encodes the readout assignment error per qubit. Edge colour/width
encodes an effective CX (CNOT) error per connection, estimated from the native
CZ error and single-qubit SX error via the standard CX = (I ⊗ H) CZ (I ⊗ H)
representation (with H decomposed as SX·RZ·SX and virtual RZ).

Output: ibm_torino_calibration_map.pdf
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import networkx as nx

from qiskit_ibm_runtime import QiskitRuntimeService


# Thesis-friendly PDF defaults
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 10,
    }
)


def get_readout_assignment_errors(properties) -> dict[int, float]:
    """Return per-qubit readout assignment error.

    Uses `readout_error` if present; otherwise reconstructs the standard average
    assignment error ½[p(1|0)+p(0|1)] when the confusion terms are available.
    """
    errors = {}
    for qi, qparams in enumerate(properties.qubits):
        val = None
        for p in qparams:
            if getattr(p, "name", "") == "readout_error":
                val = float(p.value)
                break

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
                val = 0.5 * (p01 + p10)

        errors[qi] = 0.0 if val is None else float(val)
    return errors


def cx_error_from_cz(backend, G: nx.Graph) -> dict[tuple[int, int], float]:
    """Estimate an effective CX (CNOT) error per undirected edge.

    Uses CZ(u,v) and SX(target) from `backend.target` and approximates
    p_fail(CX) ≈ 1 - (1 - e_CZ) (1 - e_SX)^2, corresponding to
    CX = (I ⊗ H) CZ (I ⊗ H) with H ≈ SX·RZ·SX (virtual RZ).
    """

    def key(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    tgt = getattr(backend, "target", None)
    if tgt is None:
        return {key(int(u), int(v)): 0.0 for u, v in G.edges()}

    def inst_error(name: str, qargs):
        try:
            inst = tgt[name]
            ip = inst.get(qargs, None)
        except Exception:
            return None
        if ip is None:
            return None
        e = getattr(ip, "error", None)
        return None if e is None else float(e)

    def sx_error(q: int) -> float:
        e = inst_error("sx", (q,))
        return 0.0 if e is None else float(e)

    errs: dict[tuple[int, int], float] = {}
    for u, v in G.edges():
        u, v = int(u), int(v)
        k = key(u, v)

        e_cz = inst_error("cz", (u, v))
        if e_cz is None:
            errs[k] = 0.0
            continue

        # Undirected map: pick a deterministic "target" for the H dressing.
        tgt_q = v if u < v else u
        e_sx = sx_error(tgt_q)
        errs[k] = 1.0 - (1.0 - float(e_cz)) * (1.0 - e_sx) ** 2

    return errs


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



# Helper functions for colorbar and timestamp

def make_colorbar(fig, cax, sm, label: str, fontsize: int = 9):
    """Create a vertical colorbar and place a short vertical label below it."""
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("")
    cax.set_xlabel(label, fontsize=fontsize, labelpad=6)
    cax.xaxis.set_label_position("bottom")
    cax.xaxis.label.set_rotation(90)
    cax.xaxis.label.set_verticalalignment("top")
    cax.xaxis.label.set_horizontalalignment("center")
    return cb


def brisbane_timestamp(backend, tz_name: str = "Australia/Brisbane") -> str:
    tz = ZoneInfo(tz_name)
    fetch_time = datetime.now(tz)
    try:
        props = backend.properties()
        calib_time = getattr(props, "last_update_date", None)
    except Exception:
        calib_time = None

    if calib_time is not None:
        if calib_time.tzinfo is None:
            calib_time = calib_time.replace(tzinfo=tz)
        return "Calibration: " + calib_time.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")
    return "Fetched: " + fetch_time.strftime("%Y-%m-%d %H:%M %Z")


def main():
    backend_name = "ibm_torino"

    token = os.environ.get("IBM_QUANTUM_TOKEN", None)
    if token:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")

    backend = service.backend(backend_name)

    time_label = brisbane_timestamp(backend)
    source_label = f"Source: IBM Quantum Platform • backend={backend_name}"

    coupling_map = backend.configuration().coupling_map
    readout_err = get_readout_assignment_errors(backend.properties())

    G = build_graph_from_coupling_map(coupling_map)
    pos = torino_grid_positions(spacing_factor_x=2.0, spacing_factor_y=3.0)
    cnot_err = cx_error_from_cz(backend, G)

    values = np.array([readout_err.get(int(q), 0.0) for q in G.nodes()], dtype=float)
    vmin, vmax = float(values.min()), float(values.max())

    # Readout-error colouring: use a perceptually-uniform map + nonlinear scaling for contrast
    cmap = plt.get_cmap("viridis")
    norm = mcolors.PowerNorm(gamma=0.6, vmin=vmin, vmax=vmax)
    node_colors = [cmap(norm(readout_err.get(int(q), 0.0))) for q in G.nodes()]

    fig, ax = plt.subplots(figsize=(8.27, 11.69))

    def ekey(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    edge_vals = np.array([cnot_err.get(ekey(int(u), int(v)), 0.0) for u, v in G.edges()], dtype=float)
    e_vmin = float(edge_vals.min())
    e_vmax = float(edge_vals.max())
    e_norm = plt.Normalize(vmin=e_vmin, vmax=e_vmax)
    e_cmap = plt.get_cmap("plasma")
    edge_colors = [e_cmap(e_norm(cnot_err.get(ekey(int(u), int(v)), 0.0))) for u, v in G.edges()]

    if e_vmax > e_vmin:
        edge_widths = [
            0.6
            + 2.2
            * (cnot_err.get(ekey(int(u), int(v)), 0.0) - e_vmin)
            / (e_vmax - e_vmin)
            for u, v in G.edges()
        ]
    else:
        edge_widths = 1.0

    edges_list = list(G.edges())
    segments = [[pos[int(u)], pos[int(v)]] for (u, v) in edges_list]

    widths = [float(edge_widths)] * len(edges_list) if isinstance(edge_widths, (int, float)) else [float(w) for w in edge_widths]

    lc = LineCollection(segments, colors=edge_colors, linewidths=widths, zorder=1)
    ax.add_collection(lc)
    ax.autoscale_view()

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=220,
        edgecolors="black",
        linewidths=0.6,
        ax=ax,
    )

    nx.draw_networkx_labels(G, pos, font_size=6, font_color="white", ax=ax)

    sm_readout = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_readout.set_array([])
    sm_2q = plt.cm.ScalarMappable(cmap=e_cmap, norm=e_norm)
    sm_2q.set_array([])

    y0, h = 0.20, 0.62
    gap = 0.055
    cb_w = 0.030

    cb1_x = 0.82
    cb2_x = cb1_x + cb_w + gap

    cax_2q = fig.add_axes([cb1_x, y0, cb_w, h])
    cax_ro = fig.add_axes([cb2_x, y0, cb_w, h])

    make_colorbar(fig, cax_2q, sm_2q, "CNOT")
    make_colorbar(fig, cax_ro, sm_readout, "Readout")

    ax.set_title(
        "IBM Torino Calibration Map",
        pad=10,
    )
    caption = (
        f"{time_label} • {source_label}\n"
        "Node colour: readout assignment error. Edge colour/width: effective CX (CNOT) error from CZ+H representation."
    )
    ax.text(
        0.01,
        -0.04,
        caption,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    ax.axis("off")

    fig.subplots_adjust(left=0.08, right=0.81, top=0.92, bottom=0.10)

    out_pdf = f"{backend_name}_calibration_map.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()