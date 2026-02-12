"""
ibm_connectivity.py

Prints and (optionally) draws the coupling map (connectivity graph) and calibration error map
for REAL IBM Quantum backends (e.g., ibm_torino).

This uses your locally saved IBM Quantum credentials via QiskitRuntimeService.

Usage:
  python ibm_connectivity.py
  python ibm_connectivity.py --draw
  python ibm_connectivity.py --save plots

Note:
  - Requires an IBM Quantum account and credentials saved locally.
  - The error map reflects LIVE calibrations at the time you run this script.
"""

import argparse
from typing import List, Tuple, Set
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm

from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_coupling_map, plot_gate_map, plot_error_map

from qiskit_ibm_runtime import QiskitRuntimeService

# Optional: IBMProvider (legacy provider) often exposes qubit coordinates used by plot_gate_map/plot_error_map.
# If installed, we prefer it to get grid-like plots.
try:
    from qiskit_ibm_provider import IBMProvider  # type: ignore
except Exception:  # pragma: no cover
    IBMProvider = None

# Optional: Fake backends provide qubit_coordinates (layout) but not live calibrations.
# We use them ONLY to obtain coordinates when the real backend omits them.
try:
    from qiskit_ibm_runtime.fake_provider import *  # type: ignore
except Exception:  # pragma: no cover
    pass

try:
    from qiskit_ibm_provider.fake_provider import *  # type: ignore
except Exception:  # pragma: no cover
    pass


# Keep this list in sync with the backends you care about.
# (You can still use --list-backends to discover what your account can access.)
REAL_BACKENDS = [
    "ibm_torino",
    "ibm_marrakesh",
    "ibm_fez",
]

DEFAULT_PLOT_DIR = Path("../plots")


def backend_name(backend) -> str:
    name_attr = getattr(backend, "name", None)
    # Runtime backends usually expose .name as a string
    if isinstance(name_attr, str):
        return name_attr
    return name_attr() if callable(name_attr) else str(backend)


def get_service():
    """Return a service/provider object for accessing IBM backends.

    Preference order:
      1) IBMProvider (if installed) for best plotting support (qubit coordinates).
      2) QiskitRuntimeService fallback.
    """
    if IBMProvider is not None:
        return IBMProvider()
    return QiskitRuntimeService()


def get_real_backend(service, name: str):
    """Get a REAL IBM backend handle.

    Works with either IBMProvider (preferred) or QiskitRuntimeService (fallback).
    """
    # IBMProvider uses .get_backend; Runtime service uses .backend
    if hasattr(service, "get_backend"):
        return service.get_backend(name)
    return service.backend(name)


def get_directed_edges(backend) -> List[Tuple[int, int]]:
    cm = getattr(backend, "coupling_map", None)
    if cm is None:
        raise ValueError("Backend has no coupling_map attribute.")
    if not hasattr(cm, "get_edges"):
        raise ValueError("Unsupported coupling_map type (expected .get_edges()).")
    return list(cm.get_edges())


def to_undirected(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    undirected: Set[Tuple[int, int]] = set()
    for u, v in edges:
        a, b = (u, v) if u <= v else (v, u)
        undirected.add((a, b))
    return sorted(undirected)


def print_summary(backend, max_edges: int = 80):
    edges_dir = get_directed_edges(backend)
    edges_undir = to_undirected(edges_dir)

    print("\n==============================")
    print(f"Connectivity for {backend_name(backend)} (REAL backend)")
    print("==============================")
    nqb = getattr(backend, "num_qubits", None)
    if callable(nqb):
        nqb = nqb()
    if nqb is None:
        # fallback for older backends
        try:
            nqb = backend.configuration().num_qubits
        except Exception:
            nqb = "?"
    print(f"Number of qubits: {nqb}")
    print(f"Couplings (directed):   {len(edges_dir)}")
    print(f"Couplings (undirected): {len(edges_undir)}")

    print(f"\nFirst {min(max_edges, len(edges_dir))} directed edges:")
    print(edges_dir[:max_edges])

    print(f"\nFirst {min(max_edges, len(edges_undir))} undirected edges:")
    print(edges_undir[:max_edges])


def get_qubit_coordinates(backend):
    """Return qubit coordinates if the backend provides them, else None.

    For grid-like IBM plots, Qiskit visualizers rely on these coordinates.
    Runtime backends sometimes omit them, while IBMProvider backends usually include them.
    """
    # Common location
    try:
        cfg = backend.configuration()
        coords = getattr(cfg, "qubit_coordinates", None)
        if coords:
            return coords
    except Exception:
        pass

    # Some backends expose configuration directly
    try:
        cfg = getattr(backend, "_configuration", None)
        coords = getattr(cfg, "qubit_coordinates", None)
        if coords:
            return coords
    except Exception:
        pass

    return None


def _guess_fake_backend_class_name(real_name: str) -> str:
    """Map 'ibm_torino' -> 'FakeTorino' etc."""
    base = real_name
    if base.startswith("ibm_"):
        base = base[len("ibm_"):]
    # Handle names like 'washington' -> 'Washington'
    return "Fake" + "".join(part.capitalize() for part in base.split("_"))


def get_fake_backend_for(real_backend):
    """Return a Fake backend instance matching the given real backend name, or None."""
    rname = backend_name(real_backend)
    cls_name = _guess_fake_backend_class_name(rname)
    cls = globals().get(cls_name)
    if cls is None:
        return None
    try:
        return cls()
    except Exception:
        return None


class _PlotBackendProxy:
    """Backend proxy that mixes layout from a fake backend with properties from a real backend.

    plot_error_map/plot_gate_map primarily need:
      - configuration().qubit_coordinates (for grid-like layout)
      - coupling_map (for edges)
      - properties() (for live error numbers)

    This proxy provides coordinates from `backend_for_layout` and calibrations from `backend_for_props`.
    """

    def __init__(self, backend_for_layout, backend_for_props):
        self._layout = backend_for_layout
        self._props = backend_for_props
        # Prefer the real coupling map if present (should match the device)
        self.coupling_map = getattr(backend_for_props, "coupling_map", getattr(backend_for_layout, "coupling_map", None))

    def configuration(self):
        return self._layout.configuration()

    def properties(self):
        return self._props.properties()

    @property
    def num_qubits(self):
        n = getattr(self._props, "num_qubits", None)
        return n if n is not None else getattr(self._layout, "num_qubits", None)

    @property
    def name(self):
        # Some backends expose .name as string, some as method
        return backend_name(self._props)


def draw_connectivity(
    backend,
    save_path: str | None = None,
    show: bool = True,
    style: str = "error_map",
):
    """Draw the coupling map visually.

    - style="gate_map": use Qiskit's plot_gate_map (shows qubit grid if coordinates exist).
    - style="error_map": use Qiskit's plot_error_map (shows readout + 2Q gate errors from backend properties).

    Note: For real backends, plot_gate_map / plot_error_map may reflect chip geometry when coordinates are available.
    """

    if style not in {"gate_map", "error_map"}:
        raise ValueError("style must be 'gate_map' or 'error_map'")

    coords = get_qubit_coordinates(backend)
    backend_for_plot = backend

    if coords is None:
        fake = get_fake_backend_for(backend)
        if fake is not None:
            coords = get_qubit_coordinates(fake)
            if coords is not None:
                print(f"[info] Using {backend_name(fake)} for qubit_coordinates (layout), and {backend_name(backend)} for live calibrations.")
                backend_for_plot = _PlotBackendProxy(fake, backend)

    if coords is None:
        # Still no coordinates: we cannot produce a grid/heavy-hex style plot.
        raise RuntimeError(
            f"Backend '{backend_name(backend)}' does not expose qubit_coordinates, and no matching Fake backend was found. "
            "Cannot make a grid-like plot on this installation."
        )

    # Explicit NetworkX-based grid plot using coordinates and LIVE calibration values
    # Build graph from coupling map
    edges = get_directed_edges(backend)
    edges_undir = to_undirected(edges)

    G = nx.Graph()
    G.add_edges_from(edges_undir)

    # Coordinates for layout
    pos = {i: tuple(coords[i]) for i in range(len(coords))}

    # Get LIVE readout error values from backend properties
    props = backend.properties()
    readout_err = {}
    for q in G.nodes():
        try:
            p10 = props.qubit_property(q, "prob_meas1_prep0")
            p01 = props.qubit_property(q, "prob_meas0_prep1")
            if p10 is not None and p01 is not None:
                readout_err[q] = 0.5 * (p10 + p01)
            else:
                readout_err[q] = 0.0
        except Exception:
            readout_err[q] = 0.0

    values = np.array([readout_err[n] for n in G.nodes()])
    vmin, vmax = values.min(), values.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    node_colors = [cmap(norm(readout_err[n])) for n in G.nodes()]

    plt.figure(figsize=(18, 15))
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.8)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
    nx.draw_networkx_labels(G, pos, font_size=6)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Readout assignment error")

    plt.title(f"{backend_name(backend)} — LIVE readout error map")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"\nSaved plot to: {save_path}")

    if show:
        plt.show()

    plt.close()
    return


def parse_initial_layout(s: str | None) -> List[int] | None:
    """Parse a comma-separated list like '45,44,43' into [45, 44, 43]."""
    if s is None or str(s).strip() == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List backend names accessible to your account and exit.",
    )

    parser.add_argument("--max-edges", type=int, default=80, help="How many edges to print.")
    parser.add_argument("--draw", action="store_true", help="Draw the connectivity graph.")
    parser.add_argument(
        "--style",
        default="error_map",
        choices=["gate_map", "error_map"],
        help=(
            "Plot style: 'error_map' (default) uses plot_error_map (grid-like + colors show readout + 2Q gate errors from calibrations); "
            "'gate_map' uses plot_gate_map (grid-like if coordinates exist)."
        ),
    )
    parser.add_argument("--save", default=None, help="Path to save the connectivity plot (png or directory).")

    # ---- Optional: demonstrate which physical qubits are chosen by transpilation ----
    parser.add_argument(
        "--demo-transpile",
        action="store_true",
        help="Build a small demo circuit and transpile it to show which physical qubits are chosen.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=6,
        help="Number of logical qubits for the demo circuit (only used with --demo-transpile).",
    )
    parser.add_argument(
        "--initial-layout",
        default=None,
        help="Comma-separated physical qubits for initial_layout, e.g. '45,44,43,46,47,48'.",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=3,
        help="Transpiler optimization level for the demo transpile.",
    )

    args = parser.parse_args()

    service = get_service()

    if args.list_backends:
        backends = service.backends() if hasattr(service, "backends") else []
        names = []
        for b in backends:
            n = getattr(b, "name", None)
            names.append(n() if callable(n) else n)
        names = sorted([str(x) for x in names if x is not None])
        print("\nAccessible backends:")
        for n in names:
            print(f"  - {n}")
        return

    for bname in REAL_BACKENDS:
        backend = get_real_backend(service, bname)
        print_summary(backend, max_edges=args.max_edges)

        # Determine save path per backend
        if args.save is None:
            save_dir = DEFAULT_PLOT_DIR
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{bname}_error_map.png"
        else:
            save_arg = args.save
            if save_arg.endswith(".png"):
                if "{backend}" in save_arg:
                    save_path_str = save_arg.replace("{backend}", bname)
                    save_path = Path(save_path_str)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    # Insert backend name before extension
                    p = Path(save_arg)
                    save_path = p.with_name(f"{p.stem}_{bname}{p.suffix}")
                    save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Treat as directory
                save_dir = Path(save_arg)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{bname}_error_map.png"

        # Always generate a plot file (default behavior). Only the on-screen window is controlled by --draw.
        if not args.draw and args.save is None:
            print(f"\nNo --draw/--save provided; saving to {save_path}")

        draw_connectivity(backend, save_path=str(save_path), show=args.draw, style=args.style)

        # If running all, separate outputs with a blank line
        print()


    if args.demo_transpile:
        backend = get_real_backend(service, REAL_BACKENDS[-1])
        print("\n==============================")
        print("Demo transpile: which physical qubits get chosen?")
        print("==============================")

        # Build a small demo circuit with some 2-qubit interactions (a chain of CX)
        qc = QuantumCircuit(args.n)
        for i in range(args.n - 1):
            qc.cx(i, i + 1)
        qc.measure_all()

        init_layout = parse_initial_layout(args.initial_layout)
        if init_layout is not None and len(init_layout) != args.n:
            raise ValueError(
                f"--initial-layout has {len(init_layout)} entries, but demo circuit has {args.n} qubits."
            )

        tqc = transpile(
            qc,
            backend=backend,
            optimization_level=args.opt_level,
            initial_layout=init_layout,
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=7,
        )

        print(f"Backend: {backend_name(backend)}")
        print(f"Demo logical qubits: {args.n}")
        print(f"Initial layout (requested): {init_layout}")
        print(f"Chosen layout (tqc.layout): {tqc.layout}")

        # Extract the set of physical qubits used
        phys = set()
        try:
            # Newer Qiskit layouts can be iterated as (virtual_bit, physical_bit)
            for _, p in tqc.layout.get_virtual_bits().items():
                phys.add(int(p))
        except Exception:
            # Fallback: try common string parsing
            phys = set(init_layout) if init_layout is not None else set()

        if phys:
            print(f"Physical qubits used: {sorted(phys)}")
        else:
            print("Physical qubits used: (could not extract reliably; see tqc.layout above)")

        print(f"Depth: {tqc.depth()}")
        print(f"Gate counts: {tqc.count_ops()}")


if __name__ == "__main__":
    main()