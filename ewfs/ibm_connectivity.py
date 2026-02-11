"""
print_ibm_connectivity.py

Offline: prints and (optionally) draws the coupling map (connectivity graph)
of an IBM *fake* backend (default: FakeBrisbane).

Usage:
  python print_ibm_connectivity.py
  python print_ibm_connectivity.py --backend brisbane --draw --save brisbane_connectivity.png
"""

import argparse
from typing import List, Tuple, Set
from qiskit import QuantumCircuit, transpile

import matplotlib.pyplot as plt
import networkx as nx
from qiskit.visualization import plot_coupling_map, plot_gate_map, plot_error_map

from qiskit_ibm_runtime.fake_provider import (
    FakeBrisbane,
    FakeFez,
    FakeKyoto,
    FakeOsaka,
)


FAKE_BACKENDS = {
    "brisbane": FakeBrisbane,
    "fez": FakeFez,
    "kyoto": FakeKyoto,
    "osaka": FakeOsaka,
}


def get_backend(name: str):
    name = name.lower().strip()
    if name not in FAKE_BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {sorted(FAKE_BACKENDS)}")
    return FAKE_BACKENDS[name]()


def backend_name(backend) -> str:
    name_attr = getattr(backend, "name", None)
    return name_attr() if callable(name_attr) else str(backend)


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
    print(f"Connectivity for {backend_name(backend)} (FAKE backend)")
    print("==============================")
    print(f"Number of qubits: {backend.num_qubits}")
    print(f"Couplings (directed):   {len(edges_dir)}")
    print(f"Couplings (undirected): {len(edges_undir)}")

    print(f"\nFirst {min(max_edges, len(edges_dir))} directed edges:")
    print(edges_dir[:max_edges])

    print(f"\nFirst {min(max_edges, len(edges_undir))} undirected edges:")
    print(edges_undir[:max_edges])


def draw_connectivity(
    backend,
    save_path: str | None = None,
    show: bool = True,
    style: str = "qiskit",
):
    """Draw the coupling map visually.

    - style="qiskit": use Qiskit's `plot_coupling_map` (often the most "IBM-like" look).
    - style="gate_map": use Qiskit's plot_gate_map (shows qubit grid if coordinates exist).
    - style="coupling_map": use Qiskit's plot_coupling_map (may require Graphviz).
    - style="error_map": use Qiskit's plot_error_map (shows readout + 2Q gate errors from backend properties).
    - style="spring": use a generic NetworkX spring-layout graph.

    Note: For Fake backends, this shows connectivity (who connects to whom), not the physical chip geometry.
    """

    if style not in {"gate_map", "coupling_map", "error_map", "spring"}:
        raise ValueError("style must be 'gate_map', 'coupling_map', 'error_map', or 'spring'")

    # 1) Best option (grid-like, IBM-style) if the backend provides qubit coordinates.
    if style == "gate_map":
        try:
            fig = plot_gate_map(backend)
            if save_path:
                fig.savefig(save_path, dpi=250, bbox_inches="tight")
                print(f"\nSaved plot to: {save_path}")
            if show:
                plt.show()
            plt.close(fig)
            return
        except Exception as exc:
            print(f"\n[warn] plot_gate_map failed ({exc}); falling back to spring layout.")

    # 1b) Error map (Zeng-style): colors indicate calibration errors.
    # Works for fake backends too (but reflects the snapshot bundled with the fake backend).
    if style == "error_map":
        try:
            try:
                fig = plot_error_map(backend, figsize=(24, 20))
            except TypeError:
                fig = plot_error_map(backend)
            # Improve readability of the readout-error colorbar
            # - remove the 0 tick (not informative at this scale)
            # - rotate tick labels vertically to avoid overlap
            for cax in fig.axes:
                if hasattr(cax, "get_ylabel") and "Readout" in str(cax.get_ylabel()):
                    ticks = cax.get_yticks()
                    # drop zero tick if present
                    ticks = [t for t in ticks if abs(t) > 1e-6]
                    cax.set_yticks(ticks)
                    for lbl in cax.get_yticklabels():
                        lbl.set_rotation(90)
                        lbl.set_fontsize(8)
            # The default error map annotates *many* numbers (readout + 2Q errors), which can overlap.
            # For layout selection, colors are usually enough; keep only qubit index labels.
            for ax in fig.axes:
                for txt in list(getattr(ax, "texts", [])):
                    s = txt.get_text().strip()
                    if s.isdigit():
                        txt.set_fontsize(5)
                    else:
                        txt.set_visible(False)
            if save_path:
                fig.savefig(save_path, dpi=250, bbox_inches="tight")
                print(f"\nSaved plot to: {save_path}")
            if show:
                plt.show()
            plt.close(fig)
            return
        except Exception as exc:
            print(f"\n[warn] plot_error_map failed ({exc}); falling back to gate_map.")
            style = "gate_map"

    # 2) Qiskit coupling-map plot (often cleaner) but may require system Graphviz.
    if style == "coupling_map":
        try:
            fig = plot_coupling_map(backend)
            if save_path:
                fig.savefig(save_path, dpi=250, bbox_inches="tight")
                print(f"\nSaved plot to: {save_path}")
            if show:
                plt.show()
            plt.close(fig)
            return
        except Exception as exc:
            print(f"\n[warn] plot_coupling_map failed ({exc}); falling back to spring layout.")

    # If style is "spring" (or if the above failed), we draw the generic graph below.
    edges_undir = to_undirected(get_directed_edges(backend))
    g = nx.Graph()
    g.add_nodes_from(range(backend.num_qubits))
    g.add_edges_from(edges_undir)

    pos = nx.spring_layout(g, seed=7)
    plt.figure(figsize=(11, 9))
    nx.draw_networkx_edges(g, pos, alpha=0.35, width=1.0)
    nx.draw_networkx_nodes(g, pos, node_size=140)
    nx.draw_networkx_labels(g, pos, font_size=7)
    plt.title(f"Coupling map (connectivity graph) — {backend_name(backend)}")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"\nSaved plot to: {save_path}")

    if show:
        plt.show()

    plt.close()


def parse_initial_layout(s: str | None) -> List[int] | None:
    """Parse a comma-separated list like '45,44,43' into [45, 44, 43]."""
    if s is None or str(s).strip() == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default="brisbane",
        choices=sorted(FAKE_BACKENDS.keys()),
        help="Which fake backend to use.",
    )
    parser.add_argument("--max-edges", type=int, default=80, help="How many edges to print.")
    parser.add_argument("--draw", action="store_true", help="Draw the connectivity graph.")
    parser.add_argument(
        "--style",
        default="error_map",
        choices=["gate_map", "coupling_map", "error_map", "spring"],
        help=(
            "Plot style: 'gate_map' tries Qiskit plot_gate_map (often grid-like/IBM-style if coordinates exist); "
            "'error_map' uses plot_error_map (colors show readout + 2Q gate errors from backend calibrations); "
            "'coupling_map' uses plot_coupling_map (may require system Graphviz); "
            "'spring' uses NetworkX."
        ),
    )
    parser.add_argument("--save", default=None, help="Path to save the connectivity plot (png).")

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

    backend = get_backend(args.backend)
    print_summary(backend, max_edges=args.max_edges)

    if not args.draw and not args.save:
        args.save = f"{args.backend}_coupling_map.png"
        print(f"\nNo --draw/--save provided; saving to {args.save}")

    if args.draw or args.save:
        draw_connectivity(backend, save_path=args.save, show=args.draw, style=args.style)


    if args.demo_transpile:
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