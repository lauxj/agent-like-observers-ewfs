"""
plot_ibm_torino.py

Fetches and plots the coupling map for IBM Torino with REAL hardware data
including error rates from IBM Quantum.

Requirements:
  pip install qiskit qiskit-ibm-runtime matplotlib networkx

Usage:
  python plot_ibm_torino.py
  python plot_ibm_torino.py --save torino_error_map.png
"""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_error_map, plot_gate_map, plot_coupling_map


def main():
    parser = argparse.ArgumentParser(
        description="Plot IBM Torino coupling map with real hardware error data"
    )
    parser.add_argument(
        "--save",
        default="ibm_torino_error_map.png",
        help="Path to save the plot (default: ibm_torino_error_map.png)",
    )
    parser.add_argument(
        "--style",
        default="error_map",
        choices=["error_map", "gate_map", "coupling_map"],
        help="Plot style (default: error_map shows calibration errors)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot, only save it",
    )
    parser.add_argument(
        "--instance",
        default=None,
        help="IBM Quantum instance (e.g., 'ibm-q/open/main'). If not provided, uses default.",
    )
    parser.add_argument(
        "--annotate-date",
        action="store_true",
        help="Annotate the plot with the calibration timestamp (or fetch time if unavailable), and the IBM instance used.",
    )
    parser.add_argument(
        "--a4",
        action="store_true",
        help="Format the figure for A4 portrait and save as PDF (recommended for thesis figures).",
    )

    args = parser.parse_args()

    print("Connecting to IBM Quantum...")
    print("(This will use your saved credentials or prompt you to authenticate)")

    # Initialize the service - this will use saved credentials
    # If you haven't saved credentials yet, run:
    # QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="YOUR_TOKEN")
    try:
        if args.instance:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", instance=args.instance)
        else:
            service = QiskitRuntimeService(channel="ibm_quantum_platform")
    except Exception as e:
        print(f"\nError connecting to IBM Quantum: {e}")
        print("\nTo save your credentials, run:")
        print("from qiskit_ibm_runtime import QiskitRuntimeService")
        print('QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="YOUR_IBM_TOKEN")')
        print("\nGet your token from: https://quantum.ibm.com/")
        return

    print("\nFetching IBM Torino backend...")
    try:
        backend = service.backend("ibm_torino")
    except Exception as e:
        print(f"\nError fetching backend: {e}")
        print("\nAvailable backends:")
        backends = service.backends()
        for b in backends:
            print(f"  - {b.name}")
        return

    print(f"\nBackend: {backend.name}")
    print(f"Number of qubits: {backend.num_qubits}")
    print(f"Status: {backend.status().status_msg}")

    # Timestamp + source label (Brisbane timezone)
    tz = ZoneInfo("Australia/Brisbane")
    fetch_time = datetime.now(tz)
    calib_time = None
    try:
        props = backend.properties()
        if props is not None and getattr(props, "last_update_date", None) is not None:
            calib_time = props.last_update_date
    except Exception:
        props = None

    if calib_time is not None:
        if calib_time.tzinfo is None:
            calib_time = calib_time.replace(tzinfo=tz)
        time_label = "Calibration: " + calib_time.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")
    else:
        time_label = "Fetched: " + fetch_time.strftime("%Y-%m-%d %H:%M %Z")

    # Best-effort instance label
    instance_used = args.instance
    if not instance_used:
        instance_used = getattr(service, "instance", None) or getattr(service, "_instance", None) or "(default)"
    source_label = f"Source: IBM Quantum Platform • instance={instance_used}"

    # Get coupling map info
    coupling_map = backend.coupling_map
    if coupling_map:
        edges = coupling_map.get_edges()
        print(f"Number of connections (directed): {len(edges)}")

        # Convert to undirected for counting
        undirected = set()
        for u, v in edges:
            a, b = (u, v) if u <= v else (v, u)
            undirected.add((a, b))
        print(f"Number of connections (undirected): {len(undirected)}")

    print(f"\nGenerating {args.style} plot...")

    try:
        if args.style == "error_map":
            # This shows readout errors and 2-qubit gate errors with color coding
            fig = plot_error_map(backend, figsize=(24, 20))

            # A4 portrait formatting (for PDF)
            if args.a4:
                # A4 in inches: 8.27 x 11.69
                fig.set_size_inches(8.27, 11.69, forward=True)

            # Move the (typically two) readout-related colorbar axes to the top
            readout_caxes = []
            for ax_ in fig.axes:
                try:
                    yl = ax_.get_ylabel()
                except Exception:
                    yl = ""
                if yl and ("Readout" in str(yl) or "readout" in str(yl)):
                    readout_caxes.append(ax_)

            # Place up to two readout colorbars horizontally at the top
            if readout_caxes:
                # Reserve top margin for bars + header
                top_y = 0.90
                bar_h = 0.07
                bar_w = 0.38
                lefts = [0.10, 0.52]
                for i, cax in enumerate(readout_caxes[:2]):
                    try:
                        cax.set_position([lefts[i], top_y, bar_w, bar_h])
                        # Improve readability
                        for lbl in cax.get_yticklabels():
                            lbl.set_rotation(0)
                            lbl.set_fontsize(8)
                        cax.yaxis.label.set_fontsize(10)
                    except Exception:
                        pass

                # Push the main plot axes down slightly to make room
                for ax_ in fig.axes:
                    if ax_ not in readout_caxes:
                        try:
                            pos = ax_.get_position()
                            if pos.y1 > 0.88:
                                ax_.set_position([pos.x0, 0.08, pos.width, 0.78])
                        except Exception:
                            pass

            # Clean up the plot for better readability
            for cax in fig.axes:
                if hasattr(cax, "get_ylabel") and "Readout" in str(cax.get_ylabel()):
                    ticks = cax.get_yticks()
                    ticks = [t for t in ticks if abs(t) > 1e-6]
                    cax.set_yticks(ticks)
                    for lbl in cax.get_yticklabels():
                        lbl.set_rotation(90)
                        lbl.set_fontsize(8)

            # Keep qubit labels visible but hide error value annotations for clarity
            for ax in fig.axes:
                for txt in list(getattr(ax, "texts", [])):
                    s = txt.get_text().strip()
                    if s.isdigit():
                        txt.set_fontsize(6)
                    else:
                        txt.set_visible(False)

            # Add timestamp + source annotation
            if args.annotate_date:
                header = f"{backend.name} • {time_label}\n{source_label}"
                try:
                    fig.text(0.10, 0.985, header, ha="left", va="top", fontsize=10)
                except Exception:
                    pass

        elif args.style == "gate_map":
            fig = plot_gate_map(backend)

        elif args.style == "coupling_map":
            fig = plot_coupling_map(backend)

        # Save the plot
        save_path = args.save
        if args.a4 and not save_path.lower().endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf" if "." in save_path else save_path + ".pdf"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Plot saved to: {save_path}")

        # Show the plot unless --no-show is specified
        if not args.no_show:
            print("\nDisplaying plot...")
            plt.show()

        plt.close(fig)

    except Exception as e:
        print(f"\nError generating plot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()