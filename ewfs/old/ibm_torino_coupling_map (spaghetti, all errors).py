"""
plot_ibm_torino_live.py

Fetches LIVE calibration data from IBM Torino backend and plots the error map
with a grid-like coupling map layout.

Usage:
  python plot_ibm_torino_live.py
  python plot_ibm_torino_live.py --save torino_errors.png
  python plot_ibm_torino_live.py --save torino_errors.png --show

Requirements:
  - IBM Quantum account
  - qiskit-ibm-runtime installed
  - Valid IBM Quantum Platform token saved (use QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=...))
"""

import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_error_map


def get_live_backend(backend_name: str = "ibm_torino"):
    """
    Connect to IBM Quantum and retrieve the live backend.

    Note: You must have saved your IBM account token first using:
      from qiskit_ibm_runtime import QiskitRuntimeService
      QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="YOUR_TOKEN_HERE")
    """
    try:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        backend = service.backend(backend_name)
        return backend
    except Exception as e:
        print(f"\nError connecting to IBM Quantum: {e}")
        print("\nMake sure you have:")
        print("1. A valid IBM Quantum account")
        print("2. Saved your token using:")
        print("   from qiskit_ibm_runtime import QiskitRuntimeService")
        print('   QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="YOUR_TOKEN_HERE")')
        raise


def get_calibration_timestamp(backend) -> str:
    """
    Extract the calibration timestamp from the backend properties.
    """
    try:
        # Get backend properties which contain calibration data
        props = backend.properties()
        if hasattr(props, 'last_update_date'):
            dt = props.last_update_date
            # Format: "Day, DD Month YYYY HH:MM:SS UTC"
            return dt.strftime("%A, %d %B %Y %H:%M:%S UTC")
        else:
            return datetime.now().strftime("%A, %d %B %Y %H:%M:%S UTC")
    except Exception:
        return datetime.now().strftime("%A, %d %B %Y %H:%M:%S UTC")


def print_backend_info(backend):
    """Print summary information about the backend."""
    print("\n" + "=" * 60)
    print(f"Backend: {backend.name}")
    print("=" * 60)
    print(f"Number of qubits: {backend.num_qubits}")

    # Get coupling map info
    coupling_map = backend.coupling_map
    if coupling_map:
        edges = list(coupling_map.get_edges())
        print(f"Directed couplings: {len(edges)}")

        # Count undirected edges
        undirected = set()
        for u, v in edges:
            a, b = (u, v) if u <= v else (v, u)
            undirected.add((a, b))
        print(f"Undirected couplings: {len(undirected)}")

    # Get calibration timestamp
    timestamp = get_calibration_timestamp(backend)
    print(f"Calibration data from: {timestamp}")
    print("=" * 60)


def plot_torino_error_map(
    backend,
    save_path: str | None = None,
    show: bool = False,
):
    """
    Plot the error map for IBM Torino with improved formatting.

    The error map shows:
    - Readout errors (color-coded)
    - Two-qubit gate errors (shown on connections)
    - Grid-like layout based on backend's qubit coordinates
    """

    timestamp = get_calibration_timestamp(backend)

    try:
        # Create the error map figure
        # Try with custom figsize first
        try:
            fig = plot_error_map(backend, figsize=(20, 16))
        except TypeError:
            # Fallback if figsize not supported
            fig = plot_error_map(backend)

        # Add timestamp to the title
        # Find the main axes with the coupling map
        main_ax = None
        for ax in fig.axes:
            if hasattr(ax, 'get_title') and ax.get_title():
                main_ax = ax
                break

        if main_ax is None and fig.axes:
            main_ax = fig.axes[0]

        if main_ax:
            current_title = main_ax.get_title()
            if current_title:
                new_title = f"{current_title}\nCalibration data: {timestamp}"
            else:
                new_title = f"{backend.name} Error Map\nCalibration data: {timestamp}"
            main_ax.set_title(new_title, fontsize=12, pad=15)

        # Improve readout error colorbar formatting
        for cax in fig.axes:
            if hasattr(cax, "get_ylabel"):
                ylabel = str(cax.get_ylabel())
                if "Readout" in ylabel or "readout" in ylabel:
                    # Get and filter ticks
                    ticks = cax.get_yticks()
                    # Remove zero tick for better readability
                    ticks = [t for t in ticks if abs(t) > 1e-6]
                    cax.set_yticks(ticks)

                    # Rotate labels and adjust font size
                    for lbl in cax.get_yticklabels():
                        lbl.set_rotation(90)
                        lbl.set_verticalalignment('center')
                        lbl.set_fontsize(8)

        # Clean up text annotations to reduce clutter
        # Keep only qubit indices, hide error values
        for ax in fig.axes:
            if hasattr(ax, 'texts'):
                for txt in list(ax.texts):
                    s = txt.get_text().strip()
                    if s.isdigit():
                        # This is a qubit index - keep it visible but smaller
                        txt.set_fontsize(6)
                        txt.set_weight('bold')
                    else:
                        # This is an error value - hide it to reduce clutter
                        txt.set_visible(False)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nPlot saved to: {save_path}")

        if show:
            plt.show()

        plt.close(fig)

    except Exception as e:
        print(f"\nError creating plot: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Plot error map for IBM Torino backend with live calibration data"
    )
    parser.add_argument(
        "--backend",
        default="ibm_torino",
        help="Backend name (default: ibm_torino)",
    )
    parser.add_argument(
        "--save",
        default="torino_error_map.png",
        help="Path to save the plot (default: torino_error_map.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively",
    )

    args = parser.parse_args()

    print("Connecting to IBM Quantum...")
    backend = get_live_backend(args.backend)

    print_backend_info(backend)

    print(f"\nGenerating error map plot...")
    plot_torino_error_map(backend, save_path=args.save, show=args.show)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()

    #working script, good plot, but spaghetti