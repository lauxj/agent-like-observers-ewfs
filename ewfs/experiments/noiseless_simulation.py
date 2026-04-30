"""
noiseless_simulation.py
Runs the agent circuits on a noiseless Qiskit Aer simulator.
"""

import json
import re
from datetime import datetime
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator

EWFS_ROOT = Path(__file__).resolve().parents[1]
if str(EWFS_ROOT) not in sys.path:
    sys.path.insert(0, str(EWFS_ROOT))

from circuits.agents import AGENTS

# define directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "data_noiseless_simulation"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots" / "plots_noiseless_simulation"
NOISELESS_PLOT_FOLD = 18
NOISELESS_PLOT_FONTSIZE = 16
NOISELESS_PLOT_SUBFONTSIZE = 12


def run_noiseless_simulation(shots=10000, save=True, make_plots=True, agent_builders=None, folder_ts=None, result_filename="noiseless_simulation.json", plots_subdir="circuit_plots", ):
    """Run the selected agent circuits on a noiseless simulator."""

    print("\n=== Noiseless simulation ===")
    print(f"Shots: {shots}")

    timestamp = folder_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"noiseless_simulation_{timestamp}"

    # Use one timestamped folder for the data and optional circuit plots.
    data_dir = DATA_DIR / run_name
    plots_dir = PLOTS_DIR / run_name / plots_subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    if make_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # AerSimulator without a noise model gives the ideal/noiseless result.
    simulator = AerSimulator()
    builders = agent_builders or AGENTS

    run_data = {
        "kind": "noiseless_simulation",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "shots": int(shots),
        "agents": {},
    }

    for agent_name, build_circuit in builders:
        qc = build_circuit()
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()

        # Save the bitstring counts in a JSON-friendly form.
        run_data["agents"][agent_name] = {
            "counts": {str(key): int(value) for key, value in counts.items()},
        }

        if make_plots:
            save_circuit_plot(qc, agent_name, plots_dir)

        print(f"  {agent_name}: done")

    if save:
        output_file = data_dir / result_filename
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, sort_keys=True)

        print(f"Saved data to: {output_file}")
        if make_plots:
            print(f"Saved circuit plots to: {plots_dir}")

    return run_data


def safe_label(label: str) -> str:
    """Make a label safe to use as a file or folder name."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_")


def save_circuit_plot(qc, agent_name, plots_dir):
    """Save one circuit diagram as PNG and PDF."""
    safe_name = safe_label(agent_name)
    agent_dir = plots_dir / safe_name
    agent_dir.mkdir(parents=True, exist_ok=True)

    fig = make_circuit_plot_figure(qc, agent_name)

    png_path = agent_dir / f"{safe_name}_circuit.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    if fig._suptitle is not None:
        fig._suptitle.remove()
    fig.savefig(png_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_circuit_plot_figure(qc, agent_name, title=True, fold=NOISELESS_PLOT_FOLD):
    """Create one circuit diagram figure without saving it."""
    plot_qc = qc.copy()
    plot_qc.global_phase = 0

    draw_kwargs = {
        "output": "mpl",
        "scale": 1.2,
        "style": {
            "fontsize": NOISELESS_PLOT_FONTSIZE,
            "subfontsize": NOISELESS_PLOT_SUBFONTSIZE,
        },
    }
    if fold is not None:
        draw_kwargs["fold"] = fold

    # Qiskit's matplotlib drawer returns a figure that notebooks can display.
    fig = plot_qc.draw(**draw_kwargs)
    clean_circuit_plot_labels(fig)
    if title:
        fig.suptitle(f"{agent_name} - Quantum Circuit", fontsize=20)
    return fig


def clean_circuit_plot_labels(fig):
    """Make Qiskit's plot labels easier to read in the thesis."""
    for ax in fig.axes:
        for text in ax.texts:
            label = text.get_text()

            # Register names sometimes appear as SB_0, SA_0, etc.
            label = re.sub(r"_\{0\}", "", label)
            label = re.sub(r"_0(?=\}?\\?$|\$|$)", "", label)

            if label.startswith("c_") and "=0x" in label:
                bit_name, value = label.split("=0x")
                bit_index = bit_name.replace("c_", "")
                label = f"c[{bit_index}] = {int(value, 16)}"
                text.set_fontsize(20)
            elif label.isdigit():
                text.set_fontsize(20)

            text.set_text(label)


if __name__ == "__main__":
    # can be run using this file, but usually is called from the main run script
    run_noiseless_simulation(shots=10_000, save=True, make_plots=True)
