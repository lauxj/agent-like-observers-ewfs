"""
noiseless_simulation.py:
Performs noiseless simulation on noiseless AerSimulator from qiskit_aer.
"""

import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from pathlib import Path
import json
from datetime import datetime

try:
    from ewfs.agents import AGENTS
except ModuleNotFoundError:
    from agents import AGENTS

# Simulator:
sim = AerSimulator()


# Project root and base output folders
project_root = Path(__file__).resolve().parent.parent

# Where to save noiseless simulation raw data
DATA_DIR = project_root / "data" / "data_noiseless_simulation"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Where to save noiseless circuit plots
RESULTS_DIR = project_root / "results" / "plots" / "plots_noiseless_simulation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_")


def run_noiseless_simulation(
    shots=10000,
    save=True,
    make_plots=True,
    agent_builders=None,
    folder_ts=None,
    result_filename="noiseless_simulation.json",
    plots_subdir="circuit_plots",
):
    """Run noiseless Aer simulations for all agent circuits."""

    print("\n=== Noiseless simulation ===")
    print(f"Shots: {shots}")

    timestamp = folder_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"noiseless_simulation_{timestamp}"

    data_run_dir = DATA_DIR / run_folder_name
    data_run_dir.mkdir(parents=True, exist_ok=True)

    plots_run_dir = RESULTS_DIR / run_folder_name
    plots_dir = plots_run_dir / plots_subdir
    if make_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    run_data = {
        "kind": "noiseless_simulation",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "shots": int(shots),
        "agents": {},
    }

    selected_agents = list(agent_builders) if agent_builders is not None else AGENTS

    for name, build_fn in selected_agents:
        qc = build_fn()
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts()

        # Qiskit counts are already JSON-friendly in most cases, but ensure ints.
        counts_json = {str(k): int(v) for k, v in counts.items()}

        print(f"  {name}: done")

        run_data["agents"][name] = {
            "counts": counts_json,
        }

        if make_plots:
            safe_name = safe_label(name)
            agent_folder = plots_dir / safe_name
            agent_folder.mkdir(parents=True, exist_ok=True)
            fig = qc.draw(output="mpl", fold=-1)
            fig.suptitle(f"{name} – Quantum Circuit", fontsize=14)
            filename = f"{safe_name}_circuit.png"
            fig.savefig(agent_folder / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

    if save:
        out_path = data_run_dir / result_filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, sort_keys=True)

        print(f"Saved data → {out_path}")
        if make_plots:
            print(f"Saved circuit plots → {plots_dir}")

    return run_data


# -----------------------------------------------------------------------------

# uncomment for testing or running

if __name__ == "__main__":
    # Change shots if needed.
    run_noiseless_simulation(shots=10_000, save=True, make_plots=True)
