"""
noiseless_simulation.py:
Performs noiseless simulation on noiseless AerSimulator from qiskit_aer.
"""

import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from pathlib import Path
import json
from datetime import datetime

# Agent circuits import (ewfs/agents/agents.py)
from ewfs.agents import build_circuit_reflex, build_circuit_guessing, build_circuit_betting

AGENTS = [
    ("Reflex Agent", build_circuit_reflex),
    ("Guessing Agent", build_circuit_guessing),
    ("Betting Agent", build_circuit_betting),
]

# Simulator:
sim = AerSimulator()


# Where to save noiseless circuit plots
project_root = Path(__file__).resolve().parent.parent
plot_dir = project_root / "results" / "plots_noiseless_simulation"
plot_dir.mkdir(parents=True, exist_ok=True)

# Where to save noiseless simulation raw data
DATA_DIR = project_root / "data" / "data_noiseless_simulation"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def run_noiseless_simulation(shots=10000, save=True, make_plots=True):
    """Run noiseless Aer simulations for all agent circuits."""

    print("\n=== Noiseless simulation ===")
    print(f"Shots: {shots}")
    run_data = {
        "kind": "noiseless_simulation",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "shots": int(shots),
        "agents": {},
    }

    for name, build_fn in AGENTS:
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
            agent_folder = plot_dir / name.replace(" ", "_")
            agent_folder.mkdir(parents=True, exist_ok=True)
            fig = qc.draw(output="mpl", fold=-1)
            fig.suptitle(f"{name} – Quantum Circuit", fontsize=14)
            filename = f"{name.replace(' ', '_')}_circuit.png"
            fig.savefig(agent_folder / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

    if save:
        ts_safe = run_data["timestamp"].replace(":", "-")
        out_path = DATA_DIR / f"noiseless_run_{ts_safe}_shots{shots}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, sort_keys=True)

        print(f"Saved data → {out_path}")

    return run_data


# -----------------------------------------------------------------------------

# uncomment for testing or running

if __name__ == "__main__":
    # Change shots if needed.
    run_noiseless_simulation(shots=10_000, save=True, make_plots=True)
