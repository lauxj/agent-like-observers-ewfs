import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from pathlib import Path
import json
from datetime import datetime

# Agent circuits (located in ewfs/agents)
from ewfs.agents import guessing_agent, betting_agent, reflex_agent

# -----------------------------------------------------------------------------
# SETTINGS:

AGENTS = [
    ("Reflex Agent", reflex_agent.build_measurement),
    ("Guessing Agent", guessing_agent.build_measurement),
    ("Betting Agent", betting_agent.build_measurement),
]

# Semi-Brukner inequality settings:
SETTINGS = [
    ("A1B1", 1, 1),
    ("A1B2", 1, 2),
    ("A2B1", 2, 1),
    ("A2B2", 2, 2),
]

# Simulator:
sim = AerSimulator()

# Project directory (masters_thesis_project)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where to save noiseless circuit plots (project root → results/plots_noiseless_simulation):
PLOT_DIR = PROJECT_ROOT / "results" / "plots_noiseless_simulation"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Where to save noiseless simulation raw data (project root → data/data_noiseless_simulation):
DATA_DIR = PROJECT_ROOT / "data" / "data_noiseless_simulation"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# Analytic angles:
def analytic_optimal_angles():
    """Return the analytic angles"""
    alpha = 3.0 * np.pi / 2.0
    beta1 = 3.0 * np.pi / 4.0
    beta2 = 1.0 * np.pi / 4.0

    return alpha, beta1, beta2

# -----------------------------------------------------------------------------

def exp_values_from_counts(counts, shots):
    """Compute the correlator E_AB from a counts dictionary."""
    # counts example: {'00': 1000, '01': 500, ...}

    exp_AB = 0.0  # running sum

    for s, c in counts.items():
        B = +1 if s[0] == '0' else -1   # bit -> ±1
        A = +1 if s[1] == '0' else -1   # bit -> ±1
        p = c / shots
        exp_AB += p * A * B

    return exp_AB


def counts_to_jsonable(counts):
    """Convert Qiskit counts dict to a JSON-serializable dict."""
    # Qiskit counts are already dict[str,int] in most cases, but be defensive.
    return {str(k): int(v) for k, v in counts.items()}


def expectation_AB(build_fn, A_setting, B_setting, alpha, beta1, beta2, shots):
    """Run one setting and return (E_AB, counts)."""
    qc = build_fn(A_setting, B_setting, alpha, beta1, beta2)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    E = exp_values_from_counts(counts, shots)
    return E, counts


def S_SB(build_fn, alpha, beta1, beta2, shots):
    """Compute S_SB and return (S, E, counts_by_setting)."""
    E = {}
    counts_by_setting = {}

    for label, A, B in SETTINGS:
        E_val, counts = expectation_AB(build_fn, A, B, alpha, beta1, beta2, shots)
        E[label] = float(E_val)
        counts_by_setting[label] = counts_to_jsonable(counts)

    S = -E["A1B1"] + E["A1B2"] - E["A2B1"] - E["A2B2"] - 2
    return float(S), E, counts_by_setting



def plot_SB_circuits(build_fn, agent_name, alpha, beta1, beta2):
    """Save the four SB circuits as plots in agent-specific folders."""

    # Create a clean folder name per agent
    agent_folder = PLOT_DIR / agent_name.replace(" ", "_")
    agent_folder.mkdir(parents=True, exist_ok=True)

    for setting_name, A, B in SETTINGS:
        qc = build_fn(A, B, alpha, beta1, beta2)
        fig = qc.draw("mpl")
        fig.set_size_inches(10, 4)
        fig.suptitle(f"{agent_name} (noiseless) – Circuit {setting_name}", fontsize=14)

        # Save image inside agent-specific folder
        filename = f"{setting_name}.png"
        fig.savefig(agent_folder / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)



def main(shots):
    """Run noiseless S_SB for all agents, save circuit plots, and persist raw data."""
    alpha, beta1, beta2 = analytic_optimal_angles()

    run_data = {
        "kind": "noiseless_simulation",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "shots": int(shots),
        "angles": {
            "alpha": float(alpha),
            "beta1": float(beta1),
            "beta2": float(beta2),
        },
        "settings": [{"label": lbl, "A": int(A), "B": int(B)} for (lbl, A, B) in SETTINGS],
        "agents": {},
    }

    for name, build_fn in AGENTS:
        S_val, E, counts_by_setting = S_SB(build_fn, alpha, beta1, beta2, shots=shots)
        print(f"{name} (noiseless simulation): S_SB ≈ {S_val:.3f}")

        run_data["agents"][name] = {
            "S_SB": float(S_val),
            "E": E,
            "counts": counts_by_setting,
        }

        plot_SB_circuits(build_fn, name, alpha, beta1, beta2)

    # Save a compact JSON artifact for thesis reproducibility.
    ts_safe = run_data["timestamp"].replace(":", "-")
    out_path = DATA_DIR / f"noiseless_run_{ts_safe}_shots{shots}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2, sort_keys=True)

    print(f"Saved noiseless run data to: {out_path}")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Change shots if needed.
    main(shots=10000)
