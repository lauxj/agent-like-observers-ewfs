import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from pathlib import Path

# Agent circuits
import reflex_agent
import guessing_agent
import betting_agent

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

# Where to save plots:
PLOT_DIR = Path("plots_noiseless")
PLOT_DIR.mkdir(exist_ok=True)


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


def expectation_AB(build_fn, A_setting, B_setting, alpha, beta1, beta2, shots):
    """Run one setting and return the correlator E_AB"""
    qc = build_fn(A_setting, B_setting, alpha, beta1, beta2)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return exp_values_from_counts(counts, shots)


def S_SB(build_fn, alpha, beta1, beta2, shots):
    """Compute S_SB from the four SB correlators"""
    # Compute the four correlators.
    E = {}
    for label, A, B in SETTINGS:
        E[label] = expectation_AB(build_fn, A, B, alpha, beta1, beta2, shots)

    # Combine correlators into S_SB.
    S = -E["A1B1"] + E["A1B2"] - E["A2B1"] - E["A2B2"] - 2
    return S



def plot_SB_circuits(build_fn, agent_name, alpha, beta1, beta2):
    """Save the four SB circuits as plots"""
    for setting_name, A, B in SETTINGS:
        qc = build_fn(A, B, alpha, beta1, beta2)
        fig = qc.draw("mpl")
        fig.set_size_inches(10, 4)
        fig.suptitle(f"{agent_name} (noiseless) – Circuit {setting_name}", fontsize=14)

        # Save image
        filename = f"{agent_name.replace(' ', '_')}_{setting_name}.png"
        fig.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)



def main(shots):
    """Run noiseless S_SB for all agents and save circuit plots"""
    alpha, beta1, beta2 = analytic_optimal_angles()

    for name, build_fn in AGENTS:
        S_val = S_SB(build_fn, alpha, beta1, beta2, shots=shots)
        print(f"{name} (noiseless simulation): S_SB ≈ {S_val:.3f}")
        plot_SB_circuits(build_fn, name, alpha, beta1, beta2)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Change shots if needed.
    main(shots=10000)
