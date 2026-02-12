import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator

import reflex_agent
import guessing_agent
import betting_agent


# Simulator we use:
sim = AerSimulator()

def exp_values_from_counts(counts, shots):
    """
    maps the 0,1 values to +1 and -1
    returns expectation values A, B, and AB
    """

    exp_A = exp_B = exp_AB = 0.0 #initialize

    for s, c in counts.items():
        #s is a bitstring like 01 and c is how many times it appeared
        B = +1 if s[0] == '0' else -1
        A = +1 if s[1] == '0' else -1
        p = c / shots
        exp_A  += p * A
        exp_B  += p * B
        exp_AB += p * A * B

    return exp_A, exp_B, exp_AB


def expectation_AB(build_fn, A_setting, B_setting, alpha, beta1, beta2, shots=20000):
    """
    Build circuit for given settings and run it.
    `build_fn` is the agent's build_measurement function.

    Returns expectation values (A, B, AB) for the chosen settings.
    """
    qc = build_fn(A_setting, B_setting, alpha, beta1, beta2)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return exp_values_from_counts(counts, shots)


def S_SB(build_fn, alpha, beta1, beta2, shots=20000):
    """
    Semi-Brukner S_SB for given angles alpha, beta1, beta2.
    `build_fn` is the agent's build_measurement function.

    Uses: A1, A2, B1, B2
    """
    # Correlators needed for SB in minimal scenario with one friend Charlie
    _, _, E_A1B1 = expectation_AB(build_fn, 1, 1, alpha, beta1, beta2, shots)
    _, _, E_A1B2 = expectation_AB(build_fn, 1, 2, alpha, beta1, beta2, shots)
    _, _, E_A2B1 = expectation_AB(build_fn, 2, 1, alpha, beta1, beta2, shots)
    _, _, E_A2B2 = expectation_AB(build_fn, 2, 2, alpha, beta1, beta2, shots)

    # SB inequality for minimal scenario with one friend:
    S = -E_A1B1 + E_A1B2 - E_A2B1 - E_A2B2 - 2
    return S


def analytic_optimal_angles():
    """
    Analytic angles for the Bell-plus implementation of the Semi-Brukner scenario.
    Returns (alpha, beta1, beta2).
    """
    alpha = 3.0 * np.pi / 2.0     # 270 degrees
    beta1 = 3.0 * np.pi / 4.0      # 135 degrees
    beta2 = 1.0 * np.pi / 4.0      # 45 degrees

    return alpha, beta1, beta2


def plot_SB_circuits(build_fn, agent_name, alpha, beta1, beta2):
    """Plot the four circuits used in the Semi-Brukner inequality."""
    settings = [
        ("A1B1", 1, 1),
        ("A1B2", 1, 2),
        ("A2B1", 2, 1),
        ("A2B2", 2, 2),
    ]

    for setting_name, A, B in settings:
        qc = build_fn(A, B, alpha, beta1, beta2)
        fig = qc.draw("mpl")
        fig.set_size_inches(10, 4)
        fig.suptitle(f"{agent_name} (noiseless) – Circuit {setting_name}", fontsize=14)

        plt.show()  #uncomment if you dont want to show the plots
        plt.close(fig)



def main(shots: int = 10000, do_plots: bool = True):
    """Run noiseless S_SB for all agents.

    Parameters
    ----------
    shots:
        Number of shots for the AerSimulator.
    do_plots:
        If True, plot the four SB circuits per agent.
    """
    alpha, beta1, beta2 = analytic_optimal_angles()

    agents = [
        ("Reflex Agent", reflex_agent.build_measurement),
        ("Guessing Agent", guessing_agent.build_measurement),
        ("Betting Agent", betting_agent.build_measurement),
    ]

    for name, build_fn in agents:
        S_val = S_SB(build_fn, alpha, beta1, beta2, shots=shots)
        print(f"{name} (noiseless simulation): S_SB ≈ {S_val:.3f}")
        if do_plots:
            plot_SB_circuits(build_fn, name, alpha, beta1, beta2)


if __name__ == "__main__":
    # When run as a script, compute and print S_SB and (optionally) draw circuits.
    # When imported, nothing runs automatically.
    main(shots=10000, do_plots=True)
