"""Depolarizing-noise simulation for Semi-Brukner S_SB.

We have three agent files:
  - reflex_agent.py
  - guessing_agent.py
  - betting_agent.py
Each one must define:
  build_measurement(A_setting, B_setting, alpha, beta1, beta2) -> QuantumCircuit

This script runs the 4 SB circuits for each agent using an AerSimulator
with a depolarizing noise model.

NOTE: We *do* need the expectation-value extraction from counts.
Qiskit returns raw measurement counts; S_SB is built from correlators ⟨AB⟩,
so we must compute them (either here or by importing a helper from your noiseless file).
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

import reflex_agent
import guessing_agent
import betting_agent


# ----------------------------
# Helpers: angles + counts -> expectations
# ----------------------------

def optimal_angles():
    """Angles you already use in the noiseless script (Bell-plus optimal)."""
    alpha = 3.0 * np.pi / 2.0
    beta1 = 3.0 * np.pi / 4.0
    beta2 = 1.0 * np.pi / 4.0
    return alpha, beta1, beta2


def exp_from_counts(counts, shots):
    """Return (⟨A⟩, ⟨B⟩, ⟨AB⟩) from 2-bit counts.

    Convention (must match your circuits):
      - bitstring s like "01"
      - B is s[0], A is s[1]
      - "0" -> +1, "1" -> -1
    """
    EA = 0.0
    EB = 0.0
    EAB = 0.0

    for s, c in counts.items():
        B = +1 if s[0] == "0" else -1
        A = +1 if s[1] == "0" else -1
        p = c / shots
        EA += p * A
        EB += p * B
        EAB += p * A * B

    return EA, EB, EAB


# ----------------------------
# Noise model
# ----------------------------

def make_noise(p1, p2, p_meas=0.0):
    """Simple depolarizing noise model.

    p1: depolarizing prob on every 1-qubit gate
    p2: depolarizing prob on every 2-qubit gate
    p_meas: optional symmetric bit-flip readout error
    """
    noise = NoiseModel()

    one_q = ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "u", "u1", "u2", "u3"]
    two_q = ["cx", "cz", "ecr", "swap"]

    if p1 > 0:
        e1 = depolarizing_error(p1, 1)
        for g in one_q:
            noise.add_all_qubit_quantum_error(e1, g)

    if p2 > 0:
        e2 = depolarizing_error(p2, 2)
        for g in two_q:
            noise.add_all_qubit_quantum_error(e2, g)

    if p_meas and p_meas > 0:
        ro = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
        noise.add_all_qubit_readout_error(ro)

    return noise


# ----------------------------
# SB computation
# ----------------------------

def run_circuit(build_fn, A_setting, B_setting, alpha, beta1, beta2, sim, shots):
    qc = build_fn(A_setting, B_setting, alpha, beta1, beta2)
    counts = sim.run(qc, shots=shots).result().get_counts()
    return exp_from_counts(counts, shots)  # returns (EA, EB, EAB)


def S_SB(build_fn, alpha, beta1, beta2, sim, shots):
    # We only need the correlators ⟨AB⟩ for the SB expression
    _, _, E11 = run_circuit(build_fn, 1, 1, alpha, beta1, beta2, sim, shots)
    _, _, E12 = run_circuit(build_fn, 1, 2, alpha, beta1, beta2, sim, shots)
    _, _, E21 = run_circuit(build_fn, 2, 1, alpha, beta1, beta2, sim, shots)
    _, _, E22 = run_circuit(build_fn, 2, 2, alpha, beta1, beta2, sim, shots)

    return -E11 + E12 - E21 - E22 - 2


# ----------------------------
# Run all agents
# ----------------------------

def run_all_agents(p1, p2, p_meas=0.0, shots=10_000):
    alpha, beta1, beta2 = optimal_angles()
    sim = AerSimulator(noise_model=make_noise(p1, p2, p_meas))

    agents = [
        ("Reflex Agent", reflex_agent.build_measurement),
        ("Guessing Agent", guessing_agent.build_measurement),
        ("Betting Agent", betting_agent.build_measurement),
    ]

    print(f"Depolarizing noise: p1={p1}, p2={p2}, p_meas={p_meas}, shots={shots}")
    for name, build_fn in agents:
        val = S_SB(build_fn, alpha, beta1, beta2, sim, shots)
        print(f"{name}: S_SB (depolarizing) ≈ {val:.3f}")


def sweep_plot(p_list, p2_scale=10.0, p_meas=0.0, shots=5_000, filename="S_SB_depolarizing_sweep.pdf"):
    """Make a simple thesis-ready PDF plot of S_SB vs p1."""
    alpha, beta1, beta2 = optimal_angles()

    agents = [
        ("Reflex Agent", reflex_agent.build_measurement),
        ("Guessing Agent", guessing_agent.build_measurement),
        ("Betting Agent", betting_agent.build_measurement),
    ]

    plt.figure(figsize=(6.5, 4.0))

    for name, build_fn in agents:
        y = []
        for p1 in p_list:
            p2 = p2_scale * p1
            sim = AerSimulator(noise_model=make_noise(p1, p2, p_meas))
            y.append(S_SB(build_fn, alpha, beta1, beta2, sim, shots))

        plt.plot(p_list, y, marker="o", label=name)

    plt.xlabel("1-qubit depolarizing probability p1")
    plt.ylabel(r"$S_{\mathrm{SB}}$")
    plt.title("Depolarizing Noise Sweep")
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {filename}")


def find_violation_thresholds(
    p_list,
    p2_scale=10.0,
    p_meas=0.0,
    shots=10_000,
    filename="S_SB_depolarizing_thresholds.pdf",
):
    """Sweep p1 and find the last p1 where each agent still violates (S_SB > 0).

    Saves a thesis-ready PDF plot with:
      - S_SB(p1) curves for each agent
      - a horizontal line at S_SB = 0 (violation boundary)
      - a vertical line at the estimated threshold for each agent (last sampled p1 with S_SB > 0)
    """
    alpha, beta1, beta2 = optimal_angles()

    agents = [
        ("Reflex Agent", reflex_agent.build_measurement),
        ("Guessing Agent", guessing_agent.build_measurement),
        ("Betting Agent", betting_agent.build_measurement),
    ]

    results = {}

    plt.figure(figsize=(6.5, 4.2))

    for name, build_fn in agents:
        y = []
        for p1 in p_list:
            p2 = p2_scale * p1
            sim = AerSimulator(noise_model=make_noise(p1, p2, p_meas))
            y.append(S_SB(build_fn, alpha, beta1, beta2, sim, shots))

        # Determine threshold: last p1 where S_SB > 0
        threshold = None
        for p1, s_val in zip(p_list, y):
            if s_val > 0:
                threshold = p1

        results[name] = {"p_list": list(p_list), "S": y, "threshold": threshold}

        plt.plot(p_list, y, marker="o", label=name)

        if threshold is not None:
            plt.axvline(threshold, linestyle="--", linewidth=1)

    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.xlabel("1-qubit depolarizing probability p1")
    plt.ylabel(r"$S_{\mathrm{SB}}$")
    plt.title("Depolarizing Noise: Violation Threshold (S_SB > 0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n=== Estimated thresholds (last sampled p1 with S_SB > 0) ===")
    for name in results:
        thr = results[name]["threshold"]
        if thr is None:
            print(f"{name}: no violation for any tested p1")
        else:
            print(f"{name}: threshold ≈ {thr}")

    print(f"Saved plot: {filename}")
    return results


# ----------------------------
# Sweep vs p2 (fixed p1)
# ----------------------------

def sweep_plot_p2(p2_list, p1=0.0, p_meas=0.0, shots=5_000, filename="S_SB_depolarizing_sweep_p2.pdf"):
    """Thesis-ready PDF plot of S_SB vs p2 for a fixed p1 and p_meas."""
    alpha, beta1, beta2 = optimal_angles()

    agents = [
        ("Reflex Agent", reflex_agent.build_measurement),
        ("Guessing Agent", guessing_agent.build_measurement),
        ("Betting Agent", betting_agent.build_measurement),
    ]

    plt.figure(figsize=(6.5, 4.0))

    for name, build_fn in agents:
        y = []
        for p2 in p2_list:
            sim = AerSimulator(noise_model=make_noise(p1, p2, p_meas))
            y.append(S_SB(build_fn, alpha, beta1, beta2, sim, shots))

        plt.plot(p2_list, y, marker="o", label=name)

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("2-qubit depolarizing probability p2")
    plt.ylabel(r"$S_{\mathrm{SB}}$")
    plt.title(f"Depolarizing Noise Sweep vs p2 (fixed p1={p1}, p_meas={p_meas})")
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {filename}")


def find_violation_thresholds_p2(
    p2_list,
    p1=0.0,
    p_meas=0.0,
    shots=10_000,
    filename="S_SB_depolarizing_thresholds_p2.pdf",
):
    """Sweep p2 (with fixed p1) and find the last p2 where each agent violates (S_SB > 0)."""
    alpha, beta1, beta2 = optimal_angles()

    agents = [
        ("Reflex Agent", reflex_agent.build_measurement),
        ("Guessing Agent", guessing_agent.build_measurement),
        ("Betting Agent", betting_agent.build_measurement),
    ]

    results = {}
    plt.figure(figsize=(6.5, 4.2))

    for name, build_fn in agents:
        y = []
        for p2 in p2_list:
            sim = AerSimulator(noise_model=make_noise(p1, p2, p_meas))
            y.append(S_SB(build_fn, alpha, beta1, beta2, sim, shots))

        threshold = None
        for p2, s_val in zip(p2_list, y):
            if s_val > 0:
                threshold = p2

        results[name] = {"p2_list": list(p2_list), "S": y, "threshold": threshold}

        plt.plot(p2_list, y, marker="o", label=name)
        if threshold is not None:
            plt.axvline(threshold, linestyle="--", linewidth=1)

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("2-qubit depolarizing probability p2")
    plt.ylabel(r"$S_{\mathrm{SB}}$")
    plt.title(f"Violation Threshold vs p2 (fixed p1={p1}, p_meas={p_meas})")
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n=== Estimated p2 thresholds (last sampled p2 with S_SB > 0) ===")
    for name in results:
        thr = results[name]["threshold"]
        if thr is None:
            print(f"{name}: no violation for any tested p2")
        else:
            print(f"{name}: threshold ≈ {thr}")

    print(f"Saved plot: {filename}")
    return results


def sweep_grid_p1_p2(
    p1_list,
    p2_list,
    p_meas=0.0,
    shots=5_000,
    out_prefix="S_SB_grid",
):
    """Beginner-friendly 2D sweep over (p1, p2).

    Saves one PDF per agent: {out_prefix}_{agent}_pmeas{p_meas}.pdf
    Heatmap shows S_SB(p1,p2) with a contour at S_SB=0.
    """
    alpha, beta1, beta2 = optimal_angles()

    agents = [
        ("Reflex Agent", reflex_agent.build_measurement),
        ("Guessing Agent", guessing_agent.build_measurement),
        ("Betting Agent", betting_agent.build_measurement),
    ]

    p1_list = list(p1_list)
    p2_list = list(p2_list)

    for agent_name, build_fn in agents:
        S_grid = np.zeros((len(p2_list), len(p1_list)))

        for i, p2 in enumerate(p2_list):
            for j, p1 in enumerate(p1_list):
                sim = AerSimulator(noise_model=make_noise(p1, p2, p_meas))
                S_grid[i, j] = S_SB(build_fn, alpha, beta1, beta2, sim, shots)

        plt.figure(figsize=(7.2, 5.0))

        # Heatmap of S_SB values
        img = plt.imshow(
            S_grid,
            origin="lower",
            aspect="auto",
            extent=[min(p1_list), max(p1_list), min(p2_list), max(p2_list)],
        )
        plt.colorbar(img, label=r"$S_{\mathrm{SB}}$")

        # Make the violation region (S_SB > 0) visually obvious
        X, Y = np.meshgrid(p1_list, p2_list)
        # Red boundary where violation disappears: S_SB = 0
        plt.contour(
            X,
            Y,
            S_grid,
            levels=[0.0],
            colors="red",
            linewidths=2.5,
        )

        plt.xlabel("1-qubit depolarizing probability p1")
        plt.ylabel("2-qubit depolarizing probability p2")
        plt.title(f"{agent_name}: Violation Region (S_SB > 0), p_meas={p_meas}")

        # Add a small label on the plot
        plt.text(
            0.02,
            0.98,
            "Red line: S_SB = 0 (violation boundary)",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            fontsize=10,
        )

        plt.tight_layout()
        plt.show()

        best = float(np.max(S_grid))
        any_violation = bool(np.any(S_grid > 0.0))
        print(f"Shown plot for {agent_name} | max S_SB={best:.3f} | any violation={any_violation}")


if __name__ == "__main__":
    # Edit numbers
    run_all_agents(p1=0.002, p2=0.02, p_meas=0.01, shots=10_000)

    # 2D combined sweep over (p1, p2)
    sweep_grid_p1_p2(
        p1_list=np.linspace(0.0, 0.03, 11),   # same range, fewer points (faster)
        p2_list=np.linspace(0.0, 0.12, 11),   # same range, fewer points (faster)
        p_meas=0.01,
        shots=5_000,
    )