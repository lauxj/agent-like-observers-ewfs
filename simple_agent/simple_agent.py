# simple_agent.py / lf_simulation.py

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Default shots for final evaluation
SHOTS = 4096

# Shots for angle scan (can be lower for speed)
SCAN_SHOTS = 1024

# Angles for the bases (radians)
# A2 and B2 are now rotated bases; A3/B3 reserved for later.
THETA_A2 = np.pi / 4
PHI_B2 = np.pi / 4
THETA_A3 = np.pi / 4
PHI_B3 = np.pi / 4

# Qubit indices
S_C = 0  # system qubit for Charlie
F_C = 1  # friend memory for Charlie
S_D = 2  # system qubit for Debbie
F_D = 3  # friend memory for Debbie

# Classical bits: 0 = Alice outcome, 1 = Bob outcome
A_CL = 0
B_CL = 1

backend = AerSimulator()


# ----------------------------------------------------------------------
# Helper: map bit '0'/'1' to +1/-1
# ----------------------------------------------------------------------

def bit_to_pm1(bit: str) -> int:
    """
    Map a classical bit ('0' or '1') to a ±1 value.
    0 -> +1, 1 -> -1
    """
    return +1 if bit == "0" else -1


# ----------------------------------------------------------------------
# Base EWFS circuit: Bell + friend pre-measurements
# ----------------------------------------------------------------------

def base_ewfs_circuit() -> QuantumCircuit:
    """
    Prepare the base EWFS circuit:
    - 4 qubits: S_C, F_C, S_D, F_D
    - 2 classical bits: Alice, Bob
    - Bell state on S_C, S_D
    - friend CNOTs: S_C -> F_C, S_D -> F_D
    """
    qc = QuantumCircuit(4, 2)

    # Bell state on S_C, S_D
    qc.h(S_C)
    qc.cx(S_C, S_D)

    # Friend pre-measurements (CNOTs)
    qc.cx(S_C, F_C)  # Charlie measures S_C into F_C
    qc.cx(S_D, F_D)  # Debbie measures S_D into F_D

    return qc


# ----------------------------------------------------------------------
# Alice's settings A1, A2, A3
# ----------------------------------------------------------------------

def apply_alice(qc: QuantumCircuit, x: int) -> None:
    """
    Apply Alice's setting A_x on (S_C, F_C).
    Measurement result goes to classical bit A_CL.
    """
    global THETA_A2, THETA_A3

    if x == 1:
        # A1: PEEK -> measure friend memory F_C in Z
        qc.measure(F_C, A_CL)

    elif x == 2:
        # A2: reverse friend CNOT, then measure S_C in rotated basis R_y(THETA_A2)
        qc.cx(S_C, F_C)        # reverse CNOT (self-inverse)
        qc.ry(THETA_A2, S_C)   # rotated basis (includes X if THETA_A2 = pi/2)
        qc.measure(S_C, A_CL)

    elif x == 3:
        # A3: reverse friend, then measure S_C in rotated basis R_y(THETA_A3)
        qc.cx(S_C, F_C)
        qc.ry(THETA_A3, S_C)
        qc.measure(S_C, A_CL)

    else:
        raise ValueError(f"Invalid Alice setting x={x}, must be 1,2,3.")


# ----------------------------------------------------------------------
# Bob's settings B1, B2, B3
# ----------------------------------------------------------------------

def apply_bob(qc: QuantumCircuit, y: int) -> None:
    """
    Apply Bob's setting B_y on (S_D, F_D).
    Measurement result goes to classical bit B_CL.
    """
    global PHI_B2, PHI_B3

    if y == 1:
        # B1: PEEK -> measure friend memory F_D in Z
        qc.measure(F_D, B_CL)

    elif y == 2:
        # B2: reverse friend CNOT, then measure S_D in rotated basis R_y(PHI_B2)
        qc.cx(S_D, F_D)
        qc.ry(PHI_B2, S_D)
        qc.measure(S_D, B_CL)

    elif y == 3:
        # B3: reverse friend, then measure S_D in rotated basis R_y(PHI_B3)
        qc.cx(S_D, F_D)
        qc.ry(PHI_B3, S_D)
        qc.measure(S_D, B_CL)

    else:
        raise ValueError(f"Invalid Bob setting y={y}, must be 1,2,3.")


# ----------------------------------------------------------------------
# Run one setting pair (x,y) and get counts
# ----------------------------------------------------------------------

def run_setting(x: int, y: int, shots: int) -> dict:
    """
    Build the EWFS circuit for settings (x,y), run it, and return counts.
    """
    qc = base_ewfs_circuit()
    apply_alice(qc, x)
    apply_bob(qc, y)

    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    return counts


# ----------------------------------------------------------------------
# Extract correlator ⟨A_x B_y⟩ from counts
# ----------------------------------------------------------------------

def correlator_from_counts(counts: dict) -> float:
    """
    Compute ⟨A_x B_y⟩ from counts for a given (x,y).
    Assumes 2 classical bits: B_CL (1) and A_CL (0),
    so bitstring order is 'b a' (left to right).
    """
    total_shots = sum(counts.values())
    exp_val = 0.0

    for bitstring, n in counts.items():
        if len(bitstring) != 2:
            continue
        a_bit = bitstring[-1]  # Alice (c0)
        b_bit = bitstring[-2]  # Bob   (c1)

        a = bit_to_pm1(a_bit)
        b = bit_to_pm1(b_bit)
        p = n / total_shots

        exp_val += a * b * p

    return exp_val


# ----------------------------------------------------------------------
# Extract marginal ⟨A_x⟩ from counts (for a fixed y)
# ----------------------------------------------------------------------

def marginal_A_from_counts(counts: dict) -> float:
    """
    Compute ⟨A_x⟩ from counts for a fixed (x,y).
    Uses only Alice's bit.
    """
    total_shots = sum(counts.values())
    exp_val = 0.0

    for bitstring, n in counts.items():
        if len(bitstring) < 1:
            continue
        a_bit = bitstring[-1]  # Alice is c0
        a = bit_to_pm1(a_bit)
        p = n / total_shots
        exp_val += a * p

    return exp_val


# ----------------------------------------------------------------------
# Compute S_SB for current global angles
# ----------------------------------------------------------------------

def compute_S_SB(shots: int) -> float:
    """
    Compute the semi-Brukner S_SB for the current values of
    THETA_A2, PHI_B2 (and A1,B1 fixed).
    """
    settings = [(1, 1), (1, 2), (2, 1), (2, 2)]
    counts_dict = {}

    for (x, y) in settings:
        counts_dict[(x, y)] = run_setting(x, y, shots=shots)

    A1B1 = correlator_from_counts(counts_dict[(1, 1)])
    A1B2 = correlator_from_counts(counts_dict[(1, 2)])
    A2B1 = correlator_from_counts(counts_dict[(2, 1)])
    A2B2 = correlator_from_counts(counts_dict[(2, 2)])
    A1 = marginal_A_from_counts(counts_dict[(1, 1)])

    S_SB = A1B1 + A1B2 + A2B1 - A2B2 - A1
    return S_SB, (A1B1, A1B2, A2B1, A2B2, A1)


# ----------------------------------------------------------------------
# Simple angle scan over THETA_A2, PHI_B2
# ----------------------------------------------------------------------

def scan_angles():
    """
    Coarse grid search over (THETA_A2, PHI_B2) to find a good S_SB.
    EXTENDED RANGE: angles in [0, 2π].
    """
    global THETA_A2, PHI_B2

    # FULL RANGE SCAN (this enables LF violation)
    thetas = np.linspace(0, 2*np.pi, 61)  # 0°,6°,12°,...360°  (fine enough)

    best_S = -999.0
    best_params = None

    print("Scanning angles (0 → 2π)...")

    for theta in thetas:
        for phi in thetas:
            THETA_A2 = theta
            PHI_B2 = phi
            S_SB, _ = compute_S_SB(shots=SCAN_SHOTS)

            if S_SB > best_S:
                best_S = S_SB
                best_params = (float(theta), float(phi))

    print("\nBest coarse S_SB =", best_S)
    print("Best (THETA_A2, PHI_B2) =", best_params)
    return best_S, best_params


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    # 1) Optional: scan to find good angles
    best_S, best_params = scan_angles()

    # 2) Fix angles to the best coarse values
    global THETA_A2, PHI_B2
    THETA_A2, PHI_B2 = best_params

    print("\nRe-evaluating with best coarse angles and full SHOTS...")
    S_SB, (A1B1, A1B2, A2B1, A2B2, A1) = compute_S_SB(shots=SHOTS)

    print("\n--- Results ---")
    print(f"THETA_A2 = {THETA_A2:.4f} rad")
    print(f"PHI_B2   = {PHI_B2:.4f} rad\n")
    print(f"<A1 B1> = {A1B1:.4f}")
    print(f"<A1 B2> = {A1B2:.4f}")
    print(f"<A2 B1> = {A2B1:.4f}")
    print(f"<A2 B2> = {A2B2:.4f}")
    print(f"<A1>     = {A1:.4f}")
    print(f"S_SB     = {S_SB:.4f}")
    print("LF bound: S_SB <= 2.0")
    if S_SB > 2.0:
        print("=> Local Friendliness is violated (ideal simulation, these angles).")
    else:
        print("=> No violation with current angles (try finer scan or different ranges).")


if __name__ == "__main__":
    main()