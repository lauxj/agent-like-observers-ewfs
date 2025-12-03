import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from sympy.abc import theta


#build all circuits
def build_measurement(A_setting, B_setting, alpha2, alpha3, beta2, beta3):
    """
    Build the circuit for Alice-setting A_setting (1,2,3)
    and Bob-setting B_setting (1,2,3).
    alpha2 = angle for Alice REVERSE-1
    alpha3 = angle for Alice REVERSE-2
    beta2 = angle for Bob REVERSE-1
    beta3 = angle for Bob REVERSE-2
    returns the quantum circuit qc for the input settings
    """

    # quantum registers
    qr1 = QuantumRegister(1, "S_C")
    qr2 = QuantumRegister(1, "F_C")
    qr3 = QuantumRegister(1, "S_D")
    cr = ClassicalRegister(2, "c")  # classical register to store the measurement outcomes

    qc = QuantumCircuit(qr1, qr2, qr3, cr)

    # --- pre-measurement ---
    qc.h(qr1[0])          # create superposition on S_C
    qc.cx(qr1[0], qr3[0]) # entangle S_C with S_D (Bell state)
    qc.cx(qr1[0], qr2[0]) # Charlie "measures" S_C into F_C

    # --------------------------------
    # Alice's setting
    # --------------------------------
    if A_setting == 1:
        # PEEK: measure friend's memory directly
        qc.measure(qr2[0], cr[0])
        #qc.draw("mpl")
        #plt.show()
    else:
        # REVERSE needed
        qc.cx(qr1[0], qr2[0])  # undo Charlie
        # choose angle
        if A_setting == 2:
            qc.ry(alpha2, qr1[0])
        elif A_setting == 3:
            qc.ry(alpha3, qr1[0])
        qc.measure(qr1[0], cr[0])
        #qc.draw("mpl")
        #plt.show()
    # --------------------------------
    # Bob's setting (no Debbie friend)
    # --------------------------------
    if B_setting == 1:
        # PEEK-like: direct Z-basis measurement on S_D
        qc.measure(qr3[0], cr[1])
        #qc.draw("mpl")
        #plt.show()
    else:
        # Different bases on S_D via R_y rotations
        if B_setting == 2:
            qc.ry(beta2, qr3[0])
        elif B_setting == 3:
            qc.ry(beta3, qr3[0])
        qc.measure(qr3[0], cr[1])
        #qc.draw("mpl")
        #plt.show()
    return qc


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


def expectation_AB(A_setting, B_setting, alpha2, alpha3, beta2, beta3, shots=20000):
    """
    Build circuit for given (A_i, B_j), run it
    using the function exp_values_from_counts above
    returns <A>, <B>, <AB> for the chosen settings A_setting and B_setting element of {1,2,3}
    """
    qc = build_measurement(A_setting, B_setting, alpha2, alpha3, beta2, beta3)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return exp_values_from_counts(counts, shots)


def S_SB(alpha3, beta2, beta3, shots=20000):
    """
    Semi-Brukner S_SB for given angles alpha3, beta2, beta3.
    Uses:
        A1 = PEEK (1)
        A2 = REVERSE-1 (2) fixed at 0
        A3 = REVERSE-2 (3)
        B2 = REVERSE-1 (2)
        B3 = REVERSE-2 (3)
    """
    alpha2 = 0.0
    # Only need the correlators <A_i B_j>
    _, _, E_A1B2 = expectation_AB(1, 2, alpha2, alpha3, beta2, beta3, shots)
    _, _, E_A1B3 = expectation_AB(1, 3, alpha2, alpha3, beta2, beta3, shots)
    _, _, E_A3B2 = expectation_AB(3, 2, alpha2, alpha3, beta2, beta3, shots)
    _, _, E_A3B3 = expectation_AB(3, 3, alpha2, alpha3, beta2, beta3, shots)

    S = -E_A1B2 + E_A1B3 - E_A3B2 - E_A3B3 - 2
    return S

def analytic_optimal_angles():
    """
    Analytic angles for the Bell-plus implementation of the Semi-Brukner scenario.
    These angles yield the theoretical maximum S_SB = 2*sqrt(2) - 2 for the ideal noiseless model.
    Returns (alpha3, beta2, beta3).
    """
    beta2 = 3.0 * np.pi / 4.0      # 135 degrees
    beta3 = 1.0 * np.pi / 4.0      # 45 degrees
    alpha3 = 3.0 * np.pi / 2.0     # 270 degrees
    return alpha3, beta2, beta3

def optimise_S_SB(shots=10000, n_grid=100):
    angles = np.linspace(0, 2*np.pi, n_grid)
    best_S = -np.inf
    best_thetas = (None, None, None)

    for a3 in angles:
        for b2 in angles:
            for b3 in angles:
                S = S_SB(a3, b2, b3, shots=shots)
                if S > best_S:
                    best_S = S
                    best_thetas = (a3, b2, b3)

    return best_S, best_thetas

def plot_SB_circuits(alpha3, beta2, beta3):
    """
    Plot the four circuits used in the Semi-Brukner inequality.
    """
    alpha2 = 0.0  # fixed for S_SB

    settings = [
        ("A1B2", 1, 2),
        ("A1B3", 1, 3),
        ("A3B2", 3, 2),
        ("A3B3", 3, 3),
    ]

    for name, A, B in settings:
        qc = build_measurement(A, B, alpha2, alpha3, beta2, beta3)
        fig = qc.draw("mpl")
        fig.suptitle(f"Circuit {name}", fontsize=14)
        plt.show()



best_S, best_thetas = optimise_S_SB(shots=10000, n_grid=15)

print(f"Best S_SB ≈ {best_S:.3f}")
print(f"Best angles: alpha3 ≈ {best_thetas[0]:.3f}, beta2 ≈ {best_thetas[1]:.3f}, beta3 ≈ {best_thetas[2]:.3f}")

if best_S > 0:
    print("→ Local Friendliness (semi-Brukner) is violated.")
else:
    print("→ No violation found on this grid.")

# Analytic optimal angles (Bell-plus state)
alpha3_opt, beta2_opt, beta3_opt = analytic_optimal_angles()
S_analytic = S_SB(alpha3_opt, beta2_opt, beta3_opt, shots=10000)

print(f"Analytic angles: alpha3 = {alpha3_opt:.3f}, beta2 = {beta2_opt:.3f}, beta3 = {beta3_opt:.3f}")
print(f"S_SB at analytic angles ≈ {S_analytic:.3f}")

# Plot the circuits
a3, b2, b3 = alpha3_opt, beta2_opt, beta3_opt
plot_SB_circuits(a3, b2, b3)
