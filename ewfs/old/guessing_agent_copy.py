import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator


#Build all circuits for guessing agent:
def build_measurement(A_setting, B_setting, alpha, beta1, beta2):
    """
    Build the circuit for settings A_setting (1,2) and B_setting (1,2) and angles alpha1 and beta1, beta2.
    Includes Charlie's friend F_C.
    S_D is measured directly by Bob with no Debbie.
    """

    # quantum registers
    qr_SC = QuantumRegister(1, "S_C")
    qr_GC = QuantumRegister(1, "G")
    qr_M1 = QuantumRegister(1, "M1")
    qr_M2 = QuantumRegister(1, "M2")
    qr_SD = QuantumRegister(1, "S_D")
    cr = ClassicalRegister(2, "c")         # store A and B outcomes

    qc = QuantumCircuit(qr_SC, qr_GC, qr_M1, qr_M2, qr_SD, cr)

    # --- PRE-MEASUREMENT ---
    qc.h(qr_SC[0])                 # create |+> on S_C
    qc.cx(qr_SC[0], qr_SD[0])      # entangle → Bell pair
    qc.cx(qr_SC[0], qr_M1[0])      # Charlie pre-measures S_C
    qc.cx(qr_M1[0], qr_GC[0])      # Guess
    qc.ry(np.pi/3, qr_SC[0])       # Rotation on SC
    qc.cx(qr_SC[0], qr_M2[0])      # Charlie measures SC again after rotation
    qc.cx(qr_M2[0], qr_GC[0])      # Bookkeeping

    # --------------------------------
    # Alice setting
    # --------------------------------
    if A_setting == 1:
        # Measure F_C directly
        qc.measure(qr_M1[0], cr[0])

    if A_setting == 2:
        # Undo Charlie, then rotate S_C and measure
        qc.cx(qr_M2[0], qr_GC[0])
        qc.cx(qr_SC[0], qr_M2[0])
        qc.ry(-np.pi / 3, qr_SC[0])
        qc.cx(qr_M1[0], qr_GC[0])
        qc.cx(qr_SC[0], qr_M1[0])
        qc.ry(alpha, qr_SC[0])
        qc.measure(qr_SC[0], cr[0])

    # --------------------------------
    # Bob setting
    # --------------------------------
    if B_setting == 1:
        # Measure S_D directly in rotated basis beta1
        qc.ry(beta1, qr_SD[0])
        qc.measure(qr_SD[0], cr[1])

    if B_setting == 2:
        qc.ry(beta2, qr_SD[0])
        qc.measure(qr_SD[0], cr[1])

    return qc

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


def expectation_AB(A_setting, B_setting, alpha, beta1, beta2, shots=20000):
    """
    Build circuit for given settings and run it
    using the function exp_values_from_counts above
    returns <AB> for the chosen settings A_setting and B_setting in {1,2}
    """
    qc = build_measurement(A_setting, B_setting, alpha, beta1, beta2)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return exp_values_from_counts(counts, shots)


def S_SB(alpha, beta1, beta2, shots=20000):
    """
    Semi-Brukner S_SB for given angles alpha, beta1, beta2.
    Uses: A1, A2, B1, B2
    """
    #alpha = 0.0
    # Correlators needed for SB in minimal scenario with one friend Charlie
    _, _, E_A1B1 = expectation_AB(1, 1, alpha, beta1, beta2, shots)
    _, _, E_A1B2 = expectation_AB(1, 2, alpha, beta1, beta2, shots)
    _, _, E_A2B1 = expectation_AB(2, 1, alpha, beta1, beta2, shots)
    _, _, E_A2B2 = expectation_AB(2, 2, alpha, beta1, beta2, shots)

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


def plot_SB_circuits(alpha, beta1, beta2):
    """
    Plot the four circuits used in the Semi-Brukner inequality.
    """
    #alpha2 = 0.0  # fixed for S_SB

    settings = [
        ("A1B1", 1, 1),
        ("A1B2", 1, 2),
        ("A2B1", 2, 1),
        ("A2B2", 2, 2),
    ]

    for name, A, B in settings:
        qc = build_measurement(A, B, alpha, beta1, beta2)
        fig = qc.draw("mpl")
        fig.suptitle(f"Circuit {name}", fontsize=14)
        plt.show()


# Analytic optimal angles (Bell-plus state)
alpha = 3.0 * np.pi / 2.0     # 270 degrees
beta1 = 3.0 * np.pi / 4.0      # 135 degrees
beta2 = 1.0 * np.pi / 4.0      # 45 degrees

S_analytic = S_SB(alpha, beta1, beta2, shots=10000)
print(f"S_SB≈ {S_analytic:.3f}")

# Plot the circuits
plot_SB_circuits(alpha, beta1, beta2)

