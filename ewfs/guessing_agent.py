import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


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

