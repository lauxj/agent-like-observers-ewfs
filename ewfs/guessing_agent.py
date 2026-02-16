import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


# Build circuit for guessing agent:
def build_measurement(A_setting, B_setting, alpha, beta1, beta2):
    """
    Builds the circuit for the guessing agent for settings and angles given to the function
    """

    # quantum registers
    qr_SC = QuantumRegister(1, "S_C")
    qr_GC = QuantumRegister(1, "G")
    qr_M1 = QuantumRegister(1, "M1")
    qr_M2 = QuantumRegister(1, "M2")
    qr_SD = QuantumRegister(1, "S_D")
    cr = ClassicalRegister(2, "c")         # store A and B outcomes

    qc = QuantumCircuit(qr_SD, qr_SC,qr_M1, qr_GC, qr_M2,  cr)

    # PRE-MEASUREMENT
    qc.h(qr_SC[0])                 # create |+> on S_C
    qc.cx(qr_SC[0], qr_SD[0])      # entangle → Bell pair
    qc.cx(qr_SC[0], qr_M1[0])      # Charlie pre-measures S_C
    qc.cx(qr_M1[0], qr_GC[0])      # Guess
    qc.ry(np.pi/3, qr_SC[0])       # Rotation on SC
    qc.cx(qr_SC[0], qr_M2[0])      # Charlie measures SC again after rotation
    qc.cx(qr_M2[0], qr_GC[0])      # Bookkeeping (feedback if guess was correct)

    # Alice setting:
    if A_setting == 1:
        # Measure F_C directly
        qc.measure(qr_M1[0], cr[0])

    if A_setting == 2:
        # Undo friend, then rotate S_C and measure in computational basis
        qc.cx(qr_M2[0], qr_GC[0])
        qc.cx(qr_SC[0], qr_M2[0])
        qc.ry(-np.pi / 3, qr_SC[0])
        qc.cx(qr_M1[0], qr_GC[0])
        qc.cx(qr_SC[0], qr_M1[0])
        qc.ry(alpha, qr_SC[0])
        qc.measure(qr_SC[0], cr[0])

    # Bob setting:
    if B_setting == 1:
        # Measure S_D directly in basis defined by beta1
        qc.ry(beta1, qr_SD[0])
        qc.measure(qr_SD[0], cr[1])

    if B_setting == 2:
        qc.ry(beta2, qr_SD[0])
        qc.measure(qr_SD[0], cr[1])

    return qc

