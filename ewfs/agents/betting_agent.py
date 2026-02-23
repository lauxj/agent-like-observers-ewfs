import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


# Build circuit for betting agent:
def build_measurement(A_setting, B_setting, alpha, beta1, beta2):
    """Builds the circuit for the betting agent for settings and angles given to the function"""

    # quantum registers
    qr_SD = QuantumRegister(1, "S_D")
    qr_SC = QuantumRegister(1, "S_C")
    qr_M1 = QuantumRegister(1, "M1")
    qr_M2 = QuantumRegister(1, "M2")
    qr_W0 = QuantumRegister(1, "W_0")
    qr_W1 = QuantumRegister(1, "W_1")
    cr = ClassicalRegister(2, "c")

    qc = QuantumCircuit( qr_SD, qr_SC, qr_M1, qr_M2, qr_W0, qr_W1, cr)

    # PRE-MEASUREMENT
    qc.h(qr_SC[0])                 # create |+> on S_C
    qc.cx(qr_SC[0], qr_SD[0])      # entangle → Bell pair
    qc.x(qr_W0[0])                  # Initialize Wallet W0 = 1
    qc.cx(qr_SC[0], qr_M1[0])      # Charlie pre-measures S_C (store in M1)
    qc.cx(qr_M1[0], qr_W0[0])      # Bet bookkeeping in Wallet
    qc.ry(np.pi/3, qr_SC[0])       # Rotation on SC
    qc.cx(qr_SC[0], qr_M2[0])      # Charlie measures SC again after rotation (store in M2)
    qc.cx(qr_M2[0], qr_W1[0])      # Bookkeeping of wallet after the bet


    # Alice setting:
    if A_setting == 1:
        # Measure F_C directly
        qc.measure(qr_M1[0], cr[0])

    if A_setting == 2:
        # Undo Charlie, then rotate S_C and measure in computational basis
        qc.cx(qr_M2[0], qr_W1[0])
        qc.cx(qr_SC[0], qr_M2[0])
        qc.ry(-np.pi / 3, qr_SC[0])
        qc.cx(qr_M1[0], qr_W0[0])
        qc.cx(qr_SC[0], qr_M1[0])
        qc.x(qr_W0[0])


        qc.ry(alpha, qr_SC[0])
        qc.measure(qr_SC[0], cr[0])

    # Bob setting:
    if B_setting == 1:
        # Measure S_D directly in rotated basis beta1
        qc.ry(beta1, qr_SD[0])
        qc.measure(qr_SD[0], cr[1])

    if B_setting == 2:
        qc.ry(beta2, qr_SD[0])
        qc.measure(qr_SD[0], cr[1])

    return qc
