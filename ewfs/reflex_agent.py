from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


# Build circuit for reflex agent:
def build_measurement(A_setting, B_setting, alpha, beta1, beta2):
    """Builds the circuit for the reflex agent for settings and angles given to the function"""

    # quantum registers
    qr_SC = QuantumRegister(1, "S_C")
    qr_LC = QuantumRegister(1, "L")
    qr_FC = QuantumRegister(1, "M")
    qr_SD = QuantumRegister(1, "S_D")
    cr = ClassicalRegister(2, "c")         # store A and B outcomes

    qc = QuantumCircuit(qr_SD, qr_SC, qr_FC, qr_LC,  cr)

    # PRE-MEASUREMENT
    qc.h(qr_SC[0])                 # create |+> on S_C
    qc.cx(qr_SC[0], qr_SD[0])      # entangle → Bell pair
    qc.cx(qr_SC[0], qr_FC[0])      # Charlie pre-measures S_C
    qc.cx(qr_FC[0], qr_LC[0])      # turn light switch on


    # Alice setting:
    if A_setting == 1:
        # Measure F_C directly
        qc.measure(qr_FC[0], cr[0])

    if A_setting == 2:
        # Undo Charlie, then rotate S_C and measure in computational basis
        qc.cx(qr_FC[0], qr_LC[0])
        qc.cx(qr_SC[0], qr_FC[0])
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


