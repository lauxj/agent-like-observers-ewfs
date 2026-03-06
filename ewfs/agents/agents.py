import qiskit
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# optimal angles
alpha = 3 * np.pi / 2
beta1 = 3 * np.pi / 4
beta2 = 1 * np.pi / 4




def build_circuit_betting() -> QuantumCircuit:
    """Betting agent."""

    # Quantum registers
    qr_SD = QuantumRegister(1, "SD")
    qr_SC = QuantumRegister(1, "SC")
    qr_M1 = QuantumRegister(1, "M1")
    qr_M2 = QuantumRegister(1, "M2")
    qr_W0 = QuantumRegister(1, "W0")
    qr_W1 = QuantumRegister(1, "W1")
    qr_A_choice = QuantumRegister(1, "Achoice")
    qr_B_choice = QuantumRegister(1, "Bchoice")

    # Classical register (joint counts)
    c = ClassicalRegister(5, "c") # A_choice, B_choice, A_record (Arec), B_meas

    #cA = ClassicalRegister(1, "cA")
    #cB = ClassicalRegister(1, "cB")
    #cM1Dump = ClassicalRegister(1, "cM1Dump")
    #cSCDump = ClassicalRegister(1, "cSCDump")
    #cBr = ClassicalRegister(1, "cBr")

    qc = QuantumCircuit(
        qr_SD, qr_SC, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice,
        c,
        name="betting_agent",
    )


    qc.h(qr_SC[0])
    qc.h(qr_A_choice[0])
    qc.h(qr_B_choice[0])

    qc.x(qr_W0[0])

    qc.cx(qr_SC[0], qr_SD[0])

    qc.measure(qr_A_choice[0], c[0])
    qc.measure(qr_B_choice[0], c[1])

    qc.cx(qr_SC[0], qr_M1[0])

    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SD[0])

    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SD[0])

    qc.ry(np.pi / 3, qr_SC[0])

    qc.cx(qr_M1[0], qr_W0[0])

    qc.measure(qr_SD[0], c[4])

    qc.cx(qr_SC[0], qr_M2[0])

    qc.cx(qr_M2[0], qr_W1[0])

    with qc.if_test((c[0], 1)):
        qc.cx(qr_M2[0], qr_W1[0])
        qc.cx(qr_SC[0], qr_M2[0])
        qc.ry(-np.pi / 3, qr_SC[0])
        qc.cx(qr_M1[0], qr_W0[0])
        qc.cx(qr_SC[0], qr_M1[0])
        qc.x(qr_W0[0])
        qc.ry(alpha, qr_SC[0])

    qc.measure(qr_M1[0], c[2])
    qc.measure(qr_SC[0], c[3])

    return qc


def build_circuit_guessing() -> QuantumCircuit:
    """Guessing agent."""

    qr_SD = QuantumRegister(1, "SD")
    qr_SC = QuantumRegister(1, "SC")
    qr_M1 = QuantumRegister(1, "M1")
    qr_M2 = QuantumRegister(1, "M2")
    qr_G = QuantumRegister(1, "G")
    qr_A_choice = QuantumRegister(1, "Achoice")
    qr_B_choice = QuantumRegister(1, "Bchoice")

    c = ClassicalRegister(5, "c")

    qc = QuantumCircuit(
        qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice,
        c,
        name="guessing_agent",
    )

    qc.h(qr_SC[0])
    qc.h(qr_A_choice[0])
    qc.h(qr_B_choice[0])

    qc.cx(qr_SC[0], qr_SD[0])

    qc.measure(qr_A_choice[0], c[0])

    qc.cx(qr_SC[0], qr_M1[0])

    qc.measure(qr_B_choice[0], c[1])

    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SD[0])

    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SD[0])

    qc.cx(qr_M1[0], qr_G[0])

    qc.ry(np.pi / 3, qr_SC[0])

    qc.cx(qr_SC[0], qr_M2[0])

    qc.measure(qr_SD[0], c[4])

    qc.cx(qr_M2[0], qr_G[0])

    with qc.if_test((c[0], 1)):
        qc.cx(qr_M2[0], qr_G[0])
        qc.cx(qr_SC[0], qr_M2[0])
        qc.ry(-np.pi / 3, qr_SC[0])
        qc.cx(qr_M1[0], qr_G[0])
        qc.cx(qr_SC[0], qr_M1[0])
        qc.ry(alpha, qr_SC[0])

    qc.measure(qr_M1[0], c[2])
    qc.measure(qr_SC[0], c[3])

    return qc


def build_circuit_reflex() -> QuantumCircuit:
    """Reflex agent."""

    qr_SD = QuantumRegister(1, "SD")
    qr_SC = QuantumRegister(1, "SC")
    qr_M = QuantumRegister(1, "M")
    qr_L = QuantumRegister(1, "L")
    qr_A_choice = QuantumRegister(1, "Achoice")
    qr_B_choice = QuantumRegister(1, "Bchoice")

    c = ClassicalRegister(5, "c")

    qc = QuantumCircuit(
        qr_SD, qr_SC, qr_M, qr_L, qr_A_choice, qr_B_choice,
        c,
        name="reflex_agent",
    )

    qc.h(qr_B_choice[0])
    qc.h(qr_SC[0])
    qc.h(qr_A_choice[0])

    qc.measure(qr_A_choice[0], c[0])

    qc.cx(qr_SC[0], qr_SD[0])

    qc.measure(qr_B_choice[0], c[1])

    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SD[0])

    qc.cx(qr_SC[0], qr_M[0])

    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SD[0])

    qc.cx(qr_M[0], qr_L[0])

    qc.measure(qr_SD[0], c[4])

    with qc.if_test((c[0], 1)):
        qc.cx(qr_M[0], qr_L[0])
        qc.cx(qr_SC[0], qr_M[0])
        qc.ry(alpha, qr_SC[0])

    qc.measure(qr_M[0], c[2])
    qc.measure(qr_SC[0], c[3])

    return qc
