"""
agents.py
Builds all quantum circuits for the different agents using qiskit
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# optimal angles
alpha = 3 * np.pi / 2
beta1 = 3 * np.pi / 4
beta2 = 1 * np.pi / 4

def build_circuit_betting() -> QuantumCircuit:
    """Betting agent circuit builder."""

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
    qc = QuantumCircuit(
        qr_SD, qr_SC, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice,
        c,
        name="betting_agent",
    )

    # Create entangled pair between both labs:
    qc.h(qr_SC[0])
    qc.cx(qr_SC[0], qr_SD[0])
    # wallet initialization
    qc.x(qr_W0[0])

    # Charlie's 1st measurement:
    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.cx(qr_SC[0], qr_M1[0])

    # Charlies bet: wallet update
    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.cx(qr_M1[0], qr_W0[0]) #Charlie placing the bet

    #
    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.ry(np.pi / 3, qr_SC[0]) # Rotation on system for second measurement of Charlie
    qc.cx(qr_SC[0], qr_M2[0]) # Charlie's second measurement on rotated system

    # Wallet update
    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.cx(qr_M2[0], qr_W1[0]) # Wallet update based on outcome of second measurement

    # Alice's choice:
    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.h(qr_A_choice[0])
    qc.measure(qr_A_choice[0], c[0])

    # Conditional block for setting A=1 (Undo and measure system):
    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    with qc.if_test((c[0], 1)):
        qc.cx(qr_M2[0], qr_W1[0])
        qc.cx(qr_SC[0], qr_M2[0])
        qc.ry(-np.pi / 3, qr_SC[0])
        qc.cx(qr_M1[0], qr_W0[0])
        qc.cx(qr_SC[0], qr_M1[0])
        qc.x(qr_W0[0])
        qc.ry(alpha, qr_SC[0])

    # Alice's measurement(s)
    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.measure(qr_M1[0], c[2]) # Alice measurement for setting A1 (ask Charlie)
    qc.measure(qr_SC[0], c[3]) # Alice measurement for setting A2 (Undo)

    # Bob's choice operation
    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.h(qr_B_choice[0])
    qc.measure(qr_B_choice[0], c[1])

    # Bob's measurements conditioned on setting:
    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    with qc.if_test((c[4], 0)): # setting B=1
        qc.ry(beta1, qr_SD[0])
    with qc.if_test((c[4], 1)): # setting B=2
        qc.ry(beta2, qr_SD[0])

    # Bob's measurement
    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.measure(qr_SD[0], c[4])

    return qc


def build_circuit_guessing() -> QuantumCircuit:
    """Guessing agent circuit builder."""

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

    # Initialization
    qc.h(qr_SC[0])
    qc.cx(qr_SC[0], qr_SD[0])

    # Charlie's measurement
    qc.barrier(qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)

    qc.h(qr_B_choice[0])
    qc.h(qr_A_choice[0])
    qc.barrier(qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)

    qc.cx(qr_SC[0], qr_M1[0])
    qc.measure(qr_A_choice[0], c[0])
    qc.measure(qr_B_choice[0], c[1])


    # Charlie's guess
    qc.barrier(qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    qc.cx(qr_M1[0], qr_G[0])

    # Charlie's 2nd measurement in rotated basis
    qc.barrier(qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    qc.ry(np.pi / 3, qr_SC[0])
    qc.cx(qr_SC[0], qr_M2[0])

    # Alice's choice
    qc.barrier(qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)

    with qc.if_test((c[0], 1)):
        qc.cx(qr_SC[0], qr_M2[0])
        qc.ry(-np.pi / 3, qr_SC[0])
        qc.cx(qr_M1[0], qr_G[0])
        qc.cx(qr_SC[0], qr_M1[0])
        qc.ry(alpha, qr_SC[0])

    # Final measurements at the end.



    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SD[0])

    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SD[0])

    qc.measure(qr_M1[0], c[2])
    qc.measure(qr_SC[0], c[3])
    qc.measure(qr_SD[0], c[4])

    return qc


def build_circuit_reflex() -> QuantumCircuit:
    """Reflex agent circuit builder."""

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
    qc.cx(qr_SC[0], qr_SD[0])
    qc.cx(qr_SC[0], qr_M[0])
    qc.barrier(qr_SC[0], qr_M[0], qr_L[0], qr_B_choice[0])
    qc.cx(qr_M[0], qr_L[0])

    with qc.if_test((c[0], 1)):
        qc.cx(qr_M[0], qr_L[0])
        qc.cx(qr_SC[0], qr_M[0])
        qc.ry(alpha, qr_SC[0])

    # Final measurements at the end.
    qc.measure(qr_A_choice[0], c[0])
    qc.measure(qr_B_choice[0], c[1])

    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SD[0])

    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SD[0])

    qc.measure(qr_M[0], c[2])
    qc.measure(qr_SC[0], c[3])
    qc.measure(qr_SD[0], c[4])

    return qc
