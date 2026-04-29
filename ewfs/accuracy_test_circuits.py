"""
accuracy_test_circuits.py
Builds the relaxed LF accuracy-test circuits for all agents

The  circuits are the same idea as the main agent circuits, but
Charlie's first memory bit is fixed to |0> or |1> instead of being written from
the system. This lets us test how accurately the hardware preserves and reads
the memory bit, ie P(c=a|x=1)
"""

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

# Optimal angles:
alpha = 3 * np.pi / 2
beta1 = 3 * np.pi / 4
beta2 = 1 * np.pi / 4

# extra delay to have a strictly longer circuit duration compared to the original circuits (see paper)
REFLEX_DELAY = 83
GUESSING_DELAY = 96
BETTING_DELAY = 115
ALWAYS_LARGE_DELAY = 121


def _build_circuit_reflex_accuracy_test(init_bit: int) -> QuantumCircuit:
    """Reflex agent accuracy-test circuit builder."""

    # Quantum registers:
    qr_SB = QuantumRegister(1, "SB")
    qr_SA = QuantumRegister(1, "SA")
    qr_M = QuantumRegister(1, "M")
    qr_R = QuantumRegister(1, "R")
    qr_A_choice = QuantumRegister(1, "Achoice")
    qr_B_choice = QuantumRegister(1, "Bchoice")

    # Classical register:
    c = ClassicalRegister(6, "c")

    qc = QuantumCircuit(
        qr_SB, qr_SA, qr_M, qr_R, qr_A_choice, qr_B_choice,
        c,
        name=f"reflex_agent_accuracy_test_init{init_bit}",
    )

    # Initialization:
    qc.h(qr_SA[0])
    qc.cx(qr_SA[0], qr_SB[0])

    # Fixed memory preparation:
    qc.barrier(qr_SB, qr_SA, qr_M, qr_R, qr_A_choice, qr_B_choice)
    if init_bit == 1:
        qc.x(qr_M[0])

    # Charlie's action:
    qc.barrier(qr_SB, qr_SA, qr_M, qr_R, qr_A_choice, qr_B_choice)
    qc.cx(qr_M[0], qr_R[0])

    # Alice's and Bob's choice (x and y):
    qc.barrier(qr_SB, qr_SA, qr_M, qr_R, qr_A_choice, qr_B_choice)
    qc.h(qr_B_choice[0])
    qc.h(qr_A_choice[0])
    qc.measure(qr_A_choice[0], c[0])
    qc.measure(qr_B_choice[0], c[1])

    # Bob's setting B1:
    qc.barrier(qr_SB, qr_SA, qr_M, qr_R, qr_A_choice, qr_B_choice)
    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SB[0])
    # Bob's setting B2:
    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SB[0])

    # Alice's and Bob's measurements:
    qc.barrier(qr_SB, qr_SA, qr_M, qr_R, qr_A_choice, qr_B_choice)
    qc.delay(REFLEX_DELAY, qr_M[0], unit="dt")
    qc.measure(qr_M[0], c[2])
    qc.measure(qr_SA[0], c[3])
    qc.measure(qr_SB[0], c[4])
    qc.measure(qr_R[0], c[5])

    return qc


def build_circuit_reflex_accuracy_test_init0() -> QuantumCircuit:
    return _build_circuit_reflex_accuracy_test(init_bit=0)


def build_circuit_reflex_accuracy_test_init1() -> QuantumCircuit:
    return _build_circuit_reflex_accuracy_test(init_bit=1)


def _build_circuit_guessing_accuracy_test(init_bit) -> QuantumCircuit:
    """Guessing agent accuracy-test circuit builder."""

    # Quantum registers:
    qr_SB = QuantumRegister(1, "SB")
    qr_SA = QuantumRegister(1, "SA")
    qr_M1 = QuantumRegister(1, "M1")
    qr_M2 = QuantumRegister(1, "M2")
    qr_G = QuantumRegister(1, "G")
    qr_A_choice = QuantumRegister(1, "Achoice")
    qr_B_choice = QuantumRegister(1, "Bchoice")

    # Classical register:
    c = ClassicalRegister(7, "c")

    qc = QuantumCircuit(
        qr_SB, qr_SA, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice,
        c,
        name=f"guessing_agent_accuracy_test_init{init_bit}",
    )

    # Initialization:
    qc.h(qr_SA[0])
    qc.cx(qr_SA[0], qr_SB[0])

    # Fixed memory preparation:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    if init_bit == 1:
        qc.x(qr_M1[0])

    # Charlie's guess:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    qc.cx(qr_M1[0], qr_G[0])

    # Charlie's second measurement in rotated basis:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    qc.ry(np.pi / 3, qr_SA[0])
    qc.cx(qr_SA[0], qr_M2[0])

    # Alice's and Bob's choice (x and y):
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    qc.h(qr_B_choice[0])
    qc.h(qr_A_choice[0])
    qc.measure(qr_A_choice[0], c[0])
    qc.measure(qr_B_choice[0], c[1])

    # Bob's setting B1:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SB[0])
    # Bob's setting B2:
    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SB[0])

    # Alice's and Bob's measurements:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    qc.delay(GUESSING_DELAY, qr_M1[0], unit="dt")
    qc.measure(qr_M1[0], c[2])
    qc.measure(qr_SA[0], c[3])
    qc.measure(qr_SB[0], c[4])
    qc.measure(qr_M2[0], c[5])
    qc.measure(qr_G[0], c[6])

    return qc


def build_circuit_guessing_accuracy_test_init0() -> QuantumCircuit:
    return _build_circuit_guessing_accuracy_test(init_bit=0)


def build_circuit_guessing_accuracy_test_init1() -> QuantumCircuit:
    return _build_circuit_guessing_accuracy_test(init_bit=1)


def _build_circuit_betting_accuracy_test(init_bit) -> QuantumCircuit:
    """Betting agent accuracy-test circuit builder."""

    # Quantum registers:
    qr_SB = QuantumRegister(1, "SB")
    qr_SA = QuantumRegister(1, "SA")
    qr_M1 = QuantumRegister(1, "M1")
    qr_M2 = QuantumRegister(1, "M2")
    qr_W0 = QuantumRegister(1, "W0")
    qr_W1 = QuantumRegister(1, "W1")
    qr_A_choice = QuantumRegister(1, "Achoice")
    qr_B_choice = QuantumRegister(1, "Bchoice")

    # Classical register:
    c = ClassicalRegister(8, "c")

    qc = QuantumCircuit(
        qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice,
        c,
        name=f"betting_agent_accuracy_test_init{init_bit}",
    )

    # Initialization:
    qc.h(qr_SA[0])
    qc.cx(qr_SA[0], qr_SB[0])
    qc.x(qr_W0[0])

    # Fixed memory preparation:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    if init_bit == 1:
        qc.x(qr_M1[0])

    # Charlie's first wallet update:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    qc.cx(qr_M1[0], qr_W0[0])

    # Charlie's second measurement in rotated basis:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    qc.ry(np.pi / 3, qr_SA[0])
    qc.cx(qr_SA[0], qr_M2[0])

    # Charlie's second wallet update:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    qc.cx(qr_M2[0], qr_W1[0])

    # Alice's and Bob's choice (x and y):
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    qc.h(qr_A_choice[0])
    qc.measure(qr_A_choice[0], c[0])
    qc.h(qr_B_choice[0])
    qc.measure(qr_B_choice[0], c[1])

    # Bob's setting B1:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SB[0])
    # Bob's setting B2:
    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SB[0])

    # Alice's and Bob's measurements:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    qc.delay(BETTING_DELAY, qr_M1[0], unit="dt")
    qc.measure(qr_M1[0], c[2])
    qc.measure(qr_SA[0], c[3])
    qc.measure(qr_SB[0], c[4])

    # Other measurements:
    qc.measure(qr_W1[0], c[7])
    qc.measure(qr_W0[0], c[6])
    qc.measure(qr_M2[0], c[5])

    return qc


def build_circuit_betting_accuracy_test_init0() -> QuantumCircuit:
    return _build_circuit_betting_accuracy_test(init_bit=0)


def build_circuit_betting_accuracy_test_init1() -> QuantumCircuit:
    return _build_circuit_betting_accuracy_test(init_bit=1)


def _build_circuit_always_large_accuracy_test(init_bit) -> QuantumCircuit:
    """Always-3/4 agent accuracy-test circuit builder."""

    # Quantum registers:
    qr_SB = QuantumRegister(1, "SB")
    qr_SA = QuantumRegister(1, "SA")
    qr_M1 = QuantumRegister(1, "M1")
    qr_M2 = QuantumRegister(1, "M2")
    qr_W0 = QuantumRegister(1, "W0")
    qr_W1 = QuantumRegister(1, "W1")
    qr_A_choice = QuantumRegister(1, "Achoice")
    qr_B_choice = QuantumRegister(1, "Bchoice")

    # Classical register:
    c = ClassicalRegister(8, "c")

    qc = QuantumCircuit(
        qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice,
        c,
        name=f"always_large_agent_accuracy_test_init{init_bit}",
    )

    # Initialization:
    qc.h(qr_SA[0])
    qc.cx(qr_SA[0], qr_SB[0])
    qc.x(qr_W0[0])

    # Fixed memory preparation:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    if init_bit == 1:
        qc.x(qr_M1[0])

    # Charlie always places the 3/4 bet:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    qc.x(qr_W0[0])

    # Charlie's second measurement in rotated basis:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    qc.ry(np.pi / 3, qr_SA[0])
    qc.cx(qr_SA[0], qr_M2[0])

    # Charlie's second wallet update:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    qc.cx(qr_M2[0], qr_W1[0])

    # Alice's and Bob's choice (x and y):
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    qc.h(qr_A_choice[0])
    qc.measure(qr_A_choice[0], c[0])
    qc.h(qr_B_choice[0])
    qc.measure(qr_B_choice[0], c[1])

    # Bob's setting B1:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SB[0])
    # Bob's setting B2:
    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SB[0])

    # Alice's and Bob's measurements:
    qc.barrier(qr_SB, qr_SA, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice)
    qc.delay(ALWAYS_LARGE_DELAY, qr_M1[0], unit="dt")
    qc.measure(qr_M1[0], c[2])
    qc.measure(qr_SA[0], c[3])
    qc.measure(qr_SB[0], c[4])

    # Other measurements:
    qc.measure(qr_W1[0], c[7])
    qc.measure(qr_W0[0], c[6])
    qc.measure(qr_M2[0], c[5])

    return qc


def build_circuit_always_large_accuracy_test_init0() -> QuantumCircuit:
    return _build_circuit_always_large_accuracy_test(init_bit=0)


def build_circuit_always_large_accuracy_test_init1() -> QuantumCircuit:
    return _build_circuit_always_large_accuracy_test(init_bit=1)

# builders for all accuracy-test circuits, to be used in the main script for running the tests on hardware
ACCURACY_TEST_BUILDERS = [
    ("Reflex Agent_accuracy_test_init0", build_circuit_reflex_accuracy_test_init0),
    ("Reflex Agent_accuracy_test_init1", build_circuit_reflex_accuracy_test_init1),
    ("Guessing Agent_accuracy_test_init0", build_circuit_guessing_accuracy_test_init0),
    ("Guessing Agent_accuracy_test_init1", build_circuit_guessing_accuracy_test_init1),
    ("Betting Agent_accuracy_test_init0", build_circuit_betting_accuracy_test_init0),
    ("Betting Agent_accuracy_test_init1", build_circuit_betting_accuracy_test_init1),
    ("Always 3/4 Agent_accuracy_test_init0", build_circuit_always_large_accuracy_test_init0),
    ("Always 3/4 Agent_accuracy_test_init1", build_circuit_always_large_accuracy_test_init1),
]
