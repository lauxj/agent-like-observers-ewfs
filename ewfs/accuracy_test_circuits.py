"""
accuracy_test_circuits.py
Hard-coded EWFS accuracy-test circuits.

For each agent, these circuits keep the original structure but replace the
first S->M memory-write CNOT with a fixed memory preparation in |0> or |1>.
The later conditional branch on c[0] = 1 is removed, while the random choice
measurements and the Bob-side conditional branch are kept.
"""

import math

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

try:
    from .agents import beta1, beta2
except ImportError:
    from agents import beta1, beta2

ACCURACY_TEST_SUFFIX_INIT0 = "_accuracy_test_init0"
ACCURACY_TEST_SUFFIX_INIT1 = "_accuracy_test_init1"
REFLEX_ACCURACY_M_EXTRA_DELAY_DT = 83
GUESSING_ACCURACY_M1_EXTRA_DELAY_DT = 96
BETTING_ACCURACY_M1_EXTRA_DELAY_DT = 115
ALWAYS_LARGE_ACCURACY_M1_EXTRA_DELAY_DT = 121


def _initialize_memory_bit(qc: QuantumCircuit, memory_qubit, init_bit: int) -> None:
    """Prepare the memory qubit in the requested computational-basis state."""
    if init_bit not in {0, 1}:
        raise ValueError(f"init_bit must be 0 or 1, got {init_bit}.")
    if init_bit == 1:
        qc.x(memory_qubit)


def _accuracy_test_label(base_label: str, init_bit: int) -> str:
    return f"{base_label}_accuracy_test_init{init_bit}"


def _accuracy_test_circuit_name(base_name: str, init_bit: int) -> str:
    return f"{base_name}_accuracy_test_init{init_bit}"


def _build_circuit_reflex_accuracy_test(init_bit: int) -> QuantumCircuit:
    qr_SD = QuantumRegister(1, "SD")
    qr_SC = QuantumRegister(1, "SC")
    qr_M = QuantumRegister(1, "M")
    qr_L = QuantumRegister(1, "L")
    qr_A_choice = QuantumRegister(1, "Achoice")
    qr_B_choice = QuantumRegister(1, "Bchoice")
    c = ClassicalRegister(6, "c")

    qc = QuantumCircuit(
        qr_SD, qr_SC, qr_M, qr_L, qr_A_choice, qr_B_choice,
        c,
        name=_accuracy_test_circuit_name("reflex_agent", init_bit),
    )

    qc.h(qr_SC[0])
    qc.cx(qr_SC[0], qr_SD[0])

    qc.barrier(qr_SD, qr_SC, qr_M, qr_L, qr_A_choice, qr_B_choice)
    _initialize_memory_bit(qc, qr_M[0], init_bit)

    qc.barrier(qr_SD, qr_SC, qr_M, qr_L, qr_A_choice, qr_B_choice)
    qc.cx(qr_M[0], qr_L[0])

    qc.barrier(qr_SD, qr_SC, qr_M, qr_L, qr_A_choice, qr_B_choice)
    qc.h(qr_B_choice[0])
    qc.h(qr_A_choice[0])
    qc.measure(qr_A_choice[0], c[0])
    qc.measure(qr_B_choice[0], c[1])

    qc.barrier(qr_SD, qr_SC, qr_M, qr_L, qr_A_choice, qr_B_choice)
    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SD[0])
    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SD[0])

    qc.barrier(qr_SD, qr_SC, qr_M, qr_L, qr_A_choice, qr_B_choice)
    qc.delay(REFLEX_ACCURACY_M_EXTRA_DELAY_DT, qr_M[0], unit="dt")
    qc.measure(qr_M[0], c[2])
    qc.measure(qr_SC[0], c[3])
    qc.measure(qr_SD[0], c[4])
    qc.measure(qr_L[0], c[5])

    return qc


def build_circuit_reflex_accuracy_test_init0() -> QuantumCircuit:
    return _build_circuit_reflex_accuracy_test(init_bit=0)


def build_circuit_reflex_accuracy_test_init1() -> QuantumCircuit:
    return _build_circuit_reflex_accuracy_test(init_bit=1)


def _build_circuit_guessing_accuracy_test(init_bit: int) -> QuantumCircuit:
    qr_SD = QuantumRegister(1, "SD")
    qr_SC = QuantumRegister(1, "SC")
    qr_M1 = QuantumRegister(1, "M1")
    qr_M2 = QuantumRegister(1, "M2")
    qr_G = QuantumRegister(1, "G")
    qr_A_choice = QuantumRegister(1, "Achoice")
    qr_B_choice = QuantumRegister(1, "Bchoice")
    c = ClassicalRegister(7, "c")

    qc = QuantumCircuit(
        qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice,
        c,
        name=_accuracy_test_circuit_name("guessing_agent", init_bit),
    )

    qc.h(qr_SC[0])
    qc.cx(qr_SC[0], qr_SD[0])

    qc.barrier(qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    _initialize_memory_bit(qc, qr_M1[0], init_bit)

    qc.barrier(qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    qc.cx(qr_M1[0], qr_G[0])

    qc.barrier(qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    qc.ry(math.pi / 3, qr_SC[0])
    qc.cx(qr_SC[0], qr_M2[0])

    qc.barrier(qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    qc.h(qr_B_choice[0])
    qc.h(qr_A_choice[0])
    qc.measure(qr_A_choice[0], c[0])
    qc.measure(qr_B_choice[0], c[1])

    qc.barrier(qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SD[0])
    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SD[0])

    qc.barrier(qr_SD, qr_SC, qr_M1, qr_M2, qr_G, qr_A_choice, qr_B_choice)
    qc.delay(GUESSING_ACCURACY_M1_EXTRA_DELAY_DT, qr_M1[0], unit="dt")
    qc.measure(qr_M1[0], c[2])
    qc.measure(qr_SC[0], c[3])
    qc.measure(qr_SD[0], c[4])
    qc.measure(qr_M2[0], c[5])
    qc.measure(qr_G[0], c[6])

    return qc


def build_circuit_guessing_accuracy_test_init0() -> QuantumCircuit:
    return _build_circuit_guessing_accuracy_test(init_bit=0)


def build_circuit_guessing_accuracy_test_init1() -> QuantumCircuit:
    return _build_circuit_guessing_accuracy_test(init_bit=1)


def _build_circuit_betting_accuracy_test(init_bit: int) -> QuantumCircuit:
    qr_SD = QuantumRegister(1, "SD")
    qr_SC = QuantumRegister(1, "SC")
    qr_M1 = QuantumRegister(1, "M1")
    qr_M2 = QuantumRegister(1, "M2")
    qr_W0 = QuantumRegister(1, "W0")
    qr_W1 = QuantumRegister(1, "W1")
    qr_A_choice = QuantumRegister(1, "Achoice")
    qr_B_choice = QuantumRegister(1, "Bchoice")
    c = ClassicalRegister(8, "c")

    qc = QuantumCircuit(
        qr_SD, qr_SC, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice,
        c,
        name=_accuracy_test_circuit_name("betting_agent", init_bit),
    )

    qc.h(qr_SC[0])
    qc.cx(qr_SC[0], qr_SD[0])
    qc.x(qr_W0[0])

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    _initialize_memory_bit(qc, qr_M1[0], init_bit)

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.cx(qr_M1[0], qr_W0[0])

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.ry(math.pi / 3, qr_SC[0])
    qc.cx(qr_SC[0], qr_M2[0])

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.cx(qr_M2[0], qr_W1[0])

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.h(qr_A_choice[0])
    qc.measure(qr_A_choice[0], c[0])
    qc.h(qr_B_choice[0])
    qc.measure(qr_B_choice[0], c[1])

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SD[0])
    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SD[0])

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.delay(BETTING_ACCURACY_M1_EXTRA_DELAY_DT, qr_M1[0], unit="dt")
    qc.measure(qr_M1[0], c[2])
    qc.measure(qr_SC[0], c[3])
    qc.measure(qr_SD[0], c[4])
    qc.measure(qr_W1[0], c[7])
    qc.measure(qr_W0[0], c[6])
    qc.measure(qr_M2[0], c[5])

    return qc


def build_circuit_betting_accuracy_test_init0() -> QuantumCircuit:
    return _build_circuit_betting_accuracy_test(init_bit=0)


def build_circuit_betting_accuracy_test_init1() -> QuantumCircuit:
    return _build_circuit_betting_accuracy_test(init_bit=1)


def _build_circuit_always_large_accuracy_test(init_bit: int) -> QuantumCircuit:
    qr_SD = QuantumRegister(1, "SD")
    qr_SC = QuantumRegister(1, "SC")
    qr_M1 = QuantumRegister(1, "M1")
    qr_M2 = QuantumRegister(1, "M2")
    qr_W0 = QuantumRegister(1, "W0")
    qr_W1 = QuantumRegister(1, "W1")
    qr_A_choice = QuantumRegister(1, "Achoice")
    qr_B_choice = QuantumRegister(1, "Bchoice")
    c = ClassicalRegister(8, "c")

    qc = QuantumCircuit(
        qr_SD, qr_SC, qr_M1, qr_M2, qr_W0, qr_W1, qr_A_choice, qr_B_choice,
        c,
        name=_accuracy_test_circuit_name("always_large_agent", init_bit),
    )

    qc.h(qr_SC[0])
    qc.cx(qr_SC[0], qr_SD[0])
    qc.x(qr_W0[0])

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    _initialize_memory_bit(qc, qr_M1[0], init_bit)

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.x(qr_W0[0])

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.ry(math.pi / 3, qr_SC[0])
    qc.cx(qr_SC[0], qr_M2[0])

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.cx(qr_M2[0], qr_W1[0])

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.h(qr_A_choice[0])
    qc.measure(qr_A_choice[0], c[0])
    qc.h(qr_B_choice[0])
    qc.measure(qr_B_choice[0], c[1])

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    with qc.if_test((c[1], 0)):
        qc.ry(beta1, qr_SD[0])
    with qc.if_test((c[1], 1)):
        qc.ry(beta2, qr_SD[0])

    qc.barrier(qr_SC[0], qr_SD[0], qr_M1[0], qr_M2[0], qr_W0[0], qr_W1[0], qr_A_choice[0], qr_B_choice[0])
    qc.delay(ALWAYS_LARGE_ACCURACY_M1_EXTRA_DELAY_DT, qr_M1[0], unit="dt")
    qc.measure(qr_M1[0], c[2])
    qc.measure(qr_SC[0], c[3])
    qc.measure(qr_SD[0], c[4])
    qc.measure(qr_W1[0], c[7])
    qc.measure(qr_W0[0], c[6])
    qc.measure(qr_M2[0], c[5])

    return qc


def build_circuit_always_large_accuracy_test_init0() -> QuantumCircuit:
    return _build_circuit_always_large_accuracy_test(init_bit=0)


def build_circuit_always_large_accuracy_test_init1() -> QuantumCircuit:
    return _build_circuit_always_large_accuracy_test(init_bit=1)


ACCURACY_TEST_BUILDERS = [
    (_accuracy_test_label("Reflex Agent", 0), build_circuit_reflex_accuracy_test_init0),
    (_accuracy_test_label("Reflex Agent", 1), build_circuit_reflex_accuracy_test_init1),
    (_accuracy_test_label("Guessing Agent", 0), build_circuit_guessing_accuracy_test_init0),
    (_accuracy_test_label("Guessing Agent", 1), build_circuit_guessing_accuracy_test_init1),
    (_accuracy_test_label("Betting Agent", 0), build_circuit_betting_accuracy_test_init0),
    (_accuracy_test_label("Betting Agent", 1), build_circuit_betting_accuracy_test_init1),
    (_accuracy_test_label("Always 3/4 Agent", 0), build_circuit_always_large_accuracy_test_init0),
    (_accuracy_test_label("Always 3/4 Agent", 1), build_circuit_always_large_accuracy_test_init1),
]
