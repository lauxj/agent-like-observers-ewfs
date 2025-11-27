import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from sympy.abc import theta

#Build quantum register:
qr1 = QuantumRegister(1, name="S_C")
qr2 = QuantumRegister(1, name="F_C")
qr3 = QuantumRegister(1, name="S_D")
qr4 = QuantumRegister(1, name="F_D")
cr = ClassicalRegister(2, name="classical measurement")


#Build circuit for PEEK:
qc= QuantumCircuit(qr1, qr2, qr3, qr4, cr)
qc.h(0) #Hadamard to 0th qubit
qc.cx(0,2) #CNOT
qc.cx(0,1) #CNOT
qc.cx(2,3)
qc.measure([qr1[0],qr3[0]], cr)
qc.draw("mpl")
plt.show()



#Build circuit for REVERSE:
theta = np.pi/4

qc = QuantumCircuit(qr1, qr2, qr3, qr4, cr)
qc.h(0)
qc.cx(0,2)
qc.cx(0,1)
qc.cx(2,3)
qc.cx(0,1) #UNDO CNOT
qc.cx(2,3) #UNDO CNOT
qc.ry(theta,qr1[0]) #rotation into basis 1
qc.ry(theta,qr3[0]) #rotation into basis 1
qc.measure([qr1[0],qr3[0]], cr)
qc.draw("mpl")
plt.show()
