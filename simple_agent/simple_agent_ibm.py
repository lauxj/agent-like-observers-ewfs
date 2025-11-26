from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator


qr1 = QuantumRegister(1, name="System qubit")
qr2 = QuantumRegister(1, name="Wigner's friend")
cr1 = ClassicalRegister(1, name="Wigner's friend outcome")
cr2 = ClassicalRegister(1, name="Wigner's friend 2 outcome")


qc= QuantumCircuit(qr1, qr2, cr1)
qc.h(0) #Hadamard to 0th qubit
qc.cx(0,1) #CNOT
qc.measure(1,0)
qc.draw("mpl")
#plt.show()


#run the circuit

backend = AerSimulator()
result = backend.run(qc).result()

print("Wigner's friend Counts: ", result.get_counts())
plot_histogram(result.get_counts())
#plt.show()


#implementing wigner
qr3 = QuantumRegister(1, name="Wigner")
cr2 = ClassicalRegister(1, name="Wigner's outcome")

qc2= QuantumCircuit(qr1, qr2,qr3, cr2)
qc2.h(0)
qc2.cx(0,1)
qc2.cx(1,2)
qc2.measure(2,0)

qc2.draw("mpl")
#plt.show()

result = backend.run(qc2).result()

print("Wigner's Counts: ", result.get_counts())
plot_histogram(result.get_counts())
#plt.show()


#friend perspective:
qc3= QuantumCircuit(qr1, qr2, qr3, cr1, cr2)
qc3.h(0)
qc3.cx(0,1)
qc3.measure(qr2,cr1)
qc3.cx(1,2)
qc3.measure(qr3, cr2)

qc3.draw("mpl")
#plt.show()


#wigner perspective:
qc4= QuantumCircuit(qr1, qr2, qr3, cr1, cr2)
qc4.h(0)
qc4.cx(0,1)
qc4.cx(1,2)
qc4.measure(qr2,cr1)
qc4.measure(qr3, cr2)

qc4.draw("mpl")
#plt.show()



#all quantum
qc5= QuantumCircuit(qr1, qr2, qr3)
qc5.h(0)
qc5.cx(0,1)
qc5.cx(1,2)

qc5.draw("mpl")
plt.show()

