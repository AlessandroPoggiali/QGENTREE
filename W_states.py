from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from numpy import sqrt, arccos, arcsin
import numpy as np
import matplotlib.pyplot as plt
import math as m
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.visualization import plot_histogram

# https://quantumcomputing.stackexchange.com/questions/15506/how-to-implement-a-circuit-preparing-the-three-qubit-w-state
def _w_state_circuit(circuit, nn, mm, qubitC):    
    global qubitT #reference to global variable qubitT
    
    if nn == 0 and mm == 1:
        pass #do nothing in this case
    elif nn == 1 and mm == 2: #case (1,2)
        circuit.cu(theta=m.pi/2, phi=0, lam=0, gamma=0, control_qubit=q[qubitC], target_qubit=q[qubitT])
        circuit.cx(q[qubitT], q[qubitC])
        qubitT = qubitT + 1
    else: #otherwise
        theta = 2*np.arccos(m.sqrt(nn/mm))
        circuit.cu(theta=theta, phi=0, lam=0, gamma=0, control_qubit=q[qubitC], target_qubit=q[qubitT])
        circuit.cx(q[qubitT], q[qubitC])
        
        qubitTRecurse = qubitT #saving target qubit index, used as control qubit for lower child
        qubitT = qubitT + 1
        
        a = m.floor(nn/2)
        b = m.floor(mm/2)
        c = m.ceil(nn/2)
        d = m.ceil(mm/2)
        
        if a == 1 and b == 1: #upper child (1,1) => (1,2) became upper child
            circuit = _w_state_circuit(circuit, 1, 2, qubitC)
            #there is no lower child
        elif c == 1 and d == 1: #lower child (1,1) => (1,2) became lower child
            circuit = _w_state_circuit(circuit, 1, 2, qubitTRecurse)
            #there is no upper child
        else:                       
            #upper child
            circuit = _w_state_circuit(circuit, a, b, qubitC)                 
            #lower child
            circuit = _w_state_circuit(circuit, c, d, qubitTRecurse)
                
    return circuit
        
def w_state_circuit (qubits, qRegister, cRegister):
    global qubitT
    qubitT = 1 #index of a qubit a new gate acts on (hard to compute inside recursion => global variable)
    circuit = QuantumCircuit(qRegister, cRegister)
    circuit.x(q[0])
    circuit = _w_state_circuit(circuit, m.floor(qubits/2), qubits, 0)
    return circuit

#construction of 6 qubits W-state
qubits = 3

q = QuantumRegister(qubits, name = 'q')
c = ClassicalRegister(qubits, name = 'c')

circuit = w_state_circuit(qubits, q, c)

circuit.draw('mpl')
plt.show()


circuit.measure(q,c)
simulator = AerSimulator()  # Create a simulator backend
result = simulator.run(circuit, shots=4096).result()  # Run the circuit and get the result
counts = result.get_counts()
plot_histogram(counts)
plt.show()
