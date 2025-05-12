from qlasskit import qlassf
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import XGate, HGate, ZGate, CZGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.compiler import transpile
import math



def bin_value_gate(value, bin_size):
    gatebin = QuantumCircuit(bin_size, name=value)
    for i in range(bin_size):
        if value[i] == '1':
            gatebin.append(XGate(), [i])
    return gatebin


def encode_dataset(dataset):

    # should be a dataset of permutations of the same length
    n_samples = len(dataset)
    bin_size = len(dataset[0])

    assert (n_samples & (n_samples - 1)
            ) == 0, "n_samples must be a power of two"

    n_addr_samples = int(math.log2(n_samples))

    print("n_addr_samples:", n_addr_samples)
    print("bin_size:", bin_size)

    qaddr_samples = QuantumRegister(n_addr_samples, name="addr_samples")
    qbinvalue = QuantumRegister(bin_size, name="binval")
    qc = QuantumCircuit(qaddr_samples,
                        qbinvalue, name="encode")

    for qs in qaddr_samples:
        qc.append(HGate(), [qs])

    count = 0
    for i, sample in enumerate(dataset):
        print(f"i: {i}, sample: {sample}")
        controlled_bin_value_gate = bin_value_gate(sample, bin_size).control(
            num_ctrl_qubits=len(qaddr_samples), ctrl_state=count)
        count += 1
        qc.append(controlled_bin_value_gate, qaddr_samples[:]+qbinvalue[:])
        print()
        qc.barrier()

    for qf in qaddr_samples:
        qc.append(HGate(), [qf])

    return qc


@qlassf
def tree(f0: bool, f1: bool, f2: bool, l: bool) -> bool:
    def xnor(a: bool, b: bool) -> bool:
        return (a and b) or (not a and not b)

    right = xnor(f0, True) and ((xnor(f2, False) and xnor(l, True))
                                or (xnor(f2, True)) and xnor(l, False))

    left = xnor(f0, False) and ((xnor(f1, False) and xnor(l, False))
                                or (xnor(f1, True)) and xnor(l, True))

    res = right or left
    return res


def classical_tree(f0: bool, f1: bool, f2: bool, l: bool) -> bool:
    def xnor(a: bool, b: bool) -> bool:
        return (a and b) or (not a and not b)

    right = xnor(f0, True) and ((xnor(f2, False) and xnor(l, True))
                                or (xnor(f2, True)) and xnor(l, False))

    left = xnor(f0, False) and ((xnor(f1, False) and xnor(l, False))
                                or (xnor(f1, True)) and xnor(l, True))

    res = right or left
    return res


def oracle(dataset):
    qtree = tree.circuit().export('circuit', 'qiskit')
    qdb = encode_dataset(dataset)
    q_index = QuantumRegister(int(math.log2(len(dataset))))
    q_checker = QuantumRegister(1)
    q = QuantumRegister(qtree.num_qubits)
    oracle = QuantumCircuit(q_index, q, q_checker, name="oracle")
    oracle.append(qdb, q_index[:]+q[:4])
    oracle.append(qtree, q[:])
    oracle.append(CZGate(), [q[-1], q_checker[0]])
    oracle.append(qtree.inverse(), q[:])
    oracle.append(qdb.inverse(), q_index[:]+q[:4])
    return oracle


def diffuser(n_parameters):
    q_par = QuantumRegister(n_parameters)
    q_checker = QuantumRegister(1)
    diffuser = QuantumCircuit(q_par, q_checker, name="diffuser")

    for i in range(len(q_par)):
        diffuser.append(HGate(), [q_par[i]])
        diffuser.append(XGate(), [q_par[i]])

    CZGate = ZGate().control(n_parameters)
    diffuser.append(CZGate, q_par[:]+[q_checker[0]])

    for i in range(len(q_par)):
        diffuser.append(XGate(), [q_par[i]])
        diffuser.append(HGate(), [q_par[i]])

    return diffuser


dataset = ['0101', '0111',
           '1001', '1100']


q_index = 2
oracle_circuit = oracle(dataset)
c = ClassicalRegister(q_index, name="c")
q = QuantumRegister(oracle_circuit.num_qubits, name="q")

qc = QuantumCircuit(q, c)
qc.append(XGate(), [q[-1]])
iterations = 1
for i in range(iterations):
    qc.append(oracle_circuit, qc.qubits)
    qc.append(diffuser(2), qc.qubits[0:2]+[qc.qubits[-1]])


qc.measure(q[:q_index], c[:])
# qc.decompose().draw()
# plt.show()
# exit()

simulator = AerSimulator()
transpiled_qc = transpile(qc, simulator)
result = simulator.run(transpiled_qc, shots=1024).result()
counts = result.get_counts()
plot_histogram(counts)
plt.show()


# TASK: FIND THE ORDER OF THE FEATURES THAT SATISFY THE TREE FUNCTION.
# The number is factorial of the number of features, so n!
# However, the number of features is limited. Is it a reasonable assumption?
# What is the cost of generating all the possible combinations?

# Create a dataset with the specified features and entries
# f0, f1, f2
dataset0 = [
    {"f0": 0, "f1": 1, "f2": 0, "l": 1},
    {"f0": 0, "f1": 1, "f2": 1, "l": 1},
    {"f0": 1, "f1": 0, "f2": 0, "l": 1},
    {"f0": 1, "f1": 1, "f2": 0, "l": 0},
]

# # f1, f0, f2
# dataset1 = [
#     {"f0": 1, "f1": 0, "f2": 0, "l": 1},
#     {"f0": 1, "f1": 0, "f2": 1, "l": 1},
#     {"f0": 0, "f1": 1, "f2": 0, "l": 1},
# ]

# # f0, f2, f1
# dataset2 = [
#     {"f0": 0, "f1": 0, "f2": 1, "l": 1},
#     {"f0": 0, "f1": 1, "f2": 1, "l": 1},
#     {"f0": 1, "f1": 0, "f2": 0, "l": 1},
# ]


# datasets = [dataset0, dataset1, dataset2]
datasets = [dataset0]
assignments_satisfied = []
# Evaluate the tree function for each entry in the dataset

for i, dataset in enumerate(datasets):
    satisfied = True
    for entry in dataset:
        result = classical_tree(
            entry["f0"], entry["f1"], entry["f2"], entry["l"])
        print(f"Input: {entry}, Result: {result}")
        if result != 1:
            satisfied = False
    if satisfied:
        assignments_satisfied.append(i)

print("Assignments satisfied:", assignments_satisfied)
