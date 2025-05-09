# Variational Quantum Circuit (VQC) to learn optimal decision trees

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit_algorithms.optimizers import COBYLA, SLSQP
import numpy as np
import matplotlib.pyplot as plt

# Dataset (very small for feasibility)
data = [
    ([0, 0, 0, 0],0),
    ([0, 0, 1, 0],1),
    ([0, 1, 0, 0],1),
    ([0, 1, 1, 1],1),
    ([1, 0, 0, 1],0),
    ([1, 0, 1, 0],0),
    ([1, 0, 1, 1],1),
    ([1, 1, 0, 1],1),
    ([1, 1, 1, 0],1),
    ([0, 0, 0, 1],1),
    ([0, 0, 1, 1],0)
]

# Utility to decode tree from bitstring
# Tree: first 6 bits = internal node feature indices (3 nodes × 2 bits)
#       last 4 bits = leaf labels

n_nodes = 3
n_leaf = n_nodes + 1
n_features = len(data[0][0])
log_f = int(np.ceil(np.log2(n_features)))
n_qubits = n_nodes*log_f + n_leaf


def decode_tree(bitstring):
    feature_ids = [int(bitstring[i:i+2], 2) for i in range(0, 6, 2)]
    leaves = [int(b) for b in bitstring[6:]]
    return feature_ids, leaves

# Evaluate tree accuracy on dataset
# Simple depth-2 binary decision tree over boolean features

def evaluate_tree(feature_ids, leaves, data):
    correct = 0
    for x, y in data:
        # Traverse depth-2 tree
        node0 = x[feature_ids[0]]
        if node0 == 0:
            node1 = x[feature_ids[1]]
            pred = leaves[0] if node1 == 0 else leaves[1]
        else:
            node2 = x[feature_ids[2]]
            pred = leaves[2] if node2 == 0 else leaves[3]
        #print("Prediction of ",x,": ", pred)
        if pred == y:
            correct += 1
    #print("correct: ", correct)
    return correct / len(data)

# Create VQC with n_qubits qubits representing tree bits
params = [Parameter(f'theta_{i}') for i in range(n_qubits)]
def vqc():
    qc = QuantumCircuit(n_qubits, n_qubits)
    for i in range(n_qubits):
        qc.h(i)
        qc.rx(params[i], i)
    for i in range(n_qubits-1):
        qc.cx(i, i+1)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

# Backend
simulator = AerSimulator()

# Loss = 1 - average accuracy
loss_history = []
theta_history = []
def loss(theta):
    qc = vqc()
    bound_qc = qc.assign_parameters({p: t for p, t in zip(params, theta)})
    job = simulator.run(bound_qc, shots=1024)
    counts = job.result().get_counts()
    avg_acc = 0
    total = 0
    for bitstring, count in counts.items():
        bitstring = bitstring[::-1]  # Qiskit uses little endian
        try:
            features, leaves = decode_tree(bitstring)
            acc = evaluate_tree(features, leaves, data)
            avg_acc += acc * count
            total += count
        except:
            continue  # skip invalid encodings
    
    return 1 - avg_acc / total if total > 0 else 1

def loss2(theta):
    print("THETA",theta)
    qc = vqc()
    bound_qc = qc.assign_parameters({p: t for p, t in zip(params, theta)})
    job = simulator.run(bound_qc, shots=1024)
    counts = job.result().get_counts()

    # Find the bitstring with the maximum count
    best_bitstring = max(counts, key=counts.get)
    
    # Decode the tree and compute accuracy for this best bitstring
    best_bitstring = best_bitstring[::-1]  # Reverse due to little-endian representation
    try:
        features, leaves = decode_tree(best_bitstring)
        acc = evaluate_tree(features, leaves, data)
        loss_val = 1 - acc
    except:
        loss_val = 1  # High loss for invalid encodings

    loss_history.append(loss_val)
    theta_history.append(theta.copy())
    return loss_val

# Optimization
opt = COBYLA(disp=False)
np.random.seed(123)
theta_init = np.random.uniform(0, np.pi/2, n_qubits)

opt_result = opt.minimize(loss2,theta_init)

# Plot loss history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss (1 - accuracy)")
plt.title("VQC Training Loss Over Iterations")
plt.grid(True)

# Plot theta evolution
plt.subplot(1, 2, 2)
theta_history = np.array(theta_history)
for i in range(theta_history.shape[1]):
    plt.plot(theta_history[:, i], label=f"θ{i}")
plt.xlabel("Iteration")
plt.ylabel("Theta Value")
plt.title("Evolution of θ Parameters")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("\nOptimized theta:", opt_result.x)
print("Result on test set ")

test_data = [
    ([0, 1, 1, 0],0),
    ([1, 0, 0, 0],1),
    ([1, 1, 0, 0],0),
    ([1, 1, 1, 1],0),
    ([0, 1, 0, 1],0)
]
#print("Best accuracy achieved:", 1 - opt_result[1])
qc = vqc()
bound_qc = qc.assign_parameters({p: t for p, t in zip(params, opt_result.x)})
job = simulator.run(bound_qc, shots=1024)
counts = job.result().get_counts()

# Find the bitstring with the maximum count
best_bitstring = max(counts, key=counts.get)

# Decode the tree and compute accuracy for this best bitstring
best_bitstring = best_bitstring[::-1]  # Reverse due to little-endian representation
try:
    features, leaves = decode_tree(best_bitstring)
    print("features mapping",features)
    print("leaves",leaves)
    acc = evaluate_tree(features, leaves, test_data)
    print("accuracy ",acc)
except:
    print("invalid tree")