import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.providers.fake_provider import *
from qiskit.primitives import Sampler, StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score

class TreeNode:
    def __init__(self, feature_idx=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.left = left
        self.right = right

def build_tree(bit_list, n_nodes, n_features):
    """ Constructs the decision tree from the given bit list. """
    log_f = int(np.log2(n_features))
    
    presence_bits = bit_list[:n_nodes]
    feature_bits = bit_list[n_nodes:]
    #print('p',presence_bits)
    #print('f',feature_bits)
    # Extract all feature indices (keeping their positions)
    feature_indices = [
        int("".join(map(str, feature_bits[i * log_f:(i + 1) * log_f])), 2)
        for i in range(n_nodes)
    ]
    
    def construct_tree(i):
        if i >= len(presence_bits) or presence_bits[i] == 0:
            return None  # Skip absent nodes
        
        node = TreeNode(feature_indices[i])  # Assign the feature index
        node.left = construct_tree(2 * i + 1)  # Recurse on left child
        node.right = construct_tree(2 * i + 2)  # Recurse on right child
        return node

    return construct_tree(0)

def get_majority(feature_values, X, Y):
    mask = np.ones(X.shape[0], dtype=bool)  # Start with all True (include all samples)

    for feature_idx, value in feature_values.items():
        mask &= (X[:, feature_idx] == value)  # Keep only rows where feature matches value

    # Filter Y based on the mask
    filtered_Y = Y[mask]

    # If no data remains, return a default label (e.g., 0) non puo succedere, almeno un sample ce l'ho sicuro
    #if filtered_Y.size == 0:
    #    return 0  # Default label if no matching samples

    # Compute the majority label
    unique_labels, counts = np.unique(filtered_Y, return_counts=True)
    majority_label = unique_labels[np.argmax(counts)]  # Most frequent label

    return majority_label

def predict(tree, sample, X, Y):
    """ Predict the label for a single sample. """
    feature_values = {}
    node = tree  
    while node.left or node.right:  # Traverse until a leaf
        if sample[node.feature_idx] == 0:
            feature_values[node.feature_idx] = 0
            node = node.left
        else:
            feature_values[node.feature_idx] = 1
            node = node.right
        if node is None: 
            return get_majority(feature_values, X, Y)
    # Predict based on the feature at the leaf node
    return 0 if sample[node.feature_idx] == 0 else 1

def loss_function(tree, X, Y):
    epsilon = 1e-9 
    predictions = np.array([predict(tree, x, X, Y) for x in X])
    loss = -np.mean(Y * np.log(predictions + epsilon) + (1 - Y) * np.log(1 - predictions + epsilon))
    return loss

def evaluate(tree, X, Y):
    """ Computes the accuracy of the tree on the dataset. """
    predictions = np.array([predict(tree, x, X, Y) for x in X])
    #print("predictions",predictions)
    #return np.mean(predictions == Y)

    return accuracy_score(Y, predictions)



# Dataset booleano (Feature1, Feature2) -> Classe
X = np.array([
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
])

y = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1])

n_nodes = 3
n_features = len(X[0])
num_qubits = n_nodes+n_nodes*int(np.log2(n_features))

def print_tree(root, level=0, prefix="Root: "):
    """ Recursively prints the binary tree in a readable format. """
    if root is not None:
        print("  " * level + prefix + f"Feature {root.feature_idx}")
        print_tree(root.left, level + 1, "L--- ")
        print_tree(root.right, level + 1, "R--- ")

# Creiamo un circuito variazionale parametrico
def variational_circuit():
    qc = QuantumCircuit(num_qubits)
    qc.x(0)
    # Inizializziamo ogni qubit con uno stato parametrico
    for i in range(1, num_qubits):
        qc.ry(params[i-1], i)  # Rotazione dipendente da parametri
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    
    qc.measure_all()
    return qc

# Parametri variazionali
params = ParameterVector('θ', num_qubits-1)

def softmax(values):
 
    # Computing element wise exponential value
    exp_values = np.exp(values)
 
    # Computing sum of these values
    exp_values_sum = np.sum(exp_values)
 
    # Returing the softmax output.
    return exp_values/exp_values_sum
# Cross Entropy function.
def cross_entropy(y_true, y_pred):
 
    # computing softmax values for predicted values
    y_pred = softmax(y_pred)
    loss = 0
     
    # Doing cross entropy Loss
    for i in range(len(y_pred)):
 
        # Here, the loss is computed using the
        # above mathematical formulation.
        loss = loss + (-1 * y_true[i]*np.log(y_pred[i]))
 
    return loss

# Funzione di costo: misura quanto bene il decision tree fitta i dati
def cost_function(params_values):
    #qc = variational_circuit()
    #qc.assign_parameters(params_values, inplace=True)
    #qc = variational_circuit(params_values)
    qc = variational_circuit()
    qc.assign_parameters(params_values, inplace=True)
    
    # Simuliamo il circuito
    sampler = StatevectorSampler()
    #sampler.set_options(backend=backend)
    result = sampler.run([qc], shots=1000).result()
    counts = result[0].data.meas.get_counts()
    '''
    # Stimiamo la qualità dell'albero: più zero, migliore è l'accuratezza
    error_score = counts.get('1' * num_qubits, 0)  # Quanti stati "pessimi" misuriamo
    return error_score / sum(counts.values())  # Normalizziamo
    '''

    bitmap = list(map(int, max(counts, key=counts.get)))[::-1]
    decision_tree = build_tree(bitmap, n_nodes, n_features)
    #print_tree(decision_tree)
    predictions = np.array([predict(decision_tree, x, X, y) for x in X])
    return cross_entropy(y, predictions)
    return -evaluate(decision_tree, X, y)
    return loss_function(decision_tree, X, y)

#initial_params = np.linspace(-np.pi, np.pi, num_qubits-1) 
initial_params = np.random.uniform(-2*np.pi, 2*np.pi, size=1*(num_qubits-1))  # Inizializziamo parametri casuali
#optimal_params = optimizer.minimize(cost_function, initial_params).x
params_history = [initial_params]
loss_history = []

params = initial_params.copy()
max_epochs = 100

for epoch in range(max_epochs):
    # Minimizzazione step-by-step usando scipy.optimize
    result = minimize(cost_function, params, method="COBYLA")
    
    loss = result.fun  # Valore della loss attuale
    params = result.x  # Aggiorniamo i parametri
    print(params)
    # Memorizziamo la perdita e i parametri
    loss_history.append(loss)
    params_history.append(params)
    
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
'''
out = minimize(cost_function,
                   x0=params,
                   method="COBYLA",
                   tol=0.001,
                   options={'maxiter': max_epochs})
best_parameters = out.x
# Visualizziamo l'andamento della loss
'''
'''
plt.plot(range(1, epochs+1), loss_history, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over epochs")
plt.show()
'''
'''
# Mostriamo il miglior circuito trovato
qc_optimal = variational_circuit(current_params)
qc_optimal = qc_optimal.assign_parameters(current_params)
'''
