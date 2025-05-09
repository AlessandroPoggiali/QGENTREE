import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import accuracy_score

def load_data():
    # Define the dataset
    X = np.array([
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    ], requires_grad=False)

    y = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1], requires_grad=False)

    # Split into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X
    y_train = y
    return X_train, y_train

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

def print_tree(root, level=0, prefix="Root: "):
    """ Recursively prints the binary tree in a readable format. """
    if root is not None:
        print("  " * level + prefix + f"Feature {root.feature_idx}")
        print_tree(root.left, level + 1, "L--- ")
        print_tree(root.right, level + 1, "R--- ")
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

def square_loss(labels, predictions, results):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2, requires_grad=True)+0.01*np.mean(qml.math.stack(results), requires_grad=True)

def ReLU(x):
    x = -x
    return x * (x > 0)

def bitmap_threshold(x):
    return (x*-1+1)/2

# Defining the softmax function
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

# Cost function (Binary Cross Entropy Loss)
def cost(params, X, y, n_nodes, n_features):
    circuit_result = variational_circuit(params, X)
    #print(circuit_result)
    #bitmap = bitmap_threshold(np.array(circuit_result))
    #print(bitmap)
    bitmap = np.array([1 if x < 0 else 0 for x in circuit_result], requires_grad=True)  # Trasforma i valori in 0 o 1 senza perdere la derivabilità
    decision_tree = build_tree(bitmap, n_nodes, n_features)
    print_tree(decision_tree)
    #return evaluate(decision_tree, X, y)
    predictions = np.array([predict(decision_tree, x, X, y) for x in X])
    #return -evaluate(decision_tree, X, y)+0.01*np.mean(qml.math.stack(circuit_result), requires_grad=True)
    #sl =  square_loss(y, predictions, circuit_result)
    return cross_entropy(y, predictions)#+np.mean(qml.math.stack(circuit_result), requires_grad=True)
    #return sl

X_train, y_train = load_data()
# Define quantum device
n_nodes = 3
n_features = len(X_train[0])
n_wires = n_nodes+n_nodes*int(np.log2(n_features))
dev = qml.device("default.qubit", wires=n_wires) # un qubit per ogni possibile nodo, logf qubit per ogni possibile nodo

# Define quantum node
@qml.qnode(dev,  diff_method="backprop")
def variational_circuit(weights, X):
    # Apply variational parameters
    qml.X(wires=0) # nodo 0 radice c'è sempre
    #print(params)
    for i in range(1, n_wires):
        qml.H(wires=i)
        qml.RY(weights[i-1], wires=i)

    for i in range(n_wires-1):
        qml.CNOT(wires=[i, i+1])
    
    
    #return qml.counts()  
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]


# Initialize parameters
params = qml.numpy.array(np.random.uniform(-2*np.pi, 2*np.pi, size=n_wires-1), requires_grad=True)
#params = np.linspace(-np.pi, np.pi, n_wires-1)  # Discrete set of angles
optimizer = qml.GradientDescentOptimizer(0.1) 

#fig, ax = qml.draw_mpl(variational_circuit)(params)
#plt.show()

# Training loop
epochs = 100
for epoch in range(epochs):
    #print("params prima", params)
    params = optimizer.step(cost, params, X_train, y_train, n_nodes, n_features)[0]
    #print("params dopo", params)
    
    if epoch % 10 == 0:
        loss_value = cost(params, X_train, y_train, n_nodes, n_features)
        print(f"Epoch {epoch}, Cost: {loss_value:.4f}, Params: {params}")
    
'''
# Evaluate accuracy
preds = np.array([quantum_model(params, x) for x in X_test])
preds = (preds > 0).astype(int)
accuracy = np.mean(preds == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
'''