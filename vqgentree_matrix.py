import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import accuracy_score
import math as m
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def load_data():
    # Define the dataset
    X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1],
    ], requires_grad=False)

    y = np.array([0, 1, 1, 1, 0, 1], requires_grad=False)
    y =  [[1,0] if not x else [0,1] for x in y]
    # Split into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X
    y_train = y
    return X_train, y_train

def square_loss(labels, predictions):
    return np.mean((labels - qml.math.stack(predictions)) ** 2, requires_grad=True)

def _w_state_circuit(circuit, nn, mm, qubitC):    
    global qubitT #reference to global variable qubitT
    
    if nn == 0 and mm == 1:
        pass #do nothing in this case
    elif nn == 1 and mm == 2: #case (1,2)
        #circuit.cu(theta=m.pi/2, phi=0, lam=0, gamma=0, control_qubit=q[qubitC], target_qubit=q[qubitT])
        circuit.cu(theta=m.pi/2, phi=0, lam=0, gamma=0, control_qubit=q[qubitT], target_qubit=q[qubitC])
        circuit.cx(q[qubitT], q[qubitC])
        qubitT = qubitT + 1
    else: #otherwise
        theta = 2*np.arccos(m.sqrt(nn/mm))
        #circuit.cu(theta=theta, phi=0, lam=0, gamma=0, control_qubit=q[qubitC], target_qubit=q[qubitT])
        circuit.cu(theta=theta, phi=0, lam=0, gamma=0, control_qubit=q[qubitT], target_qubit=q[qubitC])
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

qubits = 3
q = QuantumRegister(qubits, name = 'q')
c = ClassicalRegister(qubits, name = 'c')
qc = w_state_circuit(qubits, q, c)
my_func = qml.from_qiskit(qc)

#fig, ax = qml.draw_mpl(my_func)()
#plt.show()
#exit()

X_train, y_train = load_data()

# Define quantum device
n_nodes = 3
n_features = len(X_train[0])
n_leaves = n_nodes+1
n_wires = n_nodes*n_features+n_leaves*2
dev = qml.device("default.qubit", wires=n_wires,shots=1)

# Define quantum node
@qml.qnode(dev,  diff_method="parameter-shift")
def variational_circuit(weights):
    ## circuit part for matrix A
    
    for i in range(n_nodes*n_features):
        #qml.H(wires=i)
        qml.RY(weights[i], wires=i)
    '''
    for i in range(n_nodes*n_features-1):
        qml.CNOT(wires=[i, i+1])
    '''
    
    #for i in range(n_nodes*n_features):
        #if i%3==2:
        #qml.RY(weights[i], wires=i)

    my_func()
    my_func(wires=[3,4,5])
    my_func(wires=[6,7,8])

    ## circuit part for matrix E
    
    j = n_nodes*n_features
    for i in range(n_nodes*n_features, n_wires):
        if i%2==(n_nodes*n_features%2):
            qml.RY(weights[j], wires=i)
            j + 1
        else:
            qml.X(wires=i)
            qml.CNOT(wires=[i-1, i])
    

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]



def construct_pattern_matrix(e):
    assert (e + 1) & e == 0, "e + 1 must be a power of two"
    n = e + 1
    mat = np.zeros((e, n), dtype=int)

    # pattern filling
    level = 0
    block_size = n // 2
    while block_size >= 1:
        for i in range(0, n, 2 * block_size):
            row_index = level
            if row_index >= e:
                break
            mat[row_index, i:i + block_size] = 1
            mat[row_index, i + block_size:i + 2 * block_size] = -1
            level += 1
        block_size = block_size // 2

    return mat

def compute_matrices(A):
    B = np.full((1, n_nodes), 0.5)
    #C = np.full((n_nodes, n_leaves), 1) ## DA COSTRUIRE IN BASE AD A
    C = construct_pattern_matrix(n_nodes)
    D = np.sum(np.where(C >= 0, C, 0), axis=0)
    return B,C,D

def predict(A, B, C, D, E, x):
    #(((xA < B)*C)==D)*E
    prediction = ((x@A < B).astype(int)@C == D).astype(int)@E
    return prediction
    
def bitmap_threshold(x):
    return (x*-1+1)/2

# Cost function (Binary Cross Entropy Loss)
def cost(params, X, y, n_nodes, n_features):
    circuit_result = variational_circuit(params)
    bitmap = bitmap_threshold(np.array(circuit_result))
    ## FARE RESHAPE PER AVERE MATRICE A ed E
    A = np.reshape(bitmap[:n_nodes*n_features],(n_features,n_nodes))
    A = A.T
    E = np.reshape(bitmap[n_nodes*n_features:],(n_leaves,2))
    print("A", A)
    #print("E", E)
    ## CREARE MATRICI B,C,D A PARTIRE DA MATRICE A 
    B, C, D = compute_matrices(A)
    #print("B", B)
    #print("C", C)
    #print("D", D)
    ## PREDICTION: (((xA < B)*C)==D)*E forall X
    predictions = np.array([predict(A,B,C,D,E, x) for x in X], requires_grad=True)
    ## RESTITUIRE VALORE LOSS
    return square_loss(y, predictions)
 
 # Initialize parameters
params = qml.numpy.array(np.random.uniform(-2*np.pi, 2*np.pi, size=n_nodes*n_features+n_leaves), requires_grad=True)
optimizer = qml.GradientDescentOptimizer(0.1) 

#fig, ax = qml.draw_mpl(variational_circuit)(params)
#plt.show()
#exit()


# Training loop
epochs = 100
for epoch in range(epochs):
    #print("params prima", params)
    params = optimizer.step(cost, params, X_train, y_train, n_nodes, n_features)[0]
    #print("params dopo", params)
    
    if epoch % 10 == 0:
        loss_value = cost(params, X_train, y_train, n_nodes, n_features)
        print(f"Epoch {epoch}, Cost: {loss_value:.4f}, Params: {params}")
