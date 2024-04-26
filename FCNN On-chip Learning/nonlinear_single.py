# Import necessary libraries
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(123)

# Fetch the MNIST dataset
# X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
print(X.shape, y.shape)

# Normalize the data
X = X / 255
digits = 10
examples = y.shape[0]

# Reshape the labels
y = y.reshape(1, examples)

# Create one-hot encoded labels
Y_new=np.eye(digits)[y.astype('int32')]
Y_new=Y_new.T.reshape(digits, examples)
m=60000
m_test=X.shape[0] - m

# Split the data into training and testing sets
X_train, X_test=X[:m].T, X[m:].T
Y_train, Y_test=Y_new[:, :m], Y_new[:, m:]

# Shuffle the training data
shuffle_index=np.random.permutation(m)
X_train, Y_train=X_train[:, shuffle_index], Y_train[:, shuffle_index]

# Display a sample image
i=12
plt.imshow(X_train[:, i].reshape(28, 28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
Y_train[:, i]


# Define the tanh activation function
def tanh(z):
    """
    Compute the hyperbolic tangent (tanh) of an input value.

    Args:
    z (numpy.ndarray): Input value or array.

    Returns:
    numpy.ndarray: Output values after applying the tanh function element-wise.
    """
    return np.tanh(z)


# Define the multiclass loss function
def compute_multiclass_loss(Y, Y_hat):
    """
    Compute the multiclass cross-entropy loss.

    Args:
    Y (numpy.ndarray): True labels (one-hot encoded).
    Y_hat (numpy.ndarray): Predicted probabilities.

    Returns:
    float: Multiclass cross-entropy loss.
    """
    L_sum=np.sum(np.multiply(Y, np.log(Y_hat)))
    m=Y.shape[1]
    L=-(1 / m) * L_sum
    return L


# Define functions to calculate weights for potentiation and depression
def potentiation_weight(ini = -1, fin = 1, A_P = 0.5, N = 5, pulse_no = 1):
    """
    Calculate the weight value for a given pulse number in the potentiation curve.

    Args:
    ini (float): Initial weight value.
    fin (float): Final weight value.
    A_P (float): Parameter controlling the shape of the curve.
    N (int): Number of bits.
    pulse_no (int): Pulse number.

    Returns:
    float: Weight value.
    """
    Gmax=fin
    Gmin=ini
    n_max=2 ** (N)

    n=pulse_no / n_max
    pot_wt=(((Gmax - Gmin) * (1 - np.exp(-n / A_P))) /
              (1 - np.exp(-1 / A_P))) + Gmin

    return pot_wt


def depression_weight(ini = -1, fin = 1, A_D = -0.5, N = 5, pulse_no = 1):
    """
    Calculate the weight value for a given pulse number in the depression curve.

    Args:
    ini (float): Initial weight value.
    fin (float): Final weight value.
    A_D (float): Parameter controlling the shape of the curve.
    N (int): Number of bits.
    pulse_no (int): Pulse number.

    Returns:
    float: Weight value.
    """
    Gmax=fin
    Gmin=ini
    n_max=2 ** (N)

    n=pulse_no / n_max
    dep_wt=-((Gmax - Gmin) * (1 - np.exp((1 - n) / A_D)) /
               (1 - np.exp((1) / A_D))) + Gmax
    return dep_wt


# Define functions to calculate pulse numbers for given weight values
def inv_potentiation_weight(ini = -1, fin = 1, A_P = 0.5, N = 5, G = -1):
    """
    Calculate the pulse number for a given weight value in the potentiation curve.

    Args:
    ini (float): Initial weight value.
    fin (float): Final weight value.
    A_P (float): Parameter controlling the shape of the curve.
    N (int): Number of bits.
    G (float): Weight value.

    Returns:
    int: Pulse number.
    """
    Gmax=fin
    Gmin=ini
    n_max=2 ** (N)
    B=(Gmax - Gmin) / (1 - np.exp(-1 / A_P))
    n=np.round(-A_P * np.log(1 - (G - Gmin) / B) * n_max)
    return n


def inv_depression_weight(ini = -1, fin = 1, A_D = 0.5, N = 5, G = -1):
    """
    Calculate the pulse number for a given weight value in the depression curve.

    Args:
    ini (float): Initial weight value.
    fin (float): Final weight value.
    A_D (float): Parameter controlling the shape of the curve.
    N (int): Number of bits.
    G (float): Weight value.

    Returns:
    int: Pulse number.
    """
    Gmax=fin
    Gmin=ini
    n_max=2 ** (N)

    B=-(Gmax - Gmin) / (1 - np.exp(1 / A_D))
    n=np.round((1 - A_D * np.log(1 - (G - Gmax) / B)) * n_max)
    return n



def next_weight(W_init_int, W_f, del_W, W_init_int_dep, W_init_int_pot):
    global weight_up_pos, weight_up_neg
    weight_up_pos += np.sum(del_W < 0)
    weight_up_neg += np.sum(del_W > 0)
    # del_W>0 and W>0 -> Potentiation -> Depression, pulse no. should decrease in magnitude
    W_init_int[np.logical_and(del_W > 0, W_init_int > 0)]=(-W_init_int_dep + np.sign(del_W))[np.logical_and(del_W > 0, W_init_int > 0)]
    # del_W<0 and W>0 -> Potentiation -> Potentiation, pulse no. should increase in magnitude
    W_init_int[np.logical_and(del_W < 0, W_init_int > 0)] = (W_init_int_pot - np.sign(del_W))[np.logical_and(del_W < 0, W_init_int > 0)]
    # del_W>0 and W>0 -> Depression -> Depression, pulse no. should decrease in magnitude
    W_init_int[np.logical_and(del_W > 0, W_init_int < 0)] = (-W_init_int_dep + np.sign(del_W))[np.logical_and(del_W > 0, W_init_int < 0)]
    # del_W<0 and W>0 -> Depression -> Potentiation, pulse no. should increase in magnitude
    W_init_int[np.logical_and(del_W < 0, W_init_int < 0)] = (W_init_int_pot - np.sign(del_W))[np.logical_and(del_W < 0, W_init_int < 0)]
    W_f[W_init_int > 0] = potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, pulse_no=W_init_int)[W_init_int > 0]
    W_f[W_init_int < 0] = depression_weight(ini=-1, fin=1, A_D=A_D, N=N, pulse_no=-W_init_int)[W_init_int < 0]

    return W_f


N = 10
n_x = X_train.shape[0]
n_h = 100
learning_rate = 1e-3
weight_up_pos = 0
weight_up_neg = 0

A_P = 126.269
A_D = -3.7101

W1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(digits, n_h)
b2 = np.zeros((digits, 1))
W1 = np.clip(W1, -1, 1)
W2 = np.clip(W2, -1, 1)
b1 = np.clip(b1, -1, 1)
b2 = np.clip(b2, -1, 1)
W1_init_int = inv_potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, G=W1)
W1 = potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, pulse_no=W1_init_int)
W2_init_int = inv_potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, G=W2)
W2 = potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, pulse_no=W2_init_int)
b1_init_int = inv_potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, G=b1)
b1 = potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, pulse_no=b1_init_int)
b2_init_int = inv_potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, G=b2)
b2 = potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, pulse_no=b2_init_int)

losses, accuracies, test_accuracies = [], [], []
W1_f = W1
W2_f = W2
b1_f = b1
b2_f = b2
X = X_train
Y = Y_train


batch_size = 1
epochs = 101
epoch = []
test_accuracies = []
for i in range(epochs):
    shuffled_indxs = np.random.permutation(X_train.shape[0])
    batches = np.split(shuffled_indxs, batch_size)
    accuracy = 0

    for batch in batches:
        X = X_train[:, batch]
        Y = Y_train[:, batch]
        Z1 = np.matmul(W1, X) + b1
        A1 = tanh(Z1)
        Z2 = np.matmul(W2, A1) + b2
        A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

        cost = compute_multiclass_loss(Y, A2)

        dZ2 = A2 - Y
        dW2 = (1. / m) * np.matmul(dZ2, A1.T)
        db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(W2.T, dZ2)
        dZ1 = dA1 * (1 - tanh(Z1)**2)
        dW1 = (1. / m) * np.matmul(dZ1, X.T)
        db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Finds the corresponding depression states
        W1_init_int_dep = inv_depression_weight(ini=-1, fin=1, A_D=A_D, N=N, G=W1)
        # Finds the corresponding potentiation states
        W1_init_int_pot = inv_potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, G=W1)
        # Finds the corresponding depression states
        W2_init_int_dep = inv_depression_weight(ini=-1, fin=1, A_D=A_D, N=N, G=W2)
        # Finds the corresponding potentiation states
        W2_init_int_pot = inv_potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, G=W2)

        # Finds the corresponding depression states
        b1_init_int_dep = inv_depression_weight(ini=-1, fin=1, A_D=A_D, N=N, G=b1)
        # Finds the corresponding depression states
        b2_init_int_dep = inv_depression_weight(ini=-1, fin=1, A_D=A_D, N=N, G=b2)
        # Finds the corresponding potentiation states
        b2_init_int_pot = inv_potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, G=b2)
        # Finds the corresponding depression states
        b1_init_int_pot = inv_potentiation_weight(ini=-1, fin=1, A_P=A_P, N=N, G=b1)

        W1_f = next_weight(W1_init_int, W1_f, dW1,
                           W1_init_int_dep, W1_init_int_pot)
        W2_f = next_weight(W2_init_int, W2_f, dW2,
                           W2_init_int_dep, W2_init_int_pot)

        b1_f = next_weight(b1_init_int, b1_f, db1,
                           b1_init_int_dep, b1_init_int_pot)
        b2_f = next_weight(b2_init_int, b2_f, db2,
                           b2_init_int_dep, b2_init_int_pot)

        W1 = W1_f
        W2 = W2_f
        b1 = b1_f
        b2 = b2_f

        Z1_test = np.matmul(W1, X_test) + b1
        A1_test = tanh(Z1_test)
        Z2_test = np.matmul(W2, A1_test) + b2
        A2_test = np.exp(Z2_test) / np.sum(np.exp(Z2_test), axis=0)

        predictions = np.argmax(A2_test, axis=0)
        labels = np.argmax(Y_test, axis=0)

        epoch.append(i)
        test_accuracies.append(accuracy_score(predictions, labels))

        if (i % 10 == 0):

            print(accuracy_score(predictions, labels))
            # print(dW1>0)

            print("Epoch", i, "cost: ", cost, " Positive Updates", weight_up_pos," Negative Updates", weight_up_neg, " Total Updates", weight_up_pos + weight_up_neg)

print("Final cost:", cost)
a = epoch
b = test_accuracies

df = pd.DataFrame({"Epoch": a, "Test Accuracy": b})
df.to_csv("train.csv", index=False)
