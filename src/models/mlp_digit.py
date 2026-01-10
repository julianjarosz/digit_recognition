import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def cross_entropy(y_pred, y_true):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss


def cross_entropy_derivative(y_pred, y_true):

    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    m = y_true.shape[0]
    grad = - (y_true / y_pred) / m

    return grad


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(Y, num_classes=10):
    one_hot_Y = np.zeros((Y.size, num_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y


def softmax_derivative(z):
    return softmax(z) * (1 - softmax(z))


def kaiming_normal(layer_sizes):
    weights = []
    biases = []

    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]
        print(f"n_in : {n_in} , n_out : {n_out}")
        std = np.sqrt(2 / (n_in + n_out))
        w = np.random.randn(n_in, n_out) * std
        b = np.zeros((1, n_out))
        weights.append(w)
        biases.append(b)

    return weights, biases


class MLP:
    def __init__(self, hidden_layers, learning_rate=0.01):
        self.hidden_layers = hidden_layers
        self.layer_sizes = [784] + self.hidden_layers + [10]
        self.weights, self.biases = kaiming_normal(self.layer_sizes)
        self.learning_rate = learning_rate
        self.activation_function = relu
        self.activation_function_derivative = relu_derivative

    def forward(self, inputs):
        self.activations = [inputs]
        self.z_values = []
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.activation_function(z)
            self.activations.append(a)

        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        a = softmax(z)
        self.activations.append(a)
        return a

    def backward(self, y_true_indices):
        y_true = one_hot(y_true_indices)
        output = self.activations[-1]
        m = y_true.shape[0]

        delta = (output - y_true) / m

        weight_grads = []
        bias_grads = []

        for i in range(len(self.weights) - 1, -1, -1):

            w_grad = np.dot(self.activations[i].T, delta)
            b_grad = np.sum(delta, axis=0, keepdims=True)

            weight_grads.insert(0, w_grad)
            bias_grads.insert(0, b_grad)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_function_derivative(self.z_values[i - 1])

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_grads[i]
            self.biases[i] -= self.learning_rate * bias_grads[i]

    def train_full_batch(self, X, Y, iterations):
        for i in range(iterations):
            self.forward(X)
            self.backward(Y)
            if i % 100 == 0:
                loss = cross_entropy(self.activations[-1], one_hot(Y))
                print(f"Iteration: {i}, Loss: {loss:.4f}")

    def train(self, X, Y, epochs, batch_size=32):
        m = X.shape[0]
        print(f"Type of bs: {type(batch_size)}, and value: {batch_size}")
        print(f"M : {m}")
        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]

            epoch_loss = 0

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]

                if X_batch.shape[0] < batch_size:
                    break

                self.forward(X_batch)
                self.backward(Y_batch)

                batch_loss = cross_entropy(self.activations[-1], one_hot(Y_batch))
                epoch_loss += batch_loss

            avg_loss = epoch_loss / (m // batch_size)

            if epoch % 1 == 0:
                print(f"Epoch: {epoch}, Avg Loss: {avg_loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
