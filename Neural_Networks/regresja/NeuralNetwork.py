import numpy as np

from preprocessed_data import X_train, y_train, scaler_y, X_test, y_test

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.activation_name = activation.lower()

    def relu(self, z): return np.maximum(0, z)
    def relu_derivative(self, z): return (z > 0).astype(float)

    def sigmoid(self, z): return 1 / (1 + np.exp(-z))
    def sigmoid_derivative(self, z): a = self.sigmoid(z); return a * (1 - a)

    def tanh(self, z): return np.tanh(z)
    def tanh_derivative(self, z): return 1 - np.tanh(z) ** 2

    def linear(self, z): return z
    def linear_derivative(self, z): return np.ones_like(z)

    def activate(self, z):
        return {
            'relu': self.relu,
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'linear': self.linear
        }.get(self.activation_name, self.relu)(z)

    def activate_derivative(self, z):
        return {
            'relu': self.relu_derivative,
            'sigmoid': self.sigmoid_derivative,
            'tanh': self.tanh_derivative,
            'linear': self.linear_derivative
        }.get(self.activation_name, self.relu_derivative)(z)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.activate(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        return self.Z2

    def backward(self, X, y, output, learning_rate=0.005):
        m = X.shape[0]
        dZ2 = output - y
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = dZ2 @ self.W2.T * self.activate_derivative(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs=1000, learning_rate=0.005):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y)**2)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.5f}")


    def predict(self, X):
        return self.forward(X)

model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=64, output_size=1)
model.train(X_train, y_train, epochs=1000, learning_rate=0.05)

def create_model(activation='relu'):
    return NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=64,
        output_size=1,
        activation=activation
    )

y_pred_scaled = model.predict(X_test)

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)


def train_batch(network, X, y, epochs=1000, learning_rate=0.01):
    for epoch in range(epochs):
        output = network.forward(X)
        loss = np.mean((output - y)**2)
        network.backward(X, y, output, learning_rate)
        if epoch % 100 == 0:
            print(f"[Batch] Epoch {epoch}, Loss: {loss:.5f}")

def train_rmsprop(network, X, y, epochs=1000, learning_rate=0.001, beta=0.9, epsilon=1e-8):
    cache_W1 = np.zeros_like(network.W1)
    cache_b1 = np.zeros_like(network.b1)
    cache_W2 = np.zeros_like(network.W2)
    cache_b2 = np.zeros_like(network.b2)

    for epoch in range(1, epochs + 1):
        output = network.forward(X)
        m = X.shape[0]

        dZ2 = output - y
        dW2 = network.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = dZ2 @ network.W2.T * network.activate_derivative(network.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        cache_W1 = beta * cache_W1 + (1 - beta) * dW1**2
        cache_b1 = beta * cache_b1 + (1 - beta) * db1**2
        cache_W2 = beta * cache_W2 + (1 - beta) * dW2**2
        cache_b2 = beta * cache_b2 + (1 - beta) * db2**2

        network.W1 -= learning_rate * dW1 / (np.sqrt(cache_W1) + epsilon)
        network.b1 -= learning_rate * db1 / (np.sqrt(cache_b1) + epsilon)
        network.W2 -= learning_rate * dW2 / (np.sqrt(cache_W2) + epsilon)
        network.b2 -= learning_rate * db2 / (np.sqrt(cache_b2) + epsilon)

        if epoch % 100 == 0:
            loss = np.mean((output - y) ** 2)
            print(f"[RMSProp] Epoch {epoch}, Loss: {loss:.5f}")

def train_minibatch(network, X, y, epochs=1000, learning_rate=0.001, batch_size=32):
    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, X.shape[0], batch_size):
            xb = X_shuffled[i:i+batch_size]
            yb = y_shuffled[i:i+batch_size]
            output = network.forward(xb)
            network.backward(xb, yb, output, learning_rate)
        if epoch % 100 == 0:
            loss = np.mean((network.forward(X) - y)**2)
            print(f"[MiniBatch] Epoch {epoch}, Loss: {loss:.5f}")

def train_adam(network, X, y, epochs=1000, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    mW1 = np.zeros_like(network.W1)
    mb1 = np.zeros_like(network.b1)
    mW2 = np.zeros_like(network.W2)
    mb2 = np.zeros_like(network.b2)

    vW1 = np.zeros_like(network.W1)
    vb1 = np.zeros_like(network.b1)
    vW2 = np.zeros_like(network.W2)
    vb2 = np.zeros_like(network.b2)

    for epoch in range(1, epochs + 1):
        output = network.forward(X)

        m = X.shape[0]
        dZ2 = output - y
        dW2 = network.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = dZ2 @ network.W2.T * network.activate_derivative(network.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        mW1 = beta1 * mW1 + (1 - beta1) * dW1
        mb1 = beta1 * mb1 + (1 - beta1) * db1
        mW2 = beta1 * mW2 + (1 - beta1) * dW2
        mb2 = beta1 * mb2 + (1 - beta1) * db2

        vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
        vb1 = beta2 * vb1 + (1 - beta2) * (db1 ** 2)
        vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
        vb2 = beta2 * vb2 + (1 - beta2) * (db2 ** 2)

        mW1_hat = mW1 / (1 - beta1 ** epoch)
        mb1_hat = mb1 / (1 - beta1 ** epoch)
        mW2_hat = mW2 / (1 - beta1 ** epoch)
        mb2_hat = mb2 / (1 - beta1 ** epoch)

        vW1_hat = vW1 / (1 - beta2 ** epoch)
        vb1_hat = vb1 / (1 - beta2 ** epoch)
        vW2_hat = vW2 / (1 - beta2 ** epoch)
        vb2_hat = vb2 / (1 - beta2 ** epoch)

        network.W1 -= learning_rate * mW1_hat / (np.sqrt(vW1_hat) + epsilon)
        network.b1 -= learning_rate * mb1_hat / (np.sqrt(vb1_hat) + epsilon)
        network.W2 -= learning_rate * mW2_hat / (np.sqrt(vW2_hat) + epsilon)
        network.b2 -= learning_rate * mb2_hat / (np.sqrt(vb2_hat) + epsilon)

        if epoch % 100 == 0:
            loss = np.mean((output - y) ** 2)
            print(f"[Adam] Epoch {epoch}, Loss: {loss:.5f}")



print("\nPrzykładowe przewidywania (prawdziwa cena vs przewidziana):")
for i in range(5):
    print(f"Rzeczywista: {y_true[i][0]:,.2f} \t|\tPrzewidziana: {y_pred[i][0]:,.2f} ")