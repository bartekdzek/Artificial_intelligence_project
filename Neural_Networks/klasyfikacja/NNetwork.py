import numpy as np
from sklearn.preprocessing import LabelEncoder
from klasyfikacja.processdata import y_train, y_test, X_train, y


class SimpleNN:
    def __init__(self, input_size, hidden_size=64, output_size=4, learning_rate=0.001, epochs=1000, random_state=0):
        np.random.seed(random_state)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs


        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def cross_entropy(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
        return loss

    def to_one_hot(self, y):
        one_hot = np.zeros((y.size, self.output_size))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def fit(self, X_train, y_train, verbose=False):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        y_train_one_hot = self.to_one_hot(y_train)

        for epoch in range(self.epochs):

            z1 = X_train @ self.W1 + self.b1
            a1 = self.relu(z1)
            z2 = a1 @ self.W2 + self.b2
            a2 = self.softmax(z2)


            loss = self.cross_entropy(y_train_one_hot, a2)


            dz2 = a2 - y_train_one_hot
            dW2 = a1.T @ dz2 / X_train.shape[0]
            db2 = np.sum(dz2, axis=0, keepdims=True) / X_train.shape[0]

            dz1 = dz2 @ self.W2.T * self.relu_derivative(z1)
            dW1 = X_train.T @ dz1 / X_train.shape[0]
            db1 = np.sum(dz1, axis=0, keepdims=True) / X_train.shape[0]


            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            if verbose and (epoch % 500 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch} — Loss: {loss:.4f}")

    def predict(self, X):
        X = np.array(X)
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.softmax(z2)
        return np.argmax(a2, axis=1)

    @staticmethod
    def encode_labels(y):
        le = LabelEncoder()
        return le.fit_transform(y), le

def train_batch_gradient_descent(X_train, y_train_encoded, X_test, y_test_encoded, input_size, output_size):
    model = SimpleNN(input_size=input_size, output_size=output_size, epochs=10000)
    model.fit(X_train, y_train_encoded, verbose=True)
    return model

def train_on_batch(self, X_batch, y_batch):
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    y_one_hot = self.to_one_hot(y_batch)

    z1 = X_batch @ self.W1 + self.b1
    a1 = self.relu(z1)
    z2 = a1 @ self.W2 + self.b2
    a2 = self.softmax(z2)

    loss = self.cross_entropy(y_one_hot, a2)

    dz2 = a2 - y_one_hot
    dW2 = a1.T @ dz2 / X_batch.shape[0]
    db2 = np.sum(dz2, axis=0, keepdims=True) / X_batch.shape[0]

    dz1 = dz2 @ self.W2.T * self.relu_derivative(z1)
    dW1 = X_batch.T @ dz1 / X_batch.shape[0]
    db1 = np.sum(dz1, axis=0, keepdims=True) / X_batch.shape[0]

    self.W1 -= self.learning_rate * dW1
    self.b1 -= self.learning_rate * db1
    self.W2 -= self.learning_rate * dW2
    self.b2 -= self.learning_rate * db2

    return loss


def train_mini_batch_gradient_descent(X_train, y_train_encoded, X_test, y_test_encoded, input_size, output_size, batch_size=32):
    model = SimpleNN(input_size=input_size, output_size=output_size, epochs=10000)
    import types
    model.train_on_batch = types.MethodType(train_on_batch, model)

    epochs = 10000
    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_encoded[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            loss = model.train_on_batch(X_batch, y_batch)

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch} — Loss: {loss:.4f}")

    return model

def train_momentum(X_train, y_train_encoded, X_test, y_test_encoded, input_size, output_size,
                   learning_rate=0.001, momentum=0.9, epochs=10000, batch_size=32):
    model = SimpleNN(input_size=input_size, output_size=output_size, epochs=1, learning_rate=learning_rate)

    import types
    model.train_on_batch = types.MethodType(train_on_batch, model)

    v_W1 = np.zeros_like(model.W1)
    v_b1 = np.zeros_like(model.b1)
    v_W2 = np.zeros_like(model.W2)
    v_b2 = np.zeros_like(model.b2)

    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_encoded[indices]

        total_loss = 0
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            y_one_hot = model.to_one_hot(y_batch)

            z1 = X_batch @ model.W1 + model.b1
            a1 = model.relu(z1)
            z2 = a1 @ model.W2 + model.b2
            a2 = model.softmax(z2)

            dz2 = a2 - y_one_hot
            dW2 = a1.T @ dz2 / batch_size
            db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

            dz1 = dz2 @ model.W2.T * model.relu_derivative(z1)
            dW1 = X_batch.T @ dz1 / batch_size
            db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

            v_W1 = momentum * v_W1 - learning_rate * dW1
            model.W1 += v_W1

            v_b1 = momentum * v_b1 - learning_rate * db1
            model.b1 += v_b1

            v_W2 = momentum * v_W2 - learning_rate * dW2
            model.W2 += v_W2

            v_b2 = momentum * v_b2 - learning_rate * db2
            model.b2 += v_b2

            total_loss += model.cross_entropy(y_one_hot, a2)

        if epoch % 500 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / (X_train.shape[0] // batch_size)
            print(f"Epoch {epoch} — Loss: {avg_loss:.4f}")

    return model

def train_adagrad(X_train, y_train_encoded, X_test, y_test_encoded, input_size, output_size,
                  learning_rate=0.001, epsilon=1e-8, epochs=10000, batch_size=32):
    model = SimpleNN(input_size=input_size, output_size=output_size, epochs=1, learning_rate=learning_rate)

    import types
    model.train_on_batch = types.MethodType(train_on_batch, model)

    G_W1 = np.zeros_like(model.W1)
    G_b1 = np.zeros_like(model.b1)
    G_W2 = np.zeros_like(model.W2)
    G_b2 = np.zeros_like(model.b2)

    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_encoded[indices]

        total_loss = 0
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            y_one_hot = model.to_one_hot(y_batch)

            z1 = X_batch @ model.W1 + model.b1
            a1 = model.relu(z1)
            z2 = a1 @ model.W2 + model.b2
            a2 = model.softmax(z2)

            dz2 = a2 - y_one_hot
            dW2 = a1.T @ dz2 / batch_size
            db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

            dz1 = dz2 @ model.W2.T * model.relu_derivative(z1)
            dW1 = X_batch.T @ dz1 / batch_size
            db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

            G_W1 += dW1 ** 2
            G_b1 += db1 ** 2
            G_W2 += dW2 ** 2
            G_b2 += db2 ** 2

            model.W1 -= learning_rate * dW1 / (np.sqrt(G_W1) + epsilon)
            model.b1 -= learning_rate * db1 / (np.sqrt(G_b1) + epsilon)
            model.W2 -= learning_rate * dW2 / (np.sqrt(G_W2) + epsilon)
            model.b2 -= learning_rate * db2 / (np.sqrt(G_b2) + epsilon)

            total_loss += model.cross_entropy(y_one_hot, a2)

        if epoch % 500 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / (X_train.shape[0] // batch_size)
            print(f"Epoch {epoch} — Loss: {avg_loss:.4f}")

    return model


y_train_encoded, label_encoder = SimpleNN.encode_labels(y_train)
y_test_encoded = label_encoder.transform(y_test)
input_size = X_train.shape[1]
output_size = len(np.unique(y))
model = SimpleNN(input_size=input_size, output_size=output_size, epochs=10000, learning_rate=0.001)
model.fit(X_train, y_train_encoded, verbose=True)
