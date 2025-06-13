import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class SGDRegressor:
    def __init__(self, lr=0.01, n_iter=1000, batch_size=32, scaling='standard'):
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.scaling = scaling
        self.scaler = None
        self.weights = None
        self.bias = None

    def _scale_features(self, X, fit=False):
        if self.scaling == 'standard':
            if fit or self.scaler is None:
                self.scaler = StandardScaler()
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        elif self.scaling == 'minmax':
            if fit or self.scaler is None:
                self.scaler = MinMaxScaler()
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        elif self.scaling == 'robust':
            if fit or self.scaler is None:
                self.scaler = RobustScaler()
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        elif self.scaling == 'none':
            return X
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling}")

    def fit(self, X, y):
        X = self._scale_features(X, fit=True)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_iter):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_pred = X_batch.dot(self.weights) + self.bias
                error = y_pred - y_batch

                grad_w = (1 / len(y_batch)) * (X_batch.T @ error)
                grad_b = (1 / len(y_batch)) * np.sum(error)

                max_grad = 1e3
                grad_w = np.clip(grad_w, -max_grad, max_grad)
                grad_b = np.clip(grad_b, -max_grad, max_grad)

                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b

                if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
                    print("Wagi zawierają NaN lub Inf, zatrzymuję trening.")
                    return

    def predict(self, X):
        X = self._scale_features(X, fit=False)
        return X.dot(self.weights) + self.bias

df = pd.read_csv(r"diamonds.csv").drop_duplicates()
df = df.rename(columns={'x': 'length_in_mm', 'y': 'width_in_mm', 'z': 'depth_in_mm'})
df['cut'] = df['cut'].map({'Ideal': 4, 'Premium': 3, 'Very Good': 2, 'Good': 1, 'Fair': 0})
df['color'] = df['color'].map({'D': 6, 'E': 5, 'F': 4, 'G': 3, 'H': 2, 'I': 1, 'J': 0})
df['clarity'] = df['clarity'].map({'IF': 7, 'VVS1': 6, 'VVS2': 5, 'VS1': 4, 'VS2': 3, 'SI1': 2, 'SI2': 1, 'I1': 0})

X = df.drop(columns='price').values
y = df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'lr': [0.001, 0.01, 0.05, 0.1],
    'n_iter': [500, 1000, 1500, 2000],
    'batch_size': [16, 32, 64, 128],
    'scaling': ['none', 'standard', 'minmax', 'robust']
}

results = []

print("Analiza SGD regresji - start\n")

for param_name, param_values in param_grid.items():
    print(f"Analiza parametru: {param_name}")
    for value in param_values:
        kwargs = {
            'lr': 0.01,
            'n_iter': 1000,
            'batch_size': 32,
            'scaling': 'standard'
        }
        kwargs[param_name] = value

        print(f"  Testuję {param_name} = {value}")
        model = SGDRegressor(**kwargs)
        model.fit(X_train, y_train)

        for zbior, X_set, y_set in [('train', X_train, y_train), ('test', X_test, y_test)]:
            y_pred = model.predict(X_set)
            r2 = r2_score(y_set, y_pred)
            rmse = np.sqrt(mean_squared_error(y_set, y_pred))
            mae = mean_absolute_error(y_set, y_pred)

            results.append({
                'parametr': param_name,
                'wartość': value,
                'zbiór': zbior,
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae
            })
    print(f"Zakończono analizę {param_name}\n")

df_results = pd.DataFrame(results)
df_results.to_excel("wyniki_sgd_regresja.xlsx", index=False)
print("Wyniki zapisane do pliku Excel.")
