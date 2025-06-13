from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sklearn.preprocessing import LabelEncoder

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000, batch_size=32, lambda_reg=0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.scaler = StandardScaler()

    def _one_hot(self, y, num_classes):
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        n_samples, n_features = X.shape
        n_classes = np.unique(y).size
        y_one_hot = self._one_hot(y, n_classes)

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        for epoch in range(self.n_iter):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                logits = X_batch @ self.weights + self.bias
                probs = self._softmax(logits)

                error = probs - y_batch
                grad_w = (X_batch.T @ error) / X_batch.shape[0] + self.lambda_reg * self.weights
                grad_b = np.sum(error, axis=0) / X_batch.shape[0]

                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b

    def predict(self, X):
        X = self.scaler.transform(X)
        logits = X @ self.weights + self.bias
        return np.argmax(self._softmax(logits), axis=1)



weather = pd.read_csv("weather_classification_data.csv")

categorical_cols = ['Cloud Cover', 'Season', 'Location']
numeric_cols = [
    'Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
    'Atmospheric Pressure', 'UV Index', 'Visibility (km)'
]
le_weather = LabelEncoder()
weather["Weather Type"] = le_weather.fit_transform(weather["Weather Type"])

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    weather[col] = le.fit_transform(weather[col])
    label_encoders[col] = le

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]
    return cleaned_df

cleaned_df = remove_outliers_iqr(weather, numeric_cols)

X = cleaned_df.drop("Weather Type", axis=1).values
y = cleaned_df["Weather Type"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'lr': [0.01, 0.05, 0.1, 0.2],
    'n_iter': [500, 1000, 1500, 2000],
    'batch_size': [16, 32, 64, 128],
    'lambda_reg': [0.0, 0.01, 0.1, 1.0]
}

results = []

for param_name, values in param_grid.items():
    for val in values:
        kwargs = {'lr': 0.05, 'n_iter': 1000, 'batch_size': 32, 'lambda_reg': 0.0}
        kwargs[param_name] = val

        model = LogisticRegression(**kwargs)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        prf_train = precision_recall_fscore_support(y_train, y_train_pred, average=None, zero_division=0)
        prf_test = precision_recall_fscore_support(y_test, y_test_pred, average=None, zero_division=0)

        result = {
            "parametr": param_name,
            "wartość": val,
            "Accuracy_train": acc_train,
            "Accuracy_test": acc_test
        }

        for i in range(len(prf_train[0])):
            result[f"Precision_train_class_{i}"] = prf_train[0][i]
            result[f"Recall_train_class_{i}"] = prf_train[1][i]
            result[f"F1_train_class_{i}"] = prf_train[2][i]
            result[f"Precision_test_class_{i}"] = prf_test[0][i]
            result[f"Recall_test_class_{i}"] = prf_test[1][i]
            result[f"F1_test_class_{i}"] = prf_test[2][i]

        results.append(result)
        print(f"Testowane: {param_name} = {val} - Accuracy: {acc_test:.4f}")

results_df = pd.DataFrame(results)
results_df.to_excel("logistic_regression_results.xlsx", index=False)
