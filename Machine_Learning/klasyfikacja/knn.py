import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import random

class FastKNNClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', p=2, tie_breaker='random', batch_size=1000):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.tie_breaker = tie_breaker
        self.batch_size = batch_size

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _compute_distances(self, X_batch):
        diff = X_batch[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
        return np.linalg.norm(diff, ord=self.p, axis=2)

    def _get_top_k(self, distances):
        indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        top_distances = np.take_along_axis(distances, indices, axis=1)
        top_labels = self.y_train[indices]
        return top_labels, top_distances

    def _vote_batch(self, labels, dists):
        preds = []
        for lbls, dsts in zip(labels, dists):
            weights = {}
            for label, dist in zip(lbls, dsts):
                if self.weights == 'uniform':
                    w = 1
                elif self.weights == 'distance':
                    w = 1 / max(dist, 1e-3)
                elif self.weights == 'custom_linear':
                    w = max(1e-3, 1 - dist)
                elif self.weights == 'custom_inverse':
                    w = 1 / (1 + dist)
                else:
                    raise ValueError("Nieznana metoda wagowania")
                weights[label] = weights.get(label, 0) + w
            max_w = max(weights.values())
            top_labels = [lbl for lbl, w in weights.items() if w == max_w]
            preds.append(self._break_ties(top_labels))
        return np.array(preds)

    def _break_ties(self, labels):
        if self.tie_breaker == 'random':
            return random.choice(labels)
        elif self.tie_breaker == 'first':
            return labels[0]
        elif self.tie_breaker == 'min_label':
            return min(labels)
        elif self.tie_breaker == 'max_label':
            return max(labels)
        else:
            raise ValueError("Nieznany tie_breaker")

    def predict(self, X):
        preds = []
        for i in range(0, len(X), self.batch_size):
            X_batch = X[i:i + self.batch_size]
            distances = self._compute_distances(X_batch)
            if X.shape[0] == self.X_train.shape[0] and np.allclose(X, self.X_train):
                np.fill_diagonal(distances, np.inf)
            labels, dists = self._get_top_k(distances)
            preds.extend(self._vote_batch(labels, dists))
        return np.array(preds)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_recall_f1(y_true, y_pred):
    classes = np.unique(y_true)
    precision, recall, f1 = [], [], []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec  = tp / (tp + fn) if tp + fn > 0 else 0
        f1s  = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        precision.append(prec)
        recall.append(rec)
        f1.append(f1s)
    return np.array(precision), np.array(recall), np.array(f1)

weather = pd.read_csv("klasyfikacja/weather_classification_data.csv")
le_weather = LabelEncoder()
weather["Weather Type"] = le_weather.fit_transform(weather["Weather Type"])

categorical_cols = ['Cloud Cover', 'Season', 'Location']
for col in categorical_cols:
    le = LabelEncoder()
    weather[col] = le.fit_transform(weather[col])

numeric_cols = [
    'Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
    'Atmospheric Pressure', 'UV Index', 'Visibility (km)'
]

def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df

weather = remove_outliers_iqr(weather, numeric_cols)
X = weather.drop("Weather Type", axis=1).values
y = weather["Weather Type"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

np.random.seed(42)
sample_idx = np.random.choice(len(X), size=10000, replace=False)
X = X[sample_idx]
y = y[sample_idx]

param_grid = {
    'weights': ['uniform', 'distance', 'custom_linear', 'custom_inverse'],
    'p': [1, 2, 3, 4],
    'tie_breaker': ['random', 'first', 'min_label', 'max_label'],
    'n_neighbors': [2, 3, 5, 10]
}


results = []

for param, values in param_grid.items():
    for val in tqdm(values, desc=f"Testowanie {param}"):
        acc_train_all, acc_test_all = [], []
        pr_train_all, pr_test_all = [], []
        rc_train_all, rc_test_all = [], []
        f1_train_all, f1_test_all = [], []

        for seed in range(5):
            np.random.seed(seed)
            random.seed(seed)
            idx = np.random.permutation(len(X))
            split = int(0.8 * len(X))
            X_train, X_test = X[idx[:split]], X[idx[split:]]
            y_train, y_test = y[idx[:split]], y[idx[split:]]

            kwargs = {
                'n_neighbors': 5,
                'weights': 'uniform',
                'p': 2,
                'tie_breaker': 'random',
                'batch_size': 1000
            }
            kwargs[param] = val

            clf = FastKNNClassifier(**kwargs)
            clf.fit(X_train, y_train)
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)

            acc_train_all.append(accuracy(y_train, y_train_pred))
            acc_test_all.append(accuracy(y_test, y_test_pred))

            pr_tr, rc_tr, f1_tr = precision_recall_f1(y_train, y_train_pred)
            pr_te, rc_te, f1_te = precision_recall_f1(y_test, y_test_pred)

            pr_train_all.append(pr_tr)
            pr_test_all.append(pr_te)
            rc_train_all.append(rc_tr)
            rc_test_all.append(rc_te)
            f1_train_all.append(f1_tr)
            f1_test_all.append(f1_te)

        result = {
            "parametr": param,
            "wartość": val,
            "accuracy_train_avg": np.mean(acc_train_all),
            "accuracy_test_avg": np.mean(acc_test_all)
        }

        for i in range(len(pr_train_all[0])):
            result[f"precision_train_class_{i}"] = np.mean([x[i] for x in pr_train_all])
            result[f"recall_train_class_{i}"] = np.mean([x[i] for x in rc_train_all])
            result[f"f1_train_class_{i}"] = np.mean([x[i] for x in f1_train_all])
            result[f"precision_test_class_{i}"] = np.mean([x[i] for x in pr_test_all])
            result[f"recall_test_class_{i}"] = np.mean([x[i] for x in rc_test_all])
            result[f"f1_test_class_{i}"] = np.mean([x[i] for x in f1_test_all])

        results.append(result)

df_results = pd.DataFrame(results)
df_results.to_csv("wyniki_knn_classification.csv", index=False)
print("Wyniki zapisane do wyniki_knn_classification.csv")