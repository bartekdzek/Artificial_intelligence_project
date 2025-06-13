import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from time import time
from tqdm import tqdm


df = pd.read_csv("diamonds.csv").drop_duplicates()
df = df.rename(columns={'x': 'length_in_mm', 'y': 'width_in_mm', 'z': 'depth_in_mm'})
df['cut'] = df['cut'].map({'Ideal': 4, 'Premium': 3, 'Very Good': 2, 'Good': 1, 'Fair': 0})
df['color'] = df['color'].map({'D': 6, 'E': 5, 'F': 4, 'G': 3, 'H': 2, 'I': 1, 'J': 0})
df['clarity'] = df['clarity'].map({'IF': 7, 'VVS1': 6, 'VVS2': 5, 'VS1': 4, 'VS2': 3, 'SI1': 2, 'SI2': 1, 'I1': 0})

df = df.sample(10000, random_state=42)

X = df.drop(columns='price').values
y = df['price'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

class MyKNNRegressor:
    def __init__(self, n_neighbors=5, p=2, weights='uniform', aggregation='mean'):
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights
        self.aggregation = aggregation

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _get_weights(self, distances):
        if self.weights == 'uniform':
            return np.ones_like(distances)
        elif self.weights == 'distance':
            return 1 / (distances + 1e-5)
        elif self.weights == 'custom_linear':
            max_dist = np.max(distances, axis=1, keepdims=True) + 1e-5
            return 1 - distances / max_dist
        elif self.weights == 'custom_inverse':
            return 1 / (distances + 1e-5)
        else:
            return np.ones_like(distances)

    def _weighted_median(self, values, weights):
        result = []
        for val, w in zip(values, weights):
            if np.sum(w) == 0:
                result.append(np.mean(val))
                continue
            sorted_idx = np.argsort(val)
            val_sorted = val[sorted_idx]
            w_sorted = w[sorted_idx]
            cum_weights = np.cumsum(w_sorted)
            cutoff = np.sum(w_sorted) / 2.0
            median_idx = np.searchsorted(cum_weights, cutoff)
            result.append(val_sorted[min(median_idx, len(val_sorted)-1)])
        return np.array(result)

    def predict(self, X):
        if self.p == 1:
            distances = np.sum(np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
        elif self.p == 2:
            distances = np.sqrt(np.sum((X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2, axis=2))
        else:
            from scipy.spatial.distance import cdist
            distances = cdist(X, self.X_train, metric='minkowski', p=self.p)

        if np.array_equal(X, self.X_train):
            np.fill_diagonal(distances, np.inf)

        neighbors_idx = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        row_idx = np.arange(X.shape[0])[:, None]
        selected_distances = distances[row_idx, neighbors_idx]
        neighbors_y = self.y_train[neighbors_idx]
        weights = self._get_weights(selected_distances)

        if self.aggregation == 'mean':
            return np.mean(neighbors_y, axis=1)
        elif self.aggregation == 'median':
            return np.median(neighbors_y, axis=1)
        elif self.aggregation == 'weighted_mean':
            weighted_sum = np.sum(weights * neighbors_y, axis=1)
            weight_sum = np.sum(weights, axis=1)
            return weighted_sum / (weight_sum + 1e-8)
        elif self.aggregation == 'weighted_median':
            return self._weighted_median(neighbors_y, weights)
        else:
            raise ValueError("Nieznana metoda agregacji")

param_values = {
    'n_neighbors': [1, 3, 5, 10],
    'p': [1, 2, 3, 4],
    'weights': ['uniform', 'distance', 'custom_linear', 'custom_inverse'],
    'aggregation': ['mean', 'median', 'weighted_mean', 'weighted_median']
}
default_params = {
    'n_neighbors': 5,
    'p': 2,
    'weights': 'uniform',
    'aggregation': 'mean'
}
n_iterations = 5
results_list = []

for param, values in param_values.items():
    print(f"\n🔍 Analiza parametru: {param}")
    for value in tqdm(values, desc=f"{param}"):
        start = time()
        metric_accumulator = {
            'train': {'R2': [], 'RMSE': [], 'MAE': []},
            'test': {'R2': [], 'RMSE': [], 'MAE': []}
        }

        for i in range(n_iterations):
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42 + i)
            kwargs = default_params.copy()
            kwargs[param] = value

            model = MyKNNRegressor(**kwargs)
            model.fit(X_train, y_train)

            for subset_name, X_set, y_set in [('train', X_train, y_train), ('test', X_test, y_test)]:
                y_pred = model.predict(X_set)
                metric_accumulator[subset_name]['R2'].append(r2_score(y_set, y_pred))
                metric_accumulator[subset_name]['RMSE'].append(np.sqrt(mean_squared_error(y_set, y_pred)))
                metric_accumulator[subset_name]['MAE'].append(mean_absolute_error(y_set, y_pred))

        for subset_name in ['train', 'test']:
            results_list.append({
                'parametr': param,
                'wartość': value,
                'zbiór': subset_name,
                'R2': np.mean(metric_accumulator[subset_name]['R2']),
                'RMSE': np.mean(metric_accumulator[subset_name]['RMSE']),
                'MAE': np.mean(metric_accumulator[subset_name]['MAE'])
            })

        print(f" Zakończono {param} = {value} | Czas: {time() - start:.2f} sek")

df_results = pd.DataFrame(results_list)
df_results.to_csv("wyniki_knn_regression.csv", index=False)
print("Wyniki zapisano do 'wyniki_knn_regression.csv'")