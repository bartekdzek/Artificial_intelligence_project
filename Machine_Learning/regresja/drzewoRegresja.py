import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import random


df = pd.read_csv("diamonds.csv").drop_duplicates()
df = df.rename(columns={'x': 'length_in_mm', 'y': 'width_in_mm', 'z': 'depth_in_mm'})
df['cut'] = df['cut'].map({'Ideal': 4, 'Premium': 3, 'Very Good': 2, 'Good': 1, 'Fair': 0})
df['color'] = df['color'].map({'D': 6, 'E': 5, 'F': 4, 'G': 3, 'H': 2, 'I': 1, 'J': 0})
df['clarity'] = df['clarity'].map({'IF': 7, 'VVS1': 6, 'VVS2': 5, 'VS1': 4, 'VS2': 3, 'SI1': 2, 'SI2': 1, 'I1': 0})

X = df.drop(columns='price').values
y = df['price'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, feature_subset_ratio=1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_subset_ratio = feature_subset_ratio

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self._build_tree(np.array(X), np.array(y), depth=0)

    def predict(self, X):
        return np.array([self._predict_input(x, self.tree) for x in X])

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X, y):
        best_mse = float('inf')
        best_split = None
        n_features_to_use = max(1, int(self.n_features * self.feature_subset_ratio))
        features = random.sample(range(self.n_features), n_features_to_use)

        for feature_index in features:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if (len(y[left_mask]) < self.min_samples_split or
                    len(y[right_mask]) < self.min_samples_split or
                    len(y[left_mask]) < self.min_samples_leaf or
                    len(y[right_mask]) < self.min_samples_leaf):
                    continue
                mse = (
                    len(y[left_mask]) * self._mse(y[left_mask]) +
                    len(y[right_mask]) * self._mse(y[right_mask])
                ) / len(y)
                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left': (X[left_mask], y[left_mask]),
                        'right': (X[right_mask], y[right_mask])
                    }
        return best_split

    def _build_tree(self, X, y, depth):
        if (self.max_depth is not None and depth >= self.max_depth) or len(X) < self.min_samples_split:
            return {'value': np.mean(y)}
        split = self._best_split(X, y)
        if split is None:
            return {'value': np.mean(y)}
        return {
            'feature_index': split['feature_index'],
            'threshold': split['threshold'],
            'left': self._build_tree(*split['left'], depth + 1),
            'right': self._build_tree(*split['right'], depth + 1)
        }

    def _predict_input(self, x, tree):
        if 'value' in tree:
            return tree['value']
        if x[tree['feature_index']] <= tree['threshold']:
            return self._predict_input(x, tree['left'])
        else:
            return self._predict_input(x, tree['right'])


param_values = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'feature_subset_ratio': [1.0, 0.75, 0.5, 0.25]
}

results_list = []

print("Rozpoczynam analizę parametrów drzewa decyzyjnego...\n")

for param, values in param_values.items():
    print(f"Analiza parametru: {param}")
    for value in values:
        kwargs = {
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'feature_subset_ratio': 1.0
        }
        kwargs[param] = value
        print(f"  Testuję {param} = {value}")
        model = DecisionTreeRegressor(**kwargs)
        model.fit(X_train, y_train)

        for subset_name, X_set, y_set in [('train', X_train, y_train), ('test', X_test, y_test)]:
            y_pred = model.predict(X_set)
            r2 = r2_score(y_set, y_pred)
            rmse = np.sqrt(mean_squared_error(y_set, y_pred))
            mae = mean_absolute_error(y_set, y_pred)
            results_list.append({
                'parametr': param,
                'wartość': value,
                'zbiór': subset_name,
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae
            })
    print(f"Zakończono analizę {param}.\n")


df_results = pd.DataFrame(results_list)
df_results.to_excel("wyniki_drzewo_decyzyjne.xlsx", index=False)
print("Wyniki zapisano do pliku 'wyniki_drzewo_decyzyjne.xlsx'\n")


metrics = ['R2', 'RMSE', 'MAE']
fig, axes = plt.subplots(len(param_values), len(metrics), figsize=(18, 16))

for i, (param, values) in enumerate(param_values.items()):
    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        for zbior in ['train', 'test']:
            df_subset = df_results[(df_results['parametr'] == param) & (df_results['zbiór'] == zbior)]
            ax.plot(df_subset['wartość'], df_subset[metric], marker='o', label=zbior)
        ax.set_title(f'{metric} vs {param}')
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

plt.tight_layout()
plt.show()
