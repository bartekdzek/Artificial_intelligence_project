import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

weather = pd.read_csv("klasyfikacja/weather_classification_data.csv")
weather.head()


le_weather = LabelEncoder()
weather["Weather Type"] = le_weather.fit_transform(weather["Weather Type"])

categorical_cols = ['Cloud Cover', 'Season', 'Location']
label_encoders = {}

numeric_cols = [
    'Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
    'Atmospheric Pressure', 'UV Index', 'Visibility (km)'
]

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
        before = cleaned_df.shape[0]
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]
        after = cleaned_df.shape[0]
        print(f"Removed {before - after} outliers from '{col}'")
    return cleaned_df

cleaned_df = remove_outliers_iqr(weather, numeric_cols)

X = cleaned_df.drop("Weather Type", axis=1)
y = cleaned_df["Weather Type"].values

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, feature_subset_ratio=1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_subset_ratio = feature_subset_ratio
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_single(sample, self.tree) for sample in X])

    def _predict_single(self, sample, node):
        while isinstance(node, dict):
            feature, threshold = node['feature'], node['threshold']
            if sample[feature] <= threshold:
                node = node['left']
            else:
                node = node['right']
        return node

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y, features):
        best_gain = -1
        best_split = None

        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                gain = self._gini(y) - (
                    len(y_left) / len(y) * self._gini(y_left) +
                    len(y_right) / len(y) * self._gini(y_right)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        return best_split

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or \
           num_samples < self.min_samples_split or \
           num_classes == 1:
            return np.bincount(y).argmax()

        feature_indices = np.random.choice(num_features,
                                           int(self.feature_subset_ratio * num_features),
                                           replace=False)
        split = self._best_split(X, y, feature_indices)
        if split is None:
            return np.bincount(y).argmax()

        left = self._grow_tree(X[split['left_mask']], y[split['left_mask']], depth + 1)
        right = self._grow_tree(X[split['right_mask']], y[split['right_mask']], depth + 1)
        return {'feature': split['feature'], 'threshold': split['threshold'], 'left': left, 'right': right}


scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_values = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [5, 10, 50, 100],
    'min_samples_leaf': [3, 10, 20, 30],
    'feature_subset_ratio': [0.1, 0.2, 0.4, 1.0]
}

results = {}
results_list = []

for param, values in param_values.items():
    print(f"\nParametr: {param}")
    for value in values:
        print(f" Sprawdzam {param} = {value}")
        kwargs = {
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'feature_subset_ratio': 1.0
        }
        kwargs[param] = value

        clf = DecisionTreeClassifier(**kwargs)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)


        acc_train = np.mean(y_train_pred == y_train)
        acc_test = np.mean(y_test_pred == y_test)

        print(f"    Accuracy (train): {acc_train:.4f}")
        print(f"    Accuracy (test):  {acc_test:.4f}")


        precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, y_train_pred, zero_division=0)
        precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, y_test_pred, zero_division=0)

        for cls in range(len(precision_train)):
            print(f"    Klasa {cls}:")
            print(f"      Train - Precision: {precision_train[cls]:.3f}, Recall: {recall_train[cls]:.3f}, F1: {f1_train[cls]:.3f}")
            print(f"      Test  - Precision: {precision_test[cls]:.3f}, Recall: {recall_test[cls]:.3f}, F1: {f1_test[cls]:.3f}")

        results_list.append({
            'parametr': param,
            'wartość': value,
            'Accuracy_train': acc_train,
            'Accuracy_test': acc_test,
            **{f'Precision_train_class_{cls}': precision_train[cls] for cls in range(len(precision_train))},
            **{f'Recall_train_class_{cls}': recall_train[cls] for cls in range(len(recall_train))},
            **{f'F1_train_class_{cls}': f1_train[cls] for cls in range(len(f1_train))},
            **{f'Precision_test_class_{cls}': precision_test[cls] for cls in range(len(precision_test))},
            **{f'Recall_test_class_{cls}': recall_test[cls] for cls in range(len(recall_test))},
            **{f'F1_test_class_{cls}': f1_test[cls] for cls in range(len(f1_test))},
        })

        if param not in results:
            results[param] = []
        results[param].append(acc_test)


for param, acc in results.items():
    plt.figure()
    plt.plot(param_values[param], acc, marker='o')
    plt.xlabel(param)
    plt.ylabel("Accuracy (test)")
    plt.title(f"Accuracy (test) vs {param}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

df_results = pd.DataFrame(results_list)
df_results.to_excel("wyniki_drzewo_klasyfikacja.xlsx", index=False)
print("Wyniki zapisane do wyniki_drzewo_klasyfikacja.xlsx")