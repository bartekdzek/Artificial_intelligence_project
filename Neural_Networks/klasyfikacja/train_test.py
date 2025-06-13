import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from NNetwork import SimpleNN
from processdata import remove_outliers_iqr, label_encoders

weather = pd.read_csv("weather_classification_data.csv")

categorical_cols = ['Cloud Cover', 'Season', 'Location']
numeric_cols = [
    'Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
    'Atmospheric Pressure', 'UV Index', 'Visibility (km)'
]

for col in categorical_cols:
    weather[col] = label_encoders[col].transform(weather[col])

cleaned_df = remove_outliers_iqr(weather, numeric_cols)


X = cleaned_df.drop("Weather Type", axis=1)
y_raw = cleaned_df["Weather Type"]


y, label_encoder = SimpleNN.encode_labels(y_raw.values)
X = X.values

n_classes = len(np.unique(y))
class_names = label_encoder.inverse_transform(np.arange(n_classes))

test_sizes = np.arange(0.1, 0.4, 0.1)
repeats = 5

all_results = []

for test_size in test_sizes:
    print(f"\nAnaliza dla test_size = {test_size:.2f}")

    temp_results = []

    for i in range(repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=i
        )

        model = SimpleNN(
            input_size=X.shape[1],
            hidden_size=64,
            output_size=n_classes,
            learning_rate=0.001,
            epochs=10000,
            random_state=i
        )
        model.fit(X_train, y_train, verbose=False)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(
            y_train, y_train_pred, labels=range(n_classes), zero_division=0
        )
        precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(
            y_test, y_test_pred, labels=range(n_classes), zero_division=0
        )

        temp_results.append({
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_test': f1_test
        })

        print(f"Iteracja {i + 1}: test_acc={test_acc:.4f} | " +
              " | ".join([f"F1_{class_names[idx]}={f1_test[idx]:.2f}" for idx in range(n_classes)]))


    avg_train_acc = np.mean([r['train_accuracy'] for r in temp_results])
    avg_test_acc = np.mean([r['test_accuracy'] for r in temp_results])
    avg_precision_train = np.mean([r['precision_train'] for r in temp_results], axis=0)
    avg_recall_train = np.mean([r['recall_train'] for r in temp_results], axis=0)
    avg_f1_train = np.mean([r['f1_train'] for r in temp_results], axis=0)
    avg_precision_test = np.mean([r['precision_test'] for r in temp_results], axis=0)
    avg_recall_test = np.mean([r['recall_test'] for r in temp_results], axis=0)
    avg_f1_test = np.mean([r['f1_test'] for r in temp_results], axis=0)

    result_row = {
        'test_size': test_size,
        'avg_train_accuracy': avg_train_acc,
        'avg_test_accuracy': avg_test_acc
    }

    for idx, class_name in enumerate(class_names):
        result_row[f'avg_precision_train_{class_name}'] = avg_precision_train[idx]
        result_row[f'avg_recall_train_{class_name}'] = avg_recall_train[idx]
        result_row[f'avg_f1_train_{class_name}'] = avg_f1_train[idx]
        result_row[f'avg_precision_test_{class_name}'] = avg_precision_test[idx]
        result_row[f'avg_recall_test_{class_name}'] = avg_recall_test[idx]
        result_row[f'avg_f1_test_{class_name}'] = avg_f1_test[idx]

    all_results.append(result_row)

df_results = pd.DataFrame(all_results)
df_results.to_excel("results_avg.xlsx", index=False)
print(f"\nWyniki średnie zapisane do pliku: results_avg.xlsx")
