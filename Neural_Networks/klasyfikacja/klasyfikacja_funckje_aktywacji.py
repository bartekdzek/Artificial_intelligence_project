import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from processdata import X, y
from NNetwork import SimpleNN


activation_functions = ['relu', 'sigmoid', 'tanh', 'linear']
epochs = 5000
learning_rate = 0.001
repeats = 5
test_size = 0.2


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.inverse_transform(np.arange(len(np.unique(y))))


results = []


for activation in activation_functions:
    print(f"\n Funkcja aktywacji: {activation}")

    for i in range(repeats):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=i, stratify=y_encoded
        )

        input_size = X_train.shape[1]
        output_size = len(np.unique(y_encoded))

        model = SimpleNN(
            input_size=input_size,
            output_size=output_size,
            epochs=epochs,
            learning_rate=learning_rate,
            activation=activation,
            random_state=i
        )
        model.fit(X_train, y_train, verbose=False)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)


        prec_train, rec_train, f1_train, _ = precision_recall_fscore_support(
            y_train, y_train_pred, labels=range(output_size), zero_division=0
        )
        prec_test, rec_test, f1_test, _ = precision_recall_fscore_support(
            y_test, y_test_pred, labels=range(output_size), zero_division=0
        )

        row = {
            'activation': activation,
            'repeat': i + 1,
            'test_size': len(y_test),
            'avg_train_accuracy': train_acc,
            'avg_test_accuracy': test_acc
        }

        for idx, class_name in enumerate(class_names):
            row[f'avg_precision_train_{class_name}'] = prec_train[idx]
            row[f'avg_recall_train_{class_name}'] = rec_train[idx]
            row[f'avg_f1_train_{class_name}'] = f1_train[idx]
            row[f'avg_precision_test_{class_name}'] = prec_test[idx]
            row[f'avg_recall_test_{class_name}'] = rec_test[idx]
            row[f'avg_f1_test_{class_name}'] = f1_test[idx]

        results.append(row)

        print(f"   Powtórzenie {i+1}: Test Acc={test_acc:.4f} | " +
              " | ".join([f"F1_test_{class_names[idx]}={f1_test[idx]:.2f}" for idx in range(output_size)]))


df_results = pd.DataFrame(results)
df_avg = df_results.groupby('activation').mean(numeric_only=True).reset_index()

df_results.to_csv("porownanie_detaliczne.csv", index=False)
df_avg.to_csv("porownanie_srednie.csv", index=False)

print("\n Pliki wynikowe zapisane:")
print(" - porownanie_detaliczne.xlsx")
print(" - porownanie_srednie.xlsx")
