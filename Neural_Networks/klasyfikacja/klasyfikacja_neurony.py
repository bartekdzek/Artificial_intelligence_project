import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from processdata import X_train, X_test, y_train, y_test, y
from NNetwork import SimpleNN


neurons_list = [5, 10, 20, 50]
num_runs = 5
epochs = 10000
learning_rate = 0.001
class_names = np.unique(y)
output_size = len(class_names)
input_size = X_train.shape[1]

y_train_encoded, label_encoder = SimpleNN.encode_labels(y_train)
y_test_encoded = label_encoder.transform(y_test)
class_names = label_encoder.inverse_transform(np.arange(output_size))

results = []

for hidden_size in neurons_list:
    print(f"\n Neurony: {hidden_size}")

    metrics_all_runs = []

    for run in range(num_runs):
        model = SimpleNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            epochs=epochs,
            learning_rate=learning_rate,
            random_state=run
        )
        model.fit(X_train, y_train_encoded, verbose=False)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        acc_tr = accuracy_score(y_train_encoded, y_train_pred)
        acc_te = accuracy_score(y_test_encoded, y_test_pred)

        prec_tr, rec_tr, f1_tr, _ = precision_recall_fscore_support(
            y_train_encoded, y_train_pred, labels=range(output_size), zero_division=0
        )
        prec_te, rec_te, f1_te, _ = precision_recall_fscore_support(
            y_test_encoded, y_test_pred, labels=range(output_size), zero_division=0
        )

        metrics_all_runs.append({
            "train_acc": acc_tr,
            "test_acc": acc_te,
            "train_prec": prec_tr,
            "train_rec": rec_tr,
            "train_f1": f1_tr,
            "test_prec": prec_te,
            "test_rec": rec_te,
            "test_f1": f1_te
        })


    def avg_metric(metric_name, class_idx):
        return np.mean([run[metric_name][class_idx] for run in metrics_all_runs])

    result = {"Neurony": hidden_size}
    result["avg_train_acc"] = np.mean([run["train_acc"] for run in metrics_all_runs])
    result["avg_test_acc"] = np.mean([run["test_acc"] for run in metrics_all_runs])

    for i, class_name in enumerate(class_names):
        result[f"precision_train_{class_name}"] = avg_metric("train_prec", i)
        result[f"recall_train_{class_name}"] = avg_metric("train_rec", i)
        result[f"f1_train_{class_name}"] = avg_metric("train_f1", i)
        result[f"precision_test_{class_name}"] = avg_metric("test_prec", i)
        result[f"recall_test_{class_name}"] = avg_metric("test_rec", i)
        result[f"f1_test_{class_name}"] = avg_metric("test_f1", i)

    results.append(result)

df_res = pd.DataFrame(results)
df_res.to_csv("porownanie_neuronow_klasyfikacja.csv", index=False)

print("\nZapisano plik 'porownanie_neuronow_klasyfikacja.csv'")
print(df_res[["Neurony", "avg_train_acc", "avg_test_acc"]].round(3))
