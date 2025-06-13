import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from processdata import X_train, X_test, y_train, y_test, y


from NNetwork import (
    SimpleNN,
    train_batch_gradient_descent,
    train_mini_batch_gradient_descent,
    train_adagrad,
    train_momentum,
)

def prepare_data():
    y_train_encoded, label_encoder = SimpleNN.encode_labels(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    input_size = X_train.shape[1]
    output_size = len(np.unique(y))
    return y_train_encoded, y_test_encoded, input_size, output_size, label_encoder

def evaluate_model(model_func, method_name, X_train, y_train_encoded, X_test, y_test_encoded, input_size, output_size, label_encoder):
    model = model_func(X_train, y_train_encoded, X_test, y_test_encoded, input_size, output_size)

    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)

    y_train_labels = label_encoder.inverse_transform(y_train_encoded)
    y_test_labels = label_encoder.inverse_transform(y_test_encoded)
    y_train_pred_labels = label_encoder.inverse_transform(y_train_preds)
    y_test_pred_labels = label_encoder.inverse_transform(y_test_preds)

    train_report = classification_report(y_train_labels, y_train_pred_labels, output_dict=True, zero_division=0)
    test_report = classification_report(y_test_labels, y_test_pred_labels, output_dict=True, zero_division=0)

    results = {
        "learning_method": method_name,
        "avg_train_accuracy": accuracy_score(y_train_encoded, y_train_preds),
        "avg_test_accuracy": accuracy_score(y_test_encoded, y_test_preds),
    }

    classes = ["Cloudy", "Rainy", "Snowy", "Sunny"]
    for cls in classes:
        for dataset, report in [("train", train_report), ("test", test_report)]:
            results[f"avg_precision_{dataset}_{cls}"] = report.get(cls, {}).get("precision", 0)
            results[f"avg_recall_{dataset}_{cls}"] = report.get(cls, {}).get("recall", 0)
            results[f"avg_f1_{dataset}_{cls}"] = report.get(cls, {}).get("f1-score", 0)

    return results

if __name__ == "__main__":
    y_train_encoded, y_test_encoded, input_size, output_size, label_encoder = prepare_data()

    all_results = []
    repeats = 5

    methods = [
        
         ("Batch Gradient Descent", train_batch_gradient_descent),
        ("Mini-Batch Gradient Descent", train_mini_batch_gradient_descent),
        ("AdaGrad", train_adagrad),
        ("Momentum",train_momentum),
    ]

    for name, method in methods:
        print(f"\n--- Training with: {name} ---")
        repeated_results = []

        for i in range(repeats):
            print(f"   Repeat {i + 1}/{repeats}")
            result = evaluate_model(method, name, X_train, y_train_encoded, X_test, y_test_encoded, input_size, output_size, label_encoder)
            if result:
                repeated_results.append(result)

        avg_result = {
            key: np.mean([r[key] for r in repeated_results])
            for key in repeated_results[0].keys()
            if key != "learning_method"
        }
        avg_result["learning_method"] = name
        all_results.append(avg_result)


    results_df = pd.DataFrame(all_results)

cols = ['learning_method'] + [col for col in results_df.columns if col != 'learning_method']
results_df = results_df[cols]

results_df.to_excel("analiza_metod_klas2.xlsx", index=False)
print("\nŚrednie wyniki zapisane do pliku: analiza_metod_klas.xlsx")

