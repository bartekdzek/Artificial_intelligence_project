import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessed_data import X_train, X_test, y_train, y_test, scaler_y
from NeuralNetwork import NeuralNetwork, train_batch, train_minibatch, train_adam, train_rmsprop

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, mape, r2

methods = [
    ("Batch Gradient Descent", train_batch, {"learning_rate": 0.001}),
    ("Mini-Batch Gradient Descent", train_minibatch, {"batch_size": 64, "learning_rate": 0.001}),
    ("Adam Optimizer", train_adam, {"learning_rate": 0.001}),
    ("RMSProp", train_rmsprop, {"learning_rate": 0.001})  
]


epochs = 1000
runs = 5

results = []

for name, train_func, extra_params in methods:
    print(f"\n{'='*60}")
    print(f"Training with method: {name}")
    print(f"{'='*60}")

    train_metrics_list = []
    test_metrics_list = []

    for run in range(runs):
        print(f" Run {run+1}/{runs}")

        model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=64, output_size=1)

        train_func(model, X_train, y_train, epochs=epochs, **extra_params)

        train_preds = scaler_y.inverse_transform(model.predict(X_train))
        test_preds = scaler_y.inverse_transform(model.predict(X_test))
        y_train_true = scaler_y.inverse_transform(y_train)
        y_test_true = scaler_y.inverse_transform(y_test)

        train_metrics = calculate_metrics(y_train_true, train_preds)
        test_metrics = calculate_metrics(y_test_true, test_preds)

        train_metrics_list.append(train_metrics)
        test_metrics_list.append(test_metrics)

    train_avg = np.mean(train_metrics_list, axis=0)
    test_avg = np.mean(test_metrics_list, axis=0)

    results.append({
        "Method": name,
        "Set": "Train",
        "MSE": train_avg[0],
        "RMSE": train_avg[1],
        "MAE": train_avg[2],
        "MAPE (%)": train_avg[3],
        "R²": train_avg[4]
    })

    results.append({
        "Method": name,
        "Set": "Test",
        "MSE": test_avg[0],
        "RMSE": test_avg[1],
        "MAE": test_avg[2],
        "MAPE (%)": test_avg[3],
        "R²": test_avg[4]
    })

results_df = pd.DataFrame(results)
results_df.to_excel("analiza_metod.xlsx", index=False)
print("\nWyniki zapisane do pliku: analiza_metod.xlsx")
