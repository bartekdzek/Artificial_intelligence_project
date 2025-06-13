import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessed_data import X_train, X_test, y_train, y_test, scaler_y
from NeuralNetwork import model


activation_functions = ['relu', 'sigmoid', 'tanh', 'linear']


num_runs = 5


results = []


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100


for activation in activation_functions:
    print(f"\n🔍 Analiza dla funkcji aktywacji: {activation.upper()}")
    
    all_metrics_train, all_metrics_test = [], []

    for run in range(num_runs):
        print(f"  ➤ Iteracja {run+1}/{num_runs}")


        net = model(activation=activation)
        net.train(X_train, y_train)


        y_train_pred_scaled = net.predict(X_train)
        y_test_pred_scaled = net.predict(X_test)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
        y_train_true = scaler_y.inverse_transform(y_train)
        y_test_true = scaler_y.inverse_transform(y_test)

        def compute_metrics(y_true, y_pred):
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            return [mse, rmse, mae, r2, mape]

        metrics_train = compute_metrics(y_train_true, y_train_pred)
        metrics_test = compute_metrics(y_test_true, y_test_pred)

        all_metrics_train.append(metrics_train)
        all_metrics_test.append(metrics_test)


    metrics_train_avg = np.mean(all_metrics_train, axis=0)
    metrics_test_avg = np.mean(all_metrics_test, axis=0)


    results.append({
        "Aktywacja": activation,
        "Zbiór": "Train",
        "MSE": metrics_train_avg[0],
        "RMSE": metrics_train_avg[1],
        "MAE": metrics_train_avg[2],
        "R2": metrics_train_avg[3],
        "MAPE (%)": metrics_train_avg[4]
    })
    results.append({
        "Aktywacja": activation,
        "Zbiór": "Test",
        "MSE": metrics_test_avg[0],
        "RMSE": metrics_test_avg[1],
        "MAE": metrics_test_avg[2],
        "R2": metrics_test_avg[3],
        "MAPE (%)": metrics_test_avg[4]
    })


df_results = pd.DataFrame(results)
df_results.to_csv("porownanie_aktywcji.csv", index=False)

print("\nWyniki zapisane do pliku 'porownanie_aktywcji.csv'")
print(df_results)
