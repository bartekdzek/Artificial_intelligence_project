

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessed_data import X_train, X_test, y_train, y_test, scaler_y
from NeuralNetwork import NeuralNetwork


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(
        np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))
    ) * 100

def compute_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mse, rmse, mae, r2, mape


neurons_list    = [5, 10, 20, 50]
num_runs        = 5
epochs          = 1000
learning_rate   = 0.01


results = []

for neurons in neurons_list:
    print(f"\n Hidden size = {neurons}")
    train_metrics, test_metrics = [], []

    for run in range(1, num_runs + 1):
        print(f"  run {run}/{num_runs}", end="\r")


        net = NeuralNetwork(
            input_size=X_train.shape[1],
            hidden_size=neurons,
            output_size=1,
            activation="relu"
        )


        net.train(
            X_train, y_train,
            epochs=epochs,
            learning_rate=learning_rate
        )


        y_train_pred = scaler_y.inverse_transform(net.predict(X_train))
        y_test_pred  = scaler_y.inverse_transform(net.predict(X_test))
        y_train_true = scaler_y.inverse_transform(y_train)
        y_test_true  = scaler_y.inverse_transform(y_test)


        train_metrics.append(compute_metrics(y_train_true, y_train_pred))
        test_metrics.append(compute_metrics(y_test_true,  y_test_pred))


    m_train = np.mean(train_metrics, axis=0)
    m_test  = np.mean(test_metrics,  axis=0)


    results.extend([
        {"Neurony": neurons, "Zbiór": "Train",
         "MSE": m_train[0], "RMSE": m_train[1], "MAE": m_train[2],
         "R2": m_train[3],  "MAPE (%)": m_train[4]},
        {"Neurony": neurons, "Zbiór": "Test",
         "MSE": m_test[0], "RMSE": m_test[1], "MAE": m_test[2],
         "R2": m_test[3],  "MAPE (%)": m_test[4]},
    ])


df_results = pd.DataFrame(results)
df_results.to_csv("porownanie_neuronow.csv", index=False)

print("\n  Wyniki zapisane do pliku 'porownanie_neuronow.csv'")
print(df_results.to_string(index=False))
