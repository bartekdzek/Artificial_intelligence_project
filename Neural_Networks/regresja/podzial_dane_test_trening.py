import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from NeuralNetwork import  NeuralNetwork
from preprocessed_data import X_scaled, y_scaled, scaler_y

test_sizes = [0.1, 0.2, 0.3, 0.4]
repeats = 10
results = []

for test_size in test_sizes:
    train_mae_list, train_rmse_list, train_r2_list = [], [], []
    test_mae_list, test_rmse_list, test_r2_list = [], [], []

    for seed in range(repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=seed
        )

        model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=64, output_size=1)
        model.train(X_train, y_train, epochs=1000, learning_rate=0.01)


        y_train_pred_scaled = model.predict(X_train)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
        y_train_orig = scaler_y.inverse_transform(y_train)

        train_mae = mean_absolute_error(y_train_orig, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_train_pred))
        train_r2 = r2_score(y_train_orig, y_train_pred)

        train_mae_list.append(train_mae)
        train_rmse_list.append(train_rmse)
        train_r2_list.append(train_r2)


        y_test_pred_scaled = model.predict(X_test)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
        y_test_orig = scaler_y.inverse_transform(y_test)

        test_mae = mean_absolute_error(y_test_orig, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))
        test_r2 = r2_score(y_test_orig, y_test_pred)

        test_mae_list.append(test_mae)
        test_rmse_list.append(test_rmse)
        test_r2_list.append(test_r2)


    results.append({
        'Test Size': test_size,
        'Train Size': 1 - test_size,
        'Train MAE': np.mean(train_mae_list),
        'Train RMSE': np.mean(train_rmse_list),
        'Train R2': np.mean(train_r2_list),
        'Test MAE': np.mean(test_mae_list),
        'Test RMSE': np.mean(test_rmse_list),
        'Test R2': np.mean(test_r2_list),
    })


results_df = pd.DataFrame(results)
results_df.to_excel("podzial_dane_test_trening.xlsx", index=False)
print("\nŚrednie metryki po 10 powtórzeniach (dla treningu i testu):")
print(results_df)


plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
plt.plot(results_df['Test Size'], results_df['Train MAE'], marker='o', label='Train MAE')
plt.plot(results_df['Test Size'], results_df['Test MAE'], marker='o', label='Test MAE')
plt.xlabel('Test Size')
plt.ylabel('MAE')
plt.title('MAE vs Test Size')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(results_df['Test Size'], results_df['Train R2'], marker='o', label='Train R2')
plt.plot(results_df['Test Size'], results_df['Test R2'], marker='o', label='Test R2')
plt.xlabel('Test Size')
plt.ylabel('R² Score')
plt.title('R² Score vs Test Size')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


