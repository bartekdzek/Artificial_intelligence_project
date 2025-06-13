import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

weather = pd.read_csv("weather_classification_data.csv")
weather.head()

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
y = cleaned_df["Weather Type"]

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values