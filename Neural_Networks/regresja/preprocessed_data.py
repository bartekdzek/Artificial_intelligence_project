import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('diamonds.csv')
df = df.drop_duplicates()
df = df.rename(columns={'x': 'length_in_mm', 'y': 'width_in_mm', 'z': 'depth_in_mm'})


df['cut'] = df['cut'].map({'Ideal': 4, 'Premium': 3, 'Very Good': 2, 'Good': 1, 'Fair': 0})
df['color'] = df['color'].map({'D': 6, 'E': 5, 'F': 4, 'G': 3, 'H': 2, 'I': 1, 'J': 0})
df['clarity'] = df['clarity'].map({'IF': 7, 'VVS1': 6, 'VVS2': 5, 'VS1': 4, 'VS2': 3, 'SI1': 2, 'SI2': 1, 'I1': 0})

X = df.drop(columns='price').values
y = df['price'].values.reshape(-1, 1)


scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)