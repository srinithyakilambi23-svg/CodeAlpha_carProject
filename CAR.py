
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv("car data.csv")

# Display First 5 Rows
print("FIRST 5 ROWS:")
print(df.head())


print("\nCOLUMN NAMES:")
print(df.columns)


print("\nMISSING VALUES:")
print(df.isnull().sum())



# Create Car Age Column
df['Car_Age'] = 2026 - df['Year']

# Drop Unnecessary Columns
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)


le = LabelEncoder()

df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])

df['Selling_type'] = le.fit_transform(df['Selling_type'])

df['Transmission'] = le.fit_transform(df['Transmission'])



X = df.drop('Selling_Price', axis=1)

y = df['Selling_Price']



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")

print("Mean Absolute Error:", mae)

print("Mean Squared Error:", mse)

print("Root Mean Squared Error:", rmse)

print("R2 Score:", r2)


plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Car Prices")

plt.show()