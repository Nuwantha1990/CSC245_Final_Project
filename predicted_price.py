import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Load data
df = pd.read_csv('stocks.csv')
df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime format
df.dropna(axis=0, inplace=True)

print(df.head())
print(df.shape)
print(df.info())

# Plotting Volume
df.plot(x='Date', y='Volume', marker='o')
plt.title("The Total Volume of Amazon Stock Each Day")
plt.ylabel("Volume")
plt.xlabel("Date")
plt.grid(True)
plt.show()

# Plotting Average Price
Average = (df['Open'] + df['Close'] + df['High'] + df['Low']) / 4
plt.plot(df['Date'], Average, label="Average price")
plt.title("Average Price of Amazon Stock Each Day")
plt.ylabel("Price")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.show()

# Data preprocessing and feature selection
X = df[['Open', 'High', 'Low', 'Volume']]  # Features
y = df['Close']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Get the last row of the dataset
latest_data = df.iloc[-1]

# Extract relevant values
open_price_today = latest_data['Open']
high_price_today = latest_data['High']
low_price_today = latest_data['Low']
volume_today = latest_data['Volume']

# Create new_data with estimated values for tomorrow
new_data = [[open_price_today, high_price_today, low_price_today, volume_today]]

# Predict tomorrow's stock price
predicted_price = model.predict(new_data)
print("Predicted Stock Price for Tomorrow:", predicted_price)
