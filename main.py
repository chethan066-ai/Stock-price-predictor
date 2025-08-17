import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Stock Data
stock = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
stock.reset_index(inplace=True)

# 2. Use 'Close' price for prediction
stock['Prev_Close'] = stock['Close'].shift(1)
stock = stock.dropna()

X = stock[['Prev_Close']]
y = stock['Close']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 7. Plot
plt.figure(figsize=(10,6))
plt.plot(stock['Date'].iloc[-len(y_test):], y_test, label="Actual Price")
plt.plot(stock['Date'].iloc[-len(y_pred):], y_pred, label="Predicted Price")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Apple Stock Price Prediction")
plt.savefig("screenshot.png")
plt.show()