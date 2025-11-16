import pandas as pd
import pmdarima as pm
import pickle
from config import *

# Load merged dataset
df = pd.read_csv(MERGED_CSV, parse_dates=["date"])

# Target variable
y = df["Close"]

# Exogenous regressors for ARIMAX
X = df[["sentiment_avg", "articles_count"]]

print("Training ARIMAX model...")

# AutoARIMA (p,d,q) search with external regressors
model = pm.auto_arima(
    y,
    exogenous=X,
    seasonal=False,
    stepwise=True,
    trace=True,
    suppress_warnings=True,
    max_p=5,
    max_q=5,
    max_d=2
)

print("\nBest model:", model.summary())

# Save model
with open(ARIMAX_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("Model saved to:", ARIMAX_MODEL_PATH)
