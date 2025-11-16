import pandas as pd
import numpy as np
import pickle
from config import *

# Load model
with open(ARIMAX_MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load merged dataset
df = pd.read_csv(MERGED_CSV, parse_dates=["date"])

# Last known date
last_date = df["date"].max()

# Future dates (30 days)
future_dates = pd.date_range(start=last_date, periods=31, freq="D")[1:]

# Future sentiment assumption: neutral (0)
future_exog = pd.DataFrame({
    "sentiment_avg": np.zeros(30),
    "articles_count": np.zeros(30)
})

# Forecast next 30 days
forecast = model.predict(n_periods=30, exogenous=future_exog)

# Create forecast dataframe
df_forecast = pd.DataFrame({
    "date": future_dates,
    "forecast_close": forecast
})

df_forecast.to_csv(FORECAST_CSV, index=False)
print("Saved forecast:", FORECAST_CSV)
