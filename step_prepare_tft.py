import pandas as pd
import pickle
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from config import *

# Load merged dataset
df = pd.read_csv(MERGED_CSV, parse_dates=["date"])

# Sort by date
df = df.sort_values("date").reset_index(drop=True)

# Create continuous time index
df["time_idx"] = (df["date"] - df["date"].min()).dt.days

# Group ID (since you have only AAL)
df["company"] = "AAL"

# Target = Next day close
df["target"] = df["Close"].shift(-1)
df.dropna(subset=["target"], inplace=True)

# TFT hyperparameters
max_encoder_length = 60
max_prediction_length = 1

training = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="target",
    group_ids=["company"],
    allow_missing_timesteps=True,          # ⭐ IMPORTANT FIX ⭐
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=[
        "time_idx",
        "MA_7",
        "MA_21",
        "volatility_10",
        "sentiment_avg",
        "articles_count"
    ],
    time_varying_unknown_reals=[
        "target",
        "Close",
        "returns"
    ],
    target_normalizer=GroupNormalizer(groups=["company"])
)

# Save dataset for TFT training
with open(TRAIN_DATASET, "wb") as f:
    pickle.dump(training, f)

print("Saved TFT training dataset:", TRAIN_DATASET)
