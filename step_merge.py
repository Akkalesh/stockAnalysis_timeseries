import pandas as pd
import numpy as np
from config import *

# Load news with sentiment
news = pd.read_csv(SENT_NEWS)
news["date"] = pd.to_datetime(news["date"]).dt.date

# Load stock data
stock = pd.read_csv(STOCK_CSV)

# Fix date column
stock["date"] = pd.to_datetime(stock["date"]).dt.date

# Detect and rename close column
if "close" in stock.columns:
    stock.rename(columns={"close": "Close"}, inplace=True)
elif "Close" in stock.columns:
    pass
elif "adj close" in stock.columns:
    stock.rename(columns={"adj close": "Close"}, inplace=True)
else:
    print("❌ No close price column found. Columns:", stock.columns)
    raise SystemExit

# Aggregate sentiment per day
daily = news.groupby("date").agg(
    sentiment_avg=("sentiment_score", "mean"),
    articles_count=("clean_text", "count")
).reset_index()

# Merge stock + sentiment
df = pd.merge(stock, daily, on="date", how="left")
df["sentiment_avg"].fillna(0, inplace=True)
df["articles_count"].fillna(0, inplace=True)

# Create technical indicators
df["returns"] = df["Close"].pct_change().fillna(0)
df["MA_7"] = df["Close"].rolling(7).mean()
df["MA_21"] = df["Close"].rolling(21).mean()
df["volatility_10"] = df["returns"].rolling(10).std().fillna(0)

# Drop rows with NAs (from moving averages)
df.dropna(subset=["MA_7", "MA_21"], inplace=True)

# Save output
df.to_csv(MERGED_CSV, index=False)
print("Saved merged dataset:", MERGED_CSV)
print("Rows:", df.shape)
