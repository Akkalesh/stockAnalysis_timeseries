import pandas as pd
from config import *

df = pd.read_csv(MERGED_CSV, parse_dates=["date"])
df_fore = pd.read_csv(FORECAST_CSV, parse_dates=["date"])

# Export daily dataset for Power BI
df_daily = df.copy()
df_daily.to_csv(PBI_DAILY, index=False)

# Monthly summary
df_month = df.groupby(df["date"].dt.to_period("M")).agg(
    close_avg=("Close", "mean"),
    sentiment_avg=("sentiment_avg", "mean"),
    articles=("articles_count", "sum")
).reset_index()
df_month["date"] = df_month["date"].dt.to_timestamp()

df_month.to_csv(PBI_MONTHLY, index=False)

# Forecast file
df_fore.to_csv(PBI_FORECAST, index=False)

print("Power BI files exported:")
print(PBI_DAILY)
print(PBI_MONTHLY)
print(PBI_FORECAST)
