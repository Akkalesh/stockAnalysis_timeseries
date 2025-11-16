import pandas as pd
from config import *

df = pd.read_csv(STOCK_CSV)
print("\n=== Columns in AAL_stock.csv ===")
print(df.columns)
print("\nFirst 5 rows:")
print(df.head())
