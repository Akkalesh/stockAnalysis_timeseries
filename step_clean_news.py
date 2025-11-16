import polars as pl
import re
from config import *

df = pl.read_csv(EXTRACTED_NEWS)

# Remove Anglo American
df = df.filter(~pl.col("Stock_symbol").cast(pl.Utf8).str.contains("AAL.L", literal=False))
df = df.filter(~pl.col("Article_title").cast(pl.Utf8).str.contains("Anglo American", literal=False))
df = df.filter(~pl.col("Article").cast(pl.Utf8).str.contains("Anglo American", literal=False))

# Fix date column
date_col = [c for c in df.columns if "date" in c.lower()][0]
df = df.with_columns([
    pl.col(date_col)
      .str.replace(" UTC", "")
      .str.strptime(pl.Datetime, strict=False)
      .dt.date()
      .alias("date")
])

# Clean text
def clean(x):
    if x is None: return ""
    x = re.sub(r"http\S+", "", str(x))
    x = re.sub(r"[^a-zA-Z0-9\s]", " ", x)
    return re.sub(r"\s+", " ", x).strip()

df = df.with_columns([
    pl.col("Article").map_elements(clean).alias("clean_text"),
    pl.col("Article_title").map_elements(clean).alias("clean_title")
])

df = df.unique(subset=["date","clean_title"])
df.write_csv(CLEAN_NEWS)
print("Saved:", CLEAN_NEWS, "Rows:", df.shape)
