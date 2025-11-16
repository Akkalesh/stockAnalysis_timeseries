import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import *

df = pd.read_csv(CLEAN_NEWS)
analyzer = SentimentIntensityAnalyzer()

def vader(s):
    c = analyzer.polarity_scores(str(s))["compound"]
    if c >= 0.05: return 1
    if c <= -0.05: return -1
    return 0

df["sentiment_score"] = df["clean_text"].apply(vader)
df.to_csv(SENT_NEWS, index=False)
print("Saved:", SENT_NEWS)
