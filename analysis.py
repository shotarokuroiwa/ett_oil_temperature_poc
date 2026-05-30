import pandas as pd
import numpy as np

path = "./data/ETTm1.csv"

df = pd.read_csv(path, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df.set_index("date")

target_col = "OT"
feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

print(df.head())
print(df.dtypes())
print(df.index.min(), df.index.max())
print(df.isna().sum())
