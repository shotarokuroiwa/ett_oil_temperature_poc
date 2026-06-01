import pandas as pd
import numpy as np

path = "./data/ETTm1.csv"

df = pd.read_csv(path, parse_dates=["date"]) # csvファイルを読み込む
df = df.sort_values("date").reset_index(drop=True) # 古い順に並び変える
df = df.set_index("date") # data列をindexに指定、dateベースで操作する

target_col = "OT" # 予測変数
feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"] # 説明変数

print(df.head()) # データの先頭５行を表示
print(df.dtypes)
print(df.index.min(), df.index.max())
print(df.isna().sum())

# ADF
from statsmodels.tsa.stattools import adfuller

y = df["OT"].dropna() # NaNがあるデータを削除する
# y = df["OT"].diff().dropna() 非定常なら変化量の変化量を見る(diff())

# d = 1のみ自動
adf_result = adfuller(
    y, #予測対象
    maxlag=None,
    regression="c", # ΔYt-Yt-1 = α + ε(α(右肩上がり、右肩下がりのトレンド)がある場合の引数)
    autolag="AIC" # AR(p)のpを最適値を自動で探す
)

print("adf統計量: ", adf_result[0]) # γの値(臨界値5%のメモリより低ければ定常)
print("p値: ", adf_result[1]) # γ=0である確率(p<0.05であれば定常)
print("採用ラグ:", adf_result[2])
print("有効観測数:", adf_result[3])
print("臨界値: {") # このデータ群における1%, 5%, 10% のメモリ
critical_values = adf_result[4]
for key, value in critical_values.items():
    print(f"  {key}: {value}")
print("}")
print("選ばれた情報量基準値:", adf_result[5])


