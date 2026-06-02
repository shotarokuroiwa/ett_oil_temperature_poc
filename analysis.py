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

# STL
import matplotlib.pyplot as plt # グラフ描画ライブラリ
from statsmodels.tsa.seasonal import STL

y = df["OT"].asfreq("15min") # 15分間隔に統一されたオブジェクトとを返す
y = y.interpolate(method="time", limit_direction="both") # asfreqで保管したNaN行を前後から推測して埋める

# STLクラス（インスタンスを作るだけ)
stl = STL( 
        y,
        period=96,
        seasonal=13, # 近傍幅
        trend=None, # トレンド周期を自動判定
        robust=True # 大きな差異を持つ値の重みを小さくする
)

stl_result = stl.fit() # Tt + St + Rtに分解
trend = stl_result.trend # 長期トレンド
seasonal = stl_result.seasonal # # 周期変動
resid = stl_result.resid # 残差（ノイズ）

fig = stl_result.plot() # グラフ作成(メモリ上)
fig.set_size_inches(12,8) # グラフのサイズを設定
plt.show() # 画面に描画

#残差チェック
resid_check = y - trend - seasonal
print(np.nanmean(np.abs(resid_check - resid)))

# ACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14,8)) # fig = 部全体のオブジェクト, axes =　二つのグラフの配列

plot_acf(
    df["OT"].dropna(),
    lags=192, # 一周先(TtとTt-s)のデータに回帰性がないか
    alpha=0.05, # 有意水準
    ax=axes[0], #描画先
    title="ACF of OT"
)

plot_pacf(
    df["OT"],
    lags=192,
    alpha=0.05,
    method="ywm",
    ax=axes[1],
    title="PACF of OT"
)

plt.tight_layout()
plt.show()


