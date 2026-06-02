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

# # ADF
# from statsmodels.tsa.stattools import adfuller
#
# y = df["OT"].dropna() # NaNがあるデータを削除する
# # y = df["OT"].diff().dropna() 非定常なら変化量の変化量を見る(diff())
#
# # d = 1のみ自動
# adf_result = adfuller(
#     y, #予測対象
#     maxlag=None,
#     regression="c", # ΔYt-Yt-1 = α + ε(α(右肩上がり、右肩下がりのトレンド)がある場合の引数)
#     autolag="AIC" # AR(p)のpを最適値を自動で探す
# )
#
# print("adf統計量: ", adf_result[0]) # γの値(臨界値5%のメモリより低ければ定常)
# print("p値: ", adf_result[1]) # γ=0である確率(p<0.05であれば定常)
# print("採用ラグ:", adf_result[2])
# print("有効観測数:", adf_result[3])
# print("臨界値: {") # このデータ群における1%, 5%, 10% のメモリ
# critical_values = adf_result[4]
# for key, value in critical_values.items():
#     print(f"  {key}: {value}")
# print("}")
# print("選ばれた情報量基準値:", adf_result[5])
#
# # STL
# import matplotlib.pyplot as plt # グラフ描画ライブラリ
# from statsmodels.tsa.seasonal import STL
#
# y = df["OT"].asfreq("15min") # 15分間隔に統一されたオブジェクトとを返す
# y = y.interpolate(method="time", limit_direction="both") # asfreqで保管したNaN行を前後から推測して埋める
#
# # STLクラス（インスタンスを作るだけ)
# stl = STL( 
#         y,
#         period=96,
#         seasonal=13, # 近傍幅
#         trend=None, # トレンド周期を自動判定
#         robust=True # 大きな差異を持つ値の重みを小さくする
# )
#
# stl_result = stl.fit() # Tt + St + Rtに分解
# trend = stl_result.trend # 長期トレンド
# seasonal = stl_result.seasonal # # 周期変動
# resid = stl_result.resid # 残差（ノイズ）
#
# fig = stl_result.plot() # グラフ作成(メモリ上)
# fig.set_size_inches(12,8) # グラフのサイズを設定
# plt.show() # 画面に描画
#
# #残差チェック
# resid_check = y - trend - seasonal
# print(np.nanmean(np.abs(resid_check - resid)))
#
# # ACF
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14,8)) # fig = 部全体のオブジェクト, axes =　二つのグラフの配列
#
# plot_acf(
#     df["OT"].dropna(),
#     lags=192, # 一周先(TtとTt-s)のデータに回帰性がないか
#     alpha=0.05, # 有意水準
#     ax=axes[0], #描画先
#     title="ACF of OT"
# )
#
# # AR(p)のpを選定
# plot_pacf(
#     df["OT"],
#     lags=192,
#     alpha=0.05,
#     method="ywm",
#     ax=axes[1],
#     title="PACF of OT"
# )
#
# plt.tight_layout()
# plt.show()


# 最大最小スケーリング
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

minmax_scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=False)
scaled_values = minmax_scaler.fit_transform(df[feature_cols])
scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=feature_cols)

print(scaled_df.head())
print(minmax_scaler.data_min_)
print(minmax_scaler.data_max_)

# 標準化
standard_scaler = StandardScaler(with_mean=True, with_std=True)
standard_values = standard_scaler.fit_transform(df[feature_cols])
standard_df = pd.DataFrame(standard_values, index=df.index, columns=feature_cols)

print(standard_df.head())
print(standard_scaler.mean_)
print(standard_scaler.scale_)

# 外れ値の影響を数値化
x = np.array([[10.0], [11.0], [12.0], [13.0], [100.0]])

minmax = MinMaxScaler(feature_range=(0, 1))
standard = StandardScaler()

print("\nMinMax: ", minmax.fit_transform(x).ravel())
print("Standard: ", standard.fit_transform(x).ravel()) #　標準化は外れ値が存在しても、間隔は担保している


# テンソル
def make_sliding_windows_numpy(
    data: np.ndarray, # 全サンプルデータ
    target: np.ndarray, # 予測対象
    input_length: int, # AR(p)のp
    forecast_horizon: int # 何ステップ予測するか
) -> tuple[np.ndarray, np.ndarray]:
    x_list = [] # 行
    y_list = [] # 列
    total_length = len(data)
    max_start = total_length - input_length - forecast_horizon + 1

    # 一回のループである時間t, t-1, t-2 × 説明変数 の行列ができる
    for start_idx in range(max_start):
        end_idx = start_idx + input_length
        target_end_idx = end_idx + forecast_horizon
        x_window = data[start_idx:end_idx, :] # 左側: a:bでa~bまでの行, 右側: :ですべての列
        y_window = target[end_idx:target_end_idx]
        x_list.append(x_window)
        y_list.append(y_window)

    # この時点でx_listはp × 特徴量の行列の配列のなっている
    x = np.stack(x_list, axis=0) # stackはテンソルにし、[ループ数, 行, 列]でアクセスできる
    y = np.stack(y_list, axis=0)
    return x, y

input_length = 96
forecast_horizon = 24

scaler = StandardScaler()
values_scaled = scaler.fit_transform(df[feature_cols].values)
target_values = df[target_col].values

x_np, y_np = make_sliding_windows_numpy(
    data=values_scaled,
    target=target_values,
    input_length=input_length,
    forecast_horizon=forecast_horizon
)

print("\nx_np shalpe: ", x_np.shape)
print("y_np shalpe: ", y_np.shape)
print("一回目, 2行目, 4列目: ", x_np[0, 1, 4]) 

"""
pytorchによるテンソル化

"""

