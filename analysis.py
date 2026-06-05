import pandas as pd
import numpy as np

path = "./data/ETTm1.csv"

df = pd.read_csv(path, parse_dates=["date"]) # csvファイルを読み込む (オブジェクトを返す)
df = df.sort_values("date").reset_index(drop=True) # 古い順に並び変える
df = df.set_index("date") # data列をindexに指定、dateベースで操作する

target_col = "OT" # 予測変数
feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"] # 説明変数

print(df.head()) # データの先頭５行を表示
print(df.dtypes)
print(df.index.min(), df.index.max())
print(df.isna().sum())

# # ADF
from statsmodels.tsa.stattools import adfuller
#
# y = df["OT"].() # NaNがあるデータを削除する
# # y = df["OT"].diff().() 非定常なら変化量の変化量を見る(diff())
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
# # ACF PACF
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14,8)) # fig = 全体のオブジェクト, axes =　二つのグラフの配列
#
# plot_acf(
#     df["OT"].(),
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
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
#
# minmax_scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=False)
# scaled_values = minmax_scaler.fit_transform(df[feature_cols])
# scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=feature_cols)
#
# print(scaled_df.head())
# print(minmax_scaler.data_min_)
# print(minmax_scaler.data_max_)
#
# # 標準化
# standard_scaler = StandardScaler(with_mean=True, with_std=True)
# standard_values = standard_scaler.fit_transform(df[feature_cols])
# standard_df = pd.DataFrame(standard_values, index=df.index, columns=feature_cols)
#
# print(standard_df.head())
# print(standard_scaler.mean_)
# print(standard_scaler.scale_)
#
# # 外れ値の影響を数値化
# x = np.array([[10.0], [11.0], [12.0], [13.0], [100.0]])
#
# minmax = MinMaxScaler(feature_range=(0, 1))
# standard = StandardScaler()
#
# print("\nMinMax: ", minmax.fit_transform(x).ravel())
# print("Standard: ", standard.fit_transform(x).ravel()) #　標準化は外れ値が存在しても、間隔は担保している
#
#
# # テンソル
# def make_sliding_windows_numpy(
#     data: np.ndarray, # 全サンプルデータ
#     target: np.ndarray, # 予測対象
#     input_length: int, # AR(p)のp
#     forecast_horizon: int # 何ステップ予測するか
# ) -> tuple[np.ndarray, np.ndarray]:
#     x_list = [] # 行
#     y_list = [] # 列
#     total_length = len(data)
#     max_start = total_length - input_length - forecast_horizon + 1
#
#     # 一回のループである時間t, t-1, t-2 × 説明変数 の行列ができる
#     for start_idx in range(max_start):
#         end_idx = start_idx + input_length
#         target_end_idx = end_idx + forecast_horizon
#         x_window = data[start_idx:end_idx, :] # 左側: a:bでa~bまでの行, 右側: :ですべての列
#         y_window = target[end_idx:target_end_idx]
#         x_list.append(x_window)
#         y_list.append(y_window)
#
#     # この時点でx_listはp × 特徴量の行列の配列のなっている
#     x = np.stack(x_list, axis=0) # stackはテンソルにし、[ループ数, 行, 列]でアクセスできる
#     y = np.stack(y_list, axis=0)
#     return x, y
#
# input_length = 96
# forecast_horizon = 24
#
# scaler = StandardScaler()
# values_scaled = scaler.fit_transform(df[feature_cols].values) # .values = 見出しを取り除いて、純粋は配列だけを返す
# target_values = df[target_col].values
#
# x_np, y_np = make_sliding_windows_numpy(
#     data=values_scaled,
#     target=target_values,
#     input_length=input_length,
#     forecast_horizon=forecast_horizon
# )
#
# print("\nx_np shalpe: ", x_np.shape) # .shape = テンソルの大きさを[z, x, y]で出力
# print("y_np shalpe: ", y_np.shape)
# print("一回目, 2行目, 4列目: ", x_np[0, 1, 4]) 
#
# """
# pytorchによるテンソル化
#
# """
#
# # ラグ特徴量(sarimaxのように時系列をから推測するモデル用に配列を整形)
# df_feat = df.copy()
# lag_step = [1, 2] # AR(2)
# for i in lag_step:
#     df_feat[f"OT_lag_{i}"] = df_feat["OT"].shift(i) # .shift = OT列のi個前の値そのものを取ってくる
#
# # 差分特徴量
# dif_steps = [1, 4, 96]
# ot_history = df_feat["OT"].shift(1)
# for step in dif_steps:
#     df_feat[f"OT_diff_{step}"] = ot_history.diff(periods=step) # .diff = step個前の値との差分をとる
#
# # 移動平均特徴量
# rolling_windows = [4, 16, 96]
# ot_history = df_feat["OT"].shift(1)
# for window in rolling_windows:
#     # rolling() = windowの数だけ今いる行の上から値をそのまま持ってくる　複数行持ってくる.shift()
#     df_feat[f"OT_rooll_mean_{window}"] = ot_history.rolling(window=window, min_periods=window).mean() # mena() = 平均
#     df_feat[f"OT_rooll_std_{window}"] = ot_history.rolling(window=window, min_periods=window).std() # std() = 標準偏差
#
# print("\n",df_feat.head(20))
#
# # 周期特徴量(時間をラジアンに変換)
# df["minute"] = df.index.minute
# df["hour"] = df.index.hour
# df["dayofweek"] = df.index.dayofweek
#
# total_munutes = df.index.hour * 60 + df.index.minute
# df["minute_sin"] = np.sin(2.0 * np.pi * df["minute"] / 60.0) # sinとconに分けるベクトルを一マスに入れれないから
# df["minute_cos"] = np.cos(2.0 * np.pi * df["minute"] / 60.0)
# df["hour_sin"] = np.sin(2.0 * np.pi * total_munutes / 1440.0)
# df["hour_cos"] = np.cos(2.0 * np.pi * total_munutes / 1440.0)
# df["dayofweek_sin"] = np.sin(2.0 * np.pi * df["dayofweek"] / 7.0)
# df["dayofweek_cos"] = np.cos(2.0 * np.pi * df["dayofweek"] / 7.0)
#
# print(df.head(30))
#
"""
SARIMAX
"""
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

def estimate_d_by_adf(series: pd.Series, max_d: int = 2, alpha: float = 0.05) -> int:
    current = series.dropna()
    for d in range(max_d + 1):
        result = adfuller(current, maxlag=None, regression="c", autolag="AIC")
        p_value = result[1]
        print(f"d={d}, p-value={p_value}")
        if p_value < alpha:
            return d
        current = current.diff().dropna()
    return max_d

# (1 - L)ᵈ
d = estimate_d_by_adf(df["OT"], max_d=2, alpha=0.05) # 何回変化量をとったか(定常になるまで)
s = 96

# (1 - L⁹⁶)ᵒ
seasonal_diff = df["OT"].diff(periods=s).dropna() # diff() = 一周期前のデータを引くと季節やトレンドの波を消える
seasonal_p_value = adfuller(seasonal_diff, maxlag=None, regression="c", autolag="AIC")[1] # p値
D = 1 if seasonal_p_value < 0.05 else 0

# 定常化
stationary_data = df["OT"].copy() 
for _ in range(d):
    stationary_data = stationary_data.diff()
stationary_data = stationary_data.dropna() # 空白の行を削除

# acf, pacf
acf_values = acf(stationary_data, nlags=20, fft=True) # acf()は配列を返す ⇔  plot_acf()はグラフを返す
pacf_values = pacf(stationary_data, nlags=20, method="ywm")

print("PACF_VALUE = \n", pacf_values)

threshold = 1.96 / np.sqrt(len(stationary_data)) # 信頼区間(これより外に出ると偶然のノイズではない)
print("信頼区間", threshold)

p_candidates = [lag for lag in range(1, 6) if abs(pacf_values[lag]) > threshold] 
q_candidates = [lag for lag in range(1, 6) if abs(acf_values[lag]) > threshold]
p = max(p_candidates) if p_candidates else 1
q = max(q_candidates) if q_candidates else 1

print("\norder", (p, d, q))
print("seasonal_order", (1, D, 1, 96))

y = df["OT"].asfreq("15min").interpolate(method="time")
exog_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"] # 外部変数
exog = df[exog_cols].asfreq("15min").interpolate(method="time")

# 訓練用8割, 答え合わせ用2割
train_size = int(len(y) * 0.8)
# 目的変数
y_train = y.iloc[:train_size] # :a = indexがaまで 
y_test = y.iloc[train_size:] # a: = indexがaから最後まで
exog_train = exog.iloc[:train_size]
exog_test = exog.iloc[:train_size:]

# SIRIMAX()
model = SARIMAX(
    endog=y_train,
    exog=exog_train,
    order=(p, d, q), # 次数(自己回帰, 変化量の変化量,　移動平均)
    seasonal_order=(1, D, 1, 96,), # (一期前の変化量の重み, D, 一期前のノイズの重み, 周期)
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False
)

# model学習
# fit() = 微分, 二回微分を実行し最適された重みの確定値を出力
result = model.fit(disp=False, maxiter=200) # maxiter = 何回重み更新を行うか
print(result.summary()) # 最適化した重みのレポート

# model評価
forecast_reuslt = result.get_forecast(steps=len(y_test), exog=exog_test) # 
y_pred_sarimax = forecast_reuslt.predicted_mean() # 確率分布で一番確率が高い予測値
conf_int = forecast_reuslt.conf_int(alpha=0.05) # 95%の確率でこの範囲に収まるという信頼区間(最大最小)

