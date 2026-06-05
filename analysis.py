import pandas 
import numpy as np
import matplotlib.pyplot as plt # グラフ描画ライブラリ

path = "./data/ETTm1.csv"

df = pandas.read_csv(path, parse_dates=["date"]) # csvファイルを読み込む (オブジェクトを返す)
df = df.sort_values("date").reset_index(drop=True) # 古い順に並び変える
df = df.set_index("date") # data列をindexに指定、dateベースで操作する

target_col = "OT" # 予測変数
feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"] # 説明変数

# ラグ特徴量(sarimaxのように時系列をから推測するモデル用に配列を整形)
df_feat = df.copy()
lag_step = [1, 2] # AR(2)
for i in lag_step:
    df_feat[f"OT_lag_{i}"] = df_feat["OT"].shift(i) # .shift = OT列のi個前の値そのものを取ってくる

# 差分特徴量
dif_steps = [1, 4, 96]
ot_history = df_feat["OT"].shift(1)
for step in dif_steps:
    df_feat[f"OT_diff_{step}"] = ot_history.diff(periods=step) # .diff = step個前の値との差分をとる

# 移動平均特徴量
rolling_windows = [4, 16, 96]
ot_history = df_feat["OT"].shift(1)
for window in rolling_windows:
    # rolling() = windowの数だけ今いる行の上から値をそのまま持ってくる　複数行持ってくる.shift()
    df_feat[f"OT_rooll_mean_{window}"] = ot_history.rolling(window=window, min_periods=window).mean() # mena() = 平均
    df_feat[f"OT_rooll_std_{window}"] = ot_history.rolling(window=window, min_periods=window).std() # std() = 標準偏差


# 周期特徴量(時間をラジアンに変換)
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek

total_munutes = df.index.hour * 60 + df.index.minute
df["hour_sin"] = np.sin(2.0 * np.pi * total_munutes / 1440.0)
df["hour_cos"] = np.cos(2.0 * np.pi * total_munutes / 1440.0)
df["dayofweek_sin"] = np.sin(2.0 * np.pi * df["dayofweek"] / 7.0)
df["dayofweek_cos"] = np.cos(2.0 * np.pi * df["dayofweek"] / 7.0)


"""
勾配ブースティング決定木(LightGBM)
"""
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# データ加工
df_lgbm = df_feat.copy()
df_lgbm["target_T"] = df_lgbm[target_col]
df_lgbm = df_lgbm.dropna()

y_lgbm = df_lgbm["target_T"]
x_lgbm = df_lgbm.drop(columns=["target_T", target_col])

# TimeSeriesSplit() = ジェネレータ
# 訓練用と検証用を渡すので
# 5分割なら、訓練用が1/6, 6/2, 6/3....、検証用は1/6固定
tss = TimeSeriesSplit(n_splits=5, gap=24) # 5分割, 15*24の空白時間を挟む
fold_scores = []

# split()は訓練用と検証用のデータindexを返す
# enumerate()はループindexとsplit()の出力を返す 
for fold, (train_idx, test_idx) in enumerate(tss.split(x_lgbm)):  
    x_train = x_lgbm.iloc[train_idx]
    y_train = y_lgbm.iloc[train_idx]
    x_test = x_lgbm.iloc[test_idx]
    y_test = y_lgbm.iloc[test_idx]

    # model定義
    lgbm_model = LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=1000,
        learning_rate=0.01, # 学習率
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0, # λ = 1 (正規化)
        random_state=42 + fold,
        n_jobs=-1
    )

    # model学習(得点ゲインの計算, 重み修正)
    lgbm_model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="l2", # 最小二乗法
        callbacks=[ # ヒストグラム化して高速化
            lgb.early_stopping(stopping_rounds=150, verbose=False), # 誤差が150回反復改善しなければ停止
            lgb.log_evaluation(period=0)
        ],
    )

    # predict() = 各行の予測値配列を返す
    y_pred_lgbm = lgbm_model.predict(x_test, num_iteration=lgbm_model.best_iteration_) # best_iteration_ = 最も誤差が小さかった時点

    mae = mean_absolute_error(y_test, y_pred_lgbm) # 予測値と実測値の残差平均
    mse = mean_squared_error(y_test, y_pred_lgbm) # 残差二乗平均
    rmse = np.sqrt(mse)

    # 最大値,最小値のずれ(3度以下で成功とする)
    residual_max = abs(y_pred_lgbm.max() - y_test.max())
    residual_min = abs(y_pred_lgbm.min() - y_test.min())
    result = "succeed" if (
        residual_max < 3 and residual_min < 3
    ) else "failed"

    fold_scores.append({
        "fold": fold,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "best_iteration": lgbm_model.best_iteration_,
        "best_score": lgbm_model.best_score_["valid_0"]["l2"],
        "residual_max": residual_max,
        "residual_min": residual_min,
        "result": result
    })

    print("\nfold: ", fold + 1)
    print("mae: ", mae)
    print("mse: ", mse)
    print("rmse: ",rmse)
    print("best_iteration: ", lgbm_model.best_iteration_)
    print("best_score: ", lgbm_model.best_score_["valid_0"]["l2"])
    print("residual_max: ", residual_max)
    print("residual_min: ", residual_min)
    print("result: ", result)

# result
score_df = pandas.DataFrame(fold_scores)
print("\n平均")
print(score_df[["mae", "mse", "rmse", "best_iteration", "best_score", "residual_max", "residual_min"]].mean())
print("\n標準偏差")
print(score_df[["mae", "mse", "rmse", "best_iteration", "best_score", "residual_max", "residual_min"]].std())

# 表出力
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:500], label="Actual value")
plt.plot(y_pred_lgbm[:500], label="Predicted value")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=12)
plt.show()
