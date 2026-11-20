"""
勾配ブースティング決定木を用いた分類モデル
LightGBM使用
"""

import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import lightgbm as lgb


# ============================================================
# Setting
# ============================================================
RANDOM_STATE = 42

# # Early Stopping
# PATIENCE = 5       
# PERIOD = 10 #表示頻度

# LEARNING_RATE = 0.05
# ESTIMATORS = 1000


# ==========================================
# データ読み込み・前処理
# ==========================================
# Excel形式で保存されている場合
df = pd.read_excel("data/Concrete_Data.xlsx")

# ()内の記述削除
df.columns = df.columns.str.replace(r"\s*\([^)]*\)", "", regex=True)


X = df.drop(columns=["Concrete compressive strength "])
Y = df["Concrete compressive strength "]


# クラス数
OUTPUT_DIM = 1



# NumPy配列に変換
X = X.values.astype(np.float32)
Y = Y.astype(np.float32)


print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")


# 訓練データと検証データに分割（8:2）
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# LightGBM Datasetの作成
lgb_train = lgb.Dataset(X_train, label=Y_train)
# lgb_valid = lgb.Dataset(X_va, label=Y_va, reference=lgb_train)

# # 標準化(決定木モデルは無意味)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

start_time = time.perf_counter()

# ============================================================
# Initialize Model
# ============================================================

params = {
            "objective": "regression",
            'num_class': OUTPUT_DIM,
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'random_state': 42,
            'verbose': -1,
            
            # チューニング後のパラメータ
            'learning_rate': 0.1,
            'num_leaves': 19,
            'max_depth': 7,
            'min_child_samples': 15,
            'subsample': 0.85,
            'colsample_bytree': 0.71,
            'reg_alpha': 1,
            'reg_lambda': 0.01,
            }

# 履歴保存用の空辞書
evals_result = {}

callbacks = [
    # lgb.early_stopping(stopping_rounds=PATIENCE, verbose=False),
    lgb.log_evaluation(period=0), # 学習中の標準出力を抑制
    lgb.record_evaluation(evals_result) # 履歴を記録
]

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_train],
    valid_names=['training'],
    callbacks=callbacks,
)

# 違う書き方
# model = lgb.LGBMRegressor(
#     random_state=RANDOM_STATE,
#     n_estimators=ESTIMATORS, # 木の本数（Early Stoppingと併用するため多めに設定）
#     learning_rate=LEARNING_RATE,  # 学習率       
#     num_leaves=31,               # 葉の最大数
#     max_depth=-1,                # 木の深さ(-1制限なし)
#     min_child_samples=20,        # 葉の最小データ数
#     subsample=0.8,               # データ（行）のサンプリング率
#     colsample_bytree=0.8,        # 特徴量（列）のサンプリング率
#     reg_alpha=0.1,               # L1正則化（0~10 対数）
#     reg_lambda=0.1,              # L2正則化（0~10 対数）
# )
# # Early Stopping とログ出力を設定
# # eval_setの最後のデータセットに対して評価される。
# callbacks = [
#             lgb.early_stopping(stopping_rounds=PATIENCE, verbose=False),
#             # lgb.log_evaluation(period=PERIOD)
#             ]

# model.fit(
#           X_train, Y_train,
#           eval_set=[(X_train, Y_train), (X_test, Y_test)],  # result出力させたいデータセット。0番目に訓練、1番目に検証
#           eval_metric="rmse",
#           callbacks=callbacks
#           )


print("\nModel Structure:")
print(model)


# ==========================================
# 4. 学習ループ (Early Stopping含む)
# ==========================================


print("--- 学習完了 ---")

elapsed_time = time.perf_counter() - start_time
print(f"処理時間: {elapsed_time:.2f}秒")


# ==========================================
# 5. 学習済みモデルの保存
# ==========================================
model_save_path = "./results/GBDT/GBDT_model.pkl"

# 保存先ディレクトリを自動作成
Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "input_dim": X.shape[1],
        "output_dim": OUTPUT_DIM,
    },
    model_save_path
)

print(f"モデルを保存しました: {model_save_path}")


# ==========================================
# 6. 学習曲線の描画
# ==========================================
lgb.plot_metric(evals_result, metric='rmse')
plt.title('Full Data Learning Curve')
plt.show()


# evals_result = model.evals_result_
# print(evals_result.keys()) #キー項目名確認

# fig, axes = plt.subplots(1, 1, figsize=(6, 4))

# # Loss (Multi-Logloss)
# axes.plot(
#     evals_result['training']['rmse'],
#     label="Train rmse"
# )

# # axes.plot(
# #     evals_result['valid_1']['rmse'],
# #     label="Validation rmse"
# # )

# axes.set_title("LightGBM Training & Validation Loss")
# axes.set_xlabel("Boosting Iteration")
# axes.set_ylabel("Loss")
# axes.legend()
# axes.grid()

# plt.tight_layout()
# plt.show()

# ==========================================
# 正解-予測プロット
# ==========================================

# 予測
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

fig, ax = plt.subplots(figsize=(7, 7))

# 学習データ
ax.scatter(
    Y_train, y_train_pred,
    alpha=0.5,
    label="Train"
)

# テストデータ
ax.scatter(
    Y_test, y_test_pred,
    alpha=0.7,
    label="Test"
)

# y = x の基準線
min_val = min(
    np.min(Y_train), np.min(y_train_pred),
    np.min(Y_test), np.min(y_test_pred)
)
max_val = max(
    np.max(Y_train), np.max(y_train_pred),
    np.max(Y_test), np.max(y_test_pred)
)

ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    "k--",
    label="Ideal prediction"
)

ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted")

ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)

ax.set_aspect("equal", adjustable="box")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 7. データの精度評価
# ==========================================

train_rmse = np.sqrt(mean_squared_error(Y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
train_mae = mean_absolute_error(Y_train, y_train_pred)
test_mae = mean_absolute_error(Y_test, y_test_pred)
train_r2 = r2_score(Y_train, y_train_pred)
test_r2 = r2_score(Y_test, y_test_pred)

print(f"--- 精度評価 ---")
print(f"RMSE :  train {train_rmse:.4f}   test {test_rmse:.4f}")
print(f"MAE  :  train {train_mae:.4f}    test {test_mae:.4f}")
print(f"R2   :  train {train_r2:.4f}     teat {test_r2:.4f}")


# # ==========================================
# # 8. 予測結果の保存
# # ==========================================
# predicted_labels = encoder.inverse_transform(pred_np)

# result_df = pd.DataFrame({
#     "Actual": encoder.inverse_transform(Y_test),
#     "Predicted": predicted_labels
# })

# result_df.to_csv(
#     "./data/Dry_Bean_Dataset/predictions_lgbm.csv",
#     index=False,
#     encoding="utf-8-sig"
# )

# print(result_df.head())
# print("予測結果を保存しました。")