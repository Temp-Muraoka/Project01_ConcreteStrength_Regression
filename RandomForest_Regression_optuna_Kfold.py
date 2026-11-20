"""
Optunaを用いたハイパーパラメータチューニング
RandomForest_Regression.pyのモデルを対象

交差検証法

"""

import pandas as pd
import numpy as np
import time

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

import optuna
import os


# ============================================================
# Setting
# ============================================================
TRIAL_N = 500 #探索回数
N_SPLITS = 5    # 交差検証の分割数


# Early Stopping
PATIENCE = 5       # 検証Lossが改善しないことを許容するエポック数



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

# ==========================================
# optunaによるハイパーパラメータ最適化（交差検証）
# ==========================================


start_time = time.perf_counter()

print("--- ハイパーパラメータ探索開始 (Cross-Validation) ---")

skf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def objective(trial):

    # --------------------------------------
    # 探索したいハイパーパラメータの設定 
    # --------------------------------------
    params = {
        'n_estimators': 300,       # 決定木の本数

        # 特徴量の選択割合（0.1〜1.0の間で探索）
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        
        # 木の深さ
        'max_depth': trial.suggest_int('max_depth', 3, 25),
        
        # 葉の最小サンプル数
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        
        # 分割の最小サンプル数
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),

        'max_features': 'sqrt',    # 分割時に考慮する特徴量の数
        'random_state': 42,        # 乱数シードの固定
        'n_jobs': -1               # 全CPUコアを使って高速化
    }


    # 各フォールドのスコアを格納するリスト
    fold_valid_rmses = []
    fold_train_rmses = []
    fold_best_rounds = []

    # 交差検証ループ
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, Y_train)):

        X_tr, X_va = X[train_idx], X[valid_idx]
        Y_tr, Y_va = Y[train_idx], Y[valid_idx]


        # ==========================================
        # 学習
        # ==========================================
        model = RandomForestRegressor(**params)

        # 学習の実行（全データで学習する場合）
        model.fit(X_tr, Y_tr)


        # 各データセットに対する予測とメトリクス計算
        train_pred = model.predict(X_tr)
        valid_pred = model.predict(X_va)

        train_rmse = np.sqrt(mean_squared_error(Y_tr, train_pred))
        valid_rmse = np.sqrt(mean_squared_error(Y_va, valid_pred))

        # 進捗の動的上書き表示
        print(f" Trial {trial.number+1} | {fold + 1}/{N_SPLITS} fold finished", end="\r", flush=True)

        # 各フォールドの結果を保存
        fold_valid_rmses.append(valid_rmse)
        fold_train_rmses.append(train_rmse)


    # 全フォールドの平均を計算
    mean_valid_rmse = np.mean(fold_valid_rmses)
    mean_train_rmse = np.mean(fold_train_rmses)


    # 各トライアルの最後に、全フォールド平均でのLossとAccを表示
    print(
        f"Trial {trial.number+1} "
        f"Train RMSE: {mean_train_rmse:.4f} | Valid RMSE: {mean_valid_rmse:.4f} | "
    )

    # UserAttrs へ保存
    trial.set_user_attr("best_train_rmse", mean_train_rmse)

    return mean_valid_rmse

study = optuna.create_study(
    study_name="subrutin_lgbm_tuning",
    # storage=f"sqlite:///{db_path}",  # Google Drive 上の SQLite DB を使う場合はコメント解除
    # load_if_exists=True,             # ★重要: DBが存在すれば自動で続きから再開
    direction="minimize",            # 目的関数（Loss）を最小化
    pruner=optuna.pruners.MedianPruner()  # 枝刈り用
)

print(f"現在完了している Trial 数: {len(study.trials)}")

# 探索の実行
study.optimize(objective, n_trials=TRIAL_N)

csv_path = "./results/RandomForest/trials_history.csv"

# 保存先ディレクトリを自動作成
Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

# 完了後に結果を CSV としても出力
df_trials = study.trials_dataframe()
df_trials.to_csv(csv_path, index=False, encoding="utf-8-sig")


# --- 最も良かったTrialから情報を取得 ---
best_trial = study.best_trial

best_epoch = best_trial.user_attrs.get("best_epoch")
best_train_loss = best_trial.user_attrs.get("best_train_rmse")

print("\nハイパーパラメータチューニング完了！")
print(f"ベストTrial番号: {best_trial.number}")
print(f"ベスト検証 RMSE: {best_trial.value:.4f}")
print(f"ベスト訓練 RMSE: {best_train_loss:.4f}")
print("ベストパラメータ:")
for key, val in best_trial.params.items():
    print(f"  - {key}: {val}")

elapsed_time = time.perf_counter() - start_time
print(f"処理時間: {elapsed_time:.2f}秒")


# # ==========================================
# # 最も良いパラメータでのモデル構築・再学習と保存
# # ==========================================
# print("\n--- ベストパラメータによるモデルの再学習と保存 ---")

# best_params = study.best_params
# # 再学習用の設定パラメータを構築
# final_params = {
#     'objective': 'multiclass',
#     'num_class': OUTPUT_DIM,
#     'metric': 'multi_logloss',
#     'boosting_type': 'gbdt',
#     'random_state': 42,
#     'verbose': -1,
#     **best_params
# }

# # 全データを用いて、クロスバリデーションの平均最適ラウンド数（または少し余裕を持たせた回数）で再学習
# # あるいは全データに対するtrain/val分割で厳密にEarly Stoppingを行うことも可能です
# best_best_iter = int(best_epoch) if not np.isnan(best_epoch) else 100

# train_data_full = lgb.Dataset(X, label=Y)
# best_model = lgb.train(
#     final_params,
#     train_data_full,
#     num_boost_round=best_best_iter
# )

# # モデルとクラス情報・エンコーダをまとめて joblib で保存
# model_save_path = "./data/Dry_Bean_Dataset/dry_bean_lgbm_model.pkl"
# joblib.dump(
#     {
#         "model": best_model,
#         "classes": encoder.classes_,
#         "params": final_params
#     },
#     model_save_path
# )

# print(f"ベストモデルを保存しました: {model_save_path}")

# elapsed_time = time.perf_counter() - start_time
# print(f"総処理時間: {elapsed_time:.2f}秒")