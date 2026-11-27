import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import lightgbm as lgb

import optuna
import optuna.visualization.matplotlib as vis_plt
from sklearn.model_selection import KFold


# 自作関数
from src.Data_load_And_Preprocess import data_load_and_preprocess
from src.GBDT_model_train import GBDT_train

# Setting
RANDOM_STATE = 42
PATIENCE = 5

def Optimize_lightGBM(trial_num, X_train, Y_train, CV_num=5):

    start_time = time.perf_counter()

    def objective(trial):

        skf = KFold(n_splits=CV_num, shuffle=True, random_state=RANDOM_STATE)

        # --------------------------------------
        # 探索したいハイパーパラメータの設定 (LightGBM用)
        # --------------------------------------
        params = {
                    "objective": "regression",
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'random_state': RANDOM_STATE,
                    'verbose': -1,

                    "n_estimators": 1000,

                    'subsample_freq': 1,
                    
                    # チューニング対象パラメータ（対数変数はキリの良い数値のみ）
                    'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1]),
                    'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.01),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.01),
                    'reg_alpha': trial.suggest_categorical('reg_alpha', [0.0, 0.01, 0.1, 1.0, 10.0]),
                    'reg_lambda': trial.suggest_categorical('reg_lambda',[0.0, 0.01, 0.1, 1.0, 10.0]),
                }

        # 各フォールドのスコアを格納するリスト
        fold_valid_rmses = []
        fold_train_rmses = []
        fold_best_rounds = []

        # 交差検証ループ
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, Y_train)):

            X_tr, X_va = X_train[train_idx], X_train[valid_idx]
            Y_tr, Y_va = Y_train[train_idx], Y_train[valid_idx]

            # 決定木系モデルのため、StandardScaler等のスケーリングは不要（そのまま渡す）

            model = GBDT_train(params, X_tr, Y_tr, X_va, Y_va)

            # ベストイテレーション（何本目で止まったか）
            best_iter = model._best_iteration
            if best_iter == 0:
                best_iter = model.current_iteration()

            # 各データセットに対する予測とメトリクス計算
            train_pred = model.predict(X_tr, num_iteration=best_iter)
            valid_pred = model.predict(X_va, num_iteration=best_iter)

            train_rmse = np.sqrt(mean_squared_error(Y_tr, train_pred))
            valid_rmse = np.sqrt(mean_squared_error(Y_va, valid_pred))

            # 進捗の動的上書き表示
            print(f" Trial {trial.number+1} | {fold + 1}/{CV_num} fold finished (Best Iter: {best_iter})", end="\r", flush=True)

            # 各フォールドの結果を保存
            fold_valid_rmses.append(valid_rmse)
            fold_train_rmses.append(train_rmse)

            fold_best_rounds.append(best_iter)

        # 全フォールドの平均を計算
        mean_valid_rmse = np.mean(fold_valid_rmses)
        mean_train_rmse = np.mean(fold_train_rmses)

        mean_best_round = np.mean(fold_best_rounds)

        # 各トライアルの最後に、全フォールド平均でのLossとAccを表示
        print(
            f"Trial {trial.number+1} CV Mean [Avg Best Iter: {mean_best_round:.1f}] "
            f"Train RMSE: {mean_train_rmse:.4f} | Valid RMSE: {mean_valid_rmse:.4f} | "
        )

        # UserAttrs へ保存
        trial.set_user_attr("best_epoch", mean_best_round) # 互換性のためキー名を合わせる
        trial.set_user_attr("best_train_rmse", mean_train_rmse)

        return mean_valid_rmse


    study = optuna.create_study(
    study_name="subrutin_lgbm_tuning",
    direction="minimize",            # 目的関数（Loss）を最小化
    )

    study.optimize(objective, n_trials=trial_num)

    csv_path = "./results/GBDT/trials_history.csv"

    # 保存先が存在しない場合、自動作成（上書き注意）
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
    print(f"ベストイテレーション（平均）: {best_epoch:.1f} ラウンド")
    print(f"ベスト検証 RMSE: {best_trial.value:.4f}")
    print(f"ベスト訓練 RMSE: {best_train_loss:.4f}")
    print("ベストパラメータ:")
    for key, val in best_trial.params.items():
        print(f"  - {key}: {val}")

    # 1. 最適化の履歴
    vis_plt.plot_optimization_history(study)
    plt.show()

    # 2. パラメータの重要度
    vis_plt.plot_param_importances(study)
    plt.show()
    

    elapsed_time = time.perf_counter() - start_time
    print(f"処理時間: {elapsed_time:.2f}秒")

    fixed_params = {
        "objective": "regression",
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'random_state': RANDOM_STATE,
        'verbose': -1,        
        "n_estimators": int(best_epoch),        
        'subsample_freq': 1,
    }

    # 探索パラメータと固定パラメータ合体
    final_params = {**fixed_params, **best_trial.params}

    import json

    with open('results/GBDT/GBDT_best_params.json', 'w') as f:
        json.dump(final_params, f, indent=4)

  

if __name__=="__main__":

    X_train, X_test, Y_train, Y_test = data_load_and_preprocess()

    
    Optimize_lightGBM(5, X_train, Y_train)

    