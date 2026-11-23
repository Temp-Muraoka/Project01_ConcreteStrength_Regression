import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from sklearn.ensemble import RandomForestRegressor

import optuna
import optuna.visualization.matplotlib as vis_plt
from sklearn.model_selection import KFold


# 自作関数
from src.Data_load_And_Preprocess import data_load_and_preprocess
from src.RandomForest_model_train import RadomForest_train

# Setting
RANDOM_STATE = 42
PATIENCE = 5

def Optimize_RndomForest(trial_num, X_train, Y_train, CV_num=5):

    start_time = time.perf_counter()

    def objective(trial):

        skf = KFold(n_splits=CV_num, shuffle=True, random_state=RANDOM_STATE)

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

            X_tr, X_va = X_train[train_idx], X_train[valid_idx]
            Y_tr, Y_va = Y_train[train_idx], Y_train[valid_idx]

            # 決定木系モデルのため、StandardScaler等のスケーリングは不要（そのまま渡す）

            model = RadomForest_train(params, X_tr, Y_tr)


            # 各データセットに対する予測とメトリクス計算
            train_pred = model.predict(X_tr)
            valid_pred = model.predict(X_va)

            train_rmse = np.sqrt(mean_squared_error(Y_tr, train_pred))
            valid_rmse = np.sqrt(mean_squared_error(Y_va, valid_pred))

            # 進捗の動的上書き表示
            print(f" Trial {trial.number+1} | {fold + 1}/{CV_num} fold finished", end="\r", flush=True)

            # 各フォールドの結果を保存
            fold_valid_rmses.append(valid_rmse)
            fold_train_rmses.append(train_rmse)



        # 全フォールドの平均を計算
        mean_valid_rmse = np.mean(fold_valid_rmses)
        mean_train_rmse = np.mean(fold_train_rmses)



        # 各トライアルの最後に、全フォールド平均でのLossとAccを表示
        print(
            f"Trial {trial.number+1}  "
            f"Train RMSE: {mean_train_rmse:.4f} | Valid RMSE: {mean_valid_rmse:.4f} | "
        )

        # UserAttrs へ保存
        trial.set_user_attr("best_train_rmse", mean_train_rmse)

        return mean_valid_rmse


    study = optuna.create_study(
    study_name="subrutin_lgbm_tuning",
    direction="minimize",            # 目的関数（Loss）を最小化
    pruner=optuna.pruners.MedianPruner()  # 枝刈り用
    )

    study.optimize(objective, n_trials=trial_num)

    csv_path = "./results/RndomForest/trials_history.csv"

    # 保存先が存在しない場合、自動作成（上書き注意）
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    # 完了後に結果を CSV としても出力
    df_trials = study.trials_dataframe()
    df_trials.to_csv(csv_path, index=False, encoding="utf-8-sig")


    # --- 最も良かったTrialから情報を取得 ---
    best_trial = study.best_trial

    # best_epoch = best_trial.user_attrs.get("best_epoch")
    best_train_loss = best_trial.user_attrs.get("best_train_rmse")

    print("\nハイパーパラメータチューニング完了！")
    print(f"ベストTrial番号: {best_trial.number}")
    # print(f"ベストイテレーション（平均）: {best_epoch:.1f} ラウンド")
    print(f"ベスト検証 RMSE: {best_trial.value:.4f}")
    print(f"ベスト訓練 RMSE: {best_train_loss:.4f}")
    print("ベストパラメータ:")
    for key, val in best_trial.params.items():
        print(f"  - {key}: {val}")

    vis_plt.plot_optimization_history(study)
    plt.show()
    

    elapsed_time = time.perf_counter() - start_time
    print(f"処理時間: {elapsed_time:.2f}秒")

  

if __name__=="__main__":

    X_train, X_test, Y_train, Y_test = data_load_and_preprocess()

    
    Optimize_RndomForest(5, X_train, Y_train)

    