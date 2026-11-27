import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch

from sklearn.preprocessing import StandardScaler

import optuna
import optuna.visualization.matplotlib as vis_plt
from sklearn.model_selection import KFold


# 自作関数
from src.Data_load_And_Preprocess import data_load_and_preprocess
from src.NN_model_train import NN_train

# Setting
RANDOM_STATE = 42
PATIENCE = 5

def Optimize_NN(trial_num, X_train, Y_train, device_0, CV_num=5):

    start_time = time.perf_counter()

    device = device_0

    def objective(trial):

        skf = KFold(n_splits=CV_num, shuffle=True, random_state=RANDOM_STATE)

        # --------------------------------------
        # 探索したいハイパーパラメータの設定 
        # --------------------------------------
        
        # 中間層の「層数」を決める（例: 1〜4層）
        n_layers = trial.suggest_int('n_layers', 1, 4)
        
        # 各層の「ノード数」をループで決定してリストを作成する
        hidden_dims_triral = []
        prev_dim = 256 # 最初の層数最大値

        for i in range(n_layers):
            # 層ごとに個別のアディショナルパラメータ（hidden_dim_0, hidden_dim_1...）として定義
            # 次の層は前の層のサイズ（prev_dim）を超えないように上限を設定
            dim = trial.suggest_int(f'hidden_dim_{i}', 16, prev_dim, step=16)
            hidden_dims_triral.append(dim)
            prev_dim = dim  # 上限を更新
            
        # ドロップアウト率を決める（例: 0.0〜0.5）
        drop_rate_trial = trial.suggest_float('drop_rate', 0.0, 0.5, step=0.1)
        
        # # 学習率を決める（対数スケール log=True で 1e-4 〜 1e-2 を探す）
        # lr_trial = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        lr_trial = trial.suggest_categorical('learning_rate', [0.0001, 0.0005, 0.001, 0.005, 0.01])

        params = {
                    "input_dim": X_train.shape[1],
                    "output_dim": Y_train.shape[1],
                    "hidden_layers": hidden_dims_triral,
                    "dropout_rate": drop_rate_trial,

                    "epochs": 300,
                    "learning_rate": lr_trial,

                    "device":device
                    }

        # 各フォールドのスコアを格納するリスト
        fold_valid_rmses = []
        fold_train_rmses = []
        fold_best_rounds = []

        # 交差検証ループ
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, Y_train)):

            X_tr, X_va = X_train[train_idx], X_train[valid_idx]
            Y_tr, Y_va = Y_train[train_idx], Y_train[valid_idx]

            # 各フォールド内で標準化を実施（データリーク防止）
            scaler_x = StandardScaler()
            X_tr_scaled = scaler_x.fit_transform(X_tr)
            X_va_scaled = scaler_x.transform(X_va)

            scaler_y = StandardScaler()
            Y_tr_scaled = scaler_y.fit_transform(Y_tr.reshape(-1, 1))
            Y_va_scaled = scaler_y.transform(Y_va.reshape(-1, 1))

            # Tensorに変換
            X_tr_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32)
            Y_tr_tensor = torch.tensor(Y_tr_scaled, dtype=torch.float32)
            X_va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32)
            Y_va_tensor = torch.tensor(Y_va_scaled, dtype=torch.float32)

            # train_dataset = TensorDataset(X_tr_tensor, Y_tr_tensor)
            # valid_dataset = TensorDataset(X_va_tensor, Y_va_tensor)

            # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            # valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

            model, best_epoch = NN_train(params, X_tr_tensor, Y_tr_tensor, X_va_tensor, Y_va_tensor)

            # 予測
            with torch.no_grad():
                y_train_pred_scaled = model(X_tr_tensor.to(device)).cpu().numpy()
                y_valid_pred_scaled = model(X_va_tensor.to(device)).cpu().numpy()

            # 元のスケールに戻す
            y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
            y_valid_pred = scaler_y.inverse_transform(y_valid_pred_scaled)

            train_rmse = np.sqrt(mean_squared_error(Y_tr, y_train_pred))
            valid_rmse = np.sqrt(mean_squared_error(Y_va, y_valid_pred))

            # 進捗の動的上書き表示
            print(f" Trial {trial.number+1} | {fold + 1}/{CV_num} fold finished", end="\r", flush=True)

            # 各フォールドの結果を保存
            fold_valid_rmses.append(valid_rmse)
            fold_train_rmses.append(train_rmse)

            fold_best_rounds.append(best_epoch)

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

    csv_path = "./results/NN/trials_history.csv"

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

  

if __name__=="__main__":

    X_train, X_test, Y_train, Y_test = data_load_and_preprocess()

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    
    Optimize_NN(5, X_train, Y_train, device)

    