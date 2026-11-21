"""
Optunaを用いたハイパーパラメータチューニング
NN_Classification_pytorch.pyのモデルを対象

交差検証法

Colab上での実行を想定
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset

import optuna
import os


# ============================================================
# Setting
# ============================================================
TRIAL_N = 200 #探索回数
N_SPLITS = 5    # 交差検証の分割数

# HIDDEN_LAYERS = [32, 16]

LEARNING_RATE = 0.001

EPOCHS = 300 #Earlystoppingするので、十分大きい数

# DROPOUT_RATE = 0.0

# Early Stopping
PATIENCE = 5       # 検証Lossが改善しないことを許容するエポック数
MIN_DELTA = 0.0     # 改善とみなす最小のLoss減少量



# =====================================================================
# デバイス設定と Google Drive 保存先の設定
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用中の実行デバイス: {device}")

# # Google Drive 上の保存用ディレクトリを作成
# save_dir = "/content/drive//MyDrive/Colab Notebooks/Project01_ConcreteStrength_Regression/optuna_results"
# os.makedirs(save_dir, exist_ok=True)

# # データベース（.db）と CSV の保存パスを設定
# db_path = os.path.join(save_dir, "optuna_study.db")
# csv_path = os.path.join(save_dir, "trials_history.csv")


# ============================================================
# Model Definition
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_layers,
                 dropout_rate=0.0
                 ):

       
        """
        Args:
            input_dim (int): 入力特徴量の次元数
            hidden_layers (list of int): 各中間層のノード数のリスト (例: [64, 32, 16])
            output_dim (int): 出力次元数
        """

        super().__init__() #親クラス(NN.module))の初期化(必須)


        # Sequential の箱を作る
        self.network = nn.Sequential()
        
        # .append() で順々に追加していく
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            self.network.append(nn.Linear(prev_dim, hidden_dim))
            self.network.append(nn.ReLU())

            if dropout_rate > 0:
                self.network.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim
            
        # 3. 最後に出力層を追加
        self.network.append(nn.Linear(prev_dim, output_dim))
        # self.network.append(nn.Sigmoid()) # 2値分類の場合


    def forward(self, x):
        return self.network(x)


# ==========================================
# データ読み込み・前処理
# ==========================================
df = pd.read_excel("data/Concrete_Data.xlsx")

# ()内の記述削除
df.columns = df.columns.str.replace(r"\s*\([^)]*\)", "", regex=True)


X = df.drop(columns=["Concrete compressive strength "])
Y = df["Concrete compressive strength "]


# クラス数
OUTPUT_DIM = 1



# NumPy配列に変換
X = X.to_numpy().astype(np.float32)
Y = Y.to_numpy().astype(np.float32)


print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")



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
    
    # 中間層の「層数」を決める（例: 1〜4層）
    n_layers = trial.suggest_int('n_layers', 1, 4)
    
    # 各層の「ノード数」をループで決定してリストを作成する
    hidden_dims_triral = []
    prev_dim = 256

    for i in range(n_layers):
        # 層ごとに個別のアディショナルパラメータ（hidden_dim_0, hidden_dim_1...）として定義
        # 例: 16〜256の間で、16の倍数（step=16）や2の累乗で探す
        # 次の層は前の層のサイズ（prev_dim）を超えないように上限を設定
        dim = trial.suggest_int(f'hidden_dim_{i}', 16, prev_dim, step=16)
        hidden_dims_triral.append(dim)
        prev_dim = dim  # 上限を更新
        
    # ドロップアウト率を決める（例: 0.0〜0.5）
    drop_rate_trial = trial.suggest_float('drop_rate', 0.0, 0.5, step=0.1)
    
    # # 学習率を決める（対数スケール log=True で 1e-4 〜 1e-2 を探す）
    # lr_trial = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    # 各フォールドのスコアを格納するリスト
    fold_valid_rmse = []
    fold_train_rmse = []
    fold_best_epochs = []

    # 交差検証ループ
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, Y)):

        X_tr, X_va = X[train_idx], X[valid_idx]
        Y_tr, Y_va = Y[train_idx], Y[valid_idx]

        # 各フォールド内で標準化を実施（データリーク防止）
        scaler_x = StandardScaler()
        X_tr = scaler_x.fit_transform(X_tr)
        X_va = scaler_x.transform(X_va)

        scaler_y = StandardScaler()
        Y_tr = scaler_y.fit_transform(Y_tr.reshape(-1, 1))
        Y_va = scaler_y.transform(Y_va.reshape(-1, 1))

        # Tensorに変換
        X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
        Y_tr_tensor = torch.tensor(Y_tr, dtype=torch.float32)
        X_va_tensor = torch.tensor(X_va, dtype=torch.float32)
        Y_va_tensor = torch.tensor(Y_va, dtype=torch.float32)

        train_dataset = TensorDataset(X_tr_tensor, Y_tr_tensor)
        valid_dataset = TensorDataset(X_va_tensor, Y_va_tensor)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

        # ============================================================
        # Initialize Model
        # ============================================================

        model = NeuralNetwork(
                            input_dim=X.shape[1],
                            output_dim=OUTPUT_DIM,
                            hidden_layers=hidden_dims_triral,
                            dropout_rate=drop_rate_trial
                            ).to(device)


        # Loss Function and Optimizer

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


        # ==========================================
        # 学習ループ
        # ==========================================

        # Early Stopping用の変数 (フォールドごと)
        best_valid_rmse_fold = float("inf")
        best_epoch_fold = 0
        best_train_rmse_fold = 0.0

        patience_counter = 0

        for epoch in range(EPOCHS):

            # ----------------------------------------------------
            # 進捗の動的上書き表示
            # ----------------------------------------------------
            print(f" {fold + 1}/{N_SPLITS} fold : {epoch + 1}epoch", end="\r", flush=True)

            # ----- Training -----
            model.train()

            train_loss = 0.0

            for batch_X, batch_y in train_loader:

                # .to(device)
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()

                outputs = model(batch_X)

                loss = criterion(outputs, batch_y)

                loss.backward()

                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)

            train_mse = train_loss / len(train_loader.dataset)
            train_rmse = train_mse ** 0.5
            

            # ----- Validation -----
            model.eval()

            valid_loss = 0.0

            with torch.no_grad():

                for batch_X, batch_y in valid_loader:

                    # テストデータも .to(device)
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)

                    outputs = model(batch_X)

                    loss = criterion(outputs, batch_y)

                    valid_loss += loss.item() * batch_X.size(0)
            
                valid_mse = valid_loss / len(valid_loader.dataset)
                valid_rmse = valid_mse ** 0.5


            # ----- Early Stopping & Best Metrics Tracking -----
            if valid_rmse < best_valid_rmse_fold - MIN_DELTA:
                best_valid_rmse_fold = valid_rmse
                best_epoch_fold = epoch
                best_train_rmse_fold = train_rmse

                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early Stopping: Epoch {epoch + 1}", end="\r", flush=True)
                break

        
        # 各フォールドの結果を保存
        fold_valid_rmse.append(best_valid_rmse_fold)
        fold_train_rmse.append(best_train_rmse_fold)
        fold_best_epochs.append(best_epoch_fold + 1)

    # 全フォールドの平均を計算
    mean_valid_rmse = np.mean(fold_valid_rmse)
    mean_train_rmse = np.mean(fold_train_rmse)
    mean_best_epoch = np.mean(fold_best_epochs)

    # 各トライアルの最後に、全フォールド平均でのLossとAccを表示
    print(
        f"Trial CV Mean [Avg Best Epoch: {mean_best_epoch:.1f}] "
        f"Train Loss: {mean_train_rmse:.4f} | Valid Loss: {mean_valid_rmse:.4f} | "
    )

    # UserAttrs へ保存
    trial.set_user_attr("best_epoch", mean_best_epoch)
    trial.set_user_attr("best_train_rmse", mean_train_rmse)

    return mean_valid_rmse

# Optuna Study の作成または既存 DB からの読み込み
study = optuna.create_study(
    study_name="subrutin_nn_tuning",
    # storage=f"sqlite:///{db_path}",  # Google Drive 上の SQLite DB
    # load_if_exists=True,             # ★重要: DBが存在すれば自動で続きから再開
    direction="minimize",            # 目的関数（Lossなど）を最小化する場合
    pruner=optuna.pruners.MedianPruner()  # 枝刈り用（任意）
)

print(f"現在完了している Trial 数: {len(study.trials)}")

# 探索の実行
study.optimize(objective, n_trials=TRIAL_N)

csv_path = "./results/NN/trials_history.csv"

# 保存先ディレクトリを自動作成
Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

# 完了後に結果を CSV としても出力
df_trials = study.trials_dataframe()
df_trials.to_csv(csv_path, index=False, encoding="utf-8-sig")



# --- 最も良かったTrialから情報を取得 ---
best_trial = study.best_trial

best_epoch = best_trial.user_attrs.get("best_epoch")
best_train_loss = best_trial.user_attrs.get("best_train_rmse")


print("ハイパーパラメータチューニング完了！")
print(f"ベストTrial番号: {best_trial.number}")
print(f"ベストエポック: {best_epoch} エポック目")
print(f"ベスト検証 Loss (Valid Loss): {best_trial.value:.4f}")
print(f"ベスト訓練 Loss: {best_train_loss:.4f}")

print("ベストパラメータ:")
for key, val in best_trial.params.items():
    print(f"  - {key}: {val}")

print(f"履歴保存先: {csv_path}")

elapsed_time = time.perf_counter() - start_time

print(f"処理時間: {elapsed_time:.2f}秒")



# # ==========================================
# # 最も良いパラメータでのモデル構築・再学習と保存
# # ==========================================
# print("\n--- ベストパラメータによるモデルの再学習と保存 ---")

# best_params = study.best_params
# best_n_layers = best_params['n_layers']
# best_hidden_layers = [best_params[f'hidden_dim_{i}'] for i in range(best_n_layers)]
# best_drop_rate = best_params['drop_rate']
# best_lr = best_params['lr']

# # ベストモデルのインスタンス化
# best_model = NeuralNetwork(
#                             input_dim=X_train.shape[1],
#                             output_dim=OUTPUT_DIM,
#                             hidden_layers=best_hidden_layers,
#                             dropout_rate=best_drop_rate
#                             ).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(best_model.parameters(), lr=best_lr)

# # 全訓練データを用いてベストパラメータで再学習（必要に応じてエポック数を調整してください）
# best_model.train()
# for epoch in range(EPOCHS):
#     for batch_X, batch_y in train_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#         optimizer.zero_grad()
#         outputs = best_model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
    

# # 状態や前処理スケーラーを含めて保存
# torch.save(
#     {
#         "model_state_dict": best_model.state_dict(),
#         "input_dim": X.shape[1],
#         "hidden_dims": best_hidden_layers,
#         "output_dim": OUTPUT_DIM,
#         "drop_rate": best_drop_rate,
#         "classes": encoder.classes_,
#         "scaler_mean": scaler.mean_,
#         "scaler_scale": scaler.scale_,
#     },
#     "./data/Dry_Bean_Dataset/dry_bean_model.pth"
# )

# print(f"ベストモデルを保存しました")

# elapsed_time = time.perf_counter() - start_time
# print(f"総処理時間: {elapsed_time:.2f}秒")

