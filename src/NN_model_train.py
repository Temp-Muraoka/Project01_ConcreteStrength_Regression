"""
ニューラルネットワーク回帰モデル
Pytorch使用
"""

import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

from sklearn.preprocessing import StandardScaler

# 自作関数
from src.Data_load_And_Preprocess import data_load_and_preprocess
from src.utils import Plot_actual_vs_predict


# Setting
RANDOM_STATE = 42
PATIENCE = 5 #Early Stoppingの停止基準（改善しない回数）
MIN_DELTA = 0.0 #Early Stoppingの改善判定許容差

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
        
        # .append() で順々に中間層追加していく
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            self.network.append(nn.Linear(prev_dim, hidden_dim))
            self.network.append(nn.ReLU())

            if dropout_rate > 0:
                self.network.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim
            
        # 出力層
        self.network.append(nn.Linear(prev_dim, output_dim))


    def forward(self, x):
        return self.network(x)



def NN_train(params, X_train_tensor, Y_train_tensor, X_test_tensor=None, Y_test_tensor=None, log_disp=False):

    device = params["device"]

    # DataLoader
    # 入力、出力をペアにする。画像など型が違ってもペアにできる
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    
    # ミニバッチに分ける（ジェネレータとして使う）。shuffle=True：1epochごとにデータシャッフル
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)   
    
    
    # 注:パラメータ調整後の最終的なモデル学習時は検証データなし
    if X_test_tensor is not None:
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    
    # ============================================================
    # Initialize Model
    # ============================================================

    model = NeuralNetwork(
                        input_dim = params["input_dim"],
                        output_dim = params["output_dim"],
                        hidden_layers = params["hidden_layers"],
                        dropout_rate=params["dropout_rate"]
                        ).to(device)

    if log_disp:
        print("\nModel Structure:")
        print(model)


    # Loss Function and Optimizer

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])


    # ==========================================
    # 4. 学習ループ
    # ==========================================

    history = { 
                "train_rmse": [],
                "valid_rmse": [],
                }

    # Early Stopping用の変数
    best_valid_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    best_epoch = 0


    for epoch in range(params["epochs"]):

        # ----- Training -----
        model.train()

        train_loss = 0.0

        for batch_X, batch_y in train_loader:

            # .to(device) GPU転送　
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            loss.backward()

            optimizer.step()

            # バッチ内の平均MSEに「そのバッチのデータ数」を掛けて、二乗誤差の合計に戻す
            # .item() をつけることで計算グラフを切り離しメモリリークを防ぐ
            train_loss += loss.item() * batch_X.size(0)

        # 全データ数で割って全体のMSEを出し、最後に1回だけルートをとってRMSEにする
        train_mse = train_loss / len(train_loader.dataset)
        train_rmse = train_mse ** 0.5

        history["train_rmse"].append(train_rmse)


        # ----- Validation -----
        model.eval()

        valid_loss = 0.0

        # 注:パラメータ調整後の最終的なモデル学習時は検証データなし
        if X_test_tensor is not None:

            with torch.no_grad():

                for batch_X, batch_y in test_loader:

                    # テストデータも .to(device)
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)

                    outputs = model(batch_X)

                    loss = criterion(outputs, batch_y)

                    valid_loss += loss.item() * batch_X.size(0)
            
                valid_mse = valid_loss / len(test_loader.dataset)
                valid_rmse = valid_mse ** 0.5
                history["valid_rmse"].append(valid_rmse)


            # ----- Early Stopping -----
            if valid_loss < best_valid_loss - MIN_DELTA:

                best_valid_loss = valid_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0

            else:
                patience_counter += 1

            # patience回連続で改善しなければ終了
            if patience_counter >= PATIENCE:
                print(f"Early Stopping: Epoch {epoch + 1}", end="\r", flush=True)

                # 最良だったエポックのモデルに戻す
                model.load_state_dict(best_model_state)

                break



        # # 10エポックごとに表示
        # if (epoch + 1) % 10 == 0:

        #     print(
        #         f"Epoch [{epoch+1}/{params['epochs']}] "
        #         f"Train Loss: {train_rmse:.4f} | "
        #         f"Valid Loss: {valid_rmse:.4f} | ", end="\r", flush=True
        #     )


    return model, best_epoch


if __name__=="__main__":

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train, X_test, Y_train, Y_test = data_load_and_preprocess()


    start_time = time.perf_counter()


    params = {
                "input_dim": X_train.shape[1],
                "output_dim": Y_train.shape[1],
                "hidden_layers": [224, 208, 192],
                "dropout_rate": 0.0,

                "epochs": 300,
                "learning_rate": 0.001,

                "device":device
                }

    # 標準化
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    Y_train_scaled = scaler_y.fit_transform(Y_train)
    Y_test_scaled = scaler_y.transform(Y_test)

    # Tensorに変換（ここでGPU転送もできるが、データ量多い場合メモリ不足する。ここではbachごと伝送）
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)


    model, best_epoch = NN_train(params, X_train_tensor, Y_train_tensor, log_disp=True)


    # ==========================================
    # 学習済みモデルの保存
    # ==========================================
    model_save_path = "./results/NN/NN_model.pth"

    # 保存先ディレクトリを自動作成
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)


    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": params["input_dim"],
            "output_dim": params["output_dim"],
            "hidden_layers": params["hidden_layers"],
            "dropout_rate": params["dropout_rate"],

            # X（入力）のスケーラー情報
            "scaler_x_mean": scaler_x.mean_,
            "scaler_x_scale": scaler_x.scale_,
            
            # Y（出力）のスケーラー情報
            "scaler_y_mean": scaler_y.mean_,
            "scaler_y_scale": scaler_y.scale_,
        },
        model_save_path
    )

    print("モデルを保存しました。")

    # ==========================================
    # 正解-予測プロット
    # ==========================================

    # 予測
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    Plot_actual_vs_predict(Y_train,y_train_pred,Y_test,y_test_pred)

    

    # ==========================================
    # データの精度評価
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
    # # 予測結果の保存
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