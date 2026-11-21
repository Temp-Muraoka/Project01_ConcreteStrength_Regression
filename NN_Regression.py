"""
基本的なニューラルネットワークモデル
pytorch
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import copy
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Setting
# ============================================================

HIDDEN_LAYERS = [224, 208, 192]
# HIDDEN_LAYERS = [32, 16]

LEARNING_RATE = 0.001

EPOCHS = 300

DROPOUT_RATE = 0.0

# Early Stopping
PATIENCE = 5       # 検証Lossが改善しないことを許容するエポック数
MIN_DELTA = 0.0     # 改善とみなす最小のLoss減少量



# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")


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


# 訓練データと検証データに分割（8:2）
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# 標準化
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

scaler_y = StandardScaler()
Y_train_scaled = scaler_y.fit_transform(Y_train.reshape(-1, 1))
Y_test_scaled = scaler_y.transform(Y_test.reshape(-1, 1))

# Tensorに変換（ここでGPU転送もできるが、データ量多い場合メモリ不足する。ここではbachごと伝送）
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)

# DataLoader
# 入力、出力をペアにする。画像など型が違ってもペアにできる
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

# ミニバッチに分ける（ジェネレータとして使う）。shuffle=True：1epochごとにデータシャッフル
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# ============================================================
# Initialize Model
# ============================================================

model = NeuralNetwork(
                      input_dim=X_train.shape[1],
                      output_dim=OUTPUT_DIM,
                      hidden_layers=HIDDEN_LAYERS,
                      dropout_rate=DROPOUT_RATE
                      ).to(device)

print("\nModel Structure:")
print(model)

# Loss Function and Optimizer

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ==========================================
# 4. 学習ループ
# ==========================================

start_time = time.perf_counter()

print("--- 学習開始 ---")

history = { 
            "train_rmse": [],
            "valid_rmse": [],
            }

# Early Stopping用の変数
best_valid_loss = float("inf")
best_model_state = copy.deepcopy(model.state_dict())
patience_counter = 0
best_epoch = 0


for epoch in range(EPOCHS):

    # ----- Training -----
    model.train()

    train_loss = 0.0

    all_predictions = []
    all_targets = []

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

    all_predictions = []
    all_targets = []

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


    # 10エポックごとに表示
    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_rmse:.4f} | "
            f"Valid Loss: {valid_rmse:.4f} | "
        )

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
        print(f"\nEarly Stopping: Epoch {epoch + 1}")
        break


# 最良だったエポックのモデルに戻す
model.load_state_dict(best_model_state)


print("--- 学習完了 ---")

elapsed_time = time.perf_counter() - start_time

print(f"処理時間: {elapsed_time:.2f}秒")


# ==========================================
# 5. 学習済みモデルの保存
# ==========================================
model_save_path = "./results/NN/NN_model.pth"

# 保存先ディレクトリを自動作成
Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "hidden_dims": HIDDEN_LAYERS,
        "output_dim": OUTPUT_DIM,
        "drop_rate": DROPOUT_RATE,

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
# 6. 学習曲線の描画
# ==========================================

fig, axes = plt.subplots(figsize=(7, 7))

# Loss
axes.plot(
    history["train_rmse"],
    label="Train Loss"
)

axes.plot(
    history["valid_rmse"],
    label="Validation Loss"
)

axes.set_title("Loss")
axes.set_xlabel("Epoch")
axes.set_ylabel("Loss")
axes.legend()
axes.grid()


plt.tight_layout()
plt.show()




# ==========================================
# 正解-予測プロット
# ==========================================

model.eval()

# 予測
with torch.no_grad():
    y_train_pred_scaled = model(X_train_tensor.to(device)).cpu().numpy()
    y_test_pred_scaled = model(X_test_tensor.to(device)).cpu().numpy()

# 元のスケールに戻す
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

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
#     "./data/Dry_Bean_Dataset/predictions.csv",
#     index=False,
#     encoding="utf-8-sig"
# )

# print(result_df.head())
# print("予測結果を保存しました。")
