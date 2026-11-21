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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Setting
# ============================================================

HIDDEN_LAYERS = [128]
# HIDDEN_LAYERS = [32, 16]

LEARNING_RATE = 0.001

EPOCHS = 100

DROPOUT_RATE = 0.2

# Early Stopping
PATIENCE = 15       # 検証Lossが改善しないことを許容するエポック数
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

# 標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tensorに変換（ここでGPU転送もできるが、データ量多い場合メモリ不足する。ここではbachごと伝送）
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

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
            "train_loss": [],
            "valid_loss": [],
            "train_acc": [],
            "valid_acc": []
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
    train_correct = 0  # GPU上で正解率計算のために使用

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

        train_loss += loss.item() * batch_X.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (predictions == batch_y).sum().item()

        # all_predictions.extend(predictions.cpu().numpy())
        # all_targets.extend(batch_y.cpu().numpy())

    train_loss /= len(train_loader.dataset)

    train_acc = train_correct / len(train_loader.dataset)
    # train_acc = accuracy_score(all_targets,all_predictions)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)


    # ----- Validation -----
    model.eval()

    valid_loss = 0.0
    valid_correct = 0
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

            predictions = outputs.argmax(dim=1)

            valid_correct += (predictions == batch_y).sum().item()
            # all_predictions.extend(predictions.cpu().numpy())    
            # all_targets.extend(batch_y.cpu().numpy())
    
        valid_loss /= len(test_loader.dataset)

        valid_acc = valid_correct / len(test_loader.dataset)
        # valid_acc = accuracy_score(all_targets,all_predictions)
    
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)


    # 10エポックごとに表示
    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Valid Acc: {valid_acc:.4f}"
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

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "hidden_dims": HIDDEN_LAYERS,
        "output_dim": OUTPUT_DIM,
        "drop_rate": DROPOUT_RATE,
        "classes": encoder.classes_,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    },
    "./data/Dry_Bean_Dataset/dry_bean_model.pth"
)

print("モデルを保存しました。")


# ==========================================
# 6. 学習曲線の描画
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(
    history["train_loss"],
    label="Train Loss"
)

axes[0].plot(
    history["valid_loss"],
    label="Validation Loss"
)

axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid()

# Accuracy
axes[1].plot(
    history["train_acc"],
    label="Train Accuracy"
)

axes[1].plot(
    history["valid_acc"],
    label="Validation Accuracy"
)

axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()


# ==========================================
# 7. 検証データの予測・精度評価
# ==========================================

model.eval()

with torch.no_grad():

    # .to(device)
    X_test_tensor = X_test_tensor.to(device)

    outputs = model(X_test_tensor)

    # 最も確率の高いクラスを取得
    predicted_classes = outputs.argmax(dim=1)

    # ここで1回だけCPUのNumPy配列に変換して使い回す
    pred_np = predicted_classes.cpu().numpy()  

    accuracy = accuracy_score(Y_test, pred_np)


print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 8. 予測結果の保存
# ==========================================

predicted_labels = encoder.inverse_transform(pred_np)

result_df = pd.DataFrame({

    "Actual": encoder.inverse_transform(Y_test),

    "Predicted": predicted_labels

})

result_df.to_csv(
    "./data/Dry_Bean_Dataset/predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print(result_df.head())
print("予測結果を保存しました。")
