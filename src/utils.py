import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==========================================
# 正解-予測プロット
# ==========================================
def Plot_actual_vs_predict(Y_train,y_train_pred,Y_test,y_test_pred):

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
# データの精度評価
# ==========================================
def evaluation_(Y_train, y_train_pred, Y_test, y_test_pred):

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
