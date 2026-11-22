import numpy as np
import matplotlib.pyplot as plt


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