import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# データ読み込み・前処理
# ==========================================

def data_load_and_preprocess():

    ROOT_DIR = Path(__file__).resolve().parent.parent

    data_path = ROOT_DIR / "data" / "Concrete_Data.xlsx"

    df = pd.read_excel(data_path)

    # 項目名簡略化のため、()内の記述削除
    df.columns = df.columns.str.replace(r"\s*\([^)]*\)", "", regex=True)


    X = df.drop(columns=["Concrete compressive strength "])
    Y = df["Concrete compressive strength "]


    # NumPy配列に変換
    X = X.to_numpy().astype(np.float32)
    Y = Y.to_numpy().astype(np.float32).reshape(-1, 1)


    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")


    # 訓練データと検証データに分割（8:2）
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, Y_train, Y_test


if __name__=="__main__":
    X_train, X_test, Y_train, Y_test = data_load_and_preprocess()

    print(X_train)
    print(Y_train)
