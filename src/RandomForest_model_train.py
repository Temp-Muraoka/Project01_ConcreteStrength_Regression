"""
ランダムフォレストを用いた回帰モデル

"""

import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor

# 自作関数
from src.Data_load_And_Preprocess import data_load_and_preprocess
from src.utils import Plot_actual_vs_predict


# Setting
RANDOM_STATE = 42
PATIENCE = 5 #Early Stoppingの停止基準（改善しない回数）


def RandomForest_train(params, X_train, Y_train):
    
    if Y_train.shape[1]==1:
        Y_train = Y_train.ravel()

    model = RandomForestRegressor(**params)

    model.fit(X_train, Y_train)

    return model


if __name__=="__main__":

    X_train, X_test, Y_train, Y_test = data_load_and_preprocess()


    start_time = time.perf_counter()


    params = {
        'n_estimators': 300,       # 決定木の本数
        'max_depth': 10,           # 木の深さの上限（Noneで制限なし）
        'min_samples_leaf': 5,     # 葉に必要な最小サンプル数
        'max_features': 'sqrt',    # 分割時に考慮する特徴量の数
        'random_state': 42,        # 乱数シードの固定
        'n_jobs': -1               # 全CPUコアを使って高速化
    }

    RandomForest_model = RandomForest_train(params, X_train, Y_train)


    # ==========================================
    # 学習済みモデルの保存
    # ==========================================
    model_save_path = "./results/RandomForest/RandomForest_model.pkl"

    # 保存先ディレクトリを自動作成
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": RandomForest_model,
            "input_dim": X_train.shape[1],
            "output_dim": Y_train.shape[1],
        },
        model_save_path
    )

    print(f"モデルを保存しました: {model_save_path}")

    # ==========================================
    # 正解-予測プロット
    # ==========================================

    # 予測
    y_train_pred = RandomForest_model.predict(X_train)
    y_test_pred = RandomForest_model.predict(X_test)

    Plot_actual_vs_predict(Y_train,y_train_pred,Y_test,y_test_pred)


