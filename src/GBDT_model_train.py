"""
勾配ブースティング決定木を用いた分類モデル
LightGBM使用
"""

import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import lightgbm as lgb

from Data_load_And_Preprocess import data_load_and_preprocess


# Setting

RANDOM_STATE = 42

PATIENCE = 5


def GBDT_train(params, X_train, Y_train, X_valid=None, Y_valid=None):
    

    eval_sets = [(X_train, Y_train)]

    # 履歴保存用の空辞書
    evals_result = {}

    callbacks = [
        lgb.log_evaluation(period=0), # 学習中の標準出力を抑制
        lgb.record_evaluation(evals_result) # 履歴を記録
    ]


    # 注:パラメータ調整後の最終的なモデル学習時は検証データなし
    if X_valid is not None:
        eval_sets.append((X_valid, Y_valid))
        callbacks.append(lgb.early_stopping(stopping_rounds=PATIENCE, verbose=False))


    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train, Y_train,
        eval_set=eval_sets,
        callbacks=callbacks
    )

    return model


if __name__=="__main__":

    X_train, X_test, Y_train, Y_test = data_load_and_preprocess()

    # クラス数
    OUTPUT_DIM = Y_train.shape[1]

    start_time = time.perf_counter()


    params = {
                "objective": "regression",
                'num_class': OUTPUT_DIM,
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'random_state': RANDOM_STATE,
                'verbose': -1,

                "n_estimators": 1000,
                
                # チューニング後のパラメータ
                'learning_rate': 0.1,
                'num_leaves': 19,
                'max_depth': 7,
                'min_child_samples': 15,
                'subsample': 0.85,
                'colsample_bytree': 0.71,
                'reg_alpha': 1,
                'reg_lambda': 0.01,
                }


    model = GBDT_train(params, X_train, Y_train)

    
    print("--- 学習完了 ---")

    elapsed_time = time.perf_counter() - start_time
    print(f"処理時間: {elapsed_time:.2f}秒")


    # ==========================================
    # 学習済みモデルの保存
    # ==========================================
    model_save_path = "./results/GBDT/GBDT_model.pkl"

    # 保存先ディレクトリを自動作成
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "input_dim": X_train.shape[1],
            "output_dim": OUTPUT_DIM,
        },
        model_save_path
    )

    print(f"モデルを保存しました: {model_save_path}")


    # ==========================================
    # 学習曲線の描画
    # ==========================================
    lgb.plot_metric(model, metric='rmse')
    plt.title('Full Data Learning Curve')
    plt.show()

    # ==========================================
    # 正解-予測プロット
    # ==========================================

    # 予測
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

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