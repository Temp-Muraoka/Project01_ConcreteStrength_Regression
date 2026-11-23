"""
勾配ブースティング決定木を用いた回帰モデル
LightGBM使用
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


def RadomForest_train(params, X_train, Y_train):
    
    if Y_train.shape[1]==1:
        Y_train = Y_train.ravel()

    model = RandomForestRegressor(**params)

    model.fit(X_train, Y_train)

    return model


if __name__=="__main__":

    X_train, X_test, Y_train, Y_test = data_load_and_preprocess()


    start_time = time.perf_counter()


    params = {
                "objective": "regression",
                'num_class': Y_train.shape[1],
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


    model =RadomForest_train(params, X_train, Y_train)

    
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
            "output_dim": Y_train.shape[1],
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