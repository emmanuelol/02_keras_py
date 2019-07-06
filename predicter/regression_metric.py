# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statistics import mean

def calc_regression_metrics(y_true_list, y_pred_list, mask_value=-1):
    """
    回帰の評価指標を計算する
    Args:
        y_true_list:正解ラベルのnumpy.ndarrayのリスト [array([0,1,0,…]), array([1,0,0,…]),…]
        y_pred_list:予測スコアのnumpy.ndarrayのリスト [array([0.65,0.99,0.01,…]), array([0.1,0.33,0.95,…]),…]
        mask_value:ラベルの欠損値
    """
    mae_list = []
    mse_list = []
    rmse_list = []
    r2_list = []
    count = 0
    for (y_pred, y_true) in zip(y_pred_list, y_true_list):
        y_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        y_df = y_df[y_df['y_true'] != mask_value]# 欠損ラベル=-1.0 以外の行だけにする
        y_true = np.array(y_df['y_true'])
        y_pred = np.array(y_df['y_pred'])
        y_true_list[count] = y_true
        y_pred_list[count] = y_pred
        if y_true_list[count].shape[0] > 0:
            mae_list.append( mean_absolute_error(y_true, y_pred) )# 平均絶対誤差 (MAE, Mean Absolute Error) ：実際の値と予測値の絶対値を平均したもの
            mse_list.append( mean_squared_error(y_true, y_pred) ) # 平均二乗誤差 (MSE, Mean Squared Error) ：実際の値と予測値の絶対値の 2 乗を平均したもの
            rmse_list.append( np.sqrt(mean_squared_error(y_true, y_pred)) ) # 二乗平均平方根誤差 (RMSE: Root Mean Squared Error) ：MSE の平方根
            # 決定係数 (R2, R-squared, coefficient of determination) ：モデルの当てはまりの良さを示す指標
            # 最も当てはまりの良い場合、1.0 となります (当てはまりの悪い場合、マイナスとなることもあり)。寄与率 (きよりつ) とも呼ばれる
            r2_list.append( r2_score(y_true, y_pred) )
        #else:
        #    mae_list.append(-1)
        #    mse_list.append(-1)
        #    rmse_list.append(-1)
        #    r2_list.append(-1)
        count += 1
    print('MAE, Mean Absolute Error:', mae_list, mean(mae_list))
    print('MSE, Mean Squared Error:', mse_list, mean(mse_list))
    print('RMSE: Root Mean Squared Error:', rmse_list, mean(rmse_list))
    print('R2, R-squared, coefficient of determination:', r2_list, mean(r2_list))
    return y_true_list, y_pred_list, mae_list, mse_list, rmse_list, r2_list

if __name__ == '__main__':
    print('regression_metric.py: loaded as script file')
else:
    print('regression_metric.py: loaded as module file')
