# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_residual(out_dir, y_true_list=[], y_pred_list=[], label_list=[], title='Residual Plot'):
    """
    回帰モデルの正解と予測の残差plotする
    参考:http://tekenuko.hatenablog.com/entry/2016/09/19/151547
    Args:
        y_true_list:正解ラベルのnumpy.ndarrayのリスト [array([0,1,0,…]), array([1,0,0,…]),…]
        y_pred_list:予測スコアのnumpy.ndarrayのリスト [array([0.65,0.99,0.01,…]), array([0.1,0.33,0.95,…]),…]
        label_list:y_true_list,y_pred_listのリストの要素に対応するクラス名のリスト。凡例名になる。
        title:残差グラフのタイトル名
    """
    pred_max = 0.0
    pred_min = 0.0
    # 正解と予測の残差をプロット
    if len(y_true_list) != 0 and len(y_pred_list) != 0:
        for i, y_true_pred in enumerate(zip(y_true_list, y_pred_list)):
            y_true = y_true_pred[0]
            y_pred = y_true_pred[1]
            if len(label_list) != 0:
                label = label_list[i]
            else:
                label = 'Data'+str(i)

            if y_true.shape[0] != 0:
                plt.scatter(y_pred, y_pred - y_true, label = label) # 残差をプロット

                # 残差グラフの横軸の最大・最小値にする
                if np.max(y_pred) > pred_max:
                    pred_max = np.max(y_pred)
                if np.min(y_pred) < pred_min:
                    pred_min = np.min(y_pred)

    plt.title(title)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    #plt.legend(loc = 'upper left') # 凡例を左上に表示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12) # 凡例を枠外に書く
    plt.xlim(pred_min-1, pred_max+1) # X軸の表示範囲
    plt.hlines(y = 0, xmin = pred_min-1, xmax = pred_max+1, lw = 2, color = 'red') # y = 0に直線を引く
    plt.grid() # グリッド線を表示
    plt.savefig(os.path.join(out_dir, 'plot_residual.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()
    plt.clf()# plt.clf() [Clear the current figure]

if __name__ == '__main__':
    print('plot_residual.py: loaded as script file')
else:
    print('plot_residual.py: loaded as module file')
