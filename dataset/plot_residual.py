# -*- coding: utf-8 -*-
"""
残差をplotする
"""
import os, pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_residual(out_dir, y_true_list=[], y_pred_list=[], label_list=[], title='Residual Plot'):
    """
    回帰モデルの正解と予測の残差plotする
    参考:http://tekenuko.hatenablog.com/entry/2016/09/19/151547
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

            #print(y_true.shape[0])
            if y_true.shape[0] != 0:
                plt.scatter(y_pred, y_pred - y_true, label = label) # 残差をプロット

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


def scatter_plot_from_regression_pred_tsv(df, out_dir, title):
    """ 回帰モデルの予測スコアのtsvファイルロードして散布図描画 """
    import warnings
    warnings.filterwarnings('ignore') # 実行に影響のない警告（ワーニング）を非表示にする
    plt.style.use('ggplot') # チャート綺麗に書くおまじない
    import seaborn as sns; sns.set()
    from scipy import stats

    df = df[df['y_true'] != -1] # 欠損ラベル=-1.0 以外の行だけにする

    g = sns.jointplot(x=df.columns[1], y=df.columns[0], data=df).annotate(stats.pearsonr,frameon=False) # .annotate(stats.pearsonr,frameon=False) が無いと相関係数書いてくれない
    g.fig.suptitle(title)

    plt.tight_layout() #グラフを整えて表示
    plt.savefig(os.path.join(out_dir, title+'.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()

def scatter_plot_from_regression_score_dir(pred_dir, out_path, df_task_id_name, title='NPI_test_set', figsize=(7, 7)):
    """
    指定ディレクトリ内の回帰の予測scoreファイルを全てロードして1枚の散布図にplot
    Args:
        pred_dir:回帰の予測scoreファイル格納ディレクトリ
        out_path:出力する散布図画像のパス
        df_task_id_name:task_idとtask名の対応表
        figsize:出力する散布図のサイズ
    """
    plt.style.use('ggplot') # チャート綺麗に書くおまじない

    os.makedirs(pathlib.Path(out_path).parent, exist_ok=True)

    plt.figure(figsize=figsize)

    for index, series in df_task_id_name.iterrows():

        df = pd.read_csv(pred_dir+'/'+series['task_num']+'.tsv', sep='\t')

        # 欠損ラベル=-1.0 以外の行だけにする
        df = df[df['y_true'] != -1]

        # ピアソンの相関係数r
        res = df['y_true'].corr(df['y_pred'], method='pearson')

        # 散布図plot
        plt.scatter( df['y_pred'], df['y_true'], label=series['task_name']+' r='+str(round(res, 2)) )

    plt.title(title)
    plt.xlabel('y_pred [pIC50]')
    plt.ylabel('y_true [pIC50]')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12) # 凡例を枠外に書く
    plt.savefig(out_path, bbox_inches="tight") # plt.savefig はplt.show() の前に書かないと白紙で保存される # label見切れ対策 bbox_inches="tight"
    plt.show()
