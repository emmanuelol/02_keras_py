"""
データフレームの各列の分布可視化関数
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys

def mean_hist_plot(df_describe_T, title='', xlabel='column', ylabel='mean', outputdir='describe'):
    """データフレームのdescribe().Tから平均値のヒストグラムplot"""
    y = list(df_describe_T['mean'])
    x = np.arange(len(y))

    plt.figure() # グラフ初期化
    #plt.rcParams['figure.figsize'] = (10, 4) # グラフが見きれないようにするためサイズを大きくしておく

    plt.bar(x, y, align="center") # 棒グラフ
    plt.xticks(x, list(df_describe_T['mean'].index), rotation=90) # 棒グラフの時のx軸の目盛り 90度回転

    plt.ylabel(ylabel) # y軸のラベル
    plt.xlabel(xlabel) # x軸のラベル
    plt.title(title, fontsize=20) # タイトルの文字大きくする

    plt.tight_layout() # グラフ同士が重ならないようにする関数
    plt.savefig(os.path.join(outputdir, title+'.png'), dpi=200) # グラフの保存+解像度指定(デフォルトdpi=80)
    plt.show() # グラフの表示
    plt.close('all') # 繰り返しplt.figure()を実行時に出る警告対策
    plt.clf() # グラフの設定クリア

# hist
def df_hist_plot(df, title='', outputdir='describe', figsize=(10.0, 10.0)):
    """データフレームの各列でヒストグラムplot"""
    plt.figure() # グラフ初期化
    plt.rcParams['figure.figsize'] = figsize # グラフが見きれないようにするためサイズを大きくしておく

    df.hist(range=(-200, 200), bins=200) # 一括でヒストグラムを並べて描画する

    plt.tight_layout() # グラフ同士が重ならないようにする関数
    plt.savefig(os.path.join(outputdir, title+'.png')) # グラフの保存
    plt.show() # グラフの表示
    plt.close('all') # 繰り返しplt.figure()を実行時に出る警告対策
    plt.clf() # グラフの設定クリア

# box plot
def df_box_plot(df, title='', xlabel='column', outputdir='describe', figsize=None, ylim=(-200,200)):
    """データフレームの各列でboxplot"""
    plt.figure() # グラフ初期化
    if figsize is not None:
        plt.rcParams['figure.figsize'] = figsize # グラフが見きれないようにするためサイズを大きくしておく

    df.plot.box() # 一括で箱ひげグラフを描画する
    plt.xticks(rotation=90) # 棒グラフの時のx軸の目盛り 90度回転

    plt.xlabel(xlabel) # x軸のラベル
    plt.title(title) # タイトル
    plt.ylim(ylim) # y軸に範囲

    plt.tight_layout() # グラフ同士が重ならないようにする関数
    plt.savefig(os.path.join(outputdir, title+'.png')) # グラフの保存
    plt.show() # グラフの表示
    plt.close('all') # グラフクリア。繰り返しplt.figure()を実行時に出る警告対策

def plot_barh_series(series, title=None, xlabel=None, xlim=None, out_png=None, color='blue'):
    """ seriesの横棒グラフplot """
    series_inv = series[series.index[::-1]] # seriesのindexの順番逆にする（plot.barhはindex逆順にしないとだめ）
    series_inv.plot.barh(alpha=0.6, figsize=(8,8*series.shape[0]//15), color=color)
    plt.grid(True)
    if title is not None:
        plt.title(title, size=12)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if xlim is not None:
        plt.xlim(xlim)
    if out_png is not None:
        plt.savefig(out_png, bbox_inches="tight")
    plt.show()
    plt.clf()
