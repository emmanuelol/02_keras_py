# -*- coding: utf-8 -*-
"""
CSVLogger で出した学習の損失関数と推測確率のファイルをplotする
ニューラルネットワーク分岐したマルチタスク用

Usage:
import plot_12task_log
plot_12task_log.plot_results(out_dir, hist_file)
"""
import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_results(out_dir, hist_file
                , tsv_col = ['task0_pred', 'task1_pred', 'task2_pred', 'task3_pred', 'task4_pred', 'task5_pred', 'task6_pred', 'task7_pred', 'task8_pred', 'task9_pred', 'task10_pred', 'task11_pred']
                , gpu_count=1):
    """"
    CSVLogger で出した学習の損失関数と推測確率のファイル（history.tsv）をplotする
    Args：
        out_dir:plot画像出力先ディレクトリ
        hist_file:history.tsv
        tsv_col: history.tsvの列名
        gpu_count: multigpuかどうかをgpuの数で判定
    Returns:
        なし（lossやaccのplotファイル出力）
    """
    df = pd.read_table(hist_file)# カンマ区切りならpd.read_csv(hist_file)

    # loss plot
    plt.figure()
    plt.plot(df['epoch'], df['val_loss'], 'g-', marker='.', label='validation')
    plt.plot(df['epoch'], df['loss'], 'r-', marker='.', label='train')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(out_dir, 'loss.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()
    plt.clf()# plt.clf() [Clear the current figure]

    maker_list = ['.', 'v', '1', 'd', 'x', '8', 's', 'p', '*', 'h', '3', '4']
    # task loss plot
    plt.figure()
    for i, col in enumerate(tsv_col):
        if gpu_count==1:
            plt.plot(df['epoch'], df[col+'_loss'], 'r-', marker=maker_list[i], label=col)
        else:
            # multigpuだと列名が変わる
            plt.plot(df['epoch'], df['concatenate_'+str(i)+'_loss'], 'r-', marker=maker_list[i], label=col)
    for i, col in enumerate(tsv_col):
        if gpu_count==1:
            plt.plot(df['epoch'], df['val_'+col+'_loss'], 'g-', marker=maker_list[i], label='val_'+col)
        else:
            # multigpuだと列名が変わる
            plt.plot(df['epoch'], df['val_concatenate_'+str(i)+'_loss'], 'g-', marker=maker_list[i], label='val_'+col)
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)# 凡例を枠外に書く
    plt.xlabel('epoch')
    plt.ylabel('task loss')
    plt.savefig(os.path.join(out_dir, 'task_loss.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()
    plt.clf()# plt.clf() [Clear the current figure]

    # task acc plot
    plt.figure()
    for i, col in enumerate(tsv_col):
        if gpu_count==1:
            plt.plot(df['epoch'], df[col+'_acc'], 'r-', marker=maker_list[i], label=col)
        else:
            # multigpuだと列名が変わる
            plt.plot(df['epoch'], df['concatenate_'+str(i)+'_acc'], 'r-', marker=maker_list[i], label=col)
    for i, col in enumerate(tsv_col):
        if gpu_count==1:
            plt.plot(df['epoch'], df['val_'+col+'_acc'], 'g-', marker=maker_list[i], label='val_'+col)
        else:
            # multigpuだと列名が変わる
            plt.plot(df['epoch'], df['val_concatenate_'+str(i)+'_acc'], 'g-', marker=maker_list[i], label='val_'+col)
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)# 凡例を枠外に書く
    plt.xlabel('epoch')
    plt.ylabel('task acc')
    plt.ylim((0, 1))
    plt.savefig(os.path.join(out_dir, 'task_acc.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()
    plt.clf()# plt.clf() [Clear the current figure]

    if 'lr' in df.columns: # lrカラムあるか確認
        # lr plot
        plt.figure()
        plt.plot(df['epoch'], df['lr'], 'g-', marker='.', label='lr')
        plt.grid()
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('lr')
        plt.savefig(os.path.join(out_dir, 'lr.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
        plt.show()
        plt.clf()# plt.clf() [Clear the current figure]

if __name__ == '__main__':
    print('plot_12task_log.py: loaded as script file')
else:
    print('plot_12task_log.py: loaded as module file')
