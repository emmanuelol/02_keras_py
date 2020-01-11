# -*- coding: utf-8 -*-
"""
CSVLogger で出した学習の損失関数と推測確率のファイルをplotする

Usage:
    import plot_log
    hist_file = model.log
    plot_log.plot_results(out_dir, hist_file)
"""
import os, pathlib
import matplotlib.pyplot as plt
import pandas as pd

def plot_results(out_dir, hist_file, acc_metric=None):
    """"CSVLogger で出した学習の損失関数と推測確率のファイルをplotする"""
    df = pd.read_csv(hist_file, sep='\t')# カンマ区切りならpd.read_csv(hist_file)

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

    # acc plot
    plt.figure()
    if acc_metric is None:
        if 'binary_accuracy' in df.columns.values:
            plt.plot(df['epoch'], df['val_binary_accuracy'], 'g-', marker='.', label='validation')
            plt.plot(df['epoch'], df['binary_accuracy'], 'r-', marker='.', label='train')
        else:
            plt.plot(df['epoch'], df['val_acc'], 'g-', marker='.', label='validation')
            plt.plot(df['epoch'], df['acc'], 'r-', marker='.', label='train')
        plt.ylabel('acc')
        plt.ylim((0, 1))
        img_name = 'acc.png'
    else:
        plt.plot(df['epoch'], df['val_'+acc_metric], 'g-', marker='.', label='validation')
        plt.plot(df['epoch'], df[acc_metric], 'r-', marker='.', label='train')
        plt.ylabel(acc_metric)
        img_name = acc_metric+'.png'
    plt.grid()
    plt.legend(loc='lower right')# 凡例を右下に書く
    plt.xlabel('epoch')
    plt.savefig(os.path.join(out_dir, img_name), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()

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

def overlaid_plot_results(out_dir, hist_file_list):
    """"CSVLogger で出した学習の損失関数と推測確率のファイルを重ねてplotする"""
    # train loss plot
    plt.figure()
    for hist_file in hist_file_list:
        df = pd.read_csv(hist_file, sep='\t')# カンマ区切りならpd.read_csv(hist_file)
        plt.plot(df['epoch'], df['loss'], marker='.', label=pathlib.Path(hist_file).stem+'_train')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)# 凡例をグラフ外に書く
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(out_dir, 'train_loss.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()
    plt.clf()# plt.clf() [Clear the current figure]

    # validation loss plot
    plt.figure()
    for hist_file in hist_file_list:
        df = pd.read_csv(hist_file, sep='\t')# カンマ区切りならpd.read_csv(hist_file)
        plt.plot(df['epoch'], df['val_loss'], marker='.', label=pathlib.Path(hist_file).stem+'_validation')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)# 凡例をグラフ外に書く
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(out_dir, 'validation_loss.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()
    plt.clf()# plt.clf() [Clear the current figure]

    # train acc plot
    metric = 'acc'
    plt.figure()
    for hist_file in hist_file_list:
        df = pd.read_csv(hist_file, sep='\t')# カンマ区切りならpd.read_csv(hist_file)
        if 'mean_absolute_error' in df.columns.values:
            plt.plot(df['epoch'], df['mean_absolute_error'], marker='.', label=pathlib.Path(hist_file).stem+'_train')
            metric = 'mean_absolute_error'
        elif 'binary_accuracy' in df.columns.values:
            plt.plot(df['epoch'], df['binary_accuracy'], marker='.', label=pathlib.Path(hist_file).stem+'_train')
        else:
            plt.plot(df['epoch'], df['acc'], marker='.', label=pathlib.Path(hist_file).stem+'_train')
    plt.grid()
    #plt.legend(loc='lower right')# 凡例を右下に書く
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)# 凡例をグラフ外に書く
    plt.xlabel('epoch')
    if metric == 'acc':
        plt.ylim((0, 1))
    plt.ylabel(metric)
    plt.savefig(os.path.join(out_dir, 'train_'+metric+'.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()

    # validation acc plot
    metric = 'acc'
    plt.figure()
    for hist_file in hist_file_list:
        df = pd.read_csv(hist_file, sep='\t')# カンマ区切りならpd.read_csv(hist_file)
        if 'mean_absolute_error' in df.columns.values:
            plt.plot(df['epoch'], df['val_mean_absolute_error'], marker='.', label=pathlib.Path(hist_file).stem+'_validation')
            metric = 'mean_absolute_error'
        elif 'binary_accuracy' in df.columns.values:
            plt.plot(df['epoch'], df['val_binary_accuracy'], marker='.', label=pathlib.Path(hist_file).stem+'_validation')
        else:
            plt.plot(df['epoch'], df['val_acc'], marker='.', label=pathlib.Path(hist_file).stem+'_validation')
    plt.grid()
    #plt.legend(loc='lower right')# 凡例を右下に書く
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)# 凡例をグラフ外に書く
    plt.xlabel('epoch')
    if metric == 'acc':
        plt.ylim((0, 1))
    plt.ylabel(metric)
    plt.savefig(os.path.join(out_dir, 'validation_'+metric+'.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()

    if 'lr' in df.columns: # lrカラムあるか確認
        # lr plot
        plt.figure()
        for hist_file in hist_file_list:
            df = pd.read_csv(hist_file, sep='\t')# カンマ区切りならpd.read_csv(hist_file)
            plt.plot(df['epoch'], df['lr'], marker='.', label=pathlib.Path(hist_file).stem+'_lr')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)# 凡例をグラフ外に書く
        plt.xlabel('epoch')
        plt.ylabel('lr')
        plt.savefig(os.path.join(out_dir, 'lr.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
        plt.show()
        plt.clf()# plt.clf() [Clear the current figure]

def plot_multitask_results(out_dir, hist_file
                , tsv_col = ['task0_pred', 'task1_pred', 'task2_pred', 'task3_pred', 'task4_pred', 'task5_pred', 'task6_pred', 'task7_pred', 'task8_pred', 'task9_pred', 'task10_pred', 'task11_pred']
                , tsv_metric_col = None
                , gpu_count=1):
    """"
    CSVLogger で出した学習の損失関数と推測確率のファイル（history.tsv）をplotする
    ニューラルネットワーク分岐したマルチタスク用
    Args：
        out_dir:plot画像出力先ディレクトリ
        hist_file:history.tsv
        tsv_col: history.tsvの列名
        tsv_metric_col: history.tsvのmetricの列名
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
        #print(i, col)
        if gpu_count==1:
            plt.plot(df['epoch'], df[col+'_loss'], 'r-', marker=maker_list[i%len(maker_list)], label=col)
        else:
            # multigpuだと列名が変わる
            plt.plot(df['epoch'], df['concatenate_'+str(i)+'_loss'], 'r-', marker=maker_list[i%len(maker_list)], label=col)
    for i, col in enumerate(tsv_col):
        if gpu_count==1:
            plt.plot(df['epoch'], df['val_'+col+'_loss'], 'g-', marker=maker_list[i%len(maker_list)], label='val_'+col)
        else:
            # multigpuだと列名が変わる
            plt.plot(df['epoch'], df['val_concatenate_'+str(i)+'_loss'], 'g-', marker=maker_list[i%len(maker_list)], label='val_'+col)
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)# 凡例を枠外に書く
    plt.xlabel('epoch')
    plt.ylabel('task loss')
    plt.savefig(os.path.join(out_dir, 'task_loss.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()
    plt.clf()# plt.clf() [Clear the current figure]

    # task acc plot
    plt.figure()
    if tsv_metric_col is None:
        for i, col in enumerate(tsv_col):
            #print(i, col)
            if gpu_count==1:
                plt.plot(df['epoch'], df[col+'_acc'], 'r-', marker=maker_list[i%len(maker_list)], label=col)
            else:
                # multigpuだと列名が変わる
                plt.plot(df['epoch'], df['concatenate_'+str(i)+'_acc'], 'r-', marker=maker_list[i%len(maker_list)], label=col)
        for i, col in enumerate(tsv_col):
            if gpu_count==1:
                plt.plot(df['epoch'], df['val_'+col+'_acc'], 'g-', marker=maker_list[i%len(maker_list)], label='val_'+col)
            else:
                # multigpuだと列名が変わる
                plt.plot(df['epoch'], df['val_concatenate_'+str(i)+'_acc'], 'g-', marker=maker_list[i%len(maker_list)], label='val_'+col)
    else:
        for i, col in enumerate(tsv_metric_col):
                plt.plot(df['epoch'], df[col], 'r-', marker=maker_list[i%len(maker_list)], label=col)
                plt.plot(df['epoch'], df['val_'+col], 'g-', marker=maker_list[i%len(maker_list)], label='val_'+col)
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
