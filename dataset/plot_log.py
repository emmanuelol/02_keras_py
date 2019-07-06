# -*- coding: utf-8 -*-
"""
CSVLogger で出した学習の損失関数と推測確率のファイルをplotする

Usage:
import plot_log
plot_log.plot_results(out_dir, hist_file)
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

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
        plt.plot(df['epoch'], df['loss'], marker='.', label=Path(hist_file).stem+'_train')
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
        plt.plot(df['epoch'], df['val_loss'], marker='.', label=Path(hist_file).stem+'_validation')
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
            plt.plot(df['epoch'], df['mean_absolute_error'], marker='.', label=Path(hist_file).stem+'_train')
            metric = 'mean_absolute_error'
        elif 'binary_accuracy' in df.columns.values:
            plt.plot(df['epoch'], df['binary_accuracy'], marker='.', label=Path(hist_file).stem+'_train')
        else:
            plt.plot(df['epoch'], df['acc'], marker='.', label=Path(hist_file).stem+'_train')
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
            plt.plot(df['epoch'], df['val_mean_absolute_error'], marker='.', label=Path(hist_file).stem+'_validation')
            metric = 'mean_absolute_error'
        elif 'binary_accuracy' in df.columns.values:
            plt.plot(df['epoch'], df['val_binary_accuracy'], marker='.', label=Path(hist_file).stem+'_validation')
        else:
            plt.plot(df['epoch'], df['val_acc'], marker='.', label=Path(hist_file).stem+'_validation')
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
            plt.plot(df['epoch'], df['lr'], marker='.', label=Path(hist_file).stem+'_lr')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)# 凡例をグラフ外に書く
        plt.xlabel('epoch')
        plt.ylabel('lr')
        plt.savefig(os.path.join(out_dir, 'lr.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
        plt.show()
        plt.clf()# plt.clf() [Clear the current figure]

if __name__ == '__main__':
    print('plot_log.py: loaded as script file')
else:
    print('plot_log.py: loaded as module file')
