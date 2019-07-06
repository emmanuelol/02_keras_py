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

def plot_results(out_dir, hist_file):
    """"CSVLogger で出した学習の損失関数と推測確率のファイルをplotする"""
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

    # acc plot
    plt.figure()
    if 'binary_accuracy' in df.columns.values:
        plt.plot(df['epoch'], df['val_binary_accuracy'], 'g-', marker='.', label='validation')
        plt.plot(df['epoch'], df['binary_accuracy'], 'r-', marker='.', label='train')
    else:
        plt.plot(df['epoch'], df['val_acc'], 'g-', marker='.', label='validation')
        plt.plot(df['epoch'], df['acc'], 'r-', marker='.', label='train')
    plt.grid()
    plt.legend(loc='lower right')# 凡例を右下に書く
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.ylim((0, 1))
    plt.savefig(os.path.join(out_dir, 'acc.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
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

if __name__ == '__main__':
    print('plot_log.py: loaded as script file')
else:
    print('plot_log.py: loaded as module file')
