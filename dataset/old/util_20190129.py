# -*- coding: utf-8 -*-
"""
util関数群
"""
import os, glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob

def find_all_files(directory):
    """再帰的にファイル・ディレクトリを探して出力するgenerator"""
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

def file_count(path , search):
    """ディレクトリ再帰的になめて<search>の文字持つファイルの数を返す"""
    files = glob.glob(os.path.join(path, '**'), recursive=True)
    newlist = []
    for l in files:
        if search in l:
            newlist.append(l)
    return(len(newlist))

def glob_files(dir):
    """
    サブディレクトリも含めてファイル一覧を取得
    https://weblabo.oscasierra.net/python/python3-beginning-file-list.html
    """
    return glob.glob(dir+"/**/*")

def show_np_img(x):
    """numpy配列の画像データを表示させる"""
    plt.imshow(x)
    plt.show()

def show_file_img(img_path):
    """ファイルパスから画像データを表示させる"""
    #画像の読み込み
    im = Image.open(img_path)
    #画像をarrayに変換
    im_list = np.asarray(im)
    #貼り付け
    plt.imshow(im_list)
    #表示
    plt.show()

if __name__ == '__main__':
    print('util.py: loaded as script file')
else:
    print('util.py: loaded as module file')
