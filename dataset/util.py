# -*- coding: utf-8 -*-
"""
util関数群
"""
import os, glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

def show_np_img(x, is_grayscale=False):
    """
    numpy配列の画像データを表示させる
    matplotlibはgrayscaleにした画像は白色が黄色になるのでgrayscale化した場合はis_grayscale=Trueにすること
    """
    plt.imshow(x)
    if is_grayscale == True:
        plt.gray()
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

def find_img_files(path):
    """
    ファイルパス再帰的に探索し、pngかjpgのパスだけリストで返す
    https://qiita.com/hasepy/items/8e6a0757da1ce074ce87
    """
    import os
    imagePaths = []
    for pathname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            # フィルタ処理
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.PNG') or filename.endswith('.JPG'):
                imagePaths.append(os.path.join(pathname,filename))
    return imagePaths

def get_jpg_png_path_in_dir(dir):
    """
    pathlibを使って指定ディレクトリを再帰的になめ、jpg,pngのファイルパスをリストで返す
    """
    jpg_files = []
    for p in list(Path(dir).glob("**/*jpg")):
        # Pathオブジェクトを通常の文字列に変換
        jpg_files.append(p.as_posix())
    png_files = []
    for p in list(Path(dir).glob("**/*png")):
        # Pathオブジェクトを通常の文字列に変換
        png_files.append(p.as_posix())
    files = []
    files.extend(jpg_files)
    files.extend(png_files)
    print('jpg_png_count:', len(files))
    return sorted(files)

def resize_np_Nearest_Neighbor(img_np, n_resize):
    """
    numpy型の画像データ1つの縦横サイズをn_resize倍にする
    （CIFAR10そのままのサイズ(32,32,3)ではkerasの学習済みモデルつかえないので）
    https://blog.shikoan.com/numpy-upsampling-image/
    """
    return img_np.repeat(n_resize, axis=0).repeat(n_resize, axis=1)


def ipywidgets_show_img(img_path_list):
    """
    ipywidgetsでインタラクティブに画像表示
    https://github.com/pfnet-research/chainer-chemistry/blob/master/examples/tox21/tox21_dataset_exploration.ipynb
    """
    from ipywidgets import interact
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    def view_image(index):
        img_path = img_path_list[index]
        print('index={}, img_path={}'.format(index, img_path))
        img = Image.open(img_path)
        img_array = np.asarray(img)
        #plt.figure(figsize=(9, 9))
        plt.imshow(img_array)
        #if is_grayscale == True:
        #    plt.gray()
        plt.show()
    interact(view_image, index=(0, len(img_path_list) - 1))

if __name__ == '__main__':
    print('util.py: loaded as script file')
else:
    print('util.py: loaded as module file')
