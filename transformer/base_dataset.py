"""
ディープラーニングを学習させるためのデータセットの準備
https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_dataset
"""
import os, sys, argparse, glob, pathlib, copy
import cv2
import numpy as np
from sklearn import preprocessing as pp

def load_my_data(dir_path:str, classes=[], img_height=0, img_width=0, channel=3, is_pytorch=True):
    """
    自分で用意した学習データセット読み込み
    https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_dataset

    フレームワークに関係なく、学習データセットの入力画像は４次元配列で準備する
    ４次元は[データ数、画像の縦サイズ、画像の横サイズ、チャネル(RGBの3かグレースケールの1)]
    → Tensorflow, Keras(Tensorflow)	[データ数、画像の縦サイズ、画像の横サイズ、チャネル]
    → PyTorch, Chainer, Caffe	[データ数、チャネル、画像の縦サイズ、画像の横サイズ]

    教師ラベルのデータは、１次元もしくは二次元配列で用意します。
    1次元の場合はクラスのインデックス(例えば３クラス分類にて犬なら0、イモリなら1、ヤモリなら2みたいな)を指定するが、二次元の場合はone-hot表現を用いる(犬なら[1,0,0]、イモリなら[0,1,0]、ヤモリなら[0,0,1]みたいな)。
    これもフレームワークによって変わります。
    → PyTorch, Chainer	index [データ数]
    → Tensorflow, Keras(Tensorflow), Caffe	one-hot [データ数、クラス数]

    画像の学習データセット、教師データセットを用意して、それぞれxs, tsという変数に格納してxs, tsを返す仕様
    ディレクトリ構成は
    dir_path
     |-class1
      |-*jpg
     |-class2
      |-*jpg
    """
    # クラス名の指定ない場合ディレクトリ名をクラス名にする
    if len(classes) == 0:
        cla_dir_paths = sorted(glob.glob(str(pathlib.Path(dir_path) / '*')))
        classes = [pathlib.Path(cla_dir_path).name for cla_dir_path in cla_dir_paths]
        classes = [f for f in classes if os.path.isdir(os.path.join(dir_path, f))] # ディレクトリだけ取得
    print(f"classes: {classes}")

    # 画像の学習データセット、教師データセット、ファイルパスを用意
    xs = np.ndarray((0, img_height, img_width, channel))
    ts = np.ndarray((0))
    paths = []
    for cla_dir_path in glob.glob(str(pathlib.Path(dir_path) / '*')):
        #print(cla_dir_path)
        for path in glob.glob(str(pathlib.Path(cla_dir_path) / '*')):
            # 画像サイズの指定があれば画像ロード
            if img_width != 0 and img_height != 0:
                x = cv2.imread(path)
                x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
                x /= 255.
                xs = np.r_[xs, x[None, ...]]

            t = np.zeros((1))
            for i,cla in enumerate(classes):
                #print(i, cla)
                if cla in path:
                    t = np.array((i))
                    break
            ts = np.r_[ts, t]

            paths += [path]

    if is_pytorch:
        if img_width != 0 and img_height != 0:
            # PyTorch, Chainer, Caffe [データ数、チャネル、画像の縦サイズ、画像の横サイズ]
            xs = xs.transpose(0,3,1,2)
    else:
        # 教師データone-hot化
        lb = pp.LabelBinarizer()
        ts = lb.fit_transform(ts)

    return xs, ts, paths

def make_batch(paths, ts=None, mb=12):
    """
    ミニバッチ単位でファイルパスとラベルを返す
    https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_dataset
    Usage:
        xs, ts, paths = load_my_data('./data/', img_height=0, img_width=0)
        mb_paths, mb_ts = ds.make_batch(paths, ts=ts)
    """
    mbi = 0
    train_ind = np.arange(len(paths))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    mb_paths = []
    mb_ts = [] # ラベル
    for i in range(len(paths)//mb):
        if mbi + mb > len(paths):
            mb_ind = copy.copy(train_ind)[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(paths)-mbi))]))
            mbi = mb - (len(paths) - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb
        print(f"{i} {mb_ind}")
        mb_paths.append([paths[ind] for ind in mb_ind])
        mb_ts.append([ts[ind] for ind in mb_ind])
    return mb_paths, np.array(mb_ts)

def load_mb_xs(mb_path:list, img_height=224, img_width=224, channel=3, is_pytorch=True):
    """
    ミニバッチ単位で画像ロード
    https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_dataset
    Usage:
        xs, ts, paths = load_my_data('./data/', img_height=0, img_width=0)
        mb_paths, mb_ts = make_batch(paths, ts=ts)
        for mb_path, mb_t in zip(mb_paths, mb_ts)
            mb_xs = load_mb_xs(mb_path)
    """
    mb_xs = np.ndarray((0, img_height, img_width, channel))
    for path in mb_path:
        x = cv2.imread(path)
        x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
        x /= 255.
        mb_xs = np.r_[mb_xs, x[None, ...]]
    if is_pytorch:
        # PyTorch, Chainer, Caffe [データ数、チャネル、画像の縦サイズ、画像の横サイズ]
        mb_xs = mb_xs.transpose(0,3,1,2)
    return mb_xs