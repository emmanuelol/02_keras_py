# -*- coding: utf-8 -*-
"""
山登り法による画像再構成
- https://colab.research.google.com/drive/1NPmqKFMpXeLxZk8MB5ozC7SMSvbxzsEb?usp=sharing
灰色画像からスタートして、正解画像との画素の差分（=勾配とみたてる）を比較して、色付き円を書き加え、元画像を再構成していく

※山下り法：近傍の内で最も成績の良いものが現在の解より良ければ入れ替えるを繰り返す最適化手法。初期値によって局所解になりうる。
　バッチ学習:
　　勾配法 - 最急勾配法，(準)ニュートン法，L-BFGS
　オンライン学習
　　確率的勾配降下法 (SGD)
　https://hayashibe.jp/note/ml/optimization/

Usage:
    $ activate tfgpu20
    $ python img_reconstruct_hill_climbing.py -o D:\work\02_keras_py\experiment\01_code_test\output_test\img_reconstruct
    $ python img_reconstruct_hill_climbing.py -i D:\work\02_keras_py\experiment\01_code_test\output_test\ndimage.save.jpg
    $ python img_reconstruct_hill_climbing.py -i D:\iPhone_pictures\2019-11\IMG_2194.JPG
    $ python img_reconstruct_hill_climbing.py -i D:\iPhone_pictures\2019-11\IMG_2491.PNG -r 10
    $ python img_reconstruct_hill_climbing.py -i D:\iPhone_pictures\2019-10\IMG_2102.JPG
"""
import os
import argparse
import pathlib
import matplotlib.pyplot as plt
import cv2
import numpy
import requests
import urllib
from tqdm import tqdm

annealing_im, current_radius = None, None


def annealing_draw(original_im, current_im=None, max_circle_radius=100.0, round=1, cool_down_count=100):
    """
    山登り法による画像再構成
    灰色画像からスタートして、正解画像との画素の差分（=勾配とみたてる）を比較して、色付き円を書き加え、元画像を再構成していく
    """
    if current_im is None:
        # make gray image（灰色画像からスタート）
        current_im = numpy.full(original_im.shape, 127, dtype=numpy.uint8)
        # current_im = numpy.zeros(original_im.shape, dtype=numpy.uint8)

    working_im = current_im.copy()
    # cv2.absdiff():画像の差の絶対値を計算
    current_diffrence = cv2.absdiff(original_im, current_im).sum()

    for r in range(round):
        failed_count = 0
        while 1:
            circle_image = working_im.copy()
            # ランダムに色付き円を書き加えていく
            radius = numpy.random.randint(1, int(max_circle_radius + 2))
            center_x = numpy.random.randint(0, original_im.shape[1])
            center_y = numpy.random.randint(0, original_im.shape[0])
            color_r = numpy.random.randint(0, 255)
            color_g = numpy.random.randint(0, 255)
            color_b = numpy.random.randint(0, 255)
            circle_image = cv2.circle(
                circle_image,
                (center_x, center_y),
                radius,
                (color_r, color_g, color_b),
                -1)
            alpha = 0.3
            beta = 1 - alpha
            # cv2.addWeighted():2つの配列の加重和を計算 色付き円と画像混ぜ合わせ
            alphaed_im = cv2.addWeighted(circle_image, alpha, working_im, beta, 0)

            # 色付き円を書き加えた画像と正解画像との画素の差分（=勾配とみたてる）
            annealing_diffrence = cv2.absdiff(original_im, alphaed_im).sum()

            # 勾配(=annealing_diffrence)が更新されたら、このラウンド終わる。ラウンドは最適化を行う回数。epoch数と同じと思っていい
            if annealing_diffrence < current_diffrence:
                current_diffrence = annealing_diffrence
                working_im = alphaed_im
                break
            else:
                # 勾配(=annealing_diffrence)が何度も更新されなかったら、色付き円の半径を狭める
                failed_count += 1
                if failed_count > cool_down_count:
                    failed_count = 0
                    max_circle_radius *= 0.95

    return working_im, max_circle_radius, current_diffrence


def next_generation(is_plot=True):
    global annealing_im, current_radius
    annealing_im, current_radius, diffrence = annealing_draw(original_im, annealing_im, current_radius, round=100)
    print('current_radius, diffrence:', current_radius, diffrence)
    if is_plot:
        plt.imshow(annealing_im)
        plt.show()
    return annealing_im


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default=r'D:\work\02_keras_py\experiment\01_code_test\output_test\img_reconstruct', help="results output dir path.")
    ap.add_argument("-i", "--input_img_path", type=str, default=None, help="input image path.")
    ap.add_argument("-r", "--round", type=int, default=100, help="勾配計算round数.")
    return vars(ap.parse_args())


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    args = get_args()

    # 入力画像無ければセーラームーン画像で実行
    if args['input_img_path'] is None:
        filename = "sailormoonredraw.jpeg"
        raw_image_data = urllib.request.urlopen("https://pbs.twimg.com/media/EYf0zecUEAAPPO2?format=jpg&name=large").read()
        with open(filename, "wb") as fp:
            fp.write(raw_image_data)
    else:
        filename = args['input_img_path']

    # 画像ロード
    original_im = cv2.imread(filename, cv2.IMREAD_COLOR)
    original_im = cv2.cvtColor(original_im, cv2.COLOR_BGR2RGB)
    # plt.imshow(original_im)  # 確認用
    print('original_im.dtype, original_im.shape, original_im.sum():', original_im.dtype, original_im.shape, original_im.sum())  # 確認用

    # 実行
    annealing_im, current_radius, diffrence = annealing_draw(original_im, None, round=1)
    pbar = tqdm(range(args['round']))
    for i in pbar:
        pbar.set_description(f"Generation {(i+1) * 100}")
        annealing_im = next_generation(is_plot=False)
        # 処理時間かかるから6roundごとに画像出すようにする
        if args['round'] % 5 == 1:
            plt.imsave(os.path.join(args['output_dir'], str(pathlib.Path(filename).stem) + '_reconstruct' + '.png'), annealing_im)
    plt.imsave(os.path.join(args['output_dir'], str(pathlib.Path(filename).stem) + '_reconstruct' + '.png'), annealing_im)
