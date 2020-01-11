# -*- coding: utf-8 -*-
""" LIME """
import os, sys
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import lime
import pandas as pd
import numpy as np

from tensorflow import keras

import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得
sys.path.append( str(current_dir) + '/../' )
from transformer import get_train_valid_test

def lime_explanation_from_keras_model(x, model, top_labels=5, hide_color=0, num_samples=1000):
    """
    kerasの画像モデルからLIMEのexplanationインスタンス作成
    Args:
        x: 画像を1枚分の3次元テンソル(img_rows, img_cols, 3)+前処理ずみ
        model: kerasの画像モデル
        top_labels: 予測スコア高い順にLIMEでの予測クラスいくつ保持するか
        hide_color: スーパーピクセルがオフになっている色です。 あるいは、NONEの場合、スーパーピクセルはそのピクセルの平均で置き換えられます。
        num_samples: 特徴をランダムに摂動させて作る近傍データいくつ生成するか
    Return:
        LIMEのexplanationインスタンス
    """
    explainer = lime.lime_image.LimeImageExplainer()
    # インスタンスから特徴をランダムに摂動させることによって近傍データを生成する
    explanation = explainer.explain_instance(x, model.predict
                                             , top_labels=top_labels
                                             , hide_color=hide_color
                                             , num_samples=num_samples)
    return explanation

def plot_lime_skimage_segmentation(explanation, out_jpg=None, id=0, positive_only=True, num_features=10, hide_rest=True, min_weight=0.0):
    """
    skimageでLIMEの結果可視化
    Args:
        explanation: LIMEのexplanationインスタンス
        out_jpg: 出力画像のパス
        id: 予測クラスのid. 1位のデータだけ見たいなら explanation.top_labels[0]
        positive_only: Trueの場合、ラベルの予測に寄与するスーパーピクセルのみ表示させる
        num_features: 可視化する領域の数
        hide_rest: 注目領域以外を隠すか。Trueならnum_features*注目領域だけ表示させる
        min_weight: min_weight以上のスコアのクラスのみ表示させる（positive_only=Falseのときでnegativeクラス消したいときとかに使う）
    Return:
        なし
    """
    temp, mask = explanation.get_image_and_mask(id
                                                , positive_only=positive_only
                                                , num_features=num_features
                                                , hide_rest=hide_rest
                                                , min_weight=min_weight)
    lime_array = mark_boundaries(temp / 2 + 0.5, mask)
    if out_jpg is not None:
        lime_img = keras.preprocessing.image.array_to_img(lime_array)
        lime_img.save(out_jpg, 'JPEG', quality=100, optimize=True)
    plt.imshow(lime_array)
    plt.show()
    plt.clf()
