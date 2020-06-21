# -*- coding: utf-8 -*-
"""
Keras: CNN中間層出力の可視化
http://pynote.hatenablog.com/entry/keras-visualize-kernel-and-feature-map
https://hazm.at/mox/machine-learning/computer-vision/keras/intermediate-layers-visualization/index.html
"""
import os, sys
import keras
from keras import models
from keras import backend as K
from keras.preprocessing import image
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def visual_cnn_output(model, X, layer_id, vis_cols=8, figsize=(15, 15), output_dir=''):
    """
    モデルのcnnカーネル（フィルター）で畳み込みを行った出力結果を可視化
    画像をカーネルの数だけ並べて表示する
    Args:
        model: modelオブジェクト
        x: 前処理済みのnp.array型の画像データ
        layer_id: 可視化する層id
        vis_cols: 並べて可視化する画像の列数
        figsize: 並べて可視化する画像のサイズ
        output_dir: 並べて可視化する画像の出力ディレクトリ
    """
    # 指定のレイヤー
    layer = model.layers[layer_id]
    name = layer.name
    # 中間層の特徴マップを返す関数を作成
    get_feature_map = K.function(inputs=[model.input, K.learning_phase()], outputs=[layer.output])
    # 順伝搬して畳み込み層の特徴マップを取得する。
    features, = get_feature_map([X, False])
    print('features.shape', features.shape)  # features.shape (1, 112, 112, 64)
    kernel_num = features.shape[3]
    print('kernel_num', kernel_num)
    # 特徴マップをカーネルごとに分割し、画像化する。
    feature_imgs = []
    for f in np.split(features, kernel_num, axis=3):
        f = np.squeeze(f, axis=0)  # バッチ次元を削除する。
        f = image.array_to_img(f)  # 特徴マップを画像化する。
        f = np.array(f)  # PIL オブジェクトを numpy 配列にする。
        feature_imgs.append(f)
    print('len(feature_imgs):', len(feature_imgs))
    # カーネルで畳み込みを行った出力結果を並べて表示する
    cols = vis_cols
    rows = int(round(kernel_num / cols))
    fig, ax = plt.subplots(rows, cols, figsize=figsize, dpi=200)
    print(ax.shape)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            f_ax = ax[r, c]
            f_ax.imshow(feature_imgs[i], cmap='gray')
            #f_ax.set_title('Feature '+str(i))
            f_ax.set_axis_off()
    plt.savefig(os.path.join(output_dir, "%s.png" % name))
    plt.show(fig)
    plt.close(fig)

def visual_cnn_kernel(model, layer_id, vis_cols=8, figsize=(15, 15)):
    """
    モデルのcnnカーネル（フィルター）の重みを可視化
    画像をカーネルの数だけ並べて表示する
    重みのチャネルが3になってない層はエラーになる
    """
    # モデルの重みを取得
    weights, bias = model.layers[layer_id].get_weights()
    print('layer.name', model.layers[layer_id].name)  # layer.name conv1
    print('weights.shape', weights.shape)  # weights.shape (7, 7, 3, 64)
    print('bias.shape', bias.shape)  # bias.shape (64,)
    kernel_num = int(weights.shape[3])
    print('kernel_num', kernel_num)
    # 重みをカーネルごとに分割し、画像化する。
    weight_imgs = []
    #print(weights)
    for w in np.split(weights, kernel_num, axis=3):
        w = np.squeeze(w, axis=weights.shape[2])  # (KernelH, KernelW, KernelC, 1) -> (KernelH, KernelW, KernelC)
        w = image.array_to_img(w) # 重みを画像化する。
        w = np.array(w)  # PIL オブジェクトを numpy 配列にする。
        weight_imgs.append(w)
    # カーネルの重みを並べて表示する
    cols = vis_cols
    rows = int(kernel_num / cols)
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            w_ax = ax[r, c]
            w_ax.imshow(weight_imgs[i])
            w_ax.set_title('Kernel '+str(i))
            w_ax.set_axis_off()

def make_intermediate_images(model, img, vis_keras_layers=keras.layers.Activation, output_dir=''):
    """
    指定の各層の出力となった特徴強度(output)をすべて連結してヒートマップをファイル出力
    注) seaborn ライブラリ使ってheatmapで可視化している
    Args:
        model: modelオブジェクト
        img: 前処理済みのnp.array型の画像データ
        vis_keras_layers: 可視化する層のデータ型。デフォルトは Activation の層をすべて可視化する
    """
    # 各レイヤー毎のoutputを取得する
    layers = model.layers[1:]
    layer_outputs = [layer.output for layer in layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activation_model.summary()
    # 特徴強度の取得
    activations = activation_model.predict(img)
    for i, activation in enumerate(activations):
        print("%2d: %s" % (i, str(activation.shape)))
    # 特徴強度の保存
    # プーリング層の出力のみに絞る (畳み込み層の出力も可視化できるが量が多くなるため)
    activations = [(layer.name, activation) for layer, activation in zip(layers, activations) if isinstance(layer, vis_keras_layers)]
    # 出力層ごとに特徴画像を並べてseabornでヒートマップ画像として出力
    for i, (name, activation) in enumerate(activations):
        num_of_image = activation.shape[3]
        print('num_of_image', num_of_image)
        cols = math.ceil(math.sqrt(num_of_image))
        rows = math.floor(num_of_image / cols)
        screen = []
        for y in range(0, rows):
            row = []
            for x in range(0, cols):
                j = y * cols + x
                if j < num_of_image:
                    row.append(activation[0, :, :, j])
                else:
                    row.append(np.zeros())
            screen.append(np.concatenate(row, axis=1))
        screen = np.concatenate(screen, axis=0)
        plt.figure(dpi = 200)
        sns.heatmap(screen, xticklabels=False, yticklabels=False, cmap='gray', cbar=False ) # seabornのheatmap グレースケールにして、カラーバー消しとく
        plt.savefig(os.path.join(output_dir, "%s.png" % name)) # 余白なるべく消す , bbox_inches='tight', pad_inches=0
        print('output:', os.path.join(output_dir, "%s.png" % name))
        plt.close()

if __name__ == '__main__':
    print('visualize_cnn.py: loaded as script file')
else:
    print('visualize_cnn.py: loaded as module file')
