# -*- coding: utf-8 -*-
"""
ディープラーニングの注視領域の可視化
https://qiita.com/bele_m/items/a7bb15313e2a52d68865
"""
import os, sys
import cv2
import numpy as np
from tqdm import tqdm
from os import path
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras

def load_img_for_model(filename, target_size=(224, 224)):
    """判定対象画像の読み込み"""
    img = keras.preprocessing.image.load_img(filename, target_size=target_size)
    x_orig = keras.preprocessing.image.img_to_array(img)  # ndarray: (224, 224, 3), float32
    x = np.expand_dims(x_orig, axis=0)  # ndarray: (1, 224, 224, 3), float32
    return x, x_orig

def visualize_mono_map(map_array, base_image=None, output_path=None):
    """
    ヒートマップ表示、重ね合わせ表示
    map_arrayが注視領域の可視化をしたarrayです。
    重ね合わせについては、base_imageに元画像を指定することで行うことができます。
    """
    # 座標軸の削除処理（google colでそのまま画像表示すると画像に格子が出てくるので）
    # https://uepon.hatenadiary.com/entry/2018/03/30/185145
    fig,ax = plt.subplots()
    ax.tick_params(labelbottom="off",bottom="off")
    ax.tick_params(labelleft="off",left="off")
    ax.set_xticklabels([])
    ax.axis('off')
    if map_array.ndim == 3:
        mono_map = np.sum(np.abs(map_array), axis=2)  # マップがカラーだった場合はモノクロに変換する
    else:
        mono_map = map_array
    # マップを正規化（上位・下位10%の画素は見やすさのため飽和させる）
    minimum_value = np.percentile(mono_map, 10)
    maximum_value = np.percentile(mono_map, 90)
    normalized_map = (np.minimum(mono_map, maximum_value) - minimum_value) / (maximum_value - minimum_value)
    normalized_map = np.maximum(normalized_map, 0.)
    if base_image is None:
        plt.imshow(normalized_map, cmap='jet')
    else:
        image_norm = (base_image - base_image.min()) / (base_image.max() - base_image.min())  # 背景画像の正規化
        overlay = np.stack([normalized_map * image_norm[:,:,i] for i in range(3)], axis=2)
        plt.imshow(overlay)
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)

def img_gradient(model, class_idx, x_processed):
    """
    勾配の可視化（単純な方法）
    注視領域を可視化するための、一番基本的な考え方としては、該当クラスのスコア（確率）に影響を与える入力画素を表示することです。
    具体的な算出方法としては、クラススコア（出力）の各画素（入力）に対する微分（各画素に対する勾配）を算出し、そこから画像を作ります。
    クラススコアのTensorと入力画像のTensorから、勾配を表すTensorを定義
    Tensorの値をそれを計算するための関数を定義したのち、具体的な入力値であるx_processedに対して勾配の値を算出
    Arges:
        model: モデルオブジェクト
        class_idx: 予測クラスid
        x_processed: ndarray: (1, 224, 224, 3), float32
    Usage:
        # 勾配の可視化（単純な方法）
        gradient = vis_gradient(model, class_idx, x_processed)
        # ヒートマップ表示
        visualize_mono_map(gradient, base_image=None)
        # 重ね合わせ表示
        visualize_mono_map(gradient, base_image=x_orig)
    """
    class_output = model.output[:, class_idx]  # Tensor / クラススコア
    grad_tensor = keras.backend.gradients(class_output, model.input)[0]  # Tensor / クラススコアに対する入力の勾配
    grad_func = keras.backend.function([model.input], [grad_tensor])  # Function / 勾配の値を算出するための関数
    gradient = grad_func([x_processed])[0][0]  # ndarray: (224, 224, 3), float32 / 算出された勾配の値
    return gradient

def smooth_grad(model, class_idx, x_processed):
    """
    SmoothGradによるノイズ削減
    勾配の可視化はノイズが多いという課題があるため、ノイズを削減するために提案されたのがSmoothgradと呼ばれる手法です。
    これは、ランダムノイズを与えた複数の入力画像に対して、勾配をそれぞれ計算したうえで、平均する手法のことです
    単純な方法ですが、ノイズを与えても共通して現れる画素のみを抽出することで、単純な勾配よりも注視領域を安定して可視化できます。
    ランダムノイズを乗せた100枚の画像を入力画像から算出し、for文の中で1枚ずつ勾配を算出して平均を求めて画像を算出しています
    単純勾配では、象の領域全体が赤くなっていたのに対して、象の顔や耳付近がより強い特徴として強調されていることがわかります。
    Arges:
        model: モデルオブジェクト
        class_idx: 予測クラスid
        x_processed: ndarray: (1, 224, 224, 3), float32
    Usage:
        # SmoothGradによるノイズ削減
        smooth_grad = smooth_grad(model, class_idx, x_processed)
        # ヒートマップ表示
        visualize_mono_map(smooth_grad, base_image=None)
        # 重ね合わせ表示
        visualize_mono_map(smooth_grad, base_image=x_orig)
    """
    n_samples = 100  # ランダムノイズを乗せて生成される画像の数
    stdev_spread = 0.1  # ライダムノイズの分散のパラメータ（大きいほどランダムノイズを強くする）
    stdev = stdev_spread * (np.max(x_processed) - np.min(x_processed))  # 画像の最大値-最小値でランダムノイズの大きさをスケーリング
    total_gradient = np.zeros_like(x_processed)  # ndarray: (1, 224, 224, 3), float32 / 勾配の合計値を加算していく行列（0で初期化）
    class_output = model.output[:, class_idx]  # Tensor / クラススコア
    grad_tensor = keras.backend.gradients(class_output, model.input)[0]  # Tensor / クラススコアに対する入力の勾配
    grad_func = keras.backend.function([model.input], [grad_tensor])  # Function / 勾配の値を算出するための関数
    pbar = tqdm(range(n_samples))
    for i in range(n_samples):
        #print("SmoothGrad: {}/{}".format(i+1, n_samples))
        pbar.set_description("SmoothGrad: {}/{}".format(i+1, n_samples))
        x_plus_noise = x_processed \
            + np.random.normal(0, stdev, x_processed.shape)  # ndarray: (1, 224, 224, 3), float32 / xにノイズを付加
        total_gradient += grad_func([x_plus_noise])[0]  # ndarray: (1, 224, 224, 3), float32 / サンプルに対する勾配を算出して合計値に加算
    smooth_grad = total_gradient[0] / n_samples  # ndarray: (224, 224, 3), float32 / 勾配の合計値から平均の勾配を算出
    return smooth_grad

def guided_packpropagation(model, class_idx, x_processed, model_save_dir):
    """
    GuidedPackpropagation
    単純な勾配を算出する場合、クラススコアに対して正の影響を与える画素と、負の影響を与える画素を両方合わせて計算しています。
    逆伝搬で勾配を算出する各段階で、負の影響を与えるものを取り除き、正の影響を持つものだけで逆伝搬させていく手法をGuidedBackpropagationと呼ばれています。
    ReLUを使っているアーキテクチャの場合、ReLUを逆伝搬させるときに以下の条件で勾配を0にします。
      純伝搬時に負の要素に対する勾配は0になる（通常の勾配計算と同じ）
      勾配が負であれば、それの要素の勾配も0にする
    ReLU層を、勾配を算出する際に上記の特性を持った層（GuidedReLU層）に置き換えたうえで、勾配を算出しています。
    単純な勾配よりは輪郭を抽出する傾向が強まっているように感じられます。
    Arges:
        model: モデルオブジェクト
        class_idx: 予測クラスid
        x_processed: ndarray: (1, 224, 224, 3), float32
        model_save_dir: モデル保存先
    Usage:
        # GuidedPackpropagation
        guided_packpropagation = guided_packpropagation(model, class_idx, x_processed)
        # ヒートマップ表示
        visualize_mono_map(guided_packpropagation, base_image=None)
        # 重ね合わせ表示
        visualize_mono_map(guided_packpropagation, base_image=x_orig)
    """
    #model_save_dir = base_dir+"keras_visual/model"
    os.makedirs(model_save_dir, exist_ok=True)
    model_temp_path = path.join(model_save_dir, "temp_orig.h5")
    train_save_path = path.join(model_save_dir, "guided_backprop_ckpt")
    # 一回実行したら登録済みになったのか、再実行すると名前重複エラーでる場合はコメントアウトすること
    @tf.RegisterGradient("GuidedRelu")
    def _GuidedReluGrad(op, grad):
        gate_g = tf.cast(grad > 0, "float32")
        gate_y = tf.cast(op.outputs[0] > 0, "float32")
        return gate_y * gate_g * grad
    model.save(model_temp_path)
    with tf.Graph().as_default():
        with tf.Session().as_default():
            keras.backend.set_learning_phase(0)
            keras.models.load_model(model_temp_path)
            session = keras.backend.get_session()
            tf.train.export_meta_graph()
            saver = tf.train.Saver()
            saver.save(session, train_save_path)
    guided_graph = tf.Graph()
    with guided_graph.as_default():
        guided_sess = tf.Session(graph=guided_graph)
        with guided_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            saver = tf.train.import_meta_graph(train_save_path + ".meta")
            saver.restore(guided_sess, train_save_path)
            imported_y = guided_graph.get_tensor_by_name(model.output.name)[0][class_idx]
            imported_x = guided_graph.get_tensor_by_name(model.input.name)
            guided_grads_node = tf.gradients(imported_y, imported_x)
        guided_feed_dict = {imported_x: x_processed}
        sample_gradient = guided_sess.run(guided_grads_node, feed_dict=guided_feed_dict)[0][0]
    return sample_gradient

def gradcam(model, class_idx, x_processed, layer_name="block5_conv3"):
    """
    GradCAM
    これまでの手法は、各画素の出力結果に対する影響度を求めていますが、画素ごとに影響力を求めるため、情報として細かすぎて、結局どこの領域が重要なのかがわかりづらい場合があります。
    CNNでは入力からの出力に近づいた層ほど、解像度が低く、最終的な判定結果と紐づいた特徴を抽出するとされており、領域の抽出には適していると考えられます。
    したがって、出力に最も近い、最後の畳み込み層の特徴マップを可視化すれば、判定結果と最も紐づく領域が得られそうです。そのためには、多くの次元を持つ特徴マップのうち、出力結果に最も関連する次元を求める必要があります。
    そのために、特徴量マップの各次元の出力結果に対する影響度を算出し、特徴量マップの各次元を重みづけ平均して可視化することにより、最終出力に影響度の高い領域を可視化します。
    最終出力(class_output)に対する畳み込み層の出力(counvout_tensor)に対する勾配を算出するための関数を考慮し、勾配を算出します。
    次に各、チャネルに対して、重要度を表すweightsを勾配の大きさを元に作成します。最後に畳み込み層の最終出力（convout_val）に対して、重要度をかけて足し合わせることにより、GradCAMの出力とします。
    これまでの手法に比べると、解像度が低く、よりエリアに焦点が上がっていることがわかります。
    Arges:
        model: モデルオブジェクト
        class_idx: 予測クラスid
        x_processed: ndarray: (1, 224, 224, 3), float32
        layer_name: 可視化する層の名前
    Usage:
        # GradCAM
        gradcam = gradcam(model, class_idx, x_processed)
        # ヒートマップ表示
        visualize_mono_map(gradcam, base_image=None)
        # 重ね合わせ表示
        visualize_mono_map(gradcam, base_image=x_orig)
    """
    class_output = model.output[:, class_idx]  # Tensor / クラススコア
    convout_tensor = model.get_layer(layer_name).output  # convolutionの出力/Tensor
    grad_tensor = keras.backend.gradients(class_output, convout_tensor)[0]  # 勾配/Tensor
    grad_func = keras.backend.function([model.input], [convout_tensor, grad_tensor])  # 勾配を算出する関数
    convout_val, grads_val = grad_func([x_processed])
    convout_val, grads_val = convout_val[0], grads_val[0]  # array: (14, 14, 512), float32 (両方とも）
    weights = np.mean(grads_val, axis=(0,1))  # チャネルの重み/array: (512,), float32
    grad_cam = np.dot(convout_val, weights)  # 畳み込みの出力をチャネルで重みづけ/array, (14, 14), float32
    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = cv2.resize(grad_cam, (224, 224), cv2.INTER_LINEAR)  # 上記をリサイズ
    return grad_cam
