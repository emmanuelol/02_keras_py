# -*- coding: utf-8 -*-
"""
Grad-CAM

Usage1:  全クラス(タスク)のgradcam画像作成する
    keras.backend.set_learning_phase(0) #Test時には0にセット modelロード前にこれがないとGradCamエラーになる

    # GradCam出力先
    out_grad_cam_dir = os.path.join(out_dir, 'grad_cam/test')

    model = keras.models.load_model(os.path.join(out_dir, 'best_model.h5'), compile=False)

    layer_name = 'mixed10'

    # 3次元numpy.array型の画像データ（*1./255.前）
    x = d_cls.X_train[0]*255.0
    input_img_name = 'train0'

    # 1画像について各タスクのGradCamを計算
    grad_cam.nobranch_multi_grad_cam(model, out_grad_cam_dir, input_img_name, x, layer_name, shape[0], shape[1])

Usage2: コマンドラインから指定のクラス(タスク)のgradcam画像作成する
    $ PYTHON=/home/bioinfo/.conda/envs/tfgpu_py36/bin/python
    $ export PATH=/usr/local/cuda-8.0/bin:${PATH}
    $ export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:${LD_LIBRARY_PATH}
    $ export CUDA_HOME=/usr/local/cuda-8.0
    $ export KERAS_BACKEND=tensorflow
    $ cat_dog=/gpfsx01/home/aaa00162/jupyterhub/notebook/other/bioinfo_tfgpu_py36_work/grad_cam_test/input/cat_dog.png
    $ out_jpg=./tmp/grad_cat_dog.png
    $ ${PYTHON} tf_grad_cam.py --image_path ${cat_dog} # テスト用。imagenet_vgg16でgradcam。gradcam画像はimage_pathと同じディレクトリに出力
    $ CUDA_VISIBLE_DEVICES=1 ${PYTHON} tf_grad_cam.py --image_path ${cat_dog} --model_path model.h5 # 予測スコア最大クラスを指定モデルの最後のPooling層でgradcam
    $ CUDA_VISIBLE_DEVICES=2 ${PYTHON} tf_grad_cam.py --image_path ${cat_dog} --model_path model.h5 --layer_name mix10 --class_idx 0 --out_jpg ${out_jpg} # gradcamのクラスidや層指定、出力画像パスも措定
"""
import os, sys, time, shutil, glob, pathlib, argparse
import cv2
import pandas as pd
import numpy as np
from scipy.misc import imresize
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras

class GradCAM:
    def __init__(self, model, classIdx:int, layerName=None):
        """
        tensorflow2.0でのGrad-CAM
        https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
        Args:
            model: モデルオブジェクト
            classIdx: 勾配計算したいクラスid
            layerName: 可視化する層の名前。NoneならPoolingの直前の層になる
        """
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def cv2pil(self, image):
        """
        OpenCV型 -> PIL型
        https://qiita.com/derodero24/items/f22c22b22451609908ee
        """
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = new_image[:, :, ::-1]
        elif new_image.shape[2] == 4:  # 透過
            new_image = new_image[:, :, [2, 1, 0, 3]]
        new_image = Image.fromarray(new_image)
        return new_image

    def pil2cv(self, image):
        """
        PIL型 -> OpenCV型
        https://qiita.com/derodero24/items/f22c22b22451609908ee
        """
        new_image = np.array(image, dtype=np.uint8)
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = new_image[:, :, ::-1]
        elif new_image.shape[2] == 4:  # 透過
            new_image = new_image[:, :, [2, 1, 0, 3]]
        return new_image

    def find_target_layer(self):
        """ モデルオブジェクトの最後のPooling層の名前取得（gradcamで出力層に一番近い畳込み層の名前必要なので）"""
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, X: np.ndarray, eps=1e-8):
        """
        Grad-CAM計算してヒートマップ返す
        Args:
            X: 4次元テンソル(1, img_rows, img_cols, 3)へ変換+前処理後画像データ
            eps: [0,255]->[0,1]に正規化するのに使う極小値。変更必要なし
        """
        # 1. modelから勾配計算するモデルを構築
        # 出力(outputs)をlayerNameの層と出力層の2つ出す
        gradModel = keras.models.Model(inputs=[self.model.inputs], outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        # 2. tf.GradientTape（自動微分するのためのAPI。値を計算し、その値の導関数を計算する）で勾配を計算
        with tf.GradientTape() as tape:
            # 画像テンソルをfloat-32データ型にキャストし、画像を勾配モデルに渡し、特定のクラスインデックスに関連する損失を取得
            inputs = tf.cast(X, tf.float32)  # 勾配計算するために画像テンソルをfloat-32データ型にキャスト
            # 勾配計算するモデルでXを順伝播
            # layerNameの層の出力(=convOutputs:サイズが(1, 3, 3, 512)みたいな4次元テンソル)と出力層の出力(=predictions:クラス数の1次元テンソル。tf.Tensor([[0.98 0.02]])みたいなの)がでてくる
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]  # 確認するclassIdxについての予測値だけとる。tf.Tensor([[0.98]])みたいなの
        # tf.GradientTapeを使用し、予測値とlayerNameの層について勾配を計算。convOutputsのサイズと同じ4次元テンソルができる
        grads = tape.gradient(loss, convOutputs)

        # 3. ガイド付き勾配(grads)を計算
        # 0より大きい（=意味のある）layerNameの層の出力だけ取って、float-32データ型にキャストしてテンソルにする。castConvOutputsは 0.0 か 1.0 のどちらかになる
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        # 0より大きい（=特定のclassIdxに寄与している）勾配だけ取って、float-32データ型にキャストしてテンソルにする。castGradsは 0.0 か 1.0 のどちらかになる
        castGrads = tf.cast(grads > 0, "float32")
        # 乗算してcastConvOutputsもcastGradsどちらも1.0である箇所の勾配だけ残す
        guidedGrads = castConvOutputs * castGrads * grads

        # 4. convOutputsとguidedGradsはバッチ次元があるが必要ないので0番目(バッチの1番目の画像)だけ取る
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # 5. フィルターごとに勾配値の平均を計算し、それらを重みとして使用
        # 各フィルターの平均(=Global Average Pooling)をlayerNameの層の出力の重みにするのがGrad-CAMが強力な理由
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # 6. 入力画像のサイズを取得し、cam(layerNameの層の出力に勾配の重みを掛けたもの)のサイズを変更
        (w, h) = (X.shape[2], X.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # 7. すべての値が範囲内に収まるようにヒートマップを正規化する
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom  # 値が[0、1]の範囲になる
        heatmap = (heatmap * 255).astype("uint8")  # 画像に戻すために255掛けて、符号なし8ビット整数に変換

        # heatmapは画像内でネットワークがアクティブ化された場所の単一チャネルのグレースケール表現
        # 大きい値は高いアクティブ化に対応し、小さい値は低いアクティブ化に対応
        return heatmap

    def overlay_heatmap(self, heatmap: np.ndarray, x: np.ndarray, alpha=0.5, colormap=cv2.COLORMAP_JET):  # cv2.COLORMAP_VIRIDIS
        """
        Grad-CAMのヒートマップと元画像を重ねた画像を返す
        Args:
            heatmap: self.compute_heatmap()で出したGrad-CAMのヒートマップ（*1./255.前）
            x: 3次元numpy.array型の画像データ（*1./255.前）
            alpha: heatmapの色の濃さ
            colormap: モノクロのheatmapに疑似的に色をつけるためのcv2のカラーマップ
        """
        x_cv2 = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)#self.pil2cv(x) # cv2で画像オーバレイするからRGBをBGRに変換
        heatmap_cv2 = cv2.applyColorMap(heatmap, colormap) # cv2.applyColorMap()でモノクロのheatmapに疑似的に色をつける
        output_cv2 = cv2.addWeighted(np.uint8(x_cv2), alpha, heatmap_cv2, 1 - alpha, 0) # ヒートマップを入力画像にオーバーレイ
        # RGBに戻しておく
        heatmap = cv2.cvtColor(heatmap_cv2, cv2.COLOR_BGR2RGB)
        output = cv2.cvtColor(output_cv2, cv2.COLOR_BGR2RGB)
        return (heatmap, output)



def preprocess_x(x):
    """
    4次元テンソルへ変換+前処理
    Args:
        x: 3次元numpy.array型の画像データ（*1./255.前）
    Returns:
        X: 4次元テンソル(1, img_rows, img_cols, 3)へ変換+前処理後画像データ
    """
    x = np.expand_dims(x, axis=0)# 4次元テンソルへ変換
    x = x.astype('float32')
    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！ これを忘れると結果がおかしくなるので注意
    X = x/255.0# 前処理
    return X

def judge_evaluate(y_pred, y_true, positive=1, negative=0):
    """
    TP/FN/FP/TN を判定した結果を返す
    Args:
        y_pred:予測ラベル
        y_true:正解ラベル
        positive:positiveラベル
        negative:negativeラベル
    Returns:
        judge:TP/FN/FP/TN/NAN いずれか
    """
    if (positive == y_true) and (positive == y_pred):
        judge = 'TP'
    elif(negative == y_true) and (positive == y_pred):
        judge = 'FP'
    elif(positive == y_true) and (negative == y_pred):
        judge = 'FN'
    elif(negative == y_true) and (negative == y_pred):
        judge = 'TN'
    else:
        # 欠損ラベルの時NANとする
        judge = 'NAN'
    return judge

def nobranch_multi_grad_cam(model, out_grad_cam_dir, input_img_name, x, y_true
                            , layer_name=None
                            , img_rows=None, img_cols=None
                            , pred_threshold=None#0.5
                            , grad_threshold=None#-1.0
                            , is_gradcam_plus=False, predicted_score=None, run_gradcam_task_idx_list=None):
    """
    出力層のニューラルネットワークに分岐がない場合で
    1画像について各タスクのGradCamを計算
    Args:
       model: モデルオブジェクト
       out_grad_cam_dir: GradCam画像出力先ディレクトリ
       input_img_name: 入力画像名（出力画像名に使う）
       x: 3次元numpy.array型の画像データ（*1./255.前）
       y_true: 正解ラベル 5taskなら[0,1,1,0,0]のようなラベル
       layer_name: Poolingの直前の層の名前
       img_rows, img_cols: モデルの入力層のサイズ
       pred_threshold: 予測スコアのposi/nega分ける閾値。バイナリ分類なので、デフォルトはNone(0.5)とする
       grad_threshold: GradCam実行するか決める予測スコアの閾値。デフォルトはNoneとして必ず実行
       is_gradcam_plus: gradcam++で実行するか。Falseだと普通のgradcam実行
       predicted_score: 予測済みpredict score
       run_gradcam_task_idx_list: GradCam実行するタスクidのリスト。Noneの時は全タスクGradCam実行する
    Returns:
       最後のtaskのGrad-cam画像1枚（内部処理で全taskのGradCam画像を出力してる）
    """
    if img_rows is None or img_cols is None:
        shape = [model.input_shape[1], model.input_shape[2], model.input_shape[3]] # モデルオブジェクトの入力層のサイズ取得
        img_rows, img_cols = shape[0], shape[1]

    if layer_name is None:
        layer_name = get_last_conv_layer_name(model) # layer_nameなければモデルオブジェクトの最後の畳込み層の名前取得

    # 推論実行するための前処理（画像を読み込んで4次元テンソルへ変換+前処理）
    X = preprocess_x(x)
    if predicted_score is None:
        # 推論 予測スコア算出
        pred_score = model.predict(X)
    else:
        pred_score = predicted_score

    # 各タスクごとに予測及びGrad-Camのしきい値変えるか判定
    # pred_threshold, grad_threshold がNoneでないならそのまま各タスクに対するしきい値のリストにするだけ
    pred_threshold_list = pred_threshold
    if pred_threshold_list is None:
        pred_threshold_list = [0.5] * pred_score.shape[1]
    grad_threshold_list = grad_threshold
    if grad_threshold_list is None:
        grad_threshold_list = [-1.0] * pred_score.shape[1]

    # 5タスクならpred_score = [[ 0.046  0.04977  0.4  0.96  0.085]] のようなスコア出る
    # Grad-Camで勾配計算するところで、特定のタスクの出力 model.output[:, task_idx] が必要
    # マルチタスクだから各タスクのcamを計算すべき
    #（シングルタスクの時はスコア最大のクラス（class_output = model.output[:, np.argmax(pred_score[0])]）を選んでいる）
    grad_cam_img = None
    for task_idx in range(pred_score.shape[1]):
        # taskのscore
        pred_score_task = pred_score[0,task_idx]
        # バイナリ分類なので、確信度がpred_threshold_list[task_idx]より大きい推論を1、それ以外を0に置換する
        y_pred = (pred_score_task > pred_threshold_list[task_idx]) * 1.0
        #print(pred_score_task, pred_threshold_list[task_idx], y_pred)
        # スコアの桁省略
        pred_score_task_form = "{0:.2f}".format(pred_score_task)
        #print(pred_score.shape, len(grad_threshold_list), task_idx, pred_score_task_form)
        # GradCam実行するタスク指定する場合
        is_run_gradcam_task_idx = True
        if run_gradcam_task_idx_list is not None:
            is_run_gradcam_task_idx = task_idx in run_gradcam_task_idx_list
        # run_gradcam_task_idx_list に存在しない task_idx はGradCam実行しない
        if is_run_gradcam_task_idx == False:
            continue

        # taskのscoreが閾値を超えていたらGradCam実行する
        #print(f'{pred_score_task} < {grad_threshold_list[task_idx]}')
        if float(pred_score_task) < float(grad_threshold_list[task_idx]):
            #print(f'{pred_score_task} < {grad_threshold_list[task_idx]}')
            continue

        # task_id を出力パスに含める
        task_out_grad_cam_dir = os.path.join(out_grad_cam_dir, 'task'+str(task_idx))

        # task_idの特定のクラスでGrad-Cam実行
        cam = GradCAM(model, task_idx, layer_name)
        heatmap = cam.compute_heatmap(X)
        heatmap, grad_cam_img = cam.overlay_heatmap(heatmap, x)
        grad_cam_img = keras.preprocessing.image.array_to_img(grad_cam_img)

        if y_true is not None:
            # 正解ラベルあれば TP/FN/FP/TN/NAN を判定し、判定結果を出力パスに含める
            judge = judge_evaluate(y_pred, y_true[task_idx], positive=1., negative=0.)
            judge_out_grad_cam_dir = os.path.join(task_out_grad_cam_dir, judge)
            # ファイル出力先作成
            os.makedirs(judge_out_grad_cam_dir, exist_ok=True)
            # Grad-cam画像保存
            out_jpg = input_img_name+'_task'+str(task_idx)+'_score='+str(pred_score_task_form)+'.jpg'
            grad_cam_img.save(os.path.join(judge_out_grad_cam_dir, out_jpg), 'JPEG', quality=100, optimize=True)
        else:
            # ファイル出力先作成
            os.makedirs(task_out_grad_cam_dir, exist_ok=True)
            # Grad-cam画像保存
            out_jpg = input_img_name+'_task'+str(task_idx)+'_score='+str(pred_score_task_form)+'.jpg'
            grad_cam_img.save(os.path.join(task_out_grad_cam_dir, out_jpg), 'JPEG', quality=100, optimize=True)

        # Grad-cam画像表示
        #print(out_jpg)
        #plt.imshow(grad_cam_img)
        #plt.show()
        #plt.clf() # plotの設定クリアにする
    return grad_cam_img

def get_last_conv_layer_name(model):
    """ モデルオブジェクトの最後のPooling層の名前取得（gradcamで出力層に一番近い畳込み層の名前必要なので） """
    for i in range(len(model.layers)):
        #print(i, model.layers[i], model.layers[i].name, model.layers[i].trainable, model.layers[i].output)
        if len(model.layers[i].output.shape) == 4:
            last_conv_layer_name = model.layers[i].name
    return last_conv_layer_name

def image2numpy_keras(image_path:str, shape):
    """
    kerasのAPIで画像ファイルをリサイズしてnp.arrayにする
    Args:
        image_path:画像ファイルパス
        target_size:リサイズする画像サイズ.[331,331,3]みたいなの
    """
    img = keras.preprocessing.image.load_img(image_path, target_size=shape[:2])
    x = keras.preprocessing.image.img_to_array(img)
    return x

def image2gradcam(model, image_path:str, X=None, layer_name=None, class_idx=None, out_jpg=None, out_dir=None):
    """
    画像ファイル1枚からGradCam実行して画像保存
    Args:
       model:モデルオブジェクト
       image_path:入力画像パス
       X:4次元numpy.array型の画像データ（*1./255.後）。Noneならimage_pathから作成する
       layer_name:GradCamかける層の名前。Noneならモデルの最後のPooling層の名前取得にする
       class_idx:GradCamかけるクラスid。Noneなら予測スコア最大クラスでGradCamかける
       out_jpg,out_dir:GradCam画像出力先パス。Noneならimage_pathから作成する
    """
    shape = [model.input_shape[1], model.input_shape[2], model.input_shape[3]] # モデルオブジェクトの入力層のサイズ取得
    #x_cv = cv2.imread(image_path) # gradcamの処理は全てcv2でロードするため、cv2で画像ロードしないとエラーになる
    #x_cv = cv2.resize(x_cv, (shape[0], shape[0]))
    x = image2numpy_keras(image_path, shape) # 画像ファイルをリサイズしてnp.arrayにする

    if X is None:
        X = preprocess_x(x) # np.arrayの画像前処理

    if layer_name is None:
        layer_name = get_last_conv_layer_name(model) # layer_nameなければモデルオブジェクトの最後の畳込み層の名前取得

    if class_idx is None:
        pred_score = model.predict(X)[0]
        class_idx = np.argmax(pred_score) # class_idxなければ予測スコア最大クラスでgradcamかける

    # class_idxの特定のクラスでGrad-Cam実行
    cam = GradCAM(model, class_idx, layer_name)
    heatmap = cam.compute_heatmap(X)
    heatmap, grad_cam_img = cam.overlay_heatmap(heatmap, x)
    grad_cam_img = keras.preprocessing.image.array_to_img(grad_cam_img)

    # Grad-Cam画像保存
    if out_jpg is None:
        if out_dir is None:
            out_dir = str(pathlib.Path(image_path).parent)
        out_jpg = out_dir+'/'+str(pathlib.Path(image_path).stem)+f"_classidx{class_idx}_gradcam.jpg"
        print(f"out_jpg: {out_jpg}")
    grad_cam_img.save(out_jpg, 'JPEG', quality=100, optimize=True)

    return grad_cam_img

def main(args):
    if args.model_path is None:
        #import urllib.request
        ## proxy の設定
        #proxy_support = urllib.request.ProxyHandler({'http' : 'http://apiproxy:8080', 'https': 'https://apiproxy:8080'})
        #opener = urllib.request.build_opener(proxy_support)
        #urllib.request.install_opener(opener)
        # テスト用にモデルファイルなしなら imagenet_vgg16 で gradcam 実行できるようにしておく
        model = keras.applications.vgg16.VGG16(weights='imagenet') # imagenet_vgg16モデルロード
        x = image2numpy_keras(args.image_path, [224,224,3]) # 画像ファイルをリサイズしてnp.arrayにする
        X = np.expand_dims(x, axis=0)
        X = X.astype('float32')
        X = keras.applications.vgg16.preprocess_input(X) # imagenet_vgg16の画像前処理
        grad_cam_img = image2gradcam(model, args.image_path, X=X, layer_name='block5_conv3')
    else:
        model = keras.models.load_model(args.model_path, compile=False) # モデルロード
        grad_cam_img = image2gradcam(model, args.image_path, layer_name=args.layer_name, class_idx=args.class_idx, out_jpg=args.out_jpg)
    return

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_path", type=str, required=True, help="input image path.")
    ap.add_argument("-m", "--model_path", type=str, default=None, help="model path.")
    ap.add_argument("-l", "--layer_name", type=str, default=None, help="gradcam layer_name.")
    ap.add_argument("-c_i", "--class_idx", type=int, default=None, help="gradcam class_idx.")
    ap.add_argument("-o_j", "--out_jpg", type=str, default=None, help="output gradcam jpg path.")
    args = ap.parse_args()

    keras.backend.clear_session()
    keras.backend.set_learning_phase(0)

    main(args)
