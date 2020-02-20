# -*- coding: utf-8 -*-
"""
Grad-CAM
https://qiita.com/haru1977/items/45269d790a0ad62604b3 を参考に作成

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
    grad_cam.branch_multi_grad_cam(model, out_grad_cam_dir, input_img_name, x, layer_name, shape[0], shape[1])
    grad_cam.nobranch_multi_grad_cam(model, out_grad_cam_dir, input_img_name, x, layer_name, shape[0], shape[1])

Usage2: コマンドラインから指定のクラス(タスク)のgradcam画像作成する
    $ PYTHON=/home/bioinfo/.conda/envs/tfgpu_py36/bin/python
    $ export PATH=/usr/local/cuda-8.0/bin:${PATH}
    $ export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:${LD_LIBRARY_PATH}
    $ export CUDA_HOME=/usr/local/cuda-8.0
    $ export KERAS_BACKEND=tensorflow
    $ cat_dog=/gpfsx01/home/aaa00162/jupyterhub/notebook/other/bioinfo_tfgpu_py36_work/grad_cam_test/input/cat_dog.png
    $ out_jpg=./tmp/grad_cat_dog.png
    $ ${PYTHON} grad_cam.py --image_path ${cat_dog} # テスト用。imagenet_vgg16でgradcam。gradcam画像はimage_pathと同じディレクトリに出力
    $ CUDA_VISIBLE_DEVICES=1 ${PYTHON} grad_cam.py --image_path ${cat_dog} --model_path model.h5 # 予測スコア最大クラスを指定モデルの最後のPooling層でgradcam
    $ CUDA_VISIBLE_DEVICES=2 ${PYTHON} grad_cam.py --image_path ${cat_dog} --model_path model.h5 --layer_name mix10 --class_idx 0 --out_jpg ${out_jpg} # gradcamのクラスidや層指定、出力画像パスも措定
    $ CUDA_VISIBLE_DEVICES=3 ${PYTHON} grad_cam.py --image_path ${cat_dog} --model_path model.h5 --is_gradcam_plus # gradcam++で実行
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

def grad_cam(model, X, x, layer_name, img_rows, img_cols, class_output):
    '''
    GradCam
    Args:
       model: モデルオブジェクト
       X: 4次元テンソル(1, img_rows, img_cols, 3)へ変換+前処理後画像データ
       x: 3次元numpy.array型の画像データ（*1./255.前）
       layer_name: Poolingの直前の層の名前
       img_rows, img_cols: モデルの入力層のサイズ
       class_output: 勾配計算したいクラスの出力
    Returns:
       jetcam: 影響の大きい箇所を色付けした画像(array)
    '''
    # 勾配を取得
    conv_output = model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット

    # https://www.aipacommander.com/entry/2019/05/14/000250
    g = tf.Graph()
    with g.as_default():
        grads = tf.gradients(class_output, conv_output)[0]
    ##grads = keras.backend.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    gradient_function = keras.backend.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数

    output, grads_val = gradient_function([X])
    output, grads_val = output[0], grads_val[0]

    # 重みを平均化して、レイヤーのアウトプットに乗じる
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像化してヒートマップにして合成
    # ヒートマップでない問題は畳み込み層が小さいせいではなく、着目してるところが無いため（camの値が全てマイナス。注目してるところはプラスの値で出る）
    cam = cv2.resize(cam, (img_rows, img_cols), cv2.INTER_LINEAR)# 線形補完（cv2.INTER_LINEAR）でモデルの入力層のサイズにcamを引き延ばす
    cam = np.maximum(cam, 0)# camから0より大きい要素はそのままそれ以外は0にする。np.maximum:要素ごと比較して、大きい方を格納した配列を返す
    #cam = zoom(cam, img_cols/cam.shape[0]) # githubのGradCam++にあわせる場合 cv2.resizeするよりもヒートマップきれいに出るがノイズが混じるときがある
    cam = cam / cam.max()# camのピクセルの値を0-1に正規化

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)# camのピクセルの値を0-255に戻す（モノクロ画像に疑似的に色をつける）
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)# 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2)# もとの画像に合成 ここのｘはピクセルの値を0-255の3次元のもと画像データを使うこと！！！（4次元テンソルではエラーになる）

    return jetcam

def grad_cam_plus(model, X, x, layer_name, img_rows, img_cols, class_output):
    """
    GradCam++
    https://github.com/totti0223/gradcamplusplus
    """
    input_model = model
    img, H, W = X, img_cols, img_rows

    y_c = class_output
    #cost = 全部のラベルの値。cost*label_indexでy_cになる
    conv_output = input_model.get_layer(layer_name).output
    #conv_output = target_conv_layer, mixed10の出力1,5,5,2048

    # https://www.aipacommander.com/entry/2019/05/14/000250
    g = tf.Graph()
    with g.as_default():
        grads = tf.gradients(y_c, conv_output)[0]
    ####grads = keras.backend.gradients(y_c, conv_output)[0]
    #grads = normalize(grads)

    first = keras.backend.exp(y_c)*grads
    second = keras.backend.exp(y_c)*grads*grads
    third = keras.backend.exp(y_c)*grads*grads*grads

    gradient_function = keras.backend.function([input_model.input], [y_c,first,second,third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([img])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)

    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)

    alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))

    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
    #print deep_linearization_weights
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    # Passing through ReLU
    grad_CAM_map = cv2.resize(grad_CAM_map, (img_rows, img_cols), cv2.INTER_LINEAR)# 線形補完（cv2.INTER_LINEAR）でモデルの入力層のサイズにcamを引き延ばす
    cam = np.maximum(grad_CAM_map, 0)
    #cam = zoom(cam,H/cam.shape[0]) # githubのコードはこれにしている。cv2.resizeするよりもヒートマップきれいに出るがノイズが混じるときがある
    cam = cam / np.max(cam) # scale 0 to 1.0
    #cam = resize(cam, (224,224))

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)# camのピクセルの値を0-255に戻す（モノクロ画像に疑似的に色をつける）
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)# 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2)# もとの画像に合成 ここのｘはピクセルの値を0-255の3次元のもと画像データを使うこと！！！（4次元テンソルではエラーになる）

    return jetcam

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

def branch_multi_grad_cam(model, out_grad_cam_dir, input_img_name, x, y_true
                          , layer_name=None
                          , img_rows=None, img_cols=None
                          , pred_threshold=None#0.5
                          , grad_threshold=None#-1.0
                          , is_gradcam_plus=False, predicted_score=None, run_gradcam_task_idx_list=None):
    """
    出力層のニューラルネットワークに分岐がある場合で
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

    X = preprocess_x(x)
    if predicted_score is None:
        pred_score = np.array([p_task[0] for p_task in model.predict(X)])
    else:
        pred_score = predicted_score
    #print(f"pred_score, pred_score.shape: {pred_score}, {pred_score.shape}")
    #print(f"pred_score[0][0], pred_score[0][1]: {pred_score[0][0]}, {pred_score[0][1]}")

    # 各タスクごとに予測及びGrad-Camのしきい値変えるか判定
    # pred_threshold, grad_threshold がNoneでないならそのまま各タスクに対するしきい値のリストにするだけ
    pred_threshold_list = pred_threshold
    if pred_threshold_list is None:
        pred_threshold_list = [0.5] * pred_score.shape[0]
    grad_threshold_list = grad_threshold
    if grad_threshold_list is None:
        grad_threshold_list = [-1.0] * pred_score.shape[0]
    #print(f"pred_threshold_list, grad_threshold_list:{pred_threshold_list}, {grad_threshold_list}")

    grad_cam_img = None
    for task_idx in range(len(model.output)):
        for class_idx in range(y_true.ndim):
            # task_idの各クラスのscore
            if pred_score.ndim == 2:
                pred_score_task = pred_score[task_idx][class_idx]
            elif pred_score.ndim == 3:
                pred_score_task = pred_score[task_idx][0][class_idx]
            #print(f"pred_score_task_{task_idx}_{class_idx}: {pred_score_task}")

            # バイナリ分類なので、確信度がpred_threshold_list[task_idx]より大きい推論を1、それ以外を0に置換する
            y_pred = (pred_score_task > pred_threshold_list[task_idx]) * 1.0
            # スコアの桁省略
            pred_score_task_form = "{0:.2f}".format(pred_score_task)

            # GradCam実行するタスク指定する場合
            is_run_gradcam_task_idx = True
            if run_gradcam_task_idx_list is not None:
                is_run_gradcam_task_idx = task_idx in run_gradcam_task_idx_list
            # run_gradcam_task_idx_list に存在しない task_idx はGradCam実行しない
            if is_run_gradcam_task_idx == False:
                continue

            # taskのscoreが閾値を超えていたらGradCam実行する
            if float(pred_score_task) < float(grad_threshold_list[task_idx]):
                continue

            # task_id を出力パスに含める
            task_out_grad_cam_dir = os.path.join(out_grad_cam_dir, 'task'+str(task_idx))

            # branchのmultitaskだと、model.outputがリスト
            task_output = model.output[task_idx]
            #print(f"task_output: {task_output}")

            # task_idの特定のクラスでgradcam計算
            class_output = task_output[:, class_idx] # task_output[0, class_idx] でないとエラーになるケースあった
            #print(f"class_output: {class_output}")

            # Grad-Cam実行
            if is_gradcam_plus == True:
                jetcam = grad_cam_plus(model, X, x, layer_name, img_rows, img_cols, class_output)
            else:
                jetcam = grad_cam(model, X, x, layer_name, img_rows, img_cols, class_output)
            grad_cam_img = keras.preprocessing.image.array_to_img(jetcam)

            if y_true is not None:
                if y_true.ndim == 1:
                    y_t = y_true[task_idx]
                else:
                    y_t = y_true[task_idx][class_idx]
                #print(f"y_t: {y_t}")

                # 正解ラベルあれば TP/FN/FP/TN/NAN を判定し、判定結果を出力パスに含める
                judge = judge_evaluate(y_pred, y_t, positive=1., negative=0.)
                judge_out_grad_cam_dir = os.path.join(task_out_grad_cam_dir, judge)
                #print(f"judge_out_grad_cam_dir: {judge_out_grad_cam_dir}")
                # ファイル出力先作成
                os.makedirs(judge_out_grad_cam_dir, exist_ok=True)
                # Grad-cam画像保存
                out_jpg = input_img_name+'_task'+str(task_idx)+'_'+str(class_idx)+'_score='+str(pred_score_task_form)+'.jpg'
                grad_cam_img.save(os.path.join(judge_out_grad_cam_dir, out_jpg), 'JPEG', quality=100, optimize=True)
            else:
                # ファイル出力先作成
                os.makedirs(task_out_grad_cam_dir, exist_ok=True)
                # Grad-cam画像保存
                out_jpg = input_img_name+'_task'+str(task_idx)+'_'+str(class_idx)+'_score='+str(pred_score_task_form)+'.jpg'
                grad_cam_img.save(os.path.join(task_out_grad_cam_dir, out_jpg), 'JPEG', quality=100, optimize=True)
            # Grad-cam画像表示
            #print(out_jpg)
            #plt.imshow(grad_cam_img)
            #plt.show()
            #plt.clf() # plotの設定クリアにする
    return grad_cam_img

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
        # Grad-Cam実行
        task_output = model.output[:, task_idx] # model.output[0, task_idx] でないとエラーになるケースあった
        #print(keras.backend.dtype(task_output))
        #print(f"keras.backend.eval(task_output), model.output.shape:, {keras.backend.eval(task_output)}, {model.output.shape}")
        if is_gradcam_plus == True:
            jetcam = grad_cam_plus(model, X, x, layer_name, img_rows, img_cols, task_output)
        else:
            jetcam = grad_cam(model, X, x, layer_name, img_rows, img_cols, task_output)
        grad_cam_img = keras.preprocessing.image.array_to_img(jetcam)

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

def image2gradcam(model, image_path:str, X=None, layer_name=None, class_idx=None, out_jpg=None, out_dir=None, is_gradcam_plus=False):
    """
    画像ファイル1枚からGradCam実行して画像保存
    Args:
       model:モデルオブジェクト
       image_path:入力画像パス
       X:4次元numpy.array型の画像データ（*1./255.後）。Noneならimage_pathから作成する
       layer_name:GradCamかける層の名前。Noneならモデルの最後のPooling層の名前取得にする
       class_idx:GradCamかけるクラスid。Noneなら予測スコア最大クラスでGradCamかける
       out_jpg,out_dir:GradCam画像出力先パス。Noneならimage_pathから作成する
       is_gradcam_plus:gradcam++で実行するか。Falseだと普通のgradcam実行
    """
    shape = [model.input_shape[1], model.input_shape[2], model.input_shape[3]] # モデルオブジェクトの入力層のサイズ取得
    x = image2numpy_keras(image_path, shape) # 画像ファイルをリサイズしてnp.arrayにする

    if X is None:
        X = preprocess_x(x) # np.arrayの画像前処理

    if layer_name is None:
        layer_name = get_last_conv_layer_name(model) # layer_nameなければモデルオブジェクトの最後の畳込み層の名前取得

    if class_idx is None:
        pred_score = model.predict(X)[0]
        class_idx = np.argmax(pred_score) # class_idxなければ予測スコア最大クラスでgradcamかける

    # Grad-Cam実行
    class_output = model.output[:, class_idx]
    if is_gradcam_plus == True:
        jetcam = grad_cam_plus(model, X, x, layer_name, shape[0], shape[1], class_output) # gradcam++で実行
    else:
        jetcam = grad_cam(model, X, x, layer_name, shape[0], shape[1], class_output)
    grad_cam_img = keras.preprocessing.image.array_to_img(jetcam)

    # Grad-Cam画像保存
    if out_jpg is None:
        if out_dir is None:
            out_dir = str(pathlib.Path(image_path).parent)
        if is_gradcam_plus == True:
            out_jpg = out_dir+'/'+str(pathlib.Path(image_path).stem)+f"_classidx{class_idx}_gradcam++.jpg"
        else:
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
        grad_cam_img = image2gradcam(model, args.image_path, X=X, layer_name='block5_conv3', is_gradcam_plus=args.is_gradcam_plus)
    else:
        model = keras.models.load_model(args.model_path, compile=False) # モデルロード
        grad_cam_img = image2gradcam(model, args.image_path, layer_name=args.layer_name, class_idx=args.class_idx, out_jpg=args.out_jpg, is_gradcam_plus=args.is_gradcam_plus)
    return

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", type=str, required=True, help="input image path.")
    ap.add_argument("--model_path", type=str, default=None, help="model path.")
    ap.add_argument("--layer_name", type=str, default=None, help="gradcam layer_name.")
    ap.add_argument("--class_idx", type=int, default=None, help="gradcam class_idx.")
    ap.add_argument("--out_jpg", type=str, default=None, help="output gradcam jpg path.")
    ap.add_argument("--is_gradcam_plus", action='store_const', const=True, default=False, help="Grad-Cam++ flag.")
    args = ap.parse_args()

    keras.backend.clear_session()
    keras.backend.set_learning_phase(0)

    main(args)
