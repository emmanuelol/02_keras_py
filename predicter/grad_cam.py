# -*- coding: utf-8 -*-
"""
Grad-CAM
https://qiita.com/haru1977/items/45269d790a0ad62604b3 を参考に作成

Usage:
K.set_learning_phase(0) #Test時には0にセット modelロード前にこれがないとGradCamエラーになる

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
"""
import os
import keras
import cv2
import pandas as pd
import numpy as np
from scipy.misc import imresize
from keras import backend as K
from keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom

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
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数

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
    print(img.shape)

    y_c = class_output
    #cost = 全部のラベルの値。cost*label_indexでy_cになる
    conv_output = input_model.get_layer(layer_name).output
    #conv_output = target_conv_layer, mixed10の出力1,5,5,2048
    grads = K.gradients(y_c, conv_output)[0]
    #grads = normalize(grads)

    first = K.exp(y_c)*grads
    second = K.exp(y_c)*grads*grads
    third = K.exp(y_c)*grads*grads*grads

    gradient_function = K.function([input_model.input], [y_c,first,second,third, conv_output, grads])
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

def branch_multi_grad_cam(model, out_grad_cam_dir, input_img_name, x, y_true, layer_name, img_rows, img_cols,
                          pred_threshold=0.5, grad_threshold=-1.0, is_gradcam_plus=False):
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
       pred_threshold: 予測スコアのposi/nega分ける閾値。バイナリ分類なので、デフォルトは0.5とする
       grad_threshold: GradCam実行するか決める予測スコアの閾値。デフォルトは-1.0として必ず実行
       is_gradcam_plus: gradcam++で実行するか。Falseだと普通のgradcam実行
    Returns:
       最後のtaskのGrad-cam画像1枚（内部処理で全taskのGradCam画像を出力してる）
    """
    # 推論実行するための前処理（画像を読み込んで4次元テンソルへ変換+前処理）
    X = preprocess_x(x)
    # 推論 予測スコア算出
    pred_score = model.predict(X)

    # 5タスクならpred_score = [array([[0.046]]), array([[0.04977]]), array([[0.4]]), array([[0.96]]), array([[0.085]])] のようなスコア出る
    # Grad-Camで勾配計算するところで、特定のタスクの出力 model.output[:, task_idx] が必要
    # マルチタスクだから各タスクのcamを計算すべき
    #（シングルタスクの時はスコア最大のクラス（class_output = model.output[:, np.argmax(pred_score[0])]）を選んでいる）
    grad_cam_img = None
    for task_idx in range(len(model.output)):
        # taskのscore
        pred_score_task = pred_score[task_idx][0][0]
        # Tox21はバイナリ分類なので、確信度が0.5より大きい推論を1、それ以外を0に置換する
        y_pred = (pred_score_task > pred_threshold) * 1.0
        # スコアの桁省略
        pred_score_task_form = "{0:.2f}".format(pred_score_task)

        # taskのscoreが閾値を超えていたらGradCam実行する
        if float(pred_score_task) < float(grad_threshold):
            continue

        # task_id を出力パスに含める
        task_out_grad_cam_dir = os.path.join(out_grad_cam_dir, 'task'+str(task_idx))
        # branchのmultitaskだと、model.outputがリスト
        task_output = model.output[task_idx]
        # binaryなのでクラスは1つ
        class_idx = 0
        class_output = task_output[:, class_idx]
        # Grad-Cam実行
        if is_gradcam_plus == True:
            jetcam = grad_cam_plus(model, X, x, layer_name, img_rows, img_cols, task_output)
        else:
            jetcam = grad_cam(model, X, x, layer_name, img_rows, img_cols, class_output)
        grad_cam_img = image.array_to_img(jetcam)

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

def nobranch_multi_grad_cam(model, out_grad_cam_dir, input_img_name, x, y_true, layer_name, img_rows, img_cols,
                            pred_threshold=0.5, grad_threshold=-1.0, is_gradcam_plus=False):
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
       pred_threshold: 予測スコアのposi/nega分ける閾値。バイナリ分類なので、デフォルトは0.5とする
       grad_threshold: GradCam実行するか決める予測スコアの閾値。デフォルトは-1.0として必ず実行
       is_gradcam_plus: gradcam++で実行するか。Falseだと普通のgradcam実行
    Returns:
       最後のtaskのGrad-cam画像1枚（内部処理で全taskのGradCam画像を出力してる）
    """
    # 推論実行するための前処理（画像を読み込んで4次元テンソルへ変換+前処理）
    X = preprocess_x(x)
    # 推論 予測スコア算出
    pred_score = model.predict(X)

    # 5タスクならpred_score = [[ 0.046  0.04977  0.4  0.96  0.085]] のようなスコア出る
    # Grad-Camで勾配計算するところで、特定のタスクの出力 model.output[:, task_idx] が必要
    # マルチタスクだから各タスクのcamを計算すべき
    #（シングルタスクの時はスコア最大のクラス（class_output = model.output[:, np.argmax(pred_score[0])]）を選んでいる）
    grad_cam_img = None
    for task_idx in range(pred_score.shape[1]):
        # taskのscore
        pred_score_task = pred_score[0,task_idx]
        # Tox21はバイナリ分類なので、確信度が0.5より大きい推論を1、それ以外を0に置換する
        y_pred = (pred_score_task > pred_threshold) * 1.0
        # スコアの桁省略
        pred_score_task_form = "{0:.2f}".format(pred_score_task)

        # taskのscoreが閾値を超えていたらGradCam実行する
        if float(pred_score_task) < float(grad_threshold):
            #print(f'{pred_score_task} < {grad_threshold}')
            continue

        # task_id を出力パスに含める
        task_out_grad_cam_dir = os.path.join(out_grad_cam_dir, 'task'+str(task_idx))
        # Grad-Cam実行
        task_output = model.output[:, task_idx]
        if is_gradcam_plus == True:
            jetcam = grad_cam_plus(model, X, x, layer_name, img_rows, img_cols, task_output)
        else:
            jetcam = grad_cam(model, X, x, layer_name, img_rows, img_cols, task_output)
        grad_cam_img = image.array_to_img(jetcam)

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

if __name__ == "__main__":
    print('grad_cam.py: loaded as script file')
else:
    print('grad_cam.py: loaded as module file')