# -*- coding: utf-8 -*-
"""
Grad-CAM++
(ピクセル平均を出していたGrad-CAMの平均について２階微分、３階微分で重み付けをした、いわばGrad-CAMの一般化版)
https://qiita.com/Dason08/items/a8013b3fa4d303f5c41c を参考に作成

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
import os, sys
import numpy as np
import cv2
import argparse
import keras
import sys
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array, load_img

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

def grad_cam_plus_plus(model, preprocessed_input, x, layer_name, img_rows, img_cols, class_idx):
    '''
    Grad-CAM++
    Args:
       model: モデルオブジェクト
       preprocessed_input: 4次元テンソル(1, img_rows, img_cols, 3)へ変換+前処理後画像データ
       x: 3次元numpy.array型の画像データ（*1./255.前）
       layer_name: Poolingの直前の層の名前
       img_rows, img_cols: モデルの入力層のサイズ
       class_idx: 勾配計算したいクラスid
    Returns:
       jetcam: 影響の大きい箇所を色付けした画像(array)
    '''
    # 勾配を取得
    # grad-cam++では、活性化関数はReLU関数を使っている層にしないとだめらしい。
    # VGG16,ResNet,InceptionV3などの学習済みモデルはReLUしか使ってなかったはず
    # なのでこの辺の学習済みモデルは大丈夫そうだが、自作モデルでReLUじゃない活性化関数使っている場合多分微分がうまくいかない
    class_output = model.layers[-1].output
    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す

    ## Grad-CAM++の肝のGrad-CAMの平均についての２階微分、３階微分
    #first_derivative：１階微分
    first_derivative = K.exp(class_output)[0][class_idx] * grads
    #second_derivative：２階微分
    second_derivative = K.exp(class_output)[0][class_idx] * grads * grads
    #third_derivative：３階微分
    third_derivative = K.exp(class_output)[0][class_idx] * grads * grads * grads
    # 関数の定義
    # model.inputを入力すると、conv_outputとgradsを出力する関数
    gradient_function = K.function([model.input], [conv_output, first_derivative, second_derivative, third_derivative]) # model.inputを入力すると、conv_outputとgradsを出力する関数
    conv_output, conv_first_grad, conv_second_grad, conv_third_grad = gradient_function([preprocessed_input])
    conv_output, conv_first_grad, conv_second_grad, conv_third_grad = conv_output[0], conv_first_grad[0], conv_second_grad[0], conv_third_grad[0]

    ## Grad-CAM++の式の計算
    # α算出
    global_sum = np.sum(conv_output.reshape((-1, conv_first_grad.shape[2])), axis=0)
    alpha_num = conv_second_grad
    alpha_denom = conv_second_grad * 2.0 + conv_third_grad * global_sum.reshape(( 1, 1, conv_first_grad.shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num / alpha_denom
    # αを0-1に正規化
    alpha_normalization_constant = np.sum(np.sum(alphas, axis = 0), axis = 0)
    alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))
    alphas /= alpha_normalization_constant_processed.reshape((1,1,conv_first_grad.shape[2]))
    # wの計算
    weights = np.maximum(conv_first_grad, 0.0)
    deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad.shape[2])),axis = 0)
    # L計算+0-1に正規化
    grad_CAM_map = np.sum(deep_linearization_weights * conv_output, axis=2)
    grad_CAM_map = np.maximum(grad_CAM_map, 0) # camから0より大きい要素はそのままそれ以外は0にする。np.maximum:要素ごと比較して、大きい方を格納した配列を返す
    grad_CAM_map = grad_CAM_map / np.max(grad_CAM_map) # ピクセルの値を0-1に正規化

    # 画像化してヒートマップにして合成
    # ヒートマップでない問題は畳み込み層が小さいせいではなく、着目してるところが無いため（camの値が全てマイナス。注目してるところはプラスの値で出る）
    grad_CAM_map = cv2.resize(grad_CAM_map, (img_rows, img_cols), cv2.INTER_LINEAR)# 線形補完（cv2.INTER_LINEAR）でモデルの入力層のサイズにcamを引き延ばす
    jetcam = cv2.applyColorMap(np.uint8(255 * grad_CAM_map), cv2.COLORMAP_JET) # camのピクセルの値を0-255に戻す（モノクロ画像に疑似的に色をつける）
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)# 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2) # もとの画像に合成 ここのｘはピクセルの値を0-255の3次元のもと画像データを使うこと！！！（4次元テンソルではエラーになる）

    return jetcam

def nobranch_grad_cam(model, out_grad_cam_dir, input_img_name, x, layer_name, img_rows, img_cols):
    """
    出力層のニューラルネットワークに分岐がない場合で
    Args:
       model: モデルオブジェクト
       out_grad_cam_dir: GradCam画像出力先ディレクトリ
       input_img_name: 入力画像名（出力画像名に使う）
       x: 3次元numpy.array型の画像データ（*1./255.前）
       layer_name: Poolingの直前の層の名前
       img_rows, img_cols: モデルの入力層のサイズ
    Returns:
       最大スコアのGrad-cam画像1枚
    """
    # 推論実行するための前処理（画像を読み込んで4次元テンソルへ変換+前処理）
    X = preprocess_x(x)
    # 推論 予測スコア算出
    pred_score = model.predict(X)
    #print("pred_score =", pred_score)
    class_idx = np.argmax(pred_score[0])
    # Grad-CAM++
    jetcam = grad_cam_plus_plus(model, X, x, layer_name, img_rows, img_cols, class_idx)
    grad_cam_img = image.array_to_img(jetcam)

    # classのscore
    pred_score_class = "{0:.2f}".format(pred_score[0,class_idx])

    # Grad-cam++画像保存
    out_jpg = input_img_name+'_class='+str(class_idx)+'_score='+str(pred_score_class)+'.jpg'
    grad_cam_img.save(os.path.join(out_grad_cam_dir, out_jpg), 'JPEG', quality=100, optimize=True)
    print('out_jpg:', out_jpg)

    # Grad-cam++画像表示
    #plt.imshow(grad_cam_img)
    #plt.show()
    #plt.clf() # plotの設定クリアにする
    return grad_cam_img


if __name__ == "__main__":
    print('grad_cam_plus_plus.py: loaded as script file')
else:
    print('grad_cam_plus_plus.py: loaded as module file')
