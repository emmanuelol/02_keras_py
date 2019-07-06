# -*- coding: utf-8 -*-
"""
Grad-CAM
https://qiita.com/haru1977/items/45269d790a0ad62604b3 を参考に作成

学習済みモデルのディレクトリ（モデルのファイル名は architecture.json, finetuning.h5 としてる）、
全結合層の直前のCNN層の名前（Grad-CAMではConvolution最終層の勾配を取るため）、
モデルの入力層のサイズ、
入力画像、
Grad-CAM 実行画像格納先
をコマンドライン引数で指定する

クラス変更する場合はソース内のclassesを変更すること！！！！

Usage: CUDA_VISIBLE_DEVICES=0 python grad_cam.py [model_dir] [layer_name] [img_rows] [img_cols] [input_img] [output_dir]
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2
from keras import backend as K
from keras.preprocessing import image
#from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model, model_from_json

##### 引数で渡すように変更するの大変なので、良くないけどソース内でクラス名を記憶しとく #####
# クラス変更するときは下記の2変数を変更すること
# 分類クラス
classes = ['negative', 'positive']
# positive のTP/FN/FP/TN を判定
check_class = 'positive'
#####################################################################################

# 引数チェック
args = sys.argv
argc = len(args)
print(argc)
if argc != 7:
    print('Usage: python %s [model_dir] [layer_name] [img_rows] [img_cols] [input_img] [output_dir]' % args[0])
    sys.exit('Commandline arguments error')

# 学習済みモデルのディレクトリ
model_dir = args[1]
print(model_dir)
# Poolingの直前の層の名前
layer_name = args[2]
print(layer_name)
# モデルの入力層のサイズ
img_rows = int(args[3])
img_cols = int(args[4])
# 入力画像
input_img = args[5]
# 再帰的にファイル・ディレクトリを探して出力する関数
def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)
input_img_list = []
for file in find_all_files(input_img):
    if '.jpg' in file:
        input_img_list.append(file)
# 出力画像格納先
output_dir = args[6]

# https://qiita.com/supersaiakujin/items/568605f999ef5cc741be より
# finetuning したモデルはDropout やBatchNorm のようにTraining とTest でネットワーク構造が変わる操作がある
# ここのGrad_Camではそのようなモデルの勾配の取得を行うためset_learning_phase()の指定が無いとエラーになる
# また、Test時のset_learning_phase(0)でないと推測の結果はおかしくなる
K.set_learning_phase(0) #Test時には０にセット
def Grad_Cam(model, x, layer_name):
    '''
    Args:
       model: モデルオブジェクト
       x: 画像(array)
       layer_name: 畳み込み層の名前
    Returns:
       jetcam: 影響の大きい箇所を色付けした画像(array)
    '''
    # 推論実行するための前処理（画像を読み込んで4次元テンソルへ変換）
    preprocessed_input = load_array(x)
    # 予測クラスの算出
    predictions = model.predict(preprocessed_input)
    print("predictions =", predictions)
    class_idx = np.argmax(predictions[0])
    print("max_class_idx =", class_idx)
    class_output = model.output[:, class_idx]
    #print(class_output)

    # 予測したクラス名確認
    pred_class = classes[class_idx]
    print("pred_class =", pred_class)

    # 勾配を取得
    conv_output = model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # 重みを平均化して、レイヤーのアウトプットに乗じる
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像化してヒートマップにして合成
    cam = cv2.resize(cam, (img_rows, img_cols), cv2.INTER_LINEAR) # モデルの入力層のサイズ指定
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成 ここは3次元のxのデータを使うこと！！！（4次元テンソルではエラーになる）

    return jetcam, pred_class

def output_img(input_img, img_rows, img_cols, model):
    """grad_cam画像出力"""
    # 入力画像ロード
    x = image.img_to_array(image.load_img(input_img, target_size=(img_rows, img_cols)))
    print(x)

    # Grad_Cam実行
    img, pred_class = Grad_Cam(model, x, layer_name)
    Grad_Cam_img = image.array_to_img(img)
    input_img_base = os.path.basename(input_img)
    # GradCAMの結果画像のファイル名は predict_予測クラス_正解クラス_.jpg とする
    out_img_name = "predict_"+pred_class+"_"+input_img_base
    Grad_Cam_img.save(os.path.join(output_dir, out_img_name), 'JPEG', quality=100, optimize=True)

def get_stat_err(cla, correct_class, pred_class):
    """
    TP/FN/FP/TN を判定した結果を返す
    """
    if (cla == correct_class) and (cla == pred_class):
        stat_err = 'TP'
    elif(cla != correct_class) and (cla == pred_class):
        stat_err = 'FP'
    elif(cla == correct_class) and (cla != pred_class):
        stat_err = 'FN'
    elif(cla != correct_class) and (cla != pred_class):
        stat_err = 'TN'
    return stat_err

def load_array(x):
    """array型の（テスト）画像を読み込んで4次元テンソルへ変換"""
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32')
    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！ これを忘れると結果がおかしくなるので注意
    x = x/255.0
    return x

def output_img_Classification(input_img, img_rows, img_cols, model):
    """grad_cam画像をTP/FN/FP/TN を判定して、TP/FN/FP/TN でわけたディレクトリに出力"""
    # 入力画像ロード(array型に変換)
    x = image.img_to_array(image.load_img(input_img, target_size=(img_rows, img_cols)))
    # Grad_Cam実行
    img, pred_class = Grad_Cam(model, x, layer_name)
    #print("img.shape =",img.shape)
    Grad_Cam_img = image.array_to_img(img)
    input_img_base = os.path.basename(input_img)
    out_img_name = "predict_"+pred_class+"_"+input_img_base
    print("out_img_name =", out_img_name)

    # 入力画像のパスから正解ラベル取得
    correct_class = ''
    for cla in classes:
        if cla in input_img:
            correct_class = cla
    print("input_img =", input_img)
    print("correct_class =", correct_class)
    # TP/FN/FP/TN を判定
    stat_err = get_stat_err(check_class, correct_class, pred_class)
    print(stat_err+"\n")

    Grad_Cam_img.save(os.path.join(output_dir, stat_err, out_img_name), 'JPEG', quality=100, optimize=True)

def main():
    # モデルのグラフロード
    model = model_from_json(open(os.path.join(model_dir, 'architecture.json')).read())
    # モデルの重みをロード
    model.load_weights(os.path.join(model_dir, 'finetuning.h5'))

    # 入力画像は複数か
    if len(input_img_list) > 0:
        for img in input_img_list:
            #output_img(img, img_rows, img_cols, model)
            output_img_Classification(img, img_rows, img_cols, model)
    else:
        #output_img(input_img, img_rows, img_cols, model)
        output_img_Classification(input_img, img_rows, img_cols, model)

    # TypeError: 'NoneType' object is not callable を避けるために処理終了時に下記をコール
    # メモリ解放的な操作
    # http://infotech.hateblo.jp/entry/2017/09/12/021951 より
    K.clear_session()

if __name__ == "__main__":
    main()
