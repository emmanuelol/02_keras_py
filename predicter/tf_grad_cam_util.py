# -*- coding: utf-8 -*-
"""
Grad-CAM に関連したutil関数
"""
import os, sys, glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
from predicter import grad_cam, base_predict
from dataset import util

def gradcam_from_img_path(load_model, pred_path, output_dir, classes, img_rows, img_cols
                        , layer_name='multiply_1', show_img=True, is_gradcam_plus=False):
    """
    画像1件のファイルパスからgradcam実行+ファイル出力
    Arges:
        load_model: modelオブジェクト
        pred_path: 予測する画像ファイルパス
        output_dir: gradcam画像保存先ディレクトリのパス
        classes: クラス名リスト
        img_rows, img_cols: 画像サイズ
        layer_name: gradcam掛ける層の名前
        show_img: 元画像とgradcam画像及びprint文を表示させるか。Trueならnotebook上に画像とprint文を出す
        is_gradcam_plus: gradcam++で実行するか。Falseだと普通のgradcam実行
    Return:
        Grad-cam画像
    """
    # grad_cam掛けるクラスid取得
    pred_id, pred_score = base_predict.pred_from_1img(load_model, pred_path, img_rows, img_cols, classes=classes, show_img=show_img)
    #print('pred_id:', str(pred_id))
    pred_label = classes[pred_id]
    #print('pred_label', pred_label)

    # 入力画像ロード(image.load_img)してをarray型に変換(image.img_to_array)
    x = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(pred_path, target_size=(img_rows, img_cols)))
    X = grad_cam.preprocess_x(x)

    # grad_cam
    #class_output = load_model.output[:, pred_id]
    class_output = load_model.output[0, pred_id] # こうしないとgradcam++が動かない
    if is_gradcam_plus == True:
        jetcam = grad_cam.grad_cam_plus(load_model, X, x, layer_name, img_rows, img_cols, class_output)
    else:
        jetcam = grad_cam.grad_cam(load_model, X, x, layer_name, img_rows, img_cols, class_output)
    grad_cam_img = keras.preprocessing.image.array_to_img(jetcam)

    # Grad-cam画像表示
    if show_img == True:
        plt.imshow(grad_cam_img)
        plt.show()

    # TP/FN/FP/TN/NAN を判定し、判定結果を出力パスに含める
    y_true_label = os.path.basename(os.path.dirname(pred_path)) # 正解ラベルであるファイルの直上のフォルダ名のみを取得
    #print(y_true_label)
    judge = grad_cam.judge_evaluate(pred_label, y_true_label, positive=y_true_label, negative=pred_label)
    judge_out_grad_cam_dir = os.path.join(output_dir, 'gradcam', judge)
    out_jpg = os.path.join(judge_out_grad_cam_dir, y_true_label+'_'+os.path.basename(pred_path)+'_pred_'+classes[pred_id]+"_{0:.2f}".format(pred_score)+'.jpg')
    print('out_jpg:', out_jpg)

    # ファイル出力
    os.makedirs(judge_out_grad_cam_dir, exist_ok=True)
    grad_cam_img.save(out_jpg, 'JPEG', quality=100, optimize=True)

    return grad_cam_img

def gradcam_from_x(load_model, x, output_dir, img_rows, img_cols
                    , classes=None
                    , layer_name='activation_49', show_img=True, is_gradcam_plus=False):
    """
    画像1件のxからgradcam実行+ファイル出力
    Arges:
        load_model: modelオブジェクト
        x: 予測する画像のx(前処理前)
        output_dir: gradcam画像保存先ディレクトリのパス
        img_rows, img_cols: 画像サイズ
        classes: クラス名リスト。Noneなら予測ラベル名はわからない
        layer_name: gradcam掛ける層の名前
        show_img: 元画像とgradcam画像及びprint文を表示させるか。Trueならnotebook上に画像とprint文を出す
        is_gradcam_plus: gradcam++で実行するか。Falseだと普通のgradcam実行
    Return:
        Grad-cam画像
    """
    # 前処理
    X = x/255.0
    # 元画像表示
    util.show_np_img(X)

    # grad_cam掛けるクラスid取得
    pred_id, pred_score = base_predict.pred_from_1X(load_model, X, classes=classes)
    #print('pred_id:', str(pred_id))
    if classes is not None:
        pred_label = classes[pred_id]
        #print('pred_label', pred_label)

    # grad_cam
    #class_output = load_model.output[:, pred_id]
    class_output = load_model.output[0, pred_id] # こうしないとgradcam++が動かない
    if is_gradcam_plus == True:
        jetcam = grad_cam.grad_cam_plus(load_model, X, x, layer_name, img_rows, img_cols, class_output)
    else:
        jetcam = grad_cam.grad_cam(load_model, np.expand_dims(X, axis=0), x, layer_name, img_rows, img_cols, class_output)
    grad_cam_img = keras.preprocessing.image.array_to_img(jetcam)

    # Grad-cam画像表示
    if show_img == True:
        plt.imshow(grad_cam_img)
        plt.show()

    # TP/FN/FP/TN/NAN を判定し、判定結果を出力パスに含める
    #y_true_label = os.path.basename(os.path.dirname(pred_path)) # 正解ラベルであるファイルの直上のフォルダ名のみを取得
    #print(y_true_label)
    #judge = grad_cam.judge_evaluate(pred_label, y_true_label, positive=y_true_label, negative=pred_label)
    #judge_out_grad_cam_dir = os.path.join(output_dir, 'gradcam', judge)
    #out_jpg = os.path.join(judge_out_grad_cam_dir, y_true_label+'_'+os.path.basename(pred_path)+'_pred_'+classes[pred_id]+'.jpg')
    #print(out_jpg)

    # ファイル出力
    #os.makedirs(judge_out_grad_cam_dir, exist_ok=True)
    #grad_cam_img.save(out_jpg, 'JPEG', quality=100, optimize=True)

    return grad_cam_img
