# -*- coding: utf-8 -*-
"""
tfapiの予測結果から分類する
- cudaのエラーでtfapiでkerasのモデル同時に予測できなかったので（おそらくtensorflowのバージョンがtfapiとkerasで違うため）
"""
import os, sys
import keras
from keras.preprocessing import image
import glob, time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import pathlib
import cv2
import argparse

# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得
# 自作モジュールimport
sys.path.append( str(current_dir) + '/../' )
from transformer import get_train_valid_test, my_generator, ndimage
from predicter import base_predict

def predict_class_model(dst, model, img_height, img_width, TTA=None, TTA_rotate_deg=0, TTA_crop_num=0, *args, **kwargs):
    """
    学習済み分類モデルで予測
    引数：
        dst:切り出したarray型の画像
        model:学習済み分類モデル
        img_height, img_width:モデルの入力画像サイズ（modelのデフォルトのサイズである必要あり）
        TTA*:TTA option
        *args, **kwargs:TTA用に引数任意に追加できるようにする。*argsはオプション名なしでタプル型で引数渡せる。**kwargsは辞書型で引数渡せる
    返り値：
        pred[pred_max_id]:確信度
        pred_max_id:予測ラベル
    """
    # 画像のサイズ変更
    #x = cv2.resize(dst,(img_height,img_width))
    x = ndimage.resize(dst, img_height, img_width)
    # 4次元テンソルへ変換
    x = np.expand_dims(x, axis=0)
    # 前処理
    X = x/255.0
    # 予測1画像だけ（複数したい場合は[0]をとる）
    if (TTA == "None") and (TTA_rotate_deg == 0) and (TTA_crop_num == 0):
        pred = model.predict(X)[0]
    else:
        # TTA
        #print(f'\nTTA={TTA}, TTA_rotate_deg={TTA_rotate_deg}, TTA_crop_num={TTA_crop_num}')
        pred = base_predict.predict_tta(model, x, TTA=TTA#'flip'
                                        , TTA_rotate_deg=TTA_rotate_deg
                                        , TTA_crop_num=TTA_crop_num, TTA_crop_size=[(img_height*3)//4, (img_width*3)//4]
                                        , preprocess=1.0/255.0, resize_size=[img_height, img_width])
    #print(pred)
    # 予測確率最大のクラスidを取得
    pred_max_id = np.argmax(pred)#,axis=1)
    return pred[pred_max_id], pred_max_id

def out_pred_df_mid_posi(df_output_dict_merge, output_csv):
    """
    kaggleのくずし字コンペ用
    予測したx,yの位置情報をx1,x2,y1,y2の中心位置にしたデータフレーム出力
    """
    image_id = []
    labels = []
    #for index, series in tqdm(df_output_dict_merge.iterrows()): # df を1行づつループ回す
    for index, series in df_output_dict_merge.iterrows(): # df を1行づつループ回す
        y_mid = (series['ymin'] + series['ymax'])//2
        x_mid = (series['xmin'] + series['xmax'])//2
        labels.append( str(series['class_name'])+" "+str(x_mid)+" "+str(y_mid) )
        image_id.append( str(pathlib.Path(series['image_path']).stem) )
    df = pd.DataFrame({'image_id':image_id, 'labels':labels})

    # image_id一意にする
    image_id = sorted(set(image_id), key=image_id.index)
    #print(image_id)

    # 画像id, ラベル1 x_mid1 y mid1 ラベル2 x_mid2 y mid2 … の形式に変形する
    labels_join = []
    #for id in tqdm(image_id):
    for id in image_id:
        labels_list = list(df[df['image_id'] == id]['labels'])
        labels_join_1img = ' '.join(labels_list)
        labels_join.append(labels_join_1img)
    df = pd.DataFrame({'image_id':image_id, 'labels':labels_join})
    #display(df)
    df.to_csv(output_csv, index=False)

def predict_cut_img_from_tfapi_pred_df_one(df
                                           , class_model
                                           , dict_class
                                           , output_dir
                                           , model_height=32, model_width=32
                                           , class_min_score_thresh=0.1
                                           , is_save_cut_img=False
                                           , TTA=None, TTA_rotate_deg=0, TTA_crop_num=0
                                          ):
    """
    1画像についてのtfapiの予測結果のデータフレームから画像の予測領域の切り出してpredict
    Args:
        df:1画像についてのtfapiの予測結果のデータフレーム
        class_model:予測するkerasのモデルオブジェクト
        dict_class:kerasのモデルのクラスidとクラス名の辞書
        output_dir:予測結果の出力先ディレクトリ
        model_height, model_width:kerasのモデルの入力サイズ
        class_min_score_thresh:スコアの足切り値。予測スコアがこの値未満なら予測無しとする
        is_save_cut_img:切り出し画像保存するか
        TTA*:TTA option
    Return:
        予測結果のデータフレーム
    """
    # 画像単位にディレクトリ切ってファイル出力
    os.makedirs(output_dir, exist_ok=True)

    # 画像ロード
    img_RGB = ndimage.load(df.image_path.values[0])
    #print(img_RGB.shape)

    image_paths = []
    boxes_ymin = []
    boxes_ymax = []
    boxes_xmin = []
    boxes_xmax = []
    class_names = []
    scores = []
    # 1画像についての検出領域のデータフレーム1行ずつなめる
    for i, series in df.iterrows():
        ymin = series.ymin
        xmin = series.xmin
        ymax = series.ymax
        xmax = series.xmax

        # 画像切り出し
        dst = img_RGB[int(ymin):int(ymax), int(xmin):int(xmax)] # スライス[:]はint型でないとエラー

        # 切り出し画像を分類モデルでpredict
        class_conf, class_label_id = predict_class_model(dst, class_model, model_height, model_width, TTA=TTA, TTA_rotate_deg=TTA_rotate_deg, TTA_crop_num=TTA_crop_num)

        # 切り出し画像保存するか
        if is_save_cut_img == True:
            #plt.imshow(dst)
            #plt.show()
            # ファイル名は ymin_xmin_ymax_xmax_scores_class_names とする
            f_name = str(ymin)+'_'+str(xmin)+'_'+str(ymax)+'_'+str(xmax)+'_'+str(class_conf)+'_'+str(dict_class[class_label_id])+'.jpg'
            Image.fromarray(dst).save(os.path.join(output_dir, f_name)) # ファイル出力

        # 分類モデルの閾値以上の予測結果のみ採用する
        if class_conf >= class_min_score_thresh:
            image_paths.append(series.image_path)# str(pathlib.Path(series.image_path).stem)
            boxes_ymin.append(int(ymin))
            boxes_ymax.append(int(ymax))
            boxes_xmin.append(int(xmin))
            boxes_xmax.append(int(xmax))
            class_names.append(str(dict_class[class_label_id]))
            scores.append(float(class_conf))

    # 1画像の予測結果をデータフレームに詰め直す
    df_pred_result = pd.DataFrame({'image_path':image_paths
                                   , 'ymin':boxes_ymin
                                   , 'xmin':boxes_xmin
                                   , 'ymax':boxes_ymax
                                   , 'xmax':boxes_xmax
                                   , 'class_scores':scores
                                   , 'class_name':class_names
                                  })

    # 予測結果の出力先csv出力
    output_csv = os.path.join(output_dir, str(pathlib.Path(series.image_path).stem) + '.csv')
    out_pred_df_mid_posi(df_pred_result, output_csv)

    return df_pred_result

def predict_cut_img_from_tfapi_pred_df_all(tf_pred_df
                                           , class_model
                                           , dict_class
                                           , output_dir
                                           , model_height=32, model_width=32
                                           , class_min_score_thresh=0.1
                                           , is_save_cut_img=False
                                           , TTA=None, TTA_rotate_deg=0, TTA_crop_num=0):
    """
    tfapiの予測結果のデータフレームから画像の予測領域の切り出してpredict
    Args:
        tf_pred_df:tfapiの予測結果のデータフレーム
        class_model:予測するkerasのモデルオブジェクト
        dict_class:kerasのモデルのクラスidとクラス名の辞書
        output_dir:予測結果の出力先ディレクトリ
        model_height, model_width:kerasのモデルの入力サイズ
        class_min_score_thresh:スコアの足切り値。予測スコアがこの値未満なら予測無しとする
        is_save_cut_img:切り出し画像保存するか
        TTA*:TTA option
    Return:
        予測結果のデータフレーム
    """
    # 出力ディレクトリ
    os.makedirs(output_dir, exist_ok=True)

    # 一意の画像パス取得
    unique_img_paths = tf_pred_df.image_path.unique()

    df_pred_result_all = None
    #for p in tqdm(unique_img_paths[:3]): # 3件テスト用
    for p in tqdm(unique_img_paths):

        # 1画像についての検出領域のデータフレーム
        df = tf_pred_df[tf_pred_df.image_path == p]
        #print(df.shape)

        # 画像名
        img_name = str(pathlib.Path(df.image_path.values[0]).stem)
        #print(img_name)

        # 途中で処理落ちても、途中までの結果をファイル出力する用
        out_one_img_dir = os.path.join(output_dir, img_name)
        os.makedirs(out_one_img_dir, exist_ok=True)
        #print(out_one_img_dir)

        # 1画像についてのtfapiの予測結果のデータフレームから画像の予測領域の切り出してpredict
        df_pred_result = predict_cut_img_from_tfapi_pred_df_one(df
                                                                , class_model
                                                                , dict_class
                                                                , output_dir=out_one_img_dir
                                                                , model_height=model_height, model_width=model_width
                                                                , class_min_score_thresh=class_min_score_thresh
                                                                , is_save_cut_img=is_save_cut_img
                                                                , TTA=TTA, TTA_rotate_deg=TTA_rotate_deg, TTA_crop_num=TTA_crop_num)
        if df_pred_result is None:
            df_pred_result_all = df_pred_result
        else:
            df_pred_result_all = pd.concat([df_pred_result_all, df_pred_result], ignore_index=True)

    # 予測結果の出力先csv出力
    df_pred_result_all.to_csv(os.path.join(output_dir, 'df_pred_result_all.csv'), index=False)

    # kaggleのくずし字コンペ用
    out_pred_df_mid_posi(df_pred_result_all, os.path.join(output_dir, 'kuzushiji_submission.csv'))

    return df_pred_result_all


def kuzushiji_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=str, required=True,#, default=r'D:\work\kaggle_kuzushiji-recognition\work\object_detection\20190812\tfapi_result_predict_20190930_pseudo_label_label_smooth',
        help="path to output directory")
    ap.add_argument("--tf_pred_csv", type=str, required=True,#, default=r'D:\work\kaggle_kuzushiji-recognition\work\object_detection\20190812\predict_keras_all_20190909_model\output_dict.tsv',
        help="tfapiの予測結果tsvのpath")
    ap.add_argument("--class_model", type=str, required=True,#, default=r'D:\work\kaggle_kuzushiji-recognition\work\classes\20190930_pseudo_label_label_smooth\best_val_loss.h5',
        help="予測するkerasのモデルのpath")
    ap.add_argument("--tfAPI_dict_class", type=str, required=True,#, default=r'D:\work\kaggle_kuzushiji-recognition\work\classes\20190930_pseudo_label_label_smooth\tfAPI_dict_class.tsv',
        help="kerasのモデルのクラスidとクラス名のpath")
    ap.add_argument("--TTA_flip", type=str, default="None",
        help="kerasの分類モデルのTTA flip")
    ap.add_argument("--TTA_rotate_deg", type=int, default=0,
        help="kerasの分類モデルのTTA rotate deg")
    ap.add_argument("--TTA_crop_num", type=int, default=0,
        help="kerasの分類モデルのTTA crop num")
    ap.add_argument("--is_save_cut_img", action='store_const', const=True, default=False,
        help="切り出し画像保存するか")
    args = vars(ap.parse_args())

    # tfapiの予測結果ロード
    tf_pred_df = pd.read_csv(args['tf_pred_csv'], sep='\t')
    #print(tf_pred_df.head())

    # 予測するkerasのモデルロード
    sys.path.append( r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py\Git\OctConv-TFKeras' )
    from oct_conv2d import OctConv2D
    class_model = keras.models.load_model(args['class_model']
                                          , custom_objects={'OctConv2D': OctConv2D} # OctConvは独自レイヤーだからcustom_objects の指定必要
                                          , compile=False)

    # kerasのモデルのクラスidとクラス名取得
    df_dict_class = pd.read_csv(args['tfAPI_dict_class'], sep='\t')
    #print(df_dict_class.head())
    dict_class = df_dict_class.set_index('classes_ids').to_dict()['unicode']
    #print(dict_class)

    # 予測
    df_pred_result = predict_cut_img_from_tfapi_pred_df_all(tf_pred_df
                                                            , class_model
                                                            , dict_class
                                                            , output_dir=args['output']
                                                            , is_save_cut_img=args['is_save_cut_img']
                                                            , TTA=args["TTA_flip"], TTA_rotate_deg=args["TTA_rotate_deg"], TTA_crop_num=args["TTA_crop_num"]
                                                            )

if __name__ == '__main__':
    kuzushiji_main()
else:
    print('tfapi_result_predict.py: loaded as module file')
