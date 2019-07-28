# -*- coding: utf-8 -*-

# iPhoneの画像ディレクトリpredict用コマンドラインスクリプト
# $ cd C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py\predicter
# $ activate tfgpu_py36_v3
# (tfgpu_py36_v3)$ python output_pred_dir_cp.py -o D:\work\keras_iPhone_pictures\01_classes_results_tfgpu_py36\20190531\train_all\prediction_cp_img\2019-06 -i D:\iPhone_pictures\2019-06

import os, sys
import argparse
import matplotlib
matplotlib.use('Agg')
import keras
import shutil
from tqdm import tqdm

# 自作モジュールimport
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
from predicter import base_predict
from model import define_model
from dataset import util

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--pred_output_dir", type=str, required=True,
                    help="output dir path")
    ap.add_argument("-i", "--img_data_dir", type=str, required=True,
                    help="img dir path (ex. D:\iPhone_pictures\2019-06")
    ap.add_argument("-m", "--model_dir", type=str, default='default', nargs='*', # nargs='*'を指定すると可変長で複数個をリスト形式で受け取ることができます
                    help="model files path")
    ap.add_argument("-m_d", "--merge_dir", type=str, default=r'D:\work\keras_iPhone_pictures\01_classes_results_tfgpu_py36\20190531\train_all\prediction_cp_img\merge',
                    help="merge dir")
    args = vars(ap.parse_args())

    # modelファイルのしていなければこの2モデル使う
    if args['model_dir'] == 'default':
        model_InceptionResNetV2_path = r'D:\work\keras_iPhone_pictures\01_classes_results_tfgpu_py36\20190529\train_all\finetuning.h5'
        model_InceptionResNetV2 = keras.models.load_model(model_InceptionResNetV2_path, compile=False)

        # EfficientNetはネットワークロードできないので重みからロード
        output_dir = r'D:\work\keras_iPhone_pictures\01_classes_results_tfgpu_py36\20190531\train_all'
        model_EfficientNetB3_path = os.path.join(output_dir, 'finetuning.h5')
        model_EfficientNetB3, orig_model = define_model.get_fine_tuning_model(
            output_dir
            , 331, 331, 3
            , 11
            , 'EfficientNet', 190
            , FCnum=2
            , activation='softmax'
            , add_se=False
            , efficientnet_num=3
        )
        model_EfficientNetB3.load_weights(model_EfficientNetB3_path)

        models = [model_InceptionResNetV2, model_EfficientNetB3]

    # 分類クラスごとにフォルダ切ってコピーする
    os.makedirs(args['pred_output_dir'], exist_ok=True)
    base_predict.pred_dir_cp(data_dir = args['img_data_dir'] # 入力ディレクトリ
                            , pred_output_dir = args['pred_output_dir'] # 出力先ディレクトリ
                            , model = models)
    # mergeディレクトリに全コピーする
    if args['merge_dir'] is not None:
        img_data_dir_paths = util.get_jpg_png_path_in_dir(args['pred_output_dir'])
        for p in tqdm(img_data_dir_paths):
            out_dir = os.path.join(args['merge_dir'], pathlib.Path(p).parent.name)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copyfile(p, os.path.join(out_dir, str(pathlib.Path(p).parent.parent.name +'_'+ pathlib.Path(p).name)))


if __name__ == '__main__':
    main()
