# -*- coding: utf-8 -*-
"""
分類問題の基本predict関数

Usage:
import base_predict

model = keras.models.load_model(os.path.join(out_dir, 'best_model.h5'), compile=False)
# 出力層のニューラルネットワークに分岐がある場合のpredict
y_test_list, y_pred_list = multi_predict.branch_set_predict(model, d_cls.X_test, d_cls.y_test)
"""
import os, sys, glob, shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import cv2

# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得

# imgaug は["scipy", "scikit-image>=0.11.0", "numpy>=1.15.0", "six", "imageio", "Pillow", "matplotlib","Shapely", "opencv-python"] の依存ライブラリ必要
sys.path.append( str(current_dir) + '/../Git/imgaug' )
import imgaug
# albumentations はimgaug をimport しておかないとimport できない
sys.path.append( str(current_dir) + '/../Git/albumentations' )
import albumentations
# https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, PadIfNeeded, CenterCrop, Rotate, RandomCrop)

# 自作モジュールimport
sys.path.append( str(current_dir) + '/../' )
from predicter import conf_matrix
from dataset import util
from predicter import ensemble_predict

def pred_classes_generator(model, generator, steps=None, classes_list=None):
    """
    多クラス分類のgenerator からgenerator の指定ディレクトリの画像全件予測する
    Args:
        model: modelオブジェクト
        generator: generatorオブジェクト
        steps: batch回す回数。 None ならlen(generator)をステップ数として使用
        classes_list: クラス名リスト.クラスid をクラス名に変換しない場合はNoneにする
    Return:
        データフレームでファイル名と予測ラベル返す
    """
    # predict_generator でgenerator から予測確率だせる
    # generator をreset() せずにpredict_generator 実行すると順番がグチャグチャでpredict してしまうらしい
    # https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
    generator.reset()
    pred = model.predict_generator(generator, steps=steps)#steps=(totalTest // BS) + 1)
    #print('pred :', pred)
    # 予測確率最大のクラスidを取得
    top_indices = np.argmax(pred,axis=1)
    # 予測確率top位のクラスidと予測確率を取得
    pred_id_list = []
    pred_score_list = []
    for i, top_id in enumerate(top_indices):
        pred_id_list.append(top_id)
        pred_score_list.append("{:.3}".format(pred[i,top_id]))

    if classes_list != None:
        # クラス名あればクラスid をクラス名に変換
        pred_name_list = []
        for pred_id in pred_id_list:
            pred_name_list.append(classes_list[pred_id])
    else:
        pred_name_list = pred_id_list
    # データフレームでファイル名と予測ラベル返す
    if hasattr(generator, 'filenames') == True:
        # 画像ファイルからデータ取る場合
        results = pd.DataFrame({"Filename":generator.filenames, "PredictionLabel":pred_name_list, "PredictionScore":pred_score_list})
    else:
        # 画像ファイルからデータ取らない場合
        _label = np.argmax(generator.y, axis=1)
        #print(_label)
        _label = np.array(_label, dtype=str)
        _label = [os.path.join(l, 'none') for l in _label]
        #print(_label)
        results = pd.DataFrame({"Filename":_label, "PredictionLabel":pred_name_list, "PredictionScore":pred_score_list})
    return results

def conf_matrix_from_pred_classes_generator(pred_df, classes, output_dir, is_label_print=True, figsize=(6, 4)):
    """
    pred_classes_generator() でだした値からconf_matrix.make_confusion_matrix() で混同行列作る
    Arges:
        pred_df: pred_classes_generator()の返り値である予測結果のデータフレーム
        classes: 分類クラス名リスト
        output_dir: 出力ディレクトリ
        is_label_print: ラベル名表示させるか。Trueならprint()でだす
        figsize: 混同行列のplotサイズ
    Return:
        混同行列のファイル作成
    """
    #----pred_classes_generator() でだした値から conf_matrix.make_confusion_matrix() で混同行列作るための前処理 ----#
    # 予測ラベル(id)をnumpyに変換
    y_pred_list = pred_df['PredictionLabel'].values.tolist() # リストに変換
    y_pred_list_str = [str(i) for i in y_pred_list] # リストの中身を文字型に変換
    y_pred_np = np.array(y_pred_list_str) # リストをnumpyに変換

    # ファイルパスから正解ラベル取得し、numpyに変換
    y_true_list = pred_df['Filename']
    y_true_label_list = []
    for y in y_true_list:
        y_true_label = os.path.basename(os.path.dirname(y)) # 正解ラベルであるファイルの直上のフォルダ名のみを取得
        y_true_label_list.append(y_true_label)
    y_true_label_np = np.array(y_true_label_list) # リストをnumpyに変換

    if is_label_print==True:
        print('y_pred_list_str:', y_pred_np)
        print('y_true_label_np:', y_true_label_np)
    #-----------------------------------------------------------------------------------------------------------#
    # 混同行列作成
    return conf_matrix.make_confusion_matrix(classes, y_true_label_np, y_pred_np, output_dir, figsize=figsize)


def pred_classes_evaluate_generator(model, generator, steps=None):
    """
    多クラス分類のgenerator から、evaluate_generator 関数使って、generator の指定ディレクトリ全体の正解率とlossを出す
        # evaluate_generator はflow_from_directory でラベルクラスが親ディレクトリである場合しか使えない評価関数（lossとaccを出す）
        # なので、基本train/validation setにしか使えない（test setは正解ラベルないケースが普通なので）
        # evaluate_generator はgenerator だけ引数に渡したら全件予測してくれる
        # ただし、batch size ごとに予測するから総数/batch_size が割り切れる数になっていないと、あまりの分が2回predictされてしまう
        # 安全な方法はbatch_size=1 にしたgenerator で実行（複数一気にpredictしないから時間かかる）
        # https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
    Args:
        model: modelオブジェクト(compile=True(既定値)でload_modelしたモデル)
        generator: generatorオブジェクト
        steps: batch回す回数。 None ならlen(generator)をステップ数として使用
    Return:
        loss, acuuracy
    """
    scoreSeg = model.evaluate_generator(generator, steps=steps)#steps=(totalTest // BS) + 1)
    print("loss = ",scoreSeg[0])
    print("acc = ",scoreSeg[1])
    return scoreSeg[0], scoreSeg[1]


def pred_from_1img(model, pred_path, img_rows, img_cols
                    , classes=None, show_img=True):
    """
    1件のファイルパスから予測実行し、確率最大のラベル表示し、確率最大のidとscoreを返す
    classesはクラス名のリスト。Noneなら予測ラベル名出さない
    """
    if show_img == True:
        print(pred_path)
        # ファイルパスから画像データを表示させる
        util.show_file_img(pred_path)

    # 4次元テンソルへ変換してpredict
    X = load_img(pred_path, img_rows, img_cols)
    predict = model.predict(X)
    if show_img == True:
        print('predict_score:', predict)

    # 確率最大のクラスid返す
    y = predict[0].argmax()

    # classesがあればラベル名表示
    if classes is not None and show_img==True:
        print('classes:', classes)
        print("max_pred_label: ", classes[y])

    return y, predict[0][y]

def pred_from_1X(model, X, classes=None, show_img=True):
    """
    1件の前処理済みの画像データから予測実行し、確率最大のラベル表示し、確率最大のid返す
    classesはクラス名のリスト。Noneなら予測ラベル名出さない
    """
    # 4次元テンソルへ変換してpredict
    X = np.expand_dims(X, axis=0)
    X = X.astype('float32')
    predict = model.predict(X)
    if show_img == True:
        print('predict_score:', predict)

    # 確率最大のクラスid返す
    y = predict[0].argmax()

    # classesがあればラベル名表示
    if classes is not None and show_img==True:
        print('classes:', classes)
        print("max_pred_label: ", classes[y])

    return y, predict[0][y]

def predict_tta(model, x, TTA='flip'
                , TTA_rotate_deg=0
                , TTA_crop_num=0, TTA_crop_size=[224, 224]
                , preprocess=1.0/255.0, resize_size=[331, 331]
                ):
    """
    albumentationsで水平反転,回転,切り抜き画像作成してTTA
    Usege:
        from keras.preprocessing import image
        for f_name in d_cls.test_gen.filenames[0:2]:
                path = os.path.join(d_cls.test_gen.directory, f_name)
                img = image.load_img(path, target_size=(img_rows, img_cols))
                x = image.img_to_array(img)
                pred = predict_tta(model, x, TTA_rotate_deg=120, TTA_crop_num=5)
    """
    x_list = [x]
    X_list = []
    data = {'image': x}
    # 水平反転
    if TTA == 'flip':
        augmentation = albumentations.HorizontalFlip(p=1)
        x_list.append(augmentation(**data)['image'])
    # 回転
    if TTA_rotate_deg > 0 :
        r_deg = int(TTA_rotate_deg)
        r_num = 360//r_deg
        for r_i in range(1, r_num):
            augmentation = albumentations.Rotate((r_deg*r_i, r_deg*r_i), always_apply=True, p=1, border_mode=0)
            x_list.append(augmentation(**data)['image'])
    # 切り抜き
    if TTA_crop_num > 0 :
        for r_i in range(0, TTA_crop_num):
            augmentation = albumentations.RandomCrop(int(TTA_crop_size[0]), int(TTA_crop_size[1]), always_apply=True, p=1)
            x_list.append(augmentation(**data)['image'])
    # サイズ変更+前処理
    for x in x_list:
        x = np.array(x, np.float32)
        if resize_size is not None:
            x = cv2.resize(x, (resize_size[0], resize_size[1]))
        if preprocess is None:
            X_list.append(x)
        else:
            X_list.append(preprocess * x)

    # TTA predict score avg
    pred_avg = None
    for X in X_list:
        # 4次元テンソルへ変換してpredict
        X = np.expand_dims(X, axis=0)
        X = X.astype('float32')
        if pred_avg is None:
            if isinstance(model, list) == True:
                pred_avg = ensemble_predict.ensembling_soft(model, X)[0]
            else:
                pred_avg = model.predict(X)[0]
        else:
            if isinstance(model, list) == True:
                pred_avg += ensemble_predict.ensembling_soft(model, X)[0]
            else:
                pred_avg += model.predict(X)[0]
    # TTA画像確認用
    # import matplotlib.pyplot as plt
    #for X in X_list:
    #    plt.imshow(X)
    #    plt.show()
    #print(len(X_list))
    return pred_avg/len(X_list)

def predict_tta_generator(model, generator
                            , TTA='flip'
                            , TTA_rotate_deg=0
                            , TTA_crop_num=0, TTA_crop_size=[224, 224]
                            , preprocess=None, resize_size=None
                            ):
    """
    flowやflow_from_directory 済みのImageDataGenerator からTTA を実行
    ※validation とtest のgenerator でしか実効不可
    　train のgenerator はshuffle=True されてるためgenerator.n (サンプルの総数) が無いから
    """
    count = 0
    generator.reset()
    pred_tta = []
    # 書き込み進捗バーの書き込み先をsys.stdout(標準出力)指定しないと進捗バーが改行される
    # https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
    #pbar = tqdm(generator, file=sys.stdout)
    #for step, (x_batch, y_batch) in enumerate(pbar):
    for step, (x_batch, y_batch) in enumerate(generator):
        #pbar.set_description("Processing %s" % step) # tqdmの進捗バー enumerateで回すと進捗バー改行されるためコメントアウト
        for x, y in zip(x_batch, y_batch):
            pred = predict_tta(model, x
                                , TTA=TTA
                                , TTA_rotate_deg=TTA_rotate_deg
                                , TTA_crop_num=TTA_crop_num, TTA_crop_size=TTA_crop_size
                                , preprocess=preprocess, resize_size=resize_size
                                )
            pred_tta.append(pred)
            count+=1
        #print(y_batch)
        #print('Step, Count:', step, count)
        if count >= generator.n:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            # https://keras.io/ja/preprocessing/image/#imagedatagenerator_1
            break
    return np.array(pred_tta)

def get_predict_generator_results(pred, generator, classes_list=None):
    """
    多クラス分類予測結果（スコア）のリスト とgenerator からデータフレームでファイル名と予測ラベル返す
    Args:
        pred: 多クラス分類予測結果（スコア）のリスト
        generator: generatorオブジェクト
        classes_list: クラス名リスト.クラスid をクラス名に変換しない場合はNoneにする
    Return:
        データフレームでファイル名と予測ラベル返す
    """
    generator.reset()
    # 予測確率最大のクラスidを取得
    top_indices = np.argmax(pred,axis=1)
    # 予測確率top位のクラスidと予測確率を取得
    pred_id_list = []
    pred_score_list = []
    for i, top_id in enumerate(top_indices):
        pred_id_list.append(top_id)
        pred_score_list.append("{:.3}".format(pred[i,top_id]))

    if classes_list != None:
        # クラス名あればクラスid をクラス名に変換
        pred_name_list = []
        for pred_id in pred_id_list:
            pred_name_list.append(classes_list[pred_id])
    else:
        pred_name_list = pred_id_list

    # データフレームでファイル名と予測ラベル返す
    if hasattr(generator, 'filenames') == True:
        # 画像ファイルからデータ取る場合
        results = pd.DataFrame({"Filename":generator.filenames, "PredictionLabel":pred_name_list, "PredictionScore":pred_score_list})
    else:
        # 画像ファイルからデータ取らない場合
        _label = np.argmax(generator.y, axis=1)
        #print(_label)
        _label = np.array(_label, dtype=str)
        _label = [os.path.join(l, 'none') for l in _label]
        #print(_label)
        results = pd.DataFrame({"Filename":_label, "PredictionLabel":pred_name_list, "PredictionScore":pred_score_list})
    return results


def pred_tta_from_paths(model, img_paths, img_rows, img_cols, classes=None, show_img=False
                        , TTA=''
                        , TTA_rotate_deg=0
                        , TTA_crop_num=0, TTA_crop_size=[224, 224]
                        , preprocess=1.0/255.0
                       ):
    """
    指定のファイルパス全件 predict_tta() で予測実行し、ファイルパス、確率最大のラベル名、scoreの予測結果のデータフレームを返す
    classesはクラス名のリスト。Noneなら予測ラベル名出さないで予測idを返す
    TTAつけたくない場合はTTAのオプション引数の指定を入れないこと
    """
    from keras.preprocessing import image

    df = pd.DataFrame(columns=['Filename', 'PredictionLabel', 'PredictionScore'])
    # 書き込み進捗バーの書き込み先をsys.stdout(標準出力)指定しないと進捗バーが改行される
    # https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
    pbar = tqdm(img_paths, file=sys.stdout)
    for p in pbar:
        pbar.set_description("Processing %s" % p) # tqdmの進捗バー
        if show_img == True:
            print(p)
            # ファイルパスから画像データを表示させる
            util.show_file_img(p)

        # 画像をarrayに変換
        img = image.load_img(p, target_size=(img_rows, img_cols))
        x = image.img_to_array(img)

        # TTA predict
        predict = predict_tta(model, x, TTA=TTA
                    , TTA_rotate_deg=TTA_rotate_deg
                    , TTA_crop_num=TTA_crop_num, TTA_crop_size=TTA_crop_size
                    , preprocess=preprocess, resize_size=[img_rows, img_cols]
                    )
        if show_img == True:
            print('predict_score:', predict)

        # 確率最大のクラスid返す
        y = predict.argmax()
        score = predict[y]

        # classesがあればラベル名表示
        if classes is not None and show_img==True:
            print('classes:', classes)
            print("max_pred_label: ", classes[y])

        # TTAなしのpredict
        #y, score = pred_from_1img(load_model, p, img_rows, img_cols, classes=classes, show_img=show_img)
        #print(y, score)

        # データフレームに1行ずつ追加
        # https://qiita.com/567000/items/d8a29bb7404f68d90dd4
        if classes is None:
            series = pd.Series([p, y, score], index=df.columns)
        else:
            series = pd.Series([p, classes[y], score], index=df.columns)
        df = df.append(series, ignore_index=True)
    return df


def copy_pred_img_by_pred_df(pred_img_output_dir, pred_df):
    """
    予測結果のデータフレームから、予測ラベルごとに出力先ディレクトリを分けて、予測した画像コピーする
    """
    for img_path, label, score in tqdm(zip(pred_df['Filename'], pred_df['PredictionLabel'], pred_df['PredictionScore'])):
        # 予測ラベルごとに出力先ディレクトリ
        out_class_dir = os.path.join(pred_img_output_dir, label)
        os.makedirs(out_class_dir, exist_ok=True)
        # コピーする画像ファイル名に予測ラベル名+スコアをつける
        out_path = os.path.join(out_class_dir, Path(img_path).stem+'_pred_'+label+'_'+'{:0.2f}'.format(score)+Path(img_path).suffix)
        # 画像コピー
        shutil.copyfile(img_path, out_path)

def pred_dir_cp(data_dir, pred_output_dir, model,
                img_rows=331, img_cols=331,
                classes = ['beagle', 'bikini', 'boke', 'cat', 'comic_book', 'fashion', 'marin', 'other', 'shingo', 'suit', 'tumblr'],
                TTA_dict={'TTA':'flip',
                          'TTA_rotate_deg':0,
                          'TTA_crop_num':4,
                          'TTA_crop_row':224, 'TTA_crop_col':224}):
    """
    指定ディレクトリの画像分類し、分類クラスごとにフォルダ切ってコピーする
    iPhone画像の予測分類用
    Args:
        data_dir: 入力ディレクトリ
        pred_output_dir: 出力先ディレクトリ
        model: モデルオブジェクト。アンサンブルする場合はリストでわたす
        img_rows, img_cols: モデルの入力サイズ
        classes: クラス名のリスト
        TTA_dict: TTAのオプション
    """
    os.makedirs(pred_output_dir, exist_ok=True)

    img_paths = util.get_jpg_png_path_in_dir(data_dir)

    # 1件づつpredictして、予測結果のデータフレームを返す
    pred_df = pred_tta_from_paths(model
                                  , img_paths
                                  , img_rows, img_cols
                                  , classes=classes
                                  , show_img=False
                                  , TTA=TTA_dict['TTA']
                                  , TTA_rotate_deg=TTA_dict['TTA_rotate_deg']
                                  , TTA_crop_num=TTA_dict['TTA_crop_num']
                                  , TTA_crop_size=[TTA_dict['TTA_crop_row'], TTA_dict['TTA_crop_col']]
                                  , preprocess=1.0/255.0
                                 )

    # 予測結果のデータフレーム出力
    pred_df.to_csv(os.path.join(pred_output_dir, 'pred.tsv'), sep='\t')

    # 予測ラベルごとに出力先ディレクトリを分けて、予測した画像コピー
    copy_pred_img_by_pred_df(pred_output_dir, pred_df)
    return


def load_img(img_file_path, img_rows, img_cols, preprocess=1.0/255.0):
    """（テスト）画像を読み込んで4次元テンソルへ変換"""
    from keras.preprocessing import image

    img = image.load_img(img_file_path, target_size=(img_rows, img_cols))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32')
    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！ これを忘れると結果がおかしくなるので注意
    x = x * preprocess
    return x

if __name__ == '__main__':
    print('base_predict.py: loaded as script file')
else:
    print('base_predict.py: loaded as module file')
