# -*- coding: utf-8 -*-
"""
ArcFace
- l2 softmax net と同じように正則化を入れてmetric出す方法のArcFaceのkerasの実装
- ArcFaceは正解ラベルも入力データに用いる
    - https://github.com/Yohei-Kawakami/201901_self_checkout/blob/develop/model/model2.ipynb
    - https://qiita.com/noritsugu_yamada/items/2e049cd7a8fd77eee0f5
"""

import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input,Dropout,Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import sys
# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得
sys.path.append( str(current_dir) + '/../' )
from predicter import base_predict
from transformer import get_train_valid_test, my_generator

class Arcfacelayer(Layer):
    """
    arcface layer
    https://github.com/Yohei-Kawakami/201901_self_checkout/blob/develop/model/arcface.py
    """
    # s:softmaxの温度パラメータ, m:margin
    def __init__(self, output_dim, s=30, m=0.50, easy_magin=False
                 , **kwargs):
        super(Arcfacelayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.s = s
        self.m = m
        self.easy_magin = easy_magin

    # 重みの作成
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Arcfacelayer, self).build(input_shape)

    def call(self, x):
        y = x[1]
        x_normalize = tf.nn.l2_normalize(x[0], axis=1) #tf.math.l2_normalize(x[0]) #|x = x'/ ||x'||2
        k_normalize = tf.nn.l2_normalize(self.kernel, axis=0) #tf.math.l2_normalize(self.kernel) # Wj = Wj' / ||Wj'||2
        cos_m = K.cos(self.m)
        sin_m = K.sin(self.m)
        th = K.cos(np.pi - self.m)
        mm = K.sin(np.pi - self.m) * self.m
        cosine = K.dot(x_normalize, k_normalize) # W.Txの内積
        sine = K.sqrt(1.0 - K.square(cosine))
        phi = cosine * cos_m - sine * sin_m # cos(θ+m)の加法定理
        if self.easy_magin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > th, phi, cosine - mm)
        # 正解クラス:cos(θ+m) 他のクラス:cosθ
        output = (y * phi) + ((1.0 - y) * cosine) # true cos(θ+m), False cos(θ)
        output *= self.s
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim) # 入力[x,y]のためx[0]はinput_shape[0][0]
        #return self.output_dim

def create_mpdel_with_arcface(n_categories, arc_s=30, arc_m=0.50, easy_magin=False,
                              base_model=MobileNet(input_shape=(224,224,3), weights='imagenet', include_top=False),
                              x=None,
                              weight_file_path=None):
    """
    train用model
    ベースのモデルの出力層の直前にarcface layerをつける
    論文ではsoftmaxに掛けるarc_s=64、角度のマージンarc_m=0.50
    """
    if x is None:
        # add new layers instead of FC networks
        x = base_model.output
        # stock hidden model
        hidden = GlobalAveragePooling2D()(x)
    else:
        hidden = x

    # stock hidden model
    #hidden = GlobalAveragePooling2D()(x)

    # stock Feature extraction
    yinput = Input(shape=(n_categories,), name='arcface_input')
    #x = Dropout(0.5)(hidden)
    x = Arcfacelayer(n_categories, s=arc_s, m=arc_m, easy_magin=easy_magin)([hidden,yinput])
    #x = Dense(1024,activation='relu')(x)

    prediction = Activation('softmax', name='arcface_activation')(x)

    model = Model(inputs=[base_model.input, yinput],outputs=prediction)

    if weight_file_path:
        model.load_weights(weight_file_path)
        print('weightは{}'.format(weight_file_path))

    return model

def create_predict_model(n_categories, arc_s=30, arc_m=0.50, easy_magin=False,
                         base_model=MobileNet(input_shape=(224,224,3), weights='imagenet', include_top=False),
                         x=None,
                         weight_file_path=None):
    """
    predict用model
    """
    arcface_model = create_mpdel_with_arcface(n_categories, arc_s=arc_s, arc_m=arc_m, easy_magin=easy_magin,
                                              base_model=base_model,
                                              x=x,
                                              weight_file_path=weight_file_path)
    predict_model = Model(arcface_model.get_layer(index=0).input, arcface_model.get_layer(index=-4).output)
    #predict_model.summary()
    return predict_model


# cos sim numpy
def cosine_similarity(x1, x2):
    """
    input
    x1 : shape (n_sample, n_features)
    x2 : shape (n_classes, n_features)
    ------
    output
    cos : shape (n_sample, n_classes)
    """
    x1_norm = np.linalg.norm(x1,axis=1) # 行毎のベクトルノルムをとり列ベクトルを求める ord=2 がデフォルトでl2ノルム
    x2_norm = np.linalg.norm(x2,axis=1) # 行毎のベクトルノルムをとり列ベクトルを求める ord=2 がデフォルトでl2ノルム
    return np.dot(x1, x2.T)/(x1_norm*x2_norm+1e-10)

# new画像のcos類似度を比較して一番値が高いindexを取り出しその値が閾値を超えるならindexを閾値以下ならをNoneを返す
def judgment(predict_vector, hold_vector, thresh=0.0):
    """
    predict_vector : shape(1,1028)
    hold_vector : shape(5, 1028)
    """
    cos_similarity = cosine_similarity(predict_vector, hold_vector) # shape(1, 5)
    #print(cos_similarity[0])
    # 最も値が高いindexを取得
    high_index = np.argmax(cos_similarity[0]) # int
    # cos類似度が閾値を超えるか
    if cos_similarity[0][high_index] > thresh:
        return high_index
    else:
        return None

# マスターのpredictをhold_vectorとして保存して,予測が正解かどうか判断し精度を算出
def test_acc(model, test_generator, hold_vector, thresh=0.0):
    # test_acc(model, test_dir, hold_dir, classes, thresh=0, sample=100):
    """
    テスト用
    model: 特徴抽出用モデル(predict)
    X: array
    test_dir: str 画像入ってるフォルダ
    hold_dir:str 登録データのフォルダ　ファイル名はclass名.jpgにしてください
    classes:　フォルダ名のリスト
    """
    correct = 0
    # 商品のvectorの呼び出し
    #hold_vector = get_hold_vector(model, classes, hold_dir)

    # test画像の作成
    #test_datagen=ImageDataGenerator(rescale=1.0/255.)
    #test_generator=test_datagen.flow_from_directory(
    #        test_dir,
    #        target_size=target_size,
    #        batch_size=1,
    #        class_mode='categorical',
    #        classes=classes)

    # 判定
    #for i in range(sample):
    #    X, Y = test_generator.next()
    label_index_list = []
    test_generator.reset()
    for step, (X, Y) in tqdm(enumerate(test_generator)):
        Y = np.argmax(Y, axis=1) # YはOneHotされているのでlabelのみ取り出す
        predict_vector = model.predict(X)
        #print(predict_vector.shape)
        index = judgment(predict_vector, hold_vector, thresh)
        label_index = index #// 4 # hold_vector用の画像は4枚ずつあるのでlabelは4で割った商になる
        label_index_list.append(label_index)
        if Y == label_index:
            correct += 1
            #print('pred_index: {}'.format(label_index))
            #print('Y: {}'.format(Y))
        if step+1 >= test_generator.n:
            break
    acc = correct / test_generator.n # 正解/全体
    print("acc: {}".format(acc))
    return acc, label_index_list

class train_Generator_xandy(object): # rule1
    def __init__(self, train_generator):
        #datagen = ImageDataGenerator(
        #                     vertical_flip = False,
        #                     width_shift_range = 0.1,
        #                     height_shift_range = 0.1,
        #                     rescale=1.0/255.,
        #                     zoom_range=0.2,
        #                     fill_mode = "constant",
        #                     cval=0)
        #train_generator=datagen.flow_from_directory(
        #  train_dir,
        #  target_size=(224,224),
        #  batch_size=25,
        #  class_mode='categorical',
        #  shuffle=True)
        self.gene = train_generator
    def __iter__(self):
    # __next__()はselfが実装してるのでそのままselfを返す
        return self
    def __next__(self):
        X, Y = self.gene.__next__()#next()
        return [X,Y], Y

class val_Generator_xandy(object):
    def __init__(self, validation_generator):
        #validation_datagen=ImageDataGenerator(rescale=1.0/255.)
        #validation_generator=validation_datagen.flow_from_directory(
        #    validation_dir,
        #    target_size=(224,224),
        #    batch_size=25,
        #    class_mode='categorical',
        #    shuffle=True)
        self.gene = validation_generator
    def __iter__(self):
    # __next__()はselfが実装してるのでそのままselfを返す
        return self
    def __next__(self):
        X, Y = self.gene.__next__()#next()
        return [X,Y], Y

########################### kaggleくずし字コンペ用 ###########################
def pred_hold_vec(predict_model, df_hold_file_id, output_dir
                   , class_count_max=5
                   , img_rows=32, img_cols=32, is_grayscale=False):
    """
    予測モデル（predict_model）と
    正解となるマスター画像のファイルパスとラベルidのデータフレーム（df_hold_file_id）から
    推論実行してマスター画像のベクトル(np.array(クラス数, class_count_max, モデルの出力層のサイズ))を取得する。

    予測したベクトルはnpyファイルでoutput_dirに保存する

    マスター画像は1クラス複数枚（class_count_max枚）用意しないと精度下がる
    Args:
        predict_model:ArcFaceモデル
        df_hold_file_id: 正解となるマスター画像のファイルパスとラベルidのデータフレーム
        output_dir:出力ディレクトリ
        class_count_max:1クラスあたりマスター画像何枚とるか
        img_rows,img_cols:モデルの入力層のサイズ
        is_grayscale:グレーにしてからpredictするか
    Return:
        np.array(pred_list):予測した正解となるマスター画像のベクトル(np.array(クラス数, class_count_max, モデルの出力層のサイズ))
    """
    df_hold_file_id.columns = ['file_path', 'label_id']
    pred_list = []

    for id in tqdm(sorted(df_hold_file_id['label_id'].unique())):
        df = df_hold_file_id[df_hold_file_id['label_id'] == id]

        if df.shape[0] > class_count_max:
            df = df[:class_count_max]

        X_list = []
        for f in df['file_path']:
            X = get_train_valid_test.load_one_img(f, img_rows, img_cols, is_grayscale=is_grayscale)
            X_list.append(X[0])

        # マスター画像は全クラス同じ枚数（class_count_max枚）用意する
        if len(X_list) < class_count_max:
            for i in range(class_count_max-len(X_list)):
                X_list.append(X[0])
        #print(np.array(X_list).shape)

        # 推論実行してマスター画像のベクトル取得
        pred = predict_model.predict(np.array(X_list))
        #print(pred.shape)
        pred_list.append(pred)

    # 予測したベクトルをnpyで保存
    np.save(output_dir+'/hold_vec.npy', np.array(pred_list))

    return np.array(pred_list)


def cossim(img1, img2):
    """
    コサイン距離：ベクトル空間モデルにおいて、文書同士を比較する際に用いられる類似度計算手法。
                  ベクトル同士の成す角度の近さを表現する。1に近ければ類似しており、0に近ければ似ていない
    """
    return np.dot(img1, img2) / (np.linalg.norm(img1) * np.linalg.norm(img2))

def pred_tta_X_hold_vec_cossim(predict_model, X, hold_vec
                               , hold_name_list=None
                               , img_rows=32, img_cols=32
                               , TTA='no_flip'#'flip'
                               , TTA_rotate_deg=0
                               , TTA_crop_num=0
                               , preprocess=1.0#1.0/255.0
                              ):
    """
    1画像についてTTAで予測ベクトル推論し、
    全クラスの正解ベクトルとの各コサイン類似度を計算して、
    コサイン類似度最大の予測クラス名とコサイン類似度を返す
    Args:
        predict_model:ArcFaceモデル
        X:予測する前処理済み4次元ベクトル
        hold_vec:正解となるマスター画像のベクトル。np.array(クラス数, class_count_max, モデルの出力層のサイズ)
        hold_name_list:hold_vec のラベル名リスト。予測クラス名出すのに使う。Noneなら予測クラス名ではなく予測クラスidを返す
        img_rows,img_cols:モデルの入力層のサイズ
        TTA*:TTAのoption
    Return:
        pred_label:コサイン類似度最大の予測クラス名(hold_name_list=Noneなら予測クラスid)
        pred_conf:pred_labelに対応するコサイン類似度
    """
    c_metric = base_predict.predict_tta(predict_model
                                        , X
                                        , TTA=TTA
                                        , TTA_rotate_deg=TTA_rotate_deg
                                        , TTA_crop_num=TTA_crop_num, TTA_crop_size=[int(img_rows*3/4), int(img_cols*3/4)]
                                        , preprocess=preprocess
                                        , resize_size=[img_rows, img_cols])

    #print(c_metric.shape)
    # 全マスター画像とcossim比較する
    cos_sim_mean_list = []
    for i,h_name in enumerate(hold_name_list):

        # マスター画像は1クラス複数枚用意しないと精度下がる
        h_vec = hold_vec[i]

        cos_sim_list = list(map(lambda h_metric: cossim(h_metric.flatten(), c_metric.flatten())
                                , h_vec)) # ndarray.flatten()で1次元にしないとコサイン類似度計算できない
        cos_sim_mean_list.append(np.mean(cos_sim_list)) # 1クラスについてのcossimの平均取る

    top_id = np.argmax(cos_sim_mean_list) # コサイン類似度が最大のindex

    if hold_name_list is None:
        pred_label = top_id
    else:
        pred_label = hold_name_list[top_id]

    pred_conf = cos_sim_mean_list[top_id]

    return pred_label, pred_conf
    #print('master+train_10:', class_conf, class_label) # cosとラベル確認用


def pred_tta_X_gen_hold_vec_cossim(predict_model
                                   , df_gen
                                   , hold_vec, hold_name_list
                                   , df_gen_data_dir=None
                                   , df_gen_x_col='char_id', df_gen_y_col='unicode_id'
                                   , img_rows=32, img_cols=32
                                   , TTA='no_flip'#'flip'
                                   , TTA_rotate_deg=0
                                   , TTA_crop_num=0
                                   , preprocess=1.0
                                   , color_mode='rgb'
                                   , class_mode='categorical'
                                   , classes=None
                                   , rescale=1.0/255.0
                                   , is_grayscale=True
                                   , max_step=None
                                  ):
    """
    batchサイズ=1のgenerator作って、
    正解となるマスター画像のベクトルのnp.array(hold_vec)
    予測した画像のベクトルとの各コサイン類似度を計算して、
    generatorでの正解率を返す。
    ①ファイルパスとラベルidのデータフレーム（df_classes）からflow_from_dataframe()関数使ってgenerator作成、
    ②TTAで予測
    ため引数多い
    Args:
        predict_model:ArcFaceモデル
        df_gen:予測する画像のファイルパスとラベル名のデータフレーム
        hold_vec:正解となるマスター画像のベクトル。np.array(クラス数, class_count_max, モデルの出力層のサイズ)
        hold_name_list:hold_vec のラベル名リスト。予測クラス名出すのに使う
        df_gen_data_dir:df_genのファイルディレクトリ
        df_gen_x_col, df_gen_y_col:df_genのファイル名とラベル名の列名
        img_rows,img_cols:モデルの入力層のサイズ
        TTA*:TTAのoption
        is_grayscale:グレーにしてからpredictするか
        max_step:generator何step回すか。Noneならgenerator最後までpredict
    Return:
        acc:正解率
    """
    datagen = my_generator.get_datagen(rescale=rescale, is_grayscale=is_grayscale)
    gen = datagen.flow_from_dataframe(
        df_gen,
        directory=df_gen_data_dir, # ラベルクラスをディレクトリ名にした画像ディレクトリのパス
        x_col=df_gen_x_col,
        y_col=df_gen_y_col,
        target_size=(img_rows, img_cols), # すべての画像はこのサイズにリサイズ
        color_mode=color_mode,# 画像にカラーチャンネルが3つある場合は「rgb」画像が白黒またはグレースケールの場合は「grayscale」
        classes=classes, # 分類クラス名リスト
        class_mode=class_mode, # 2値分類は「binary」、多クラス分類は「categorical」
        batch_size=1, # バッチごとにジェネレータから生成される画像の数
        seed=42, # 乱数シード
        shuffle=False # 生成されているイメージの順序をシャッフルする場合は「True」を設定し、それ以外の場合は「False」。train set は基本入れ替える
    )
    #gen_val = list(gen.class_indices.values())
    gen_keys = list(gen.class_indices.keys())

    gen.reset()
    if max_step is None:
        max_step = gen.n

    pred_label_list = []
    correct = 0
    for step, (X, y) in tqdm(enumerate(gen)):
        classes_id = np.argmax(y, axis=1)[0] # YはOneHotされているので、classes_id取り出す
        correct_label = gen_keys[classes_id] # 正解ラベル名
        #print(X.shape, classes_id, correct_label)

        pred_label, pred_conf = pred_tta_X_hold_vec_cossim(predict_model, X[0]
                                                           , hold_vec
                                                           , hold_name_list
                                                           , img_rows=img_rows, img_cols=img_cols
                                                           , TTA=TTA
                                                           , TTA_rotate_deg=TTA_rotate_deg
                                                           , TTA_crop_num=TTA_crop_num
                                                           , preprocess=preprocess
                                                          )
        #print(pred_label, pred_conf, '\n')
        pred_label_list.append(pred_label)

        if correct_label == pred_label:
            correct += 1

        if step+1 >= max_step:
            break

    acc = correct / max_step # 正解/全体
    print("acc: {}".format(acc))
    return acc
##########################################################################################

if __name__ == '__main__':
    print('arcface.py: loaded as script file')
else:
    print('arcface.py: loaded as module file')
