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

class Arcfacelayer(Layer):
    """
    arcface layer
    https://github.com/Yohei-Kawakami/201901_self_checkout/blob/develop/model/arcface.py
    """
    # s:softmaxの温度パラメータ, m:margin
    def __init__(self, output_dim, s=30, m=0.50, easy_magin=False):
        self.output_dim = output_dim
        self.s = s
        self.m = m
        self.easy_magin = easy_magin
        super(Arcfacelayer, self).__init__()

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
    test_generator.reset()
    for step, (X, Y) in enumerate(test_generator):
        Y = np.argmax(Y, axis=1) # YはOneHotされているのでlabelのみ取り出す
        predict_vector = model.predict(X)
        index = judgment(predict_vector, hold_vector, thresh)
        label_index = index #// 4 # hold_vector用の画像は4枚ずつあるのでlabelは4で割った商になる
        if Y == label_index:
            correct += 1
            #print('pred_index: {}'.format(label_index))
            #print('Y: {}'.format(Y))
        if step+1 >= test_generator.n:
            break
    acc = correct / test_generator.n # 正解/全体
    print("acc: {}".format(acc))
    return acc

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

if __name__ == '__main__':
    print('arcface.py: loaded as script file')
else:
    print('arcface.py: loaded as module file')
