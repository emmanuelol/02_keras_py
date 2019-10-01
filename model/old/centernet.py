# -*- coding: utf-8 -*-
"""
CenterNet
YOLOなどのようにアンカーを使用せず、セグメンテーション(U-Net)のようなヒートマップで対象物の中心点を検出する手法
COCOデータセットなら、Faster-RCNN+inception-resnetよりも、2019年に出た centernet の方がmAP(mean Average Precision)高いみたい。
mAP=37ぐらい。
tensorflow model zooの Faster-RCNN+inception-resnetもmAP=37
    - 元コード:https://www.kaggle.com/kmat2019/centernet-keypoint-detector
    - CenterNetの論文:https://arxiv.org/abs/1904.07850
    - 論文のpytorch版コード:https://github.com/xingyizhou/CenterNet
"""
# 元コード試したnotebook:
# C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\kaggle_kuzushiji-recognition\CenterNet -Keypoint Detector\test_CenterNet -Keypoint Detector.ipynb

import numpy as np
import json
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
import random
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import KFold,train_test_split
import matplotlib.pyplot as plt
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Dropout, Conv2D,Conv2DTranspose, BatchNormalization, Activation,AveragePooling2D,GlobalAveragePooling2D, Input, Concatenate, MaxPool2D, Add, UpSampling2D, LeakyReLU,ZeroPadding2D
from keras.models import Model
from keras.objectives import mean_squared_error
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
import os

from keras.optimizers import Adam, RMSprop, SGD
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession

from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw
import argparse
import keras

# グローバル変数
category_n = 1 # 分類クラス数
input_width,input_height = 512,512 # Centernetの入力層のサイズ
output_layer_n = category_n+4
output_height,output_width = input_width//4,input_height//4 # Centernetの出力層のサイズ

########################################## step1 Encoder ##########################################

def Datagen_sizecheck_model(input_for_size_estimate, batch_size, size_detection_mode=True, is_train=True, random_crop=True, input_width=512, input_height=512):
    """
    文字と画像のサイズ比を推定する回帰モデル(CenternetのEncoderだけ)で使う自作datagenerator
    水増しはrandom_cropのみ
    CenterNetの出力方式に対して過少に小さい文字は、検出できないので画像切り出す
    Args:
        input_for_size_estimate:trainデータリスト。[[画像path,log(文字サイズ÷ピクチャサイズ) の値], …]。例. [['train_images/200021712-00022_2.jpg', -6.966739922505915], …]
        batch_size:バッチサイズ
        size_detection_mode:文字と画像のサイズ比を推定する回帰モデルにするときはTrue
        is_train:学習時はTrue
        random_crop:Trueならrandom cropの水増し入れる
        input_width, input_height:モデルの入力層のサイズ
    """
    x = []
    y = []

    count=0

    while True:
        for i in range(len(input_for_size_estimate)):
            if random_crop:
                crop_ratio=np.random.uniform(0.7,1)
            else:
                crop_ratio=1
            with Image.open(input_for_size_estimate[i][0]) as f:
                #random crop
                if random_crop and is_train:
                    pic_width,pic_height=f.size
                    f=np.asarray(f.convert('RGB'),dtype=np.uint8)
                    top_offset=np.random.randint(0,pic_height-int(crop_ratio*pic_height))
                    left_offset=np.random.randint(0,pic_width-int(crop_ratio*pic_width))
                    bottom_offset=top_offset+int(crop_ratio*pic_height)
                    right_offset=left_offset+int(crop_ratio*pic_width)
                    f=cv2.resize(f[top_offset:bottom_offset,left_offset:right_offset,:],(input_height,input_width))
                else:
                    f=f.resize((input_width, input_height))
                    f=np.asarray(f.convert('RGB'),dtype=np.uint8)
                x.append(f)


            if random_crop and is_train:
                y.append(input_for_size_estimate[i][1]-np.log(crop_ratio))
            else:
                y.append(input_for_size_estimate[i][1])

            count+=1
            if count==batch_size:
                x = np.array(x, dtype=np.float32)
                y = np.array(y, dtype=np.float32)

                inputs=x/255
                targets=y
                x = []
                y = []
                count=0
                yield inputs, targets

def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):
    """ aggregation block """
    x_deep = Conv2DTranspose(deep_ch, kernel_size=2, strides=2, padding='same', use_bias=False)(x_deep)
    x_deep = BatchNormalization()(x_deep)
    x_deep = LeakyReLU(alpha=0.1)(x_deep)
    x = Concatenate()([x_shallow, x_deep])
    x = Conv2D(out_ch, kernel_size=1, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def cbr(x, out_layer, kernel, stride):
    """ Conv2D+BatchNormalization+LeakyReLU """
    x = Conv2D(out_layer, kernel_size=kernel, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def resblock(x_in,layer_n):
    """ resnet block """
    x = cbr(x_in,layer_n,3,1)
    x = cbr(x,layer_n,3,1)
    x = Add()([x,x_in])
    return x

#I use the same network at CenterNet
def create_model(input_shape, size_detection_mode=True, aggregation=True, output_layer_n=1+4):
    """
    ResnetベースのCenterNet構築
    CenterNetはU-Netみたいな構造(Encoder+Decoder)
    なので、Encoder学習して、Decorder学習みたいに2stepでtrain走らすことができる
    この関数では、
    1step目で、Encoderに1nodeの全結合層つけた回帰モデルとしてEncoder部分の重みを学習できるようにしている
    回帰モデルのラベルは「文字と画像のサイズ比」としている
    2step目で、Encoder+DecoderのCenterNetで（Encoderは1stepの重みロードするの忘れずに）
    [heatmap, category（今回は1クラスだけ）, xとy中心オフセット, 物体幅と高さ]を推定できるようにしている
    Args:
        input_shape:入力層のサイズ。(512,512,3) とか
        size_detection_mode:Trueなら文字と画像のサイズ比を推定する回帰モデル(Encorder)を返す。
                            FalseならCenterNet(Encoder+Decoder)を返す
        aggregation:TrueならDecoderにaggregation_block()を追加する
        output_layer_n:出力nodeの数。heatmap、category（デフォルトは1クラスだけ）、xとy中心オフセット、物体幅と高さの5node
    Returns:
        modelオブジェクト:
        引数のsize_detection_mode==Trueなら、文字と画像のサイズ比を推定する回帰モデル(CenternetのEncoderだけ)
        引数のsize_detection_mode==Falseなら、CenterNet
    """
    input_layer = Input(input_shape)

    #resized input
    input_layer_1=AveragePooling2D(2)(input_layer)
    input_layer_2=AveragePooling2D(2)(input_layer_1)

    #### ENCODER ####

    x_0 = cbr(input_layer, 16, 3, 2)#512->256
    concat_1 = Concatenate()([x_0, input_layer_1])

    x_1 = cbr(concat_1, 32, 3, 2)#256->128
    concat_2 = Concatenate()([x_1, input_layer_2])

    x_2 = cbr(concat_2, 64, 3, 2)#128->64

    x = cbr(x_2,64,3,1)
    x = resblock(x,64)
    x = resblock(x,64)

    x_3 = cbr(x, 128, 3, 2)#64->32
    x = cbr(x_3, 128, 3, 1)
    x = resblock(x,128)
    x = resblock(x,128)
    x = resblock(x,128)

    x_4 = cbr(x, 256, 3, 2)#32->16
    x = cbr(x_4, 256, 3, 1)
    x = resblock(x,256)
    x = resblock(x,256)
    x = resblock(x,256)
    x = resblock(x,256)
    x = resblock(x,256)

    x_5 = cbr(x, 512, 3, 2)#16->8
    x = cbr(x_5, 512, 3, 1)

    x = resblock(x,512)
    x = resblock(x,512)
    x = resblock(x,512)

    if size_detection_mode:
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        out=Dense(1,activation="linear")(x)

    else:#centernet mode
    #### DECODER ####
        x_1 = cbr(x_1, output_layer_n, 1, 1)
        x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
        x_2 = cbr(x_2, output_layer_n, 1, 1)
        x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
        x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
        x_3 = cbr(x_3, output_layer_n, 1, 1)
        x_3 = aggregation_block(x_3, x_4, output_layer_n, output_layer_n)
        x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
        x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

        x_4 = cbr(x_4, output_layer_n, 1, 1)

        x = cbr(x, output_layer_n, 1, 1)
        x = UpSampling2D(size=(2, 2))(x)#8->16 tconvのがいいか

        x = Concatenate()([x, x_4])
        x = cbr(x, output_layer_n, 3, 1)
        x = UpSampling2D(size=(2, 2))(x)#16->32

        x = Concatenate()([x, x_3])
        x = cbr(x, output_layer_n, 3, 1)
        x = UpSampling2D(size=(2, 2))(x)#32->64   128のがいいかも？

        x = Concatenate()([x, x_2])
        x = cbr(x, output_layer_n, 3, 1)
        x = UpSampling2D(size=(2, 2))(x)#64->128

        x = Concatenate()([x, x_1])
        x = Conv2D(output_layer_n, kernel_size=3, strides=1, padding="same")(x)
        out = Activation("sigmoid")(x)

    model = Model(input_layer, out)

    return model


def create_sizecheck_model(input_height, input_width, out_dir=None, hdf5_path=None, is_summary=True):
    """
    centernetのEncoderに1nodeの全結合層つけた回帰モデル
    1step目のEncoder部分の重みの学習で使うモデル
    """
    K.clear_session()
    model = create_model(input_shape=(input_height, input_width,3), size_detection_mode=True)
    if out_dir is not None:
        model.save(out_dir + '/CenterNet_1step_Encoder_model.h5', include_optimizer=False)
    if hdf5_path is not None:
        # 重みファイルあればロード
        model.load_weights(hdf5_path)
        print('load_weights')
    if is_summary == True:
        print(model.summary())
    return model

def fit_sizecheck_model(model, train_list, cv_list, n_epoch, batch_size, lr=0.005, out_dir=None):
    """
    検出領域と画像のサイズ比を推定する回帰モデル(CenternetのEncoderだけ) train
    Args:
        model:1stepのEncoderに1nodeの全結合層つけた回帰モデル
        train_list:trainデータリスト。[[画像path,log(文字サイズ÷ピクチャサイズ) の値], …]。例. [['train_images/200021712-00022_2.jpg', -6.966739922505915], …]
        cv_list:validationデータリスト。[[画像path,log(文字サイズ÷ピクチャサイズ) の値], …]。例. [['valid_images/200021712-00022_2.jpg', -6.966739922505915], …]
        n_epoch:epoch数
        batch_size:バッチサイズ
        lr:学習率
        out_dir:出力先ディレクトリ
    """
    cb = []
    """
    # EarlyStopping
    early_stopping = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 10, verbose = 1)
    # ModelCheckpoint
    weights_dir = '/model_1/'
    if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
    model_checkpoint = ModelCheckpoint(weights_dir + "val_loss{val_loss:.3f}.hdf5", monitor = 'val_loss', verbose = 1,
                                          save_best_only = True, save_weights_only = True, period = 1)
    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, verbose = 1)
    """
    def lrs(epoch, lr=lr):
        lr = lr
        if epoch > 10:
            lr = lr/5.#0.0001
        return lr
    cb.append( LearningRateScheduler(lrs) )
    if out_dir is not None:
        cb.append( ModelCheckpoint( os.path.join(out_dir, "final_weights_step1.hdf5")
                                    , monitor='val_loss', verbose=1
                                    , save_best_only=True, save_weights_only=True, period=1) )

    model.compile(loss=mean_squared_error, optimizer=Adam(lr=lr))

    hist = model.fit_generator(
        Datagen_sizecheck_model(train_list, batch_size, is_train=True, random_crop=True), # 文字と画像のサイズ比を推定する回帰モデル(CenternetのEncoderだけ)で使う自作datagenerator
        steps_per_epoch = len(train_list) // batch_size,
        epochs = n_epoch,
        validation_data = Datagen_sizecheck_model(cv_list,batch_size, is_train=False,random_crop=False),
        validation_steps = len(cv_list) // batch_size,
        callbacks = cb, #[lr_schedule, model_checkpoint],#[early_stopping, reduce_lr, model_checkpoint],
        shuffle = True,
        verbose = 2#1
    )
    return hist

def plot_predict_sizecheck_model(model, cv_list, batch_size, out_dir=None):
    """ log(文字サイズ÷ピクチャサイズ) の値を推定する回帰モデルの予測結果をplotする """
    predict = model.predict_generator(Datagen_sizecheck_model(cv_list, batch_size, is_train=False, random_crop=False), steps=len(cv_list) // batch_size)
    target = [cv[1] for cv in cv_list]
    plt.scatter(predict,target[:len(predict)])
    plt.title('---letter_size/picture_size--- estimated vs target ',loc='center',fontsize=10)
    if out_dir is not None:
        plt.savefig( os.path.join(out_dir, 'letter_size_picture_size_estimated_vs_target.jpg'), bbox_inches="tight" )
    plt.show()

def get_train_w_split_predict_sizecheck_model(model, train_input_for_size_estimate, base_detect_num_h=25, base_detect_num_w=25):
    """
    train setについてのlog(文字サイズ÷ピクチャサイズ) の値を推定する回帰モデルの予測結果を取得し、
    step2のモデルで使うために、検出できるのはせいぜい25x25くらいだと考えて、画像データの分割数を適当に決める
    """
    batch_size = 1
    # train set predict
    predict_train = model.predict_generator(Datagen_sizecheck_model(train_input_for_size_estimate, batch_size
                                            , is_train=False
                                            , random_crop=False, )
                                            , steps=len(train_input_for_size_estimate)//batch_size)
    # 画像データの分割数決める
    annotation_list_train_w_split=[]
    for i, predicted_size in enumerate(predict_train):
        detect_num_h = aspect_ratio_pic_all[i]*np.exp(-predicted_size/2)
        detect_num_w = detect_num_h/aspect_ratio_pic_all[i]
        h_split_recommend = np.maximum(1,detect_num_h/base_detect_num_h)
        w_split_recommend = np.maximum(1,detect_num_w/base_detect_num_w)
        annotation_list_train_w_split.append([annotation_list_train[i][0]
                                              , annotation_list_train[i][1]
                                              , h_split_recommend
                                              , w_split_recommend])
    # 試しに1件可視化
    for i in np.arange(0,1):
        print("recommended height split:{}, recommended width_split:{}".format(annotation_list_train_w_split[i][2],annotation_list_train_w_split[i][3]))
        img = np.asarray(Image.open(annotation_list_train_w_split[i][0]).convert('RGB'))
        plt.imshow(img)
        plt.show()

    return annotation_list_train_w_split

#def model_fit_sizecheck_model(model, train_list, cv_list, n_epoch, callbacks=[], batch_size=32):
#    """
#    検出領域と画像のサイズ比を推定する回帰モデル(CenternetのEncoderだけ) train
#    Args:
#        model:1stepのEncoderに1nodeの全結合層つけた回帰モデル
#        train_list:trainデータリスト。[[画像path,log(文字サイズ÷ピクチャサイズ) の値], …]。例. [['train_images/200021712-00022_2.jpg', -6.966739922505915], …]
#        cv_list:validationデータリスト。[[画像path,log(文字サイズ÷ピクチャサイズ) の値], …]。例. [['valid_images/200021712-00022_2.jpg', -6.966739922505915], …]
#        n_epoch:epoch数
#        callbacks:kerasのcallback
#        batch_size:バッチサイズ
#    Usage:
#        K.clear_session()
#        input_height, input_width = 512, 512
#        model = create_model(input_shape=(input_height,input_width,3), size_detection_mode=True)
#
#        def lrs(epoch):
#            lr = 0.0005
#            if epoch>10:
#                lr = 0.0001
#            return lr
#        lr_schedule = LearningRateScheduler(lrs)
#        model_checkpoint = ModelCheckpoint(out_dir + "/final_weights_step1.hdf5", monitor='val_loss', verbose=1,
#                                           save_best_only=True, save_weights_only=True, period = 1)
#
#        train_list, cv_list = train_test_split(train_input_for_size_estimate, random_state=111, test_size=0.2)
#
#        learning_rate=0.0005
#        n_epoch=12
#        batch_size=8#32
#
#        model.compile(loss=mean_squared_error, optimizer=Adam(lr=learning_rate))
#        hist = model_fit_sizecheck_model(model, train_list, cv_list, n_epoch
#                                        , callbacks=[lr_schedule, model_checkpoint]
#                                        , batch_size=batch_size)
#
#        # log(文字サイズ÷ピクチャサイズ) の値を推定する回帰モデルの予測結果の散布図をplotする
#        predict = model.predict_generator(Datagen_sizecheck_model(cv_list, batch_size, is_train=False, random_crop=False),
#                                          steps=len(cv_list) // batch_size)
#        target=[cv[1] for cv in cv_list]
#        plt.scatter(predict,target[:len(predict)])
#        plt.title('---letter_size/picture_size--- estimated vs target ',loc='center',fontsize=10)
#        plt.show()
#    """
#    hist = model.fit_generator(
#        Datagen_sizecheck_model(train_list, batch_size, is_train=True, random_crop=True), # 文字と画像のサイズ比を推定する回帰モデル(CenternetのEncoderだけ)で使う自作datagenerator
#        steps_per_epoch = len(train_list) // batch_size,
#        epochs = n_epoch,
#        validation_data = Datagen_sizecheck_model(cv_list,batch_size, is_train=False,random_crop=False),
#        validation_steps = len(cv_list) // batch_size,
#        callbacks = callbacks, #[lr_schedule, model_checkpoint],#[early_stopping, reduce_lr, model_checkpoint],
#        shuffle = True,
#        verbose = 2#1
#    )
#    return hist

########################################## step2 Centernet(Encoder+Decoder) ##########################################

def Datagen_centernet(annotation_list_w_split, batch_size
                      , annotation_list_train_w_split=None
                      , input_width=512, input_height=512
                      , output_height=128, output_width=128
                      , category_n=1):
    """
    CenterNet用のDatagenerator
    CenterNetの出力方式に対して過少に小さい文字は検出できないので画像切り出す
    Args:
        annotation_list_w_split:trainデータリスト。[[画像path, [[ラベルid, x,y中心オフセット, 物体幅と高さ], [ラベルid, x,y中心オフセット, 物体幅と高さ] …], 縦のcrop_ratio(あとでサイズを1/crop_ratioする), 横のcrop_ratio(あとでサイズを1/crop_ratioする)], …]
                   例. [['train_images/200021712-00022_2.jpg', [ 232, 1401, 1685,   92,   13], 1.7, 1.2], …]
        batch_size:バッチサイズ
        input_width, input_height:モデルの入力層のサイズ
        output_height, output_width:モデルの出力層のサイズ
        category_n:分類クラス。デフォルトは1クラスだけ
    """
    x = [] # xは(input_width, input_height, 3)をランダムなサイズに切り出した画像
    y = [] # yにはlossを計算するための位置情報として[heatmap, category（今回は1クラスだけ, xとy中心オフセット, 物体幅と高さ]が格納される

    count=0

    while True:
        for i in range(len(annotation_list_w_split)):
            h_split=annotation_list_w_split[i][2]
            w_split=annotation_list_w_split[i][3]
            max_crop_ratio_h=1/h_split
            max_crop_ratio_w=1/w_split
            crop_ratio=np.random.uniform(0.5,1)
            crop_ratio_h=max_crop_ratio_h*crop_ratio
            crop_ratio_w=max_crop_ratio_w*crop_ratio

            with Image.open(annotation_list_w_split[i][0]) as f:
                # 検出する文字サイズを同じぐらいになる画像のサイズにrandom cropで切り出す
                pic_width,pic_height=f.size
                f=np.asarray(f.convert('RGB'),dtype=np.uint8)
                top_offset=np.random.randint(0,pic_height-int(crop_ratio_h*pic_height))
                left_offset=np.random.randint(0,pic_width-int(crop_ratio_w*pic_width))
                bottom_offset=top_offset+int(crop_ratio_h*pic_height)
                right_offset=left_offset+int(crop_ratio_w*pic_width)
                f=cv2.resize(f[top_offset:bottom_offset,left_offset:right_offset,:],(input_height,input_width))
                x.append(f)

            output_layer=np.zeros((output_height,output_width,(output_layer_n+category_n)))
            for annotation in annotation_list_w_split[i][1]:
                # random cropで切り出すので位置を合わせる
                x_c=(annotation[1]-left_offset)*(output_width/int(crop_ratio_w*pic_width))
                y_c=(annotation[2]-top_offset)*(output_height/int(crop_ratio_h*pic_height))
                width=annotation[3]*(output_width/int(crop_ratio_w*pic_width))
                height=annotation[4]*(output_height/int(crop_ratio_h*pic_height))

                top=np.maximum(0,y_c-height/2)
                left=np.maximum(0,x_c-width/2)
                bottom=np.minimum(output_height,y_c+height/2)
                right=np.minimum(output_width,x_c+width/2)

                if top>=(output_height-0.1) or left>=(output_width-0.1) or bottom<=0.1 or right<=0.1:#random crop(out of picture)
                    continue
                width=right-left
                height=bottom-top
                x_c=(right+left)/2
                y_c=(top+bottom)/2

                category=category_n-1#0#not classify, just detect
                heatmap=((np.exp(-(((np.arange(output_width)-x_c)/(width/10))**2)/2)).reshape(1,-1)
                                    *(np.exp(-(((np.arange(output_height)-y_c)/(height/10))**2)/2)).reshape(-1,1))
                # heatmap
                output_layer[:,:,category]=np.maximum(output_layer[:,:,category],heatmap[:,:])
                # category（今回は1クラスだけ）
                output_layer[int(y_c//1),int(x_c//1),category_n+category]=1
                # x,y中心オフセット
                output_layer[int(y_c//1),int(x_c//1),2*category_n]=y_c%1#height offset
                output_layer[int(y_c//1),int(x_c//1),2*category_n+1]=x_c%1
                # 物体幅と高さ
                output_layer[int(y_c//1),int(x_c//1),2*category_n+2]=height/output_height
                output_layer[int(y_c//1),int(x_c//1),2*category_n+3]=width/output_width
            y.append(output_layer)

            count+=1
            if count==batch_size:
                x = np.array(x, dtype=np.float32)
                y = np.array(y, dtype=np.float32)
                inputs=x/255
                targets=y
                x = []
                y = []
                count=0
                yield inputs, targets

def all_loss(y_true, y_pred):
    """ heatmap + x,y中心オフセット + 物体幅と高さ のlossの合計 """
    mask=K.sign(y_true[...,2*category_n+2])
    N=K.sum(mask)
    alpha=2.
    beta=4.

    heatmap_true_rate = K.flatten(y_true[...,:category_n])
    heatmap_true = K.flatten(y_true[...,category_n:(2*category_n)])
    heatmap_pred = K.flatten(y_pred[...,:category_n])
    heatloss=-K.sum(heatmap_true*((1-heatmap_pred)**alpha)*K.log(heatmap_pred+1e-6)+(1-heatmap_true)*((1-heatmap_true_rate)**beta)*(heatmap_pred**alpha)*K.log(1-heatmap_pred+1e-6))
    offsetloss=K.sum(K.abs(y_true[...,2*category_n]-y_pred[...,category_n]*mask)+K.abs(y_true[...,2*category_n+1]-y_pred[...,category_n+1]*mask))
    sizeloss=K.sum(K.abs(y_true[...,2*category_n+2]-y_pred[...,category_n+2]*mask)+K.abs(y_true[...,2*category_n+3]-y_pred[...,category_n+3]*mask))

    all_loss=(heatloss+1.0*offsetloss+5.0*sizeloss)/N
    return all_loss

def size_loss(y_true, y_pred):
    """ 物体幅と高さについてのloss """
    mask=K.sign(y_true[...,2*category_n+2])
    N=K.sum(mask)
    sizeloss=K.sum(K.abs(y_true[...,2*category_n+2]-y_pred[...,category_n+2]*mask)+K.abs(y_true[...,2*category_n+3]-y_pred[...,category_n+3]*mask))
    return (5*sizeloss)/N

def offset_loss(y_true, y_pred):
    """ x,y中心オフセットについてのloss """
    mask=K.sign(y_true[...,2*category_n+2])
    N=K.sum(mask)
    offsetloss=K.sum(K.abs(y_true[...,2*category_n]-y_pred[...,category_n]*mask)+K.abs(y_true[...,2*category_n+1]-y_pred[...,category_n+1]*mask))
    return (offsetloss)/N

def heatmap_loss(y_true, y_pred):
    """ heatmapについてのloss """
    mask=K.sign(y_true[...,2*category_n+2])
    N=K.sum(mask)
    alpha=2.
    beta=4.

    heatmap_true_rate = K.flatten(y_true[...,:category_n])
    heatmap_true = K.flatten(y_true[...,category_n:(2*category_n)])
    heatmap_pred = K.flatten(y_pred[...,:category_n])
    heatloss=-K.sum(heatmap_true*((1-heatmap_pred)**alpha)*K.log(heatmap_pred+1e-6)+(1-heatmap_true)*((1-heatmap_true_rate)**beta)*(heatmap_pred**alpha)*K.log(1-heatmap_pred+1e-6))
    return heatloss/N


def model_fit_centernet(model, train_list, cv_list, n_epoch, callbacks=[], batch_size=32):
    """
    Centernet train
    Args:
        model:2stepのEncoder+DecoderのCenternet
        train_list:trainデータリスト。[[画像path, [[ラベルid, x,y中心オフセット, 物体幅と高さ], [ラベルid, x,y中心オフセット, 物体幅と高さ] …], 縦のcrop_ratio(あとでサイズを1/crop_ratioする), 横のcrop_ratio(あとでサイズを1/crop_ratioする)], …]
                   例. [['train_images/200021712-00022_2.jpg', [ 232, 1401, 1685,   92,   13], 1.7, 1.2], …]
        cv_list:validationデータリスト。[[画像path, [[ラベルid, x,y中心オフセット, 物体幅と高さ], [ラベルid, x,y中心オフセット, 物体幅と高さ] …], 縦のcrop_ratio(あとでサイズを1/crop_ratioする), 横のcrop_ratio(あとでサイズを1/crop_ratioする)], …]
                例. [['valid_images/200021712-00022_2.jpg', [ 232, 1401, 1685,   92,   13], 1.7, 1.2], …]
        n_epoch:epoch数
        callbacks:kerasのcallback
        batch_size:バッチサイズ
    Usage:
        K.clear_session()
        input_height, input_width = 512, 512
        model = create_model(input_shape=(input_height,input_width,3), size_detection_mode=False)

        def lrs(epoch):
            lr = 0.001
            if epoch >= 20: lr = 0.0002
            return lr
        lr_schedule = LearningRateScheduler(lrs)
        model_checkpoint = ModelCheckpoint(out_dir + "/final_weights_step2.hdf5", monitor = 'val_loss', verbose = 1,
                                           save_best_only = True, save_weights_only = True, period = 1)

        # step1のEncoderの重みあればload
        model.load_weights(out_dir + '/final_weights_step1.hdf5', by_name=True, skip_mismatch=True)

        train_list, cv_list = train_test_split(annotation_list_train_w_split, random_state=111, test_size=0.2)#stratified split is better

        learning_rate=0.001
        n_epoch=30
        batch_size=32

        model.compile(loss=all_loss, optimizer=Adam(lr=learning_rate), metrics=[heatmap_loss, size_loss, offset_loss])
        hist = model_fit_centernet(model,train_list,cv_list,n_epoch
                                   , callbacks=[lr_schedule, model_checkpoint]
                                   , batch_size=batch_size)
    """
    hist = model.fit_generator(
        Datagen_centernet(train_list,batch_size),
        steps_per_epoch = len(train_list) // batch_size,
        epochs = n_epoch,
        validation_data=Datagen_centernet(cv_list,batch_size),
        validation_steps = len(cv_list) // batch_size,
        callbacks = callbacks, #[lr_schedule],#early_stopping, reduce_lr, model_checkpoint],
        shuffle = True,
        verbose = 2#1
    )
    return hist

def convert_input_data(path_1, path_2, path_3, path_4, input_width= 512, input_height=512):
    """
    kaggleのくずし字コンペのCenterNet用に入力データを形成
    Usage:
        data_dir = r'D:\work\kaggle_kuzushiji-recognition\OrigData\kuzushiji-recognition'
        path_1 = data_dir+"/train.csv"
        path_2 = data_dir+"/train_images/"
        path_3 = data_dir+"/test_images/"
        path_4 = data_dir+"/sample_submission.csv"

        df_train, category_names, inv_dict_cat, annotation_list_train, id_test, df_submission = \
        centernet.convert_input_data(path_1, path_2, path_3, path_4, input_width= 512, input_height=512)
    """
    df_train=pd.read_csv(path_1)
    df_train = df_train.dropna(axis=0, how='any')#you can use nan data(page with no letter)
    df_train = df_train.reset_index(drop=True)
    print('df_train:')
    display(df_train.head())

    annotation_list_train = []
    category_names = set()
    for i in range(len(df_train)):
        ann = np.array(df_train.loc[i,"labels"].split(" ")).reshape(-1,5)#cat,x,y,width,height for each picture
        category_names = category_names.union({i for i in ann[:,0]})
    category_names = sorted(category_names)
    print('\ncategory_names:', category_names[:5])

    dict_cat = {list(category_names)[j]:str(j) for j in range(len(category_names))}
    inv_dict_cat = {str(j):list(category_names)[j] for j in range(len(category_names))}

    #print(dict_cat)
    for i in range(len(df_train)):
        ann = np.array(df_train.loc[i,"labels"].split(" ")).reshape(-1,5)#cat,left,top,width,height for each picture
        #print(ann)
        for j,category_name in enumerate(ann[:,0]):
            ann[j,0] = int(dict_cat[category_name]) # ラベルid
            #print(ann[j,0])
        ann = ann.astype('int32')
        ann[:,1] += ann[:,3]//2#center_x # ann[:,1]はx座標。ann[:,3]は幅なので、1/2かけたの足して、ann[:,1]を物体の中心位置にする
        ann[:,2] += ann[:,4]//2#center_y # ann[:,2]はy座標。ann[:,4]は高さなので、1/2かけたの足して、ann[:,2]を物体の中心位置にする
        annotation_list_train.append(["{}{}.jpg".format(path_2,df_train.loc[i,"image_id"]),ann])

    print('\nannotation_list_train:')
    print(annotation_list_train[0:1])

    print("sample image")
    img = np.asarray(Image.open(annotation_list_train[0][0]).resize((input_width, input_height)).convert('RGB'))
    plt.imshow(img)
    plt.show()

    # get directory of test images
    df_submission = pd.read_csv(path_4)
    id_test = path_3+df_submission["image_id"]+".jpg"

    return df_train, category_names, inv_dict_cat, annotation_list_train, id_test, df_submission


def preprocess_check_object_size(annotation_list_train, id_test, out_dir=None):
    """
    検出モデルを作る前に、文字サイズをチェックする。
    CenterNetの出力方式に対して過少に小さい文字は、検出できないため。
    Usage:
        aspect_ratio_pic_all, aspect_ratio_pic_all_test, average_letter_size_all, train_input_for_size_estimate = \
        centernet.preprocess_check_object_size(annotation_list_train, id_test, out_dir=out_dir)
    """
    aspect_ratio_pic_all = []
    aspect_ratio_pic_all_test = []
    average_letter_size_all = []
    train_input_for_size_estimate = []

    # train setについて
    for i in tqdm(range(len(annotation_list_train))):
        with Image.open(annotation_list_train[i][0]) as f:
            width,height = f.size
            # 画像の縦対横のサイズ
            area = width*height
            aspect_ratio_pic = height/width
            aspect_ratio_pic_all.append(aspect_ratio_pic)
            # 1画像内の各文字の縦横サイズ
            letter_size = annotation_list_train[i][1][:,3]*annotation_list_train[i][1][:,4]
            # 1画像に対する各文字の縦対横のサイズ
            letter_size_ratio = letter_size/area
            # 1画像に対する各文字の縦対横のサイズの平均
            average_letter_size = np.mean(letter_size_ratio)
            average_letter_size_all.append(average_letter_size)
            # 画像パスとその画像に対する各文字の縦対横のサイズの平均を1要素としてリストに詰める
            train_input_for_size_estimate.append([annotation_list_train[i][0], np.log(average_letter_size)])#logにしとく

    # test setについて
    for i in tqdm(range(len(id_test))):
        with Image.open(id_test[i]) as f:
            width,height=f.size
            # 画像の縦対横のサイズ
            aspect_ratio_pic=height/width
            aspect_ratio_pic_all_test.append(aspect_ratio_pic)

    # train setの画像に対する各文字の縦対横のサイズの平均をヒストグラムにする
    plt.hist(np.log(average_letter_size_all),bins=100)
    plt.title('log(ratio of letter_size to picture_size))', loc='center', fontsize=12)
    if out_dir is not None:
        plt.savefig( os.path.join(out_dir, 'log(ratio_of_letter_size_to_picture_size)).jpg'), bbox_inches="tight" )
    plt.show()

    return aspect_ratio_pic_all, aspect_ratio_pic_all_test, average_letter_size_all, train_input_for_size_estimate



def main():
    import matplotlib
    matplotlib.use('Agg')
    plt.style.use('ggplot')

    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, required=True,
                    help="output dir path.")
    ap.add_argument("-p1", "--train_csv", type=str, required=True,
                    help="train csv path.")
    ap.add_argument("-p2", "--train_images", type=str, required=True,
                    help="train_images dir path.")
    ap.add_argument("-p3", "--test_images", type=str, required=True,
                    help="test_images dir path.")
    ap.add_argument("-p4", "--sample_submission_csv", type=str, required=True,
                    help="sample_submission.csv path.")
    ap.add_argument("-c_n", "--category_n", type=int, default=1,
                    help="分類クラス数。基本、領域検出だけするので1.")
    ap.add_argument("-i_w", "--input_width", type=int, default=512,
                    help="centernetの入力層の横幅サイズ.")
    ap.add_argument("-i_h", "--input_height", type=int, default=512,
                    help="centernetの入力層の縦幅サイズ.")
    ap.add_argument("-lr_s1", "--lr_step1_model", type=float, default=0.0005,
                    help="step1モデルの学習率.")
    ap.add_argument("-epoch_s1", "--n_epoch_step1_model", type=int, default=30,
                    help="step1モデルのエポック数.")
    ap.add_argument("-batch_s1", "--batch_size_step1_model", type=int, default=8,
                    help="step1モデルのバッチサイズ.")
    args = vars(ap.parse_args())

    # グローバル変数に再代入
    global category_n, input_width, input_height, output_layer_n, output_height,output_width
    category_n = args['category_n']
    input_width, input_height = args['input_width'], args['input_height']
    output_layer_n = category_n+4
    output_height,output_width = input_width//4,input_height//4










if __name__ == '__main__':
    print('centernet.py: loaded as script file')
else:
    print('centernet.py: loaded as module file')
