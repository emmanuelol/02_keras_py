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
# C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\kaggle_kuzushiji-recognition\CenterNet -Keypoint Detector\
# test_CenterNet -Keypoint Detector.ipynb
# test_centernet.py.ipynb

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

import sys
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw
import argparse
import keras
import pathlib

sys.path.append(r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py')
from transformer import my_image
# グレースケール化関数
def_gray_aug = my_image.Compose( [my_image.RandomCompose([my_image.ToGrayScale(p=1)])] )

# グローバル変数
category_n = 1 # 分類クラス数
input_width,input_height = 512,512 # Centernetの入力層のサイズ
output_layer_n = category_n+4
output_height,output_width = input_width//4,input_height//4 # Centernetの出力層のサイズ
base_detect_num_h, base_detect_num_w = 25, 25 # 検出できる最小領域サイズ
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

def fit_sizecheck_model(model, train_list, cv_list, n_epoch, batch_size, lr=0.005, out_dir=None, lrs_epoch=10):
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
        lrs_epoch:LearningRateSchedulerの切り替えepoch
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
    def lrs(epoch, lr=lr, lrs_epoch=lrs_epoch):
        lr = lr
        if epoch > lrs_epoch: lr = lr/5.#0.0001
        if epoch > lrs_epoch*2: lr = lr/5.
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

def get_train_w_split_predict_sizecheck_model(model, train_input_for_size_estimate, aspect_ratio_pic_all, annotation_list_train):
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

def create_centernet(input_height, input_width, out_dir=None, step1_hdf5_path=None, hdf5_path=None, is_summary=True):
    """
    2stepのEncoder+DecoderのCenternet
    """
    K.clear_session()
    model = create_model(input_shape=(input_height,input_width,3), size_detection_mode=False)
    if out_dir is not None:
        model.save(out_dir + '/CenterNet_model.h5', include_optimizer=False)
    if step1_hdf5_path is not None:
        # step1のEncoderの重みあればload
        model.load_weights(step1_hdf5_path, by_name=True, skip_mismatch=True)
        print('step1_hdf5 load_weights')
    if hdf5_path is not None:
        # 重みファイルあればロード
        model.load_weights(hdf5_path)
        print('load_weights')
    if is_summary == True:
        print(model.summary())
    return model

def fit_centernet(model, train_list, cv_list, n_epoch, batch_size, lr=0.001, out_dir=None, lrs_epoch=20):
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
        lr:学習率
        out_dir:出力先ディレクトリ
    """
    cb = []
    def lrs(epoch, lr=lr, lrs_epoch=lrs_epoch):
        lr = lr
        if epoch >= lrs_epoch: lr = lr/20.#0.0002
        if epoch >= lrs_epoch*2: lr = lr/20.
        return lr
    cb.append( LearningRateScheduler(lrs) )
    if out_dir is not None:
        cb.append( ModelCheckpoint( os.path.join(out_dir, "final_weights_step2.hdf5")
                                    , monitor='val_loss', verbose=1
                                    , save_best_only=True, save_weights_only=True, period=1) )

    model.compile(loss=all_loss, optimizer=Adam(lr=lr), metrics=[heatmap_loss, size_loss, offset_loss])

    hist = model.fit_generator(
        Datagen_centernet(train_list,batch_size),
        steps_per_epoch = len(train_list) // batch_size,
        epochs = n_epoch,
        validation_data=Datagen_centernet(cv_list,batch_size),
        validation_steps = len(cv_list) // batch_size,
        callbacks = cb, #[lr_schedule],#early_stopping, reduce_lr, model_checkpoint],
        shuffle = True,
        verbose = 2#1
    )
    return hist

########################################## step3 Centernet Predict ##########################################

def NMS_all(predicts, category_n, score_thresh, iou_thresh):
    """
    Non Maximum Suppressionを予測結果全件に適応する
    Args:
        predicts:centernetの予測結果。[heatmap, category（今回は1クラスだけ）, xとy中心オフセット, 物体幅と高さ]
        category_n:分類クラス数
        score_thresh:予測した分類スコアのしきい値
        iou_thresh:NMSで使うIoUのしきい値
    """
    y_c=predicts[...,category_n]+np.arange(output_height).reshape(-1,1)
    x_c=predicts[...,category_n+1]+np.arange(output_width).reshape(1,-1)
    height=predicts[...,category_n+2]*output_height
    width=predicts[...,category_n+3]*output_width
    #print(y_c, x_c, height, width)

    count=0
    for category in range(category_n):
        predict=predicts[...,category]
        mask=(predict>score_thresh)
        #print("box_num",np.sum(mask))
        if mask.all==False:
            continue
        box_and_score=NMS(predict[mask],y_c[mask],x_c[mask],height[mask],width[mask],iou_thresh)
        box_and_score=np.insert(box_and_score,0,category,axis=1)#category,score,top,left,bottom,right
        if count==0:
            box_and_score_all=box_and_score
        else:
            box_and_score_all=np.concatenate((box_and_score_all,box_and_score),axis=0)
        count+=1
    score_sort=np.argsort(box_and_score_all[:,1])[::-1]
    box_and_score_all=box_and_score_all[score_sort]
    #print(box_and_score_all)

    _,unique_idx=np.unique(box_and_score_all[:,2],return_index=True)
    #print(unique_idx)
    return box_and_score_all[sorted(unique_idx)]

def NMS(score, y_c, x_c, height, width, iou_thresh, merge_mode=False):
    """
    Non Maximum Suppression:
    同じクラスとして認識された重なっている状態の領域を抑制するためのアルゴリズム。
    IoU値が大きければ、領域の重なりが大きいとみなして一方の領域を抑制（削除）している。IoU値が小さければ、領域の重なりが小さいとみなして両方の領域をそのままにしておく。
    ※IoU（Intersection over Union）:画像の重なりの割合を表す値。大きいほど、画像が重なっている。
    R-CNNで使ってる。（候補領域を切り出す→NNで候補領域から特徴ベクトルを抽出→SVMでクラスラベルか判断→NMSで重複領域抑制）
    NMSのアルゴリズムの解説は下記サイト参照
    http://meideru.com/archives/3538
    Args:
        score:クラススコア
        y_c:起点のy座標
        x_c:起点のx座標
        height:幅
        width:高さ
        iou_thresh:NMSで使うIoUのしきい値
        merge_mode:認識領域をNMSで抑制するかのフラグ。FalseならNMS適応。Trueなら認識領域そのまま出す
    """
    if merge_mode:
        score=score
        top=y_c
        left=x_c
        bottom=height
        right=width
    else:
        #flatten
        score=score.reshape(-1)
        y_c=y_c.reshape(-1)
        x_c=x_c.reshape(-1)
        height=height.reshape(-1)
        width=width.reshape(-1)
        size=height*width

        top=y_c-height/2
        left=x_c-width/2
        bottom=y_c+height/2
        right=x_c+width/2

        inside_pic=(top>0)*(left>0)*(bottom<output_height)*(right<output_width)
        outside_pic=len(inside_pic)-np.sum(inside_pic)
        #if outside_pic>0:
        #    print("{} boxes are out of picture".format(outside_pic))
        normal_size=(size<(np.mean(size)*10))*(size>(np.mean(size)/10))
        score=score[inside_pic*normal_size]
        top=top[inside_pic*normal_size]
        left=left[inside_pic*normal_size]
        bottom=bottom[inside_pic*normal_size]
        right=right[inside_pic*normal_size]
    #sort
    score_sort=np.argsort(score)[::-1]
    score=score[score_sort]
    top=top[score_sort]
    left=left[score_sort]
    bottom=bottom[score_sort]
    right=right[score_sort]

    area=((bottom-top)*(right-left))

    boxes=np.concatenate((score.reshape(-1,1),top.reshape(-1,1),left.reshape(-1,1),bottom.reshape(-1,1),right.reshape(-1,1)),axis=1)

    box_idx=np.arange(len(top))
    alive_box=[]
    while len(box_idx)>0:

        alive_box.append(box_idx[0])

        y1=np.maximum(top[0],top)
        x1=np.maximum(left[0],left)
        y2=np.minimum(bottom[0],bottom)
        x2=np.minimum(right[0],right)

        cross_h=np.maximum(0,y2-y1)
        cross_w=np.maximum(0,x2-x1)
        still_alive=(((cross_h*cross_w)/area[0])<iou_thresh)
        if np.sum(still_alive)==len(box_idx):
            print("error")
            print(np.max((cross_h*cross_w)),area[0])
        top=top[still_alive]
        left=left[still_alive]
        bottom=bottom[still_alive]
        right=right[still_alive]
        area=area[still_alive]
        box_idx=box_idx[still_alive]
    return boxes[alive_box]#score,top,left,bottom,right

def draw_rectangle(box_and_score, img, color, max_rect=500):
    """
    画像に枠線を書く
    Args:
        box_and_score:4隅の枠線データとスコア
        img:numpyの画像
        color:枠線の色
        max_rect:枠線最大数
    """
    number_of_rect=np.minimum(max_rect, len(box_and_score))

    for i in reversed(list(range(number_of_rect))):
        top, left, bottom, right = box_and_score[i,:]

        top = np.floor(top + 0.5).astype('int32')
        left = np.floor(left + 0.5).astype('int32')
        bottom = np.floor(bottom + 0.5).astype('int32')
        right = np.floor(right + 0.5).astype('int32')
        #label = '{} {:.2f}'.format(predicted_class, score)
        #print(label)
        #rectangle=np.array([[left,top],[left,bottom],[right,bottom],[right,top]])

        draw = ImageDraw.Draw(img)
        #label_size = draw.textsize(label)
        #print(label_size)

        #if top - label_size[1] >= 0:
        #  text_origin = np.array([left, top - label_size[1]])
        #else:
        #  text_origin = np.array([left, top + 1])

        thickness=4
        if color=="red":
            rect_color=(255, 0, 0)
        elif color=="blue":
            rect_color=(0, 0, 255)
        else:
            rect_color=(0, 0, 0)

        if i==0:
            thickness=4
        for j in range(2*thickness):#薄いから何重にか描く
            draw.rectangle([left + j, top + j, right - j, bottom - j],
                        outline=rect_color)
        #draw.rectangle(
        #            [tuple(text_origin), tuple(text_origin + label_size)],
        #            fill=(0, 0, 255))
        #draw.text(text_origin, label, fill=(0, 0, 0))
    del draw
    return img

def check_iou_score(true_boxes, detected_boxes, iou_thresh):
    """
    IoUの値を確認する
    Args:
        true_boxes:正解の4隅の枠線データ
        detected_boxes:予測した4隅の枠線データ
        iou_thresh:Iocのしきい値
    """
    iou_all=[]
    for detected_box in detected_boxes:
        y1=np.maximum(detected_box[0],true_boxes[:,0])
        x1=np.maximum(detected_box[1],true_boxes[:,1])
        y2=np.minimum(detected_box[2],true_boxes[:,2])
        x2=np.minimum(detected_box[3],true_boxes[:,3])

        cross_section=np.maximum(0,y2-y1)*np.maximum(0,x2-x1)
        all_area=(detected_box[2]-detected_box[0])*(detected_box[3]-detected_box[1])+(true_boxes[:,2]-true_boxes[:,0])*(true_boxes[:,3]-true_boxes[:,1])

        iou=np.max(cross_section/(all_area-cross_section))

        #if iou_thresh < iou:
        #argmax=np.argmax(cross_section/(all_area-cross_section))
        iou_all.append(iou)

    # IoU計算
    score=2*np.sum(iou_all)/(len(detected_boxes)+len(true_boxes))
    print("IoU score:{}".format(np.round(score,3)))
    return score

def show_predict_centernet(model, cv_list, show_count=5, score_thresh=0.3, iou_thresh=0.4, out_dir=None):
    """
    centernetでshow_count枚数予測して、予測画像を可視化する
    Args:
        model:centernet model
        cv_list:validationデータリスト。[[画像path,log(文字サイズ÷ピクチャサイズ) の値], …]。例. [['valid_images/200021712-00022_2.jpg', -6.966739922505915], …]
        show_count:可視化する画像の枚数
        score_thresh:予測した分類スコアのしきい値
        iou_thresh:NMSで使うIoUのしきい値
        out_dir:出力先ディレクトリ
    """
    for i in np.arange(0, show_count):
        img = Image.open(cv_list[i][0]).convert("RGB") # validation setの画像でテスト
        width,height = img.size
        X = (np.asarray(img.resize((input_width, input_height))).reshape(1, input_height, input_width, 3))/255
        predict = model.predict(X).reshape(output_height, output_width, (category_n+4))

        box_and_score = NMS_all(predict, category_n, score_thresh=score_thresh, iou_thresh=iou_thresh)

        #print("after NMS",len(box_and_score))
        if len(box_and_score) == 0:
            continue

        true_boxes = cv_list[i][1][:,1:]#c_x,c_y,width_height
        top = true_boxes[:,1:2]-true_boxes[:,3:4]/2
        left = true_boxes[:,0:1]-true_boxes[:,2:3]/2
        bottom = top+true_boxes[:,3:4]
        right = left+true_boxes[:,2:3]
        true_boxes = np.concatenate((top,left,bottom,right),axis=1)

        heatmap = predict[:,:,0]

        print_w, print_h = img.size
        #resize predocted box to original size
        box_and_score = box_and_score*[1,1,print_h/output_height,print_w/output_width,print_h/output_height,print_w/output_width]
        check_iou_score(true_boxes,box_and_score[:,2:],iou_thresh=0.5)
        img = draw_rectangle(box_and_score[:,2:],img,"red") # 予測の枠を赤色にする
        img = draw_rectangle(true_boxes,img,"blue") # 正解の枠を青色にする

        fig, axes = plt.subplots(1, 2,figsize=(15,15))
        #axes[0].set_axis_off()
        axes[0].imshow(img)
        #axes[1].set_axis_off()
        axes[1].imshow(heatmap)#, cmap='gray')
        #axes[2].set_axis_off()
        #axes[2].imshow(heatmap_1)#, cmap='gray')
        if out_dir is not None:
            plt.savefig( os.path.join(out_dir, pathlib.Path(cv_list[i][0]).name), bbox_inches="tight" )
        plt.show()

def split_and_detect(model, img, height_split_recommended, width_split_recommended, score_thresh, iou_thresh, maxlap=0.5):
    """
    画像分割してから検出する
    """
    width,height=img.size
    height_split=int(-(-height_split_recommended//1)+1)
    width_split=int(-(-width_split_recommended//1)+1)
    height_lap=(height_split-height_split_recommended)/(height_split-1)
    height_lap=np.minimum(maxlap,height_lap)
    width_lap=(width_split-width_split_recommended)/(width_split-1)
    width_lap=np.minimum(maxlap,width_lap)

    # 分割する画像のリスト作成
    if height>width:
        crop_size=int((height)/(height_split-(height_split-1)*height_lap))#crop_height and width
        if crop_size>=width:
            crop_size=width
            stride=int((crop_size*height_split-height)/(height_split-1))
            top_list=[i*stride for i in range(height_split-1)]+[height-crop_size]
            left_list=[0]
        else:
            stride=int((crop_size*height_split-height)/(height_split-1))
            top_list=[i*stride for i in range(height_split-1)]+[height-crop_size]
            width_split=-(-width//crop_size)
            stride=int((crop_size*width_split-width)/(width_split-1))
            left_list=[i*stride for i in range(width_split-1)]+[width-crop_size]

    else:
        crop_size=int((width)/(width_split-(width_split-1)*width_lap))#crop_height and width
        if crop_size>=height:
            crop_size=height
            stride=int((crop_size*width_split-width)/(width_split-1))
            left_list=[i*stride for i in range(width_split-1)]+[width-crop_size]
            top_list=[0]
        else:
            stride=int((crop_size*width_split-width)/(width_split-1))
            left_list=[i*stride for i in range(width_split-1)]+[width-crop_size]
            height_split=-(-height//crop_size)
            stride=int((crop_size*height_split-height)/(height_split-1))
            top_list=[i*stride for i in range(height_split-1)]+[height-crop_size]
    #print('\ntop_list left_list', top_list, left_list)
    #print('len(top_list) len(left_list)', len(top_list), len(left_list))
    count=0

    for top_offset in top_list:
        for left_offset in left_list:
            img_crop = img.crop((left_offset, top_offset, left_offset+crop_size, top_offset+crop_size)) # 画像分割
            #print('img_crop.shape', np.asarray(img_crop).shape)
            predict=model.predict((np.asarray(img_crop.resize((input_width,input_height))).reshape(1,input_height,input_width,3))/255).reshape(output_height,output_width,(category_n+4))

            box_and_score = NMS_all(predict,category_n,score_thresh,iou_thresh)#category,score,top,left,bottom,right

            #print("after NMS",len(box_and_score))
            if len(box_and_score)==0:
                continue
            #reshape and offset
            box_and_score = box_and_score*[1,1,crop_size/output_height,crop_size/output_width,crop_size/output_height,crop_size/output_width]+np.array([0,0,top_offset,left_offset,top_offset,left_offset])

            if count==0:
                box_and_score_all=box_and_score
            else:
                box_and_score_all=np.concatenate((box_and_score_all,box_and_score),axis=0)
            count+=1
    #print("all_box_num:",len(box_and_score_all))
    #print(box_and_score_all[:10,:],np.min(box_and_score_all[:,2:]))
    if count==0:
        box_and_score_all=[]
    else:
        score=box_and_score_all[:,1]
        y_c=(box_and_score_all[:,2]+box_and_score_all[:,4])/2
        x_c=(box_and_score_all[:,3]+box_and_score_all[:,5])/2
        height=-box_and_score_all[:,2]+box_and_score_all[:,4]
        width=-box_and_score_all[:,3]+box_and_score_all[:,5]
        #print(np.min(height),np.min(width))
        box_and_score_all = NMS(box_and_score_all[:,1],box_and_score_all[:,2],box_and_score_all[:,3],box_and_score_all[:,4],box_and_score_all[:,5], iou_thresh=0.5, merge_mode=True)
    return box_and_score_all

def show_predict_centernet_split(model, cv_list, show_count=5, score_thresh=0.3, iou_thresh=0.4, out_dir=None):
    """
    split_and_detect()画像分割し、centernetでshow_count枚数予測して、予測画像を可視化する
    Args:
        model:centernet model
        cv_list:validationデータリスト。[[画像path,log(文字サイズ÷ピクチャサイズ) の値], …]。例. [['valid_images/200021712-00022_2.jpg', -6.966739922505915], …]
        show_count:可視化する画像の枚数
        score_thresh:予測した分類スコアのしきい値
        iou_thresh:NMSで使うIoUのしきい値
        out_dir:出力先ディレクトリ
    """
    all_iou_score=[]
    for i in np.arange(0, show_count):
        img = Image.open(cv_list[i][0]).convert("RGB")

        # 画像分割してから検出する
        box_and_score_all = split_and_detect(model, img, cv_list[i][2], cv_list[i][3], score_thresh, iou_thresh)
        if len(box_and_score_all)==0:
            print("no box found")
            continue

        # 正解の枠
        true_boxes = cv_list[i][1][:,1:]#c_x,c_y,width_height
        top = true_boxes[:,1:2]-true_boxes[:,3:4]/2
        left = true_boxes[:,0:1]-true_boxes[:,2:3]/2
        bottom = top+true_boxes[:,3:4]
        right = left+true_boxes[:,2:3]
        true_boxes = np.concatenate((top,left,bottom,right),axis=1)

        print_w, print_h = img.size
        iou_score = check_iou_score(true_boxes, box_and_score_all[:,1:], iou_thresh=0.5)
        all_iou_score.append(iou_score)

        img = draw_rectangle(box_and_score_all[:,1:],img,"red")# 予測の枠を赤色にする
        img = draw_rectangle(true_boxes,img,"blue") # 正解の枠を青色にする

        #fig, axes = plt.subplots(1, 2,figsize=(15,15))
        #axes[0].set_axis_off()
        #axes[0].imshow(img)
        #axes[1].set_axis_off()
        #axes[1].imshow(heatmap)#, cmap='gray')
        fig, axes = plt.subplots(1, 1, figsize=(10,10))
        axes.imshow(img)
        if out_dir is not None:
            plt.savefig( os.path.join(out_dir, pathlib.Path(cv_list[i][0]).name), bbox_inches="tight" )
        plt.show()
    print("average_score:", np.mean(all_iou_score))

########################################## step4 Predict Pipeline ##########################################

def predict_pipeline(model_1, model_2, model_3
                     , i, id_test
                     , aspect_ratio_pic_all_test
                     , class_dict_cat
                     , class_width=32, class_height=32
                     , is_class_predict_gray=True
                     , score_thresh=0.3, iou_thresh=0.4
                     , out_dir=None
                     , print_img=False
                     , is_save_predict_img=False
                    ):
    """
    文字と画像のサイズ比を推定する回帰モデル（model_1）でtest setのサイズ取得、
    centernet（model_2）で領域検出して、
    分類モデル（model_3）でクラス分類する
    Args:
        model_1:文字と画像のサイズ比を推定する回帰モデル
        model_2:centernet
        model_3:分類モデル
        i:test setの画像パスリストのid
        id_test:test setの画像パスリスト
        aspect_ratio_pic_all_test:test setの画像の縦横比リスト
        class_dict_cat:分類モデルのラベルidとラベル名の辞書
        class_width, class_height:分類モデルの入力層のサイズ
        is_class_predict_gray:分類モデルの入力画像をグレーにするか
        score_thresh:予測した分類スコアのしきい値
        iou_thresh:NMSで使うIoUのしきい値
        out_dir:出力先ディレクトリ
        print_img:print文やplotを表示するか
        is_save_predict_img:検出画像保存するか
    """
    # model1: determine how to split image
    if print_img: print("model 1")
    img = np.asarray(Image.open(id_test[i]).resize((input_width,input_height)).convert('RGB'))
    predicted_size = model_1.predict(img.reshape(1,input_width,input_height,3)/255)
    detect_num_h = aspect_ratio_pic_all_test[i]*np.exp(-predicted_size/2)
    detect_num_w = detect_num_h/aspect_ratio_pic_all_test[i]
    h_split_recommend = np.maximum(1,detect_num_h/base_detect_num_h)
    w_split_recommend = np.maximum(1,detect_num_w/base_detect_num_w)
    if print_img: print("recommended split_h:{}, split_w:{}".format(h_split_recommend,w_split_recommend))

    # model2: detection
    if print_img: print("model 2")
    img = Image.open(id_test[i]).convert("RGB")
    box_and_score_all = split_and_detect(model_2, img, h_split_recommend, w_split_recommend, score_thresh, iou_thresh)#output:score,top,left,bottom,right
    if print_img: print("find {} boxes".format(len(box_and_score_all)))
    print_w, print_h = img.size
    if (len(box_and_score_all)>0) and print_img:
        img = draw_rectangle(box_and_score_all[:,1:],img,"red")
        #plt.imshow(img)
        fig, axes = plt.subplots(1, 1, figsize=(10,10))
        axes.imshow(img)
        plt.show()
    # 検出画像保存するか
    if (len(box_and_score_all)>0) and is_save_predict_img:
        img = draw_rectangle(box_and_score_all[:,1:],img,"red")
        fig, axes = plt.subplots(1, 1, figsize=(10,10))
        axes.imshow(img)
        plt.savefig( os.path.join(out_dir, pathlib.Path(id_test[i]).name), bbox_inches="tight" )

    # model3: classification
    count=0
    if (len(box_and_score_all)>0):
        for box in box_and_score_all[:,1:]:
            top,left,bottom,right = box
            img_letter = img.crop((int(left),int(top),int(right),int(bottom))).resize((class_width,class_height))#大き目のピクセルのがいいか？

            if is_class_predict_gray == True:
                # グレー画像で学習したからグレーにしておく
                x = def_gray_aug(image=np.asarray(img_letter))["image"].astype(np.float32)
            else:
                x = np.asarray(img_letter).astype(np.float32)

            predict = (model_3.predict(x.reshape(1, class_width, class_height, 3)/255))
            predict = np.argmax(predict,axis=1)[0]
            code = class_dict_cat[str(predict)]
            c_x = int((left+right)/2)
            c_y = int((top+bottom)/2)
            if count==0:
                ans=code+" "+str(c_x)+" "+str(c_y)
            else:
                ans=ans+" "+code+" "+str(c_x)+" "+str(c_y)
            count+=1
            if (len(box_and_score_all)>0) and print_img:
                #print(x_gray.shape)
                plt.imshow(x/255)
                plt.show()
                print(code)

    else:
        ans=""
    return ans

def kuzushiji_model_pipeline(model_dir, class_h5=r'D:\work\kaggle_kuzushiji-recognition\work\classes\20190816\best_val_acc.h5'):
    """
    くずし字コンペ用のpredict_pipeline()に渡すモデルロード関数
    """
    K.clear_session()
    print("loading models...")
    # 文字と画像のサイズ比を推定する回帰モデル
    model_1 = create_model(input_shape=(input_width, input_height, 3),size_detection_mode=True)
    model_1.load_weights(model_dir+'/final_weights_step1.hdf5')
    # centernet
    model_2 = create_model(input_shape=(input_width, input_height, 3),size_detection_mode=False)
    model_2.load_weights(model_dir+'/final_weights_step2.hdf5')
    # 分類モデル
    sys.path.append( r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py\Git\OctConv-TFKeras' )
    from oct_conv2d import OctConv2D
    import keras
    model_3 = keras.models.load_model(class_h5
                                      , custom_objects={'OctConv2D': OctConv2D} # OctConvは独自レイヤーだからcustom_objects の指定必要
                                      , compile=False)
    return model_1, model_2, model_3

########################################## step0 Data Preprocessing ##########################################

def convert_input_data(path_1, path_2, path_3, path_4):
    """
    kaggleのくずし字コンペのCenterNet用に入力データを形成
    Usage:
        data_dir = r'D:\work\kaggle_kuzushiji-recognition\OrigData\kuzushiji-recognition'
        path_1 = data_dir+"/train.csv"
        path_2 = data_dir+"/train_images/"
        path_3 = data_dir+"/test_images/"
        path_4 = data_dir+"/sample_submission.csv"

        df_train, category_names, inv_dict_cat, annotation_list_train, id_test, df_submission = \
        centernet.convert_input_data(path_1, path_2, path_3, path_4)
    """
    df_train=pd.read_csv(path_1)
    df_train = df_train.dropna(axis=0, how='any')#you can use nan data(page with no letter)
    df_train = df_train.reset_index(drop=True)
    print('df_train:')
    #display(df_train.head())
    print(df_train.head()) # batから叩くとdisplayエラーになるので

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


def kuzushiji_main():
    # コマンドプロンプトからだとファイルパスが読めずうまく動かない。。。
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
    ap.add_argument("-lr_s2", "--lr_step2_model", type=float, default=0.001,
                    help="step2モデルの学習率.")
    ap.add_argument("-epoch_s2", "--n_epoch_step2_model", type=int, default=50,
                    help="step2モデルのエポック数.")
    ap.add_argument("-batch_s2", "--batch_size_step2_model", type=int, default=32,
                    help="step2モデルのバッチサイズ.")
    ap.add_argument("--class_h5", type=str, default=r'D:\work\kaggle_kuzushiji-recognition\work\classes\20190816\best_val_acc.h5',
                    help="分類モデルのh5ファイルパス.")
    ap.add_argument("--class_w", type=int, default=32,
                    help="分類モデルの入力層の横幅サイズ.")
    ap.add_argument("--class_h", type=int, default=32,
                    help="分類モデルの入力層の縦幅サイズ.")
    args = vars(ap.parse_args())

    # グローバル変数に再代入
    global category_n, input_width, input_height, output_layer_n, output_height, output_width
    category_n = args['category_n']
    input_width, input_height = args['input_width'], args['input_height']
    output_layer_n = category_n+4
    output_height,output_width = input_width//4,input_height//4

    # kaggleのくずし字コンペのCenterNet用に入力データを形成
    df_train, category_names, inv_dict_cat, annotation_list_train, id_test, df_submission = \
    convert_input_data(args['train_csv'], args['train_images'], args['test_images'], args['sample_submission_csv'])

    print('STEP 0: Preprocessing (Check Object Size)')
    aspect_ratio_pic_all, aspect_ratio_pic_all_test, average_letter_size_all, train_input_for_size_estimate = \
    preprocess_check_object_size(annotation_list_train, id_test, out_dir=args['output_dir'])

    print('\nSTEP 1: Create Encoder Model')
    model = centernet.create_sizecheck_model(input_height, input_width, out_dir=args['output_dir'])
    # train/validation set作成
    train_list, cv_list = train_test_split(train_input_for_size_estimate, random_state=111, test_size=0.2)
    # Training
    hist = fit_sizecheck_model(model, train_list, cv_list
                                         , args['n_epoch_step1_model']
                                         , args['batch_size_step1_model']
                                         , lr=args['lr_step1_model']
                                         , out_dir=args['output_dir'])
    model.load_weights(out_dir+'/final_weights_step1.hdf5')
    # Result
    plot_predict_sizecheck_model(model, cv_list, args['batch_size_step1_model'], out_dir=args['output_dir'])

    # train setについてのlog(文字サイズ÷ピクチャサイズ) の値を推定する回帰モデルの予測結果を取得し、
    # step2のモデルで使うために、検出できるのはせいぜい25x25くらいだと考えて、画像データの分割数を適当に決める
    annotation_list_train_w_split = get_train_w_split_predict_sizecheck_model(model, train_input_for_size_estimate, aspect_ratio_pic_all, annotation_list_train)

    print('\nSTEP 2: Create Centernet Model')
    model = create_centernet(input_height, input_width, step1_hdf5_path=args['output_dir']+'/final_weights_step1.hdf5', out_dir=args['output_dir'])
    # train/validation set作成
    train_list, cv_list = train_test_split(annotation_list_train_w_split, random_state=111, test_size=0.2)#stratified split is better
    # Training
    hist = centernet.fit_centernet(model, train_list, cv_list
                                         , args['n_epoch_step2_model']
                                         , args['batch_size_step2_model']
                                         , lr=args['lr_step2_model']
                                         , out_dir=args['output_dir'])
    model.load_weights(args['output_dir']+'/final_weights_step2.hdf5')

    print('\nSTEP 3: Predict Pipeline')
    # model load
    model_1, model_2, model_3 = centernet.kuzushiji_model_pipeline(args['output_dir'], class_h5=args['class_h5'])

    # 分類モデルのidとunicodeの対応表ロード
    df_class = pd.read_csv(r'D:\work\kaggle_kuzushiji-recognition\work\classes\20190816\tfAPI_dict_class.tsv', sep='\t')
    print(df_class.head())
    class_inv_dict_cat = {}
    for index, series in df_class.iterrows():
        class_inv_dict_cat[str(series['classes_ids'])] = series['unicode']

    df_submission = pd.read_csv(args['sample_submission_csv'])
    # test set全件predict
    for i in tqdm(range(len(id_test))):
        ans = centernet.predict_pipeline(model_1, model_2, model_3
                                         , i, id_test
                                         , aspect_ratio_pic_all_test
                                         , class_inv_dict_cat
                                         , class_width=args['class_w'], class_height=args['class_h']
                                         , is_class_predict_gray=True
                                         , score_thresh=0.3, iou_thresh=0.4
                                         , out_dir=args['output_dir']
                                         , print_img=False
                                         #, print_img=True
                                         , is_save_predict_img=False
                                         #, is_save_predict_img=True
                                        )
        df_submission.set_value(i, 'labels', ans)
    df_submission.to_csv(args['output_dir']+"/submission.csv", index=False)

if __name__ == '__main__':
    kuzushiji_main()
else:
    print('centernet.py: loaded as module file')
