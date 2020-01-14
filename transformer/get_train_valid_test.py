# -*- coding: utf-8 -*-
"""
train/validation/testデータ用意する

Usage:
    import os, sys
    current_dir = os.path.dirname(os.path.abspath("__file__"))
    path = os.path.join(current_dir, '../')
    sys.path.append(path)
    from transformer import get_train_valid_test

    shape = [299, 299, 3]
    batch_size = 16
    # データクラス定義
    data_manager_cls = get_train_valid_test.LabeledDataset(shape, batch_size)
"""
import os, sys, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
# 自作モジュールimport
from transformer import my_generator, augmentor_util

# githubのmixupをimport
# /home/tmp10014/jupyterhub/notebook/other/lib_DL/mixup-generator
sys.path.append( str(current_dir) + '/../Git/mixup-generator' )
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser

import keras
#from keras.preprocessing.image import ImageDataGenerator # Githubのkeras-preprocessingを使う
sys.path.append( str(current_dir) + '/../Git/keras-preprocessing' )
from keras_preprocessing.image import ImageDataGenerator


def binary_generator_multi_output_wrapper(generator):
    """
    binaryラベルのgeneratorをマルチタスクgeneratorに変換するラッパー
    マルチラベルをマルチタスクに変換するときに使う
    https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24
    Args:
        generator:keras.preprocessing.image.ImageDataGenerator
    Returns:
        ラベルのshapeが(クラス数, バッチサイズ)になったgenerator
    """
    for batch_x,batch_y in generator:
        #print(batch_y.shape)
        # ラベルのshapeを[クラス数,画像の数]に変換する
        if batch_y.ndim == 1:
            yield (batch_x,batch_y.reshape(1,batch_y.shape[0])) # batch_y[0]からbatch_y[:,0]が画像1枚のラベルになる
        else:
            yield (batch_x,[batch_y[:,i] for i in range(batch_y.shape[1])]) # batch_y[0]からbatch_y[:,0]が画像1枚のラベルになる

def generator_multi_output_wrapper(generator, Xs, task_labels:list, batch_size:int, is_shuffle=True, seed=7):
    """
    flow前のgeneratorをマルチタスクgeneratorに変換するラッパー
    batch単位ではなく全画像データをメモリにロードする
    Args:
        generator:keras.preprocessing.image.ImageDataGenerator
        Xs:前処理済み画像データ
        task_labels:各タスクのone-hotラベルを詰めたリスト
        batch_size:バッチサイズ
        is_shuffle:画像シャッフルするか。trainの時はTrueにしないとだめ
        seed:flowの乱数シード
    Returns:
        ラベルがlist[<task数>, ndarray(バッチサイズ, <出力層のnode>]になったgenerator
    """
    # すべてのgenerator.flowで同じseedにしないと、各genで違うXが生成される
    # https://stackoverflow.com/questions/38972380/keras-how-to-use-fit-generator-with-multiple-outputs-of-different-type
    gens = [generator.flow(x=Xs, y=labels, batch_size=batch_size, shuffle=is_shuffle, seed=seed) \
            for labels in task_labels]
    while True:
        _X = None
        ys = []
        for gen in gens:
            X_y = gen.__next__() # python3 系ではnext() ではなく__next__()
            ys.append(X_y[1])
            ##### 各genのXが同じかのチェック #####
            #if _X is None:
            #    _X = X_y[0]
            #else:
            #    if (_X == X_y[0]).all():
            #        print('### X match ###')
            #    else:
            #        print('### X not match ###')
            ######################################
        yield X_y[0], ys

def generator_decode_wrapper(generator):
    """
    generatorのone-hotラベルをidにデコードするラッパー
    one-hotラベルをid番号で扱いたいときに使う(Augmentorのkeras_generatorはラベルを必ずone-hotに変換するので)
    Args:
        generator:keras.preprocessing.image.ImageDataGenerator
    Returns:
        one-hotラベルをidにデコードしたgenerator([0,1,0]→1になる)
    """
    for batch_x,batch_y in generator:
        yield (batch_x,np.argmax(batch_y, axis=1).astype(np.float32))

### Dataset distribution utility
# https://github.com/daisukelab/small_book_image_dataset/blob/master/IF201812%20-Train%20With%20Augmentation.ipynb
def get_class_distribution(y):
    # y_cls can be one of [OH label, index of class, class label name]
    # convert OH to index of class
    y_cls = [np.argmax(one) for one in y] if len(np.array(y).shape) == 2 else y
    # y_cls can be one of [index of class, class label name]
    classset = sorted(list(set(y_cls)))
    sample_distribution = {cur_cls:len([one for one in y_cls if one == cur_cls]) for cur_cls in classset}
    return sample_distribution

def get_class_distribution_list(y, num_classes):
    dist = get_class_distribution(y)
    assert(y[0].__class__ != str) # class index or class OH label only
    list_dist = np.zeros((num_classes))
    for i in range(num_classes):
        if i in dist:
            list_dist[i] = dist[i]
    return list_dist

def balance_class_by_over_sampling(X, y): # Naive: all sample has equal weights
    from imblearn.over_sampling import RandomOverSampler
    Xidx = [[xidx] for xidx in range(len(X))]
    y_cls = [np.argmax(one) for one in y]
    classset = sorted(list(set(y_cls)))
    sample_distribution = [len([one for one in y_cls if one == cur_cls]) for cur_cls in classset]
    nsamples = np.max(sample_distribution)
    flat_ratio = {cls:nsamples for cls in classset}
    Xidx_resampled, y_cls_resampled = RandomOverSampler(ratio=flat_ratio, random_state=42).fit_sample(Xidx, y_cls)
    sampled_index = [idx[0] for idx in Xidx_resampled]
    return np.array([X[idx] for idx in sampled_index]), np.array([y[idx] for idx in sampled_index])


def get_dict_class_counts(label_array: np.ndarray, is_maximize_all_class=False, is_minimize_all_class=False):
    """
    one-hot前のラベルからクラスidとそのクラスidの総数の辞書を返す
    Args:
        label_array:one-hot前のラベル。np.array([1,1,1,3,2,1]) や np.array(['a','a','a','c','b','a']) みたいなの
        is_maximize_all_class:全てのクラスで数が一番多いのにするか。一番多いクラスの数が4なら、dict_class_id_countsが{1: 4, 2: 4, 3: 4}みたいになる
        is_minimize_all_class:全てのクラスで数が一番少ないのにするか。一番少ないクラスの数が1なら、dict_class_id_countsが{1: 1, 2: 1 3: 1}みたいになる
    Returns:
        dict_class_id_counts:クラスidとそのクラスidの総数の辞書。{1: 4, 2: 1, 3: 1} や {'a': 4, 'b': 1, 'c': 1} みたいなの
    """
    class_unique, class_counts = np.unique(label_array, return_counts=True)
    dict_class_counts = {}

    # 全てのクラスで数が一番多いのにするか
    if is_maximize_all_class == True:
        max_count = class_counts.max()
        class_counts = np.array([max_count for c in class_counts])

    # 全てのクラスで数が一番少ないのにするか
    if is_minimize_all_class == True:
        min_count = class_counts.min()
        class_counts = np.array([min_count for c in class_counts])

    for i in range(len(class_counts)):
        dict_class_counts[class_unique[i]] = class_counts[i]
    return dict_class_counts

def imblearn_under_over_sampling(X: np.ndarray, y: np.ndarray, dict_class_counts, mode="OVER", random_state=71, is_plot=True):
    """ imblearnでunder/over_sampling（各クラスのサンプル数を全クラスの最小/最大数になるまで減らす/増やす）
    Arges:
        X: 説明変数（numpy型の画像パス:np.array[a.img,b.img,…]や、numpy型の画像データ:[[0.2554,0.59285,…][…]]みたいなの）
        y: 目的変数（numpy型のクラスidのラベル。np.array[4,0,1,2,…]みたいなの）
        dict_class_counts: {0:200, 1:300, 2:500, …}のようにクラスid:クラスidのサンプル数の辞書.クラスidのサンプル数になるようにsamplingする
        mode: "OVER"ならRandomOverSampler. "UNDER"ならRandomUnderSampler. "SMOTE"ならSMOTE
        random_state: under/over_samplingでつかう乱数シード
        is_plot: Trueならunder/over_sampling後の各クラスの分布を棒グラフで可視化する
    Returns:
        under_sampling後のX, y
    Usage:
        dict_class_counts = get_train_valid_test.get_dict_class_counts(y, is_maximize_all_class=True) # one-hot前のラベルからクラスidとそのクラスidの総数の辞書を返す
        X_resampled, y_resampled = imblearn_under_over_sampling(np.array(train_files), y_train, dict_class_counts, is_over_sampling_type=False, is_plot=True)
    """
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.over_sampling import SMOTE

    # Xが四次元ベクトルの画像データの場合
    X_img = None
    if len(X.shape) == 4:
        X_img = copy.deepcopy(X) # 参照でなく複製
        X = np.array([xidx for xidx in range(len(X))])# Xを画像idにする
        #print(X)

    def _sampling(dict_class_counts, mode, random_state):
        if mode == "OVER":
            sample = RandomOverSampler(ratio=dict_class_counts, random_state=random_state)
        if mode == "UNDER":
            sample = RandomUnderSampler(ratio=dict_class_counts, random_state=random_state)
        # SMOTEはうまくいかず
        #if mode == "SMOTE":
        #    sample = SMOTE(ratio=dict_class_counts, random_state=random_state)
        return sample

    def _imblearn_sampling(X, y, dict_class_counts, mode, random_state, is_plot):
        sample = _sampling(dict_class_counts, mode, random_state)
        X_resampled, y_resampled = sample.fit_sample(pd.DataFrame(X), y)

        if is_plot == True:
            print('X.shape y.shape:', X.shape, y.shape)
            count = pd.Series(y_resampled).value_counts()
            print('y_resampled.value_counts():')
            print(count)
            count.plot.bar()
            plt.title(u'y_resampled.value_counts()')
            plt.show()

        return X_resampled, y_resampled

    X_resampled, y_resampled = _imblearn_sampling(X, y, dict_class_counts, mode, random_state, is_plot)

    # Xが四次元ベクトルの画像データの場合
    if X_img is not None:
        #print(X_resampled) # X_resampledが画像idになっている
        X_resampled = np.array([X_img[i][0] for i in X_resampled])

    print('X_resampled.shape y_resampled.shape:', X_resampled.shape, y_resampled.shape)
    return X_resampled, y_resampled


### Dataset management class
class LabeledDataset:
    """
    データ管理用クラス
    cpuのメモリに全画像データを持つのでcpuのメモリ少ない場合は使えないかも
    """
    image_suffix = ['.jpg', '.JPG', '.png', '.PNG']

    def __init__(self, shape, train_batch_size, valid_batch_size=1, test_batch_size=1
                , train_samples=0 # train set の画像枚数
                , valid_samples=0 # validation set の画像枚数
                , test_samples=0 # test set の画像枚数
                ):
        self.shape = shape
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.X_train, self.y_train, self.train_gen = None, None, None
        self.X_valid, self.y_valid, self.valid_gen = None, None, None
        self.X_test, self.y_test, self.test_gen = None, None, None
        self.init_train_steps_per_epoch, self.init_valid_steps_per_epoch, self.init_test_steps_per_epoch = None, None, None
        # 画像総数指定あればsteps_per_epoch計算する
        if train_samples != 0:
            self.init_train_steps_per_epoch = train_samples // self.train_batch_size
            print('train_steps_per_epoch :', self.init_train_steps_per_epoch)
        if valid_samples != 0:
            self.init_valid_steps_per_epoch = valid_samples // self.valid_batch_size
            print('valid_steps_per_epoch :', self.init_valid_steps_per_epoch)
        if test_samples != 0:
            self.init_test_steps_per_epoch = test_batch_size // self.test_batch_size
            print('test_steps_per_epoch :', self.init_test_steps_per_epoch)

    def split_train_valid(self, test_size=0.2, random_state=42):
        """train データ（self.X_train, self.y_train）を8:2でtest データに分割"""
        from sklearn.model_selection import train_test_split
        self.cur_X_train, self.cur_X_valid, self.cur_y_train, self.cur_y_valid = train_test_split(
            self.X_train,
            self.y_train,
            test_size=test_size,
            random_state=random_state)
        self.cur_X_train, self.cur_y_train = balance_class_by_over_sampling(self.cur_X_train, self.cur_y_train)

    def load_image(filename, shape, rescale_factor):
        """サイズ指定して画像を読み込んでnumpy.arrayに変換"""
        img = keras.preprocessing.image.load_img(filename, target_size=shape[:2])
        return keras.preprocessing.image.img_to_array(img) * rescale_factor

    def load_train_as_image(self, train_files, y_train, rescale_factor=1/255.):
        """train画像をnumpy.arrayに変換"""
        self.X_train = np.array([LabeledDataset.load_image(filename, self.shape, rescale_factor=rescale_factor)
                                    for filename in train_files])
        self.y_train = y_train
        #return self.X_train, self.y_train

    def load_validation_as_image(self, validation_files, y_valid, rescale_factor=1/255.):
        """validation画像をnumpy.arrayに変換"""
        self.X_valid = np.array([LabeledDataset.load_image(filename, self.shape, rescale_factor=rescale_factor)
                                for filename in validation_files])
        self.y_valid = y_valid
        #return self.X_valid, self.y_valid

    def load_test_as_image(self, test_files, y_test, rescale_factor=1/255.):
        """test画像をnumpy.arrayに変換"""
        self.X_test = np.array([LabeledDataset.load_image(filename, self.shape, rescale_factor=rescale_factor)
                                for filename in test_files])
        self.y_test = y_test
        #return self.X_test, self.y_test

    def create_test_generator(self, IDG_options={}):
        """testのImageDataGenerator作成"""
        test_datagen = ImageDataGenerator(**IDG_options)
        #y_test_dummy = to_categorical([0 for _ in range(len(self.X_test))])
        self.test_gen = test_datagen.flow(self.X_test, self.y_test, #, y_test_dummy,
                                          batch_size=len(self.X_test), shuffle=False) ########## self.batch_size
        return self.test_gen

    def create_generator(self, use_mixup=False, IDG_options={}):
        """trainとvalidationのImageDataGenerator作成"""
        print('----- train_ImageDataGenerator -----')
        print('use_mixup:', use_mixup)
        print('IDG_options:', IDG_options)
        aug_datagen = ImageDataGenerator(**IDG_options)
        if use_mixup:
            # ラベルに-1が混じってるからMixup使うと、ラベルも混ぜるのでlossがマイナスになる
            # 今回はMixupつかえない
            self.train_gen = MixupGenerator(self.X_train, self.y_train, alpha=1.0, batch_size=self.train_batch_size, datagen=aug_datagen)()
        else:
            self.train_gen = aug_datagen.flow(self.X_train, self.y_train, batch_size=self.train_batch_size)

        plain_datagen = ImageDataGenerator()
        self.valid_gen = plain_datagen.flow(self.X_valid, self.y_valid, batch_size=self.valid_batch_size, shuffle=False)

        return self.train_gen, self.valid_gen

    def create_my_generator_flow(self, my_IDG_options={}):
        """
        my_generator.MyImageDataGeneratorクラスからflow()で
        train,validationのGenerator作成
        """
        print('my_IDG_options:', my_IDG_options)

        # 訓練画像水増し（Data Augmentation）
        train_datagen = my_generator.MyImageDataGenerator(**my_IDG_options)
        self.train_gen = train_datagen.flow(self.X_train, self.y_train, batch_size=self.train_batch_size)

        plain_datagen = ImageDataGenerator(rescale=my_IDG_options['rescale'])
        self.valid_gen = plain_datagen.flow(self.X_valid, self.y_valid, batch_size=self.valid_batch_size, shuffle=False)

        return self.train_gen, self.valid_gen

    def create_my_generator_flow_from_directory(self
                                                , train_data_dir, classes
                                                , valid_data_dir=None, test_data_dir=None
                                                , color_mode='rgb', class_mode='categorical'
                                                , my_IDG_options={}
                                                , is_valid_grayscale=False # validation generatorをグレースケール化するか
                                                ):
        """
        my_generator.MyImageDataGeneratorクラスからflow_from_directory()で
        train,validation,test setのGenerator作成
        """
        print('my_IDG_options:', my_IDG_options)

        # 訓練画像水増し（Data Augmentation）
        train_datagen = my_generator.MyImageDataGenerator(**my_IDG_options)
        self.train_gen = train_datagen.flow_from_directory(
            train_data_dir, # ラベルクラスをディレクトリ名にした画像ディレクトリのパス
            target_size=(self.shape[0], self.shape[1]), # すべての画像はこのサイズにリサイズ
            color_mode=color_mode,# 画像にカラーチャンネルが3つある場合は「rgb」画像が白黒またはグレースケールの場合は「grayscale」
            classes=classes, # 分類クラス名リスト
            class_mode=class_mode, # 2値分類は「binary」、多クラス分類は「categorical」
            batch_size=self.train_batch_size, # バッチごとにジェネレータから生成される画像の数
            shuffle=True # 生成されているイメージの順序をシャッフルする場合は「True」を設定し、それ以外の場合は「False」。train set は基本入れ替える
        )

        if valid_data_dir is not None:
            # 検証画像_前処理実行
            valid_datagen = my_generator.get_datagen(rescale=my_IDG_options['rescale'], is_grayscale=is_valid_grayscale)
            self.valid_gen = valid_datagen.flow_from_directory(
                valid_data_dir,
                target_size=(self.shape[0], self.shape[1]),
                color_mode=color_mode,
                classes=classes,
                class_mode=class_mode,
                batch_size=self.valid_batch_size,# batch_size はセット内の画像の総数を正確に割るような数に設定しないと同じ画像を2回使うため、validation やtest setのbatch size は割り切れる数にすること！！！
                shuffle=False# validation/test set は基本順番入れ替えない
            )

        if test_data_dir is not None:
            # テスト画像_前処理実行
            test_datagen = my_generator.get_datagen(rescale=my_IDG_options['rescale'])
            self.test_gen = test_datagen.flow_from_directory(
                test_data_dir,
                target_size=(self.shape[0], self.shape[1]),
                color_mode=color_mode,
                classes=classes,
                class_mode=class_mode,
                batch_size=self.test_batch_size,# batch_size はセット内の画像の総数を正確に割るような数に設定しないと同じ画像を2回使うため、validation やtest setのbatch size は割り切れる数にすること！！！
                shuffle=False# validation/test set は基本順番入れ替えない
            )

        return self.train_gen, self.valid_gen, self.test_gen

    def create_my_generator_flow_from_dataframe(self
                                                , x_col, y_col
                                                , train_df
                                                , train_data_dir=None
                                                , valid_df=None
                                                , valid_data_dir=None
                                                , classes=None
                                                , validation_split=0.0
                                                , color_mode='rgb', class_mode='categorical'
                                                , seed=42
                                                , my_IDG_options={}
                                                , is_valid_grayscale=False # validation generatorをグレースケール化するか
                                                #, valid_IDG_options={'rescale':1.0/255.0}
                                                ):
        """
        my_generator.MyImageDataGeneratorクラスからflow_from_dataframe()で
        train,validation,test setのGenerator作成
        x_col, y_col はtrain_dfのファイル名列
        x_col列は、引数の train_data_dir で画像ディレクトリ指定している場合は画像ファイル名(*.jpg)だけで良い
        y_col列はラベル列。
        class_mode='categorical'ならone-hotに変換してくれるのでラベルはone-hot前でよい
        回帰で使う場合はclass_mode='raw'にすること。ラベルの数値そのまま返してくれる
        """
        print('my_IDG_options:', my_IDG_options)

        if validation_split > 0.0:
            # 訓練画像水増し（Data Augmentation）
            train_datagen = my_generator.MyImageDataGenerator(**my_IDG_options, validation_split=validation_split)
        else:
            train_datagen = my_generator.MyImageDataGenerator(**my_IDG_options)

        if validation_split > 0.0:
            # validation_splitでtrainとvalidationつくると、validationにもtarinと同じ水増しが実行されるのでつかえない！！！
            self.train_gen = train_datagen.flow_from_dataframe(
                train_df,
                directory=train_data_dir, # ラベルクラスをディレクトリ名にした画像ディレクトリのパス
                x_col=x_col,
                y_col=y_col,
                subset="training",
                target_size=(self.shape[0], self.shape[1]), # すべての画像はこのサイズにリサイズ
                color_mode=color_mode,# 画像にカラーチャンネルが3つある場合は「rgb」画像が白黒またはグレースケールの場合は「grayscale」
                classes=classes, # 分類クラス名リスト
                class_mode=class_mode, # 2値分類は「binary」、多クラス分類は「categorical」
                batch_size=self.train_batch_size, # バッチごとにジェネレータから生成される画像の数
                seed=seed, # 乱数シード
                shuffle=True # 生成されているイメージの順序をシャッフルする場合は「True」を設定し、それ以外の場合は「False」。train set は基本入れ替える
            )

            self.valid_gen = train_datagen.flow_from_dataframe(
                train_df,
                directory=train_data_dir, # ラベルクラスをディレクトリ名にした画像ディレクトリのパス
                x_col=x_col,
                y_col=y_col,
                subset="validation",
                target_size=(self.shape[0], self.shape[1]), # すべての画像はこのサイズにリサイズ
                color_mode=color_mode,# 画像にカラーチャンネルが3つある場合は「rgb」画像が白黒またはグレースケールの場合は「grayscale」
                classes=classes, # 分類クラス名リスト
                class_mode=class_mode, # 2値分類は「binary」、多クラス分類は「categorical」
                batch_size=self.valid_batch_size, # バッチごとにジェネレータから生成される画像の数
                seed=seed, # 乱数シード
                shuffle=False # 生成されているイメージの順序をシャッフルする場合は「True」を設定し、それ以外の場合は「False」。train set は基本入れ替える
            )
        elif valid_df is not None:
            self.train_gen = train_datagen.flow_from_dataframe(
                train_df,
                directory=train_data_dir, # ラベルクラスをディレクトリ名にした画像ディレクトリのパス
                x_col=x_col,
                y_col=y_col,
                target_size=(self.shape[0], self.shape[1]), # すべての画像はこのサイズにリサイズ
                color_mode=color_mode,# 画像にカラーチャンネルが3つある場合は「rgb」画像が白黒またはグレースケールの場合は「grayscale」
                classes=classes, # 分類クラス名リスト
                class_mode=class_mode, # 2値分類は「binary」、多クラス分類は「categorical」
                batch_size=self.train_batch_size, # バッチごとにジェネレータから生成される画像の数
                seed=seed, # 乱数シード
                shuffle=True # 生成されているイメージの順序をシャッフルする場合は「True」を設定し、それ以外の場合は「False」。train set は基本入れ替える
            )

            valid_datagen = my_generator.get_datagen(rescale=my_IDG_options['rescale'], is_grayscale=is_valid_grayscale)
            #valid_datagen = my_generator.MyImageDataGenerator(**valid_IDG_options) # MyImageDataGeneratorでgenerator作るとfilenameなどが属性に持たなくなるので駄目
            self.valid_gen = valid_datagen.flow_from_dataframe(
                valid_df,
                directory=valid_data_dir, # ラベルクラスをディレクトリ名にした画像ディレクトリのパス
                x_col=x_col,
                y_col=y_col,
                target_size=(self.shape[0], self.shape[1]), # すべての画像はこのサイズにリサイズ
                color_mode=color_mode,# 画像にカラーチャンネルが3つある場合は「rgb」画像が白黒またはグレースケールの場合は「grayscale」
                classes=classes, # 分類クラス名リスト
                class_mode=class_mode, # 2値分類は「binary」、多クラス分類は「categorical」
                batch_size=self.valid_batch_size, # バッチごとにジェネレータから生成される画像の数
                seed=seed, # 乱数シード
                shuffle=False # 生成されているイメージの順序をシャッフルする場合は「True」を設定し、それ以外の場合は「False」。train set は基本入れ替える
            )
        else:
            self.train_gen = train_datagen.flow_from_dataframe(
                train_df,
                directory=train_data_dir, # ラベルクラスをディレクトリ名にした画像ディレクトリのパス
                x_col=x_col,
                y_col=y_col,
                target_size=(self.shape[0], self.shape[1]), # すべての画像はこのサイズにリサイズ
                color_mode=color_mode,# 画像にカラーチャンネルが3つある場合は「rgb」画像が白黒またはグレースケールの場合は「grayscale」
                classes=classes, # 分類クラス名リスト
                class_mode=class_mode, # 2値分類は「binary」、多クラス分類は「categorical」
                batch_size=self.train_batch_size, # バッチごとにジェネレータから生成される画像の数
                seed=seed, # 乱数シード
                shuffle=True # 生成されているイメージの順序をシャッフルする場合は「True」を設定し、それ以外の場合は「False」。train set は基本入れ替える
            )

            self.valid_gen = None
        return self.train_gen, self.valid_gen

    def create_augmentor_util_from_directory(self, data_dir, batch_size
                                            , output_dir='/Augmentor_output/'
                                            , scaled=True
                                            , augmentor_options={}):
        """
        AugmentorからGenerator作成
        """
        gen = augmentor_util.make_datagenerator_from_dir(
                data_dir, batch_size
                , output_dir=output_dir
                , scaled=scaled
                , IDG_options=augmentor_options)
        return gen

    def train_steps_per_epoch(self):
        """fit_generatorで指定するtrainのsteps_per_epoch"""
        return len(self.X_train) // self.train_batch_size

    def valid_steps_per_epoch(self):
        """fit_generatorで指定するvalidationのsteps_per_epoch"""
        return len(self.X_valid) // self.valid_batch_size


def load_one_img(img_file_path, img_rows, img_cols, is_grayscale=False):
    """画像を1枚読み込んで、4次元テンソル(1, img_rows, img_cols, 3)へ変換+前処理"""
    img = keras.preprocessing.image.load_img(img_file_path, target_size=(img_rows, img_cols))# 画像ロード
    x = keras.preprocessing.image.img_to_array(img)# ロードした画像をarray型に変換
    if is_grayscale == True:
        x = my_generator.preprocessing_grayscale(x)
    x = np.expand_dims(x, axis=0)# 4次元テンソルへ変換
    x = x.astype('float32')
    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！ これを忘れると結果がおかしくなるので注意
    X = x / 255.0# 前処理
    return X

def get_folds(X, y, cv_count, split_seed, stratify=None):
    """Cross validationのインデックスを返す。
    https://github.com/ak110/pytoolkit/blob/5f663d324c965726dab6ae97097552c723cf03a5/pytoolkit/ml.py
    Args:
        X: 入力データ。
        y: 出力データ。
        cv_count (int): 分割数。cv_count=5なら train:valid=8:2 のcvが5つできる
        split_seed (int): 乱数のseed。
        stratify (bool or None): StratifiedKFold (CVをする際に、ラベルの割合が揃うにtrainデータとtestデータを分けてくれてるもの) にするならTrue。
    Returns:
        list of tuple(train_indices, val_indices): インデックス
    Usage:
        cv_count = 5
        folds = get_train_valid_test.get_folds(X, y, 5, split_seed=42, stratify=None)
        print('folds:', len(folds))
        X_train_folds = []
        X_test_folds = []
        y_train_folds = []
        y_test_folds = []
        for i in range(cv_count):
            train_index, test_index = folds[i][0], folds[i][1]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # ラベルonehot化
            y_train = keras.utils.to_categorical(y_train, len(class_name))
            y_test = keras.utils.to_categorical(y_test, len(class_name))
            X_train_folds.append(X_train)
            X_test_folds.append(X_test)
            y_train_folds.append(y_train)
            y_test_folds.append(y_test)
            print(X_train.shape, X_test.shape)
            print(y_train.shape, y_test.shape)
    """
    import sklearn.model_selection

    if stratify is None:
        stratify = isinstance(y, np.ndarray) and len(y.shape) == 1
    cv = (
        sklearn.model_selection.StratifiedKFold
        if stratify
        else sklearn.model_selection.KFold
    )
    cv = cv(cv_count, shuffle=True, random_state=split_seed)
    folds = list(cv.split(X, y))
    return folds

def print_image_generator(gen, i=0):
    """
    ImageDataGeneratorの1batdh分画像とラベルをprintで確認する
    Arges:
        gen: flow済みのImageDataGeneratorのインスタンス。d_cls.train_genとか
        i: batchのid
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x,y = next(gen)
    print('x.shape:', x.shape)
    print(f'x[{i}]:\n', x[i])
    print('np.max(x):', np.max(x))
    if isinstance(y, list):
        # マルチタスクの場合
        print('len(y):', len(y))
        for ii in range(y[0].shape[0]):
            y_ii = [y_i[ii] for y_i in y]
            print(f'y[{ii}]:', y_ii)
            plt.imshow(x[ii])
            plt.grid(False)
            plt.show()
    else:
        # シングルタスク/マルチラベルの場合
        print('y.shape:', y.shape)
        for ii in range(len(y)):
            print(f'y[{ii}]:', y[ii])
            plt.imshow(x[ii])
            plt.grid(False)
            plt.show()

def label_smoothing_generator(data_generator, smooth_factor=0.1, mask_value=-1.0, is_multi_class=True):
    """
    Imagedatagenerator用label_smoothing
    label_smoothing：分類問題の正解ラベルの1を0.9みたいに下げ、0ラベルを0.1みたいに上げ、過学習軽減させる正則化手法。
    間違っている正解ラベルが混じっているデータセットのときに有効
    tensorflow.kerasなら以下のコードでもlabel_smoothing可能(https://www.pyimagesearch.com/2019/12/30/label-smoothing-with-keras-tensorflow-and-deep-learning/ より)
        from tensorflow.keras.losses import CategoricalCrossentropy
        loss = CategoricalCrossentropy(label_smoothing=0.1)
        model.compile(loss=loss, optimizer='sgd', metrics=["accuracy"])
    Args:
        data_generator:flow済みImagedatageneratorのインスタンス。d_cls.train_genとか
        smooth_factor:label_smoothingで下げる割合。smooth_factor=0.1で、5クラス[0,0,1,0,0]なら[0.02,0.02,0.92,0.02,0.02]になる
        mask_value:欠損値ラベル。label_smoothingでラベル値がマイナスになる場合、mask_valueに置き換える
        is_multi_class:マルチクラス分類のフラグ。Falseの場合、smooth_factor=0.1で、5クラス[0,0,1,0,0]なら[0.00,0.00,0.90,0.00,0.00]になる
                       マルチクラス分類の場合softmaxで合計ラベル=1になるが、multiラベルはそうではないので、ラベル値加算したくない時用
    Returns:
        Imagedatageneratorインスタンス（yはlabel_smoothing済み）
    """
    def _smooth_labels(y_i, smooth_factor, mask_value, is_multi_class):
        y_i = y_i.astype('float64') # int型だとエラーになるのでfloatに変換
        y_i *= 1 - smooth_factor # ラベル値減らす
        # ラベル値加算するか(マルチクラス分類の場合softmaxで合計ラベル=1になるが、multiラベルはそうではないので)
        if is_multi_class == True:
            y_i += smooth_factor / y_i.shape[0]
        y_i = np.where(y_i < 0.0, mask_value, y_i) # 負の値になったらマスク値に置換する
        return y_i

    for x, y in data_generator:
        smooth_y = np.empty(y.shape, dtype=np.float) # yは上書きできないので同じ大きさの空配列用意
        for i,y_i in enumerate(y):
            smooth_y[i] = _smooth_labels(y_i, smooth_factor, mask_value, is_multi_class)
        yield x, smooth_y

def label2onehot(labels:np.ndarray):
    """
    sklearnでnp.ndarrayのラベルをonehot化
    Args:
        labels:onehot化するラベル名の配列.np.array(['high' 'high' 'low' 'low'])のようなの
    Returns:
        enc:ラベル名を0-nの連番にした配列。np.array([[0] [0] [1] [1]])のようなの
        onehot:ラベル名をonehotにした配列。np.array([[1. 0.] [1. 0.] [0. 1.] [0. 1.]])のようなの
    Usage:
        labels = df['aaa'].values
        enc, onehot = label2onehot(labels)
    """
    from sklearn import preprocessing
    from sklearn.preprocessing import OneHotEncoder

    enc = preprocessing.LabelEncoder().fit_transform(labels).reshape(-1,1)
    onehot = OneHotEncoder().fit_transform(enc).toarray()
    return enc, onehot
