# -*- coding: utf-8 -*-
"""
train/validation/testデータ管理クラス
Tox21の画像のパス(train_files,validation_files,test_files)とラベル(y_train,y_valid,y_test)からtrain/validation/test の ImageDataGenerator まで作成する
ImageDataGeneratorのオプションはIDG_optionsで指定する

ラベルに-1が混じってるからMixup使うと、ラベルも混ぜるのでlossがマイナスになる
今回はMixupつかえない

Usage:
import os, sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
path = os.path.join(current_dir, '../')
sys.path.append(path)
from transformer import get_train_valid_test

shape = [299, 299, 3]
batch_size = 16
data_manager_cls = get_train_valid_test.LabeledDataset(shape, batch_size)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import image
#from keras.preprocessing.image import ImageDataGenerator # Githubのkeras-preprocessingを使う
import sklearn

# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得

# githubのmixupをimport
# /home/tmp10014/jupyterhub/notebook/other/lib_DL/mixup-generator
sys.path.append( str(current_dir) + '/../Git/mixup-generator' )
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser

sys.path.append( str(current_dir) + '/../Git/keras-preprocessing' )
from keras_preprocessing.image import ImageDataGenerator

# 自作モジュールimport
from . import my_generator

def generator_12output(generator):
    """
    12の出力（task）をだすgenerator
    # https://github.com/keras-team/keras/issues/5036
    Args:
        generator:keras.preprocessing.image.ImageDataGenerator
    Returns:
        入力画素(X), 12taskのラベルのリスト([y[:,0], y[:,1], y[:,2]…]) のgenerator
    """
    while True:
        X_y = generator.__next__() # python3 系ではnext() ではなく__next__()
        X = X_y[0]
        y = X_y[1]
        # 入力画素(X1[0]), 12taskのラベルのリスト([X1[0], X2[1], X3[1]]…)
        y_conv = []
        for i in range(y.shape[1]):
            y_conv.append(y[:,i])
        yield X, y_conv
        #yield X, [y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11]]

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


def imblearn_under_over_sampling(X: np.ndarray, y: np.ndarray, dict_ratio, is_over_sampling_type=True, random_state=71, is_plot=True):
    """ imblearnでunder/over_sampling（各クラスのサンプル数を全クラスの最小/最大数になるまで減らす/増やす）
    Arges:
        X: 説明変数（numpy型の画像パス:np.array[a.img,b.img,…]や、numpy型の画像データ:[[0.2554,0.59285,…][…]]みたいなの）
        y: 目的変数（numpy型のクラスidのラベル。np.array[4,0,1,2,…]みたいなの）
        dict_ratio: {0:200, 1:300, 2:500, …}のようにクラスid:クラスidのサンプル数の辞書
                    RandomUnderSampler: 各クラスのサンプル数が一番数が少ないクラスの数になる（この例だと{0:200, 1:200, 2:200, …}になる）
                    RandomOverSampler:  各クラスのサンプル数が一番数が多いクラスの数になる（この例だと{0:500, 1:500, 2:500, …}になる
        is_over_sampling_type: TrueならRandomOverSampler. FalseならRandomUnderSampler
        random_state: under/over_samplingでつかう乱数シード
        is_plot: Trueならunder/over_sampling後の各クラスの分布を棒グラフで可視化する
    Returns:
        under_sampling後のX, y
    Usage:
        dict_ratio = {0:pd.Series(y_train[:,0]).value_counts()[1.0], 1:pd.Series(y_train[:,0]).value_counts()[1.0]}
        X_resampled, y_resampled = imblearn_under_over_sampling(np.array(train_files), y_train, dict_ratio, is_over_sampling_type=False, is_plot=True)
    """
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler

    def _imblearn_sampling(X, y, dict_ratio, is_over_sampling_type, random_state, is_plot):
        if is_over_sampling_type == True:
            imblearn_func = RandomOverSampler
        else:
            imblearn_func = RandomUnderSampler
        ros = imblearn_func(ratio = dict_ratio, random_state=random_state)
        X_resampled, y_resampled = ros.fit_sample(pd.DataFrame(X), y)

        if is_plot == True:
            print('X.shape y.shape:', X.shape, y.shape)
            print('X_resampled.shape y_resampled.shape:', X_resampled.shape, y_resampled.shape)
            #print(pd.Series(X_resampled[:,0]).value_counts().head())
            if is_over_sampling_type == True:
                count = pd.Series(y_resampled).value_counts()
            else:
                count = pd.Series(y_resampled[:,0]).value_counts()
            print('y_resampled.value_counts():')
            print(count)
            count.plot.bar()
            plt.title(u'y_resampled.value_counts()')

        return X_resampled, y_resampled

    return _imblearn_sampling(X, y, dict_ratio, is_over_sampling_type, random_state, is_plot)


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

    def load_train_as_image(self, train_files, y_train):
        """train画像をnumpy.arrayに変換"""
        self.X_train = np.array([LabeledDataset.load_image(filename, self.shape, rescale_factor=1/255.)
                                    for filename in train_files])
        self.y_train = y_train
        #return self.X_train, self.y_train

    def load_validation_as_image(self, validation_files, y_valid):
        """validation画像をnumpy.arrayに変換"""
        self.X_valid = np.array([LabeledDataset.load_image(filename, self.shape, rescale_factor=1/255.)
                                for filename in validation_files])
        self.y_valid = y_valid
        #return self.X_valid, self.y_valid

    def load_test_as_image(self, test_files, y_test):
        """test画像をnumpy.arrayに変換"""
        self.X_test = np.array([LabeledDataset.load_image(filename, self.shape, rescale_factor=1/255.)
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
        else:
            self.valid_gen = None

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
        else:
            self.test_gen = None

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

    def train_steps_per_epoch(self):
        """fit_generatorで指定するtrainのsteps_per_epoch"""
        return len(self.X_train) // self.train_batch_size

    def valid_steps_per_epoch(self):
        """fit_generatorで指定するvalidationのsteps_per_epoch"""
        return len(self.X_valid) // self.valid_batch_size


def load_one_img(img_file_path, img_rows, img_cols, is_grayscale=False):
    """画像を1枚読み込んで、4次元テンソル(1, img_rows, img_cols, 3)へ変換+前処理"""
    img = image.load_img(img_file_path, target_size=(img_rows, img_cols))# 画像ロード
    x = image.img_to_array(img)# ロードした画像をarray型に変換
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
        cv_count (int): 分割数。
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
    print('y.shape:', y.shape)
    for ii in range(len(y)):
        print(f'y[{ii}]:', y[ii])
        plt.imshow(x[ii])
        plt.grid(False)
        plt.show()

if __name__ == '__main__':
    print('get_train_valid_test.py: loaded as script file')
else:
    print('get_train_valid_test.py: loaded as module file')
