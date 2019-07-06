# -*- coding: utf-8 -*-
"""
画像dataの一部を各setに分ける

Uses:
# クラス名のリスト
class_name = ['beagle', 'bikini', 'boke', 'cat', 'comic_book', 'fashion', 'marin', 'other', 'shingo', 'suit', 'tumblr']

# 全画像格納しているディレクトリ
source_dir = r'D:\work\keras_iPhone_pictures\InceptionResNetV2'

# train/validation/test set に分ける画像ディレクトリ
img_dir = r'D:\work\keras_iPhone_pictures\InceptionResNetV2_small_set\train_400'

train_1class_count = 400#100 #10 # train 1クラスの画像枚数
valid_1class_count = 40#10 #5 # validaiton 1クラスの画像枚数
test_1class_count = 10#10 #5  # test 1クラスの画像枚数

# クラスディレクトリごとに別れた指定ディレクトリの画像を tarin/val set に分けてコピーする
split_class_train_valid_test_set(class_name, source_dir, img_dir
                                 , train_count_org=train_1class_count
                                 , valid_count_org=valid_1class_count
                                 , test_count_org=test_1class_count
                                )
"""
import os, sys, shutil, glob
from tqdm import tqdm
import random

def split_class_train_valid_test_set(class_name_list, source_dir, img_dir
                                     , train_count_org=10, valid_count_org=5, test_count_org=5
                                     , train_percent=0.8, valid_percent=0.2
                                     , is_test_set=True
                                     , random_seed=42
                                     ):
    """
    クラスディレクトリごとに別れた指定ディレクトリの画像を tarin/val set に分けてコピーする
    クラスごとの画像枚数の指定（train_count_org, valid_count_org, test_count_org）がなければ
    train:0.8(train_percent), test:0.2(valid_percent) の割合で分けて
    （テストデータを分割した後の）トレーニングデータの10％(valid_percent//2.0)  をvalidに使用
    is_test_set=Falseならtest set 作らない(train:0.8(train_percent),valid:0.2(valid_percent)で分ける)
    random_seedは乱数シードで画像の並び順変えるための数値
    """
    print('source_dir :', source_dir)
    print('img_dir :', img_dir)
    print('---------------------------------------------------------------')
    # 書き込み進捗バーの書き込み先をsys.stdout(標準出力)指定しないと進捗バーが改行される
    # https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
    pbar = tqdm(class_name_list, file=sys.stdout)
    for class_name in pbar:
        pbar.set_description("Processing %s" % class_name) # tqdmの進捗バー
        #print('class_name :', class_name)

        # クラスごとの画像枚数初期化
        train_count = train_count_org
        valid_count = valid_count_org
        test_count = test_count_org

        # クラスごとの画像総数
        class_source_dir = os.path.join(source_dir, class_name)
        # ファイル名一応ソートしておく
        id_imgs = sorted(glob.glob(os.path.join(class_source_dir, '*jpg')))
        id_imgs_png = sorted(glob.glob(os.path.join(class_source_dir, '*png')))
        id_imgs.extend(id_imgs_png)
        id_imgs = sorted(id_imgs)

        # 順番ランダムシャッフル
        random.seed(random_seed)
        random.shuffle(id_imgs)

        print('imgs:', len(id_imgs))

        # 画像ディレクトリ作成
        train_dir = os.path.join(img_dir, 'train', class_name)
        valid_dir = os.path.join(img_dir, 'validation', class_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        if is_test_set == True:
            test_dir = os.path.join(img_dir, 'test', class_name)
            os.makedirs(test_dir, exist_ok=True)

        # train/validation/test set に分ける画像枚数の指定なし or クラス画像の総数の方が分ける画像数より多い 場合
        # train:0.8, test:0.2 の割合で分けて、（テストデータを分割した後の）トレーニングデータの10％  をvalidに使用
        if ( (train_count_org is None) or (train_count_org+valid_count_org+test_count_org > len(id_imgs)) ) and is_test_set == True:
            #test_count =  int(len(id_imgs)*0.05)
            #valid_count = int(len(id_imgs)*0.15)
            #train_count = len(id_imgs) - valid_count - test_count

            # 参考：https://www.pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/
            # compute the training and testing split
            i = int(len(id_imgs) * train_percent)
            trainPaths = id_imgs[:i]
            testPaths = id_imgs[i:]
            # we'll be using part of the training data for validation
            i = int(len(trainPaths) * valid_percent/2.0)
            valPaths = trainPaths[:i]
            trainPaths = trainPaths[i:]
            # define the datasets that we'll be building
            datasets = [
                ("training", trainPaths, train_dir),
                ("validation", valPaths, valid_dir),
                ("testing", testPaths, test_dir)]
            # loop over the datasets
            for (dType, imagePaths, baseOutput) in datasets:
                # loop over the input image paths
                for inputPath in imagePaths:
                    shutil.copy2(inputPath, os.path.join(baseOutput, os.path.basename(inputPath)))
        elif is_test_set == False:
            # train:train_percent, valid:1-train_percent の割合で分ける
            # compute the training and testing split
            i = int(len(id_imgs) * train_percent)
            trainPaths = id_imgs[:i]
            valPaths = id_imgs[i:]
            # define the datasets that we'll be building
            datasets = [
                ("training", trainPaths, train_dir),
                ("validation", valPaths, valid_dir)]
            # loop over the datasets
            for (dType, imagePaths, baseOutput) in datasets:
                # loop over the input image paths
                for inputPath in imagePaths:
                    shutil.copy2(inputPath, os.path.join(baseOutput, os.path.basename(inputPath)))
        else:
            # test img copy
            for img in id_imgs[0: test_count]:
                shutil.copyfile(img, os.path.join(test_dir, os.path.basename(img)))
            # validation img copy
            for img in id_imgs[test_count: valid_count+test_count]:
                shutil.copyfile(img, os.path.join(valid_dir, os.path.basename(img)))
            # train img copy
            for img in id_imgs[valid_count+test_count: train_count+valid_count+test_count]:
                shutil.copyfile(img, os.path.join(train_dir, os.path.basename(img)))

if __name__ == '__main__':
    print('set_split.py: loaded as script file')
else:
    print('set_split.py: loaded as module file')
