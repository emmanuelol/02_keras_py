"""
KerasのImageDataGeneratorのカスタムジェネレーター

Usage:
    from keras_preprocessing.image import ImageDataGenerator
    import tf_my_generator as my_generator

    classes = ['nega', 'posi']
    shape = [331,512,3]
    batch_size = 15

    gen = ImageDataGenerator(rescale=1.0/255.0, shear_range=20, zoom_range=[0.8,1.5])

    gen = gen.flow_from_directory(
        train_data_dir, # ラベルクラスをディレクトリ名にした画像ディレクトリのパス
        target_size=(shape[0], shape[1]), # すべての画像はこのサイズにリサイズ
        color_mode='rgb',# 画像にカラーチャンネルが3つある場合は「rgb」画像が白黒またはグレースケールの場合は「grayscale」
        classes=classes, # 分類クラスリスト
        class_mode='categorical', # 2値分類は「binary」(=ラベルidに変換する)、多クラス分類は「categorical」(=onehotラベルに変換する)
        batch_size=batch_size, # バッチごとにジェネレータから生成される画像の数
        shuffle=True # 生成されているイメージの順序をシャッフルする場合は「True」を設定し、それ以外の場合は「False」。train set は基本入れ替える
    )
    # gen = gen.flow(x_train, y_train, batch_size=batch_size)
    # custom_gen = my_generator.get_cifar10_best_train_generator(x_train, y_train, batch_size)

    custom_gen = my_generator.randaugment_generator(gen, N=3, M=4)
    custom_gen = my_generator.get_kuzushiji_generator(custom_gen)
    custom_gen = my_generator.gray_generator(custom_gen, p=0.5)
    custom_gen = my_generator.random_crop_generator(custom_gen, random_crop_size=[100,100])
    custom_gen = my_generator.random_erasing_generator(custom_gen, p=0.5)
    custom_gen = my_generator.ricap_generator(custom_gen) # ラベルいじるmixupやricapは最後にすること
    custom_gen = my_generator.mixup_generator(custom_gen) # ラベルいじるmixupやricapは最後にすること

    # もしくはMyImageDataGeneratorで複数の水増し一気に指定できる
    my_IDG_options = {'rescale':1.0/255.0,
                      'cutmix_alpha':0.5,
                      'randaugment_N':2,
                      'randaugment_M':3}
    custom_gen = my_generator.MyImageDataGenerator(**my_IDG_options)
"""
import os
import sys
import numpy as np
import pandas as pd
import imgaug
import albumentations
from PIL import Image
import threading

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator

import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得
sys.path.append( str(current_dir) + '/../' )
from transformer import ndimage, my_image

#sys.path.append(r'C:\Users\shingo\Git\randaugment')
#import Rand_Augment
from Git.randaugment import Rand_Augment


_gray_aug = my_image.Compose([my_image.RandomCompose([my_image.ToGrayScale(p=1)])])
def preprocessing_grayscale(x):
    """ ImageDataGeneratorのpreprocessing_functionで3次元画像をグレースケール化 """
    x = _gray_aug(image=x)["image"].astype(np.float32)
    return x


def get_datagen(rescale=1.0 / 255, is_grayscale=False):
    """画像_前処理実行"""
    if is_grayscale == True:
        datagen = ImageDataGenerator(rescale=rescale # 前処理：画像を0.0~1.0の範囲に変換
                                     , preprocessing_function=preprocessing_grayscale) # グレースケール化
    else:
        datagen = ImageDataGenerator(rescale=rescale) # 前処理：画像を0.0~1.0の範囲に変換
    return datagen


def random_crop(original_img, random_crop_size):
    """
    ランダムクロップ
    https://jkjung-avt.github.io/keras-image-cropping/
    """
    # Note: image_data_format is 'channel_last'
    assert original_img.shape[2] == 3
    if original_img.shape[0] < random_crop_size[0] or original_img.shape[1] < random_crop_size[1]:
        raise ValueError(f"Invalid random_crop_size : original = {original_img.shape}, crop_size = {random_crop_size}")
    height, width = original_img.shape[0], original_img.shape[1]
    #dy, dx = random_crop_size
    #x = np.random.randint(0, width - dx + 1)
    #y = np.random.randint(0, height - dy + 1)
    #x_crop = original_img[y:(y+dy), x:(x+dx), :]
    data = {'image': original_img}
    augmentation = albumentations.RandomCrop(random_crop_size[0], original_img.shape[1], p=0.5) # albumentationsでやってみる
    x_crop = augmentation(**data)['image']
    #x_crop_img = Image.fromarray(np.uint8(x_crop * 255.0)) # numpy->PIL
    #x_crop_img = x_crop_img.resize((width, height), Image.LANCZOS) # resize
    #return np.asarray(x_crop_img)/255.0 # PIL->numpy
    return ndimage.resize(x_crop, width, height) # x_cropの画像サイズはcropしたサイズなのでもとの画像サイズに戻す


def mix_up(X1, y1, X2, y2, mix_up_alpha=0.2):
    """
    Mix-up
    https://qiita.com/yu4u/items/70aa007346ec73b7ff05
    """
    assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
    batch_size = X1.shape[0]
    l = np.random.beta(mix_up_alpha, mix_up_alpha, batch_size)
    X_l = l.reshape(batch_size, 1, 1, 1)
    y_l = l.reshape(batch_size, 1)
    X = X1 * X_l + X2 * (1 - X_l)
    y = y1 * y_l + y2 * (1 - y_l)
    return X, y


def random_erasing(x, p=0.5, s=(0.02, 0.4), r=(0.3, 3), max_pic=255):
    """
    Random Erasing（真っ黒に塗りつぶすから、正確にはcutout）
    https://www.kumilog.net/entry/numpy-data-augmentation#Random-Erasing
    """
    image = np.copy(x)
    # マスクするかしないか
    if np.random.rand() > p:
        return image
    # マスクする画素値をランダムで決める
    mask_value = np.random.randint(0, max_pic)
    h, w, _ = image.shape
    # マスクのサイズを元画像のs(0.02~0.4)倍の範囲からランダムに決める
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])
    # マスクのアスペクト比をr(0.3~3)の範囲からランダムに決める
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]
    # マスクのサイズとアスペクト比からマスクの高さと幅を決める
    # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正する
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1
    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    image[top:bottom, left:right, :].fill(mask_value)
    return image


def ricap(image_batch, label_batch, beta=0.3, use_same_random_value_on_batch=True):
    """
    RICAP（Data Augmentation using Random Image Cropping and Patching for Deep CNNs）:
    4枚の画像をクリッピングして画像合成するData Augmentation
    https://qiita.com/koshian2/items/1a6b93ee5724a6d63730
    Arges:
        image_batch: generatorから生成される1バッチのnumpy画像データ
        label_batch: generatorから生成される1バッチの画像のラベルデータ
        beta: 画像合成する大きさ
        use_same_random_value_on_batch:
            use_same_random_value_on_batch=Trueとすれば論文と同じように「ミニバッチ間で共通の乱数を使う例」となります
            また、この値をFalseにすれば、「サンプル間で別々の乱数を使う例」となります
    """
    # if use_same_random_value_on_batch = True : same as the original papaer
    assert image_batch.shape[0] == label_batch.shape[0]
    assert image_batch.ndim == 4
    batch_size, image_y, image_x = image_batch.shape[:3]
    # crop_size w, h from beta distribution
    if use_same_random_value_on_batch:
        w_dash = np.random.beta(beta, beta) * np.ones(batch_size)
        h_dash = np.random.beta(beta, beta) * np.ones(batch_size)
    else:
        w_dash = np.random.beta(beta, beta, size=(batch_size))
        h_dash = np.random.beta(beta, beta, size=(batch_size))
    w = np.round(w_dash * image_x).astype(np.int32)
    h = np.round(h_dash * image_y).astype(np.int32)
    # outputs
    output_images = np.zeros(image_batch.shape)
    output_labels = np.zeros(label_batch.shape)

    def create_masks(start_xs, start_ys, end_xs, end_ys):
        mask_x = np.logical_and(np.arange(image_x).reshape(1,1,-1,1) >= start_xs.reshape(-1,1,1,1),
                                np.arange(image_x).reshape(1,1,-1,1) < end_xs.reshape(-1,1,1,1))
        mask_y = np.logical_and(np.arange(image_y).reshape(1,-1,1,1) >= start_ys.reshape(-1,1,1,1),
                                np.arange(image_y).reshape(1,-1,1,1) < end_ys.reshape(-1,1,1,1))
        mask = np.logical_and(mask_y, mask_x)
        mask = np.logical_and(mask, np.repeat(True, image_batch.shape[3]).reshape(1,1,1,-1))
        return mask

    def crop_concatenate(wk, hk, start_x, start_y, end_x, end_y):
        nonlocal output_images, output_labels
        xk = (np.random.rand(batch_size) * (image_x-wk)).astype(np.int32)
        yk = (np.random.rand(batch_size) * (image_y-hk)).astype(np.int32)
        target_indices = np.arange(batch_size)
        np.random.shuffle(target_indices)
        weights = wk * hk / image_x / image_y
        dest_mask = create_masks(start_x, start_y, end_x, end_y)
        target_mask = create_masks(xk, yk, xk+wk, yk+hk)
        output_images[dest_mask] = image_batch[target_indices][target_mask]
        output_labels += weights.reshape(-1, 1) * label_batch[target_indices]

    # left-top crop
    crop_concatenate(w, h,
                     np.repeat(0, batch_size), np.repeat(0, batch_size),
                     w, h)
    # right-top crop
    crop_concatenate(image_x-w, h,
                     w, np.repeat(0, batch_size),
                     np.repeat(image_x, batch_size), h)
    # left-bottom crop
    crop_concatenate(w, image_y-h,
                     np.repeat(0, batch_size), h,
                     w, np.repeat(image_y, batch_size))
    # right-bottom crop
    crop_concatenate(image_x-w, image_y-h,
                     w, h, np.repeat(image_x, batch_size),
                     np.repeat(image_y, batch_size))
    return output_images, output_labels


def cutmix(X1, y1, X2, y2, cutmix_alpha=1.0):
    """
    CutMix
    https://www.kaggle.com/code1110/mixup-cutmix-in-keras を基に作成

    CutoutとMixupの技術それぞれを合わせたような手法。Cutoutの部分を別ラベルの画像の一部を入れる
    参考:https://qiita.com/wakame1367/items/82316feb7268e56c6161

    従来まで行われていた領域の欠如を行う手法（Random Erasing/Cutout）は学習に必要な情報を削り、非効率になるため、その改善を図ったそう
    参考:https://nonbiri-tereka.hatenablog.com/entry/2020/01/06/082921
    """
    def get_rand_bbox(width, height, l):
        """Cutoutの領域切り出すための関数"""
        r_x = np.random.randint(width)
        r_y = np.random.randint(height)
        r_l = np.sqrt(1 - l)
        r_w = np.int(width * r_l)
        r_h = np.int(height * r_l)
        return r_x, r_y, r_l, r_w, r_h

    assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
    lam = np.random.beta(cutmix_alpha, cutmix_alpha)
    width = X1.shape[2]
    height = X1.shape[1]
    r_x, r_y, r_l, r_w, r_h = get_rand_bbox(width, height, lam)
    bx1 = np.clip(r_x - r_w // 2, 0, width)
    bx2 = np.clip(r_x + r_w // 2, 0, width)
    by1 = np.clip(r_y - r_h // 2, 0, height)
    by2 = np.clip(r_y + r_h // 2, 0, height)
    X1[:, bx1:bx2, by1:by2, :] = X2[:, bx1:bx2, by1:by2, :]
    X = X1
    lam = 1 - ((bx2 - bx1) * (by2 - by1) / (width * height))
    y = lam * y1 + (1 - lam) * y2
    return X, y


def get_kuzushiji_generator(gen):
    """
    下山さんがくずし字コンペでやっていたData AugmentationをするGenerator
    Args:
        gen:flow済みImageDataGenerator
    """
    for batch_x, batch_y in gen:
        aug = my_image.Compose(
            [
                my_image.RandomTransform(width=batch_x[0].shape[1], height=batch_x[0].shape[0], flip_h=False), # Flip, Scale, Resize, Rotateをまとめて処理。
                my_image.RandomCompose(
                    [
                        my_image.RandomBlur(p=0.125), # ぼかし。
                        my_image.RandomUnsharpMask(p=0.125), # シャープ化。
                        my_image.GaussNoise(p=0.125), # ガウシアンノイズ。
                        my_image.RandomBrightness(p=0.25), # 明度の変更。
                        my_image.RandomContrast(p=0.25), # コントラストの変更。
                        my_image.RandomEqualize(p=0.0625), # ヒストグラム平坦化。
                        my_image.RandomAutoContrast(p=0.0625), # オートコントラスト。
                        my_image.RandomPosterize(p=0.0625), # ポスタリゼーション。
                        my_image.RandomAlpha(p=0.125), # 画像の一部にランダムな色の半透明の矩形を描画する
                    ]
                ),
            ]
        )
        batch_x = np.array([aug(image=x)["image"] for x in batch_x])
        yield batch_x, batch_y


def get_cifar10_best_train_generator(x_train, y_train, batch_size):
    """
    CIFAR10のbest ImageDataGenerator 取得
    C:/Users/shingo/jupyter_notebooktfgpu_py36_work/02_keras_py/experiment/cifar10_wrn_acc_97/cifar10_wrn_acc_97.py より
    Args:
        x_train:前処理前の画像array
        y_train:ラベルarray
        batch_size:バッチサイズ
    """
    from experiment.cifar10_wrn_acc_97 import cifar10_wrn_acc_97
    train_gen = cifar10_wrn_acc_97.mode7_generator(x_train, y_train, batch_size)
    return train_gen


def randaugment_generator(gen, N=None, M=None, rescale=1.0/255.0):
    """
    Rand_AugmentでData AugmentationするGenerator
    RandAugment:ランダムにN個のAugmentation手法(=transformation)を選ぶData Augmentation。AutoAugmentに匹敵する精度。
    パラメータはN,Mだけなので最適なNとMはグリッドサーチで見つけれる。
    例:
        _N = trial.suggest_categorical('N', list(range(2,14)))
        _M = trial.suggest_categorical('M', list(range(0,10)))
    Nは14種類。
    Mはそれぞれのtransformationごとに決めるのではなく、全transformationに一貫して同じMを使うことで探索空間をさらに減らしています。
    日本語解説:https://ai-scholar.tech/treatise/randaugment-ai-370/

    論文では以下のパターンの時best results
    - cifar10 + WideResNet-28-2:   N=1,M=2
    - cifar10 + Wide-ResNet-28-10: N=2,M=14
    Args:
        gen:flow済みImageDataGenerator
        N: 選択するtransformation(水増し)の数。
        M: Augmentationをどれだけ強くかけるか。Mは0から10のいずれかの整数。
        rescale: 1.0/255.0の前処理.genは1/255済みでないと
    """
    img_augment = Rand_Augment.Rand_Augment(Numbers=N, max_Magnitude=M)
    for batch_x, batch_y in gen:
        if np.max(batch_x) > 1.0:
            # 前処理してない場合
            batch_img = [Image.fromarray(np.uint8(x)) for x in batch_x] # numpy->PIL
        else:
            # 前処理済みの場合
            batch_img = [Image.fromarray(np.uint8(x / rescale)) for x in batch_x] # numpy->PIL
        batch_img = [img_augment(image=img) for img in batch_img] # Rand_Augment.Rand_AugmentはPILでないとエラー
        batch_x = np.array([np.asarray(img, np.float32) * rescale for img in batch_img]) # PIL->numpy
        yield batch_x, batch_y


def ricap_generator(gen, beta=0.3, use_same_random_value_on_batch=True):
    """
    RICAPでData AugmentationするGenerator
    Args:
        gen:flow済みImageDataGenerator
        beta, use_same_random_value_on_batch: RICAPのパラメータ
    """
    for batch_x, batch_y in gen:
        yield ricap(batch_x, batch_y, beta=beta, use_same_random_value_on_batch=use_same_random_value_on_batch)


def random_erasing_generator(gen, p=0.5, s=(0.02, 0.4), r=(0.3, 3), max_pic=1.0):
    """
    Random ErasingでData AugmentationするGenerator
    Args:
        gen:flow済みImageDataGenerator
        p, s, r, max_pic: Random Erasingのパラメータ
    """
    for batch_x, batch_y in gen:
        batch_x = np.array([random_erasing(x, p=p, s=s, r=r, max_pic=max_pic) for x in batch_x])
        yield batch_x, batch_y


def mixup_generator(gen, mix_up_alpha=0.2):
    """
    MixupでData AugmentationするGenerator
    Args:
        gen:flow済みImageDataGenerator
        mix_up_alpha: Mixupのパラメータ
    """
    for batch_x, batch_y in gen:
        while True:
            batch_x_2, batch_y_2 = next(gen)
            m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
            if m1 < m2:
                batch_x_2 = batch_x_2[:m1]
                batch_y_2 = batch_y_2[:m1]
                break
            elif m1 == m2:
                break
        batch_x, batch_y = mix_up(batch_x, batch_y, batch_x_2, batch_y_2, mix_up_alpha=mix_up_alpha)
        yield batch_x, batch_y


def cutmix_generator(gen, cutmix_alpha=1.0):
    """
    CutmixでData AugmentationするGenerator
    Args:
        gen:flow済みImageDataGenerator
        mix_up_alpha: Cutmixのパラメータ
    """
    for batch_x, batch_y in gen:
        while True:
            batch_x_2, batch_y_2 = next(gen)
            m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
            if m1 < m2:
                batch_x_2 = batch_x_2[:m1]
                batch_y_2 = batch_y_2[:m1]
                break
            elif m1 == m2:
                break
        batch_x, batch_y = cutmix(batch_x, batch_y, batch_x_2, batch_y_2, cutmix_alpha=cutmix_alpha)
        yield batch_x, batch_y


def random_crop_generator(gen, random_crop_size=[224,224]):
    """
    Random cropでData AugmentationするGenerator
    Args:
        gen:flow済みImageDataGenerator
        random_crop_size: Random cropのパラメータ
    """
    for batch_x, batch_y in gen:
        batch_x = np.array([random_crop(x, random_crop_size) for x in batch_x])
        yield batch_x, batch_y


def gray_generator(gen, p=1):
    """
    grayscaleでData AugmentationするGenerator
    Args:
        gen:flow済みImageDataGenerator
        p:グレー化する確率
    """
    aug = my_image.Compose( [my_image.RandomCompose([my_image.ToGrayScale(p=p)])] )
    for batch_x, batch_y in gen:
        batch_x = np.array([aug(image=x * 255.0)["image"] for x in batch_x])
        yield batch_x / 255.0, batch_y


def label_smoothing_generator(gen, smooth_factor=0.1, mask_value=-1.0, is_multi_class=True):
    """
    Imagedatagenerator用label_smoothing
    label_smoothing：分類問題の正解ラベルの1を0.9みたいに下げ、0ラベルを0.1みたいに上げ、過学習軽減させる正則化手法。
    間違っている正解ラベルが混じっているデータセットのときに有効
    tensorflow.kerasなら以下のコードでもlabel_smoothing可能(https://www.pyimagesearch.com/2019/12/30/label-smoothing-with-keras-tensorflow-and-deep-learning/ より)
        from tensorflow.keras.losses import CategoricalCrossentropy
        loss = CategoricalCrossentropy(label_smoothing=0.1)
        model.compile(loss=loss, optimizer='sgd', metrics=["accuracy"])
    Args:
        gen:flow済みImagedatageneratorのインスタンス。d_cls.train_genとか
        smooth_factor:label_smoothingで下げる割合。smooth_factor=0.1で、5クラス[0,0,1,0,0]なら[0.02,0.02,0.92,0.02,0.02]になる
        mask_value:欠損値ラベル。label_smoothingでラベル値がマイナスになる場合、mask_valueに置き換える
        is_multi_class:マルチクラス分類のフラグ。Falseの場合、smooth_factor=0.1で、5クラス[0,0,1,0,0]なら[0.00,0.00,0.90,0.00,0.00]になる
                       マルチクラス分類の場合softmaxで合計ラベル=1になるが、multiラベルはそうではないので、ラベル値加算したくない時用
    Returns:
        Imagedatageneratorインスタンス（yはlabel_smoothing済み）
    """
    def _smooth_labels(y_i, smooth_factor, mask_value, is_multi_class):
        y_i = y_i.astype('float64')  # int型だとエラーになるのでfloatに変換
        y_i *= 1 - smooth_factor  # ラベル値減らす
        # ラベル値加算するか(マルチクラス分類の場合softmaxで合計ラベル=1になるが、multiラベルはそうではないので)
        if is_multi_class == True:
            y_i += smooth_factor / y_i.shape[0]
        y_i = np.where(y_i < 0.0, mask_value, y_i)  # 負の値になったらマスク値に置換する
        return y_i

    for x, y in gen:
        smooth_y = np.empty(y.shape, dtype=np.float)  # yは上書きできないので同じ大きさの空配列用意
        for i, y_i in enumerate(y):
            smooth_y[i] = _smooth_labels(y_i, smooth_factor, mask_value, is_multi_class)
        yield x, smooth_y


def print_image_generator(gen, i=0):
    """
    ImageDataGeneratorの1batdh分画像とラベルをprintで確認する
    Arges:
        gen: flow済みのImageDataGeneratorのインスタンス。d_cls.train_genとか
        i: batchのid
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x, y = next(gen)
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


class MyImageDataGenerator(ImageDataGenerator):
    """
    KerasのImageDataGeneratorを継承してMix-upやRandom Croppingのできる独自のジェネレーターを作る
    https://qiita.com/koshian2/items/909360f50e3dd5922f32
    """
    def __init__(self
                 , featurewise_center=False  # featurewise_center に True を指定すると、データセット全体で各チャンネルごとの画素値の平均を0にする
                 , featurewise_std_normalization=False  # featurewise_std_normalization に True を指定すると、データセット全体で各チャンネルごとの画素値の分散を1にする。featurewise_std_normalization=True にした場合、featurewise_center=True も指定しなければならない
                 , samplewise_center=False  # samplewise_center に True を指定すると、サンプルごとの画素値の平均を0にする
                 , samplewise_std_normalization=False  # samplewise_std_normalization に True を指定すると、サンプルごとの画素値の分散を1にする
                 , zca_whitening=False  # zca_whitening に True を指定すると、白色化を行う。zca_whitening=True にした場合、featurewise_center=True も指定しなければならない
                 , zca_epsilon=1e-06  # zca_whitening のイプシロン
                 , rotation_range=0.0  # rotation_range=20 とした場合、-20° ~ 20° の範囲でランダムに回転する
                 , width_shift_range=0.0  # width_shift_range=0.3 とした場合、[-0.3 * Width, 0.3 * Width] の範囲でランダムに左右平行移動
                 , height_shift_range=0.0 # height_shift_range=0.3 とした場合、[-0.3 * Height, 0.3 * Height] の範囲でランダムに上下平行移動
                 , brightness_range=None  # ランダムに明度を変更
                 , shear_range=0.0  # shear_range=5 とした場合、-5° ~ 5° の範囲でランダムにせん断
                 , zoom_range=0.0  # zoom_range=[0.5, 1.2] とした場合、[0.5, 1.2] の範囲でランダムに拡大縮小
                 , channel_shift_range = 0.0 # channel_shift_range=5. とした場合、[-5.0, 5.0] の範囲でランダムに画素値に値を足す
                 , fill_mode='nearest'  # 回転や平行移動等の結果、値がないピクセルをどのように埋めるかを指定 'nearest': 最も近い値で埋める
                 , cval=0.0  # 回転や平行移動等の結果、境界外の点に使用される値
                 , horizontal_flip=False  # horizontal_flip=True とした場合、ランダムに左右反転
                 , vertical_flip=False  # vertical_flip=True とした場合、ランダムに上下反転
                 , rescale=1. / 255  # 各変換を行う前に画素値を rescale 倍する
                 , preprocessing_function=None  # コールバック関数による前処理。rescaleより前に一番始めに実行される
                 , data_format='channels_last' # "channels_last" のままにする
                 , validation_split=0.0  # 0.1に設定するとデータの最後の10％が検証のために利用される
                 , random_crop=None  # [224,224]とすると[224,224]のサイズでランダムに画像切り抜く。使わない場合はNone
                 , mix_up_alpha=0.0  # mixupの混ぜ具合。使わない場合は0.0にする。使う場合は0.2とか
                 , random_erasing_prob=0.0  # random_erasing の確率
                 , random_erasing_maxpixel=255  # random_erasing で消す領域の画素の最大値
                 , ricap_beta=0.0  # RICAP。使う場合は0.3とか
                 , ricap_use_same_random_value_on_batch=True  # RICAP
                 , is_kuzushiji_gen=False  # 下山さんが使っていたAutoAugmentのデフォルト？変換入れるか
                 , grayscale_prob=0.0  # グレースケール化の確率
                 , randaugment_N=None  # randaugmentのN
                 , randaugment_M=None  # randaugmentのM
                 , cutmix_alpha=0.0  # cutmixの混ぜ具合
                 , *args, **kwargs
                ):
        # 親クラスのコンストラクタ
        super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization, samplewise_std_normalization, zca_whitening, zca_epsilon, rotation_range, width_shift_range, height_shift_range, brightness_range, shear_range, zoom_range, channel_shift_range, fill_mode, cval, horizontal_flip, vertical_flip, rescale, preprocessing_function, data_format, validation_split
                         , *args, **kwargs)
        # 拡張処理のパラメーター
        self.is_kuzushiji_gen = is_kuzushiji_gen
        self.mix_up_alpha = mix_up_alpha
        self.random_crop_size = random_crop
        self.random_erasing_prob = random_erasing_prob
        if rescale == 1.0/255.0 and random_erasing_maxpixel == 255:
            self.random_erasing_maxpixel = 1.0
        else:
            self.random_erasing_maxpixel = random_erasing_maxpixel
        self.grayscale_prob = grayscale_prob
        self.ricap_beta = ricap_beta
        self.cutmix_alpha = cutmix_alpha
        self.ricap_use_same_random_value_on_batch = ricap_use_same_random_value_on_batch
        self.randaugment_N = randaugment_N
        self.randaugment_M = randaugment_M

    def custom_process(self, gen):
        """ 上記の自作generatorを組み合わせてnextをreturnする """

        if self.is_kuzushiji_gen:
            gen = get_kuzushiji_generator(gen)

        if (self.randaugment_N is not None) and (self.randaugment_M is not None):
            gen = randaugment_generator(gen, N=self.randaugment_N, M=self.randaugment_M)

        if self.grayscale_prob > 0.0:
            gen = gray_generator(gen, p=self.grayscale_prob)

        if self.random_crop_size is not None:
            gen = random_crop_generator(gen, random_crop_size=self.random_crop_size)

        if self.random_erasing_prob > 0.0:
            gen = random_erasing_generator(gen, p=self.random_erasing_prob)

        if self.mix_up_alpha > 0.0:
            gen = mixup_generator(gen, mix_up_alpha=self.mix_up_alpha)

        if self.ricap_beta > 0.0:
            gen = ricap_generator(gen, beta=self.ricap_beta, use_same_random_value_on_batch=self.ricap_use_same_random_value_on_batch)

        if self.cutmix_alpha > 0.0:
            gen = cutmix_generator(gen, cutmix_alpha=self.cutmix_alpha)

        return next(gen)

    def flow_from_directory(self, directory, target_size = (256,256), color_mode = 'rgb',
                            classes = None, class_mode = 'categorical', batch_size = 32, shuffle = True,
                            seed = None, save_to_dir = None, save_prefix = '', save_format = 'png',
                            follow_links = False, subset = None, interpolation = 'nearest'):
        # 親クラスのflow_from_directory
        gen = super().flow_from_directory(directory, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, follow_links, subset, interpolation)
        while True:
            yield self.custom_process(gen)

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None
            , save_to_dir=None, save_prefix='', save_format='png', subset=None):
        # 親クラスのflow_from_directory
        gen = super().flow(x, y, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, subset)
        while True:
            yield self.custom_process(gen)

    def flow_from_dataframe(self, dataframe,
                            directory=None, x_col="filename", y_col="class", weight_col=None,
                            target_size=(256, 256), color_mode='rgb', classes=None,
                            class_mode='categorical', batch_size=32, shuffle=True,
                            seed=None, save_to_dir=None, save_prefix='', save_format='png',
                            subset=None, interpolation='nearest', validate_filenames=True):
        # 親クラスのflow_from_dataframe
        gen = super().flow_from_dataframe(dataframe, directory, x_col, y_col, weight_col,
                                              target_size, color_mode, classes,
                                              class_mode, batch_size, shuffle,
                                              seed, save_to_dir, save_prefix, save_format,
                                              subset, interpolation, validate_filenames)
        while True:
            yield self.custom_process(gen)



def get_load_image_balanced_generator(paths:np.ndarray, labels:np.ndarray, n_samples=5, shape=[331,331,3], rescale_factor=1.0/255.0):
    """
    画像パスとラベル渡してミニバッチごとにunder samplingするGenerator
    ※jupyterではカーネル再起動しないとkerasのfit_generator()実行できない。
    　fit_generator()連続で実行するとthread系のエラーで学習失敗する。。。
    Args:
        paths:全画像パス
        labels:ラベル。np.array(['high','low','mid'])みたいラベル名やidもしくはonehotラベルでもいける
        n_samples:各クラスのサンプル数。バッチサイズ//クラス数にすること
        shape:リサイズする画像サイズ
        rescale_factor:画像前処理
    Usage:
        data_dir = r'D:\work\kaggle_data\Cats_VS._Dogs\images\small_set\train\Cat'
        img_paths = glob.glob(data_dir+'/*jpg')
        img_names = img_paths

        x = np.array(img_names)
        # ラベル不均衡にする
        y = np.array([0]*10 + [1]*(x.shape[0]-40) + [2]*30)
        enc, y = get_train_valid_test.label2onehot(y)

        # ミニバッチでunder samplingするGenerator
        gen = my_generator.get_load_image_balanced_generator(x, y)
        x,y = next(gen)
        util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])

        # 別Generatorと組み合わせる
        custom_gen = my_generator.randaugment_generator(gen, N=3, M=4)
        custom_gen = my_generator.mixup_generator(custom_gen)
        x,y = next(custom_gen)
        util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])
    """
    def _load_image_tfkeras(filename, shape, rescale_factor):
        """サイズ指定して画像を読み込んでnumpy.arrayに変換"""
        img = keras.preprocessing.image.load_img(filename, target_size=shape[:2])
        return keras.preprocessing.image.img_to_array(img) * rescale_factor

    #with threading.Lock():
    _iter = BatchBalancedSampler(features=paths, labels=labels, n_samples=n_samples)
    for _paths, _labels in _iter:
        _x = np.array([_load_image_tfkeras(p, shape, rescale_factor=rescale_factor) for p in _paths])
        yield _x, _labels


class BatchBalancedSampler:
    def __init__(self, features:np.ndarray, labels:np.ndarray, n_samples:int):
        """
        ミニバッチをunder sampling（不均衡データマルチクラス分類用）
        ※under sampling: 多数派データをランダムに減らして少数派データと均一にする
        参考:https://devblog.thebase.in/entry/2020/02/29/110000?utm_campaign=piqcy&utm_medium=email&utm_source=Revue%20newsletter
        Args:
            features:説明変数。全特徴量や全画像パスとか
            labels:ラベル。np.array(['high','low','mid'])みたいラベル名やidもしくはonehotラベルでもいける
            n_samples:各クラスのサンプル数。バッチサイズ//クラス数にすること
        Returns:
            under samplingしたfeaturesとlabels
        Usage:
            test_iter = BatchBalancedSampler(features=x, labels=y, n_samples=5)
            for paths, labels in test_iter:
                print(paths, paths.shape)
                print(labels, labels.shape)
                break
        """
        self.n_samples = n_samples
        self.features = features
        self.labels = labels

        # labelsがonehotならidに戻す
        if len(np.array(labels).shape) == 2:
            labels = np.array([np.argmax(one) for one in labels])

        # 各ラベルの数集計
        label_counts = pd.Series(labels).value_counts()# labelsが文字列の場合でもいけるようにする #np.bincount(labels)
        major_label = label_counts.argmax()
        minor_labels = [l for l in label_counts.index if l != major_label]#minor_label = label_counts.argmin()

        # 各ラベルのindex取得
        self.major_indices = np.where(labels == major_label)[0]
        self.minor_indices_list = [np.where(labels == minor_label)[0] for minor_label in minor_labels]#self.minor_indices_list = np.where(labels == minor_label)[0]

        # 各ラベルのindexシャッフル
        np.random.shuffle(self.major_indices)
        for minor_indices in self.minor_indices_list:
            np.random.shuffle(minor_indices)

        self.used_indices = 0
        self.count = 0
        self.n_class = label_counts.shape[0]
        self.batch_size = self.n_samples * self.n_class

    def __iter__(self):
        while True:
            np.random.shuffle(self.major_indices)
            for minor_indices in self.minor_indices_list:
                np.random.shuffle(minor_indices)
            self.used_indices = 0
            self.count = 0

            while self.count + self.batch_size < len(self.major_indices):

                # 多数派データ(major_indices)からは順番に選び出し
                indices = self.major_indices[self.used_indices:self.used_indices + self.n_samples].tolist()

                # 少数派データ(minor_indices)からはランダムに選び出す操作を繰り返す
                for minor_indices in self.minor_indices_list:
                    indices = indices + np.random.choice(minor_indices, self.n_samples, replace=False).tolist()

                yield self.features[indices], self.labels[indices]#yield torch.tensor(self.features[indices]), torch.tensor(self.labels[indices])

                self.used_indices += self.n_samples
                self.count += self.n_samples * self.n_class
