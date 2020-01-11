"""
KerasのImageDataGeneratorのカスタムジェネレータークラス
ImageDataGenerator+Mix-up+Random_Cropping+Random_Erasing

Usage:
    import my_generator
    # 訓練画像水増し（Data Augmentation）
    train_datagen = my_generator.MyImageDataGenerator(
        rescale=1.0,
        shear_range = shear_range,
        zoom_range=zoom_range,
        mix_up_alpha=mix_up_alpha,
        #random_crop=random_crop,
        random_erasing_prob=random_erasing_prob,
        random_erasing_maxpixel=1.0
        )
    # 検証画像_前処理実行
    val_datagen = my_generator.get_datagen(rescale=1.0)

    # 訓練画像用意
    train_generator = datagen.flow_from_directory(
        train_data_dir, # ラベルクラスをディレクトリ名にした画像ディレクトリのパス
        target_size=(shape[0], shape[1]), # すべての画像はこのサイズにリサイズ
        color_mode='rgb',# 画像にカラーチャンネルが3つある場合は「rgb」画像が白黒またはグレースケールの場合は「grayscale」
        classes=classes, # 分類クラスリスト
        class_mode='categorical', # 2値分類は「binary」、多クラス分類は「categorical」
        batch_size=train_batch_size, # バッチごとにジェネレータから生成される画像の数
        shuffle=True # 生成されているイメージの順序をシャッフルする場合は「True」を設定し、それ以外の場合は「False」。train set は基本入れ替える
    )
    # train_generator = train_datagen.flow(x_train, y_train, batch_size=train_batch_size)

    # 検証画像用意
    validation_generator = valid_datagen.flow_from_directory(
        train_data_dir,
        target_size=(shape[0], shape[1]),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=train_batch_size,# batch_size はセット内の画像の総数を正確に割るような数に設定しないと同じ画像を2回使うため、validation やtest setのbatch size は割り切れる数にすること！！！
        shuffle=False# validation/test set は基本順番入れ替えない
    )
    # val_datagen.flow(x_test, y_test, batch_size=valid_batch_size)

"""
import os, sys
import numpy as np

# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得
sys.path.append( str(current_dir) + '/../' )
from transformer import ndimage, my_image

# imgaug は["scipy", "scikit-image>=0.11.0", "numpy>=1.15.0", "six", "imageio", "Pillow", "matplotlib","Shapely", "opencv-python"] の依存ライブラリ必要
sys.path.append( str(current_dir) + '/../Git/imgaug' )
import imgaug
# albumentations はimgaug をimport しておかないとimport できない
sys.path.append( str(current_dir) + '/../Git/albumentations' )
import albumentations

sys.path.append( str(current_dir) + '/../Git/keras-preprocessing' )
from keras_preprocessing.image import ImageDataGenerator

_gray_aug = my_image.Compose( [my_image.RandomCompose([my_image.ToGrayScale(p=1)])] )
def preprocessing_grayscale(x):
    """ ImageDataGeneratorのpreprocessing_functionで3次元画像をグレースケール化 """
    x = _gray_aug(image=x)["image"].astype(np.float32)
    #print(x, x.shape, type(x), type(x[0][0][0]))
    return x

def get_datagen(rescale=1.0/255, is_grayscale=False):
    """画像_前処理実行"""
    if is_grayscale == True:
        datagen = ImageDataGenerator(rescale=rescale # 前処理：画像を0.0~1.0の範囲に変換
                                     , preprocessing_function=preprocessing_grayscale) # グレースケール化
    else:
        datagen = ImageDataGenerator(rescale=rescale) # 前処理：画像を0.0~1.0の範囲に変換
    return datagen

class MyImageDataGenerator(ImageDataGenerator):
    """
    KerasのImageDataGeneratorを継承してMix-upやRandom Croppingのできる独自のジェネレーターを作る
    Random Erasing も追加した
    https://qiita.com/koshian2/items/909360f50e3dd5922f32
    """
    def __init__(self
                 , featurewise_center = False # featurewise_center に True を指定すると、データセット全体で各チャンネルごとの画素値の平均を0にする
                 , featurewise_std_normalization = False# featurewise_std_normalization に True を指定すると、データセット全体で各チャンネルごとの画素値の分散を1にする。featurewise_std_normalization=True にした場合、featurewise_center=True も指定しなければならない
                 , samplewise_center = False # samplewise_center に True を指定すると、サンプルごとの画素値の平均を0にする
                 , samplewise_std_normalization = False #  samplewise_std_normalization に True を指定すると、サンプルごとの画素値の分散を1にする
                 , zca_whitening = False # zca_whitening に True を指定すると、白色化を行う。zca_whitening=True にした場合、featurewise_center=True も指定しなければならない
                 , zca_epsilon = 1e-06 # zca_whitening のイプシロン
                 , rotation_range = 0.0 # rotation_range=20 とした場合、-20° ~ 20° の範囲でランダムに回転する
                 , width_shift_range = 0.0 # width_shift_range=0.3 とした場合、[-0.3 * Width, 0.3 * Width] の範囲でランダムに左右平行移動
                 , height_shift_range = 0.0 # height_shift_range=0.3 とした場合、[-0.3 * Height, 0.3 * Height] の範囲でランダムに上下平行移動
                 , brightness_range = None # ランダムに明度を変更
                 , shear_range = 0.0 # shear_range=5 とした場合、-5° ~ 5° の範囲でランダムにせん断
                 , zoom_range = 0.0 # zoom_range=[0.5, 1.2] とした場合、[0.5, 1.2] の範囲でランダムに拡大縮小
                 , channel_shift_range = 0.0 # channel_shift_range=5. とした場合、[-5.0, 5.0] の範囲でランダムに画素値に値を足す
                 , fill_mode = 'nearest' # 回転や平行移動等の結果、値がないピクセルをどのように埋めるかを指定 'nearest': 最も近い値で埋める
                 , cval = 0.0 # 回転や平行移動等の結果、境界外の点に使用される値
                 , horizontal_flip = False # horizontal_flip=True とした場合、ランダムに左右反転
                 , vertical_flip = False # vertical_flip=True とした場合、ランダムに上下反転
                 , rescale = 1. / 255 # 各変換を行う前に画素値を rescale 倍する
                 , preprocessing_function = None # コールバック関数による前処理。rescaleより前に一番始めに実行される
                 , data_format = 'channels_last' # "channels_last" のままにする
                 , validation_split = 0.0 # 0.1に設定するとデータの最後の10％が検証のために利用される
                 , random_crop = None # [224,224]とすると[224,224]のサイズでランダムに画像切り抜く。使わない場合はNone
                 , mix_up_alpha = 0.0 # mixupの混ぜ具合。使わない場合は0.0にする。使う場合は0.2とか
                 , random_erasing_prob = 0.0 # random_erasing の確率
                 , random_erasing_maxpixel = 255 # random_erasing で消す領域の画素の最大値
                 , ricap_beta = 0.0 # RICAP。使う場合は0.3とか
                 , ricap_use_same_random_value_on_batch = True # RICAP
                 , is_base_aug = False # 下山さんが使っていたAutoAugmentのデフォルト？変換入れるか
                 , is_grayscale = False # グレースケール化
                 , *args, **kwargs
                ):
        # 親クラスのコンストラクタ
        super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization, samplewise_std_normalization, zca_whitening, zca_epsilon, rotation_range, width_shift_range, height_shift_range, brightness_range, shear_range, zoom_range, channel_shift_range, fill_mode, cval, horizontal_flip, vertical_flip, rescale, preprocessing_function, data_format, validation_split
                         , *args, **kwargs)

        # 拡張処理のパラメーター
        # 下山さんが使っていたAutoAugmentのデフォルト？変換入れるか
        self.is_base_aug = is_base_aug
        # Mix-up
        assert mix_up_alpha >= 0.0
        self.mix_up_alpha = mix_up_alpha
        # Random Crop
        assert random_crop == None or len(random_crop) == 2
        self.random_crop_size = random_crop
        # Random Erasing
        assert random_erasing_prob >= 0.0
        self.random_erasing_prob = random_erasing_prob
        if rescale == 1. / 255 and random_erasing_maxpixel == 255:
            self.random_erasing_maxpixel = 1
        else:
            self.random_erasing_maxpixel = random_erasing_maxpixel
        # グレースケール化
        self.is_grayscale = is_grayscale
        # RICAP
        assert ricap_beta >= 0.0
        self.ricap_beta = ricap_beta
        self.ricap_use_same_random_value_on_batch = ricap_use_same_random_value_on_batch

    def random_crop(self, original_img):
        """
        ランダムクロップ
        https://jkjung-avt.github.io/keras-image-cropping/
        """
        # Note: image_data_format is 'channel_last'
        assert original_img.shape[2] == 3
        if original_img.shape[0] < self.random_crop_size[0] or original_img.shape[1] < self.random_crop_size[1]:
            raise ValueError(f"Invalid random_crop_size : original = {original_img.shape}, crop_size = {self.random_crop_size}")
        height, width = original_img.shape[0], original_img.shape[1]
        #dy, dx = self.random_crop_size
        #x = np.random.randint(0, width - dx + 1)
        #y = np.random.randint(0, height - dy + 1)
        #x_crop = original_img[y:(y+dy), x:(x+dx), :]
        data = {'image': original_img}
        augmentation = albumentations.RandomCrop(self.random_crop_size[0], original_img.shape[1], p=0.5) # albumentationsでやってみる
        x_crop = augmentation(**data)['image']
        return ndimage.resize(x_crop, height, width) # x_cropの画像サイズはcropしたサイズなのでもとの画像サイズに戻す

    def mix_up(self, X1, y1, X2, y2):
        """
        Mix-up
        https://qiita.com/yu4u/items/70aa007346ec73b7ff05
        """
        assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
        batch_size = X1.shape[0]
        l = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, batch_size)
        X_l = l.reshape(batch_size, 1, 1, 1)
        y_l = l.reshape(batch_size, 1)
        X = X1 * X_l + X2 * (1-X_l)
        y = y1 * y_l + y2 * (1-y_l)
        return X, y

    def random_erasing(self, x, p=0.5, s=(0.02, 0.4), r=(0.3, 3), max_pic=255):
        """
        Random Erasing
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

    def ricap(self, image_batch, label_batch, beta=0.3, use_same_random_value_on_batch=True):
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


    def custom_process(self, batches):
        """
        拡張処理
        (回転や拡大,ぼかしとか明度変更とか（下山さんが使っていたAutoAugmentのデフォルト？）→ Mix-up → Random_Crop → RICAP → Random_Erasing)
        """
        batch_x, batch_y = next(batches) # <-- 追加が必要ですね

        if self.is_base_aug == True:
            # Flip, Scale, Resize, Rotateをまとめて処理、
            # ぼかし, シャープ化, ガウシアンノイズ, 明度変更, コントラストの変更, ヒストグラム平坦化, オートコントラスト, ポスタリゼーション, 画像の一部にランダムな色の半透明の矩形を描画する
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
            x = np.zeros(batch_x.shape)
            for i in range(batch_x.shape[0]):
                x[i] = aug(image=batch_x[i])["image"]
            batch_x = x

        # Mix-up
        if self.mix_up_alpha > 0:
            while True:
                batch_x_2, batch_y_2 = next(batches)
                m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
                if m1 < m2:
                    batch_x_2 = batch_x_2[:m1]
                    batch_y_2 = batch_y_2[:m1]
                    break
                elif m1 == m2:
                    break
            batch_x, batch_y = self.mix_up(batch_x, batch_y, batch_x_2, batch_y_2)

        # Random crop うまく機能しない。。。
        if self.random_crop_size != None:
            x = np.zeros((batch_x.shape[0], self.random_crop_size[0], self.random_crop_size[1], 3))
            for i in range(batch_x.shape[0]):
                x[i] = self.random_crop(batch_x[i])
            batch_x = x

        # RICAP
        if self.ricap_beta > 0:
            batch_x, batch_y = self.ricap(batch_x, batch_y, beta=self.ricap_beta, use_same_random_value_on_batch=self.ricap_use_same_random_value_on_batch)

        # グレースケール化
        if self.is_grayscale == True:
            aug = my_image.Compose( [my_image.RandomCompose([my_image.ToGrayScale(p=1)])] )
            x = np.zeros(batch_x.shape)
            for i in range(batch_x.shape[0]):
                # rescale済みだから255.0かけないとおかしくなる
                tmp = aug(image=batch_x[i]*255.0)["image"]
                x[i] = tmp/255.0
            batch_x = x

        # Random Erasing
        if self.random_erasing_prob > 0:
            x = np.zeros(batch_x.shape)
            for i in range(batch_x.shape[0]):
                x[i] = self.random_erasing(batch_x[i], p=self.random_erasing_prob, max_pic=self.random_erasing_maxpixel)
            batch_x = x

        return (batch_x, batch_y)


    def flow_from_directory(self, directory, target_size = (256,256), color_mode = 'rgb',
                            classes = None, class_mode = 'categorical', batch_size = 32, shuffle = True,
                            seed = None, save_to_dir = None, save_prefix = '', save_format = 'png',
                            follow_links = False, subset = None, interpolation = 'nearest'):
        # 親クラスのflow_from_directory
        batches = super().flow_from_directory(directory, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, follow_links, subset, interpolation)
        # 拡張処理
        while True:
            # 返り値
            yield self.custom_process(batches)

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None):
        # 親クラスのflow_from_directory
        batches = super().flow(x, y, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, subset)
        # 拡張処理
        while True:
            # 返り値
            yield self.custom_process(batches)

    def flow_from_dataframe(self, dataframe,
                            directory=None, x_col="filename", y_col="class", weight_col=None,
                            target_size=(256, 256), color_mode='rgb', classes=None,
                            class_mode='categorical', batch_size=32, shuffle=True,
                            seed=None, save_to_dir=None, save_prefix='', save_format='png',
                            subset=None, interpolation='nearest', validate_filenames=True):
        # 親クラスのflow_from_dataframe
        print('color_mode', color_mode)
        batches = super().flow_from_dataframe(dataframe, directory, x_col, y_col, weight_col,
                                              target_size, color_mode, classes,
                                              class_mode, batch_size, shuffle,
                                              seed, save_to_dir, save_prefix, save_format,
                                              subset, interpolation, validate_filenames)
        # 拡張処理
        while True:
            # 返り値
            yield self.custom_process(batches)


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
