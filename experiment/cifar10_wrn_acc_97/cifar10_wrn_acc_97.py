# -*- coding: utf-8 -*-
"""
CIFAR-10の高精度（acc=97.3%）コード（元はtensorflow.kerasだけどkerasにしてる）
オレオレAugmentation + Wide ResNetでacc=97.3%（Auto Augmentとほぼ同じ（-0.1％）精度）
Data Augmentationの「なぜ？」に注目しながら、エラー分析をしてCIFAR-10の精度向上を目指した
https://gist.github.com/koshian2/6c47d4dc9b0c4c252a8290b17ec88f11#file-error_analysis_cifar-py

■バッチサイズ：128
■エポック数：300
■学習率：Cosine Decay（上限0.1、下限0.001のコサインカーブ）
■Augmentation：
　RandomZoom(75%-125%)
　Color Shift(50)
　回転(10°)
　Random Erasing(5%-20%)
　Auto Contast(cutoff 1%)
　Random Sharpness(1±1)
　Mixup(beta=0.5)
■ネットワーク：Wide ResNet 28-10
"""

import tensorflow as tf
#from tensorflow.keras import layers
#from tensorflow.keras.models import Model
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.callbacks import Callback, History, LearningRateScheduler
#from tensorflow.keras.optimizers import SGD
#from tensorflow.contrib.tpu.python.tpu import keras_support
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.utils import to_categorical
from keras import layers
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, History, LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import to_categorical
import keras

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
import shutil
import pickle
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import itertools
import matplotlib
matplotlib.use('Agg')

# ネットワーク
def create_block(input, ch, reps):
    x = input
    for i in range(reps):
        x = layers.Conv2D(ch, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x

def create_network():
    input = layers.Input((32,32,3))
    x = create_block(input, 64, 3)
    x = layers.AveragePooling2D(2)(x)
    x = create_block(x, 128, 3)
    x = layers.AveragePooling2D(2)(x)
    x = create_block(x, 256, 3)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)
    return Model(input, x)

# Validation用のジェネレーター（共通）
def validation_generator(X, y, batch_size):
    gen = ImageDataGenerator(rescale=1.0/255)
    return gen.flow(X, y, batch_size=batch_size, shuffle=False)

# ベースラインのStandard Augmentationするジェネレーター
def baseline_generator(X, y, batch_size):
    gen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True,
                            width_shift_range=4.0/32.0, height_shift_range=4.0/32.0)
    return gen.flow(X, y, batch_size=batch_size)

# ランダムズーム
def random_zoom(original_image, zoom_range):
    uint8_array = (original_image * 255.0).astype(np.uint8)
    with Image.fromarray(uint8_array) as img:
        ratio = np.random.uniform(zoom_range[0], zoom_range[1])
        resized = img.resize((int(img.width*ratio), int(img.height*ratio)), Image.LANCZOS)
        crop_x = np.random.uniform(0, resized.width-32)
        crop_y = np.random.uniform(0, resized.height-32)
        crop = resized.crop((crop_x, crop_y, crop_x+32, crop_y+32))
        return np.asarray(crop, np.float32) / 255.0

# ランダムにズームを追加
def mode1_generator(X, y, batch_size):
    gen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True,
                            width_shift_range=4.0/32.0, height_shift_range=4.0/32.0,
                            zoom_range=[0.75, 1.25])
    return gen.flow(X, y, batch_size=batch_size)
    #for X_base, y_base in baseline_generator(X, y, batch_size):
    #    X = X_base.copy()
    #    for i in range(X.shape[0]):
            # 犬か猫の場合
    #        if y_base[i, 3] == 1 or y_base[i, 5] == 1:
    #            X[i] = random_zoom(X_base[i], [1.0, 1.5])
    #    yield X, y_base

# カラーシフトを追加
def mode2_generator(X, y, batch_size):
    gen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True,
                            width_shift_range=4.0/32.0, height_shift_range=4.0/32.0,
                            zoom_range=[0.75, 1.25], channel_shift_range=50.0)
    return gen.flow(X, y, batch_size=batch_size)

# ポーズを見ているっぽいんで回転を追加
def mode3_generator(X, y, batch_size):
    gen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True,
                            width_shift_range=4.0/32.0, height_shift_range=4.0/32.0,
                            zoom_range=[0.75, 1.25], channel_shift_range=50.0,
                            rotation_range=10)
    return gen.flow(X, y, batch_size=batch_size)

def random_erasing(image, prob=0.5, sl=0.05, sh=0.2, r1=0.2, r2=0.8):
    # パラメーター
    # - image = 入力画像
    # - prob = random erasingをする確率
    # - sl, sh = random erasingする面積の比率[sl, sh]
    # - r1, r2 = random erasingのアスペクト比[r1, r2]
    assert image.ndim == 3
    assert image.dtype == np.float32
    if np.random.rand() >= prob:
        return image
    else:
        H, W, C = image.shape # 縦横チャンネル
        S = H * W # 面積
        while True:
            S_eps = np.random.uniform(sl, sh) * S
            r_eps = np.random.uniform(r1, r2)
            H_eps, W_eps = np.sqrt(S_eps*r_eps), np.sqrt(S_eps/r_eps)
            x_eps, y_eps = np.random.uniform(0, W), np.random.uniform(0, H)
            if x_eps + W_eps <= W and y_eps + H_eps <= H:
                out_image = image.copy()
                out_image[int(y_eps):int(y_eps+H_eps), int(x_eps):int(x_eps+W_eps),
                          :] = np.random.uniform(0, 1.0)
                return out_image

# Random Erasing
def mode4_generator(X, y, batch_size):
    for X_base, y_base in mode3_generator(X, y, batch_size):
        X = X_base.copy()
        for i in range(X.shape[0]):
            X[i] = random_erasing(X_base[i])
        yield X, y_base

def mixup(X, y, beta=0.5):
    shuffle_ind = np.random.permutation(X.shape[0])
    rand = np.random.beta(beta, beta)
    X_mix = rand * X + (1-rand) * X[shuffle_ind]
    y_mix = rand * y + (1-rand) * y[shuffle_ind]
    return X_mix, y_mix

# mixup
def mode5_generator(X, y, batch_size):
    for X_base, y_base in mode4_generator(X, y, batch_size):
        yield mixup(X_base, y_base, 0.5)

# auto contrast
def auto_contrast(image, cutoff=0):
    if np.random.rand() >= 0.5:
        array = (image * 255).astype(np.uint8)
        with Image.fromarray(array) as img:
            autocon = ImageOps.autocontrast(img, cutoff)
            return np.asarray(autocon, np.float32) / 255.0
    else:
        return image

# auto contrast + mixup
def mode6_generator(X, y, batch_size):
    for X_base, y_base in mode4_generator(X, y, batch_size):
        X_batch = X_base.copy()
        for i in range(X_batch.shape[0]):
            X_batch[i] = auto_contrast(X_base[i], 1)
        # mixup
        yield mixup(X_batch, y_base, 0.5)

def solarize(image, threshold):
    if np.random.rand() >= 0.5:
        array = (image * 255).astype(np.uint8)
        with Image.fromarray(array) as img:
            equal = ImageOps.solarize(img, threshold)
            return np.asarray(equal, np.float32) / 255.0
    else:
        return image

def sharpen(image, magnitude):
    array = (image * 255).astype(np.uint8)
    with Image.fromarray(array) as img:
        factor = np.random.uniform(1.0-magnitude, 1.0+magnitude)
        sharp = ImageEnhance.Sharpness(img).enhance(factor)
        return np.asarray(sharp, np.float32) / 255.0

# auto contrast + sharpen
def mode7_generator(X, y, batch_size):
    for X_base, y_base in mode4_generator(X, y, batch_size):
        X_batch = X_base.copy()
        for i in range(X_batch.shape[0]):
            X_batch[i] = auto_contrast(X_base[i], 1)
            X_batch[i] = sharpen(X_batch[i], 1)
        # mixup
        yield mixup(X_batch, y_base, 0.5)

def lr_scheduler(epoch):
    x = 0.1
    if epoch >= 75: x /= 10.0
    elif epoch >= 125: x /= 10.0
    return x

def cosine_decay(epoch):
    lr_min = 0.001
    lr_max = 0.1
    T_max = 300
    return lr_min + 1/2*(lr_max-lr_min)*(1+np.cos(epoch/T_max*np.pi))

def error_analysis(model, X_test, y_test, mode, outdir):
    y_true = np.argmax(y_test, axis=-1)
    y_pred_prob = model.predict_generator(validation_generator(X_test, y_test, 128),
                                          steps=X_test.shape[0]//128+1)
    y_pred = np.argmax(y_pred_prob[:X_test.shape[0]], axis=-1)
    confusion = confusion_matrix(y_true, y_pred)

    labels = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]

    # report(各種スコア)の作成と保存
    #print(y_true, y_pred, np.max(y_pred))
    report = classification_report(y_true, y_pred, labels=range(10))
    with open(outdir+"/report.txt", mode='w') as f:
        f.write(report)
    print(report)
    # confusion matrixのプロット、保存、表示
    # http://adliblog.hatenablog.com/entry/2018/02/15/184020 より
    title="overall accuracy:"+str(accuracy_score(y_true, y_pred))
    # 混同行列
    plot_confusion_matrix(confusion, classes=labels, output_file=outdir+"/CM_without_normalize.png", title=title, figsize=(6, 4))

    # confusion matrixから不正解数が高いインデックスを選ぶ
    indices = np.argsort(confusion, axis=None)[::-1] # 全体の降順インデックス
    rows = indices // 10
    columns = indices % 10
    flag = rows != columns # 対角成分は正解なので除外
    rows, columns = rows[flag], columns[flag]
    cnt = 1

    for r, c in zip(rows, columns):
        selected_flag = np.logical_and(y_true==r, y_pred==c)
        selected_image = X_test[selected_flag]
        text = f"True={labels[r]}, Pred={labels[c]}, #={confusion[r,c]}, rank={cnt}"
        n = min(selected_image.shape[0], 100)
        for i in range(n):
            ax = plt.subplot(10, 10, 1+i)
            ax.imshow(selected_image[i])
            ax.axis("off")
        plt.suptitle(text)
        plt.savefig(outdir+f"/error_{cnt}.png")
        plt.clf()
        cnt += 1
        if cnt > 10: break

# 係数保存用コールバック
class CheckpointCallback(Callback):
    def __init__(self, model, output_dir):
        self.model = model
        self.max_val_acc = 0.0
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs):
        if logs["val_acc"] > self.max_val_acc:
            #self.model.save_weights("model.hdf5", save_format="h5")
            self.model.save(os.path.join(self.output_dir, 'model.h5'), include_optimizer=False)
            print(f"Val acc improved from {self.max_val_acc:.04} to {logs['val_acc']:.04}")
            self.max_val_acc = logs["val_acc"]

# confusion matrixをプロットし画像として保存する関数
# 参考： http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes, output_file,
                          normalize=False,
                          figsize=(6, 4),
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # 混同行列表示
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_file, bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()
    plt.clf()# plotの設定クリアにする

######################################### Wide ResNet #########################################
# https://github.com/tensorflow/models/blob/master/research/autoaugment/wrn.py
# TensorFlowからKerasに移植
def residual_block(x, in_filter, out_filter, stride, activate_before_residual=False):
    if activate_before_residual:
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        orig_x = x
    else:
        orig_x = x

    block_x = x
    if not activate_before_residual:
        block_x = layers.BatchNormalization()(block_x)
        block_x = layers.Activation("relu")(block_x)

    block_x = layers.Conv2D(out_filter, 3, padding="same", strides=stride)(block_x)

    block_x = layers.BatchNormalization()(block_x)
    block_x = layers.Activation("relu")(block_x)
    block_x = layers.Conv2D(out_filter, 3, padding="same", strides=1)(block_x)

    if in_filter != out_filter:
        orig_x = layers.AveragePooling2D(stride)(orig_x)
        orig_x = layers.Lambda(zero_pad, arguments={"in_filter":in_filter, "out_filter":out_filter})(orig_x)
    x = layers.Add()([orig_x, block_x])
    return x

def zero_pad(inputs, in_filter=1, out_filter=1):
  """Zero pads `input` tensor to have `out_filter` number of filters."""
  outputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0],
                            [(out_filter - in_filter) // 2,
                             (out_filter - in_filter) // 2]])
  return outputs

def _res_add(in_filter, out_filter, stride, x, orig_x):
    if in_filter != out_filter:
        orig_x = layers.AveragePooling2D(stride)(orig_x)
        orig_x = layers.Lambda(zero_pad, arguments={"in_filter":in_filter, "out_filter":out_filter})(orig_x)
    x = layers.Add()([x, orig_x])
    return x, orig_x

def build_wrn_model(shape=(32,32,3), num_classes=10, wrn_size=160):
    kernel_size = wrn_size
    filter_size = 3
    num_blocks_per_resnet = 4
    filters = [
        min(kernel_size, 16), kernel_size, kernel_size*2, kernel_size*4
    ]
    strides = [1,2,2]

    # first conv
    input = layers.Input(shape)
    x = layers.Conv2D(filters[0], filter_size, padding="same")(input)
    first_x = x # Res from the begging
    orig_x = x # Res from previous block

    for block_num in range(1, 4):
        activate_before_residual = True if block_num == 1 else False
        x = residual_block(x, filters[block_num-1],
                filters[block_num], strides[block_num-1],
                activate_before_residual=activate_before_residual)
        for i in range(1, num_blocks_per_resnet):
            x = residual_block(x, filters[block_num], filters[block_num], 1,
                activate_before_residual=False)
        x, orig_x = _res_add(filters[block_num-1], filters[block_num],
                            strides[block_num-1], x, orig_x)

    final_stride_val = int(np.prod(strides))
    x, _ = _res_add(filters[0], filters[3], final_stride_val, x, first_x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    return Model(input, x)
################################################################################################

def train(mode, output_dir, epochs=300, batch_size=128, is_tpu=False, load_model_path=None, verbose=1):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    tf.logging.set_verbosity(tf.logging.FATAL)

    # モードに応じて訓練ジェネレーターを選択
    if mode == 0:
        train_gen = baseline_generator(X_train, y_train, batch_size)
    elif mode == 999:
        train_gen = validation_generator(X_train, y_train, batch_size)
    elif mode == 1:
        train_gen = mode1_generator(X_train, y_train, batch_size)
    elif mode == 2:
        train_gen = mode2_generator(X_train, y_train, batch_size)
    elif mode == 3:
        train_gen = mode3_generator(X_train, y_train, batch_size)
    elif mode == 4:
        train_gen = mode4_generator(X_train, y_train, batch_size)
    elif mode == 5:
        train_gen = mode5_generator(X_train, y_train, batch_size)
    elif mode == 6:
        train_gen = mode6_generator(X_train, y_train, batch_size)
    elif mode == 7:
        train_gen = mode7_generator(X_train, y_train, batch_size)


    # 検証ジェネレーターは共通
    val_gen = validation_generator(X_test, y_test, batch_size)

    #model = create_network() # 10層CNN
    model = build_wrn_model() # Wide ResNet 28-10
    # モデルファイルあれば重みロード（tensorflowでネットワーク作ってる所あるためmodel.laodはエラーになる）
    if load_model_path is not None:
        model.load_weights(load_model_path)

    model.compile(SGD(0.1, 0.9), "categorical_crossentropy", ["acc"])

    if is_tpu == True:
        tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
        strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
        model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    #scheduler = LearningRateScheduler(lr_scheduler) # lr段階的に下げる場合はこっち
    scheduler = LearningRateScheduler(cosine_decay) # cosine decayの場合はこっち

    cb = CheckpointCallback(model, output_dir)
    hist = History()
    # 訓練
    model.fit_generator(train_gen, steps_per_epoch=X_train.shape[0]//batch_size,
                        validation_data=val_gen, validation_steps=X_test.shape[0]//batch_size,
                        callbacks=[scheduler, cb, hist], epochs=epochs
                        , verbose=verbose) # cosine decayの場合は300epoch
    # 係数読み込み
    #model.load_weights("model.hdf5")
    model.load_weights(load_model_path)

    # エラー分析
    error_analysis(model, X_test, y_test, mode, output_dir)
    # 係数コピー
    #outdir = f"mode{mode:02}"
    #shutil.copy("model.hdf5", outdir+"/model.hdf5")
    # 訓練ログ保存
    history = hist.history
    #with open(outdir+"/logs.pkl", "wb") as fp:
    with open(os.path.join(self.output_dir, "logs.pkl"), "wb") as fp:
        pickle.dump(history, fp)

if __name__ == "__main__":
    output_dir = r'D:\work\02_keras_py\experiment\01_code_test\output_test\cifar10_wrn_acc_97'
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists( os.path.join(output_dir, 'model.h5') ) == False:
        train(7, output_dir)
    else:
        train(7, output_dir, load_model_path=os.path.join(output_dir, 'model.h5'))
