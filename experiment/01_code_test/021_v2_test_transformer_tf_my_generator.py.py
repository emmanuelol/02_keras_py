# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# !pwd

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:96% !important; }</style>"))  # デフォルトは75%

import sys

sys.executable
# -

# # tf_my_generator.pyテスト

# +
# モジュールimport
import os, sys

current_dir = os.path.dirname(os.path.abspath("__file__"))
path = os.path.join(current_dir, "../../")
sys.path.append(path)
from transformer import tf_my_generator as my_generator
from transformer import tf_get_train_valid_test as get_train_valid_test
from dataset import util

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

out_dir = "output_test"

# +
import os, sys, glob, pathlib
import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator

# img_dir = r"D:\work\keras_iPhone_pictures\InceptionResNetV2_small_set\train_100\train"
img_dir = r"D:\work\kaggle_data\Cats_VS._Dogs\images\small_set\train"
classes = sorted([pathlib.Path(d).name for d in glob.glob(img_dir + "\*")])
display(pd.DataFrame(classes))

shape = 331, 331, 3
batch_size = 30

gen = ImageDataGenerator(rescale=1.0 / 255.0)
# gen = ImageDataGenerator()
gen = gen.flow_from_directory(
    img_dir,
    target_size=(shape[0], shape[1]),
    color_mode="rgb",
    classes=classes,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
)
# -

# ### Imagedatagenerator用label_smoothing
# ### ImageDataGeneratorの1batdh分画像とラベルをprintで確認する

# +
custom_gen = my_generator.label_smoothing_generator(gen)

my_generator.print_image_generator(custom_gen, i=0)
# -

# ### grayscaleでData AugmentationするGenerator

# +
custom_gen = my_generator.gray_generator(gen, p=1.0)

x, y = next(custom_gen)
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])
# -

# ### Random cropでData AugmentationするGenerator

# +
custom_gen = my_generator.random_crop_generator(custom_gen, random_crop_size=[100, 100])

x, y = next(custom_gen)
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])
# -

# ### Random ErasingでData AugmentationするGenerator

# +
custom_gen = my_generator.random_erasing_generator(gen, p=0.5)

x, y = next(custom_gen)
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])
# -

# ### RICAPでData AugmentationするGenerator

# +
custom_gen = my_generator.ricap_generator(custom_gen)

x, y = next(custom_gen)
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])
# -

# ### MixupでData AugmentationするGenerator

# +
custom_gen = my_generator.mixup_generator(custom_gen)

x, y = next(custom_gen)
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])
# -

# ### Rand_AugmentでData AugmentationするGenerator

# +
custom_gen = my_generator.randaugment_generator(gen, N=3, M=4)

x, y = next(custom_gen)
print(np.max(x))
print(y[0])
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])
# -

# ### CutmixでData AugmentationするGenerator

# +
custom_gen = my_generator.cutmix_generator(gen, cutmix_alpha=1.0)

x, y = next(custom_gen)
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])
# -

# ### 下山さんがくずし字コンペでやっていたData AugmentationをするGenerator

# +
custom_gen = my_generator.get_kuzushiji_generator(gen)

x, y = next(custom_gen)
print(np.max(x))
print(y[0])
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])
# -

# ### GridMaskでData AugmentationするGenerator

# +
custom_gen = my_generator.gridmask_generator(gen, p=1.0, num_grid=3)

x, y = next(custom_gen)
print(np.max(x))
print(y[0])
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])
# -

# ### MyImageDataGeneratorクラス確認

# +
my_IDG_options = {"rescale": 1.0 / 255.0}
custom_gen = my_generator.MyImageDataGenerator(**my_IDG_options)

custom_gen = custom_gen.flow_from_directory(
    img_dir,
    target_size=(shape[0], shape[1]),
    color_mode="rgb",
    classes=classes,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
)

x, y = next(custom_gen)
print(np.max(x))
print(y[0])
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])

# +
my_IDG_options = {
    "rescale": 1.0 / 255.0,
    # 'is_kuzushiji_gen':True,
    # 'grayscale_prob':0.5,
    # 'random_crop':[24,112],
    # 'random_erasing_prob':0.5,
    # 'mix_up_alpha':0.2,
    # 'ricap_beta':0.3,
    "randaugment_N": 2,
    "randaugment_M": 3,
    "cutmix_alpha": 1.0,
    "gridmask_prob": 0.5,
}
custom_gen = my_generator.MyImageDataGenerator(**my_IDG_options)

custom_gen = custom_gen.flow_from_directory(
    img_dir,
    target_size=(shape[0], shape[1]),
    color_mode="rgb",
    classes=classes,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
)

x, y = next(custom_gen)
print(np.max(x))
print(y[0])
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])
# -



# ### ミニバッチでunder samplingするGenerator

# +
data_dir = r"D:\work\kaggle_data\Cats_VS._Dogs\images\small_set\train\Cat"
img_paths = glob.glob(data_dir + "/*jpg")
img_names = img_paths
shape = [100, 100, 3]
n_samples = 50
print(len(img_paths))

x = np.array(img_names)
# ラベル不均衡にする
y = np.array([0] * 80 + [1] * (x.shape[0] - 80))
# y = np.array([0]*10 + [1]*(x.shape[0]-40) + [2]*30)
enc, y = get_train_valid_test.label2onehot(y)

# ミニバッチでunder samplingするGenerator
gen = my_generator.get_load_image_balanced_generator(
    x, y, n_samples=n_samples, shape=shape
)
x, y = next(gen)
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])

# 別Generatorと組み合わせる
custom_gen = my_generator.randaugment_generator(gen, N=6, M=8)
custom_gen = my_generator.gridmask_generator(custom_gen)
custom_gen = my_generator.mixup_generator(custom_gen)
x, y = next(custom_gen)
util.plot_5imgs(x, plot_num=20, labels=[np.argmax(_y) for _y in y])
# -

# #### trainためす  
# ※jupyterではカーネル再起動しないとkerasのfit_generator()実行できない。  
# fit_generator()連続で実行するとthread系のエラーで学習失敗する。。。

# +
# モジュールimport
import os, sys

current_dir = os.path.dirname(os.path.abspath("__file__"))
path = os.path.join(current_dir, "../../")
sys.path.append(path)
from transformer import tf_my_generator as my_generator
from transformer import tf_get_train_valid_test as get_train_valid_test
from dataset import util

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

out_dir = "output_test"


import os, sys, glob, pathlib
import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator

# img_dir = r"D:\work\keras_iPhone_pictures\InceptionResNetV2_small_set\train_100\train"
img_dir = r"D:\work\kaggle_data\Cats_VS._Dogs\images\small_set\test"
classes = sorted([pathlib.Path(d).name for d in glob.glob(img_dir + "\*")])
display(pd.DataFrame(classes))

shape = 100, 100, 3
batch_size = 8

my_IDG_options = {
    "rescale": 1.0 / 255.0,
    #'ricap_beta':0.3,
    "cutmix_alpha": 0.5,
    "randaugment_N": 14,
    "randaugment_M": 9,
}
custom_gen = my_generator.MyImageDataGenerator(**my_IDG_options)
custom_gen = custom_gen.flow_from_directory(
    img_dir,
    target_size=(shape[0], shape[1]),
    color_mode="rgb",
    classes=classes,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
)

valid_gen = ImageDataGenerator(rescale=1.0 / 255.0)
valid_gen = valid_gen.flow_from_directory(
    img_dir,
    target_size=(shape[0], shape[1]),
    color_mode="rgb",
    classes=classes,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
)

from model import tf_define_model as define_model
from model import tf_my_callback as my_callback
from model import tf_lr_finder as lr_finder
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator

keras.backend.clear_session()

loss = "categorical_crossentropy"
metrics = ["accuracy"]
activation = "softmax"
choice_model, trainable = "VGG16", 15
choice_optim = "sgd"
model_path = None
classes = ["Cat", "Dog"]
num_epoch = 3

# model
model, orig_model = define_model.get_fine_tuning_model(
    out_dir,
    shape[0],
    shape[1],
    shape[2],
    len(classes),
    choice_model,
    trainable,
    activation=activation,
)
optim = define_model.get_optimizers(choice_optim)
model.compile(loss=loss, optimizer=optim, metrics=metrics)

history = model.fit_generator(
    custom_gen,
    steps_per_epoch=200 // batch_size,
    epochs=num_epoch,
    validation_data=valid_gen,
    validation_steps=200 // batch_size,
    verbose=1,  # 1:ログをプログレスバーで標準出力 2:最低限の情報のみ出す
    workers=1,
    use_multiprocessing=False,
)
# -


