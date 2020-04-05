# -*- coding: utf-8 -*-
"""
Poolingを定義する
"""
import os, sys
import numpy as np

import tensorflow as tf
from tensorflow import keras

# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得

class GeM2D(keras.layers.Layer):
    """
    下山さんのGeneralized Mean Pooling (GeM) <https://github.com/filipradenovic/cnnimageretrieval-pytorch>
    https://github.com/ak110/pytoolkit/blob/657e38f7519877ab28f84647f10975e87681f2cd/pytoolkit/layers/pooling.py

    Generalized Mean Pooling (GeM)：Poolingパラメータ(p)のべきを掛けたGlobal Average Pooling
    p=1なら Global Average Pooling(GAP)
    p=無限大ならGlobal Max pooling(GMP)
    Poolingパラメータ(p)は手動で設定することも，学習することもできるみたい（この実装はデフォルト3で固定されている）
    Global Average Poolingの代わりに使えばいい。論文では物体検出のmAPで特にGAP<GeMだった
    """

    def __init__(self, p=3, epsilon=1e-6, **kargs):
        super().__init__(**kargs)
        self.p = p
        self.epsilon = epsilon

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        return (input_shape[0], input_shape[3])

    def call(self, inputs, **kwargs):
        del kwargs
        x = tf.math.maximum(inputs, self.epsilon) ** self.p
        x = tf.math.reduce_mean(x, axis=(1, 2))  # GAP
        x = x ** (1 / self.p)
        return x

    def get_config(self):
        config = {"p": self.p, "epsilon": self.epsilon}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _normalize_tuple(value, n: int) -> tuple:
    """n個の要素を持つtupleにして返す。"""
    assert value is not None
    if isinstance(value, int):
        return (value,) * n
    else:
        value = tuple(value)
        assert len(value) == n
        return value

class BlurPooling2D(keras.layers.Layer):
    """
    Blur Pooling Layer <https://arxiv.org/abs/1904.11486>
    普通のpoolingの間にBlur kernel（画像のぼかしで使うカーネル。カーネルの範囲内にある全画素の画素値の平均をとってぼかす）
    で アンチエイリアシング（ギザギザを目立たなくする）を行う
    各Conv層の最後のPoolingをこのレイヤーに置き換える
    https://github.com/ak110/pytoolkit/blob/master/pytoolkit/layers/pooling.py#L212
    C:/Users/shingo/Git/blur-pool-keras
    """

    def __init__(self, taps=5, strides=2, **kwargs):
        super().__init__(**kwargs)
        self.taps = taps
        self.strides = _normalize_tuple(strides, 2)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        input_shape = list(input_shape)
        input_shape[1] = (
            input_shape[1] + int(input_shape[1]) % self.strides[0]
        ) // self.strides[0]
        input_shape[2] = (
            input_shape[2] + int(input_shape[2]) % self.strides[1]
        ) // self.strides[1]
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        in_filters = inputs.shape.as_list()[-1]

        pascals_tr = np.zeros((self.taps, self.taps))
        pascals_tr[0, 0] = 1
        for i in range(1, self.taps):
            pascals_tr[i, :] = pascals_tr[i - 1, :]
            pascals_tr[i, 1:] += pascals_tr[i - 1, :-1]
        filter1d = pascals_tr[self.taps - 1, :]
        filter2d = filter1d[np.newaxis, :] * filter1d[:, np.newaxis]
        filter2d = filter2d * (self.taps ** 2 / filter2d.sum())
        kernel = np.tile(filter2d[:, :, np.newaxis, np.newaxis], (1, 1, in_filters, 1))
        kernel = tf.constant(kernel, dtype=inputs.dtype)

        return tf.nn.depthwise_conv2d(
            inputs, kernel, strides=(1,) + self.strides + (1,), padding="SAME"
        )

    def get_config(self):
        config = {"taps": self.taps, "strides": self.strides}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))