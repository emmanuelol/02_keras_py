# -*- coding: utf-8 -*-
""" 自作loss関数 """
from tensorflow import keras
import tensorflow as tf
import numpy as np

def build_masked_loss(loss_function, mask_value=-1.0):
    """
    ラベル付けされたデータを「マスクする」損失関数
    https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/
    https://github.com/keras-team/keras/issues/3893
    Args:
        loss_func: The loss function to mask（keras.backend.binary_crossentropy or keras.backend.categorical_crossentropy）
        mask_value: The value to mask in the targets（欠損ラベルとしてマスクする値）
    Returns:
        function: a loss function that acts like loss_function with masked inputs
    Usage:
        # コンパイル済みモデルでロードする場合はcustom_objectsが必要
        import dill
        multi_label_focal_loss = my_loss.multi_label_focal_loss([6354407, 8213], alpha=0.25, gamma=2)
        custom_object = {'masked_loss_function': dill.loads(dill.dumps(multi_loss.build_masked_loss(multi_label_focal_loss, mask_value=multi_loss.mask_value))}
        model = keras.models.load_model(r'output_test\model.h5', custom_objects=custom_object)
    """
    # 型を合わせないとエラーになる
    mask_value = keras.backend.cast(mask_value, keras.backend.floatx())
    def masked_loss_function(y_true, y_pred):
        y_true = keras.backend.cast(y_true, keras.backend.floatx())
        y_pred = keras.backend.cast(y_pred, keras.backend.floatx())
        mask = keras.backend.cast(keras.backend.not_equal(y_true, mask_value), keras.backend.floatx())
        # 欠損の要素が削除される。テンソルのshapeが[batch_size, クラス数+1(欠損クラス)]→[batch_size, クラス数]になる
        #print(y_true, y_pred, mask, mask_value)
        return loss_function(y_true * mask, y_pred * mask)
    return masked_loss_function

def masked_accuracy(y_true, y_pred, mask_value=-1.0):
    """
    ラベル付けされたデータを「マスクする」正解率
    https://github.com/keras-team/keras/issues/3893
    sigmoid+keras.backend.binary_crossentropyのときはうまくいく
    """
    # 型を合わせないとエラーになる
    mask_value = keras.backend.cast(mask_value, keras.backend.floatx())
    y_true = keras.backend.cast(y_true, keras.backend.floatx())
    y_pred = keras.backend.cast(y_pred, keras.backend.floatx())

    total = keras.backend.sum(keras.backend.cast(keras.backend.not_equal(y_true, mask_value), keras.backend.floatx()))# 要素の総数。ラベルなし要素は総数から除外
    correct = keras.backend.sum(keras.backend.cast(keras.backend.equal(y_true, keras.backend.round(y_pred)), keras.backend.floatx()))# 正解と予測が等しい数
    return correct / total# 正解数/ラベルなし除外した総数


# focal loss: 通常の cross entropy loss (CE) を動的に scaling（高い推論確率で正しく推論できているサンプルの損失をCross Entropy Lossより小さく）させる損失関数
# Focal Loss は - (1-p_t)^γ ln(p_t) のような形
# Positive-Negative間の不均衡が大きい場合に効くlossらしい.
# マイノリティクラスでも高い精度を達成できるようにするために、フォーカルロスを使用して、トレーニング中にこれらのマイノリティクラスの例に相対的な重みを付ける
# ハイパーパラメータは gamma(0-5程度) と alpha(0.25-0.75程度) があり、この2つのbest値の探索必要
# gamma:高い推論確率のサンプルに対する損失の減衰量を調整するハイパーパラメタ。γ=0 の場合、Cross Entropy Lossと一致
# alpha:クラスアンバランスを補正するハイパーパラメタ。クラス間の重み付けで、classes_weightに関連するパラメ
# FocalLossの論文解説日本語記事：https://qiita.com/agatan/items/53fe8d21f2147b0ac982
def binary_focal_loss(gamma=2., alpha=.25, mask_value=None):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

     # モデルロード時は以下のようにcustom_objectが必要
     import dill
     custom_object = {'binary_focal_loss_fixed': dill.loads(dill.dumps(binary_focal_loss(gamma=2., alpha=.25)))
                      , 'categorical_focal_loss_fixed': dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=.25)))
                      , 'categorical_focal_loss': categorical_focal_loss
                      , 'binary_focal_loss': binary_focal_loss}
      model = keras.models.load_model(input_model_path, custom_objects=custom_object)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        # 型を合わせないとエラーになる
        y_true = keras.backend.cast(y_true, keras.backend.floatx())
        y_pred = keras.backend.cast(y_pred, keras.backend.floatx())

        # 欠損ラベル指定あれば取り除く
        if mask_value is not None:
            _mask_value = keras.backend.cast(mask_value, keras.backend.floatx())
            mask = keras.backend.cast(keras.backend.not_equal(y_true, _mask_value), keras.backend.floatx())
            y_true = y_true * mask
            y_pred = y_pred * mask

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = keras.backend.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = keras.backend.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = keras.backend.clip(pt_0, epsilon, 1. - epsilon)

        return -keras.backend.sum(alpha * keras.backend.pow(1. - pt_1, gamma) * keras.backend.log(pt_1)) \
               -keras.backend.sum((1 - alpha) * keras.backend.pow(pt_0, gamma) * keras.backend.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25, mask_value=None):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

     # モデルロード時は以下のようにcustom_objectが必要
     import dill
     custom_object = {'binary_focal_loss_fixed': dill.loads(dill.dumps(binary_focal_loss(gamma=2., alpha=.25)))
                      , 'categorical_focal_loss_fixed': dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=.25)))
                      , 'categorical_focal_loss': categorical_focal_loss
                      , 'binary_focal_loss': binary_focal_loss}
      model = keras.models.load_model(input_model_path, custom_objects=custom_object)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # 型を合わせないとエラーになる
        y_true = keras.backend.cast(y_true, keras.backend.floatx())
        y_pred = keras.backend.cast(y_pred, keras.backend.floatx())

        # 欠損ラベル指定あれば取り除く
        if mask_value is not None:
            _mask_value = keras.backend.cast(mask_value, keras.backend.floatx())
            mask = keras.backend.cast(keras.backend.not_equal(y_true, _mask_value), keras.backend.floatx())
            y_true = y_true * mask
            y_pred = y_pred * mask

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= keras.backend.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * keras.backend.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * keras.backend.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return keras.backend.sum(loss, axis=1)

    return categorical_focal_loss_fixed

class FocalLoss_multilabel(object):
    """
    マルチラベル+cross_entropy_lossの時のFocalLoss
    https://github.com/keras-team/keras/issues/10371
    loss関数を引数で渡す様にした

    FocalLossの論文解説英語記事：
    https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab
    論文ではCIFAR10のクラス不均衡が100倍、200倍の時focal loss有効だった

    FocalLossの論文解説日本語記事：
    https://qiita.com/agatan/items/53fe8d21f2147b0ac982

    Usage:
        f_loss = FocalLoss_multilabel(loss_function=build_masked_loss(keras.backend.binary_crossentropy)).compute_loss
        model.compile(loss=f_loss, optimizer='nadam', metrics=['accuracy', masked_accuracy])
        model.fit_generator(…)
    """
    def __init__(self, gamma=2, alpha=0.25, loss_function=keras.backend.binary_crossentropy):
        self._gamma = gamma
        self._alpha = alpha
        self._loss_function = loss_function

    def compute_loss(self, y_true, y_pred):
        cross_entropy_loss = self._loss_function(y_true, y_pred)
        p_t = ((y_true * y_pred) + ((1 - y_true) * (1 - y_pred)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = tf.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (y_true * self._alpha + (1 - y_true) * (1 - self._alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * cross_entropy_loss)

        return keras.backend.mean(focal_cross_entropy_loss, axis=-1)
