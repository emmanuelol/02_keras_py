# -*- coding: utf-8 -*-
"""自作loss関数"""
import keras
import tensorflow as tf

# focal loss: 通常の cross entropy loss (CE) を動的に scaling（高い推論確率で正しく推論できているサンプルの損失をCross Entropy Lossより小さく）させる損失関数
# Focal Loss は - (1-p_t)^γ ln(p_t) のような形
# Positive-Negative間の不均衡が大きい場合に効くlossらしい.
# マイノリティクラスでも高い精度を達成できるようにするために、フォーカルロスを使用して、トレーニング中にこれらのマイノリティクラスの例に相対的な重みを付ける
# ハイパーパラメータは gamma(0-5程度) と alpha(0.25-0.75程度) があり、この2つのbest値の探索必要
# gamma:高い推論確率のサンプルに対する損失の減衰量を調整するハイパーパラメタ。γ=0 の場合、Cross Entropy Lossと一致
# alpha:クラスアンバランスを補正するハイパーパラメタ。クラス間の重み付けで、classes_weightに関連するパラメ
class FocalLoss(object):
    """
    cross_entropy_lossの時のFocalLoss
    https://github.com/keras-team/keras/issues/10371
    loss関数を引数で渡す様にした

    FocalLossの論文解説記事：
    https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab
    論文ではCIFAR10のクラス不均衡が100倍、200倍の時focal loss有効だった

    Usage:
        f_loss = multi_loss.FocalLoss(loss_function=multi_loss.build_masked_loss(keras.backend.binary_crossentropy)).compute_loss
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
