# -*- coding: utf-8 -*-
"""自作loss関数"""

from keras import backend as K
import tensorflow as tf

def smooth_labels(y, smooth_factor=0.1):
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.
    label_smoothing：分類問題の正解ラベルの1を0.9みたいに下げ、0ラベルを0.1みたいに上げ、過学習軽減させる。間違っている正解ラベルが混じっているデータセットのときに有効
    train直前のワンホットラベルにラベルスムージングを適用する
    https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/
    Args
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)
    Returns
        A matrix of smoothed labels.
    Usage
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        print('Before smoothing: {}'.format(y_train[0]))
        smooth_labels(y_train, .1)
        print('After smoothing: {}'.format(y_train[0]))
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y

def build_label_smoothing_loss(loss_function=tf.losses.sigmoid_cross_entropy, smooth_factor=0.1):
    """
    label_smoothing：分類問題の正解ラベルの1を0.9みたいに下げ、0ラベルを0.1みたいに上げ、過学習軽減させる。間違っている正解ラベルが混じっているデータセットのときに有効
    tensorflowの分類lossにラベルスムージングを適用する
    https://stackoverflow.com/questions/55894459/how-to-use-tf-losses-sigmoid-cross-entropy-with-label-smoothing-in-keras
    Args
        loss_function:損失関数
            二値分類の場合:tf.losses.sigmoid_cross_entropy(tensorflow v1.8のloss関数(Lib/site-packages/tensorflow/python/ops/losses/losses_impl.py)にはlabel_smoothing実装されている)
            多クラス分類の場合:tf.losses.softmax_cross_entropy(tensorflow v1.8のloss関数(Lib/site-packages/tensorflow/python/ops/losses/losses_impl.py)にはlabel_smoothing実装されている)
        smooth_factor: label smoothing factor (between 0 and 1)
    """
    def label_smoothing_loss(y_true, y_pred):
        return loss_function(y_true, y_pred, label_smoothing=smooth_factor)

# focal loss: 通常の cross entropy loss (CE) を動的に scaling（高い推論確率で正しく推論できているサンプルの損失をCross Entropy Lossより小さく）させる損失関数
# Focal Loss は - (1-p_t)^γ ln(p_t) のような形
# Positive-Negative間の不均衡が大きい場合に効くlossらしい.
# マイノリティクラスでも高い精度を達成できるようにするために、フォーカルロスを使用して、トレーニング中にこれらのマイノリティクラスの例に相対的な重みを付ける
# ハイパーパラメータは gamma(0-5程度) と alpha(0.25-0.75程度) があり、この2つのbest値の探索必要
# gamma:高い推論確率のサンプルに対する損失の減衰量を調整するハイパーパラメタ。γ=0 の場合、Cross Entropy Lossと一致
# alpha:クラスアンバランスを補正するハイパーパラメタ。クラス間の重み付けで、classes_weightに関連するパラメ
def binary_focal_loss(gamma=2., alpha=.25):
    """
    https://github.com/umbertogriffo/focal-loss-keras
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    コンパイル済みモデルでロードする場合はcustom_objectsが必要
    # https://github.com/umbertogriffo/focal-loss-keras
    import dill
    custom_object = {'binary_focal_loss_fixed': dill.loads(dill.dumps(my_loss.binary_focal_loss(gamma=2., alpha=.25))),
                     'binary_focal_loss': my_loss.binary_focal_loss}
    model = keras.models.load_model(r'output_test\model.h5', custom_objects=custom_object)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return binary_focal_loss_fixed

def categorical_focal_loss(gamma=2., alpha=.25):
    """
    https://github.com/umbertogriffo/focal-loss-keras
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

    コンパイル済みモデルでロードする場合はcustom_objectsが必要
    # https://github.com/umbertogriffo/focal-loss-keras
    import dill
    custom_object = {'categorical_focal_loss_fixed': dill.loads(dill.dumps(my_loss.categorical_focal_loss(gamma=2., alpha=.25))),
                     'categorical_focal_loss': my_loss.categorical_focal_loss}
    model = keras.models.load_model(r'output_test\model.h5', custom_objects=custom_object)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)
    return categorical_focal_loss_fixed

# focal loss with multi label
def multi_label_focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    """
    マルチラベル用 focal loss うまく機能しない。。。
    https://github.com/MLearing/Keras-Focal-Loss
    Args:
        classes_num:各クラスのサンプル数のリスト. ex [100, 1, 1]
        gamma:高い推論確率のサンプルに対する損失の減衰量を調整するハイパーパラメタ。γ=0 の場合、Cross Entropy Lossと一致
        alpha:クラスアンバランスを補正するハイパーパラメタ
    """
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        #import tensorflow as tf
        from tensorflow.python.ops import array_ops
        #from keras import backend as K
        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))
        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor
        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)
        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_sum(balanced_fl)
        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)
        return fianal_loss
    return focal_loss_fixed

if __name__ == '__main__':
    print('my_loss.py: loaded as script file')
else:
    print('my_loss.py: loaded as module file')
