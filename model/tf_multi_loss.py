# -*- coding: utf-8 -*-
"""
マルチタスク用loss関数

Usage:
    import os, sys
    current_dir = os.path.dirname(os.path.abspath("__file__"))
    path = os.path.join(current_dir, '../')
    sys.path.append(path)
    from model import define_model

    output_dir = r'output_test\100x100'
    img_rows,img_cols,channels = 100,100,3
    num_classes = 10
    chaice_model = 'EfficientNet'
    choice_optim = 'adam'

    # モデル作成
    model, orig_model = define_model.get_fine_tuning_model(output_dir, img_rows, img_cols, channels, num_classes, chaice_model, efficientnet_num=7)
    # オプティマイザ
    optim = define_model.get_optimizers(choice_optim)
    # モデルコンパイル
    model.compile(loss=loss.build_masked_loss(keras.backend.binary_crossentropy)
                    , optimizer=optim, metrics=['accuracy'])
"""
from tensorflow import keras

mask_value=-1

def build_masked_loss(loss_function, mask_value=-1):
    """
    ラベル付けされたデータを「マスクする」損失関数
    https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/
    https://github.com/keras-team/keras/issues/3893
    Args:
        loss_func: The loss function to mask（keras.backend.binary_crossentropy or keras.backend.categorical_crossentropy）
        mask_value: The value to mask in the targets（欠損ラベルとしてマスクする値）
    Returns:
        function: a loss function that acts like loss_function with masked inputs

    コンパイル済みモデルでロードする場合はcustom_objectsが必要
    import dill
    multi_label_focal_loss = my_loss.multi_label_focal_loss([6354407, 8213], alpha=0.25, gamma=2)
    custom_object = {'masked_loss_function': dill.loads(dill.dumps(multi_loss.build_masked_loss(multi_label_focal_loss, mask_value=multi_loss.mask_value))}
    model = keras.models.load_model(r'output_test\model.h5', custom_objects=custom_object)
    """
    def masked_loss_function(y_true, y_pred):
        mask = keras.backend.cast(keras.backend.not_equal(y_true, mask_value), keras.backend.floatx())
        #print(y_true, y_pred, mask, mask_value)
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function

def masked_accuracy(y_true, y_pred):
    """
    ラベル付けされたデータを「マスクする」正解率
    https://github.com/keras-team/keras/issues/3893
    sigmoid+keras.backend.binary_crossentropyのときはうまくいく
    """
    total = keras.backend.sum(keras.backend.cast(keras.backend.not_equal(y_true, mask_value), keras.backend.floatx()))# 要素の総数。ラベルなし要素は総数から除外
    correct = keras.backend.sum(keras.backend.cast(keras.backend.equal(y_true, keras.backend.round(y_pred)), keras.backend.floatx()))# 正解と予測が等しい数
    return correct / total# 正解数/ラベルなし除外した総数
