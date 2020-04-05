# -*- coding: utf-8 -*-
"""自作metric"""

import tensorflow as tf

def get_lr_metric(optimizer):
    """
    エポックごとの学習率出力
    https://stackoverflow.com/questions/48198031/keras-add-variables-to-progress-bar/48206009#48206009
    """
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def get_auc_metric(optimizer):
    """
    エポックごとのAUC出力
    https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
    マルチラベルではうまくいかない（欠損値除く処理入れたいが、変数がTensorオブジェクトなのでうまく変更できない）
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    def roc_auc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
    return roc_auc
