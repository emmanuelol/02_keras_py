# -*- coding: utf-8 -*-
"""
クラスごとに重み変えたいときに使う
fit_generator( , class_weight={} )class_weight 作成用関数
"""

import os, sys, math
import numpy as np

def cal_weight_flow(class_id_list):
    """
    https://qiita.com/Alshain/items/3415b14c077cbfdadaea
    flowを使用する場合に使える、classs_weightの辞書型作成関数
    class_id_list はラベルのリストをkerasのto_categoricalで変換したもの
    class_id_list作成例：
        from keras.utils.np_utils import to_categorical
        class_name = ['猫','犬']
        class_list = ['猫','犬','猫','犬','猫','猫']
        def make_id_list (label):
            return class_name.index(label)
        class_id_list = to_categorical(list(map(make_id_list, class_list)))
        print(class_id_list) # array([[ 1.,  0.],[ 0.,  1.],…
    """
    amounts_of_class_array = np.zeros(len(class_id_list[0]))
    for class_id in class_id_list:
        amounts_of_class_array = amounts_of_class_array + class_id
    mx = np.max(amounts_of_class_array)
    class_weights = {}
    for q in range(0,len(amounts_of_class_array)):
        class_weights[q] = round(float(math.pow(amounts_of_class_array[q]/mx, -1)),2)
    return class_weights

def cal_weight(class_name_list, IN_DIR):
    """
    https://qiita.com/Alshain/items/3415b14c077cbfdadaea
    flow_from_directoryを使用する場合に使える、classs_weightの辞書型作成関数
    使用する学習データはクラス名が書かれた画像のフォルダを使用することを想定しています。
    class_name_listに学習に使うクラスのlistを入力し、IN_DIRにクラス名が書かれた画像フォルダのディレクトリを入力してください。
    """
    amounts_of_class_dict = {}
    mx = 0
    for class_name in class_name_list:
        class_dir = IN_DIR + os.sep + class_name
        file_list = os.listdir(class_dir)
        amounts_of_class_dict[class_name] = len(file_list)
        if mx < len(file_list):
            mx = len(file_list)
    class_weights = {}
    count = 0
    for class_name in class_name_list:
        class_weights[count] = round(float(math.pow(amounts_of_class_dict[class_name]/mx, -1)),2) #重み＝（データ数/最大値）の逆数
        count += 1
    return class_weights
