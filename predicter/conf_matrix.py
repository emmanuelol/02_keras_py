# -*- coding: utf-8 -*-
"""
混同行列を作成する
http://adliblog.hatenablog.com/entry/2018/02/15/184020

Usage:
import conf_matrix

# ファイルからモデルロード
trained_model = define_model_classes.load_json_weight('/gpfsx01/home/tmp10014/jupyterhub/notebook/work_H3-031/submit_Multi_Images/FineTuning/storage/Results/InceptionV3/Nadam_FC3/classes_Augmentor_lower_lr/DataSets_060Degree_cut_test_all_degree_all_train/CV1')

# クラスごとに出力する確信度の推測結果ファイル格納先
out_score_dir = os.path.join(out_dir, "predict_class_score_test")

#Jupyterでインライン表示するための宣言
%matplotlib inline

# 入力画像のパスがディレクトリの場合、下位ディレクトリのファイルパスを再帰的に取得する
img_file_list = []
for file in util.find_all_files(test_dir):
    if '.jpg' in file:
        img_file_list.append(file)

# 混同行列を作成する
conf_matrix.make_confusion_matrix(trained_model, classes, img_file_list, out_score_dir, img_rows, img_cols)
#<モデル>, <クラスリスト>, <画像ファイルパスリスト>, <ファイル出力するディレクトリ>, <入力層のサイズ（縦）>, <入力層のサイズ（横）>
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys, sklearn, itertools
from sklearn import metrics# sklearn.metrics で実行するとエラーになることがあったのでfromで呼ぶ

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

def make_confusion_matrix(classes, y_true, y_pred, out_dir, figsize=(6, 4)):
    """
    混同行列を作成する
    Args:
        classes:分類クラスのリスト　[0,1,-1]
        y_true:正解ラベルのnumpy.ndarray　array([0,1,0,…]) (or array(['positive', 'positive', 'nagative', …]) )
        y_pred:予測ラベルのnumpy.ndarray　array([0,0,1,…]) (or array(['positive', 'nagative', 'nagative', …]) )
        out_dir:出力先のディレクトリパス
        figsize: 混同行列のplotサイズ
    Returns:
        なし（混同行列のファイルを出力する）
    """
    print('------------------------------------')
    print('out_dir:', out_dir)

    # 有効桁数を下2桁とする
    np.set_printoptions(precision=2)
    # accuracyの計算
    accuracy = metrics.accuracy_score(y_true, y_pred)
    # confusion matrixの作成
    cnf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=classes)
    np.savetxt(out_dir+"/confusion_matrix.txt", cnf_matrix, delimiter='\t')
    # report(各種スコア)の作成と保存
    report = metrics.classification_report(y_true, y_pred, labels=classes)
    with open(out_dir+"/report.txt", mode='w') as f:
        f.write(report)
    print(report)
    # confusion matrixのプロット、保存、表示
    # http://adliblog.hatenablog.com/entry/2018/02/15/184020 より
    title="overall accuracy:"+str(accuracy)
    # 混同行列
    plot_confusion_matrix(cnf_matrix, classes=classes, output_file=out_dir+"/CM_without_normalize.png", title=title, figsize=figsize)
    # 正規化した混同行列
    #plot_confusion_matrix(cnf_matrix, classes=classes, output_file=out_dir+"/CM_normalized.png", normalize=True,title=title)

def binary_multi_confmx(classes, y_true_list, y_pred_list, out_dir, figsize=(6, 4)):
    """
    タスクごとのpredictのスコア(y_true_list)と正解ラベル(y_pred_list)から混同行列をファイル出力
    →マルチラベルを分解して、make_confusion_matrix関数を呼び混同行列作成する
    Args:
        classes:分類クラスのリスト 例　[0,1,-1]
        y_true_list:正解ラベルのnumpy.ndarrayのリスト [array([0,1,0,…]), array([1,0,0,…]),…] ( リストの長さはタスク数、各arrayはファイルごとのラベル )
        y_pred_list:予測スコアのnumpy.ndarrayのリスト [array([0.65,0.99,0.01,…]), array([0.1,0.33,0.95,…]),…] ( リストの長さはタスク数、各arrayはファイルごとのスコア )
        out_dir:出力先のディレクトリパス
        figsize: 混同行列のplotサイズ
    """
    # タスクのループカウント
    count=0
    for (y_pred, y_true) in zip(y_pred_list, y_true_list):
        # ファイル出力先作成
        conf_out_dir = os.path.join(out_dir, 'confusion_matrix', 'task'+str(count))
        os.makedirs(conf_out_dir, exist_ok=True)
        # Tox21はバイナリ分類なので、確信度が0.5より大きい推論を1、それ以外を0に置換する
        y_pred = (y_pred > 0.5) * 1.0
        # 混同行列作成
        make_confusion_matrix(classes, y_true, y_pred, conf_out_dir, figsize=figsize)
        count+=1

if __name__ == '__main__':
    print('conf_matrix.py: loaded as script file')
else:
    print('conf_matrix.py: loaded as module file')
