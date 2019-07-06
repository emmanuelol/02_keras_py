# -*- coding: utf-8 -*-
"""
推測結果のファイルからROC_curveをplotする

Usage:
import roc_curve

model = keras.models.load_model(os.path.join(out_dir, 'best_model.h5'), compile=False)
y_pred = model.predict(x_test)

y_test_list = []
for i in range(y_test.shape[1]):
    y_test_list.append(y_test[:,i])

y_pred_list = []
for i in range(y_pred.shape[1]):
    y_pred_list.append(y_pred[:,i])

out_png = os.path.join(out_dir, 'ROC_curve.png')
roc_curve.plot_roc(out_png, y_test_list, y_pred_list)
"""
import sklearn
from sklearn import metrics# sklearn.metrics で実行するとエラーになることがあったのでfromで呼ぶ
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_roc(out_png, y_true_list, y_pred_list
            , task_name_list = ['NR.AhR', 'NR.AR', 'NR.AR.LBD', 'NR.Aromatase', 'NR.ER', 'NR.ER.LBD', 'NR.PPAR.gamma', 'SR.ARE', 'SR.ATAD5', 'SR.HSE', 'SR.MMP', 'SR.p53']
            ):
    """"
    推測結果のファイルからROC_curveをplotする
    Args:
        out_png:出力画像パス
        y_true_list:正解ラベルのnumpy.ndarrayのリスト [array([0,1,0,…]), array([1,0,0,…]),…]
        y_pred_list:予測スコアのnumpy.ndarrayのリスト [array([0.65,0.99,0.01,…]), array([0.1,0.33,0.95,…]),…]
        task_name_list:凡例につけるtask名 デフォルトはTox21のタスク名
    Returns:
        なし（ROC_curveのグラフをファイル出力する）
    """
    print('------------------------------------')
    print('out_png:',out_png)

    # 凡例につけるtask名
    #task_name_list = ['NR.AhR', 'NR.AR', 'NR.AR.LBD', 'NR.Aromatase', 'NR.ER', 'NR.ER.LBD', 'NR.PPAR.gamma', 'SR.ARE', 'SR.ATAD5', 'SR.HSE', 'SR.MMP', 'SR.p53']

    # デフォルトのfigsizeを設定して画像の表示を大きくする
    #plt.rcParams['figure.figsize'] = (7.0, 7.0)
    plt.figure(figsize=(7, 7)) # figureの縦横の大きさ設定

    tprs = [] # 真陽性率のリスト
    aucs = [] # 偽陽性率のリスト
    mean_fpr = np.linspace(0, 1, 100)

    # 複数の推測結果から出したROC_curveを1つのグラフ乗せるために、y_pred_listとy_true_listをforで回す
    count = 0
    for (y_pred, y_true) in zip(y_pred_list, y_true_list):

        # metrics.roc_curveはy_trueが2種類でないとダメなので、欠損ラベルのレコードは削除する
        if len(np.unique(y_true)) != 2:
            y_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
            y_df = y_df[y_df['y_true'] != -1.0]# 欠損ラベル=-1.0 以外の行だけにする
            y_true = np.array(y_df['y_true'])
            y_pred = np.array(y_df['y_pred'])

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        # fpr:偽陽性率 FP(間違って陽性判定した数) / (FP + TN(陰性全体の母数))
        # tpr:真陽性率 TP(正しく陽性判定した数) / (TP + FN(陽性全体の母数))
        # thresholds:閾値
        tprs.append(scipy.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # AUC計算
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        # ROC_curve plot
        plt.plot(fpr, tpr, lw=1, alpha=0.4, label='%s (AUC = %0.2f)' % (task_name_list[count], roc_auc))
        count += 1

    # y_pred_listやy_true_listが複数あるなら平均値もplotする
    if len(y_pred_list) > 1:
        # ROC_curveの平均値出しplotする
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

        # 全ROC_curveの標準偏差分の領域をグレーにする
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.plot([0, 1], [0, 1], label='Luck', color='r', linestyle='--') # 対角線に赤の点線引く（ランダムなパターンで AUC = 0.5 のライン）
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    #plt.grid(True) # グリッド線書く
    plt.savefig(out_png) # plt.savefig はplt.show() の前に書かないと白紙で保存される
    plt.show()
    plt.clf() # plotの設定クリアにする

if __name__ == '__main__':
    print('roc_curve.py: loaded as script file')
else:
    print('roc_curve.py: loaded as module file')
