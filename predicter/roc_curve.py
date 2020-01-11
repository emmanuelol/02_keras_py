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
from sklearn import metrics # sklearn.metrics で実行するとエラーになることがあったのでfromで呼ぶ
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

def plot_roc(out_png, y_true_list, y_pred_list,
             task_name_list = ['013471-0007<=10um', '013471-0007<=1um', '013471-0007<=0.1um', '013471-0007<=0.01um', '013471-0007<=0.001um', '013497-0003<=10um', '013497-0003<=1um', '013497-0003<=0.1um', '013497-0003<=0.01um', '013497-0003<=0.001um', '013498-0003<=10um', '013498-0003<=1um', '013498-0003<=0.1um', '013498-0003<=0.01um', '013498-0003<=0.001um']):
    """"
    推測結果のファイルからROC_curveをplotする
    Args:
        out_png:出力画像パス
        y_true_list:正解ラベルのnumpy.ndarrayのリスト [array([0,1,0,…]), array([1,0,0,…]),…]
        y_pred_list:予測スコアのnumpy.ndarrayのリスト [array([0.65,0.99,0.01,…]), array([0.1,0.33,0.95,…]),…]
        task_name_list:roc図の凡例につけるtask名
    Returns:
        なし（ROC_curveのグラフをファイル出力する）
    """
    print('------------------------------------')
    print('out_png:',out_png)

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
        y_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        y_df = y_df[y_df['y_true'] != -1]# 欠損ラベル=-1.0 以外の行だけにする
        y_true = np.array(y_df['y_true'])
        y_pred = np.array(y_df['y_pred'])
        # 欠損ラベルのレコードは削除して正解ラベル0件用対策
        if y_true.shape[0] == 0:
            print('all nan:', task_name_list[count])
        else:
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
            # fpr:偽陽性率 FP(間違って陽性判定した数) / (FP + TN(陰性全体の母数))
            # tpr:真陽性率 TP(正しく陽性判定した数) / (TP + FN(陽性全体の母数))
            # thresholds:閾値
            if np.isnan(tpr).all() == True:
                # posiデータ0件の時はtprがすべてnanになる
                print('positive 0:', task_name_list[count])
            elif np.isnan(fpr).all() == True:
                # negaデータ0件の時はfprがすべてnanになる
                print('negative 0:', task_name_list[count])
            else:
                tprs.append(scipy.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                # AUC計算
                roc_auc = metrics.auc(fpr, tpr)
                print(str(task_name_list[count])+'_roc_auc:', roc_auc)
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

        # 標準偏差の有効数字2桁にする(有効数字の指定はできなかったので、小数第5位を四捨五入するようにする)
        dec_std_auc = Decimal(std_auc).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        print('mean_auc:', mean_auc)
        print('std_auc:', std_auc)

        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.4f)' % (mean_auc, dec_std_auc), lw=2, alpha=.8)

        # 全ROC_curveの標準偏差分の領域をグレーにする
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.plot([0, 1], [0, 1], label='Random', color='r', linestyle='--') # 対角線に赤の点線引く（ランダムなパターンで AUC = 0.5 のライン）
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.legend(loc='lower right') # 凡例を右下に書く
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12) # 凡例を枠外に書く
    #plt.grid(True) # グリッド線書く
    plt.savefig(out_png, bbox_inches="tight") # plt.savefig はplt.show() の前に書かないと白紙で保存される
    plt.show()
    plt.clf() # plotの設定クリアにする
