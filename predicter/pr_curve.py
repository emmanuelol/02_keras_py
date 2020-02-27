# -*- coding: utf-8 -*-
"""
推測結果のファイルからPR(Precision-Recall)_curveをplotする
ROC曲線は、不均衡データに対する分類器の性能を過大評価する可能性がありますが、PR曲線は、不均衡データの再現率と同じレベルで精度の低下を示す

Usage:
    import pr_curve

    model = keras.models.load_model(os.path.join(out_dir, 'best_model.h5'), compile=False)
    y_pred = model.predict(x_test)

    y_test_list = []
    for i in range(y_test.shape[1]):
        y_test_list.append(y_test[:,i])

    y_pred_list = []
    for i in range(y_pred.shape[1]):
        y_pred_list.append(y_pred[:,i])

    out_png = os.path.join(out_dir, 'ROC_curve.png')
    pr_curve.plot_pr(y_test_list, y_pred_list, out_png)
"""
import numpy as np
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, plot_precision_recall_curve
from scipy.optimize import minimize

def plot_pr(y_true, y_score, out_png='PR.png', mask_value=-1.0):
    """
    二値分類の推測結果からPR(Precision-Recall)曲線plot
    Args:
        y_true:正解ラベルのリスト [0,1,0,…]
        y_score:予測スコアのリスト [0.65,0.99,0.01,…]
        out_png:出力画像パス
    Returns:
        なし（pr_curveのグラフをファイル出力する）
    """

    # metrics.roc_curveはy_trueが2種類でないとダメなので、欠損ラベルのレコードは削除する
    y_df = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
    y_df = y_df[y_df['y_true'] != mask_value]# 欠損ラベル=-1.0 以外の行だけにする
    y_true = np.array(y_df['y_true'])
    y_score = np.array(y_df['y_score'])

    # 欠損ラベルのレコードは削除して正解ラベル0件用対策
    if y_true.shape[0] == 0:
        print('all nan')
        return None
    else:
        _precision, _recall, _thresholds = precision_recall_curve(y_true, y_score)
        if np.isnan(_recall[0]) == True:
            # posiデータ0件の時はrecall[0]nanになる
            print('positive 0')
            return None
        else:
            average_precision = average_precision_score(y_true, y_score)
            print('Average precision-recall score: {0:0.8f}'.format(average_precision))
            precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig(out_png, bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()


def plot_micro_averaged_PR(n_classes, Y_true, Y_score, out_png='micro_averaged_PR.png', mask_value=-1.0):
    """
    マルチラベル分類（複数タスクの二値分類）のミクロ平均PR曲線plot
    ミクロ平均は全クラスの結果を1つにまとめてからPR曲線の面積取ったもの
    Args:
        n_classes: クラス（タスク）数(2とか3のint)
        Y_true:正解ラベルのリストのリスト [[0,1,0,…],[1,0,0,…],…]
        Y_score:予測スコアのリストのリスト [[0.65,0.99,0.01,…],[0.1,0.75,0.51,…],…]
        out_png:出力画像パス
    Returns:
        なし（pr_curveのグラフをファイル出力する）
    """
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    re_Y_true = []
    re_Y_score = []
    re_n_classes = []
    for i in range(n_classes):
        y_true = Y_true[i]
        y_score = Y_score[i]

        # metrics.roc_curveはy_trueが2種類でないとダメなので、欠損ラベルのレコードは削除する
        y_df = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
        y_df = y_df[y_df['y_true'] != mask_value]# 欠損ラベル=-1.0 以外の行だけにする
        y_true = np.array(y_df['y_true'])
        y_score = np.array(y_df['y_score'])
        # 欠損ラベルのレコードは削除して正解ラベル0件用対策
        if y_true.shape[0] == 0:
            print('all nan:', 'class_'+str(i))
        else:
            _precision, _recall, _thresholds = precision_recall_curve(y_true, y_score)
            # AUC計算
            _average_precision = average_precision_score(y_true, y_score)
            if np.isnan(_recall[0]) == True:
                # posiデータ0件の時はrecall[0]nanになる
                print('positive 0:', 'class_'+str(i))
            else:
                re_Y_true.append(y_true)
                re_Y_score.append(y_score)
                re_n_classes.append(i)
                precision[i] = _precision
                recall[i] = _recall
                average_precision[i] = _average_precision
                print('Average precision-recall score: {0:0.8f}'.format(average_precision[i]))

    # 後続処理の .ravel()（numpy型を1次元に変換）でエラー出さないようにlistならnumpyに変換する
    if isinstance(re_Y_true, list):
        Y_true_flatten = [flatten for inner in re_Y_true for flatten in inner]
        Y_true_flatten = np.array(Y_true_flatten)
        Y_score_flatten = [flatten for inner in re_Y_score for flatten in inner]
        Y_score_flatten = np.array(Y_score_flatten)
    else:
        Y_true_flatten = re_Y_true.ravel()
        Y_score_flatten = re_Y_score.ravel()

    # A "micro-average": quantifying score on all classes jointly
    # ミクロ平均は全クラスの結果を1つにまとめてからPR曲線の面積取ったもの
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_true_flatten, Y_score_flatten)
    average_precision["micro"] = average_precision_score(Y_true_flatten, Y_score_flatten, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.8f}'.format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    plt.savefig(out_png, bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()


def plot_each_class_PR(n_classes, Y_true, Y_score, out_png='each_class_PR'
                        , task_name_list = ['NR.AhR', 'NR.AR', 'NR.AR.LBD', 'NR.Aromatase', 'NR.ER', 'NR.ER.LBD', 'NR.PPAR.gamma', 'SR.ARE', 'SR.ATAD5', 'SR.HSE', 'SR.MMP', 'SR.p53']
                        , mask_value=-1.0
                      ):
    """
    マルチラベル分類（複数タスクの二値分類）のミクロ平均と各タスクのPR曲線plot
    ミクロ平均は全クラスの結果を1つにまとめてからPR曲線の面積取ったもの
    Args:
        n_classes: クラス（タスク）数(2とか3のint)
        Y_true:正解ラベルのリストのリスト [[0,1,0,…],[1,0,0,…],…]
        y_score:予測スコアのリストのリスト [[0.65,0.99,0.01,…],[0.1,0.75,0.51,…],…]
        out_png:出力画像パス
        task_name_list:凡例につけるtask名
    Returns:
        なし（pr_curveのグラフをファイル出力する）
    """
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    re_Y_true = []
    re_Y_score = []
    re_n_classes = []
    for i in range(n_classes):
        y_true = Y_true[i]
        y_score = Y_score[i]

        # metrics.roc_curveはy_trueが2種類でないとダメなので、欠損ラベルのレコードは削除する
        y_df = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
        y_df = y_df[y_df['y_true'] != mask_value]# 欠損ラベル=-1.0 以外の行だけにする
        y_true = np.array(y_df['y_true'])
        y_score = np.array(y_df['y_score'])

        # 欠損ラベルのレコードは削除して正解ラベル0件用対策
        if y_true.shape[0] == 0:
            print('all nan:', task_name_list[i])
        else:
            _precision, _recall, _thresholds = precision_recall_curve(y_true, y_score)
            # AUC計算
            # average_precision_score()はPR曲線では、ROC曲線のAUC出すmetrics.auc()は正確ではないみたい
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
            _average_precision = average_precision_score(y_true, y_score) # auc = metrics.auc(recall[i],precision[i])
            #print(_precision, _recall, _thresholds, _average_precision)
            if np.isnan(_recall[0]) == True:
                # posiデータ0件の時はrecall[0]nanになる
                print('positive 0:', task_name_list[i])
            else:
                re_Y_true.append(y_true)
                re_Y_score.append(y_score)
                re_n_classes.append(i)
                precision[i] = _precision
                recall[i] = _recall
                average_precision[i] = _average_precision
                print(str(task_name_list[i])+'_roc_auc:', _average_precision)


    # 後続処理の .ravel()（numpy型を1次元に変換）でエラー出さないようにlistならnumpyに変換する
    if isinstance(re_Y_true, list):
        Y_true_flatten = [flatten for inner in re_Y_true for flatten in inner]
        Y_true_flatten = np.array(Y_true_flatten)
        Y_score_flatten = [flatten for inner in re_Y_score for flatten in inner]
        Y_score_flatten = np.array(Y_score_flatten)
    else:
        Y_true_flatten = re_Y_true.ravel()
        Y_score_flatten = re_Y_score.ravel()

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], thresholds = precision_recall_curve(Y_true_flatten, Y_score_flatten)
    average_precision["micro"] = average_precision_score(Y_true_flatten, Y_score_flatten, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.8f}'.format(average_precision["micro"]))

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    lines.append(l)
    labels.append('iso-f1 curves')

    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    #for i, color in zip(range(n_classes), colors):
    for i in re_n_classes:
        l, = plt.plot(recall[i], precision[i], lw=2, alpha=0.4)
        lines.append(l)
        #labels.append('Precision-recall for class {0} (area = {1:0.2f})'
        #              ''.format(i, average_precision[i]))
        labels.append(task_name_list[i]+' (area = {0:0.2f})'
                      ''.format(average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    #plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14)) # 凡例下に置く
    plt.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12) # 凡例グラフの横に書く
    plt.savefig(out_png, bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()

def plot_pr_thresholds(test_labels, predictions, out_png='pr_thresholds', title='pr_thresholds', is_queue_rate=False, mask_value=-1.0):
    """
    二値分類でf1 score最大になる閾値確認するためprecision, recallのクロス線plot
    Args:
        test_labels: 正解ラベル e.g: np.array([0, 0, 1, 1])
        predictions: 予測ラベル e.g: np.array([0.1, 0.4, 0.35, 0.8])
        out_png: plot画像出力パス
        title: plot画像のタイトル
        is_queue_rate: plotにqueue_rate入れるか
    Returns:
        df_plot_data: df_plot_data: plot画像のデータフレーム( thresholds, precision, recall, f1, queue_rate )
        f1_max_threshold: f1最大の閾値 (f1最大の閾値=df_plot_dataのf1列のmax値)
        precision_max_threshold: precision最大の閾値 (f1最大の閾値=df_plot_dataのprecision列のmax値)
        recall_max_threshold: recall最大の閾値 (f1最大の閾値=df_plot_dataのrecall列のmax値)
    """
    import seaborn as sns
    from sklearn.metrics import precision_recall_curve

    # precision, recall, thresholds(閾値)
    precision, recall, thresholds = precision_recall_curve(test_labels, predictions)
    thresholds = np.append(thresholds, 1)

    # 閾値ごとにf1計算
    f1_list = []
    for p,r,t in zip(precision, recall, thresholds):
        f1 = 2 * (p * r) / (p + r)
        f1_list.append(f1)

    # thresholds, precision, recall, f1, queue_rate のデータフレーム作成
    if is_queue_rate == True:
        # queue_rate 入れるか
        queue_rate = []
        for threshold in thresholds:
            queue_rate.append((predictions >= threshold).mean())

        df_plot_data = pd.DataFrame({'thresholds': thresholds
                                    , 'precision': precision
                                    , 'recall': recall
                                    , 'f1': f1_list
                                    , 'queue_rate': queue_rate
                                    })
    else:
        df_plot_data = pd.DataFrame({'thresholds': thresholds
                                    , 'precision': precision
                                    , 'recall': recall
                                    , 'f1': f1_list
                                    })
    df_plot_data = df_plot_data.fillna(mask_value) # 欠損値(nan)は-1.0にしておく
    # plot
    plt.plot(thresholds, precision, color=sns.color_palette()[0])
    plt.plot(thresholds, recall, color=sns.color_palette()[1])
    plt.plot(thresholds, f1_list, color=sns.color_palette()[3])

    if is_queue_rate == True:
        # queue_rate 入れるか
        plt.plot(thresholds, queue_rate, color=sns.color_palette()[2])
        leg = plt.legend(('precision', 'recall', 'f1', 'queue_rate'), frameon=True)
    else:
        leg = plt.legend(('precision', 'recall', 'f1'), frameon=True)

    leg.get_frame().set_edgecolor('k')
    plt.xlabel('threshold')
    plt.xlim([-0.03, 1.03])
    plt.ylabel('%')
    plt.ylim([-0.03, 1.03])
    plt.title(title)
    plt.grid()
    if out_png is not None:
        plt.savefig(out_png, bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()

    # f1最大の閾値
    f1_max_threshold = df_plot_data.sort_values(by=['f1'], ascending=False)['thresholds'].values[0]
    # precision最大の閾値
    precision_max_threshold = df_plot_data.sort_values(by=['precision'], ascending=False)['thresholds'].values[0]
    # recall最大の閾値
    recall_max_threshold = df_plot_data.sort_values(by=['recall'], ascending=False)['thresholds'].values[0]

    return df_plot_data, f1_max_threshold, precision_max_threshold, recall_max_threshold

def f1_best_threshhold(y, pred_prob:np.ndarray):
    """
    scipy.optimizeのminimizeメソッドでf1最大になるしきい値求める
    Args:
        y:正解ラベル e.g: np.array([0, 0, 1, 1])やpd.Series(False, False, True, True)
        pred_prob:予測ラベル e.g: np.array([0.1, 0.4, 0.35, 0.8])
    Returns:
        best_threshold:f1最大になるしきい値
        best_f1_score:f1最大値
    Usage:
        size = 10000
        rand = np.random.RandomState(seed=41)
        y_prob = np.linspace(0,1.0,size)
        pred_prob = np.clip(y_prob * np.exp(rand.standard_normal(y_prob.shape))*0.3, 0.0, 1.0)
        print('pred_prob:', pred_prob[:5])

        y = pd.Series(rand.uniform(0.0,1.0,size) < y_prob)
        print('y:\n', y[:5])
        print(f1_best_threshhold(y, pred_prob))

        y = [1.0 if is_y else 0.0 for is_y in y]
        print('y:', y[:5])
        print(f1_best_threshhold(y, pred_prob))
    """
    def f1_opt(x):
        """f1のしきい値最適化の目的関数"""
        return -f1_score(y, pred_prob >= x)
    opt_result = minimize(f1_opt, x0=np.array([0.5]), method="Nelder-Mead")
    best_threshold = opt_result['x'].item()
    best_f1_score = f1_score(y, pred_prob >= best_threshold)
    return best_threshold, best_f1_score