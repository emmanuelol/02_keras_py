# -*- coding: utf-8 -*-
"""
複数モデルのpredictをensembleする
参考：https://qiita.com/koshian2/items/d569cd71b0e082111962

Usage:
import ensemble_predict
model1 = keras.models.load_model(os.path.join(out_dir, 'best_model1.h5'), compile=False)
model2 = keras.models.load_model(os.path.join(out_dir, 'best_model2.h5'), compile=False)
models = [model1, model2]
ensemble_soft_pred = ensemble_predict.ensembling_soft(models, X_test)
ensemble_hard_pred = ensemble_predict.ensembling_hard(models, X_test)
"""
import os, sys, glob
import numpy as np
import pandas as pd
from scipy.stats import mode

def ensembling_soft(models, X):
    """
    確率の平均を取るアンサンブル（ソフトアンサンブル）
    ラベルの推定確率どうしの平均を取る。
    例えば、分類器Aが犬＝0.8・猫＝0.2、分類器Bが犬＝0.9、猫＝0.1、分類器Cが犬＝0.4、猫＝0.6だったら、全体の出力は犬＝0.7、猫＝0.3となり、全体の出力は「犬」 となる。
    Args:
        models:modelオブジェクトのリスト
        X:前処理済みのnumpy.array型の画像データ(4次元テンソル (shapeは[ファイル数, 横ピクセル, 縦ピクセル, チャネル数]) )
    Returns:
        平均を取ったラベルの推定確率
    """
    preds_sum = None
    for model in models:
        if preds_sum is None:
            preds_sum = model.predict(X)
        else:
            preds_sum += model.predict(X)
    probs = preds_sum / len(models)
    return probs# ラベルでほしいときはnp.argmax(probs, axis=-1)

def ensembling_soft_from_task12_tsv(dirs):
    """
    Roc図書くときに使うy_true, y_predのtsvファイルから、行ごとにy_predの平均を取るソフトアンサンブル
    Args:
        dirs:タスクごとにRoc図書くときに使うy_true, y_predのtsvファイルを格納しているディレクトリパスのリスト
    Returns:
        y_true_list:タスクごとの正解ラベルのリスト
        y_pred_ensemble_list:タスクごとに平均を取ったラベルの推定確率リスト
    """
    pred_df_sum = None
    for dir in dirs:
        # taskごとのtsvファイル
        files = glob.glob(dir+'/*tsv')
        files.sort()
        # ディレクトリ単位で12taskのpredデータフレーム作成
        true_df = None
        pred_df = None
        for f in files:
            df = pd.read_csv(f, sep = '\t')
            df.rename(columns={'y_true': 't_'+os.path.basename(f), 'y_pred': 'p_'+os.path.basename(f)}, inplace=True)
            if true_df is None:
                true_df = pd.DataFrame(df.iloc[:,[0]])
                pred_df = pd.DataFrame(df.iloc[:,[1]])
            else:
                # 2ファイル以降はconcat（単純な縦積み）
                true_df = pd.concat([true_df, df.iloc[:,[0]]], axis=1)
                pred_df = pd.concat([pred_df, df.iloc[:,[1]]], axis=1)
        # ディレクトリ単位で集めた12taskのpredデータフレームの合計版作成
        if pred_df_sum is None:
            pred_df_sum = pred_df
        else:
            pred_df_sum = pred_df_sum + pred_df
    # ディレクトリ単位で集めた12taskのsoft_ensemble_predデータフレーム作成
    pred_df_mean = pred_df_sum/len(dirs)
    # Roc図作成用にデータフレームからリストに変換しておく
    y_true_list = [true_df['t_task0.tsv'], true_df['t_task1.tsv'], true_df['t_task2.tsv']
                    , true_df['t_task3.tsv'], true_df['t_task4.tsv'], true_df['t_task5.tsv']
                    , true_df['t_task6.tsv'], true_df['t_task7.tsv'], true_df['t_task8.tsv']
                    , true_df['t_task9.tsv'], true_df['t_task10.tsv'], true_df['t_task11.tsv']]
    y_pred_ensemble_list = [pred_df_mean['p_task0.tsv'], pred_df_mean['p_task1.tsv'], pred_df_mean['p_task2.tsv']
                            , pred_df_mean['p_task3.tsv'], pred_df_mean['p_task4.tsv'], pred_df_mean['p_task5.tsv']
                            , pred_df_mean['p_task6.tsv'], pred_df_mean['p_task7.tsv'], pred_df_mean['p_task8.tsv']
                            , pred_df_mean['p_task9.tsv'], pred_df_mean['p_task10.tsv'], pred_df_mean['p_task11.tsv']]
    return y_true_list, y_pred_ensemble_list


def ensembling_hard(models, X):
    """
    シングルタスクでの多数決のアンサンブル（ハードアンサンブル）
    分類器単位でラベルを求め、その多数決を取る。
    同一の場合は、ラベルの番号が若いものを優先する。例えば、ソフトな例なら、分類器Aは犬、分類器Bも犬、分類器Cは猫なので、全体の出力は多数決により「犬」となる。
    Args:
        models:modelオブジェクトのリスト
        X:前処理済みのnumpy.array型の画像データ(4次元テンソル (shapeはファイル数, 横ピクセル, 縦ピクセル, チャネル数]) )
    Returns:
        多数決で決まったラベル
    """
    pred_labels = np.zeros((X.shape[0], len(models)))
    for i, model in enumerate(models):
        pred_labels[:, i] = np.argmax(model.predict(X), axis=-1)
    return np.ravel(mode(pred_labels, axis=1)[0])# np.ravel:1次元配列に変換
