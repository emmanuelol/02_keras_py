# -*- coding: utf-8 -*-
"""
マルチタスクのpredictを実行する

Usage:
import multi_predict

model = keras.models.load_model(os.path.join(out_dir, 'best_model.h5'), compile=False)
# 出力層のニューラルネットワークに分岐がある場合のpredict
y_test_list, y_pred_list = multi_predict.branch_set_predict(model, d_cls.X_test, d_cls.y_test)
"""
import os, sys
import glob
import numpy as np
import pandas as pd

# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得
sys.path.append( str(current_dir) + '/../' )
from transformer import get_train_valid_test

def branch_set_predict(model, x, y_true, out_dir):
    """
    出力層のニューラルネットワークに分岐がある場合のpredict
    次処理の混同行列やRoc_AUCの入力データ形式に正解ラベル、推論スコアを整える
    Args:
        model:モデルオブジェクト
        x:前処理済みのnumpy.array型の画像データ(4次元テンソル (shapeはファイル数, 横ピクセル, 縦ピクセル, チャネル数]) )
        y_true:正解（マルチ）ラベルのnumpy.ndarray　例 array([[0,1,0,1], [0,1,0,0], …])　（shapeは ファイル数, タスク数）
        out_dir:ファイル出力先
    Returns:
        y_true_list:正解ラベルのnumpy.ndarrayのリスト [array([0,1,0,…]), array([1,0,0,…]),…] ( リストの長さはタスク数、各arrayはファイルごとのラベル )
        y_pred_list:予測スコアのnumpy.ndarrayのリスト [array([0.65,0.99,0.01,…]), array([0.1,0.33,0.95,…]),…] ( リストの長さはタスク数、各arrayはファイルごとのスコア )
    """
    # 推論
    # numpy.ndarrayのリストの形式でスコアでてくる（ [array([[0.01],[0.9],…], [[0.11],[0.6],…], …])　（リストの各arrayのshapeは [ファイル数, 1]） ）
    orig_y_pred_list = model.predict(x)
    # タスクごとにpredictのスコアをリストにつめる
    y_pred_list = []
    for y_pred in orig_y_pred_list:
        y_pred_list.append(y_pred.reshape(y_pred.shape[0]))

    # タスクごとに正解ラベルをリストにつめる
    y_true_list = []
    for i in range(y_true.shape[1]):
        y_true_list.append(y_true[:,i])

    # ファイル出力先作成
    score_out_dir = os.path.join(out_dir, 'score')
    os.makedirs(score_out_dir, exist_ok=True)
    # スコアと正解ラベルのファイル出力
    os.makedirs(out_dir, exist_ok=True)
    y_pred_df = pd.DataFrame(y_pred_list)
    y_true_df = pd.DataFrame(y_true_list)
    for i in range(len(y_true_df)):
        df = pd.DataFrame({'y_true': y_true_df.loc[i], 'y_pred': y_pred_df.loc[i]})
        df = df.ix[:,['y_true','y_pred']]
        df.to_csv(os.path.join(score_out_dir, 'task'+str(i)+'.tsv'), sep="\t", index=False, header=True)

    return y_true_list, y_pred_list

def branch_mean_predict(model, x):
    """
    出力層のニューラルネットワークに分岐がある場合のTTA用
    predictの平均を取る
    Args:
        model:モデルオブジェクト
        x:前処理済みのnumpy.array型の画像データ(4次元テンソル (shapeはファイル数, 横ピクセル, 縦ピクセル, チャネル数]) )
    Returns:
        y_pred_list:予測スコアのリストのリスト [[0.083, 0.0107, 0.0117,…], [0.83, 0.01, 0.1,…]…] ( リストの長さはタスク数、各値はファイルの平均スコア )
    """
    # 推論
    # numpy.ndarrayのリストの形式でスコアでてくる（ [array([[0.01],[0.9],…], [[0.11],[0.6],…], …])　（リストの各arrayのshapeは [ファイル数, 1]） ）
    orig_y_pred_list = model.predict(x)
    # タスクごとにpredictのスコアをリストにつめる
    y_pred_list = []
    for y_pred in orig_y_pred_list:
        # 平均を取る
        y_pred_list.append(np.mean(y_pred))

    return y_pred_list

def branch_TTA_predict(model, shape, dir_list, y_true, out_dir):
    """
    出力層のニューラルネットワークに分岐がある場合のTTAのpredict
    次処理の混同行列やRoc_AUCの入力データ形式に正解ラベル、推論スコアを整える
    Args:
        model:モデルオブジェクト
        shape:モデルの入力層のshape [100, 100, 3]など
        dir_list:ディレクトリ単位でpredictの平均を取る画像ディレクトリのリスト
        y_true:正解（マルチ）ラベルのnumpy.ndarray　例 array([[0,1,0,1], [0,1,0,0], …])　（shapeは ファイル数, タスク数）
        out_dir:ファイル出力先
    Returns:
        y_true_list:正解ラベルのnumpy.ndarrayのリスト [array([0,1,0,…]), array([1,0,0,…]),…] ( リストの長さはタスク数、各arrayはファイルごとのラベル )
        y_pred_list:予測スコアのnumpy.ndarrayのリスト [array([0.65,0.99,0.01,…]), array([0.1,0.33,0.95,…]),…] ( リストの長さはタスク数、各arrayはファイルごとのスコア )
    """
    count=0
    y_pred_list_dir = []
    for dir in dir_list:
        # 画像ディレクトリ単位でデータ管理クラス作成
        d_cls = get_train_valid_test.LabeledDataset(shape, 0, 0)
        d_cls.load_test_as_image(glob.glob(dir+'/*.jpg'), y_true[count])
        # 画像ディレクトリ単位でpredictの平均を取る
        pred_dir_mean = branch_mean_predict(model, d_cls.X_test)
        y_pred_list_dir.append(pred_dir_mean)
        count+=1

    # タスクごとに推論ラベルをリストにつめる
    np_y_pred_list_dir = np.array(y_pred_list_dir)
    y_pred_list=[]
    for i in range(np_y_pred_list_dir.shape[1]):
        y_pred_list.append(np_y_pred_list_dir[:,i])

    # タスクごとに正解ラベルをリストにつめる
    y_true_list = []
    for i in range(y_true.shape[1]):
        y_true_list.append(y_true[:,i])

    # ファイル出力先作成
    score_out_dir = os.path.join(out_dir, 'score')
    os.makedirs(score_out_dir, exist_ok=True)
    # スコアと正解ラベルのファイル出力
    y_pred_df = pd.DataFrame(y_pred_list)
    y_true_df = pd.DataFrame(y_true_list)
    for i in range(len(y_true_df)):
        df = pd.DataFrame({'y_true': y_true_df.loc[i], 'y_pred': y_pred_df.loc[i]})
        df = df.ix[:,['y_true','y_pred']]
        df.to_csv(os.path.join(score_out_dir, 'task'+str(i)+'.tsv'), sep="\t", index=False, header=True)

    return y_true_list, y_pred_list



def no_branch_set_predict(model, x, y_true, out_dir):
    """
    出力層のニューラルネットワークに分岐がない場合のpredict
    次処理の混同行列やRoc_AUCの入力データ形式に正解ラベル、推論スコアを整える
    Args:
        model:モデルオブジェクト
        x:前処理済みのnumpy.array型の画像データ(4次元テンソル (shapeはファイル数, 横ピクセル, 縦ピクセル, チャネル数]) )
        y_true:正解（マルチ）ラベルのnumpy.ndarray　例 array([[0,1,0,1], [0,1,0,0], …])　（shapeは ファイル数, タスク数）
        out_dir:ファイル出力先
    Returns:
        y_true_list:正解ラベルのnumpy.ndarrayのリスト [array([0,1,0,…]), array([1,0,0,…]),…] ( リストの長さはタスク数、各arrayはファイルごとのラベル )
        y_pred_list:予測スコアのnumpy.ndarrayのリスト [array([0.65,0.99,0.01,…]), array([0.1,0.33,0.95,…]),…] ( リストの長さはタスク数、各arrayはファイルごとのスコア )
    """
    # 推論
    # y_trueと同じ形式でスコアでてくる（ array([[0.01,0.9,0.85,0.001], [0.2,0.221,0.61,0.25], …])　（shapeは ファイル数, タスク数） ）
    y_pred = model.predict(x)
    # タスクごとにpredictのスコアをリストにつめる
    y_pred_list = []
    for i in range(y_pred.shape[1]):
        y_pred_list.append(y_pred[:,i])

    # タスクごとに正解ラベルをリストにつめる
    y_true_list = []
    for i in range(y_true.shape[1]):
        y_true_list.append(y_true[:,i])

    # ファイル出力先作成
    score_out_dir = os.path.join(out_dir, 'score')
    os.makedirs(score_out_dir, exist_ok=True)
    # スコアと正解ラベルのファイル出力
    y_pred_df = pd.DataFrame(y_pred_list)
    y_true_df = pd.DataFrame(y_true_list)
    for i in range(len(y_true_df)):
        df = pd.DataFrame({'y_true': y_true_df.loc[i], 'y_pred': y_pred_df.loc[i]})
        df = df.ix[:,['y_true','y_pred']]
        df.to_csv(os.path.join(score_out_dir, 'task'+str(i)+'.tsv'), sep="\t", index=False, header=True)

    return y_true_list, y_pred_list

def output_no_branch_predict_img_files(model, out_dir, img_files, task_name_list, img_rows=331, img_cols=331, pred_threshold=0.5, img_saff='_000.jpg'):
    """
    画像ファイルから予測して結果をエクセルに出力
    Args:
        model:モデルオブジェクト
        img_files:画像ファイルのリスト
        img_rows, img_cols:モデルの入力画像サイズ
        task_name_list: タスク（クラス）名のリスト
        pred_threshold: 予測スコアのposi/nega分ける閾値。バイナリ分類なので、デフォルトは0.5とする
        img_saff:画像ファイル名のサフィックス
    Returns:
        df_pred_score, df_y_pred:予測スコアと予測ラベルのデータフレーム
    """
    y_pred_list = []
    for f in img_files:
        x = get_train_valid_test.load_one_img(f, img_rows, img_cols) # 画像を1枚読み込んで、4次元テンソル(1, img_rows, img_cols, 3)へ変換+前処理
        y_pred_list.append(model.predict(x))
    pred_score = np.array(y_pred_list).reshape(len(y_pred_list), len(task_name_list))

    # 2つのタスク名でヘッダ2行にする
    cols = pd.MultiIndex.from_arrays([list(map(lambda i: 'task'+str(i), range(pred_score.shape[1]))), task_name_list])

    # 予測スコアのデータフレーム
    df_pred_score = pd.DataFrame(data=pred_score, columns=cols)
    # 予測ラベルのデータフレーム バイナリ分類なので、確信度が0.5より大きい推論を1、それ以外を0に置換する
    y_pred = (pred_score > pred_threshold) * 1.0
    df_y_pred = pd.DataFrame(data=y_pred, columns=cols)

    # SERIES_NOと画像パス列追加
    df_pred_score['img_path'] = img_files
    df_pred_score['SERIES_NO'] = list(map(lambda f: str(Path(f).name).replace(img_saff, ''), img_files))
    df_pred_score = df_pred_score.iloc[:,[-1,-2]+list(range(0,len(task_name_list)))] # 列の順番入れ替え
    df_y_pred['img_path'] = img_files
    df_y_pred['SERIES_NO'] = list(map(lambda f: str(Path(f).name).replace(img_saff, ''), img_files))
    df_y_pred = df_y_pred.iloc[:,[-1,-2]+list(range(0,len(task_name_list)))] # 列の順番入れ替え

    # エクセル複数シート作成
    with pd.ExcelWriter(os.path.join(out_dir, 'pred_score.xlsx')) as writer:
        df_pred_score.to_excel(writer, sheet_name='score') # 予測スコア
        df_y_pred.to_excel(writer, sheet_name='y_threshold_'+str(pred_threshold)) # 予測ラベル

    return df_pred_score, df_y_pred

def no_branch_mean_predict(model, x):
    """
    出力層のニューラルネットワークに分岐がない場合のTTA用
    predictの平均を取る
    Args:
        model:モデルオブジェクト
        x:前処理済みのnumpy.array型の画像データ(4次元テンソル (shapeはファイル数, 横ピクセル, 縦ピクセル, チャネル数]) )
    Returns:
        y_pred_list:予測スコアのリストのリスト [[0.083, 0.0107, 0.0117,…], [0.83, 0.01, 0.1,…]…] ( リストの長さはタスク数、各値はファイルの平均スコア )
    """
    # 推論
    # y_trueと同じ形式でスコアでてくる（ array([[0.01,0.9,0.85,0.001], [0.2,0.221,0.61,0.25], …])　（shapeは ファイル数, タスク数） ）
    y_pred = model.predict(x)
    # タスクごとにpredictのスコアをリストにつめる
    y_pred_list = []
    for i in range(y_pred.shape[1]):
        # 平均を取る
        y_pred_list.append(np.mean(y_pred[:,i]))

    return y_pred_list

def no_branch_TTA_predict(model, shape, dir_list, y_true, out_dir):
    """
    出力層のニューラルネットワークに分岐がない場合のTTAのpredict
    次処理の混同行列やRoc_AUCの入力データ形式に正解ラベル、推論スコアを整える
    Args:
        model:モデルオブジェクト
        shape:モデルの入力層のshape [100, 100, 3]など
        dir_list:ディレクトリ単位でpredictの平均を取る画像ディレクトリのリスト
        y_true:正解（マルチ）ラベルのnumpy.ndarray　例 array([[0,1,0,1], [0,1,0,0], …])　（shapeは ファイル数, タスク数）
        out_dir:ファイル出力先
    Returns:
        y_true_list:正解ラベルのnumpy.ndarrayのリスト [array([0,1,0,…]), array([1,0,0,…]),…] ( リストの長さはタスク数、各arrayはファイルごとのラベル )
        y_pred_list:予測スコアのnumpy.ndarrayのリスト [array([0.65,0.99,0.01,…]), array([0.1,0.33,0.95,…]),…] ( リストの長さはタスク数、各arrayはファイルごとのスコア )
    """
    count=0
    y_pred_list_dir = []
    for dir in dir_list:
        # 画像ディレクトリ単位でデータ管理クラス作成
        d_cls = get_train_valid_test.LabeledDataset(shape, 0, 0)
        d_cls.load_test_as_image(glob.glob(dir+'/*.jpg'), y_true[count])
        # 画像ディレクトリ単位でpredictの平均を取る
        pred_dir_mean = no_branch_mean_predict(model, d_cls.X_test)
        y_pred_list_dir.append(pred_dir_mean)
        count+=1

    # タスクごとに推論ラベルをリストにつめる
    np_y_pred_list_dir = np.array(y_pred_list_dir)
    y_pred_list=[]
    for i in range(np_y_pred_list_dir.shape[1]):
        y_pred_list.append(np_y_pred_list_dir[:,i])

    # タスクごとに正解ラベルをリストにつめる
    y_true_list = []
    for i in range(y_true.shape[1]):
        y_true_list.append(y_true[:,i])

    # ファイル出力先作成
    score_out_dir = os.path.join(out_dir, 'score')
    os.makedirs(score_out_dir, exist_ok=True)
    # スコアと正解ラベルのファイル出力
    y_pred_df = pd.DataFrame(y_pred_list)
    y_true_df = pd.DataFrame(y_true_list)
    for i in range(len(y_true_df)):
        df = pd.DataFrame({'y_true': y_true_df.loc[i], 'y_pred': y_pred_df.loc[i]})
        df = df.ix[:,['y_true','y_pred']]
        df.to_csv(os.path.join(score_out_dir, 'task'+str(i)+'.tsv'), sep="\t", index=False, header=True)

    return y_true_list, y_pred_list


if __name__ == '__main__':
    print('multi_predict.py: loaded as script file')
else:
    print('multi_predict.py: loaded as module file')
