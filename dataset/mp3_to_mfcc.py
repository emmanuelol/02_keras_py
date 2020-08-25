"""
指定ディレクトリのmp3ファイルをメル周波数スペクトラム係数(MFCC)のデータフレームに変換してcsvで保存する
ディレクトリ毎に0,1,2…とラベル列を切る
→ 出力するcsvはGBMとかの構造化モデルの入力データに使える

参考：[Pythonサンプルスクリプト] librosaによる音声データからの特徴量抽出 (MFCC)¶
https://community.datarobot.com/t5/%E5%AD%A6%E7%BF%92%E3%82%BB%E3%83%83%E3%82%B7%E3%83%A7%E3%83%B3/%E3%82%A2%E3%83%BC%E3%82%AB%E3%82%A4%E3%83%96-python%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AB%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%97%E3%83%88-librosa%E3%81%AB%E3%82%88%E3%82%8B%E9%9F%B3%E5%A3%B0%E3%83%87%E3%83%BC%E3%82%BF%E3%81%8B%E3%82%89%E3%81%AE%E7%89%B9%E5%BE%B4%E9%87%8F%E6%8A%BD%E5%87%BA-mfcc/ba-p/5106

Usage:
    $ conda activate tfgpu
    $ python mp3_to_mfcc.py -o D:\work\kaggle_data\birdsong-recognition\output\preprocess\mfcc -i D:\work\kaggle_data\birdsong-recognition\orig\birdsong-recognition\train_audio
"""
import os
import glob
import argparse
import pathlib

import pandas as pd  # おなじみpandas
import librosa  # 今回の主役librosa
from tqdm import tqdm


def load_dir_mfcc(m_dir: str, label: int):
    """指定ディレクトリの音声ファイルロードしてメル周波数スペクトラム係数(MFCC)取得"""
    list_ceps = []  # 抽出したMFCCを格納するリスト
    list_label = []  # ラベルを格納するリスト

    # 音声ファイルロード
    filelist = []
    filelist.extend(sorted(glob.glob(m_dir + "/*.wav")))  # *.wavのリストを作成
    filelist.extend(sorted(glob.glob(m_dir + "/*.mp3")))  # *.mp3のリストを作成

    for filename in tqdm(filelist):
        y, sr = librosa.core.load(filename, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # メル周波数スペクトラム係数を20次元にする

        # 複数のローリングウィンドウでそれぞれ20次元のMFCCを得られるので、その平均をとる。
        ceps = mfcc.mean(axis=1)

        # リストに追加
        list_ceps.append(ceps)  # 20次元のMFCCを追加
        list_label.append(label)  # ラベルを追加

    return list_ceps, list_label


def main(m_dirs: list):
    list_ceps = []
    list_label = []

    # 音声ファイルから、それぞれ20次元のMFCCを抽出
    for label, m_dir in tqdm(enumerate(m_dirs)):
        _list_ceps, _list_label = load_dir_mfcc(m_dir, label)
        list_ceps = [*list_ceps, *_list_ceps]
        list_label = [*list_label, *_list_label]

    # 20次元のMFCCのデータフレームを作成
    df_ceps = pd.DataFrame(list_ceps)

    columns_name = []  # カラム名を"dct+連番"でつける
    for i in range(20):
        columns_name_temp = "dct{0}".format(i)
        columns_name.append(columns_name_temp)

    df_ceps.columns = columns_name

    # ラベルのデータフレームを作成
    df_label = pd.DataFrame(list_label, columns=["label"])

    # 横にconcat
    df = pd.concat([df_label, df_ceps], axis=1)

    return df


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=r"D:\work\kaggle_data\birdsong-recognition\output\preprocess\mfcc",
        help="output dir path.",
    )
    ap.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default=r"D:\work\kaggle_data\birdsong-recognition\orig\birdsong-recognition\train_audio",
        help="input dir path.",
    )
    args = vars(ap.parse_args())
    os.makedirs(args["output_dir"], exist_ok=True)
    return args


if __name__ == "__main__":
    args = get_args()
    m_dirs = sorted(glob.glob(f'{args["input_dir"]}/*'))
    # m_dirs = m_dirs[:2]  # テスト用
    df = main(m_dirs)
    df.to_csv(
        os.path.join(args["output_dir"], pathlib.Path(args["input_dir"]).stem + ".csv"),
        index=False,
    )
