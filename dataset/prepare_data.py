# -*- coding: utf-8 -*-
"""
ラベル情報などが書いたcsvファイル（tox21_compoundData.csv）から
Tox21の画像のパスとラベル(y_train,y_valid,y_test)を取得する

Usage:
    import os, sys
    current_dir = os.path.dirname(os.path.abspath("__file__"))
    path = os.path.join(current_dir, '../')
    sys.path.append(path)
    from dataset import prepare_data

    csv_path = r'/home/tmp10014/jupyterhub/notebook/work_H3-038/Tox21/bioinf_data/tox21_gunzip/tox21_compoundData.csv'
    img_dir = r'/home/tmp10014/jupyterhub/notebook/work_H3-038/smi2img/work/smi2img_test/'
    img_suffix = r'_000.jpg'
    df = prepare_data.make_label_df(csv_path, img_dir, img_suffix=img_suffix)
    train_files, validation_files, test_files, y_train, y_valid, y_test = prepare_data.make_train_val_test(csv_path, img_dir, img_suffix=img_suffix)
"""
import os
from glob import glob
import pandas as pd

def make_label_df(csv_path, img_dir, img_suffix='_000.jpg'):
    """
    ラベル情報などが書いたcsvファイル（tox21_compoundData.csv）編集
    - 欠損値を-1に置換
    - ファイル名のフルパスをつける
    Args:
        csv_path: tox21_compoundData.csvのpath
        img_dir: 画像ディレクトリ。train_all,test,Final_Evaluation ディレクトリを置いてるpath
        img_suffix: 画像ファイルのサフィックス _000.jpg とか
    Returns:
        欠損値を-1に置換したtox21_compoundData.csvのデータフレーム
    """
    # ラベル情報などが書いたcsvファイルロード
    df = pd.read_csv(csv_path)
    # 欠損値を-1で置換
    df = df.fillna(-1)
    # ID列にファイルのフルパスつける
    train_dir = img_dir+'train_all/'
    val_dir = img_dir+'test/'
    test_dir = img_dir+'Final_Evaluation/'
    df.loc[df['set']=='training', 'ID'] = train_dir+df['ID']+img_suffix
    df.loc[df['set']=='validation', 'ID'] = val_dir+df['ID']+img_suffix
    df.loc[df['set']=='test', 'ID'] = test_dir+df['ID']+img_suffix
    return df

def make_label_df_NoNA(csv_path, img_dir, img_suffix='_000.jpg'):
    """
    ラベル情報などが書いたcsvファイル（tox21_compoundData.csv）編集
    - 欠損値なしレコードだけにする
    - ファイル名のフルパスをつける
    Args:
        csv_path: tox21_compoundData.csvのpath
        img_dir: 画像ディレクトリ。train_all,test,Final_Evaluation ディレクトリを置いてるpath
        img_suffix: 画像ファイルのサフィックス  _000.jpg とか
    Returns:
        ラベルの欠損値もつレコード削除したtox21_compoundData.csvのデータフレームのサブセット
    """
    # ラベル情報などが書いたcsvファイルロード
    df = pd.read_csv(csv_path)
    # サブセットを取得
    df_sub = df.loc[:,['ID','set','NR.AhR', 'NR.AR', 'NR.AR.LBD', 'NR.Aromatase', 'NR.ER', 'NR.ER.LBD', 'NR.PPAR.gamma', 'SR.ARE', 'SR.ATAD5', 'SR.HSE', 'SR.MMP', 'SR.p53']]
    # データフレームの欠損値を含むすべての行を削除する
    df = df_sub.dropna()
    # ID列にファイルのフルパスつける
    train_dir = img_dir+'train_all/'
    val_dir = img_dir+'test/'
    test_dir = img_dir+'Final_Evaluation/'
    df.loc[df['set']=='training', 'ID'] = train_dir+df['ID']+img_suffix
    df.loc[df['set']=='validation', 'ID'] = val_dir+df['ID']+img_suffix
    df.loc[df['set']=='test', 'ID'] = test_dir+df['ID']+img_suffix
    return df

def make_label_df_angle(csv_path, img_dir, angle_suffix_list=['_060.jpg', '_120.jpg', '_180.jpg', '_240.jpg', '_300.jpg']):
    """
    ラベル情報などが書いたcsvファイル（tox21_compoundData.csv）編集
    - 欠損値を-1に置換
    - ファイル名のフルパスをつける
    - 回転画像も含める。
        回転画像は
        化合物Id
            |-化合物Id_000.jpg
            |-化合物Id_060.jpg
            …
            となるディレクトリ構成を想定している
    Args:
        csv_path: tox21_compoundData.csvのpath
        img_dir: 画像ディレクトリ。train_all,test,Final_Evaluation ディレクトリを置いてるpath
        angle_suffix_list:回転画像のサフィックスのリスト。デフォルトは60度の時
    Returns:
        欠損値を-1に置換したtox21_compoundData.csvのデータフレーム
    """
    # ラベル情報などが書いたcsvファイルロード
    df = pd.read_csv(csv_path)
    # 欠損値を-1で置換
    df = df.fillna(-1)
    # 直前のディレクトリ名とｖ回転角0度のサフィックスを付ける
    df['ID'] = df['ID']+'/'+df['ID']+'_000.jpg'

    # 回転画像のパスも追加する
    for suf in angle_suffix_list:
        # ラベル情報などが書いたcsvファイルロード
        df_suf = pd.read_csv(csv_path)
        # 欠損値を-1で置換
        df_suf = df_suf.fillna(-1)
        # 直前のディレクトリ名と回転角のサフィックスを付ける
        df_suf['ID'] = df_suf['ID']+'/'+df_suf['ID']+suf
        # 回転角0度のデータフレームに縦づみ
        df = pd.concat([df, df_suf])

    # index振り直し
    df = df.reset_index()

    # ID列にファイルのフルパスつける
    train_dir = img_dir+'train_all/'
    val_dir = img_dir+'test/'
    test_dir = img_dir+'Final_Evaluation/'
    df.loc[df['set']=='training', 'ID'] = train_dir+df['ID']
    df.loc[df['set']=='validation', 'ID'] = val_dir+df['ID']
    df.loc[df['set']=='test', 'ID'] = test_dir+df['ID']
    return df


def get_set_label(df, set_name):
    """
    編集したtox21_compoundData.csvのデータフレーム（df）から
    各データセットの画像ファイルパスとdatasetのラベルを取得する
    Args:
        df:編集したtox21_compoundData.csvのデータフレーム
        set_name:データセット名 'training'/'validation'/'test' のいずれか
    Returns:
        files_set:データセットのファイルpath一式
        y_set:files_setのラベル一式
    """
    df_set = df.loc[df['set']==set_name]
    files_set = df_set['ID']
    y_set = df_set.loc[:,['NR.AhR', 'NR.AR', 'NR.AR.LBD', 'NR.Aromatase', 'NR.ER', 'NR.ER.LBD', 'NR.PPAR.gamma'
                          , 'SR.ARE', 'SR.ATAD5', 'SR.HSE', 'SR.MMP', 'SR.p53']]
    # values属性でNumPy配列ndarrayを取得
    y_set = y_set.values
    print(set_name+'_set_y.shape:', y_set.shape)
    return files_set, y_set

def make_train_val_test(df):
    """
    編集したtox21_compoundData.csvのデータフレーム（df）から
    Tox21の画像のパス(train_files,validation_files,test_files) と ラベル(y_train,y_valid,y_test)を取得する
    Args:
        df:編集したtox21_compoundData.csvのデータフレーム
    Returns:
        train_files,validation_files,test_files:Tox21の各データセットの画像path
        y_train,y_valid,y_test:Tox21の各データセットのラベル
    """
    # 各データセットの画像ファイルパスとdatasetのラベルを取得する
    train_files, y_train = get_set_label(df, 'training')
    validation_files, y_valid = get_set_label(df, 'validation')
    test_files, y_test = get_set_label(df, 'test')

    return train_files, validation_files, test_files, y_train, y_valid, y_test


def make_train_val_test_CVfold(df, valid_cv=0, not_set_name='test'):
    """
    編集したtox21_compoundData.csvのデータフレーム（df）から
    CV(Cross Validation)の画像ファイルパスとラベルを取得する
    検証用のtest setにもCVついてる場合があるのでtrain/validationからtest除く
    Args:
        df:編集したtox21_compoundData.csvのデータフレーム
        valid_cv:検証setにするCVFold番号
        not_set_name:train/validation setに含めないデータセット名
    Returns:
        files_cv_train,files_cv_valid,test_files:CVFoldの各データセットの画像path
        y_cv_train,y_cv_valid,y_test:CVFoldの各データセットのラベル
    """
    # tarin CV
    df_cv_train = df.loc[(df['CVfold']!=valid_cv) & (df['CVfold']!=-1) & (df['set']!=not_set_name)]
    files_cv_train = df_cv_train['ID']
    y_cv_train = df_cv_train.loc[:,['NR.AhR', 'NR.AR', 'NR.AR.LBD', 'NR.Aromatase', 'NR.ER', 'NR.ER.LBD', 'NR.PPAR.gamma'
                          , 'SR.ARE', 'SR.ATAD5', 'SR.HSE', 'SR.MMP', 'SR.p53']]
    y_cv_train = y_cv_train.values# values属性でNumPy配列ndarrayを取得
    cv_list=[0,1,2,3,4,5]
    cv_list.remove(valid_cv)
    print(str(cv_list)+'_cv_train_y.shape:', y_cv_train.shape)

    # valid CV
    df_cv_valid = df.loc[(df['CVfold']==valid_cv) & (df['set']!=not_set_name)]
    files_cv_valid = df_cv_valid['ID']
    y_cv_valid = df_cv_valid.loc[:,['NR.AhR', 'NR.AR', 'NR.AR.LBD', 'NR.Aromatase', 'NR.ER', 'NR.ER.LBD', 'NR.PPAR.gamma'
                          , 'SR.ARE', 'SR.ATAD5', 'SR.HSE', 'SR.MMP', 'SR.p53']]
    y_cv_valid = y_cv_valid.values# values属性でNumPy配列ndarrayを取得
    print(str(valid_cv)+'_cv_valid_y.shape:', y_cv_valid.shape)

    # test set
    test_files, y_test = get_set_label(df, not_set_name)

    return files_cv_train, y_cv_train, files_cv_valid, y_cv_valid, test_files, y_test


def make_train_val_test_angle(df):
    """
    回転画像用に編集したtox21_compoundData.csvのデータフレーム（df）から
    Tox21の画像のパス(train_files,validation_files,test_files) と ラベル(y_train,y_valid,y_test)を取得する
    Args:
        df:編集したtox21_compoundData.csvのデータフレーム
    Returns:
        train_files,validation_files,test_files:Tox21の各データセットの画像path
        y_train,y_valid,y_test:Tox21の各データセットのラベル
    """
    # train
    df_train = df.loc[df['set']=='training']
    train_files  = df_train['ID']
    y_train = df_train.loc[:,['NR.AhR', 'NR.AR', 'NR.AR.LBD', 'NR.Aromatase', 'NR.ER', 'NR.ER.LBD', 'NR.PPAR.gamma'
                              , 'SR.ARE', 'SR.ATAD5', 'SR.HSE', 'SR.MMP', 'SR.p53']]
    # values属性でNumPy配列ndarrayを取得
    y_train = y_train.values
    print('training_set_y.shape:', y_train.shape)

    # validation
    df_valid = df.loc[(df['set'] == 'validation') & (df['ID'].str.contains('_000.jpg'))]
    validation_files  = df_valid['ID']
    y_valid = df_valid.loc[:,['NR.AhR', 'NR.AR', 'NR.AR.LBD', 'NR.Aromatase', 'NR.ER', 'NR.ER.LBD', 'NR.PPAR.gamma'
                              , 'SR.ARE', 'SR.ATAD5', 'SR.HSE', 'SR.MMP', 'SR.p53']]
    # values属性でNumPy配列ndarrayを取得
    y_valid = y_valid.values
    print('validation_set_y.shape:', y_valid.shape)

    # test
    df_test = df.loc[(df['set'] == 'test') & (df['ID'].str.contains('_000.jpg'))]
    test_files  = df_test['ID']
    y_test = df_test.loc[:,['NR.AhR', 'NR.AR', 'NR.AR.LBD', 'NR.Aromatase', 'NR.ER', 'NR.ER.LBD', 'NR.PPAR.gamma'
                              , 'SR.ARE', 'SR.ATAD5', 'SR.HSE', 'SR.MMP', 'SR.p53']]
    # values属性でNumPy配列ndarrayを取得
    y_test = y_test.values
    print('test_set_y.shape:', y_test.shape)

    return train_files, validation_files, test_files, y_train, y_valid, y_test
