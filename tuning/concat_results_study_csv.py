# coding: utf-8
"""
複数のoptunaの結果csvを連結する
Usage:
    $ python concat_results_study_csv.py --csvs_dir ./csv --output_csv concat_results_study.csv
"""
import os, sys, glob, pathlib
import pandas as pd
import copy
import argparse
import warnings
warnings.filterwarnings("ignore")

def get_concat_csvs_header(csvs:list, header0=['number', 'state', 'value', 'datetime_start', 'datetime_complete']):
    """
    複数のoptunaの結果csvを連結して、csv2行目の一意なヘッダ名を取得
    Args:
        csvs: optunaの結果csvのパスリスト
        header0: optunaの結果csvの1行目のheaderの一部。引数にしているがデフォルトから変える必要ない
    Returns:
        csvs_header: optunaの結果csvのcsv2行目の一意なヘッダ名
        csvs_header1: optunaの結果csvのcsv1行目の一意なヘッダ名
    """
    # optunaの結果csv複数ロード
    df_all_header = pd.DataFrame()
    for c in csvs:
        df = pd.read_csv(c)
        df['csv'] = c
        df_header = df.iloc[:1] # header行だけ取り出す
        df_all_header = pd.concat([df_all_header, df_header])
    #display(df_all_header)

    csvs_header = [['csv']] + [header0]
    csvs_header1 = copy.deepcopy(csvs_header)
    # optunaの結果csvの1行目の各ヘッダ要素取得する
    for col in ['params', 'user_attrs', 'system_attrs', 'intermediate_values']:
        cols = [s for s in df_all_header.columns.to_list() if col in s]
        # ヘッダ名一意にする
        l_unique = []
        for c in cols:
            l_unique = l_unique + list(df_all_header[c].unique())
        l_unique = list(set(l_unique))
        l_unique = sorted([x for x in l_unique if str(x) != 'nan']) # nan削除

        # intermediate_values はfloat型なので文字列に直す
        if col == 'intermediate_values':
            l_unique = [str(int(c)) for c in l_unique]

        # csvの2行目の各ヘッダ
        csvs_header.append(l_unique)

        # csvの1行目の各ヘッダ
        csvs_header1.append([col]*len(l_unique))

    csvs_header = [flatten for inner in csvs_header for flatten in inner] # 1次元化
    csvs_header1 = [flatten for inner in csvs_header1 for flatten in inner] # 1次元化
    return csvs_header, csvs_header1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_csv", type=str, default='concat_results_study.csv',
                    help="output csv path.")
    ap.add_argument("--csvs_dir", type=str, required=True,
                    help="input optuna result study csvs dir path.")
    args = ap.parse_args()

    # optunaの結果csvの1行目のheaderの一部
    header0=['number', 'state', 'value', 'datetime_start', 'datetime_complete']

    # 複数のoptunaの結果csvのパス取得
    csvs = glob.glob(args.csvs_dir+'/*csv')

    # 複数のoptunaの結果csvを連結
    df_concat = None
    for c in csvs:
        #print(c)
        df = pd.read_csv(c, header=1) # 2行目をheaderとしてロード
        df['csv']  = c # csvのパス列追加
        df.columns = header0 + df.columns[len(header0):].tolist()
        if df_concat is None:
            df_concat = df
        else:
            df_concat = pd.concat([df_concat, df])
    #display(df_concat.head())
    #print(df_concat.columns.to_list())
    #print(df_concat.shape)

    # 連結後の列名取得
    csvs_header, csvs_header1 = get_concat_csvs_header(csvs, header0)
    #print(len(csvs_header), len(csvs_header1))

    # 列の順番入れ替え
    df_concat = df_concat[csvs_header]
    #display(df_concat.head())
    #print(df_concat.columns.to_list())
    #print(df_concat.shape)

    # 一旦出力
    df_concat.to_csv(args.output_csv, index=False)
    # csvの1行目のheader追加
    df_concat = pd.read_csv(args.output_csv, names=csvs_header1) # pandas 0.25.1ではエラーになる。pandas 0.24.1 ならエラーにならない
    df_concat.to_csv(args.output_csv)

if __name__ == "__main__":
    main()
