#!/home/aaa00162/.conda/envs/tfgpu114/bin/python
# coding: utf-8
"""
pandas_profilingでテーブル形式ファイルの統計情報をhtmlで出力する
htmlは以下の項目を表示
- OverView:データサマリー
- Variables:列各の統計情報。ヒストグラムも表示する
- Correlation:各列のピアソン/スピアマンの相関係数
- Sample:実データ

Usage:
    # tsvのサマリーhtmlをtmpディレクトリに出力
    $ summary_html_pandas_profiling.py \
        --output_dir tmp \
        --input_file '/gpfsx01/home/aaa00162/jupyterhub/notebook/H3-058/work/data/represent/raw_represent_all.hERG_IC50.tsv'

    # エクセルの各シートのサマリーhtmlをtmpディレクトリに出力。シートごとにhtml出力。n_skiprowで各シートの先頭8行飛ばす
    $ summary_html_pandas_profiling.py \
        --output_dir tmp \
        --input_file '/gpfsx01/home/aaa00162/jupyterhub/notebook/H3-058/OrigData/US034-0004376_Appendix_2.xlsx' \
        --n_skiprow 8 \
"""
import os, argparse
import pandas as pd
import pandas_profiling as pdp
import pathlib
import openpyxl
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="input file path.")
    parser.add_argument("--output_dir", type=str, default=None, help="html output dir path.")
    parser.add_argument("--n_skiprow", type=int, default=None, help="number of rows to skip.")
    parser.add_argument("--header", type=int, default=0, help="pd.read_csv arg header.")
    args = parser.parse_args()

    path = args.input_file

    if args.output_dir is None:
        output_dir = '.'
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = args.output_dir

    if args.n_skiprow is None:
        skiprows = None
    else:
        skiprows = range(args.n_skiprow)

    if str(pathlib.Path(path).suffix) in ['.txt', '.tsv']:
        df = pd.read_csv(path, sep='\t', skiprows=skiprows, header=args.header)
        profile = pdp.ProfileReport(df)
        profile.to_file(output_file=f"{output_dir}/{str(pathlib.Path(path).name)}_profile.html")

    if str(pathlib.Path(path).suffix) in ['.csv']:
        df = pd.read_csv(path, skiprows=skiprows, header=args.header)
        profile = pdp.ProfileReport(df)
        profile.to_file(output_file=f"{output_dir}/{str(pathlib.Path(path).name)}_profile.html")

    if str(pathlib.Path(path).suffix) in ['.xlsx', '.xlsm', '.xlsb', '.xls']:
        book = openpyxl.load_workbook(path)
        sheets = book.sheetnames # シート名をすべて取得
        #print(sheets)
        for s in sheets:
            try:
                df = pd.read_excel(path, sheet_name=s, skiprows=skiprows, header=args.header)
                #display(df.head())
                for col in df.columns.tolist():
                    if 'Unnamed' in col:
                        df = df.drop(col, axis=1)
                profile = pdp.ProfileReport(df)
                profile.to_file(output_file=f"{output_dir}/{str(pathlib.Path(path).name)}_{s}_profile.html")
            except Exception as e:
                print(e)