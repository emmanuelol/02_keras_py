# -*- coding: UTF-8 -*-
"""
tsvファイルの統計値取得するコマンドラインモジュール
Usage:
    $ python tsv_summary.py <input files. (csv, tsv etc.)> --out out.scv
"""
#__author__ = 'tmp04415'
import os, sys, argparse
import pandas as pd

def describe(file_path, sep='\t', header='infer', nb_sample=10):
    print("INFO: load file [{}]".format(file_path))
    dftmp = pd.read_csv(file_path, sep=sep, header=header, error_bad_lines=False, low_memory=True)

    recs = list()
    for col, ctype in dftmp.dtypes.items():
        if ctype == "object":
            min_val = dftmp[col].dropna().min()
            max_val = dftmp[col].dropna().max()
        else:
            min_val = dftmp[col].min()
            max_val = dftmp[col].max()

        recs.append([
            os.path.basename(file_path),
            col,
            ctype.name,
            dftmp[col].count(),
            dftmp[col].nunique(),
            dftmp[col].isnull().sum(),
            min_val,
            max_val,
            get_sample(dftmp[col], nb_sample)
        ])

    return pd.DataFrame(recs, columns=["file","col","type","count","unique","null","min","max","samples"])


def get_sample(srrow, nb_sample=10):
    srtmp = srrow.value_counts()
    if srtmp.size > nb_sample:
        srtmp = srtmp.iloc[:nb_sample]
    samples = list()
    for idx, val in srtmp.items():
        samples.append("{}({})".format(idx, val))
    return "/".join(samples)


def main(opts):
    header = 'infer'
    if opts.no_header:
        header = None

    dfout = pd.DataFrame()
    for itr in opts.files:
        dftmp = describe(itr, sep=opts.sep, header=header, nb_sample=opts.nb_sample)
        dfout = pd.concat([dfout, dftmp])

    dfout.reset_index(inplace=True, drop=True)
    dfout.index.name = '#'
    if opts.output is not None:
        dfout.to_csv(opts.output, sep="\t")
        print("INFO: save file [{}] {}".format(opts.output, dfout.shape))
    else:
        dfout.to_csv(sys.stdout, sep="\t")


def get_options() :
    usage = 'summarize table format files.'
    opt_parser = argparse.ArgumentParser(description=usage)
    opt_parser.add_argument(
        "files",
        metavar="FILE",
        action="store",
        type=str,
        nargs="+",
        help="input files. (csv, tsv etc.)"
    )
    opt_parser.add_argument(
        "-s",
        "--sep",
        dest="sep",
        action="store",
        type=str,
        default='\t',
        help="Delimiter to use. (default: TAB)"
    )
    opt_parser.add_argument(
        "-n",
        "--no-header",
        dest="no_header",
        action="store_true",
        default=False,
        help="Row number(s) to use as the column names if there is on header line. (default: off)"
    )
    opt_parser.add_argument(
        "-nb",
        "--nb-sample",
        dest="nb_sample",
        action="store",
        type=int,
        default=10,
        help="# of samples. (default: 10)"
    )
    opt_parser.add_argument(
        "-o",
        "--out",
        dest="output",
        action="store",
        type=str,
        default=None,
        help="output file. (default: stdout)"
    )
    opts = opt_parser.parse_args()
    return opts

if __name__ == '__main__':
    opts = get_options()
    main(opts)
