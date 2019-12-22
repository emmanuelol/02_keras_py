# -*- coding: UTF-8 -*-

'''
Created on 2012/09/19
@author: Takamune.Y
20180425 edit. Use aaa00162 python
(/home/aaa00162/bin/python -> /home/aaa00162/.conda/envs/py27/bin/python2.7)

Oracleのsqlを書いたファイルからsqlを実行するコマンドラインスクリプト
Usage:
    $ OWNER="db user" # アクセスするDBのユーザ名
    $ PASS="db user password" # アクセスするDBのユーザのパスワード
    $ DB="databese name" # アクセスするDBの名前
    $ python executer_tmp10014.py -d "${OWNER}/${PASS}@${DB}" tmp.sql > output.dat
'''
import sys
import os
os.environ['NLS_LANG'] = 'Japanese_Japan.JA16SJIS'

import argparse
import cx_Oracle

def get_options() :
    '''get_options() -> opts
    コマンドライン引数の解析

    o opts : オプション格納 dict
    '''

    usage = u'''input from FILE. output to STDOUT. log message to STDERR.
    SQLでbind変数を使う場合は、以下のように書きます。
    SELECT * FROM TBL WHERE ID = :id
    '''
    opt_parser = argparse.ArgumentParser(description=usage)
    opt_parser.add_argument(
        "sql_file",
        metavar="FILE",
        action="store",
        type=file,
#        nargs="+",
        help="sql file."
    )
    opt_parser.add_argument(
        "-d",
        "--driver",
        dest="driver",
        action="store",
        type=str,
        default='BIOREF/BIOREF@DDD6',
        help="driver string. (default: BIOREF/BIOREF@DDD6)"
    )
    opt_parser.add_argument(
        "-b",
        "--binds",
        dest="binds",
        action="store",
        type=str,
        default="",
        help="bind variables. 'key1:value1[,key2:value]' (default: '')"
    )
    opt_parser.add_argument(
        "-bvd",
        "--bind_val_delim",
        dest="bind_val_delim",
        action="store",
        type=str,
        default=":",
        help="delimiter between key and value. (default: :)"
    )
    opt_parser.add_argument(
        "-bpd",
        "--bind_pair_delim",
        dest="bind_pair_delim",
        action="store",
        type=str,
        default=",",
        help="delimiter between key:value pairs. (default: ,)"
    )
    opt_parser.add_argument(
        "-s",
        "--val_sep",
        dest="val_sep",
        action="store",
        type=str,
        default="\t",
        help="value separator. (default: TAB)"
    )
    opt_parser.add_argument(
        "-H",
        "--no_header",
        dest="no_header",
        action="store_true",
        default=False,
        help="no print column header. (default: False)"
    )
    opt_parser.add_argument(
        "-f",
        "--fmt",
        dest="datefmt",
        action="store",
        default=None,
        help="output datetime format. '%%Y/%%m/%%d' (default: None)"
    )
    opts = opt_parser.parse_args()
    return opts

def run_sql(opts):
    '''run_sql(opts) -> results
    o opts : コマンドラインオプション格納 dict

    o results : strのリスト
    '''
    conn = None
    curs = None
    itr = None
    try:
        # sql読み込み
        sql_content = opts.sql_file.read()

        # connection作成
        conn = cx_Oracle.connect(opts.driver)
        curs = conn.cursor()

        # bind変数を定義する文字列のリスト
        bind_strs = list()
        if os.path.isfile(opts.binds):
            with open(opts.binds, 'r') as bind_file:
                bind_strs = [itr_line.rstrip("\r\n") for itr_line in bind_file.readlines()]
        else:
            bind_strs.append(opts.binds)

        # bind変数がファイルで指定された場合は、１行ずつ繰り返し実行する
        for itr_bind in bind_strs:
            # bind変数用dict
            prms = dict()
            if itr_bind != '':
                prms = dict(item.split(opts.bind_val_delim) for item in itr_bind.split(opts.bind_pair_delim))

            # sql実行
            for itr in _execute_sql(curs, sql_content, prms, opts.datefmt):
                yield itr

    except cx_Oracle.Warning as ex:
        for itr_ex in ex.args:
            if itr is not None:
                sys.stderr.write(u"WARN: %s [%s]\n" % (itr_ex, itr))
            else:
                sys.stderr.write(u"WARN: %s\n" % (itr_ex))

    except cx_Oracle.Error as err:
        for itr_err in err.args:
            if itr is not None:
                sys.stderr.write(u"ERROR1: %s [%s]\n" % (itr_err, itr))
            else:
                sys.stderr.write(u"ERROR1: %s\n" % (itr_err))

    finally:
        if curs is not None:
            curs.close()

        if conn is not None:
            conn.close()

def _execute_sql(curs, sql_content, prms, datefmt=None):
    '''_execute_sql(curs, sql_content, prms) -> vals

    o curs : cx_Oracle.cursorオブジェクト
    o sql_content : sql文字列
    o prms : bind変数格納 dict

    o vals : strのリスト
    '''
    # SQL実行
    curs.execute(sql_content, prms)

    # header出力
    if not opts.no_header:
        yield [str(itr[0]) for itr in curs.description]
        opts.no_header = True

    # レコード出力
    col_types = [itr[1] for itr in curs.description]
    for itr_rec in curs:    # fetchall()を使わないこと
        vals = [_str(itr) for itr in itr_rec]

        for (idx, itr_col_type) in enumerate(col_types):
            if itr_col_type == cx_Oracle.CLOB:
                # CLOBが途中で切れないように対応
                if itr_rec[idx] is not None or itr_rec[idx] != '':
                    vals[idx] = itr_rec[idx].read()

            elif itr_col_type == cx_Oracle.DATETIME and datefmt is not None:
                if itr_rec[idx] is not None and itr_rec[idx] != '':
                    vals[idx] = itr_rec[idx].strftime(datefmt)

        yield vals

def _str(val):
    if val is None:
        return ''
    return str(val)

if __name__ == '__main__':
    opts = get_options()
    itr_rslt = None
    try:
        for itr_rslt in run_sql(opts):
            print(opts.val_sep.join(itr_rslt))

    except Exception as ex:
        if itr_rslt is not None:
            sys.stderr.write("ERROR0:%s [%s]\n" % (ex, itr_rslt))
        else:
            sys.stderr.write("ERROR0:%s\n" % (ex))
    finally:
        pass
