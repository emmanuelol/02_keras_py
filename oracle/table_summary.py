# -*- coding: UTF-8 -*-

'''
Created on 2011/10/26
@author: T.Yamamoto
20180425 edit. Use aaa00162 python
(本ファイル1行目 /home/aaa00162/bin/python -> /home/aaa00162/.conda/envs/py27/bin/python2.7)

Oracleの1テーブルについて、各カラムの最大値や最小値などの統計情報をタブ区切りでファイル出力するコマンドラインスクリプト
出力ファイルは以下の11列のファイル
#	OWNER	TABLE_NAME	COLUMN_NAME	DATA_TYPE	COUNT	UNIQUE	NULL	MIN	MAX	EXAMPLE	COMMENTS
Usage:
    $ OWNER="db user" # アクセスするDBのユーザ名
    $ PASS="db user password" # アクセスするDBのユーザのパスワード
    $ DB="databese name" # アクセスするDBの名前
    $ table_name="tablename" # 統計情報取るテーブル名
    $ python summary_tmp10014.py -d "${OWNER}/${PASS}@${DB}" -t ${table_name} > ./table_summary/${table_name}.summary
'''
import sys
import os
os.environ['NLS_LANG'] = 'Japanese_Japan.UTF8'

import argparse
import cx_Oracle

class table_checker(object):
    def __init__(self):
        '''コンストラクタ
        '''
        self.example_pattern_max = 30

    def init(self, driver):
        self.conect = cx_Oracle.connect(driver)
        self.cursor = self.conect.cursor()
        self.cursor_count = self.conect.cursor()
        self.cursor_example = self.conect.cursor()
        self.table_counts = dict()

    def term(self):
        self.cursor.close()
        self.conect.close()


    def get_summary(self, owner_name, table_name=None):
        try:
            for itr_item in self._get_column_names(owner_name, table_name):
                if itr_item.table_name.endswith("XML"):
                    continue
                elif itr_item.table_name.endswith("XMLDB"):
                    continue

                if not itr_item.data_type.endswith('LOB'):
                    itr_item.num_of_all = self._get_all_count(owner_name, itr_item.table_name, itr_item.column_name)
                    itr_item.num_of_distinct = self._get_distinct_count(owner_name, itr_item.table_name, itr_item.column_name)
                    itr_item.num_of_null = self._get_null_count(owner_name, itr_item.table_name, itr_item.column_name)
                    (itr_item.min_val, itr_item.max_val) = self._get_min_max(owner_name, itr_item.table_name, itr_item.column_name)
                    itr_item.example = self._get_examples(owner_name, itr_item.table_name, itr_item.column_name)

                yield itr_item

        except cx_Oracle.DatabaseError as dberr:
            sys.stderr.write("ERROR: [%s\t%s\t%s]\n" % (owner_name, itr_item.table_name, itr_item.column_name))
            raise dberr

    def _get_column_names(self, owner_name, table_name):
        if table_name is not None:
            sql = """
                SELECT
                    TMP1.OWNER
                ,   TMP1.TABLE_NAME
                ,   TMP1.COLUMN_NAME
                ,   TMP1.DATA_TYPE
                ,   TMP2.COMMENTS
                FROM
                    ALL_TAB_COLUMNS TMP1
                ,   USER_COL_COMMENTS TMP2
                WHERE
                    TMP1.TABLE_NAME = TMP2.TABLE_NAME
                AND
                    TMP1.COLUMN_NAME = TMP2.COLUMN_NAME
                AND
                    TMP1.TABLE_NAME NOT LIKE 'BIN$%'
                AND
                    TMP1.OWNER = :owner
                AND
                    TMP1.TABLE_NAME = :tbl
            """
            vals = self.cursor.execute(sql, owner=owner_name, tbl=table_name).fetchall()
            for itr in vals:
                yield ColumnNameItem(itr)
        else :
            sql = """
                SELECT
                    TMP1.OWNER
                ,   TMP1.TABLE_NAME
                ,   TMP1.COLUMN_NAME
                ,   TMP1.DATA_TYPE
                ,   TMP2.COMMENTS
                FROM
                    ALL_TAB_COLUMNS TMP1
                ,   USER_COL_COMMENTS TMP2
                WHERE
                    TMP1.TABLE_NAME = TMP2.TABLE_NAME
                AND
                    TMP1.COLUMN_NAME = TMP2.COLUMN_NAME
                AND
                    TMP1.TABLE_NAME NOT LIKE 'BIN$%'
                AND
                    TMP1.OWNER = :owner
            """
            vals = self.cursor.execute(sql, owner=owner_name).fetchall()
            for itr in vals:
                yield ColumnNameItem(itr)

    def _get_all_count(self, owner_name, table_name, col_name):
        curkey = owner_name + "." + table_name
        if curkey in self.table_counts:
            return self.table_counts[curkey]

        sql = """
            SELECT
                COUNT(*)
            FROM
                {0}.{1}
        """.format(owner_name, table_name)

        values = self.cursor_count.execute(sql).fetchone()
        if values is not None:
            self.table_counts[curkey] = values[0]
        else:
            self.table_counts[curkey] = 0

        return self.table_counts[curkey]

    def _get_distinct_count(self, owner_name, table_name, col_name):
        sql = """
            SELECT
                COUNT(*)
            FROM
                (
                    SELECT
                        DISTINCT
                        "{2}"
                    FROM
                        {0}.{1}
                    WHERE
                        "{2}" IS NOT NULL
                )
        """.format(owner_name, table_name, col_name)

        values = self.cursor_count.execute(sql).fetchone()
        if values is not None:
            return values[0]
        else:
            return 0

    def _get_null_count(self, owner_name, table_name, col_name):
        sql = """
            SELECT
                COUNT(*)
            FROM
                {0}.{1}
            WHERE
                "{2}" IS NULL
        """.format(owner_name, table_name, col_name)

        values = self.cursor_count.execute(sql).fetchone()
        if values is not None:
            return values[0]
        else:
            return 0

    def _get_min_max(self, owner_name, table_name, col_name):
        sql = """
            SELECT
                MIN({2})
            ,   MAX({2})
            FROM
                {0}.{1}
            WHERE
                "{2}" IS NOT NULL
        """.format(owner_name, table_name, col_name)

        values = self.cursor_count.execute(sql).fetchone()
        if values is not None:
            return values
        else:
            return (None, None)

    def _get_examples(self, owner_name, table_name, col_name):
        sql = """
            SELECT
            	KEY
            ,	CNT
            FROM
            	(
            		SELECT
            			TBL.{2} AS KEY
            		,	COUNT(*) AS CNT
            		,	RANK() OVER(ORDER BY COUNT(*) DESC, "{2}") AS RNK
            		FROM
            			{0}.{1} TBL
            		WHERE
            			"{2}" IS NOT NULL
            		GROUP BY TBL.{2}
            	) TMP
            WHERE
            	TMP.RNK <= {3}
        """.format(owner_name, table_name, col_name, self.example_pattern_max)

        vals = self.cursor.execute(sql).fetchall()
        examples = ["%s(%s)" % itr for itr in vals]
        return "/".join(examples)

    def _get_example_char(self, owner_name, table_name, col_name):
        sql = """
            SELECT
                LISTAGG(TMP1.KEY || '(' || TMP1.CNT || ')', '/') WITHIN GROUP(ORDER BY TMP1.CNT DESC) EXAMPLE
            FROM
                (
                    SELECT
                        TBL.%s KEY
                    ,    COUNT(*) CNT
                    FROM
                        %s.%s TBL
                    WHERE
                        TBL.%s IS NOT NULL
                    GROUP BY TBL.%s
                ) TMP1
        """ % (col_name, owner_name, table_name, col_name, col_name)

        return self.cursor_example.execute(sql).fetchone()[0]

    def _get_example_one(self, owner_name, table_name, col_name):
        sql = """
            SELECT
                TBL.%s
            FROM
                %s.%s TBL
            WHERE
                TBL.%s IS NOT NULL
            ORDER BY TBL.%s
        """ % (col_name, owner_name, table_name, col_name, col_name)

        example = self.cursor_example.execute(sql).fetchone()
        if example is not None:
            return example[0]
        else:
            return None

    def _get_example_number(self, owner_name, table_name, col_name):
        sql = """
            SELECT
                MIN(TBL.%s)
            ,    MAX(TBL.%s)
            FROM
                %s.%s TBL
            WHERE
                TBL.%s IS NOT NULL
        """ % (col_name, col_name, owner_name, table_name, col_name)
        results = map(str, self.cursor_example.execute(sql).fetchone())
        return " - ".join(results)



    def get_count(self, owner_name, table_name=None):
        for itr_item in self._get_table_names(owner_name, table_name):
#            if itr_item.table_name.endswith("XML"):
#                continue
#            elif itr_item.table_name.endswith("XMLDB"):
#                continue
            itr_item.num_of_record = self._get_record_count(owner_name, itr_item.table_name)

            yield itr_item

    def _get_table_names(self, owner_name, table_name):
        if table_name is not None:
            sql = """
                SELECT
                    :owner OWNER
                ,    TMP1.TABLE_NAME
                ,    TMP2.COMMENTS
                ,    TMP1.NUM_ROWS
                FROM
                    USER_TABLES TMP1
                ,    USER_TAB_COMMENTS TMP2
                WHERE
                    TMP1.TABLE_NAME = TMP2.TABLE_NAME
                AND
                    TMP1.TABLE_NAME = :tbl
                ORDER BY TMP1.TABLE_NAME
            """
            for itr in self.cursor.execute(sql, owner=owner_name, tbl=table_name):
                yield TableNameItem(itr)
        else :
            sql = """
                SELECT
                    :owner OWNER
                ,    TMP1.TABLE_NAME
                ,    TMP2.COMMENTS
                ,    TMP1.NUM_ROWS
                FROM
                    USER_TABLES TMP1
                ,    USER_TAB_COMMENTS TMP2
                WHERE
                    TMP1.TABLE_NAME = TMP2.TABLE_NAME
                ORDER BY TMP1.TABLE_NAME
            """
            for itr in self.cursor.execute(sql, owner=owner_name):
                yield TableNameItem(itr)

    def _get_record_count(self, owner_name, table_name):
        sql = """
            SELECT
                COUNT(*) NUM_OF_RECORD
            FROM
                %s.%s
        """ % (owner_name, table_name)

        counts = self.cursor_count.execute(sql).fetchone()
        if counts is not None:
            return counts[0]
        else:
            return 0


class ColumnNameItem(object):
    def __init__(self, values):

        self.owner = values[0]
        self.table_name = values[1]
        self.column_name = values[2]
        self.data_type = values[3]
        self.comment = values[4]
        self.num_of_all = 0
        self.num_of_null = 0
        self.num_of_distinct = 0
        self.min_val = None
        self.max_val = None
        self.example = None

    def __str__(self):
        return "\t".join(
            map(str, (self.owner, self.table_name, self.column_name, self.data_type,
                      self.num_of_all, self.num_of_distinct, self.num_of_null,
                      self.min_val, self.max_val, self.example, self.comment))
            )

class TableNameItem(object):
    def __init__(self, values):
        self.owner = values[0]
        self.table_name = values[1]
        self.comment = values[2]
        self.num_of_record = values[3]

    def __str__(self):
        return "\t".join(
            map(str, (self.owner, self.table_name, self.num_of_record, self.comment)))


def get_options() :
    '''オプション解析
    '''

    usage = """Usage: %prog [options] \n\
    output to STDOUT.
    """
    opt_parser = argparse.ArgumentParser(description=usage)
    opt_parser.add_argument(
        "-v",
        "--verbose",
        dest="is_verbose",
        default=False,
        action="store_true",
        help="verbose"
    )
    opt_parser.add_argument(
        "-d",
        "--driver",
        dest="driver",
        default="",
        help="driver. USER/PASS@SID"
    )
    opt_parser.add_argument(
        "-t",
        "--table",
        dest="table_name",
        default=None,
        help="table name for limit output."
    )
    opt_parser.add_argument(
        "-c",
        "--count",
        dest="is_count_only",
        default=False,
        action="store_true",
        help="count records per table."
    )
    opts = opt_parser.parse_args()
    if opts.driver == "":
        opt_parser.error("Please select -d options.")

    return opts



if __name__ == '__main__':
    checker = None
    try:
        opts = get_options()

        checker = table_checker()
        checker.init(opts.driver)
        user_name = opts.driver.split("/")[0]
        if opts.is_count_only:
            print("OWNER\tTABLE_NAME\tNUM_RECORD\tCOMMENTS")
            for itr_rec in checker.get_count(user_name, opts.table_name):
                print(itr_rec)

        else:
            print("OWNER\tTABLE_NAME\tCOLUMN_NAME\tDATA_TYPE\tCOUNT\tUNIQUE\tNULL\tMIN\tMAX\tEXAMPLE\tCOMMENTS")
            for itr_rec in checker.get_summary(user_name, opts.table_name):
                print(itr_rec)

    finally:
        if checker is not None:
            checker.term()
