# -*- coding: utf-8 -*-
"""
xlsxwriterでGradCamの結果一覧エクセルを作成する

tf_test環境でしか実行できない（xlsxwriter はtf_test環境にしか入れてない）
tf_test環境: /gpfsx01/home/aaa00162/.conda/envs/tf_test
tf_test環境ログインコマンド: $ source ~/conda_tmp10014_tf_test.login
"""
import xlsxwriter
import os, sys, glob

def make(out_xlsx_path, gradcam_img_dir, maxrow):
    """
    validation/test set のGradCamの結果をエクセルに出力する
    12taskの結果を1シートにまとめる
    予測ラベル0(negative)/1(positive)と確率値両方出力する版も作成する

    Args:
        wb: 作成するエクセルのファイル(xlsxwriter.Workbook)
        gradcam_img_dir: GradCamかけた validation/test set の画像ディレクトリ
        maxrow: 作成するエクセルの行数(=validation/test setの画像の種類+2)
    Return:
        なし（wbにシート追加してエクセル出力）
    """
    # task name
    task_name_list = ['NR.AhR', 'NR.AR', 'NR.AR.LBD', 'NR.Aromatase', 'NR.ER', 'NR.ER.LBD', 'NR.PPAR.gamma', 'SR.ARE', 'SR.ATAD5', 'SR.HSE', 'SR.MMP', 'SR.p53']

    # 作成するエクセルのファイル
    wb = xlsxwriter.Workbook(out_xlsx_path)

    # 作成するエクセルシート
    ws = wb.add_worksheet('Tox21_GradCam')

    # 列名用フォーマット
    title_format = wb.add_format({'bold': True, 'fg_color': '#BFBFBF'})
    # 文字用フォーマット
    text_format=wb.add_format({'bold': False, 'font_size':'11'})
    # 数字用フォーマット
    int_format=wb.add_format({'bold': False, 'font_size':'16'})

    # 作成するエクセルの列幅、列名設定
    ws.set_column(0, 0, 17.75) # ID column
    ws.write_string(1, 0, 'ID', title_format) # A2 cell
    for task_id, task_name in enumerate(task_name_list):
        # 列番号
        col_num = 1+4*task_id

        ws.set_column(col_num, col_num, 20) # Structure column
        ws.write_string(0, col_num, task_name, title_format)# task名を1行目にする
        ws.write_string(1, col_num, 'Structure', title_format)# 2行目

        ws.set_column(col_num+1, col_num+1, 4) # Answer column
        ws.write_string(1, col_num+1, 'Answer', title_format)# 2行目

        ws.set_column(col_num+2, col_num+2, 4) # Prediction column
        ws.write_string(1, col_num+2, 'Prediction', title_format)# 2行目

        ws.set_column(col_num+3, col_num+3, 6.25) # Probability column
        ws.write_string(1, col_num+3, 'Probability', title_format)# 2行目

    ws.set_column(col_num+4, col_num+4, 64) # Comment column
    ws.write_string(1, col_num+4, 'Comment', title_format)# 2行目

    # 各taskで回す
    for task_id, task_name in enumerate(task_name_list):
        print(task_id, task_name)
        # 列番号
        col_num = 1+4*task_id

        # gradCamの結果画像のパス
        filelist = glob.glob(gradcam_img_dir+'/task'+str(task_id)+'/**/*jpg', recursive=True)
        for i, filename in enumerate(sorted(filelist, key=lambda x: os.path.basename(x).split('_')[0])):
            # ヘッダは2行あるので、データが入るのは行数(i)+2行目から
            # 行の高さ変更
            ws.set_row(i+2, 110)

            # エクセルに画像ID書き込み
            id_name = os.path.basename(filename).split('_')[0]
            ws.write_string(i+2, 0, id_name, text_format)

            # エクセルに画像ファイル貼り付け
            ws.insert_image(i+2, col_num, filename, {'positioning': 1, 'x_scale': 0.4, 'y_scale': 0.4})

            # 確率値
            prob = os.path.basename(filename).split('=')[1]
            prob = prob.replace('.jpg', '')
            #print(prob)

            # ファイルパスから画像の正解クラスと予測クラス探索し、エクセルに書き込み
            posi_nega_flg = os.path.basename(os.path.dirname(filename))
            if 'TP' == posi_nega_flg:
                ws.write_number(i+2, col_num+1, 1, int_format)
                ws.write_number(i+2, col_num+2, 1, int_format)
            elif 'TN' == posi_nega_flg:
                ws.write_number(i+2, col_num+1, 0, int_format)
                ws.write_number(i+2, col_num+2, 0, int_format)
            elif 'FP' == posi_nega_flg:
                ws.write_number(i+2, col_num+1, 0, int_format)
                ws.write_number(i+2, col_num+2, 1, int_format)
            elif 'FN' == posi_nega_flg:
                ws.write_number(i+2, col_num+1, 1, int_format)
                ws.write_number(i+2, col_num+2, 0, int_format)
            # 正解がNANの時
            elif 'NAN' == posi_nega_flg:
                ws.write_number(i+2, col_num+1, -1, int_format)
                if float(prob) < 0.5:
                    ws.write_number(i+2, col_num+2, 0, int_format)
                else:
                    ws.write_number(i+2, col_num+2, 1, int_format)
            ws.write(i+2, col_num+3, float(prob), int_format)

    # 1-maxrow行でオートフィルタ
    ws.autofilter('A2:AX'+str(maxrow))

    # closeしないと保存されない
    wb.close()

if __name__ == '__main__':
    print('gradcam_xlsx_onesheet_label_prob.py: loaded as script file')
else:
    print('gradcam_xlsx_onesheet_label_prob.py: loaded as module file')
