# -*- coding: utf-8 -*-
"""
xlsxwriterを使ったメソッド
"""
import xlsxwriter
import os, sys, glob, pathlib

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
# 自作モジュールimport
from dataset import util

def gradcam_xlsx_onesheet_label_prob(out_xlsx_path, gradcam_img_dir, task_name_list
                                     , sheet_name='GradCam++', id_col_name='id', pred_threshold=0.5, is_y=True
                                     , img_extension='jpg', x_scale=0.2, y_scale=0.2):
    """
    validation/test set のGradCamの結果をエクセルに出力する
    各taskの結果を1シートにまとめる
    予測ラベル0(negative)/1(positive)と確率値両方出力する
    Args:
        out_xlsx_path: 作成するエクセルのファイル(xlsxwriter.Workbook)
        gradcam_img_dir: GradCamかけた validation/test set の画像ディレクトリ
        task_name_list: タスク名のリスト
        sheet_name: エクセルのシート名
        id_col_name:ID列の列名
        pred_threshold: 予測スコアのposi/nega分ける閾値。バイナリ分類なので、デフォルトは0.5とする
        is_y:正解ラベル列(y列)に値入れるか.Falseなら空にする
        img_extension:画像ファイルの拡張子
        x_scale, y_scale: エクセルに貼り付ける画像の縮小率
    Return:
        なし（wbにシート追加してエクセル出力）
    """
    print(out_xlsx_path)
    # 作成するエクセルのファイル
    wb = xlsxwriter.Workbook(out_xlsx_path)

    # 作成するエクセルシート
    ws = wb.add_worksheet(sheet_name)

    # 列名用フォーマット
    title_format = wb.add_format({'bold': True, 'fg_color': '#BFBFBF'})
    # 文字用フォーマット
    text_format=wb.add_format({'bold': False, 'font_size':'11'})
    # 数字用フォーマット
    int_format=wb.add_format({'bold': False, 'font_size':'16'})

    # 作成するエクセルの列幅、列名設定
    ws.set_column(0, 0, 17.75) # ID column
    ws.write_string(1, 0, id_col_name, title_format) # A2 cell
    for task_id, task_name in enumerate(task_name_list):
        # 列番号
        col_num = 1+4*task_id

        ws.set_column(col_num, col_num, 20) # Structure column
        ws.write_string(0, col_num, 'task'+str(task_id), title_format)# task_idを1行目にする
        ws.write_string(1, col_num, task_name, title_format)# task名を2行目にする
        ws.write_string(2, col_num, 'Structure', title_format)# 3行目

        ws.set_column(col_num+1, col_num+1, 4) # Answer column
        ws.write_string(2, col_num+1, 'y', title_format)# 3行目

        ws.set_column(col_num+2, col_num+2, 4) # Prediction column
        ws.write_string(2, col_num+2, 'Prediction_y_threshold='+str(pred_threshold), title_format)# 3行目

        ws.set_column(col_num+3, col_num+3, 6.25) # Probability column
        ws.write_string(2, col_num+3, 'Probability_Score', title_format)# 3行目

    ws.set_column(col_num+4, col_num+4, 64) # Comment column
    ws.write_string(2, col_num+4, 'Comment', title_format)# 3行目

    # 各taskで回す
    for task_id, task_name in enumerate(task_name_list):
        #print(task_id, task_name)
        # 列番号
        col_num = 1+4*task_id

        # gradCamの結果画像のパス
        filelist = util.find_img_files(gradcam_img_dir+'/task'+str(task_id))#filelist = glob.glob(gradcam_img_dir+'/task'+str(task_id)+'/**/*jpg', recursive=True)
        for i, filename in enumerate(sorted(filelist, key=lambda x: os.path.basename(x).split('_')[0])):
            #print(filename)
            # ヘッダは2行あるので、データが入るのは行数(i)+2行目から
            # 行の高さ変更
            ws.set_row(i+3, 110)

            # エクセルに画像ID書き込み
            id_name = os.path.basename(filename).split('_')[0]
            ws.write_string(i+3, 0, id_name, text_format)

            # エクセルに画像ファイル貼り付け
            ws.insert_image(i+3, col_num, filename, {'positioning': 1, 'x_scale': x_scale, 'y_scale': y_scale})

            # 確率値
            prob = os.path.basename(filename).split('=')[1]
            prob = prob.replace('.'+img_extension, '')
            #print(prob)

            # ファイルパスから画像の正解クラスと予測クラス探索し、エクセルに書き込み
            if is_y == True:
                posi_nega_flg = os.path.basename(os.path.dirname(filename))
                if 'TP' == posi_nega_flg:
                    ws.write_number(i+3, col_num+1, 1, int_format)
                    ws.write_number(i+3, col_num+2, 1, int_format)
                elif 'TN' == posi_nega_flg:
                    ws.write_number(i+3, col_num+1, 0, int_format)
                    ws.write_number(i+3, col_num+2, 0, int_format)
                elif 'FP' == posi_nega_flg:
                    ws.write_number(i+3, col_num+1, 0, int_format)
                    ws.write_number(i+3, col_num+2, 1, int_format)
                elif 'FN' == posi_nega_flg:
                    ws.write_number(i+3, col_num+1, 1, int_format)
                    ws.write_number(i+3, col_num+2, 0, int_format)
                # 正解がNANの時
                elif 'NAN' == posi_nega_flg:
                    ws.write_number(i+3, col_num+1, -1, int_format)
                    if float(prob) < pred_threshold:
                        ws.write_number(i+3, col_num+2, 0, int_format)
                    else:
                        ws.write_number(i+3, col_num+2, 1, int_format)
            else:
                ws.write_string(i+3, col_num+1, '')
                if float(prob) < pred_threshold:
                    ws.write_number(i+3, col_num+2, 0, int_format)
                else:
                    ws.write_number(i+3, col_num+2, 1, int_format)
            ws.write(i+3, col_num+3, float(prob), int_format)

    # 1-maxrow行でオートフィルタ maxrow: 作成するエクセルの行数(=validation/test setの画像の種類+2)
    #ws.autofilter('A3:AX'+str(maxrow))

    # closeしないと保存されない
    wb.close()

def multitask_gradcam_xlsx_onesheet_label_prob(out_xlsx_path, gradcam_img_dir, task_name_list, class_name_list
                                                , sheet_name='GradCam++', id_col_name='id', pred_threshold=0.5, is_y=True
                                                , img_extension='jpg', x_scale=0.2, y_scale=0.2):
    """
    multitaskモデルでvalidation/test set のGradCamの結果をエクセルに出力する
    各taskの結果を1シートにまとめる
    予測ラベル0(negative)/1(positive)と確率値両方出力する
    Args:
        out_xlsx_path: 作成するエクセルのファイル(xlsxwriter.Workbook)
        gradcam_img_dir: GradCamかけた validation/test set の画像ディレクトリ
        task_name_list: タスク名のリスト
        class_name_list: 各タスクのクラス名のリスト
        sheet_name: エクセルのシート名
        id_col_name:ID列の列名
        pred_threshold: 予測スコアのposi/nega分ける閾値。バイナリ分類なので、デフォルトは0.5とする
        is_y:正解ラベル列(y列)に値入れるか.Falseなら空にする
        img_extension:画像ファイルの拡張子
        x_scale, y_scale: エクセルに貼り付ける画像の縮小率
    Return:
        なし（wbにシート追加してエクセル出力）
    """
    print(out_xlsx_path)
    # 作成するエクセルのファイル
    wb = xlsxwriter.Workbook(out_xlsx_path)

    # 作成するエクセルシート
    ws = wb.add_worksheet(sheet_name)

    # 列名用フォーマット
    title_format = wb.add_format({'bold': True, 'fg_color': '#BFBFBF'})
    # 文字用フォーマット
    text_format=wb.add_format({'bold': False, 'font_size':'11'})
    # 数字用フォーマット
    int_format=wb.add_format({'bold': False, 'font_size':'16'})

    # 作成するエクセルの列幅、列名設定
    ws.set_column(0, 0, 17.75) # ID column
    ws.write_string(1, 0, id_col_name, title_format) # A2 cell
    count=0
    for task_id, task_name in enumerate(task_name_list):
        for class_id, class_name in enumerate(class_name_list):
            # 列番号
            col_num = 1+4*(count)
            #print(col_num, task_id, task_name, class_id, class_name)

            ws.set_column(col_num, col_num, 20) # Structure column
            ws.write_string(0, col_num, 'task'+str(task_id), title_format)# task_idを1行目にする
            ws.write_string(1, col_num, str(class_name), title_format)# task名を2行目にする
            ws.write_string(2, col_num, 'Structure', title_format)# 3行目

            ws.set_column(col_num+1, col_num+1, 4) # Answer column
            ws.write_string(2, col_num+1, 'y', title_format)# 3行目

            ws.set_column(col_num+2, col_num+2, 4) # Prediction column
            ws.write_string(2, col_num+2, 'Prediction_y_threshold='+str(pred_threshold), title_format)# 3行目

            ws.set_column(col_num+3, col_num+3, 6.25) # Probability column
            ws.write_string(2, col_num+3, 'Probability_Score', title_format)# 3行目

            count += 1

    ws.set_column(col_num+4, col_num+4, 64) # Comment column
    ws.write_string(2, col_num+4, 'Comment', title_format)# 3行目

    # 各taskで回す
    count=0
    for task_id, task_name in enumerate(task_name_list):
        for class_id, class_name in enumerate(class_name_list):
            # 列番号
            col_num = 1+4*(count)
            #print(col_num, task_id, task_name, class_id, class_name)

            # gradCamの結果画像のパス
            filelist = util.find_img_files(gradcam_img_dir+'/task'+str(task_id))#filelist = glob.glob(gradcam_img_dir+'/task'+str(task_id)+'/**/*jpg', recursive=True)
            #for i, filename in enumerate(sorted(filelist, key=lambda x: os.path.basename(x).split('_')[0])):
            filelist = sorted([f for f in filelist if 'task'+str(task_id)+'_'+str(class_id) in f])
            #print(filelist)
            for i, filename in enumerate(sorted(filelist, key=lambda x: os.path.basename(x).split('_')[0])):
                # ヘッダは2行あるので、データが入るのは行数(i)+2行目から
                # 行の高さ変更
                ws.set_row(i+3, 110)

                # エクセルに画像ID書き込み
                id_name = os.path.basename(filename).split('_')[0]
                ws.write_string(i+3, 0, id_name, text_format)

                # エクセルに画像ファイル貼り付け
                ws.insert_image(i+3, col_num, filename, {'positioning': 1, 'x_scale': x_scale, 'y_scale': y_scale})

                # 確率値
                prob = os.path.basename(filename).split('=')[1]
                prob = prob.replace('.'+img_extension, '')
                #print(prob)

                # ファイルパスから画像の正解クラスと予測クラス探索し、エクセルに書き込み
                if is_y == True:
                    posi_nega_flg = os.path.basename(os.path.dirname(filename))
                    if 'TP' == posi_nega_flg:
                        ws.write_number(i+3, col_num+1, 1, int_format)
                        ws.write_number(i+3, col_num+2, 1, int_format)
                    elif 'TN' == posi_nega_flg:
                        ws.write_number(i+3, col_num+1, 0, int_format)
                        ws.write_number(i+3, col_num+2, 0, int_format)
                    elif 'FP' == posi_nega_flg:
                        ws.write_number(i+3, col_num+1, 0, int_format)
                        ws.write_number(i+3, col_num+2, 1, int_format)
                    elif 'FN' == posi_nega_flg:
                        ws.write_number(i+3, col_num+1, 1, int_format)
                        ws.write_number(i+3, col_num+2, 0, int_format)
                    # 正解がNANの時
                    elif 'NAN' == posi_nega_flg:
                        ws.write_number(i+3, col_num+1, -1, int_format)
                        if float(prob) < pred_threshold:
                            ws.write_number(i+3, col_num+2, 0, int_format)
                        else:
                            ws.write_number(i+3, col_num+2, 1, int_format)
                else:
                    ws.write_string(i+3, col_num+1, '')
                    if float(prob) < pred_threshold:
                        ws.write_number(i+3, col_num+2, 0, int_format)
                    else:
                        ws.write_number(i+3, col_num+2, 1, int_format)
                ws.write(i+3, col_num+3, float(prob), int_format)

            count += 1

    # 1-maxrow行でオートフィルタ maxrow: 作成するエクセルの行数(=validation/test setの画像の種類+2)
    #ws.autofilter('A3:AX'+str(maxrow))

    # closeしないと保存されない
    wb.close()

def origimgs_xlsx_onesheet(out_xlsx_path, org_img_dir, gradcam_img_dir, task_name_list
                           , id_col_name='id', is_pred_col=True, is_y=True
                           , img_extension='png', x_scale=0.2, y_scale=0.2):
    """
    validation/test set のオリジナル画像をエクセルに出力する
    Args:
        out_xlsx_path: 作成するエクセルのファイル(xlsxwriter.Workbook)
        org_img_dir: オリジナルの validation/test set の画像ディレクトリ
        gradcam_img_dir: GradCamかけた validation/test set の画像ディレクトリ
        task_name_list: タスク名のリスト
        id_col_name:ID列の列名
        is_pred_col:Prediction列入れるか.FlaseならPrediction列無し
        is_y:正解ラベル列(y列)に値入れるか.Falseなら空にする
        x_scale, y_scale: エクセルに貼り付ける画像の縮小率
    Return:
        なし（wbにシート追加してエクセル出力）
    """
    print(out_xlsx_path)
    # 作成するエクセルのファイル
    wb = xlsxwriter.Workbook(out_xlsx_path)

    # 作成するエクセルシート
    ws = wb.add_worksheet('orig_img')

    # 列名用フォーマット
    title_format = wb.add_format({'bold': True, 'fg_color': '#BFBFBF'})
    # 文字用フォーマット
    text_format=wb.add_format({'bold': False, 'font_size':'11'})
    # 数字用フォーマット
    int_format=wb.add_format({'bold': False, 'font_size':'16'})

    # 作成するエクセルの列幅、列名設定
    ws.set_column(0, 0, 17.75) # ID column
    ws.write_string(1, 0, id_col_name, title_format) # A2 cell

    ws.set_column(1, 1, 20) # Structure column
    ws.write_string(1, 1, 'Structure', title_format)# 2行目
    for task_id, task_name in enumerate(task_name_list):
        # 列番号
        if is_pred_col == True:
            col_num = 1+2*task_id
        else:
            col_num = 1+task_id

        ws.set_column(col_num+1, col_num+1, 8) # Answer column
        ws.write_string(0, col_num+1, 'task'+str(task_id), title_format)# task_idを1行目にする
        ws.write_string(1, col_num+1, task_name, title_format)# task名を2行目にする
        ws.write_string(2, col_num+1, 'y', title_format)# 3行目

        if is_pred_col == True:
            # Prediction列入れるか
            ws.set_column(col_num+2, col_num+2, 12.5) # Prediction column
            ws.write_string(2, col_num+2, 'Prediction_Score', title_format)# 3行目

            ws.set_column(col_num+3, col_num+3, 64) # Comment column
            ws.write_string(2, col_num+3, 'Comment', title_format)# 3行目

    # 各taskで回す
    for task_id, task_name in enumerate(task_name_list):
        #print(task_id, task_name)
        # 列番号
        if is_pred_col == True:
            col_num = 1+2*task_id
        else:
            col_num = 1+task_id

        # gradCamの結果画像のパス
        filelist = util.find_img_files(gradcam_img_dir+'/task'+str(task_id))#filelist = glob.glob(gradcam_img_dir+'/task'+str(task_id)+'/**/*jpg', recursive=True)
        for i, filename in enumerate(sorted(filelist, key=lambda x: os.path.basename(x).split('_')[0])):
            # ヘッダは3行あるので、データが入るのは行数(i)+3行目から
            # 行の高さ変更
            ws.set_row(i+3, 110)

            # エクセルに画像ID書き込み
            id_name = os.path.basename(filename).split('_')[0]
            ws.write_string(i+3, 0, id_name, text_format)

            # オリジナルの画像ファイル貼り付け
            ws.insert_image(i+3, 1, os.path.join(org_img_dir, id_name+'_000.'+img_extension), {'positioning': 1, 'x_scale': x_scale, 'y_scale': y_scale})

            # ファイルパスから画像の正解クラスと予測クラス探索し、エクセルに書き込み
            if is_y == True:
                posi_nega_flg = os.path.basename(os.path.dirname(filename))
                if 'TP' == posi_nega_flg:
                    ws.write_number(i+3, col_num+1, 1, int_format)
                elif 'TN' == posi_nega_flg:
                    ws.write_number(i+3, col_num+1, 0, int_format)
                elif 'FP' == posi_nega_flg:
                    ws.write_number(i+3, col_num+1, 0, int_format)
                elif 'FN' == posi_nega_flg:
                    ws.write_number(i+3, col_num+1, 1, int_format)
                # 正解がNANの時
                elif 'NAN' == posi_nega_flg:
                    ws.write_number(i+3, col_num+1, -1, int_format)
            else:
                ws.write_string(i+3, col_num+1, '')

    # 1-maxrow行でオートフィルタ maxrow: 作成するエクセルの行数(=validation/test setの画像の種類+2)
    #ws.autofilter('A3:AA'+str(maxrow))

    # closeしないと保存されない
    wb.close()