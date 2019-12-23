"""
PDFファイルを、pdfminerを使ってtextファイルに変換する
参考: https://qiita.com/monchy-monchy/items/85ded85423be6108f05b

Usase:
    $ python pdf2txt.py --pdf_dir ./pdfs/ --txt_dir ./output/ # pdfsディレクトリのpdfファイルをtxtに変換して、outputディレクトリに出力する
    $ python pdf2txt.py --pdf_dir ./pdfs/ --txt_dir ./output/ --is_comma_replace # arXivのpdfはカンマ区切りでは改行してくれないので、カンマで改行してtxtに変換
"""
import os
import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import glob
import argparse
import pathlib
from tqdm import tqdm

def pdf_checker(name):
    """
    name がPDFファイル（末尾が.pdf）の場合はTRUE、それ以外はFALSEを返す。
    こちらの投稿を引用・一部変更しました　→　http://qiita.com/korkewriya/items/72de38fc506ab37b4f2d
    """
    pdf_regex = re.compile(r'.+\.pdf')
    if pdf_regex.search(str(name)):
        return True
    else:
        return False

def comma_replace(txt_path):
    """
    txtファイルをカンマ区切りで改行するように置換する
    """
    with open(txt_path, mode='r') as f:
        text = f.read()
        text = text.replace('\n',' ').replace('. ','.\n')
    with open(txt_path, mode='w') as f:
        f.write(text)

def convert_pdf_to_txt(pdf_path, txt_dir=None):
    """
    UTF-8のPDFから全テキストを抽出してtxtファイルに出力
    （鍵付きや編集権限があるPDFはエラーになる）
    Args:
        pdf_path: PDFファイルパス
        txt_dir: 出力するテキストファイルのディレクトリ
    """
    rsrcmgr = PDFResourceManager()
    if txt_dir is None:
        outfp = StringIO()
    else:
        outfp = open(txt_dir, mode='w', encoding='UTF-8', errors='replace')
    codec = 'utf-8'
    laparams = LAParams()
    laparams.detect_vertical = True # Trueにすることで綺麗にテキストを抽出できる
    device = TextConverter(rsrcmgr, outfp, codec=codec, laparams=laparams)

    fp = open(pdf_path, mode='rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
    fp.close()
    device.close()
    if txt_dir is None:
        text = outfp.getvalue()
        text = text.replace('\n',' ').replace('\r',' ')
        make_new_text_file = open(os.path.join(txt_dir, pdf_path+'.txt'), 'w')
        make_new_text_file.write(text)
        make_new_text_file.close()
    outfp.close()

def main(args):
    pdf_dir = args.pdf_dir
    txt_dir = args.txt_dir
    os.makedirs(txt_dir, exist_ok=True)
    pdfs = glob.glob(os.path.join(pdf_dir, '*.pdf'))
    for p in tqdm(pdfs):
        if pdf_checker(p): # TRUE（末尾が.pdfの場合）なら変換
            txt_path = str(pathlib.Path(txt_dir) / pathlib.Path(p).stem)+'.txt'
            try:
                convert_pdf_to_txt(p, txt_path)
                if args.is_comma_replace:
                    comma_replace(txt_path) # txtファイルをカンマ区切りで改行するように置換
            except Exception as e:
                print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", help="pdf dir path", type=str, required=True)
    parser.add_argument("--txt_dir", help="output txt dir path", type=str, required=True)
    parser.add_argument("--is_comma_replace", action='store_const', const=True, default=False, help="txtファイルをカンマ区切りで改行するように置換するかのフラグ.")
    args = parser.parse_args()
    main(args)