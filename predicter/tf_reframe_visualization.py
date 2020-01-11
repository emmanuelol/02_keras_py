# -*- coding: utf-8 -*-
"""
GoogleがPerfumeのライブに技術提供した
Reframe Visualization：ミュージックビデオをキャプチャし画像を似ている同士でタイル化するもの
をKeras/Scikit-learnで再現する
https://qiita.com/koshian2/items/89ada5223c18e930f801

やってることは
x枚の画像をDeepLearningモデルに順伝搬して、[1,n]次元の埋め込みベクトルx個取得
→x個の[1,n]次元の埋め込みベクトルをumapやtsneで次元削減し、x個の[n,2]の2次元の位置情報array取得（umapやtsneにより似てる画像は近くに位置する）
→x個の[n,2]の2次元の位置情報arrayのindexをk近傍法で行列に変換してタイル状に並べ替えれるようにし、x枚の画像タイル状にして画像化
"""
# test notebook:
# C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\300_Keras_iPhone_pictures_work
# reframe_visualization.ipynb

import os, sys
import numpy as np
from tqdm import tqdm
import argparse
import pathlib
from PIL import Image, ImageDraw, ImageFile
# PILは極端に大きな画像など高速にロードできない
# でかい画像は見過ごしてエラー(OSError: image file is truncated)になるのを避けるには以下の文が必要らしい
# http://snowman-88888.hatenablog.com/entry/2016/03/08/115918
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib
matplotlib.use('Agg')

from tensorflow import keras

# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得
sys.path.append( str(current_dir) + '/../' )
from dataset import util

def get_features(model, layer_id, X):
    """ model順伝搬してlayer_idのfeatures取得 """
    layer = model.layers[layer_id]
    # 中間層の特徴マップを返す関数を作成
    get_feature_map = keras.backend.function(inputs=[model.input, keras.backend.learning_phase()], outputs=[layer.output])
    # 順伝搬して畳み込み層の特徴マップを取得する。
    features, = get_feature_map([X, False])
    return features

def load_one_img(img_file_path, img_rows=331, img_cols=331):
    """画像を1枚読み込んで、4次元テンソル(1, img_rows, img_cols, 3)へ変換+前処理"""
    img = keras.preprocessing.image.load_img(img_file_path, target_size=(img_rows, img_cols))# 画像ロード
    x = keras.preprocessing.image.img_to_array(img)# ロードした画像をarray型に変換
    x = np.expand_dims(x, axis=0)# 4次元テンソルへ変換
    x = x.astype('float32')
    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！ これを忘れると結果がおかしくなるので注意
    X = x / 255.0# 前処理
    return X

def find_nearest_neighbor(data, w_plot=60.0, h_plot=30.0):
    """
    k=1のk近傍法で敷き詰める画像のindex行列を作成する
    Args:
        data: umapやtsneで次元削減した[n,2]の位置情報array
        w_plot: index行列の横の要素数
        h_plot: index行列の縦の要素数
        ※w_plot*h_plot = data.shape[0]でないと全画像描画できない
        ※デフォルトだと横に60枚、縦に30枚画像を敷き詰める
    """
    #  umapやtsneで次元削減したデータの最小位置/最大位置取得
    min_x = min(data[:, 0])
    max_x = max(data[:, 0])
    min_y = min(data[:, 1])
    max_y = max(data[:, 1])
    #print(min_x, max_x, min_y, max_y)

    # k近傍法使うためにマージン距離の単位ベクトル的なの用意
    pitch_x = (max_x - min_x) / float(h_plot)
    pitch_y = (max_y - min_y) / float(w_plot)
    #print(pitch_x, pitch_y)

    # プロットする画像のindexの行列初期化
    index_mat = np.zeros((int(h_plot), int(w_plot)), dtype=np.int32)
    #print(index_mat)

    for i in tqdm(range(int(h_plot))):
        for j in tqdm(range(int(w_plot))):
            pos_x = min_x + pitch_x * i
            pos_y = min_y + pitch_y * j
            positions = np.array([pos_x, pos_y]).reshape(1, -1)
            #print(positions)

            # 各点との距離計算
            dist = np.sum((data - positions) ** 2, axis=-1)
            #print(dist)
            #index = np.argmin(dist)

            # 同じ画像の繰り返し避けるため採用したindex以外のものを探す
            #print(np.argsort(dist))
            for sort_i in np.argsort(dist):
                if np.any(index_mat == sort_i) == False:
                    index = sort_i
                    break
            #print(index)
            index_mat[i, j] = index
    return index_mat

def paste_images(file_path, is_upper_triangle, target_image, position):
    """ 指定positionに画像貼り付ける """
    with Image.new("L", (128, 128)) as mask:
        draw = ImageDraw.Draw(mask)
        if is_upper_triangle:
            draw.polygon([(64, 0), (0, 128), (128, 128)], fill=255)
        else:
            draw.polygon([(0, 0), (128, 0), (64, 128)], fill=255)
        with Image.open(file_path) as original:
            width, height = original.size
            left = (width - height) // 2
            right = left + height

            resized = original.crop((left, 0, right, height)).resize((128,128), Image.BILINEAR).convert("RGBA")
            resized.putalpha(mask)
            target_image.paste(resized, position, resized)

def merge_images(img_paths, index_mat, out_png, h_plot=60, w_plot=30):
    """ umapやtsneで次元削減した[n,2]の位置情報array と k=1のk近傍法で敷き詰める画像のindex行列 から近い画像をパネル状に集めて1枚の画像にする """
    metadata = img_paths
    neighbors = index_mat
    with Image.new("RGBA", (64*h_plot, 128*w_plot)) as canvas:
        for i in tqdm(range(neighbors.shape[0])):
            for j in tqdm(range(neighbors.shape[1])):
                print(metadata[neighbors[i, j]])
                paste_images(metadata[neighbors[i, j]], (i+j)%2==0, canvas, (-64+64*i, 128*j))
        canvas.save(out_png)

def main():
    # iPhoneの画像モデルをデフォルト引数にしておく
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default=r'D:\work\keras_iPhone_pictures\01_classes_results_tfgpu_py36\20190529\train_all\tsne_umap',
                    help="output dir path.")
    ap.add_argument("-i", "--input_dir", type=str, required=True,
                    help="input dir path.")
    ap.add_argument("-m", "--model_path", type=str, default=r'D:\work\keras_iPhone_pictures\01_classes_results_tfgpu_py36\20190529\train_all\finetuning.h5',
                    help="model path.")
    ap.add_argument("-li", "--layer_id", type=int, default=780,
                    help="model features layer id.")
    args = vars(ap.parse_args())

    # 画像パス取得
    img_paths = util.find_img_files(args['input_dir'])
    #print('len(img_paths):', len(img_paths))

    # モデル順伝搬して、埋め込みベクトル取得
    features_list = []
    pop_index = []
    model = keras.models.load_model(args['model_path'], compile=False)
    for i,p in tqdm(enumerate(img_paths)):
        try:
            X = load_one_img(p, model.input_shape[1], model.input_shape[2])
            features = get_features(model, args['layer_id'], X)
            features_list.append(features[0])
        except Exception as e:
            print('load error:', p)
            pop_index.append(i) # ロードできなかった画像のindex
            print(e)
    # ロードできなかった画像削除
    if len(pop_index) > 0:
        for i in pop_index:
            img_paths.pop(i)

    # umap実行
    embedding = util.umap_tsne_scatter(np.array(features_list)
                                       , out_png=args['output_dir']+'/'+str(pathlib.Path(args['input_dir']).stem)+'_umap_scatter.png'
                                       , n_neighbors=5
                                       #, is_umap=False, perplexity=10.0
                                       , is_axis_off=False
                                       , is_show=False
                                       )

    # reframe_visualization
    w_plot = int(np.sqrt(embedding.shape[0]/2.0))
    h_plot = embedding.shape[0]//w_plot
    index_mat = find_nearest_neighbor(embedding, w_plot=w_plot, h_plot=h_plot)
    merge_images(img_paths, index_mat
                 , out_png=args['output_dir']+'/'+str(pathlib.Path(args['input_dir']).stem)+'_reframe_visualization.png'
                 , w_plot=w_plot, h_plot=h_plot)

if __name__ == "__main__":
    main()
