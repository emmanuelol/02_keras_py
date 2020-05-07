# -*- coding: utf-8 -*-
"""
util関数群
"""
import os
import glob
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2


def find_all_files(directory):
    """再帰的にファイル・ディレクトリを探して出力するgenerator"""
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)


def file_count(path, search):
    """ディレクトリ再帰的になめて<search>の文字持つファイルの数を返す"""
    files = glob.glob(os.path.join(path, '**'), recursive=True)
    newlist = []
    for l in files:
        if search in l:
            newlist.append(l)
    return(len(newlist))


def show_np_img(x, is_grayscale=False):
    """
    numpy配列の画像データを表示させる
    matplotlibはgrayscaleにした画像は白色が黄色になるのでgrayscale化した場合はis_grayscale=Trueにすること
    """
    plt.imshow(x)
    if is_grayscale == True:
        plt.gray()
    plt.show()


def show_file_img(img_path):
    """ファイルパスから画像データを表示させる"""
    # 画像の読み込み
    im = Image.open(img_path)
    # 画像をarrayに変換
    im_list = np.asarray(im)
    # 貼り付け
    plt.imshow(im_list)
    # 表示
    plt.show()


def show_file_crop_img(img_path, ymin, ymax, xmin, xmax, label=''):
    """ファイルパスから画像データを指定領域だけ表示させる"""
    image = cv2.imread(img_path)
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dst = img_RGB[ymin:ymax, xmin:xmax]
    plt.imshow(dst / 255.)
    plt.title(str(label) + ' ' + str(dst.shape))
    plt.show()


def save_crop_img(img_path, ymin, ymax, xmin, xmax, out_dir=None):
    """画像ファイルを指定領域だけ切り抜き保存する"""
    img = Image.open(img_path)
    crop_img = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
    if out_dir is not None:
        save_name = str(pathlib.Path(img_path).stem) + '_' + str(ymin) + '_' + str(ymax) + '_' + str(xmin) + '_' + str(xmax) + ".jpg"
        crop_img.save(os.path.join(out_dir, save_name))
    return crop_img


def plot_5imgs(img_list, plot_num=10, figsize=(10, 8), labels=None, is_gray=False):
    """
    5枚ずつ並べて画像表示。np.ndarrayの画像データでも表示可能
    Usage:
        img_list = glob.glob(data_dir+'/*/*.png')
        plot_5imgs(img_list)
    """
    if isinstance(img_list, np.ndarray):
        print("Num_Images: ", img_list.shape[0])
    else:
        print("Num_Images: ", len(img_list))
    plt.figure(figsize=figsize)
    # 並べる画像の行数
    rows = plot_num // 5
    if rows < plot_num / 5.0:
        rows += 1
    # print(f"rows: {rows}")
    for i in range(plot_num):
        if isinstance(img_list, np.ndarray):
            img = img_list[i]
        else:
            img = plt.imread(img_list[i])
            # 画像2次元ならグレースケール
            if len(img.shape) == 2:
                is_gray = True
            if is_gray == True:
                img = plt.imread(img_list[i], 0)  # 第二引数0でグレースケール画像として読み込む.
        plt.subplot(rows, 5, i + 1)
        if is_gray == True:
            plt.imshow(img, cmap="gray")  # グレースケールで表示.
        else:
            plt.imshow(img)
        if labels is not None:
            plt.title(str(labels[i]) + '\n' + str(img.shape))
        else:
            plt.title(img.shape)
        plt.xticks([])
        plt.yticks([])
        # 全件plotしたらbreak
        if len(img_list) == i + 1:
            break
    plt.tight_layout()
    plt.show()
    plt.clf()


def find_img_files(path):
    """
    ファイルパス再帰的に探索し、pngかjpgのパスだけリストで返す
    https://qiita.com/hasepy/items/8e6a0757da1ce074ce87
    """
    import os
    imagePaths = []
    for pathname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            # フィルタ処理
            if filename.endswith('.png') or filename. endswith('.jpg') or filename.endswith('.PNG') or filename.endswith('.JPG'):
                imagePaths.append(os.path.join(pathname, filename))
    return imagePaths


def get_jpg_png_path_in_dir(dir):
    """
    pathlibを使って指定ディレクトリを再帰的になめ、jpg,pngのファイルパスをリストで返す
    """
    jpg_files = []
    for p in list(pathlib.Path(dir).glob("**/*jpg")):
        # Pathオブジェクトを通常の文字列に変換
        jpg_files.append(p.as_posix())
    png_files = []
    for p in list(pathlib.Path(dir).glob("**/*png")):
        # Pathオブジェクトを通常の文字列に変換
        png_files.append(p.as_posix())
    files = []
    files.extend(jpg_files)
    files.extend(png_files)
    print('jpg_png_count:', len(files))
    return sorted(files)


def resize_np_Nearest_Neighbor(img_np, n_resize):
    """
    numpy型の画像データ1つの縦横サイズをn_resize倍にする
    （CIFAR10そのままのサイズ(32,32,3)ではkerasの学習済みモデルつかえないので）
    https://blog.shikoan.com/numpy-upsampling-image/
    """
    return img_np.repeat(n_resize, axis=0).repeat(n_resize, axis=1)


def ipywidgets_show_img(img_path_list, figsize=(9, 9), is_grayscale=False):
    """
    ipywidgetsでインタラクティブに画像表示
    https://github.com/pfnet-research/chainer-chemistry/blob/master/examples/tox21/tox21_dataset_exploration.ipynb
    """
    from ipywidgets import interact
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    def view_image(index):
        img_path = img_path_list[index]
        print('index={}, img_path={}'.format(index, img_path))
        img = Image.open(img_path)
        img_array = np.asarray(img)
        plt.figure(figsize=figsize)  # 表示サイズ指定
        plt.imshow(img_array)
        if is_grayscale == True:
            plt.gray()
        plt.show()
    interact(view_image, index=(0, len(img_path_list) - 1))


def creat_gif_from_images(output_gif_path, image_paths, duration=0.5):
    """
    imageio使って画像ファイルのリストからgif画像を作成
    https://codeday.me/jp/qa/20190323/466095.html
    Args:
        output_gif_path:出力するgif画像のパス
        image_paths:画像ファイルパスのリスト
        duration:gif画像の各フレームの表示秒。あまりにも小さい値だと反映されない
    Usage:
        # jupyterなら以下の文でgif画像表示できる
        from IPython.display import Image
        with open("images/movie.gif",'rb') as f:
            display(Image(data=f.read(), format='png'))
    """
    import imageio
    kargs = {'duration': duration}  # gif画像のフレームレイト
    gif_images = []
    for filename in image_paths:
        gif_images.append(imageio.imread(filename))
    imageio.mimsave(output_gif_path, gif_images, **kargs)


def umap_tsne_scatter(x_array, y=None, out_png='umap_scatter.png', random_state=42,
                      is_umap=True, point_size=None, is_axis_off=True, is_show=True,
                      n_neighbors=15, min_dist=0.1, perplexity=30.0):
    """
    umap/tsneで次元削減した画像出力
    Args:
        x_array: np.array型のn次元の特徴量
        y: 特徴量のラベル
        out_png: umap/tsneの出力画像のパス
        random_state: 乱数シード
        is_umap: Trueならumapで次元削減。Falseならtsneで次元削減
        point_size: plot点の大きさ
        is_axis_off: Trueなら画像のx,y軸表示させない
        is_show: Trueなら次元削減した画像plt.show()しない
        n_neighbors: umapのパラメータ。大きくするとマクロな、小さくするとミクロな構造を振る舞いを反映させる。t-SNEでいうところのperplexityのような値だと思われる。2~100が推奨されている。
        min_dist: umapのパラメータ。同一クラスタとして判定する距離。値大きいとplot点広がる。0.0~0.99。
        perplexity: tsneのパラメータ。投射した点の集まり方の密さを表すもの。この値は5～50が理論値で、低いほうが点が密になりやすく、高いとゆるいプロットになり
    Usage:
        %matplotlib inline
        from sklearn.datasets import load_digits
        digits = load_digits()
        # MNIST1画像
        x = digits.data[0].reshape(digits.data[0].shape[0], 1)
        print(x.shape) # (64, 1)
        umap_tsne_scatter(x)
        # MNIST全画像
        x = digits.data
        print(x.shape) # (1797, 64)
        util.umap_tsne_scatter(x, y=digits.target, out_png='output_test/umap_scatter.png')
    """
    import umap
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if is_umap == True:
        embedding = umap.UMAP(random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(x_array)
    else:
        tsne_model = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        embedding = tsne_model.fit_transform(x_array)

    if y is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=point_size)
    else:
        # ラベル:y指定してplot点の色変える
        plt.scatter(embedding[:, 0], embedding[:, 1], c=y, s=point_size, cmap=cm.tab10)
        plt.colorbar()
    if is_axis_off == True:
        plt.axis('off')  # x,y軸表示させない
    if out_png is not None:
        plt.savefig(out_png)
    if is_show == True:
        plt.show()
    plt.clf()  # plotの設定クリアにする
    return embedding


def show_tile_img(images_4tensor):
    """
    4次元ベクトル[画像枚数, y_shape, x_shape, チャネル数]を1枚ずつ積み上げて、タイル状の画像にする
    https://qiita.com/ka10ryu1/items/015c6a6a5fa287a47828
    Args:
        images_4tensor: 4次元ベクトル[画像枚数, y_shape, x_shape, チャネル数]
    """
    tile = None
    collage_n_x = int(np.sqrt(images_4tensor.shape[0]))
    for i in range(images_4tensor.shape[0] // collage_n_x):
        if tile is None:
            tile = np.hstack(images_4tensor[i*+collage_n_x: i*+collage_n_x + collage_n_x])  # np.hstackは水平方向積上げ
        else:
            tile = np.vstack((tile, np.hstack(images_4tensor[i*+collage_n_x: i*+collage_n_x + collage_n_x])))  # np.vstackは垂直方向積上げ
    show_np_img(tile)


def resize_ndarray(x, input_shape=(380,380,3)):
    """
    tensorflow.kerasでndarray型の画像を指定サイズにリサイズする
    http://pynote.hatenablog.com/entry/keras-image-utils
    http://pynote.hatenablog.com/entry/pillow-resize
    """
    import tensorflow.keras
    # ndarray から PIL 形式に変換する
    img = tensorflow.keras.preprocessing.image.array_to_img(x)
    # 指定した大きさにリサイズする
    img = img.resize((input_shape[0], input_shape[1]), resample=0)
    # PIL 形式から ndarray に変換する
    x = tensorflow.keras.preprocessing.image.img_to_array(img)
    return x


def img2jpg(input_dir, output_dir):
    """
    PILで画像ファイルをjpg拡張子に変換する
    https://qiita.com/hirohuntexp/items/05b7a81323dff7bdca9f
    """
    from PIL import Image
    import os
    import pathlib
    from tqdm import tqdm

    files = os.listdir(input_dir)
    print(files)

    for file in tqdm(files):
        if file[-4:].lower() in [".png", ".tif", ".jpg", ".bmp"]:
            input_im = Image.open(os.path.join(input_dir, file))
            rgb_im = input_im.convert('RGB')
            rgb_im.save(os.path.join(output_dir, pathlib.Path(file).stem + ".jpg"), quality=100)
            print("transaction finished: " + file)


def movie2gif(input_dir, output_dir):
    """
    MP4/AVIファイルをGIFに変換。1MB以下のMP4でもgifにするとファイルサイズが200MBぐらいになるので注意
    https://gist.github.com/michaelosthege/cd3e0c3c556b70a79deba6855deb2cc8
    """
    import imageio
    import os
    import sys
    import pathlib
    from tqdm import tqdm

    files = os.listdir(input_dir)
    print(files)

    for file in tqdm(files):
        if file[-4:].lower() in [".mp4", ".avi"]:
            reader = imageio.get_reader(os.path.join(input_dir, file))
            fps = reader.get_meta_data()['fps']
            with imageio.get_writer(os.path.join(output_dir, pathlib.Path(file).stem + ".gif"), fps=fps) as writer:
                for i, im in enumerate(reader):
                    sys.stdout.write("\rframe {0}".format(i))
                    sys.stdout.flush()
                    writer.append_data(im)
            print("transaction finished: " + file)


def pd_targethist(df, target: str, output_dir=None, kind='hist', **kwards):
    """
    Pandasのpivot_table使ってカテゴリデータ（ラベル,目的変数やhueと呼ぶほうがしっくりくるかもしれません）
    ごとに色分けしたヒストグラムを数値列(説明変数)ごとに作成する
    →'posi','nega'だけみたいなラベル列が1列でそれ以外は数値列（説明変数）のデータフレームで各ラベルごと（posi,negaの2種）の分布をplotできる
    https://own-search-and-study.xyz/2018/02/27/pandas%e3%81%a7%e7%9b%ae%e7%9a%84%e5%a4%89%e6%95%b0%e5%88%a5%e3%81%ab%e8%89%b2%e5%88%86%e3%81%91%e3%81%97%e3%81%9f%e3%83%92%e3%82%b9%e3%83%88%e3%82%b0%e3%83%a9%e3%83%a0%e3%82%92%e4%bd%9c%e6%88%90/
    Args:
        df:データフレーム
        target:dfのカテゴリデータの列名。dfはtarget列以外は数値でないとエラーになる（ヒストグラムなので当たり前だが）
        output_dir:出力ディレクトリ。グラフをファイル出力する場合はパス指定する
        kind:df.plot()の引数kind。数値の分布を見るものなら指定可能。e.g: 'hist', 'box', 'kde'or'density'(カーネル密度推定:確率密度関数を推定)
        **kwards:df.plot()の他の引数指定用。e.g: figsize=(20, 10), logy=True(y軸を対数にする), subplots=True(subplotにする), layout=(2, 2)(subplotで縦横2枚づつ並べる),
    Usage:
        import seaborn as sns
        df = sns.load_dataset('iris')  # irisデータ
        pd_targethist(df, 'species')  # ヒストグラム
        pd_targethist(df, 'species', kind='box')  # 箱ひげグラフ
        pd_targethist(df, 'species', kind='density')  # カーネル密度推定
    """
    columns = df.columns[df.columns != target]
    pdf = df.pivot_table(index=df.index, columns=target)
    for column in columns:
        ax = pdf.loc[:, column].plot(kind=kind, title=column, **kwards)
        if output_dir is not None:
            ax.get_figure().savefig(os.path.join(output_dir, column + ".png"))


def delete_outlier_3sigma(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    データフレームの指定列について、外れ値(3σ以上のデータ)削除
    Usage:
        df = delete_outlier_3sigma(df, 'value')
    """
    return df[(abs(df[col] - np.mean(df[col])) / np.std(df[col]) <= 3)].reset_index()


def pca_df_cols(df: pd.DataFrame, cols: list, n_components=2, is_plot=True) -> pd.DataFrame:
    """
    データフレームの指定列について、PCAで次元削減
    Usage:
        import seaborn as sns
        df = sns.load_dataset('iris')
        pca_df_cols(df, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    """
    import seaborn as sns
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)  # n_componentsに、主成分分析で変換後の次元数を設定

    # 主成分分析を実行. pcaに主成分分析の変換パラメータが保存され、返り値に主成分分析後の値が返される
    _ = pca.fit_transform(df[cols])

    # 累積寄与率と寄与率の確認
    print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))
    print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))

    # predict関数を利用し、同じ次元圧縮処理を実行
    df_pca = pd.DataFrame(pca.transform(df[cols]), columns=range(n_components))

    # pcaの散布図可視化
    if is_plot and n_components == 2:
        sns.set(style="darkgrid")
        sns.jointplot(x=df_pca.columns[0], y=df_pca.columns[1], data=df_pca, kind='reg')
        plt.show()

    return df_pca


def test_func():
    """
    テスト駆動開発での関数のテスト関数
    test用関数はpythonパッケージの nose で実行するのがおすすめ($ conda install -c conda-forge nose などでインストール必要)
    →noseは再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行する
    $ cd <このモジュールの場所>
    $ nosetests -v util.py  # 再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行
    """
    import seaborn as sns
    df = sns.load_dataset('iris')
    assert df.shape[0] > delete_outlier_3sigma(df, 'sepal_width').shape[0]  # assertでテストケースチェックしていく. Trueなら何もしない
    assert pca_df_cols(df, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], is_plot=False).shape[1] == 2
