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
from sklearn.metrics import f1_score
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


def find_all_files(directory):
    """再帰的にファイル・ディレクトリを探して出力するgenerator"""
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)


def file_count(path, search):
    """ディレクトリ再帰的になめて<search>の文字持つファイルの数を返す"""
    files = glob.glob(os.path.join(path, "**"), recursive=True)
    newlist = []
    for l in files:
        if search in l:
            newlist.append(l)
    return len(newlist)


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


def show_file_crop_img(img_path, ymin, ymax, xmin, xmax, label=""):
    """ファイルパスから画像データを指定領域だけ表示させる"""
    image = cv2.imread(img_path)
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dst = img_RGB[ymin:ymax, xmin:xmax]
    plt.imshow(dst / 255.0)
    plt.title(str(label) + " " + str(dst.shape))
    plt.show()


def save_crop_img(img_path, ymin, ymax, xmin, xmax, out_dir=None):
    """画像ファイルを指定領域だけ切り抜き保存する"""
    img = Image.open(img_path)
    crop_img = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
    if out_dir is not None:
        save_name = (
            str(pathlib.Path(img_path).stem)
            + "_"
            + str(ymin)
            + "_"
            + str(ymax)
            + "_"
            + str(xmin)
            + "_"
            + str(xmax)
            + ".jpg"
        )
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
            plt.title(str(labels[i]) + "\n" + str(img.shape))
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
            if (
                filename.endswith(".png")
                or filename.endswith(".jpg")
                or filename.endswith(".PNG")
                or filename.endswith(".JPG")
            ):
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
    print("jpg_png_count:", len(files))
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
        print("index={}, img_path={}".format(index, img_path))
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

    kargs = {"duration": duration}  # gif画像のフレームレイト
    gif_images = []
    for filename in image_paths:
        gif_images.append(imageio.imread(filename))
    imageio.mimsave(output_gif_path, gif_images, **kargs)


def umap_tsne_scatter(
    x_array,
    y=None,
    out_png="umap_scatter.png",
    random_state=42,
    is_umap=True,
    point_size=None,
    is_axis_off=True,
    is_show=True,
    n_neighbors=15,
    min_dist=0.1,
    perplexity=30.0,
):
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
        embedding = umap.UMAP(
            random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist
        ).fit_transform(x_array)
    else:
        tsne_model = TSNE(
            n_components=2, random_state=random_state, perplexity=perplexity
        )
        embedding = tsne_model.fit_transform(x_array)

    if y is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=point_size)
    else:
        # ラベル:y指定してplot点の色変える
        plt.scatter(embedding[:, 0], embedding[:, 1], c=y, s=point_size, cmap=cm.tab10)
        plt.colorbar()
    if is_axis_off == True:
        plt.axis("off")  # x,y軸表示させない
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
            tile = np.hstack(
                images_4tensor[i * +collage_n_x : i * +collage_n_x + collage_n_x]
            )  # np.hstackは水平方向積上げ
        else:
            tile = np.vstack(
                (
                    tile,
                    np.hstack(
                        images_4tensor[
                            i * +collage_n_x : i * +collage_n_x + collage_n_x
                        ]
                    ),
                )
            )  # np.vstackは垂直方向積上げ
    show_np_img(tile)


def resize_ndarray(x, input_shape=(380, 380, 3)):
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
            rgb_im = input_im.convert("RGB")
            rgb_im.save(
                os.path.join(output_dir, pathlib.Path(file).stem + ".jpg"), quality=100
            )
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
            fps = reader.get_meta_data()["fps"]
            with imageio.get_writer(
                os.path.join(output_dir, pathlib.Path(file).stem + ".gif"), fps=fps
            ) as writer:
                for i, im in enumerate(reader):
                    sys.stdout.write("\rframe {0}".format(i))
                    sys.stdout.flush()
                    writer.append_data(im)
            print("transaction finished: " + file)


def MOV2mp4(input_video, output_dir):
    """
    MOV拡張子の動画ファイルをmp4に変換する
    http://rikoubou.hatenablog.com/entry/2019/01/15/174751
    Usage:
        MOV2mp4('IMG_9303.MOV', 'output')  # output/IMG_9303.mp4 ができる
    """
    video = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    writer = cv2.VideoWriter(
        (os.path.join(output_dir, pathlib.Path(input_video).stem + ".mp4")),
        fourcc,
        20.0,  # フレームレート
        (int(video.get(4)), int(video.get(3))),  # 画像サイズ
    )

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        ret, frame = video.read()
        frame = np.rot90(
            frame, 3
        )  # 縦動画の縦横が逆になってしまったので、２７０度回転  https://oliversi.com/2019/01/16/python-opencv-movie2/
        # print(frame.shape)
        writer.write(frame)  # 画像を1フレーム分として書き込み

    writer.release()
    video.release()
    cv2.destroyAllWindows()


def pd_targethist(df, target: str, output_dir=None, kind="hist", **kwards):
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


def plot_dendrogram(df, method="ward", figsize=(10, 6), output_dir=None):
    """
    階層型クラスタリングで樹形図（デンドログラム）と距離行列のヒートマップをplotする
    Usage:
        import seaborn as sns
        df = sns.load_dataset('iris')
        df = df.drop("species", axis=1)  # 数値列だけにしないとエラー
        df_dist = plot_dendrogram(df.T, method='ward')  # データフレーム転置しないと列についてのクラスタリングにはならない
    """
    import seaborn as sns
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram, cophenet

    # 数値列だけにしないと距離測れない
    _df = df.T
    cols = [
        col
        for col in _df.columns
        if _df[col].dtype.name in ["object", "category", "bool"]
    ]
    assert len(cols) == 0, "数値以外型の列があるので階層型クラスタリングできません"

    # 階層型クラスタリング実行
    # クラスタリング手法である methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]　ある
    z = linkage(df, method=method, metric="euclidean")

    # デンドログラムを描く
    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(z, labels=df.index, ax=ax)
    plt.title(method)
    if output_dir is not None:
        plt.savefig(
            os.path.join(output_dir, method + "_dendro.png"), bbox_inches="tight"
        )
    plt.show()

    # 距離行列計算
    s = pdist(df)
    df_dist = pd.DataFrame(
        squareform(s), index=df.index, columns=df.index
    )  # 距離行列を平方行列の形にする
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_dist, cmap="coolwarm", annot=True, ax=ax)
    plt.title("distance matrix")
    if output_dir is not None:
        plt.savefig(
            os.path.join(output_dir, method + "_distmatrix.png"), bbox_inches="tight"
        )
    plt.show()

    # クラスタリング手法の評価指標計算 値大きい方が高精度らしい https://www.haya-programming.com/entry/2019/02/11/035943
    c, d = cophenet(z, s)
    print("method:{0} {1:.3f}".format(method, c))

    return df_dist


def normalize_df_cols(df: pd.DataFrame, cols: list, normal="standard") -> pd.DataFrame:
    """
    データフレームの指定列について、正規化
    Usage:
        df = normalize_df_cols(df, ['value'])
        df = normalize_df_cols(df, ['value'], normalization='minmax')
    """
    from sklearn.preprocessing import StandardScaler  # 平均0、分散1に変換する正規化（標準化）
    from sklearn.preprocessing import MinMaxScaler  # 最小値0、最大値1にする正規化

    # 正規化を行うオブジェクトを生成
    if normal == "standard":
        scaler = StandardScaler()
    elif normal == "minmax":
        scaler = MinMaxScaler()

    # 数値列の列名だけにする
    cols = [
        col for col in cols if df[col].dtype.name not in ["object", "category", "bool"]
    ]

    for col in cols:
        # 数値型の列なら実行
        df[col] = df[col].astype(float)  # 少数点以下を扱えるようにするためfloat型に変換

    # fit_transform関数は、fit関数（正規化するための前準備の計算）とtransform関数（準備された情報から正規化の変換処理を行う）
    df_scaler = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)

    for col in cols:
        df[col] = df_scaler[col]

    return df


def delete_outlier_3sigma_df_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    データフレームの指定列について、外れ値(3σ以上のデータ)削除
    Usage:
        df = delete_outlier_3sigma_df_cols(df, ['value', 'value2'])
    """
    for col in cols:
        if df[col].dtype.name not in ["object", "category", "bool"]:
            # 数値型の列なら実行
            df = df[
                (abs(df[col] - np.mean(df[col])) / np.std(df[col]) <= 3)
            ].reset_index(drop=True)
    return df


def pca_df_cols(
    df: pd.DataFrame, cols: list, n_components=2, is_plot=True
) -> pd.DataFrame:
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

    # 数値列の列名だけにする
    cols = [
        col for col in cols if df[col].dtype.name not in ["object", "category", "bool"]
    ]

    # 主成分分析を実行. pcaに主成分分析の変換パラメータが保存され、返り値に主成分分析後の値が返される
    _ = pca.fit_transform(df[cols])

    # 累積寄与率と寄与率の確認
    print("累積寄与率: {0}".format(sum(pca.explained_variance_ratio_)))
    print("各次元の寄与率: {0}".format(pca.explained_variance_ratio_))

    # predict関数を利用し、同じ次元圧縮処理を実行
    df_pca = pd.DataFrame(pca.transform(df[cols]), columns=range(n_components))

    # pcaの散布図可視化
    if is_plot and n_components == 2:
        sns.set(style="darkgrid")
        sns.jointplot(x=df_pca.columns[0], y=df_pca.columns[1], data=df_pca, kind="reg")
        plt.show()

    return df_pca


def add_label_kmeans_pca(
    df: pd.DataFrame, n_clusters=4, normal="standard", is_pca=True
) -> pd.DataFrame:
    """
    データフレーム全体をkmeansでクラスタリングしてラベル列追加 + データフレームをPCAで次元削減して追加したラベルplot
    Usage:
        df = add_label_kmeans_pca(df)  # クラス名0,1,2,3の kmeans 列が追加される
    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler  # 平均0、分散1に変換する正規化（標準化）
    from sklearn.preprocessing import MinMaxScaler  # 最小値0、最大値1にする正規化
    from sklearn.decomposition import PCA

    # 欠損レコード削除
    df = df.replace("None", np.nan).dropna()

    # 数値列の列名だけにする
    cols = [
        col
        for col in df.columns
        if df[col].dtype.name not in ["object", "category", "bool"]
    ]

    for col in cols:
        # 数値型の列なら実行
        df[col] = df[col].astype(float)  # 少数点以下を扱えるようにするためfloat型に変換

    # 正規化
    if normal == "standard":
        scaler = StandardScaler().fit_transform(df[cols])
    elif normal == "minmax":
        scaler = MinMaxScaler().fit_transform(df[cols])
    else:
        scaler = df[cols].values

    # kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit(scaler)
    df["kmeans"] = clusters.labels_  # kmeansのラベル列追加

    # pca
    if is_pca:
        X = scaler
        pca = PCA(n_components=2)
        pca.fit(X)
        x_pca = pca.transform(X)
        pca_df = pd.DataFrame(x_pca)
        pca_df["kmeans"] = df["kmeans"]

        for i in df["kmeans"].unique():
            tmp = pca_df.loc[pca_df["kmeans"] == i]
            plt.scatter(tmp[0], tmp[1])
        plt.show()

    return df


def drop_fillna_df_cols(df: pd.DataFrame, cols: list, how="delete") -> pd.DataFrame:
    """
    データフレームの指定列について、欠損値置換
    - how='delete'なら欠損値持つレコード削除
    - how=定数ならその値で欠損値置換
    - how='mean'なら平均値で欠損値置換.列(col)の値が数値でないとエラー
    - how='knn'ならk近傍法で欠損値置換.列(col)の値が文字列などカテゴリ型でないとエラー

    そもそも欠損値補完というアプローチは、以下の矛盾を抱えている
    - 欠損値が多すぎるとそもそもまともに予測できないし、予測の悪さが大勢に影響を及ぼして全体のパフォーマンスを悪化させかねない
    - 欠損値が少ないならdropしても大勢に影響はない
    なので、そこそこの欠損があるときにdropするよりちょっと良くなるかな？という可能性を追求するためにあるものだと思います。

    Usage:
        import seaborn as sns
        df = sns.load_dataset('titanic')
        _df = drop_fillna_df_col(df, ['age'], how='delete')
        _df = drop_fillna_df_col(df, ['age'], how='mean')
        _df = drop_fillna_df_col(df, ['age'], how=0)
        _df = drop_fillna_df_col(df, ['deck'], how='knn')
    """
    from sklearn.neighbors import KNeighborsClassifier

    df = df.replace("None", np.nan)  # dropnaはNone置換できないので置き換える

    for col in cols:
        if how == "delete":
            df = df.dropna(subset=[col])

        elif how == "mean":
            if df[col].dtype.name not in ["object", "category", "bool"]:
                # 数値型の列でないので平均値とれない
                df[col] = df[col].fillna(df[col].astype("float64").mean())
            else:
                print(f"{col}列は数値型の列でないので平均値とれません")

        elif how == "knn":
            if (
                df[col].dtype.name in ["object", "category", "bool"]
                and df[col].isnull().any()
            ):
                # 数値型の列なら実行
                train = df.dropna(subset=[col])  # 欠損していないデータの抽出
                test = df.loc[df.index.difference(train.index), :]  # 欠損しているデータの抽出
                integer_cols = [
                    col
                    for col in train.columns
                    if train[col].dtype.name not in ["object", "category", "bool"]
                    and train[col].isnull().any() == False
                ]  # 数値列かつ欠損が無い列名取得。説明変数として使う列

                assert len(integer_cols) > 0, "説明変数の数値型の列がないのでknnできません"
                # print(integer_cols)

                # knnモデル生成。近傍のサンプル数はクラス数+1にしておく（適当）
                kn = KNeighborsClassifier(n_neighbors=len(train[col].unique()) + 1)
                # モデル学習
                kn.fit(train[integer_cols], train[col])
                # knnモデルによって予測値を計算し、typeを補完
                test[col] = kn.predict(test[integer_cols])

                df = pd.concat([train, test]).sort_index()
            else:
                print(f"{col}列は文字型orカテゴリ型の列でない もしくは 欠損値が無いのでknnできません")

        else:
            if df[col].dtype.name not in ["category", "bool"]:
                # 文字列か数値型の列なら実行
                df[col] = df[col].fillna(how)
            else:
                print(f"{col}列はブール型orカテゴリ型の列なので指定文字で置換できません")

    return df


def summarize_df_category_col(
    df: pd.DataFrame, col: str, new_category, summarize_categories: list
) -> pd.DataFrame:
    """
    データフレームの指定カテゴリ列について、カテゴリ値を集約する（例.「60,70,80」の3カテゴリを「60以上」だけにする）
    Usage:
        import seaborn as sns
        df = sns.load_dataset('titanic')
        df = df.replace('None', np.nan).dropna(subset=['age'])
        df['age_rank'] = np.floor(df['age']/10)*10
        df = summarize_df_category_col(df, 'age_rank', '60以上', [60.0, 70.0, 80.0])
    """
    df[col] = df[col].astype("category")  # カテゴリ型に変換

    # マスタデータにnew_categoryを追加
    df[col] = df[col].cat.add_categories(new_category)

    # 集約するデータを書き換え。category型は、=または!=の判定のみ可能なので、isin関数を利用
    df.loc[df[col].isin(summarize_categories), col] = new_category

    # 利用されていないマスタデータを削除
    df[col] = df[col].cat.remove_unused_categories()

    return df


def plot_feature_importance(model, X):
    """
    決定木やランダムフォレストで使用された特徴量を可視化する
    https://qiita.com/takapy0210/items/73415599579f2588080e
    Usage:
        # ランダムフォレスト
        from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(n_estimators=100, random_state=20181101) # n_estimatorsは構築する決定木の数
        forest.fit(X_train, y_train)
        # 表示
        plot_feature_importance(forest, X_train)
    """
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()


def save_sklearn_model_info(
    model, training_info, preprocess_pipeline=None, output_path="."
):
    """
    モデルの前処理や使うファイル、ハイパーパラメータなどの情報を保存例
    ※「このモデルはどうやって学習されたんや！？」
    　「このモデルはどんな性能なんや！？」
      「このモデルにデータ投入するための、学習済みの前処理パイプがない！！」
      みたいな事態にならないようにデータ残す
    https://qiita.com/sugulu/items/c0e8a5e6b177bfe05e99
    ※sklearnだけじゃなく、statsmodelsのモデルでもいけた
    Usage:
        test_func()の_test_save_sklearn_model_info() 参照
    """
    from datetime import datetime, timedelta, timezone
    import joblib

    JST = timezone(timedelta(hours=+9), "JST")  # 日本時刻に
    now = datetime.now(JST).strftime("%Y%m%d_%H%M%S")  # 現在時間を取得

    # 出力先がディレクトリならファイル名に現在時刻付ける
    filepath = (
        os.path.join(output_path, "sklearn_model_info_" + now + ".joblib")
        if os.path.isdir(output_path)
        else output_path
    )

    # 学習データ、モデルの種類、ハイパーパラメータの情報に現在時刻も詰める
    training_info["save_date"] = now

    save_data = {
        "preprocess_pipeline": preprocess_pipeline,
        "trained_model": model,
        "training_info": training_info,
    }

    # 保存
    joblib.dump(save_data, filepath)
    print("INFO: save file. {}".format(filepath))
    return save_data, filepath


def set_tf_random_seed(seed=0):
    """
    tensorflow v2.0の乱数固定
    https://qiita.com/Rin-P/items/acacbb6bd93d88d1ca1b
    ※tensorflow-determinism が無いとgpuについては固定できないみたい
    　tensorflow-determinism はpipでしか取れない($ pip install tensorflow-determinism)ので未確認
    """
    import random
    import numpy as np
    import tensorflow as tf

    ## ソースコード上でGPUの計算順序の固定を記述
    # from tfdeterminism import patch
    # patch()

    # 乱数のseed値の固定
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)  # v1.0系だとtf.set_random_seed(seed)


def nelder_mead_th(true_y, pred_y):
    """
    ネルダーミードでf1スコアから2値分類のbestな閾値見つける
    Usage:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import f1_score
        from scipy.optimize import minimize

        # サンプルデータ生成の準備
        rand = np.random.RandomState(seed=71)
        train_y_prob = np.linspace(0, 1.0, 10000)

        # 真の値と予測値が以下のtrain_y, train_pred_probであったとする
        train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)
        train_pred_prob = np.clip(
            train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0
        )
        # print(train_y_prob, train_pred_prob)

        # 閾値を0.5とすると、F1は0.722
        init_threshold = 0.5
        init_score = f1_score(train_y, train_pred_prob >= init_threshold)
        print(init_threshold, init_score)

        best_threshold = nelder_mead_th(train_y, train_pred_prob)
        best_score = f1_score(train_y, train_pred_prob >= best_threshold)
        print(best_threshold, best_score))
    """

    def f1_opt(x):
        return -f1_score(true_y, pred_y >= x)

    result = minimize(f1_opt, x0=np.array([0.5]), method="Nelder-Mead")
    best_threshold = result["x"].item()
    return best_threshold


def nelder_mead_func(trues, preds):
    """
    ネルダーミードで任意の関数のbestな重み見つける
    - func_opt関数の中身変更すること
    - 複数モデルの結果ブレンドするのに使える
    Usage:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import f1_score
        from scipy.optimize import minimize
        from sklearn.metrics import mean_squared_error

        train_y_prob = np.linspace(0, 1.0, 10000)
        rand = np.random.RandomState(seed=71)
        train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)
        train_pred_prob = np.clip(
            train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0
        )

        best_thresholds = nelder_mead_func(
            train_y, [train_pred_prob, train_pred_prob, train_pred_prob]
        )

        best_score = (
            (train_pred_prob * best_thresholds[0])
            + (train_pred_prob * best_thresholds[1])
            + (train_pred_prob * best_thresholds[2])
        )
        print(best_thresholds, best_score)
    """

    def func_opt(x):
        # y = a*x1 + b*x2 * c*x3 の式のa,b,cのbest重み最適化
        blend_preds = (preds[0] * x[0]) + (preds[1] * x[1]) + (preds[2] * x[2])
        # 正解との平均2乗誤差返す
        return mean_squared_error(trues, blend_preds)

    result = minimize(func_opt, x0=np.array([1.0, 1.0, 1.0]), method="Nelder-Mead")
    # print(result)
    best_thresholds = result["x"]
    return best_thresholds


# クラスタリング結果の評価指標であるPseudo Fが高いクラスタ数でkmeans計算
def pseudof_best_kmeans_df(df, max_n_clusters=10, random_state=11, is_plot=True):
    """
    クラスタリング結果の評価指標であるPseudo Fが高いクラスタ数でkmeans計算
    Pseudo Fは大きいほうが良い
    元のデータフレームにbest kmeansのラベル列追加したものを返す
    Usage:
        import warnings
        warnings.filterwarnings("ignore")
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler  # 平均0、分散1に変換する正規化（標準化）
        from sklearn.preprocessing import MinMaxScaler  # 最小値0、最大値1にする正規化

        import seaborn as sns
        iris = sns.load_dataset("iris")
        df_x = iris.drop("species", axis=1)  # 説明変数
        df_x = pd.DataFrame(StandardScaler().fit_transform(df_x))  # 標準化

        df_x_kmeans = pseudof_best_kmeans_df(df_x)
    """
    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    # 欠損レコード削除
    df = df.replace("None", np.nan).dropna()
    # 数値列の列名だけにする
    cols = [
        col
        for col in df.columns
        if df[col].dtype.name not in ["object", "category", "bool"]
    ]
    df = df[cols]
    for col in cols:
        # 数値型の列なら実行
        df[col] = df[col].astype(float)  # 少数点以下を扱えるようにするためfloat型に変換

    # 準備：結果を格納するための変数
    df_score = pd.DataFrame(columns=["k", "pseudoF"])

    # 準備：1つのクラスタに分ける
    kmeans_all = KMeans(n_clusters=1, random_state=random_state).fit(df)

    # クラスタ数kを2から10までループ
    for k in range(2, max_n_clusters + 1):
        # K-means法によってk個のクラスタに分割
        kmeans = KMeans(n_clusters=k, random_state=random_state).fit(df)

        # pseudo-Fの分子の計算
        nu = (kmeans_all.inertia_ - kmeans.inertia_) / (kmeans.n_clusters - 1)
        # pseudo-Fの分母の計算
        de = kmeans.inertia_ / (df.iloc[:, 1].count() - kmeans.n_clusters)
        # pseudo-Fの計算
        pseudoF = nu / de
        # 結果の格納
        df_score = df_score.append(
            pd.DataFrame([[float(k), pseudoF]], columns=["k", "pseudoF"])
        )
    df_score = df_score.reset_index(drop=True)

    if is_plot:
        # 結果のプロット
        print(df_score)
        df_score.plot.scatter(x="k", y="pseudoF")
        plt.show()

    # best kmeans
    best_k = df_score.iloc[df_score["pseudoF"].argmax()]["k"]
    print("INFO: Pseudo F best KMeans n_clusters {}".format(int(best_k)))
    clusters = KMeans(n_clusters=int(best_k), random_state=random_state).fit(df)
    df["group"] = clusters.labels_  # kmeansのラベル列追加

    return df


# sklearnの回帰モデルをGridSearch
def gridsearch_reg_df(
    df_x,
    df_y,
    sklearn_reg,
    tuned_parameters: list,
    test_size=0.3,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=17,
    is_plot=True,
):
    """
    sklearnの回帰モデルをGridSearch
    Usage:
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression, ElasticNet
        from sklearn.preprocessing import StandardScaler  # 平均0、分散1に変換する正規化（標準化）
        from sklearn.preprocessing import MinMaxScaler  # 最小値0、最大値1にする正規化

        import seaborn as sns
        iris = sns.load_dataset("iris")
        display(iris.head())

        # Species列を取り除く
        data_all = iris.drop("species", axis=1)

        df_x = data_all.drop("petal_length", axis=1)  # 説明変数
        df_x = pd.DataFrame(StandardScaler().fit_transform(df_x))  # 標準化

        df_y = data_all[["petal_length"]]  # 目的変数

        #sklearn_reg = LinearRegression()  # 線形回帰モデル
        #tuned_parameters = [{"n_jobs" : [1]}]  # ハイパーパラメータ 使うコア数

        sklearn_reg = ElasticNet()  # Elastic Net（l1,l2正則化加えた線形回帰モデル）
        tuned_parameters = [{"l1_ratio" : 0.1 * np.arange(0,11) + 0.1,  # l1正則化の大きさ  ElasticNetなので、l2正則化の大きさは 1 − l1_ratio になる
                             "alpha"    : pow(10, -1.0 * np.arange(0,11))  # l1_ratio と 1 − l1_ratio にかける重み。デフォルトは1.0
                            }]

        gs_reg = gridsearch_reg_df(df_x, df_y, sklearn_reg, tuned_parameters)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import model_selection
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # 教師データとテストデータの分割
    train_x, test_x, train_y, test_y = train_test_split(
        df_x, df_y, test_size=test_size, random_state=random_state
    )
    # パラメータのセット
    # CVは目安ではありますが、分割数の上限としては、擬似的な テストデータ の数が20行(レコード)を下回らないように調整するとよい
    gs_reg = model_selection.GridSearchCV(
        sklearn_reg,  # model指定
        param_grid=tuned_parameters,  # ハイパーパラメータ
        cv=cv,  # 交差検証を指定
        scoring=scoring,  # GridSearchCVのscoringでMSE(平均二乗誤差)指定する場合は'neg_mean_squared_error'
    )
    # 学習
    gs_reg.fit(train_x, train_y)

    # 係数の確認
    print("\n回帰式の各重み（係数）")
    print(list(df_x.columns))
    print(gs_reg.best_estimator_.coef_)

    print("\nIntercept（回帰式の切片）")
    print(gs_reg.best_estimator_.intercept_)

    # 損失関数のスコア
    print("\nloss")
    print(gs_reg.best_score_)

    # 評価
    # テストデータ に対する予測値を算出
    # 予測値をpred列としてtest_yに追加する
    pred = gs_reg.best_estimator_.predict(test_x)
    test_y["pred"] = pred
    print(test_y.head())

    # MAE(平均絶対誤差): モデルからデータの平均距離
    # MAEは誤差をそのまま使っているため、予想を外した点があってもRMSEほどは評価値に影響しない
    print("\nMAE(平均絶対誤差)")
    print(mean_absolute_error(test_y.iloc[:, 0], test_y["pred"]))

    # RMSE(二乗平均平方根誤差): モデルからデータの平均二乗距離の平方根
    # 予測を外す点を厳しく評価したい場合にはRMSEを使用したほうがよい
    print("\nRMSE(二乗平均平方根誤差)")
    print(np.sqrt(mean_squared_error(test_y.iloc[:, 0], test_y["pred"])))

    if is_plot:
        # 予測値の可視化
        # 正解値と予測値を描画する(予測がぴったりならば線上に乗る)
        test_y.plot.scatter(x=test_y.columns[0], y="pred", figsize=(10, 8))
        plt.plot(np.arange(8), np.arange(8))  # y=xの線
        plt.xlim(1, 7)  # x軸の範囲
        plt.ylim(1, 7)  # y軸の範囲
        plt.show()

    return gs_reg


# sklearnの分類モデルをGridSearch
def gridsearch_class_df(
    df_x,
    df_y,
    sklearn_class,
    tuned_parameters: list,
    test_size=0.3,
    cv=5,
    random_state=71,
    is_plot=True,
):
    """
    sklearnの分類モデルをGridSearch
    Usage:
        import warnings
        warnings.filterwarnings("ignore")

        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler  # 平均0、分散1に変換する正規化（標準化）
        from sklearn.preprocessing import MinMaxScaler  # 最小値0、最大値1にする正規化

        import seaborn as sns
        iris = sns.load_dataset("iris")
        display(iris.head())
        data_all = iris.copy()

        df_x = data_all.drop("species", axis=1)  # 説明変数
        df_x = pd.DataFrame(StandardScaler().fit_transform(df_x))  # 標準化（決定木ベースのモデルについては規格化必要ない）

        df_y = data_all[["species"]]  # 目的変数

        # K近傍法
        sklearn_class = KNeighborsClassifier()
        tuned_parameters = [{#"n_neighbors" : [7],  # 近傍数
                             "n_neighbors" : np.arange(1, 20, 2),  # 近傍数
                             "weights"     : ["distance"]  # データ間の距離の指標
                            }]

        # ロジスティック回帰
        sklearn_class = LogisticRegression()
        tuned_parameters = [{#"C"                 : [1.0],  # Cは正則化(penalty)の度合いのパラメータ
                             "C"                 : pow(10, 1.0 * np.arange(-5, 5)),  # Cは正則化(penalty)の度合いのパラメータ
                             "penalty"           : ["l2"],  # penaltyは正則化項。l1がLasso回帰(L1正則化)、l2ならRidge回帰(L2正則化)
                             "fit_intercept"     : [False]
                            }]

        # 決定木
        sklearn_class = DecisionTreeClassifier()
        #tuned_parameters = [{"max_depth" : [3]}]  # 木の深さ
        tuned_parameters = [{"max_depth" : np.arange(2, 10)}]  # 木の深さ

        # SVM
        sklearn_class = SVC(probability=True)
        # パラメータのカーネルは RBFカーネル で固定とする
        # コストが小さい場合には数点の誤分類は許容し、コストが大きい場合にはなるべく誤分類がないように分類していきます。
        # 複雑さが小さい場合には簡単な境界線(直線)になり、複雑さが大きい場合には複雑な境界面になります。
        tuned_parameters = [{#"C"     : [1],  # C:誤分類に対するコスト(ペナルティ)
                             "C"     : pow(10 , 1.0 * np.arange(-4, 6, 2) ),
                             #"gamma" : [0.01]  # gamma:分割面の複雑さを表す(大きいほど複雑)
                             "gamma" : pow(10 , 1.0 * np.arange(-4, 6, 2))
                            }]

        gs_clf = gridsearch_class_df(df_x, df_y, sklearn_class, tuned_parameters)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import model_selection, metrics, tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve, auc
    import pydotplus

    # 教師データとテストデータの分割
    train_x, test_x, train_y, test_y = train_test_split(
        df_x, df_y, test_size=test_size, random_state=random_state
    )
    # パラメータのセット
    # CVは目安ではありますが、分割数の上限としては、擬似的な テストデータ の数が20行(レコード)を下回らないように調整するとよい
    gs_clf = model_selection.GridSearchCV(
        sklearn_class,  # model指定
        param_grid=tuned_parameters,  # ハイパーパラメータ
        cv=cv,  # 交差検証を指定
    )
    # 学習
    gs_clf.fit(train_x, train_y)

    # スコア(正確度)
    print("\nスコア")
    print(gs_clf.best_score_)

    # 評価
    # テストデータ に対する予測値を算出
    # 予測値をpred列, 各クラスの予測確率をpred_proba_列としてtest_yに追加する
    pred = gs_clf.best_estimator_.predict(test_x)
    test_y["pred"] = pred
    for i, class_name in enumerate(gs_clf.classes_):
        test_y[f"pred_proba_{class_name}"] = gs_clf.best_estimator_.predict_proba(
            test_x
        )[:, i]
    print(test_y.head())

    # 混同行列
    cm = metrics.confusion_matrix(test_y.iloc[:, 0], test_y["pred"])
    print("\n混同行列")
    print(cm)

    # 正確率
    print("\naccuracy:")
    print(cm.diagonal().sum() / cm.sum())

    # AUC
    print("\nAUC")
    for i, class_name in enumerate(gs_clf.classes_):
        precision, recall, thresholds = precision_recall_curve(
            test_y.iloc[:, 0], test_y[f"pred_proba_{class_name}"], pos_label=class_name
        )
        print(class_name + ":", auc(recall, precision))

    # best param
    print(f"\n{sklearn_class.__class__.__name__} best param")
    print(gs_clf.best_params_)

    if sklearn_class.__class__.__name__ == "LogisticRegression":
        # ロジスティクス回帰について
        print("\nオッズ比の確認")
        print(list(df_x.columns))
        print(np.exp(gs_clf.best_estimator_.coef_))

    if sklearn_class.__class__.__name__ == "DecisionTreeClassifier":
        # 決定木についてGraphviz描画用ファイルの出力
        tree.export_graphviz(
            gs_clf.best_estimator_,  # model
            out_file="dtree.dot",  # ファイル名
            feature_names=train_x.columns,  # 特徴量名
            class_names=train_y.iloc[:, 0].unique(),  # クラス名
            filled=True,  # 色を塗る
        )
        # dotファイルをpngに変換
        graph = pydotplus.graphviz.graph_from_dot_file("dtree.dot")
        graph.write_png("dtree.png")
        ## jupyterで決定木の画像表示
        # from IPython.display import Image, display_png
        # display_png(Image(graph.create_png()))

    return gs_clf


def test_func():
    """
    テスト駆動開発でのテスト関数
    test用関数はpythonパッケージの nose で実行するのがおすすめ($ conda install -c conda-forge nose などでインストール必要)
    →noseは再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行する
    $ cd <このモジュールの場所>
    $ nosetests -v -s util.py  # 再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行。-s付けるとprint()の内容出してくれる
    """
    import matplotlib

    matplotlib.use("Agg")  # グラフ表示させない
    import seaborn as sns

    # assertでテストケースチェックしていく. Trueなら何もしない

    # plot_dendrogram()
    df = sns.load_dataset("iris")
    df = df.drop("species", axis=1)
    df_dist = plot_dendrogram(
        df.T,
        method="ward",
        output_dir=r"D:\work\02_keras_py\experiment\01_code_test\output_test\tmp",
    )
    assert df_dist.shape[0] == 4

    # MOV2mp4()
    MOV2mp4(
        r"D:\iPhone_pictures\2019-04\IMG_9303.MOV",
        r"D:\work\02_keras_py\experiment\01_code_test\output_test\tmp",
    )
    assert (
        os.path.exists(
            r"D:\work\02_keras_py\experiment\01_code_test\output_test\tmp\IMG_9303.mp4"
        )
        == True
    )

    # normalize_df_cols
    df_titanic = sns.load_dataset("titanic")
    assert int(normalize_df_cols(df_titanic, df_titanic.columns)["age"].mean()) == 0
    assert (
        int(
            normalize_df_cols(df_titanic, df_titanic.columns, normal="minmax")[
                "age"
            ].max()
        )
        == 1
    )

    # delete_outlier_3sigma_df_cols()
    df_iris = sns.load_dataset("iris")
    assert (
        df_iris.shape[0]
        > delete_outlier_3sigma_df_cols(df_iris, df_iris.columns).shape[0]
    )

    # pca_df_cols()
    df_iris = sns.load_dataset("iris")
    assert pca_df_cols(df_iris, df_iris.columns, is_plot=False).shape[1] == 2

    # drop_fillna_df_col
    df_titanic = sns.load_dataset("titanic")
    assert (
        drop_fillna_df_cols(df_titanic, df_titanic.columns, how="delete")["age"]
        .isnull()
        .all()
        == False
    )
    assert (
        drop_fillna_df_cols(df_titanic, df_titanic.columns, how="mean")["age"]
        .isnull()
        .all()
        == False
    )
    assert (
        drop_fillna_df_cols(df_titanic, df_titanic.columns, how=0)["age"].dtype.name
        == "float64"
    )
    assert (
        drop_fillna_df_cols(df_titanic, df_titanic.columns, how="knn")["deck"]
        .isnull()
        .all()
        == False
    )

    # summarize_df_category_col()
    df_titanic = sns.load_dataset("titanic")
    df_titanic = df_titanic.replace("None", np.nan).dropna(subset=["age"])
    df_titanic["age_rank"] = np.floor(df_titanic["age"] / 10) * 10
    assert (
        summarize_df_category_col(df_titanic, "age_rank", "60以上", [60.0, 70.0, 80.0])[
            "age_rank"
        ].value_counts()["60以上"]
        > 0
    )

    # set_tf_random_seed()
    set_tf_random_seed(seed=71)

    # add_label_kmeans_pca()
    df_titanic = sns.load_dataset("titanic")
    assert len(add_label_kmeans_pca(df_titanic, is_pca=False)["kmeans"].unique()) == 4
    assert (
        len(
            add_label_kmeans_pca(df_titanic, normal="", is_pca=False)["kmeans"].unique()
        )
        == 4
    )

    # save_sklearn_model_info()
    def _test_save_sklearn_model_info():
        import joblib
        import sklearn.model_selection
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import RandomForestClassifier

        # アヤメのデータ
        df_iris = sns.load_dataset("iris")
        X = df_iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        y = LabelEncoder().fit_transform(df_iris["species"])
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y
        )

        def _train(_X_train, _y_train):
            # X_train 前処理
            preprocess_pipeline = Pipeline(
                steps=[("standard_scaler", StandardScaler())]
            )
            X_preprocessed = preprocess_pipeline.fit_transform(_X_train)

            # sklearn_model
            model = RandomForestClassifier(random_state=0)
            model.fit(X_preprocessed, _y_train)

            # データ保存
            training_info = {
                "training_data": "iris",
                "model_type": "RandomForestClassifier",
                "hyper_pram": "default",
            }
            model_info, model_info_filepath = save_sklearn_model_info(
                model,
                training_info,
                preprocess_pipeline=preprocess_pipeline,
                output_path=r"D:\work\02_keras_py\experiment\01_code_test\output_test\sklearn_model\iris_rf.joblib",
            )
            return model_info, model_info_filepath

        def _predict(model_info, _X_test):
            # X_test 前処理
            preprocess_pipeline = model_info["preprocess_pipeline"]
            X_preprocessed = (
                _X_test
                if preprocess_pipeline is None
                else preprocess_pipeline.fit_transform(_X_test)
            )

            # モデルロード
            model = model_info["trained_model"]
            pred = model.predict(X_preprocessed)
            print("pred:", pred)

        # モデル作成
        model_info, model_info_filepath = _train(X_train, y_train)

        # 5件テスト
        _predict(model_info, X_test[0:5])

        # ファイルからデータロードして、5件テスト
        load_data = joblib.load(model_info_filepath)
        print(load_data)
        _predict(load_data, X_test[0:5])
        return

    _test_save_sklearn_model_info()
