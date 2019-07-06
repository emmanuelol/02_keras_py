"""
画像のヒストグラム可視化コード
以下のサイトのコード参考に作成
http://optie.hatenablog.com/entry/2018/02/26/183825
http://optie.hatenablog.com/entry/2018/03/03/141427

bioinfoユーザのtfgpu_py36環境で実行確認した（20190313）
"""
import os, sys
import PIL.Image
import numpy as np
import scipy.stats as sstats
from scipy.misc import bytescale
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
import cv2

OUT_DIR = "tmp/"

def convert_uint8(img_uint16_path):
    """
    16bit grayscale image を OpenCVとscipy で 8bit grayscale image に変換する
    （HCSのtiff画像は16bit grayscale。変換せずにロードするとピクセル値が0-255ではなく、0-5000ぐらいになっておりjpg画像と比較できない）
    https://stackoverflow.com/questions/25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv
    """
    img = cv2.imread(img_uint16_path, flags = -1)
    return bytescale(img)

def grayhist(img, stats=False, out_jpg='gray.jpg', output_dir=OUT_DIR, is_conv_gray=True):
    """
    入力 : BGR画像, 統計量の表示の有無(Optional)
    出力 : グレースケールのimshow, Histgram(+統計量)
    http://optie.hatenablog.com/entry/2018/02/26/183825
    """
    # スタイルの設定。seabornの設定は残るため、常に最初に書いておく
    sns.set()
    sns.set_style(style='ticks')

    # プロット全体と個々のグラフのインスタンス
    fig = plt.figure(figsize=[15,4])
    ax1 = fig.add_subplot(1,2,1)

    sns.set_style(style='whitegrid')
    ax2 = fig.add_subplot(1,2,2)

    if is_conv_gray == True:
        # グレースケール画像化→三重配列に戻す
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        ax1.imshow(img)
        ax1.set_title(os.path.basename(out_jpg))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        ax1.imshow(img, cmap='gray') # opencvはBGR、matplotlibはRGB だから、cmapの指定ないと色は反転する
        ax1.set_title(os.path.basename(out_jpg))

    #print(img.shape)
    PIL.Image.fromarray(img).save(OUT_DIR+"jpg/"+out_jpg) # ファイル出力

    # 一次元配列化
    img = np.array(img).flatten()
    #print(img.shape)

    # 本来rangeは[0,255]となるはずだが、255に値が集まりすぎたので弾いた
    #img = img[img!=255]

    # 軸の刻み目
    ax2.set_xticks(np.linspace(0,255,9).astype(int))

    # ヒストグラムを計算→色をつける
    N, bins, patches = ax2.hist(img, range=(0,255), bins=256)
    for (i,patch) in enumerate(patches):
        color = cm.gray(bins[i]/256)
        patch.set_facecolor(color)

    if stats==True: # 統計量を表示する
        mean = img.mean()
        std = np.std(img)
        median = np.median(img)
        mode = sstats.mode(img)[0][0]

        # 統計量のラインをひく
        ax2.axvline(mean, color='#d95763', linestyle='solid', linewidth=3)
        ax2.axvline(median, color='#6abe30', linestyle='solid', linewidth=2)
        ax2.axvline(mode, color='#ba9641', linestyle='solid', linewidth=2)
        ax2.axvline(mean + std, color='#d95763', linestyle='dashdot', linewidth=1)
        ax2.axvline(mean - std, color='#d95763', linestyle='dashdot', linewidth=1)

        # 統計量の説明の文字
        ax2.text(mean,N.max()*1.075, "$\mu$",color='#d95763',horizontalalignment='center')
        ax2.text(median,N.max()*1.18,"median", color='#6abe30',rotation=45)
        ax2.text(mode,N.max()*1.15,"mode",color='#ba9641',rotation=45)
        ax2.text(mean+std,N.max()*1.075, "$\mu+\sigma$", color='#d95763',horizontalalignment='center')
        ax2.text(mean-std,N.max()*1.075, "$\mu-\sigma$", color='#d95763',horizontalalignment='center')

        # 累積度数
        hist,bins = np.histogram(img,256,[0,255])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()
        plt.plot(cdf_normalized, color = 'b')
        #plt.legend(('cdf','histogram'), loc = 'upper left')

        fig.tight_layout()
        plt.savefig(OUT_DIR+"hist/"+out_jpg, bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
        print(OUT_DIR+"hist/"+out_jpg)
        plt.show()

        print("min     : ", np.min(img))
        print("max     : ", np.max(img))
        print("mean    : ", mean)
        print("stddev  : ", std)
        print("median  : ", median)
        print("mode    : ", mode)

    else:
        fig.tight_layout()
        plt.savefig(OUT_DIR+"hist/"+out_jpg, bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
        print(OUT_DIR+"hist/"+out_jpg)
        plt.show()

def rgb_hist(rgb_img, stats=False, out_jpg='rgb.jpg', output_dir=OUT_DIR):
    """
    RGBチャンネルごとのヒストグラム
    http://optie.hatenablog.com/entry/2018/02/26/183825
    """
    sns.set()
    sns.set_style(style='ticks')
    fig = plt.figure(figsize=[15,4])
    ax1 = fig.add_subplot(1,2,1)
    sns.set_style(style='whitegrid')
    ax2 = fig.add_subplot(1,2,2)

    ax1.imshow(rgb_img)
    PIL.Image.fromarray(rgb_img).save(OUT_DIR+"jpg/"+out_jpg) # ファイル出力

    color=['r','g','b']

    for (i,col) in enumerate(color): # 各チャンネルのhist
        # cv2.calcHist([img], [channel], mask_img, [binsize], ranges)
        hist = cv2.calcHist([rgb_img], [i], None, [256], [0,256])
        # グラフの形が偏りすぎるので √ をとってみる
        #hist = np.sqrt(hist)
        ax2.plot(hist,color=col)
        ax2.set_xlim([0,256])

        if stats==True: # 統計量を表示する
            # 1色分のarrayだけにする
            img = rgb_img[:,:,i]
            # 一次元配列化
            img = np.array(img).flatten()

            mean = img.mean()
            std = np.std(img)
            median = np.median(img)
            mode = sstats.mode(img)[0][0]

            print(col)
            print("min     : ", np.min(img))
            print("max     : ", np.max(img))
            print("mean    : ", mean)
            print("stddev  : ", std)
            print("median  : ", median)
            print("mode    : ", mode)
            print("")

    fig.tight_layout()
    plt.savefig(OUT_DIR+"hist/"+out_jpg, bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    print(OUT_DIR+"hist/"+out_jpg)
    plt.show()

def axes4():
    """
    四分割したプロットを描画するときのための関数。
    axesが4つ入ったリストを返す
    http://optie.hatenablog.com/entry/2018/02/26/183825
    """
    sns.set()
    fig = plt.figure(figsize=[15,15])
    sns.set_style(style='ticks')
    ax1 = fig.add_subplot(2,2,1)

    sns.set_style(style='whitegrid')
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    axes = [ax1,ax2,ax3,ax4]
    return axes


def tri_hists(rgb_img, color_space='rgb', out_jpg='tri.jpg', output_dir=OUT_DIR):
    """
    'rgb','hsv','lab'の指定を受け取って、
    「入力画像」と「各チャンネルのヒストグラム」の計4枚のプロットを描画する
    ※HSVは色相 Hue, 彩度 Saturation, 明度 Value(Brightness)の三軸からなる色空間
    ※L*a*b*は L* が輝度, a*が赤-緑成分, b*が黄-青成分を示す補色空間(例えばa*は負値を含む軸で、プラス方向が赤、マイナス方向が緑)
    http://optie.hatenablog.com/entry/2018/02/26/183825
    """
    axes = axes4()

    axes[0].set_title('image')
    axes[0].imshow(rgb_img)

    if color_space=='rgb':
        img_ch = rgb_img
        color=['R','G','B']
    elif color_space=='hsv':
        img_ch = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        color=['Hue','Saturation','Brightness']
    elif color_space=='lab':
        img_ch = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab)
        color=['L*','a*','b*']
    else:
        return

    for (i,col) in enumerate(color): # 各チャンネルのhist
        if col=='Hue':  # Hueの値域は[0,180)
            hist = cv2.calcHist([img_ch], [i], None, [180], [0,180])
        else:
            hist = cv2.calcHist([img_ch], [i], None, [256], [0,256])
        #hist = np.sqrt(hist)
        axes[i+1].set_title(col)
        axes[i+1].plot(hist)

    plt.savefig(OUT_DIR+"hist/"+color_space+"_"+out_jpg, bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    print(OUT_DIR+"hist/"+color_space+"_"+out_jpg)
    plt.show()


def hs_hist(rgb_img, out_jpg='hs.jpg', output_dir=OUT_DIR):
    """
    [目的]
    「横軸に 色相 , 縦軸に 彩度 をとり、
    点(H,S)における頻度を 明度 で表現するグラフ」を作りたい。

    [実装]
    入力のRGB画像をHSVに変換し、(H, S)の二次元ヒストグラムを計算して(H, S, 頻度)の配列を作る。
    その配列をHSV2RGBとして変換し、RGB画像としてimshowに表示させる関数。

    http://optie.hatenablog.com/entry/2018/02/26/183825
    """
    sns.set()
    sns.set_style(style='ticks')

    img_hsv= cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    # (H,S) だけにして、H,Sも分離
    hs_2d = img_hsv[:,:,0:2].reshape(-1,2)
    h = hs_2d[:,0]
    s = hs_2d[:,1]

    # ヒストグラムのbinの設定。OpenCVにおいてHの値域は0~180である（0-255に納めるためか）
    hedges = np.arange(0,180)
    sedges = np.arange(0,255)

    # 二次元ヒストグラム
    H, xedges, yedges = np.histogram2d(h,s, bins=(hedges, sedges))
    H = H.T

    # log scaleで偏りを緩和 & 正規化
    H_log = np.log(H+1)
    H_norm = H_log/H_log.max()*255

    # (H,S,頻度)の配列にするために、まずH[S]を縦[横]にリピートし、x行y列の配列にする
    x = H_norm.shape[1]
    y = H_norm.shape[0]
    hue_xy = np.repeat(xedges[:-1],y).reshape(x,y).T
    sat_xy = np.repeat(yedges[:-1],x).reshape(y,x)

    # depth方向にくっつけて、(H,S,頻度)の配列にする。uint8型でないとcvtColorが受けつけないらしい
    HS_hist = np.dstack((hue_xy, sat_xy, H_norm)).astype('uint8')
    HS_hist_im = cv2.cvtColor(HS_hist, cv2.COLOR_HSV2RGB)
    HS_hist_im = cv2.resize(HS_hist_im,(360,100))

    # 以下、元の画像をax1に、ヒストグラムをax2に表示する
    fig = plt.figure(figsize=[15,4])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(rgb_img)
    ax1.set_title('image')

    ax2.set_title('Hue-Saturation Histgram')
    ax2.set_xlabel('Hue')
    ax2.set_ylabel('Saturation')
    ax2.set_xticks(np.linspace(0,360,13))
    ax2.set_xlim(0,360)
    ax2.set_ylim(0,100)
    ax2.imshow(HS_hist_im,origin='low',interpolation='bicubic')

    plt.savefig(OUT_DIR+"hist/hs_hist_"+out_jpg, bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    print(OUT_DIR+"hist/hs_hist_"+out_jpg)
    plt.show()

def ab_hist(rgb_img, out_jpg='ab.jpg', output_dir=OUT_DIR):
    """
    基本的にhs_histと同じ
    http://optie.hatenablog.com/entry/2018/02/26/183825
    """
    sns.set()
    sns.set_style(style='ticks')

    img_lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab)

    # (a,b) だけにして、a,bも分離
    ab_2d = img_lab[:,:,1:3].reshape(-1,2)
    a = ab_2d[:,0]
    b = ab_2d[:,1]

    aedges = np.arange(0,255)
    bedges = np.arange(0,255)

    H, xedges, yedges = np.histogram2d(a,b, bins=(aedges, bedges))
    H = H.T

    H_log = np.log(H+1)
    H_norm = H_log/H_log.max()*255

    # (頻度,a,b)の配列とするために、まずa[b]を縦[横]にリピートし、x行y列の配列にする。
    x = H_norm.shape[1]
    y = H_norm.shape[0]
    a_xy = np.repeat(xedges[:-1],y).reshape(x,y).T
    b_xy = np.repeat(yedges[:-1],x).reshape(y,x)

    # depth方向にくっつけて、(頻度,a,b)の配列にする。
    ab_hist = np.dstack((H_norm, a_xy, b_xy)).astype('uint8')
    ab_hist_im = cv2.cvtColor(ab_hist, cv2.COLOR_Lab2RGB)
    ab_hist_im = cv2.resize(ab_hist_im,(255,255))

    fig = plt.figure(figsize=[15,4])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(rgb_img)
    ax1.set_title('image')

    ax2.set_title('a*b* Histgram')
    ax2.set_xlabel('G ← a* → R')
    ax2.set_ylabel('B ← b* → Y')
    ax2.axvline(255/2, color='white', linestyle='solid', linewidth=1)
    ax2.axhline(255/2, color='white', linestyle='solid', linewidth=1)
    ax2.imshow(ab_hist_im,origin='low',interpolation='bicubic')

    plt.savefig(OUT_DIR+"hist/ab_hist_"+out_jpg, bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    print(OUT_DIR+"hist/ab_hist_"+out_jpg)
    plt.show()


def curve_2(x):
    """
    S字トーンカーブ
    （skimage.exposure.adjust_sigmoid のシグモイド曲線補正と同じようなの）
    input=127 を境として、低い入力値(暗部)はより低い出力値に、高い入力値(明部)はより高い出力値に写す
    結果として暗部・明部が強調され、出力画像を見るとコントラストが高くなる
    →出力のヒストグラムの分布が広がる
    http://optie.hatenablog.com/entry/2018/03/03/141427
    """
    y = (np.sin(np.pi * (x/255 - 0.5)) + 1)/2 * 255
    return y

def curve_gamma1(x):
    """
    ガンマ変換（gamma = 1/2）
    （skimage.exposure.adjust_gamma と同じ）
    gamma > 1では、暗部、中間部がより暗くなり、明部のコントラストが広がる
    gamma < 1の場合は明部、中間部がより明るくなり、暗部のコントラスが広がる
    http://optie.hatenablog.com/entry/2018/03/03/141427
    """
    gamma = 0.5
    y = 255*(x/255)**(1/gamma)
    return y

def curve_gamma2(x):
    """
    ガンマ変換（gamma = 2）
    （skimage.exposure.adjust_gamma と同じ）
    gamma > 1では、暗部、中間部がより暗くなり、明部のコントラストが広がる
    gamma < 1の場合は明部、中間部がより明るくなり、暗部のコントラスが広がる
    http://optie.hatenablog.com/entry/2018/03/03/141427
    """
    gamma = 2
    y = 255*(x/255)**(1/gamma)
    return y

def rgb_hist_simple(rgb_img, ax, ticks=None):
    """
    rgb_img と matplotlib.axes を受け取り、
    axes にRGBヒストグラムをplotして返す
    http://optie.hatenablog.com/entry/2018/03/03/141427
    """
    color=['r','g','b']
    for (i,col) in enumerate(color):
        hist = cv2.calcHist([rgb_img], [i], None, [256], [0,256])
        #hist = np.sqrt(hist)
        ax.plot(hist,color=col)

    if ticks:
        ax.set_xticks(ticks)
    ax.set_title('histogram')
    ax.set_xlim([0,256])

    return ax

def plot_curve(f,rgb_img):
    """
    関数 f:x->y, 画像 を受け取って
    以下のようなグラフを出す
    ----------------------------
    入力画像 | Curve | 出力画像
    histgram |       | histgram
    ----------------------------
    http://optie.hatenablog.com/entry/2018/03/03/141427
    """

    fig = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(2,3)
    x = np.arange(256)

    # トーンカーブを真ん中に
    sns.set_style('darkgrid')
    ax2 = fig.add_subplot(gs[:,1])
    ax2.set_title('Tone Curve')
    ticks = [0,42,84,127,169,211,255]
    ax2.set_xlabel('Input')
    ax2.set_ylabel('Output')
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.plot(x, f(x))

    # 入力画像を←に, 出力画像を→に
    sns.set_style('ticks')
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('input image →')
    ax1.imshow(rgb_img)
    ax1.set_xticks([]), ax1.set_yticks([])  # 目盛りを消す

    # 画素値変換。uint8で渡さないと大変なことに
    out_rgb_img = np.array([f(a).astype('uint8') for a in rgb_img])

    ax3 = fig.add_subplot(gs[0,2])
    ax3.set_title('→ output image')
    ax3.imshow(out_rgb_img)
    ax3.set_xticks([]), ax3.set_yticks([])

    #入力と出力のヒストグラム
    sns.set_style(style='whitegrid')
    ax4 = fig.add_subplot(gs[1,0])
    ax4 = rgb_hist_simple(rgb_img, ax4, ticks)
    ax5 = fig.add_subplot(gs[1,2])
    ax5 = rgb_hist_simple(out_rgb_img, ax5, ticks)

    plt.show()

def naive_curve(f,rgb_img):
    """
    1画素ずつRGB各チャンネルに変換処理を行う
    （フルHD(1920×1080)であれば2073600個の画素なので変換の計算を2073600回行う）
    http://optie.hatenablog.com/entry/2018/03/03/141427
    """
    out_rgb_img = np.array([f(a).astype('uint8') for a in rgb_img])
    return out_rgb_img

def LUT_curve(f,rgb_img):
    """
    Look Up Tableを LUT[input][0] = output という256行の配列として作る。
    例: LUT[0][0] = 0, LUT[127][0] = 160, LUT[255][0] = 255
    """
    LUT = np.arange(256,dtype='uint8').reshape(-1,1)
    LUT = np.array([f(a).astype('uint8') for a in LUT])
    out_rgb_img = cv2.LUT(rgb_img, LUT)
    return out_rgb_img

def simpleplot(rgb_img):
    """
    rgb画像(array)表示
    """
    sns.set_style('ticks')
    plt.title('output image')
    plt.imshow(rgb_img)
    plt.show()
