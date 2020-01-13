# -*- coding: utf-8 -*-
"""
Augmentorパッケージ（https://github.com/mdbloice/Augmentor）のutilモジュール
Augmentorなら色反転画像や白黒画像が簡単に作成できる
画像ディレクトリやnp.array()型のX,yからKerasのImagedatageneratorインスタンスも作成できる
Augmentorは画像変換のルールきめたPipelineを作って変換実行する
"""
import os, sys
import numpy as np

import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得
sys.path.append( str(current_dir) + '/../Git/Augmentor' )
import Augmentor

def make_pipeline(input_dir=None, output_dir='/Augmentor_output/'
                  , input_width=331, input_height=331
                  , rotate90=0.0, rotate180=0.0, rotate270=0.0, rotate_prob=0.0, rotate_max_left=20, rotate_max_right=20
                  , crop_prob=0.0, crop_area=0.8
                  , crop_by_size_prob=0.0, crop_by_width=280, crop_by_height=280, crop_by_centre=False
                  , shear_prob=0.0, shear_magni=15
                  , skew_prob=0.0, skew_magni=0.2
                  , zoom_prob=0.0, zoom_min=0.5, zoom_max=1.9 # 縮小すると開いた領域が黒塗りされる。嫌ならzoom_min=1.0として縮小やめとく
                  , flip_left_right=0.0
                  , flip_top_bottom=0.0
                  , random_erasing_prob=0.0, random_erasing_area=0.3
                  , random_dist_prob=0.0, random_dist_grid_width=4, random_dist_grid_height=4, random_dist_magnitude=8#https://github.com/mdbloice/p
                  , black_and_white=0.0
                  , greyscale=0.0
                  , invert=0.0):
    """
    Augmentor Pipeline 作成
    オプション引数の値は水増しなし。確率＝0.0とするとエラーになるため、ifでいちいち判定している
    使わない水増しはコメントアウトしてる
    """
    # パイプライン作成
    if input_dir is not None:
        # output_directoryはinput_dirの相対パス。デフォルトの場合input_dirの1つ上の階層に Augmentor_output ディレクトリが自動で作られて出力先になる
        p = Augmentor.Pipeline(input_dir, output_directory=output_dir)# 入力ディレクトリと拡張画像をディスクに保存するディレクトリ(存在しなければ自動で作られる)を指定
    else:
        p = Augmentor.Pipeline()# np.array()型のX,yからパイプライン作る場合
    # 白黒に変換
    if black_and_white > 0.0:
        p.black_and_white(probability=black_and_white, threshold=128)# thresholdはデフォルトで128。ピクセル値がthresholdを超える値はすべて白に変換され、このしきい値以下の値は黒に変換されます。
    # グレースケールに変換
    if greyscale > 0.0:
        p.greyscale(probability=greyscale)
    # 色反転
    if invert > 0.0:
        p.invert(probability=invert)
    # 反転
    if flip_left_right > 0.0:
        p.flip_left_right(probability=flip_left_right)# 左右反転
    if flip_top_bottom > 0.0:
        p.flip_top_bottom(probability=flip_top_bottom)# 上下反転
    # 回転
    if rotate90 > 0.0:
        p.rotate90(probability=rotate90)# 90度回転
    if rotate180 > 0.0:
        p.rotate180(probability=rotate180)# 180度回転
    if rotate270 > 0.0:
        p.rotate270(probability=rotate270)# 270度回転
    if rotate_prob > 0.0:
        # 回転角度の範囲は25度未満でないとエラーになる
        p.rotate(probability=rotate_prob, max_left_rotation=rotate_max_left, max_right_rotation=rotate_max_right)# 回転角度の範囲(max_left_rotation～max_right_rotation)指定して回転
    # 画像を傾ける(shear)。角度は25度未満でないとエラーになる
    if shear_prob > 0.0:
        p.shear(probability=shear_prob, max_shear_left=shear_magni, max_shear_right=shear_magni)
    # キャンバスの歪み（ランダムな方向、左から右、上から下、または8つの角の方向のいずれかに画像傾ける(shear)）:1未満でないとエラーになる
    if skew_prob > 0.0:
        p.skew_tilt(probability=skew_prob, magnitude=skew_magni)
    # ランダムな弾性歪み
    if random_dist_prob > 0.0:
        p.random_distortion(probability=random_dist_prob, grid_width=random_dist_grid_width, grid_height=random_dist_grid_height, magnitude=random_dist_magnitude)
    # ランダムな切り抜き（面積の大きさ指定した拡大縮小）
    if crop_prob > 0.0:
        p.crop_random(probability=crop_prob, percentage_area=crop_area)# percentage_area:切り取る画像の面積の割合
    # 切り取る幅と高さ決めて切り抜き（面積の大きさ指定した拡大縮小）
    if crop_by_size_prob > 0.0:
        p.crop_by_size(probability=crop_by_size_prob, width=crop_by_width, height=crop_by_height, centre=crop_by_centre)# centre:中央部分だけを切り取るか
    # 拡大縮小
    if zoom_prob > 0.0:
        p.zoom(probability=zoom_prob, min_factor=zoom_min, max_factor=zoom_max)# min_factor:ズームする最小倍率。max_factor:ズームする最大倍率
    # Random erasing
    if random_erasing_prob > 0.0:
        p.random_erasing(probability=random_erasing_prob, rectangle_area=random_erasing_area)# rectangle_area:ランダム矩形で遮蔽する画像の面積の割合
    # 画像の大きさ変更 加工の最後にサイズ変更しないとcropでサイズ変わったときエラーになる
    p.resize(probability=1.0, width=input_width, height=input_height)# 画像の大きさ変更
    # パイプラインのルール確認
    print(p.status())
    return p

def make_datagenerator_from_dir(input_dir, batch_size
                                , output_dir='../Augmentor_output'
                                , scaled=True
                                , IDG_options={}):
    """
    画像ディレクトリ指定して、KerasのImagedatageneratorインスタンス作成
    Args:
        input_dir:入力画像ディレクトリ
        batch_size:作成するdatageneratorのバッチサイズ
        output_dir:出力画像ディレクトリ。
                   input_dirの相対パス。デフォルトの場合input_dirの1つ上の階層に Augmentor_output ディレクトリが自動で作られて出力先になる
                   /gpfsx01/home/aaa00162/tmp/ のようなフルパスにすると絶対パスにできる。
        scaled:1/255.の前処理するかのフラグ
        IDG_options:Augmentor Pipelineの水増しルールの辞書。IDG_options={'invert':1.0, 'rotate90':0.5}みたいなの
    Return:
        KerasのImagedatageneratorインスタンス
    Usage:
        # 画像ディレクトリ指定し、KerasのImagedatageneratorインスタンス作成して、インスタンス確認
        IDG_options = augmentor_util.get_base_IDG_options(331, 331)
        gen = augmentor_util.make_datagenerator_from_dir('./input', 32, IDG_options=IDG_options)
        get_train_valid_test.print_image_generator(gen, i=1)
    """
    p = make_pipeline(input_dir=input_dir, output_dir=output_dir, **IDG_options) # パイプライン作成
    gen = p.keras_generator(batch_size=batch_size, scaled=scaled)
    return(gen)

def make_datagenerator_from_array(X, y_onehot, batch_size, IDG_options={}):
    """
    np.array()型のX,y(onehot済み)指定して、KerasのImagedatageneratorインスタンス作成
    Args:
        X:前処理済み入力画像np.array()。shapeは[N,w,h,c])
        y_onehot:onehot済みのラベルnp.array()。shapeは[N,onehot_label]
        batch_size:作成するdatageneratorのバッチサイズ
        IDG_options:Augmentor Pipelineの水増しルールの辞書。IDG_options={'invert':1.0, 'rotate90':0.5}みたいなの
    Return:
        KerasのImagedatageneratorインスタンス
    Usage:
        # 画像ディレクトリ指定し、KerasのImagedatageneratorインスタンス作成して、インスタンス確認
        IDG_options = augmentor_util.get_base_IDG_options(331, 331)
        gen = augmentor_util.make_datagenerator_from_array(x_train, y_train, 32, IDG_options=IDG_options)
        get_train_valid_test.print_image_generator(gen, i=1)
    """
    p = make_pipeline(**IDG_options) # パイプライン作成
    gen = p.keras_generator_from_array((X * 255).astype(np.uint8) # Xの画素値はintで0-255出ないとエラー。PILでloadするため
                                       , y_onehot
                                       , batch_size=batch_size)
    return(gen)


def save_conv_dir_images(input_dir
                         , output_dir='../Augmentor_output/conv_dir_images'
                         , input_width=331
                         , input_height=331
                         , is_greyscale=False
                         , is_invert=False
                         , is_black_and_white=False
                         , IDG_options={}
                         , sample_n=None
                        ):
    """
    指定ディレクトリの画像Augmentor Pipelineで変換して保存する（変換はマルチプロセスでしてくれるので早い）
    グレースケール、色反転、白黒に変換した画像すぐ作れる
    追加無くてもIDG_optionsは必ず{}で指定すること！！！notebookで実行すると前の情報が残る
    Args:
        input_dir:入力画像ディレクトリ
        output_dir:出力画像ディレクトリ。input_dirの相対パス。デフォルトの場合input_dirの1つ上の階層に Augmentor_output/conv_dir_images ディレクトリが自動で作られて出力先になる
        input_width:画像の横幅。この大きさにリサイズする
        input_height:画像の横幅。この大きさにリサイズする
        is_greyscale:グレースケールに変換するか
        is_invert:色反転するか
        is_black_and_white:白黒に変換するか
        IDG_options:その他追加で変換ルール追加用の水増しoptionsの辞書。notebookで実行すると前の情報が残るので必ず引数指定すること！！！
        sample_n:指定ディレクトリの画像ランダムに出す件数。100とかにしたら100枚出力される。デフォルトのNonならランダムではなくディレクトリの画像全件変換する
    Return:
        なし（output_dirに変化後の画像出力される）
    Usage:
        # 指定ディレクトリの画像全件Augmentor Pipelineで色反転して保存する
        augmentor_util.save_conv_dir_images(input_dir
                                            , output_dir='../Augmentor_output/conv_dir_images/invert'
                                            , is_invert=True
                                            , IDG_options={}
                                           )
    """
    IDG_options['input_width'] = input_width
    IDG_options['input_height'] = input_height
    if is_greyscale == True:
        IDG_options['greyscale'] = 1.0
    if is_invert == True:
        IDG_options['invert'] = 1.0
    if is_black_and_white == True:
        IDG_options['black_and_white'] = 1.0
    p = make_pipeline(input_dir=input_dir, output_dir=output_dir, **IDG_options) # パイプライン作成
    if sample_n is None:
        p.process() # 指定ディレクトリの画像全件変換
    else:
        p.sample(sample_n) # ランダムにsample_n件変換
    return


def get_base_IDG_options(input_width, input_height, prob=0.5):
    """
    make_pipeline()の引数に対応するAugmentor Pipelineの基本の水増しoptionsの辞書変数を返す
    Args:
        input_width:画像の横幅。この大きさにリサイズする
        input_height:画像の横幅。この大きさにリサイズする
        prob:各水増しoptionsの実行確率
    """
    IDG_options = {'input_width':input_width, 'input_height':input_height
                   , 'rotate90':prob, 'rotate180':prob, 'rotate270':prob, 'rotate_prob':prob
                   , 'crop_prob':prob
                   , 'shear_prob':prob
                   , 'skew_prob':prob
                   , 'zoom_prob':prob
                   , 'flip_left_right':prob
                   , 'flip_top_bottom':prob
                   , 'random_dist_prob':prob
                   , 'random_erasing_prob':prob
                   #, 'invert':prob
                   }
    return IDG_options
