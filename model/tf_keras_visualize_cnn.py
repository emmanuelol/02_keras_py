# -*- coding: utf-8 -*-
"""
Keras: CNN中間層出力の可視化
http://pynote.hatenablog.com/entry/keras-visualize-kernel-and-feature-map
https://hazm.at/mox/machine-learning/computer-vision/keras/intermediate-layers-visualization/index.html

Usage:
    使用例は ./01_code_test/033_v2_test_model_keras_visualize_cnn.py.ipynb 参照

    # layer_idのCNN中間層の出力可視化
    $ python keras_visualize_cnn.py --run_method "visual_cnn_output" \
                                    --output_dir "output_test\visual_cnn" \
                                    --img_path "horse.jpg" \
                                    --model_path "D:\work\keras_iPhone_pictures\01_classes_results_tfgpu_py36\20190529\train_all\finetuning.h5" \
                                    --layer_id 100

    # モデルのcnnカーネル（フィルター）の重みを可視化
    $ python keras_visualize_cnn.py --run_method "visual_cnn_kernel" \
                                    --output_dir "output_test\visual_cnn" \
                                    --model_path "D:\work\keras_iPhone_pictures\01_classes_results_tfgpu_py36\20190529\train_all\finetuning.h5"

    # 引数:vis_keras_layersで指定した種類の層に全て対してCNN中間層出力の可視化を行う
    $ python keras_visualize_cnn.py --run_method "make_intermediate_images" \
                                    --output_dir "output_test\visual_cnn" \
                                    --img_path "horse.jpg" \
                                    --model_path "D:\work\keras_iPhone_pictures\01_classes_results_tfgpu_py36\20190529\train_all\finetuning.h5" \
                                    --vis_keras_layers "Activation"
"""
import os, sys, math, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras

def visual_cnn_output(model, X, layer_id, vis_cols=8, figsize=(15, 15), output_dir='visual_cnn'):
    """
    CNN中間層出力の可視化
    モデルのcnnカーネル（フィルター）で畳み込みを行った出力結果=特徴マップを可視化
    画像をカーネルの数だけ並べて表示する(features.shape (1, 112, 112, 64) なら 64 がカーネル)
    Args:
        model: modelオブジェクト
        X: 前処理済みのnp.array型の4次元テンソル画像データ
        layer_id: 可視化する層id
        vis_cols: 並べて可視化する画像の列数
        figsize: 並べて可視化する画像のサイズ
        output_dir: 並べて可視化する画像の出力ディレクトリ
    """
    # 指定のレイヤー
    layer = model.layers[layer_id]
    name = layer.name
    print('layer.name', name)
    # 中間層の特徴マップを返す関数を作成
    get_feature_map = keras.backend.function(inputs=[model.input, keras.backend.learning_phase()], outputs=[layer.output])
    # 順伝搬して畳み込み層の特徴マップを取得する。
    features, = get_feature_map([X, False])
    print('features.shape', features.shape)  # features.shape (1, 112, 112, 64)
    kernel_num = features.shape[3]
    print('kernel_num', kernel_num)
    # 特徴マップをカーネルごとに分割し、画像化する。
    feature_imgs = []
    for f in np.split(features, kernel_num, axis=3):
        f = np.squeeze(f, axis=0)  # バッチ次元を削除する。
        f = keras.preprocessing.image.array_to_img(f)  # 特徴マップを画像化する。
        f = np.array(f)  # PIL オブジェクトを numpy 配列にする。
        feature_imgs.append(f)
    print('len(feature_imgs):', len(feature_imgs))
    # カーネルで畳み込みを行った出力結果を並べて表示する
    cols = vis_cols
    rows = int(round(kernel_num / cols))
    fig, ax = plt.subplots(rows, cols, figsize=figsize, dpi=200)
    print(ax.shape)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            f_ax = ax[r, c]
            f_ax.imshow(feature_imgs[i], cmap='gray')
            #f_ax.set_title('Feature '+str(i))
            f_ax.set_axis_off()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "%s.png" % name))
        print('output:', os.path.join(output_dir, "%s.png" % name))
    plt.show(fig)
    plt.close(fig)

def visual_cnn_kernel(model, layer_id=None, vis_cols=8, figsize=(15, 15), output_dir='visual_cnn'):
    """
    モデルのcnnカーネル（フィルター）の重みを可視化
    画像をカーネルの数だけ並べて表示する
    weights.shape (7, 7, 3, 64) のような重みのチャネルが3になってない層はエラーになる
    """
    # layer_id指定なければ重みのチャネルが3の層探す
    if layer_id is None:
        for i, layer in enumerate(model.layers):
            w = model.layers[i].get_weights()
            if len(w) == 2:
                weights, bias = w
                if weights.shape == 4:
                    n_ch = weights.shape[2]
                    if n_ch == 3:
                        layer_id = i
    print('layer_id', layer_id)
    if layer_id is None:
        print('### weights.shapeのチャネルが3の層ないので可視化できない ###')

    # モデルの重みを取得
    weights, bias = model.layers[layer_id].get_weights()
    name = model.layers[layer_id].name
    print('layer.name', name)  # layer.name conv1
    print('weights.shape', weights.shape)  # weights.shape (7, 7, 3, 64)
    print('bias.shape', bias.shape)  # bias.shape (64,)
    kernel_num = int(weights.shape[3])
    print('kernel_num', kernel_num)
    # 重みをカーネルごとに分割し、画像化する。
    weight_imgs = []
    #print(weights)
    for w in np.split(weights, kernel_num, axis=3):
        w = np.squeeze(w, axis=weights.shape[2])  # (KernelH, KernelW, KernelC, 1) -> (KernelH, KernelW, KernelC)
        w = keras.preprocessing.image.array_to_img(w) # 重みを画像化する。
        w = np.array(w)  # PIL オブジェクトを numpy 配列にする。
        weight_imgs.append(w)
    # カーネルの重みを並べて表示する
    cols = vis_cols
    rows = int(kernel_num / cols)
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            w_ax = ax[r, c]
            w_ax.imshow(weight_imgs[i])
            w_ax.set_title('Kernel '+str(i))
            w_ax.set_axis_off()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "%s.png" % name+'_kernel'))
        print('output:', os.path.join(output_dir, "%s.png" % name+'_kernel'))
    plt.show(fig)
    plt.close(fig)

def make_intermediate_images(model, X, vis_keras_layers=keras.layers.Activation, output_dir='visual_cnn'):
    """
    引数:vis_keras_layersで指定した種類の層に全て対してCNN中間層出力の可視化を行う
    指定の各層の出力となった特徴強度(output)をすべて連結した画像を出力
    指定した種類の層に全て対して可視化するのでめちゃめちゃ画像出力される
    注) seaborn ライブラリ使ってheatmapで可視化している
    Args:
        model: modelオブジェクト
        X: 前処理済みのnp.array型の4次元テンソル画像データ
        vis_keras_layers: 可視化する層のデータ型。
                          デフォルトは Activation の層をすべて可視化する。
                          MaxPooling2D の層にする場合は keras.layers.MaxPooling2D にする
                          Conv2D の層にする場合は keras.layers.Conv2D にする
    """
    # 各レイヤー毎のoutputを取得する
    layers = model.layers[1:]
    layer_outputs = [layer.output for layer in layers]
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    #activation_model.summary()
    # 特徴強度の取得
    activations = activation_model.predict(X)
    for i, activation in enumerate(activations):
        print("%2d: %s" % (i, str(activation.shape)))
    # 特徴強度の保存
    # プーリング層の出力のみに絞る (畳み込み層の出力も可視化できるが量が多くなるため)
    activations = [(layer.name, activation) for layer, activation in zip(layers, activations) if isinstance(layer, vis_keras_layers)]
    # 出力層ごとに特徴画像を並べてseabornでヒートマップ画像として出力
    for i, (name, activation) in enumerate(activations):
        if activation.ndim >= 3:
            num_of_image = activation.shape[3]
            print('num_of_image', num_of_image)
            cols = math.ceil(math.sqrt(num_of_image))
            rows = math.floor(num_of_image / cols)
            screen = []
            for y in range(0, rows):
                row = []
                for x in range(0, cols):
                    j = y * cols + x
                    if j < num_of_image:
                        row.append(activation[0, :, :, j])
                    else:
                        row.append(np.zeros())
                screen.append(np.concatenate(row, axis=1))
            screen = np.concatenate(screen, axis=0)
            plt.figure(dpi = 200)
            sns.heatmap(screen, xticklabels=False, yticklabels=False, cmap='gray', cbar=False ) # seabornのheatmap グレースケールにして、カラーバー消しとく
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, "%s.png" % name)) # 余白なるべく消す , bbox_inches='tight', pad_inches=0
                print('output:', os.path.join(output_dir, "%s.png" % name))
            plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_method", type=str, default='visual_cnn_output'
                        , choices=['visual_cnn_output', 'visual_cnn_kernel', 'make_intermediate_images']
                        , help="run method name.")
    parser.add_argument("--output_dir", type=str, default='visual_cnn'
                        , help="cnn visual image output dir path.")
    parser.add_argument("--model_path", type=str, default=r'D:\work\keras_iPhone_pictures\01_classes_results_tfgpu_py36\20190529\train_all\finetuning.h5'
                        , help="keras model path.")
    parser.add_argument("--img_path", type=str, default=r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py\experiment\01_code_test\horse.jpg'
                        , help="image file path.")
    parser.add_argument("--layer_id", type=int, default=1
                        , help="model layer id.")
    parser.add_argument("--vis_keras_layers", type=str, default='Activation'
                        , help="visual keras.layers name.")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')

    keras.backend.clear_session()
    model = keras.models.load_model(args.model_path, compile=False)

    if args.img_path is not None:
        shape = [model.input.shape[1].value, model.input.shape[2].value, model.input.shape[3].value]
        x = keras.preprocessing.image.load_img(args.img_path, target_size=shape[:2])
        x = np.expand_dims(x, axis=0).astype('float32') # 4次元テンソルへ変換
        X = x / 255.0 # 前処理

    if args.run_method == 'visual_cnn_output':
        # CNN中間層出力の可視化
        visual_cnn_output(model, X, args.layer_id, output_dir=args.output_dir)

    elif args.run_method == 'visual_cnn_kernel':
        # モデルのcnnカーネル（フィルター）の重みを可視化
        visual_cnn_kernel(model, args.layer_id, output_dir=args.output_dir)

    elif args.run_method == 'make_intermediate_images':
        if args.vis_keras_layers.lower() == 'activation':
            vis_keras_layers = keras.layers.Activation
        elif args.vis_keras_layers.lower() == 'conv2d':
            vis_keras_layers = keras.layers.Conv2D
        elif args.vis_keras_layers.lower() == 'maxpooling2d':
            vis_keras_layers = keras.layers.MaxPooling2D
        # vis_keras_layersで指定した種類の層に全て対してCNN中間層出力の可視化を行う
        make_intermediate_images(model, X, vis_keras_layers=vis_keras_layers, output_dir=args.output_dir)