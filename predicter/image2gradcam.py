
# coding: utf-8
"""
H3-045のコード使ってコマンドラインから画像1枚gradcam実行するコマンドラインスクリプト
Ussage:
    $ python image2gradcam_H3-045.py --image_path input/cat_dog.png # テスト用。imagenet_vgg16でgradcam。gradcam画像はimage_pathと同じディレクトリに出力
    $ CUDA_VISIBLE_DEVICES=1 python image2gradcam_H3-045.py --image_path input/cat_dog.png --model_path model.h5 # 予測スコア最大クラスを指定モデルの最後のPooling層でgradcam
    $ CUDA_VISIBLE_DEVICES=2 python image2gradcam_H3-045.py --image_path input/cat_dog.png --model_path model.h5 --layer_name mix10 --class_idx 0 --out_jpg grad.jpg # gradcamのクラスidや層指定、出力画像パスも措定
    $ CUDA_VISIBLE_DEVICES=3 python image2gradcam_H3-045.py --image_path input/cat_dog.png --model_path model.h5 --is_gradcam_plus # gradcam++で実行
"""


import os, sys, time, shutil, glob, pathlib, argparse
from PIL import Image
import numpy as np

#import urllib.request
## proxy の設定
#proxy_support = urllib.request.ProxyHandler({'http' : 'http://apiproxy:8080', 'https': 'https://apiproxy:8080'})
#opener = urllib.request.build_opener(proxy_support)
#urllib.request.install_opener(opener)

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import keras
import keras.backend as K
K.set_learning_phase(0)

# H3-045のコードimport
#sys.path.append('/gpfsx01/home/aaa00162/jupyterhub/notebook/work_H3-031/H3-045/02.DL/work/code_v2/')
#from predicter import grad_cam
import grad_cam

def image2numpy_keras(image_path:str, shape):
    """
    kerasのAPIで画像ファイルをリサイズしてnp.arrayにする
    Args:
        image_path:画像ファイルパス
        target_size:リサイズする画像サイズ.[331,331,3]みたいなの
    """
    img = keras.preprocessing.image.load_img(image_path, target_size=shape[:2])
    x = keras.preprocessing.image.img_to_array(img)
    return x

def get_last_conv_layer_name(model):
    """ モデルオブジェクトの最後のPooling層の名前取得（gradcamで出力層に一番近い畳込み層の名前必要なので） """
    for i in range(len(model.layers)):
        #print(i, model.layers[i], model.layers[i].name, model.layers[i].trainable, model.layers[i].output)
        if len(model.layers[i].output.shape) == 4:
            last_conv_layer_name = model.layers[i].name
    return last_conv_layer_name

def image2gradcam(model, image_path:str, X=None, layer_name=None, class_idx=None, out_jpg=None, is_gradcam_plus=False):
    """
    画像ファイル1枚からGradCam実行して画像保存
    Args:
       model:モデルオブジェクト
       image_path:入力画像パス
       X:4次元numpy.array型の画像データ（*1./255.後）。Noneならimage_pathから作成する
       layer_name:GradCamかける層の名前。Noneならモデルの最後のPooling層の名前取得にする
       class_idx:GradCamかけるクラスid。Noneなら予測スコア最大クラスでGradCamかける
       out_jpg:GradCam画像出力先パス。Noneならimage_pathから作成する
       is_gradcam_plus:gradcam++で実行するか。Falseだと普通のgradcam実行
    """
    shape = [model.input.shape[1].value, model.input.shape[2].value, model.input.shape[3].value] # モデルオブジェクトの入力層のサイズ取得
    x = image2numpy_keras(image_path, shape) # 画像ファイルをリサイズしてnp.arrayにする

    if X is None:
        X = grad_cam.preprocess_x(x) # np.arrayの画像前処理

    if layer_name is None:
        layer_name = get_last_conv_layer_name(model) # layer_nameなければモデルオブジェクトの最後の畳込み層の名前取得

    if class_idx is None:
        pred_score = model.predict(X)[0]
        class_idx = np.argmax(pred_score) # class_idxなければ予測スコア最大クラスでgradcamかける

    # Grad-Cam実行
    class_output = model.output[:, class_idx]
    if is_gradcam_plus == True:
        jetcam = grad_cam.grad_cam_plus(model, X, x, layer_name, shape[0], shape[1], class_output) # gradcam++で実行
    else:
        jetcam = grad_cam.grad_cam(model, X, x, layer_name, shape[0], shape[1], class_output)
    grad_cam_img = keras.preprocessing.image.array_to_img(jetcam)

    # Grad-Cam画像保存
    if out_jpg is None:
        if is_gradcam_plus == True:
            out_jpg = str(pathlib.Path(image_path).parent)+'/'+str(pathlib.Path(image_path).stem)+f"_classidx{class_idx}_gradcam++.jpg"
        else:
            out_jpg = str(pathlib.Path(image_path).parent)+'/'+str(pathlib.Path(image_path).stem)+f"_classidx{class_idx}_gradcam.jpg"
        print(f"out_jpg: {out_jpg}")
    grad_cam_img.save(out_jpg, 'JPEG', quality=100, optimize=True)

    return grad_cam_img

def main(args):
    if args.model_path is None:
        # テスト用にモデルファイルなしなら imagenet_vgg16 で gradcam 実行できるようにしておく
        from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
        model = VGG16(weights='imagenet') # imagenet_vgg16モデルロード
        x = image2numpy_keras(args.image_path, [224,224,3]) # 画像ファイルをリサイズしてnp.arrayにする
        X = np.expand_dims(x, axis=0)
        X = X.astype('float32')
        X = preprocess_input(X) # imagenet_vgg16の画像前処理
        grad_cam_img = image2gradcam(model, args.image_path, X=X, layer_name='block5_conv3', is_gradcam_plus=args.is_gradcam_plus)
    else:
        model = keras.models.load_model(args.model_path, compile=False) # モデルロード
        grad_cam_img = image2gradcam(model, args.image_path, layer_name=args.layer_name, class_idx=args.class_idx, out_jpg=args.out_jpg, is_gradcam_plus=args.is_gradcam_plus)
    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", type=str, required=True, help="input image path.")
    ap.add_argument("--model_path", type=str, default=None, help="model path.")
    ap.add_argument("--layer_name", type=str, default=None, help="gradcam layer_name.")
    ap.add_argument("--class_idx", type=int, default=None, help="gradcam class_idx.")
    ap.add_argument("--out_jpg", type=str, default=None, help="output gradcam jpg path.")
    ap.add_argument("--is_gradcam_plus", action='store_const', const=True, default=False, help="Grad-Cam++ flag.")
    args = ap.parse_args()
    main(args)
