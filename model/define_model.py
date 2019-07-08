# -*- coding: utf-8 -*-
"""
imagenetの学習済みモデルをfine_tuningしてモデルを定義する

クラス数やハイパーパラメータは引数から変更する

Usage:
import os, sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
path = os.path.join(current_dir, '../')
sys.path.append(path)
from model import define_model

# モデル作成
model, orig_model = define_model.get_fine_tuning_model(output_dir, img_rows, img_cols, channels, num_classes
                                                   , chaice_model, trainable
                                                   , FCnum)
# オプティマイザ
optim = define_model.get_optimizers(choice_optim)

# モデルコンパイル
define_model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
"""

# プロキシ apiproxy:8080 があるので下記の設定入れないとデータダウンロードでエラーになる
import urllib.request
# proxy の設定
proxy_support = urllib.request.ProxyHandler({'http' : 'http://apiproxy:8080', 'https': 'https://apiproxy:8080'})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)

import os, sys
import keras
from keras import optimizers
from keras.models import Model, load_model, model_from_json
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2  import InceptionResNetV2
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import backend as K
from keras.utils import multi_gpu_model
from keras import regularizers

# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得

# githubのNasnetをimport
# https://github.com/titu1994/Keras-NASNet/blob/master/README.md
sys.path.append( str(current_dir) + '/../Git/Keras-NASNet' )
from nasnet import NASNetLarge

# githubのSenetをimport
# https://github.com/titu1994/keras-squeeze-excite-network
sys.path.append( str(current_dir) + '/../Git/keras-squeeze-excite-network' )
import se_inception_v3, se_densenet, se_inception_resnet_v2, se_resnet, se_resnext, se

# githubのWideResnetをimport
# https://github.com/titu1994/Wide-Residual-Networks
sys.path.append( str(current_dir) + '/../Git/Wide-Residual-Networks' )
import wide_residual_network as wrn
import wide_residual_network_include_top_false as wrn_top_false # 出力層消したの

# githubのAdaBoundをimport
# https://github.com/titu1994/keras-adabound
sys.path.append( str(current_dir) + '/../Git/keras-adabound' )
from adabound import AdaBound

# githubのWideResNet + OctConvをimport
# https://qiita.com/koshian2/items/0e40a5930f1aa63a66b9
sys.path.append( str(current_dir) + '/../Git/OctConv-TFKeras' )
#import models as oct_wrn
from oct_conv2d import OctConv2D # load_model(… custom_objects={'OctConv2D':OctConv2D}, compile=False) が必要
import models_include_top_false as oct_wrn_top_false # 出力層消したの

# githubのEfficientNetをimport
# https://github.com/qubvel/efficientnet
sys.path.append( str(current_dir) + '/../Git/efficientnet' )
import efficientnet
from efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

# githubのPeleeNet
sys.path.append( str(current_dir) + '/../Git/PeleeNet-Keras' )
import pelee_net_keras

def save_architecture(model, output_dir):
    """モデルの構造保存"""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'architecture.json')
    json_string = model.to_json()
    open(json_path,"w").write(json_string)

def get_VGG16_model(output_dir, img_rows=224, img_cols=224, channels=3):
    """
    vgg16_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらVGG16のデフォルトの224*224*3になる
    """
    model_path = os.path.join(output_dir, 'vgg16.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        # 入力層の数変更
        input_tensor = Input(shape=(img_rows, img_cols, channels))
        # 最後の全結合層はいらないのでinclude_top=False
        # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになる
        model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

def get_ResNet50_model(output_dir, img_rows=224, img_cols=224, channels=3):
    """
    ResNet50_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらResNet50のデフォルトの224*224*3になる
    """
    model_path = os.path.join(output_dir, 'ResNet50.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        # 入力層の数変更
        input_tensor = Input(shape=(img_rows, img_cols, channels))
        # 最後の全結合層はいらないのでinclude_top=False
        # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになる
        model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

def get_InceptionV3_model(output_dir, img_rows=299, img_cols=299, channels=3):
    """
    InceptionV3_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらInceptionV3のデフォルトの299*299*3になる
    """
    model_path = os.path.join(output_dir, 'InceptionV3.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        # 入力層の数変更
        input_tensor = Input(shape=(img_rows, img_cols, channels))
        # 最後の全結合層はいらないのでinclude_top=False
        # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになる
        model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

def get_Xception_model(output_dir, img_rows=299, img_cols=299, channels=3):
    """
    Xception_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらXceptionのデフォルトの299*299*3になる
    """
    model_path = os.path.join(output_dir, 'Xception.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        # 入力層の数変更
        input_tensor = Input(shape=(img_rows, img_cols, channels))
        # 最後の全結合層はいらないのでinclude_top=False
        # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになる
        model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

def get_InceptionResNetV2_model(output_dir, img_rows=299, img_cols=299, channels=3):
    """
    InceptionResNetV2_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらInceptionResNetV2のデフォルトの299*299*3になる
    """
    model_path = os.path.join(output_dir, 'InceptionResNetV2.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        # 入力層の数変更
        input_tensor = Input(shape=(img_rows, img_cols, channels))
        # 最後の全結合層はいらないのでinclude_top=False
        # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになる
        model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

def get_MobileNet_model(output_dir, img_rows=224, img_cols=224, channels=3):
    """
    MobileNet_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらMobileNetのデフォルトの224*224*3になる
    """
    from keras.applications.mobilenet import MobileNet
    model_path = os.path.join(output_dir, 'MobileNet.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        model = MobileNet(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channels))
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        # load_modelからMobileNetモデルをロードするには、カスタムオブジェクトのrelu6をインポートし，custom_objectsパラメータに渡す
        model = load_model(model_path,
                           custom_objects={'relu6': keras.applications.mobilenet.relu6,
                                           'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D})
    return model

def get_MobileNetV2_model(output_dir, img_rows=224, img_cols=224, channels=3):
    """
    MobileNetV2_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらMobileNetV2のデフォルトの224*224*3になる
    Google Colab(keras2.2.4)でロードできた
    """
    from keras.applications.mobilenetv2 import MobileNetV2
    model_path = os.path.join(output_dir, 'MobileNetV2.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channels))
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

# -------------- github model --------------
def get_NASNetLarge_model(output_dir, img_rows=331, img_cols=331, channels=3):
    """
    NASNetLarge_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらNASNetLargeのデフォルトの331*331*3になる
    """
    model_path = os.path.join(output_dir, 'NASNetLarge.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        # 入力層の数変更
        input_tensor = Input(shape=(img_rows, img_cols, channels))
        # 最後の全結合層はいらないのでinclude_top=False
        # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになる
        # 畳み込みの最後のpoolingも引数で追加できるみたい
        # pooling='avg'にしたらglobal_average_pooling
        # pooling='max'にしたらGlobalMaxPooling2D
        model = NASNetLarge(weights='imagenet', include_top=False, input_tensor=input_tensor)#, pooling='avg')
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

def get_SEResNet_model(output_dir, img_rows=224, img_cols=224, channels=3, seresnet_num=154):
    """
    SEResNet_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらResNetのデフォルトの224*224*3になる
    seresnet_num の数字で以下の種類のどれ使うか選ぶ. デフォルトは SEResNet154 にしておく
    SEResNet18
    SEResNet34
    SEResNet50
    SEResNet101
    SEResNet154
    """
    model_path = os.path.join(output_dir, 'SEResNet'+str(seresnet_num)+'.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        # 入力層の数変更
        input_tensor = Input(shape=(img_rows, img_cols, channels))
        # SENetのモデルはweights='imagenet'としてもimagenetの重みファイルロードしない
        if seresnet_num == 18:
            model = se_resnet.SEResNet18(weights='imagenet', include_top=False, input_tensor=input_tensor)#, pooling='avg')
        elif seresnet_num == 34:
            model = se_resnet.SEResNet34(weights='imagenet', include_top=False, input_tensor=input_tensor)#, pooling='avg')
        elif seresnet_num == 50:
            model = se_resnet.SEResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)#, pooling='avg')
        elif seresnet_num == 101:
            model = se_resnet.SEResNet101(weights='imagenet', include_top=False, input_tensor=input_tensor)#, pooling='avg')
        elif seresnet_num == 154:
            model = se_resnet.SEResNet154(weights='imagenet', include_top=False, input_tensor=input_tensor)#, pooling='avg')
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

def get_SEInceptionV3_model(output_dir, img_rows=299, img_cols=299, channels=3):
    """
    SEInceptionV3_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらInceptionV3のデフォルトの299*299*3になる
    """
    model_path = os.path.join(output_dir, 'SEInceptionV3.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        # 入力層の数変更
        input_tensor = Input(shape=(img_rows, img_cols, channels))
        # 最後の全結合層はいらないのでinclude_top=False
        # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになる
        # 畳み込みの最後のpoolingも引数で追加できるみたい
        # pooling='avg'にしたらglobal_average_pooling
        # pooling='max'にしたらGlobalMaxPooling2D
        # SENetのモデルはweights='imagenet'としてもimagenetの重みファイルロードしない
        model = se_inception_v3.SEInceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)#, pooling='avg')
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

def get_SEInceptionResNetV2_model(output_dir, img_rows=299, img_cols=299, channels=3):
    """
    SEInceptionResNetV2_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらInceptionResNetV2のデフォルトの299*299*3になる
    """
    model_path = os.path.join(output_dir, 'SEInceptionResNetV2.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        # 入力層の数変更
        input_tensor = Input(shape=(img_rows, img_cols, channels))
        # 最後の全結合層はいらないのでinclude_top=False
        # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになる
        # 畳み込みの最後のpoolingも引数で追加できるみたい
        # pooling='avg'にしたらglobal_average_pooling
        # pooling='max'にしたらGlobalMaxPooling2D
        # SENetのモデルはweights='imagenet'としてもimagenetの重みファイルロードしない
        model = se_inception_resnet_v2.SEInceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)#, pooling='avg')
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

def get_SEDenseNet_model(output_dir, img_rows=224, img_cols=224, channels=3, sedensenet_num=169):
    """
    SEDenseNet_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらDenseNetのデフォルトの224*224*3になる
    sedensenet_num の数字で以下の種類のどれ使うか選ぶ. デフォルトはSEDenseNetImageNet169 にしておく
    SEDenseNetImageNet121
    SEDenseNetImageNet161
    SEDenseNetImageNet169
    SEDenseNetImageNet201
    SEDenseNetImageNet264
    SEDenseNetはソースコード変更している！！！！！（FC層のpoolingのコードをコメントアウトした）
    """
    model_path = os.path.join(output_dir, 'SEDenseNetImageNet'+str(sedensenet_num)+'.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        # 入力層の数変更
        input_tensor = Input(shape=(img_rows, img_cols, channels))
        # SEDenseNet のモデルはweights='imagenet'としてもimagenetの重みファイルロードしない？
        if sedensenet_num == 121:
            model = se_densenet.SEDenseNetImageNet121(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif sedensenet_num == 161:
            model = se_densenet.SEDenseNetImageNet161(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif sedensenet_num == 169:
            model = se_densenet.SEDenseNetImageNet169(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif sedensenet_num == 201:
            model = se_densenet.SEDenseNetImageNet201(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif sedensenet_num == 264:
            model = se_densenet.SEDenseNetImageNet264(weights='imagenet', include_top=False, input_tensor=input_tensor)
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

def get_SEResNext_model(output_dir, img_rows=224, img_cols=224, channels=3, seresnext_num=50):
    """
    SEResNext_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらResNetのデフォルトの224*224*3になる
    seresnext_num の数字で以下の種類のどれ使うか選ぶ. デフォルトは SEResNeXt-50 にしておく
    """
    model_path = os.path.join(output_dir, 'SEResNextImageNet'+str(seresnext_num)+'.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        # 入力層の数変更
        input_tensor = Input(shape=(img_rows, img_cols, channels))
        # SEResNext のモデルはweights='imagenet'としてもimagenetの重みファイルロードしない
        if seresnext_num == 50:
            model = se_resnext.SEResNextImageNet(weights='imagenet', depth=[3, 4, 6, 3], include_top=False, input_tensor=input_tensor)
        elif seresnext_num == 101:
            model = se_resnext.SEResNextImageNet(weights='imagenet', depth=[3, 4, 23, 3], include_top=False, input_tensor=input_tensor)
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

def get_WideResNet_model(img_rows=32, img_cols=32, channels=3, num_classes=10, wrn_N=4, wrn_k=8, include_top=True):
    """
    WideResNet_modelロード（基本CIFAR10で使うこと想定）
    入力層のサイズはオプション引数で指定可能。省略したらCIFAR10のデフォルトの32*32*3になる
    デフォルトはWRN-28-8のモデル(wrn_N = 4, wrn_k = 8) このモデルは学習済みモデルあるがロードできない
    ※ wrn_N:ネットワークの深さ(wrn_N = (n - 4) / 6）、wrn_k:ネットワークの幅
    　WRN-28-10のモデル(wrn_N = 4, wrn_k = 10）　
    　WRN-16-8のモデル(wrn_N = 2, wrn_k = 8)  このモデルは学習済みモデルあるがロードできない
    　WRN-40-4のモデル(wrn_N = 6, wrn_k = 4)
    include_top=Falseで出力層なしにする
    """
    if include_top==True:
        model = wrn.create_wide_residual_network((img_rows, img_cols, channels), nb_classes=num_classes, N=wrn_N, k=wrn_k, dropout=0.0)
    else:
        model = wrn_top_false.create_wide_residual_network((img_rows, img_cols, channels), nb_classes=num_classes, N=wrn_N, k=wrn_k, dropout=0.0)
    return model

def get_OctConv_WideResNet_model(alpha=0.25, img_rows=32, img_cols=32, channels=3, wrn_N=4, wrn_k=10):
    """
    WideResNet + OctConv modelロード（基本CIFAR10で使うこと想定）
    https://qiita.com/koshian2/items/0e40a5930f1aa63a66b9
    OctConv:
    画像の低周波成分と高周波成分の分解。低周波と高周波の成分それぞれに畳み込みを加える
    α というハイパーパラメータにより、低周波と高周波のチャンネル数の割り振りを決めます。
    α=0なら高周波だけ、つまりConv2Dと同じになり、α=1なら低周波だけの畳み込みになります。
    α=1なら1/2にダウンサンプリングした低周波成分だけのConv2Dと同じになるので、計算コストは1/4
    論文ではα＝0.25
    WideResNetのデフォルトはWRN-28-10のモデル(wrn_N = 4, wrn_k = 10）
    """
    model = oct_wrn_top_false.create_octconv_wide_resnet(alpha=alpha, N=wrn_N, k=wrn_k, input=(img_rows, img_cols, channels))
    return model

def get_EfficientNet_model(input_shape=None, efficientnet_num=3, weights='imagenet'):
    """
    EfficientNet_modelロード
    入力層のサイズはオプション引数で指定可能。省略したら各EfficientNetのデフォルトのサイズにする
    https://github.com/qubvel/efficientnet/blob/master/efficientnet/model.py より
                                        imagenet val set
                                        @top1 acc	@top5 acc
        EfficientNetB0 - (224, 224, 3)  0.7668	0.9312
        EfficientNetB1 - (240, 240, 3)  0.7863	0.9418
        EfficientNetB2 - (260, 260, 3)  0.7968	0.9475
        EfficientNetB3 - (300, 300, 3)  0.8083	0.9531
        EfficientNetB4 - (380, 380, 3)  0.8259	0.9612
        EfficientNetB5 - (456, 456, 3)  0.8309	0.9646
        EfficientNetB6 - (528, 528, 3)  20190619時点では、EfficientNetB4までしかimagenetの重みファイルない
        EfficientNetB7 - (600, 600, 3)  20190619時点では、EfficientNetB4までしかimagenetの重みファイルない
    """
    print('EfficientNetB'+str(efficientnet_num))
    if input_shape == None:
        if efficientnet_num == 0:
            input_shape = (224, 224, 3)
        elif efficientnet_num == 1:
            input_shape = (240, 240, 3)
        elif efficientnet_num == 2:
            input_shape = (260, 260, 3)
        elif efficientnet_num == 3:
            input_shape = (300, 300, 3)
        elif efficientnet_num == 4:
            input_shape = (380, 380, 3)
        elif efficientnet_num == 5:
            input_shape = (456, 456, 3)
        elif efficientnet_num == 6:
            input_shape = (528, 528, 3)
        elif efficientnet_num == 7:
            input_shape = (600, 600, 3)
    else:
        input_tensor = input_shape
    print('input_shape:', input_shape)
    # レイヤーに独自関数(efficientnet.model.conv_kernel_initializer)を使っているためモデルファイル(.h5py)がmodel.load()できない。model.load_weight()はできる
    # 毎回imagenetのモデルファイルダウンロードする必要あり
    if efficientnet_num == 0:
        model = EfficientNetB0(weights=weights, include_top=False, input_shape=input_shape)
    elif efficientnet_num == 1:
        model = EfficientNetB1(weights=weights, include_top=False, input_shape=input_shape)
    elif efficientnet_num == 2:
        model = EfficientNetB2(weights=weights, include_top=False, input_shape=input_shape)
    elif efficientnet_num == 3:
        model = EfficientNetB3(weights=weights, include_top=False, input_shape=input_shape)
    elif efficientnet_num == 4:
        model = EfficientNetB4(weights=weights, include_top=False, input_shape=input_shape)
    elif efficientnet_num == 5:
        model = EfficientNetB5(weights=weights, include_top=False, input_shape=input_shape)
    elif efficientnet_num == 6:
        model = EfficientNetB6(weights=None, include_top=False, input_shape=input_shape)
    elif efficientnet_num == 7:
        model = EfficientNetB7(weights=None, include_top=False, input_shape=input_shape)
    return model

def get_Pelee_net(input_shape=(224,224,3), include_top=True, use_stem_block=True, num_classes=10):
    """
    PeleeNet と呼ばれるDenseNetベースの軽量化なモデル
    係数が2.8Mと少ないながら、ImageNetでMobileNetV2以上の精度を出す
    ImageNetでPeleeNetがTop1:72.6%、MobileNetV2がTop1:72.0%
    StemBlock-DenseBlock-Transition Layer の構造
        StemBlock:入力のダウンサンプリング（入力層のサイズを小さくする）用のブロック
        DenseBlock:DenseNetの「知識の積み重ね、集約」といったConcatで少しずつチャンネル数が増えていく構造のブロック
        Transition Layer:全体に1x1Convを掛けてダウンサンプリングするレイヤー
    Stem blockの有無は、入力解像度・出力解像度が変わる
    Stem blockありが本来のPeleeNetの構造。なしの場合は、いきなりDenseLayerに入る

    https://qiita.com/koshian2/items/187e240f478504079e7a

    imagenetの重みファイルはないので、h5pyファイル保存せず毎回アーキテクチャ作る
    """
    if include_top == True:
        model = pelee_net_keras.PeleeNet(input_shape=input_shape
                                         , use_stem_block=use_stem_block
                                         , n_classes=num_classes)
    else:
        model = pelee_net_keras.PeleeNet(input_shape=input_shape
                                         , use_stem_block=use_stem_block
                                         , include_top=include_top)
    return model

def get_adabound(lr=0.001, final_lr=0.1, beta_1=0.9, beta_2=0.999, gamma=1e-3, epsilon=None, decay=0.0, amsbound=False, weight_decay=0.0):
    """
    githubのadaboundをimport
    https://github.com/titu1994/keras-adabound
    ※AdaBound: Adamに学習率の上限と下限を動的に加えたもの
                AMSGradに学習率の上限と下限を動的に加えたものをAMSBound
                どちらの手法も最初はAdamのように動き、後半からSGDのように動く
                Adamの良さである初期の学習の速さとSGDの良さである汎化能力を両立した最適化手法

    コンパイル済みモデルでロードする場合はcustom_objectsが必要
    from adabound import AdaBound
    model = keras.models.load_model(os.path.join(output_dir, 'finetuning.h5'), custom_objects={'AdaBound':AdaBound})
    """
    if 'adabound' in sys.modules.keys():
        return AdaBound(lr=lr, final_lr=final_lr, beta_1=beta_1, beta_2=beta_2, gamma=gamma, epsilon=epsilon, decay=decay, amsbound=amsbound, weight_decay=weight_decay)
    else:
        print("It can not be imported because there is no adabound library.Please obtain the adabound library from 'https://github.com/titu1994/keras-adabound'.")


def get_optimizers(choice_optim='sgd', lr=0.0, decay=0.0
                   , momentum=0.9, nesterov=True # SGD
                   , rmsprop_rho=0.9#, epsilon=None # RMSprop
                   , adadelta_rho=0.95 # Adadelta
                   , beta_1=0.9, beta_2=0.999, amsgrad=False # Adam, Adamax, Nadam
                  ):
    """
    オプティマイザを取得する
    引数のchoice_optim は 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam' のいずれかを指定する
    オプティマイザの学習率とlr_decayを変えれるようにする。指定しなければデフォルトの値が入る
    """
    print('---- choice_optim =', choice_optim, '----')
    optim = ''
    if choice_optim == 'sgd':
        if lr == 0.0:
            lr=0.01
        optim = keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
        print('sgd_lr sgd_momentum sgd_decay sgd_nesterov =', lr, momentum, decay, nesterov)
    elif choice_optim == 'rmsprop':
        if lr == 0.0:
            lr=0.001
        optim = keras.optimizers.RMSprop(lr=lr, rho=rmsprop_rho, decay=decay)
        print('rmsprop_lr rmsprop_decay rmsprop_rho =', lr, decay, rmsprop_rho)#, epsilon
    elif choice_optim == 'adagrad':
        if lr == 0.0:
            lr=0.01
        optim = keras.optimizers.Adagrad(lr=lr, decay=decay)
        print('adagrad_lr adagrad_decay =', lr, decay)#, epsilon)
    elif choice_optim == 'adadelta':
        if lr == 0.0:
            lr=1.0
        optim = keras.optimizers.Adadelta(lr=lr, decay=decay, rho=adadelta_rho)
        print('adadelta_lr adadelta_decay adadelta_rho =', lr, decay, adadelta_rho)#, epsilon)
    elif choice_optim == 'adam':
        if lr == 0.0:
            lr=0.001
        # keras 2.1.5 以上なら amsgrad あり
        if int(keras.__version__.split('.')[1]) > 0 and int(keras.__version__.split('.')[2]) > 4:
            optim = keras.optimizers.Adam(lr=lr, decay=decay, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)#, epsilon=epsilon)
            print('adam_lr adam_decay beta_1 beta_2, amsgrad =', lr, decay, beta_1, beta_2, amsgrad)#, epsilon)
        else:
            optim = keras.optimizers.Adam(lr=lr, decay=decay, beta_1=beta_1, beta_2=beta_2)#, amsgrad=amsgrad, epsilon=epsilon)
            print('adam_lr adam_decay beta_1 beta_2 =', lr, decay, beta_1, beta_2)#, epsilon)
    elif choice_optim == 'adamax':
        if lr == 0.0:
            lr=0.002
        optim = keras.optimizers.Adamax(lr=lr, decay=decay, beta_1=beta_1, beta_2=beta_2)#, epsilon=epsilon)
        print('adamax_lr adamax_decay beta_1 beta_2 =', lr, decay, beta_1, beta_2)#, epsilon)
    elif choice_optim == 'nadam':
        if lr == 0.0:
            lr=0.002
        if decay == 0.0:
             # Nadam はschedule_decay=0.004 がデフォルト値
            schedule_decay=0.004
        else:
            schedule_decay=decay
        optim = keras.optimizers.Nadam(lr=lr, schedule_decay=schedule_decay, beta_1=beta_1, beta_2=beta_2)#, epsilon=epsilon)
        print('nadam_lr nadam_schedule_decay beta_1 beta_2 =', lr, schedule_decay, beta_1, beta_2)#, epsilon)
    elif choice_optim == 'adabound':
        if lr == 0.0:
            lr=0.001
        optim = get_adabound(lr=lr, final_lr=lr*10, decay=decay, beta_1=beta_1, beta_2=beta_2, amsbound=amsgrad)
        print('adabound_lr adabound_final_lr adabound_decay beta_1 beta_2, amsbound =', lr, lr*10, decay, beta_1, beta_2, amsgrad)

    return optim


def FC_batch_drop(x, activation='relu', dence=1024, dropout_rate=0.5, addBatchNorm=None, kernel_initializer='he_normal', l2_rate=1e-4, name=None):
    """
    中間層の全結合1層（BatchNormalizationとDropout追加可能）
    Args:
        x: model.output
        activation: 活性化関数。デフォルトは'relu'
        dence, dropout_rate: 各層のニューロンの数。デフォルトは1024としてる
        dropout_rate: dropout_rate(0<dropout_rate<1 の範囲出ないとエラーになる)。デフォルトは0.5としてる
        addBatchNorm: BatchNormalization いれるか。デフォルトは入れない
        kernel_initializer:Denseレイヤーの重みの初期化 デフォルトをHe の正規分布('he_normal')としてる。
                           Denseのデフォルトglorot_uniform（Glorot（Xavier）の一様分布）にしたい場合は 'glorot_uniform' とすること
        l2_rate:l2正則化によるWeight decay入れるか。デフォルトは無し(1e-4)。kerasのマニュアルでは0.01とか使ってる。Deeptoxの4層では1e-04, 1e-05, 1e-06, 0にしてる
        name: 層の名前
    Returns:
        全結合1層(x)
    """
    x = Dense(dence, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(l2_rate), name=name+'_dence')(x)
    if addBatchNorm != None:
        x = BatchNormalization(name=name+'_batchNormalization')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name=name+'_dropout')(x)
    print('dence dropout addBatchNorm kernel_initializer l2_rate =', dence, dropout_rate, addBatchNorm, kernel_initializer, l2_rate)
    return x

def FC_0_5_layer(x, FCnum
                , Dence_1, Dropout_1, addBatchNorm_1, kernel_initializer_1, l2_rate_1
                , Dence_2, Dropout_2, addBatchNorm_2, kernel_initializer_2, l2_rate_2
                , Dence_3, Dropout_3, addBatchNorm_3, kernel_initializer_3, l2_rate_3
                , Dence_4, Dropout_4, addBatchNorm_4, kernel_initializer_4, l2_rate_4
                , Dence_5, Dropout_5, addBatchNorm_5, kernel_initializer_5, l2_rate_5
                , name='FC'):
    """
    全結合0-5層構築
    Args:
        x: model.output
        FCnum: 追加する層の数（5層まで追加できる） 1,2,3,4,5 のどれか指定する.それ以外の値入れると全結合層なし
        name: 層の名前
        Dence_n, Dropout_n, addBatchNorm_n, kernel_initializer_n, l2_rate_n: 各層のニューロンの数、dropout_rate、BatchNormalization いれるか、Denseレイヤーの重みの初期化、l2正則化によるWeight decay
    Returns:
        全結合0-5層(x)
    """
    print('----- FC_layer -----')
    if FCnum == 1:
        x = FC_batch_drop(x, activation='relu', dence=Dence_1, dropout_rate=Dropout_1, addBatchNorm=addBatchNorm_1, kernel_initializer=kernel_initializer_1, l2_rate=l2_rate_1, name=name+'1')
    if FCnum == 2:
        x = FC_batch_drop(x, activation='relu', dence=Dence_1, dropout_rate=Dropout_1, addBatchNorm=addBatchNorm_1, kernel_initializer=kernel_initializer_1, l2_rate=l2_rate_1, name=name+'1')
        x = FC_batch_drop(x, activation='relu', dence=Dence_2, dropout_rate=Dropout_2, addBatchNorm=addBatchNorm_2, kernel_initializer=kernel_initializer_2, l2_rate=l2_rate_2, name=name+'2')
    if FCnum == 3:
        x = FC_batch_drop(x, activation='relu', dence=Dence_1, dropout_rate=Dropout_1, addBatchNorm=addBatchNorm_1, kernel_initializer=kernel_initializer_1, l2_rate=l2_rate_1, name=name+'1')
        x = FC_batch_drop(x, activation='relu', dence=Dence_2, dropout_rate=Dropout_2, addBatchNorm=addBatchNorm_2, kernel_initializer=kernel_initializer_2, l2_rate=l2_rate_2, name=name+'2')
        x = FC_batch_drop(x, activation='relu', dence=Dence_3, dropout_rate=Dropout_3, addBatchNorm=addBatchNorm_3, kernel_initializer=kernel_initializer_3, l2_rate=l2_rate_3, name=name+'3')
    if FCnum == 4:
        x = FC_batch_drop(x, activation='relu', dence=Dence_1, dropout_rate=Dropout_1, addBatchNorm=addBatchNorm_1, kernel_initializer=kernel_initializer_1, l2_rate=l2_rate_1, name=name+'1')
        x = FC_batch_drop(x, activation='relu', dence=Dence_2, dropout_rate=Dropout_2, addBatchNorm=addBatchNorm_2, kernel_initializer=kernel_initializer_2, l2_rate=l2_rate_2, name=name+'2')
        x = FC_batch_drop(x, activation='relu', dence=Dence_3, dropout_rate=Dropout_3, addBatchNorm=addBatchNorm_3, kernel_initializer=kernel_initializer_3, l2_rate=l2_rate_3, name=name+'3')
        x = FC_batch_drop(x, activation='relu', dence=Dence_4, dropout_rate=Dropout_4, addBatchNorm=addBatchNorm_4, kernel_initializer=kernel_initializer_4, l2_rate=l2_rate_4, name=name+'4')
    if FCnum == 5:
        x = FC_batch_drop(x, activation='relu', dence=Dence_1, dropout_rate=Dropout_1, addBatchNorm=addBatchNorm_1, kernel_initializer=kernel_initializer_1, l2_rate=l2_rate_1, name=name+'1')
        x = FC_batch_drop(x, activation='relu', dence=Dence_2, dropout_rate=Dropout_2, addBatchNorm=addBatchNorm_2, kernel_initializer=kernel_initializer_2, l2_rate=l2_rate_2, name=name+'2')
        x = FC_batch_drop(x, activation='relu', dence=Dence_3, dropout_rate=Dropout_3, addBatchNorm=addBatchNorm_3, kernel_initializer=kernel_initializer_3, l2_rate=l2_rate_3, name=name+'3')
        x = FC_batch_drop(x, activation='relu', dence=Dence_4, dropout_rate=Dropout_4, addBatchNorm=addBatchNorm_4, kernel_initializer=kernel_initializer_4, l2_rate=l2_rate_4, name=name+'4')
        x = FC_batch_drop(x, activation='relu', dence=Dence_5, dropout_rate=Dropout_5, addBatchNorm=addBatchNorm_5, kernel_initializer=kernel_initializer_5, l2_rate=l2_rate_5, name=name+'5')
    return x

def get_fine_tuning_model(output_dir, img_rows, img_cols, channels, num_classes
                            , choice_model, trainable
                            , FCnum=0# 1,2,3,4,5
                            , FCpool='GlobalAveragePooling2D'
                            , Dence_1=1024, Dropout_1=0.5, addBatchNorm_1=None, kernel_initializer_1='he_normal', l2_rate_1=1e-4
                            , Dence_2=512, Dropout_2=0.5, addBatchNorm_2=None, kernel_initializer_2='he_normal', l2_rate_2=1e-4
                            , Dence_3=256, Dropout_3=0.5, addBatchNorm_3=None, kernel_initializer_3='he_normal', l2_rate_3=1e-4
                            , Dence_4=128, Dropout_4=0.5, addBatchNorm_4=None, kernel_initializer_4='he_normal', l2_rate_4=1e-4
                            , Dence_5=64, Dropout_5=0.5, addBatchNorm_5=None, kernel_initializer_5='he_normal', l2_rate_5=1e-4
                            , pred_kernel_initializer='zeros', pred_l2_rate=1e-4
                            , activation='softmax'#'sigmoid'
                            , gpu_count=1
                            , skip_bn=True
                            , seresnet_num=154 # SEResNet の種類指定 18,34,50,101,154 のいずれかしかだめ
                            , sedensenet_num=169 # SEDenseNet の種類指定 121,161,169,201,264 のいずれかしかだめ
                            , seresnext_num=50 # SEResNext の種類指定 50,101 のいずれかしかだめ
                            , add_se=False # FC層の前にSE block つけるか
                            , wrn_N=4, wrn_k=10 # WideResNetの引数
                            , oct_conv_alpha=0.25 # OctConv_WideResNet の低周波と高周波のチャンネル数の割り振り具合であるα
                            , efficientnet_num=3 # EfficientNet の種類指定 0,1,2,3,4,5,6,7 のいずれかしかだめ
                            ):
    """
    fine-tuningなど設定したモデルを返す
    オプティマイザの引数入れたくないのでコンパイルはしない
    マルチGPU対応あり gpu_count>1 なら return でマルチじゃないオリジナルのモデルとマルチGPUモデルを返す
    """
    print('----- model_param -----')
    print('output_dir =', output_dir)
    print('img_rows img_cols channels =', img_rows, img_cols, channels)
    print('num_classes =', num_classes)
    print('choice_model trainable =', choice_model, trainable)
    print('FCnum =', FCnum)
    print('FCpool =', FCpool)
    print('pred_kernel_initializer pred_l2_rate =', pred_kernel_initializer, pred_l2_rate)
    print('activation =', activation)
    print('gpu_count =', gpu_count)
    print('skip_bn =', skip_bn)

    trained_model = ''
    if choice_model == 'VGG16':
        trained_model = get_VGG16_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
        # trainable == 15
    elif choice_model == 'ResNet50':
        trained_model = get_ResNet50_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
        # trainable == 164
    elif choice_model == 'InceptionV3':
        trained_model = get_InceptionV3_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
        # trainable == 249
    elif choice_model == 'Xception':
        trained_model = get_Xception_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
        # trainable == 116
    elif choice_model == 'InceptionResNetV2':
        trained_model = get_InceptionResNetV2_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
        # trainable == 761
    elif choice_model == 'MobileNet':
        trained_model = get_MobileNet_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
    elif choice_model == 'MobileNetV2':
        trained_model = get_MobileNetV2_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
    elif choice_model == 'NASNetLarge':
        trained_model = get_NASNetLarge_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
    elif choice_model == 'SEResNet':
        print('seresnet_num =', seresnet_num)
        trained_model = get_SEResNet_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels, seresnet_num=seresnet_num)
    elif choice_model == 'SEInceptionV3':
        trained_model = get_SEInceptionV3_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
    elif choice_model == 'SEInceptionResNetV2':
        trained_model = get_SEInceptionResNetV2_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
    elif choice_model == 'SEDenseNet':
        print('sedensenet_num =', sedensenet_num)
        trained_model = get_SEDenseNet_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels, sedensenet_num=sedensenet_num)
    elif choice_model == 'SEResNext':
        print('seresnext_num =', seresnext_num)
        trained_model = get_SEResNext_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels, seresnext_num=seresnext_num)
    elif choice_model == 'WideResNet':
        print('wrn_N, wrn_k =', wrn_N, wrn_k)
        trained_model = get_WideResNet_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes, wrn_N=wrn_N, wrn_k=wrn_k, include_top=False)
    elif choice_model == 'OctConv_WideResNet':
        print('oct_conv_alpha, wrn_N, wrn_k =', oct_conv_alpha, wrn_N, wrn_k)
        trained_model = get_OctConv_WideResNet_model(alpha=oct_conv_alpha, img_rows=img_rows, img_cols=img_cols, channels=channels, wrn_N=wrn_N, wrn_k=wrn_k)
    elif choice_model == 'EfficientNet':
        if (img_rows is None) and (img_cols is None) and (channels is None):
            trained_model = get_EfficientNet_model(input_shape=None, efficientnet_num=efficientnet_num, weights='imagenet')
        else:
            trained_model = get_EfficientNet_model(input_shape=(img_rows,img_cols,channels), efficientnet_num=efficientnet_num, weights='imagenet')
    elif choice_model == 'PeleeNet':
            trained_model = get_Pelee_net(input_shape=(img_rows,img_cols,channels), include_top=False)

    #print(trained_model.summary())

    x = trained_model.output
    # SE block つける
    if add_se==True:
        print('add_se =', add_se)
        x = se.squeeze_excite_block(x)
    # 学習済みモデルのpooling指定
    if FCpool=='GlobalAveragePooling2D':
        x = GlobalAveragePooling2D(name='FC_avg')(x)
    elif FCpool=='GlobalMaxPooling2D':
        x = GlobalMaxPooling2D(name='FC_max')(x)
    # 全結合0-5層構築
    x = FC_0_5_layer(x, FCnum
                        , Dence_1, Dropout_1, addBatchNorm_1, kernel_initializer_1, l2_rate_1
                        , Dence_2, Dropout_2, addBatchNorm_2, kernel_initializer_2, l2_rate_2
                        , Dence_3, Dropout_3, addBatchNorm_3, kernel_initializer_3, l2_rate_3
                        , Dence_4, Dropout_4, addBatchNorm_4, kernel_initializer_4, l2_rate_4
                        , Dence_5, Dropout_5, addBatchNorm_5, kernel_initializer_5, l2_rate_5
                        , name='FC')
    # 出力層構築
    # 出力層のkernel_initializerとかは下山さんのコードまねた
    predictions = Dense(num_classes, activation=activation
                        , kernel_initializer=pred_kernel_initializer
                        , kernel_regularizer=regularizers.l2(pred_l2_rate)
                        , name='pred')(x)

    # 全結合層を削除したimagenetのtrainedモデルと構築した全結合層を結合
    model = Model(inputs=trained_model.input, outputs=predictions)

    if trainable == 'all':
        # 全レイヤーアンフリーズにする（全ての重みを再学習させる）
        for layer in model.layers:
            layer.trainable = True
    elif trainable == 0:
        # 全レイヤーフリーズ（重み学習させない）
        for layer in model.layers:
            # BatchNormalizationだけは重み学習 下山さんコードより
            if skip_bn and isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    else:
        # Fine-tuning 特定のレイヤまでfine-tuning
        for layer in model.layers[:trainable]:
            # BatchNormalizationだけは重み学習 下山さんコードより
            if skip_bn and isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        for layer in model.layers[trainable:]:
            layer.trainable = True

    # train layer 確認
    #for i,layer in enumerate(model.layers):
    #    print(i, layer.name, layer.trainable)
    # モデル情報表示
    #model.summary()
    # モデル描画
    #keras.utils.plot_model(model, to_file = os.path.join(output_dir, 'model.svg'), show_shapes=True)

    # マルチじゃないオリジナルのモデル確保 https://github.com/keras-team/keras/issues/8649
    orig_model = model
    if gpu_count > 1:
        # マルチGPU http://tech.wonderpla.net/entry/2018/01/09/110000
        model = multi_gpu_model(model, gpus=gpu_count)

    # モデルの構造保存
    save_architecture(orig_model, output_dir)

    return model, orig_model


def get_12branch_fine_tuning_model(output_dir, img_rows, img_cols, channels, num_classes
                            , choice_model, trainable
                            , FCnum=0# 1,2,3,4,5
                            , FCpool='GlobalAveragePooling2D'
                            , Dence_1=1024, Dropout_1=0.5, addBatchNorm_1=None, kernel_initializer_1='he_normal', l2_rate_1=1e-4
                            , Dence_2=512, Dropout_2=0.5, addBatchNorm_2=None, kernel_initializer_2='he_normal', l2_rate_2=1e-4
                            , Dence_3=256, Dropout_3=0.5, addBatchNorm_3=None, kernel_initializer_3='he_normal', l2_rate_3=1e-4
                            , Dence_4=128, Dropout_4=0.5, addBatchNorm_4=None, kernel_initializer_4='he_normal', l2_rate_4=1e-4
                            , Dence_5=64, Dropout_5=0.5, addBatchNorm_5=None, kernel_initializer_5='he_normal', l2_rate_5=1e-4
                            , pred_kernel_initializer='zeros', pred_l2_rate=1e-4
                            , activation='softmax'#'sigmoid'
                            , gpu_count=1
                            , skip_bn=True
                            , seresnet_num=154 # SEResNet の種類指定 18,34,50,101,154 のいずれかしかだめ
                            , sedensenet_num=169 # SEDenseNet の種類指定 121,161,169,201,264 のいずれかしかだめ
                            , seresnext_num=50 # SEResNext の種類指定 50,101 のいずれかしかだめ
                            , add_se=False # FC層の前にSE block つけるか
                            , wrn_N=4, wrn_k=10 # WideResNetの引数
                            , oct_conv_alpha=0.25 # OctConv_WideResNet の低周波と高周波のチャンネル数の割り振り具合であるα
                            , efficientnet_num=3 # EfficientNet の種類指定 0,1,2,3,4,5,6,7 のいずれかしかだめ
                            ):
    """
    全結合層12個に分岐し、fine-tuning設定したモデルを返す
    オプティマイザの引数入れたくないのでコンパイルはしない
    マルチGPU対応あり gpu_count>1 なら return でマルチじゃないオリジナルのモデルとマルチGPUモデルを返す
    """
    print('----- model_param -----')
    print('output_dir =', output_dir)
    print('img_rows img_cols channels =', img_rows, img_cols, channels)
    print('num_classes =', num_classes)
    print('choice_model trainable =', choice_model, trainable)
    print('FCnum =', FCnum)
    print('FCpool =', FCpool)
    print('pred_kernel_initializer pred_l2_rate =', pred_kernel_initializer, pred_l2_rate)
    print('activation =', activation)
    print('gpu_count =', gpu_count)
    print('skip_bn =', skip_bn)

    trained_model = ''
    if choice_model == 'VGG16':
        trained_model = get_VGG16_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
        # trainable == 15
    elif choice_model == 'ResNet50':
        trained_model = get_ResNet50_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
        # trainable == 164
    elif choice_model == 'InceptionV3':
        trained_model = get_InceptionV3_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
        # trainable == 249
    elif choice_model == 'Xception':
        trained_model = get_Xception_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
        # trainable == 116
    elif choice_model == 'InceptionResNetV2':
        trained_model = get_InceptionResNetV2_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
        # trainable == 761
    elif choice_model == 'MobileNet':
        trained_model = get_MobileNet_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
    elif choice_model == 'MobileNetV2':
        trained_model = get_MobileNetV2_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
    elif choice_model == 'NASNetLarge':
        trained_model = get_NASNetLarge_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
    elif choice_model == 'SEResNet':
        print('seresnet_num =', seresnet_num)
        trained_model = get_SEResNet_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels, seresnet_num=seresnet_num)
    elif choice_model == 'SEInceptionV3':
        trained_model = get_SEInceptionV3_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
    elif choice_model == 'SEInceptionResNetV2':
        trained_model = get_SEInceptionResNetV2_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels)
    elif choice_model == 'SEDenseNet':
        print('sedensenet_num =', sedensenet_num)
        trained_model = get_SEDenseNet_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels, sedensenet_num=sedensenet_num)
    elif choice_model == 'SEResNext':
        print('seresnext_num =', seresnext_num)
        trained_model = get_SEResNext_model(output_dir, img_rows=img_rows, img_cols=img_cols, channels=channels, seresnext_num=seresnext_num)
    elif choice_model == 'WideResNet':
        print('wrn_N, wrn_k =', wrn_N, wrn_k)
        trained_model = get_WideResNet_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=num_classes, wrn_N=wrn_N, wrn_k=wrn_k, include_top=False)
    elif choice_model == 'OctConv_WideResNet':
        print('oct_conv_alpha, wrn_N, wrn_k =', oct_conv_alpha, wrn_N, wrn_k)
        trained_model = get_OctConv_WideResNet_model(alpha=oct_conv_alpha, img_rows=img_rows, img_cols=img_cols, channels=channels, wrn_N=wrn_N, wrn_k=wrn_k)
    elif choice_model == 'EfficientNet':
        if (img_rows is None) and (img_cols is None) and (channels is None):
            trained_model = get_EfficientNet_model(input_shape=None, efficientnet_num=efficientnet_num, weights='imagenet')
        else:
            trained_model = get_EfficientNet_model(input_shape=(img_rows,img_cols,channels), efficientnet_num=efficientnet_num, weights='imagenet')
    elif choice_model == 'PeleeNet':
            trained_model = get_Pelee_net(input_shape=(img_rows,img_cols,channels), include_top=False)
    #print(trained_model.summary())

    x = trained_model.output
    # 学習済みモデルのpooling指定
    if FCpool=='GlobalAveragePooling2D':
        x = GlobalAveragePooling2D(name='FC_avg')(x)
    elif FCpool=='GlobalMaxPooling2D':
        x = GlobalMaxPooling2D(name='FC_max')(x)
    # マルチタスクの全結合層+出力層構築
    predictions = []
    for i in range(num_classes):
        task_x = FC_0_5_layer(x, FCnum
                            , Dence_1, Dropout_1, addBatchNorm_1, kernel_initializer_1, l2_rate_1
                            , Dence_2, Dropout_2, addBatchNorm_2, kernel_initializer_2, l2_rate_2
                            , Dence_3, Dropout_3, addBatchNorm_3, kernel_initializer_3, l2_rate_3
                            , Dence_4, Dropout_4, addBatchNorm_4, kernel_initializer_4, l2_rate_4
                            , Dence_5, Dropout_5, addBatchNorm_5, kernel_initializer_5, l2_rate_5
                            , name='task'+str(i)+'_FC')
        task_x = Dense(1, activation=activation
                            , kernel_initializer=pred_kernel_initializer
                            , kernel_regularizer=regularizers.l2(pred_l2_rate)
                            , name='task'+str(i)+'_pred')(task_x)
        predictions.append(task_x)

    # 全結合層を削除したimagenetのtrainedモデルと構築した全結合層を結合
    model = Model(inputs=trained_model.input, outputs=predictions)

    if trainable == 'all':
        # 全レイヤーアンフリーズにする（全ての重みを再学習させる）
        for layer in model.layers:
            layer.trainable = True
    elif trainable == 0:
        # 全レイヤーフリーズ（重み学習させない）
        for layer in model.layers:
            # BatchNormalizationだけは重み学習 下山さんコードより
            if skip_bn and isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    else:
        # Fine-tuning 特定のレイヤまでfine-tuning
        for layer in model.layers[:trainable]:
            # BatchNormalizationだけは重み学習 下山さんコードより
            if skip_bn and isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        for layer in model.layers[trainable:]:
            layer.trainable = True

    # train layer 確認
    #for layer in model.layers:
    #    print(layer.trainable)
    # モデル情報表示
    #model.summary()
    # finetunning用にレイヤーの数と名前を表示
    #count= 0
    #for layer in model.layers:
    #    print(count, layer.name)
    #    count+=1
    # モデル描画
    #keras.utils.plot_model(model, to_file = os.path.join(output_dir, 'model.svg'))

    # マルチじゃないオリジナルのモデル確保 https://github.com/keras-team/keras/issues/8649
    orig_model = model
    if gpu_count > 1:
        # マルチGPU http://tech.wonderpla.net/entry/2018/01/09/110000
        model = multi_gpu_model(model, gpus=gpu_count)

    # モデルの構造保存
    save_architecture(orig_model, output_dir)

    return model, orig_model

def change_l2_softmax_net(model, alpha=16):
    """
    modelをL2 softmax network に変形する
    softmax関数に通す直前にL2ノルムで割って定数倍(alpha)するだけ。alphaはハイパーパラメータ
    参考
        https://copypaste-ds.hatenablog.com/entry/2019/03/01/164155
        https://medium.com/syncedreview/l2-constrained-softmax-loss-for-discriminative-face-verification-7cee8e6e9f8f
    """
    from keras.models import Model
    from keras.layers import Lambda
    from keras import backend as K

    x = model.layers[-2].output
    x = Lambda(lambda x: alpha*K.l2_normalize(x, axis=-1), name='l2_soft')(x) # L2ノルムで割って定数倍
    predictions = model.layers[-1](x)
    model = Model(inputs=model.input, outputs=predictions)
    return model

def load_json_weight(weight_file, architecture_file):
    """
    ファイルからモデルのネットワークと重みをロード
    オプティマイザの引数入れたくないのでコンパイルはしない
    """
    # モデルのネットワークロード
    model = model_from_json(open(architecture_file).read())
    # モデルの重みをロード
    model.load_weights(weight_file)
    return model

def load_model_file(weight_file, compile=False):
    """
    ファイルからモデルをロード
    オプティマイザの引数入れたくないのでコンパイルはしない
    """
    # モデルをロード
    model = load_model(weight_file, compile=compile)
    return model

if __name__ == '__main__':
    print('define_model.py: loaded as script file')
else:
    print('define_model.py: loaded as module file')
