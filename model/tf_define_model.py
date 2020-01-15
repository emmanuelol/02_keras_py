# -*- coding: utf-8 -*-
"""
imagenetの学習済みモデルをfine_tuningしてモデルを定義する

Usage:
    import os, sys
    current_dir = os.path.dirname(os.path.abspath("__file__"))
    path = os.path.join(current_dir, '../')
    sys.path.append(path)
    from model import define_model

    output_dir = r'output_test\100x100'
    img_rows,img_cols,channels = 100,100,3
    num_classes = 10
    chaice_model = 'EfficientNet'
    choice_optim = 'adam'

    # モデル作成
    model, orig_model = define_model.get_fine_tuning_model(output_dir, img_rows, img_cols, channels, num_classes, chaice_model, efficientnet_num=7)
    # オプティマイザ
    optim = define_model.get_optimizers(choice_optim)
    # モデルコンパイル
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
"""
import os, sys
import numpy as np

from tensorflow import keras

# pathlib でモジュールの絶対パスを取得 https://chaika.hatenablog.com/entry/2018/08/24/090000
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得

def save_architecture(model, output_dir):
    """モデルの構造保存"""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'architecture.json')
    json_string = model.to_json()
    open(json_path,"w").write(json_string)

def get_imagenet_model(output_dir:str, choice_model:str, img_rows:int, img_cols:int, channels=3, weights='imagenet', is_include_top=False, is_imagenet_model_save=True):
    """ VGG16などimagenetのモデル取得 """
    ## プロキシ apiproxy:8080 があるので下記の設定入れないとデータダウンロードでエラーになる
    #import urllib.request
    ## proxy の設定
    #proxy_support = urllib.request.ProxyHandler({'http' : 'http://apiproxy:8080', 'https': 'https://apiproxy:8080'})
    #opener = urllib.request.build_opener(proxy_support)
    #urllib.request.install_opener(opener)

    model_path = os.path.join(output_dir, choice_model+'.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        input_tensor = keras.layers.Input(shape=(img_rows, img_cols, channels))
        if choice_model.lower() == 'vgg16':# trainable == 15 img_rows=224, img_cols=224, channels=3
            model = keras.applications.vgg16.VGG16(weights=weights, include_top=is_include_top, input_tensor=input_tensor)

        elif choice_model.lower() == 'resnet50':# trainable == 164 img_rows=224, img_cols=224, channels=3
            model = keras.applications.resnet50.ResNet50(weights=weights, include_top=is_include_top, input_tensor=input_tensor)

        elif choice_model.lower() == 'resnet152v2':
            model = keras.applications.resnet_v2.ResNet152V2(weights=weights, include_top=is_include_top, input_tensor=input_tensor)

        elif choice_model.lower() == 'inceptionv3':# trainable == 249 img_rows=299, img_cols=299, channels=3
            model = keras.applications.inception_v3.InceptionV3(weights=weights, include_top=is_include_top, input_tensor=input_tensor)

        elif choice_model.lower() == 'xception':# trainable == 116 img_rows=299, img_cols=299, channels=3
            model = keras.applications.xception.Xception(weights=weights, include_top=is_include_top, input_tensor=input_tensor)

        elif choice_model.lower() == 'inceptionresnetv2': # trainable == 761 img_rows=299, img_cols=299, channels=3
            model = keras.applications.inception_resnet_v2.InceptionResNetV2(weights=weights, include_top=is_include_top, input_tensor=input_tensor)

        elif choice_model.lower() == 'densenet121':
            model = keras.applications.densenet.DenseNet121(weights=weights, include_top=is_include_top, input_tensor=input_tensor)

        elif choice_model.lower() == 'nasnetlarge':
            model = keras.applications.nasnet.NASNetLarge(weights=weights, include_top=is_include_top, input_tensor=input_tensor)

        elif choice_model.lower() == 'mobilenet':
            model = keras.applications.mobilenet.MobileNet(weights=weights, include_top=is_include_top, input_tensor=input_tensor)

        elif choice_model.lower() == 'mobilenetv2':
            model = keras.applications.mobilenet_v2.MobileNetV2(weights=weights, include_top=is_include_top, input_tensor=input_tensor)

        if is_imagenet_model_save:
            model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        keras_ver = str(keras.__version__).split('.')
        if (int(keras_ver[0]) == 2) & (int(keras_ver[1]) < 2): # kerasのバージョン判定
            # 古いkerasでMobileNetをロードするには、カスタムオブジェクト必要
            if choice_model.lower() == 'mobilenet':
                model = keras.models.load_model(model_path, custom_objects={'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D})
            elif choice_model.lower() == 'mobilenetv2':
                model = keras.models.load_model(model_path, custom_objects={'relu6': keras.applications.mobilenetv2.relu6})
        else:
            model = keras.models.load_model(model_path)

    return model

# ------------------------------------------ github model ------------------------------------------
def get_NASNetLarge_model(output_dir:str, img_rows=331, img_cols=331, channels=3, weights='imagenet', is_include_top=False, is_imagenet_model_save=True):
    """
    NASNetLarge_modelダウンロード及びロード
    入力層のサイズはオプション引数で指定可能。省略したらNASNetLargeのデフォルトの331*331*3になる
    """
    # githubのNasnetをimport
    # https://github.com/titu1994/Keras-NASNet/blob/master/README.md
    sys.path.append( str(current_dir) + '/../Git/Keras-NASNet' )
    from nasnet import NASNetLarge

    model_path = os.path.join(output_dir, 'NASNetLarge.h5py')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        input_tensor = keras.layers.Input(shape=(img_rows, img_cols, channels))
        model = NASNetLarge(weights=weights, include_top=is_include_top, input_tensor=input_tensor)#, pooling='avg')
        if is_imagenet_model_save:
            model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = keras.models.load_model(model_path)
    return model

def get_SENet_model(output_dir:str, choice_model:str, img_rows=224, img_cols=224, channels=3
                        , weights=None
                        , seresnet_num=154, sedensenet_num=169, seresnext_num=50
                        , is_include_top=False, is_model_save=False):
    """
    SENet_modelダウンロード及びロード
    ※SENetのモデルはweights='imagenet'としてもimagenetの重みファイルロードしない
    入力層のサイズはオプション引数で指定可能。省略したらResNetのデフォルトの224*224*3になる
    seresnet_num の数字でどれ使うか選ぶ. デフォルトは SEResNet154 にしておく
    sedensenet_num の数字でどれ使うか選ぶ. デフォルトはSEDenseNetImageNet169 にしておく
    seresnext_num の数字でどれ使うか選ぶ. デフォルトは SEResNeXt-50 にしておく
    """
    # githubのSenetをimport
    # https://github.com/titu1994/keras-squeeze-excite-network
    sys.path.append( str(current_dir) + '/../Git/keras-squeeze-excite-network' )
    import se_inception_v3, se_densenet, se_inception_resnet_v2, se_resnet, se_resnext, se

    model_path = os.path.join(output_dir, choice_model+'.h5py')
    os.makedirs(output_dir, exist_ok=True)
    input_tensor = keras.layers.Input(shape=(img_rows, img_cols, channels))
    if not os.path.exists(model_path):
        if choice_model.lower() == 'seresnet':
            print('seresnet_num =', seresnet_num)
            if seresnet_num == 18:
                model = se_resnet.SEResNet18(weights=weights, include_top=is_include_top, input_tensor=input_tensor)#, pooling='avg')
            elif seresnet_num == 34:
                model = se_resnet.SEResNet34(weights=weights, include_top=is_include_top, input_tensor=input_tensor)#, pooling='avg')
            elif seresnet_num == 50:
                model = se_resnet.SEResNet50(weights=weights, include_top=is_include_top, input_tensor=input_tensor)#, pooling='avg')
            elif seresnet_num == 101:
                model = se_resnet.SEResNet101(weights=weights, include_top=is_include_top, input_tensor=input_tensor)#, pooling='avg')
            elif seresnet_num == 154:
                model = se_resnet.SEResNet154(weights=weights, include_top=is_include_top, input_tensor=input_tensor)#, pooling='avg')

        if choice_model.lower() == 'seinceptionv3':
            model = se_inception_v3.SEInceptionV3(weights=weights, include_top=is_include_top, input_tensor=input_tensor)#, pooling='avg')

        if choice_model.lower() == 'seinceptionresnetv2':
            model = se_inception_resnet_v2.SEInceptionResNetV2(weights=weights, include_top=is_include_top, input_tensor=input_tensor)#, pooling='avg')

        if choice_model.lower() == 'sedensenetimagenet':
            print('sedensenet_num =', sedensenet_num)
            if sedensenet_num == 121:
                model = se_densenet.SEDenseNetImageNet121(weights=weights, include_top=is_include_top, input_tensor=input_tensor)
            elif sedensenet_num == 161:
                model = se_densenet.SEDenseNetImageNet161(weights=weights, include_top=is_include_top, input_tensor=input_tensor)
            elif sedensenet_num == 169:
                model = se_densenet.SEDenseNetImageNet169(weights=weights, include_top=is_include_top, input_tensor=input_tensor)
            elif sedensenet_num == 201:
                model = se_densenet.SEDenseNetImageNet201(weights=weights, include_top=is_include_top, input_tensor=input_tensor)
            elif sedensenet_num == 264:
                model = se_densenet.SEDenseNetImageNet264(weights=weights, include_top=is_include_top, input_tensor=input_tensor)

        if choice_model.lower() == 'seresnext':
            print('seresnext_num =', seresnext_num)
            if seresnext_num == 50:
                model = se_resnext.SEResNextImageNet(weights=weights, depth=[3, 4, 6, 3], include_top=is_include_top, input_tensor=input_tensor)
            elif seresnext_num == 101:
                model = se_resnext.SEResNextImageNet(weights=weights, depth=[3, 4, 23, 3], include_top=is_include_top, input_tensor=input_tensor)

        if is_model_save:
            model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = keras.models.load_model(model_path)

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
    # githubのWideResnetをimport
    # https://github.com/titu1994/Wide-Residual-Networks
    sys.path.append( str(current_dir) + '/../Git/Wide-Residual-Networks' )
    import wide_residual_network as wrn
    import wide_residual_network_include_top_false as wrn_top_false # 出力層消したの
    print('wrn_N, wrn_k =', wrn_N, wrn_k)
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
    # githubのWideResNet + OctConvをimport
    # https://qiita.com/koshian2/items/0e40a5930f1aa63a66b9
    sys.path.append( str(current_dir) + '/../Git/OctConv-TFKeras' )
    #import models as oct_wrn
    from oct_conv2d import OctConv2D # load_model(… custom_objects={'OctConv2D':OctConv2D}, compile=False) が必要
    import models_include_top_false as oct_wrn_top_false # 出力層消したの
    print('oct_conv_alpha, wrn_N, wrn_k =', alpha, wrn_N, wrn_k)
    model = oct_wrn_top_false.create_octconv_wide_resnet(alpha=alpha, N=wrn_N, k=wrn_k, input=(img_rows, img_cols, channels))
    return model

def get_EfficientNet_model(output_dir:str, input_shape=None, efficientnet_num=3, is_keras=False, weights='imagenet', is_include_top=False, is_imagenet_model_save=True):
    """
    EfficientNet_modelロード
    入力層のサイズはオプション引数で指定可能。省略したら各EfficientNetのデフォルトのサイズにする
    https://github.com/qubvel/efficientnet より
                                        imagenet val set
                                        @top1 acc
        EfficientNetB0 - (224, 224, 3)  0.772
        EfficientNetB1 - (240, 240, 3)  0.791
        EfficientNetB2 - (260, 260, 3)  0.802
        EfficientNetB3 - (300, 300, 3)  0.816
        EfficientNetB4 - (380, 380, 3)  0.830
        EfficientNetB5 - (456, 456, 3)  0.837
        EfficientNetB6 - (528, 528, 3)  0.841
        EfficientNetB7 - (600, 600, 3)  0.844
    """
    # githubのEfficientNetをimport
    sys.path.append( str(current_dir) + '/../Git/efficientnet' )
    if is_keras:# kerasはこっち
        from efficientnet.keras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
    else:# tfkerasはこっち
        from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
    model_path = os.path.join(output_dir, 'EfficientNetB'+str(efficientnet_num)+'.h5py')
    os.makedirs(output_dir, exist_ok=True)
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
    print('input_shape:', input_shape)
    # レイヤーに独自関数(efficientnet.model.conv_kernel_initializer)を使っているためモデルファイル(.h5py)がmodel.load()できない。model.load_weight()はできる
    # 毎回imagenetのモデルファイルダウンロードする必要あり
    if efficientnet_num == 0:
        model = EfficientNetB0(weights=weights, include_top=is_include_top, input_shape=input_shape)
    elif efficientnet_num == 1:
        model = EfficientNetB1(weights=weights, include_top=is_include_top, input_shape=input_shape)
    elif efficientnet_num == 2:
        model = EfficientNetB2(weights=weights, include_top=is_include_top, input_shape=input_shape)
    elif efficientnet_num == 3:
        model = EfficientNetB3(weights=weights, include_top=is_include_top, input_shape=input_shape)
    elif efficientnet_num == 4:
        model = EfficientNetB4(weights=weights, include_top=is_include_top, input_shape=input_shape)
    elif efficientnet_num == 5:
        model = EfficientNetB5(weights=weights, include_top=is_include_top, input_shape=input_shape)
    elif efficientnet_num == 6:
        model = EfficientNetB6(weights=weights, include_top=is_include_top, input_shape=input_shape)
    elif efficientnet_num == 7:
        model = EfficientNetB7(weights=weights, include_top=is_include_top, input_shape=input_shape)
    if is_imagenet_model_save:
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
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
    """# githubのPeleeNet
    sys.path.append( str(current_dir) + '/../Git/PeleeNet-Keras' )
    import pelee_net
    if include_top == True:
        model = pelee_net_keras.PeleeNet(input_shape=input_shape
                                         , use_stem_block=use_stem_block
                                         , n_classes=num_classes)
    else:
        model = pelee_net_keras.PeleeNet(input_shape=input_shape
                                         , use_stem_block=use_stem_block
                                         , include_top=include_top)
    return model

def get_attention_ptmodel(num_classes, activation, base_pretrained_model=None, base_model_trainable=False):
    """
    imagenetの学習済みモデルにattentionレイヤーつける
    https://www.kaggle.com/kmader/attention-inceptionv3-for-blindness/notebook
    Args:
        base_pretrained_model: 出力層なしモデル
                               from keras.applications.inception_v3 import InceptionV3 as PTModel
                               base_pretrained_model = PTModel(input_tensor=keras.layers.Input((299,299,3)), include_top=False, weights='imagenet')
                               とかでimagenetの学習済みモデルをimportしてPTModelを渡せばよい
        shape: 入力層のサイズ.[331, 331, 3]とか.imagenetの学習済みモデル使うからchanel=3でないとエラーになる
        num_classes: 出力層のサイズ.15とか
        activation: 出力層の活性化関数.'sigmoid'とか
        ptmodel_trainable: base_pretrained_modelの重み学習させるか.Trueなら学習させる
    Returns:
        imagenetの学習済みモデルのお尻にattentionレイヤーを付けたモデルオブジェクト
    Usage:
        num_classes = 10
        activation = 'softmax'
        model = define_model.get_EfficientNet_model("output_dir")
        retina_model = define_model.get_attention_ptmodel(num_classes, activation, base_pretrained_model=model, base_model_trainable=False)
    """
    if base_pretrained_model is None:
        in_lay = keras.layers.Input(shape=(299,299,3))
        base_pretrained_model = keras.applications.inception_v3.InceptionV3(input_tensor=in_lay, include_top=False, weights='imagenet')
        pt_features = base_pretrained_model(in_lay)
    else:
        shape = (base_pretrained_model.input_shape[1], base_pretrained_model.input_shape[2], base_pretrained_model.input_shape[3])
        in_lay = keras.layers.Input(shape=shape)
        pt_features = base_pretrained_model(in_lay)

    base_pretrained_model.trainable = base_model_trainable
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    bn_features = keras.layers.BatchNormalization()(pt_features)

    # here we do an attention mechanism to turn pixels in the GAP on an off
    attn_layer = keras.layers.Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(keras.layers.Dropout(0.5)(bn_features))
    attn_layer = keras.layers.Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
    attn_layer = keras.layers.Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
    attn_layer = keras.layers.Conv2D(1,
                        kernel_size = (1,1),
                        padding = 'valid',
                        activation = 'sigmoid')(attn_layer)
    # ↑predictでattentionレイヤー可視化したいときはこの Conv2D(1 の層のoutputを可視化する。
    # outputのshapeが(None, 14, 14, 1)になるので画像として可視化できる。
    # 詳細は https://www.kaggle.com/kmader/attention-inceptionv3-for-blindness/notebook 確認すること

    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = keras.layers.Conv2D(pt_depth, kernel_size = (1,1), padding = 'same',
                   activation = 'linear', use_bias = False, weights = [up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    mask_features = keras.layers.multiply([attn_layer, bn_features])
    gap_features = keras.layers.GlobalAveragePooling2D()(mask_features)
    gap_mask = keras.layers.GlobalAveragePooling2D()(attn_layer)

    # to account for missing values from the attention model
    gap = keras.layers.Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
    gap_dr = keras.layers.Dropout(0.25)(gap)
    dr_steps = keras.layers.Dropout(0.25)(keras.layers.Dense(128, activation = 'relu')(gap_dr))
    out_layer = keras.layers.Dense(num_classes, activation = activation)(dr_steps)
    retina_model = keras.models.Model(inputs = [in_lay], outputs = [out_layer])
    #retina_model.summary()

    return retina_model

def show_attention_layer(retina_model, Xs:np.ndarray, ys=np.array([]), output_dir=None):
    """
    attention layerのoutputを可視化して画像ファイルに保存する
    https://www.kaggle.com/kmader/attention-inceptionv3-for-blindness/notebook
    Args;
        retina_model:attention layer付けたモデル
        Xs:4次元テンソルの入力画像複数
        ys:Xsに対応するラベル
        output_dir:attention map画像保存先ディレクトリ
    Usage:
        num_classes = 10
        activation = 'softmax'
        model = define_model.get_EfficientNet_model("output_dir")
        retina_model = define_model.get_attention_ptmodel(num_classes, activation, base_pretrained_model=model, base_model_trainable=False)
        from transformer import get_train_valid_test
        jpg = r'horse.jpg'
        X = get_train_valid_test.load_one_img(jpg, 100, 100)
        y = np.array([7])
        define_model.show_attention_layer(retina_model, X, y, output_dir=r'output_test\100x100')
    """
    import matplotlib.pyplot as plt

    # get the attention layer since it is the only one with a single output dim
    for attn_layer in retina_model.layers:
        c_shape = attn_layer.get_output_shape_at(0)
        if len(c_shape)==4:
            if c_shape[-1]==1:
                print(attn_layer)
                break
    # Xsからランダムに6枚だけ可視化する
    rand_idx = np.random.choice(range(len(Xs)), size = 6)
    attn_func = keras.backend.function(inputs = [retina_model.get_input_at(0), keras.backend.learning_phase()],
               outputs = [attn_layer.get_output_at(0)]
              )
    if len(Xs) < 6:
        figsize = (4, 16)
    else:
        figsize = (8, 4*len(rand_idx))
    fig, m_axs = plt.subplots(len(rand_idx), 2, figsize = figsize)
    [c_ax.axis('off') for c_ax in m_axs.flatten()]

    count = 0
    for c_idx, (img_ax, attn_ax) in zip(rand_idx, m_axs):
        cur_img = Xs[c_idx:(c_idx+1)]
        attn_img = attn_func([cur_img, 0])[0]
        img_ax.imshow(np.clip(cur_img[0,:,:,:]*127+127, 0, 255).astype(np.uint8))
        attn_ax.imshow(attn_img[0, :, :, 0]/attn_img[0, :, :, 0].max(), cmap = 'viridis',
                       vmin = 0, vmax = 1,
                       interpolation = 'lanczos')
        if ys.shape != (0,):
            real_cat = ys[c_idx]
            img_ax.set_title('Cat:%2d' % (real_cat))
        pred_cat = retina_model.predict(cur_img)
        if ys.shape != (0,):
            attn_ax.set_title('Attention Map\nPred:%2.2f%%' % (100*pred_cat[0,int(real_cat)]))
        else:
            pred_cat = [str((p*100).round(1)) for p in pred_cat[0]]
            #pred_cat = ', '.join(pred_cat)
            attn_ax.set_title('Attention Map\nPred%:'+str(pred_cat))

        count += 1
        if len(Xs) <= count:
            break

    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, 'attention_map.png'), dpi = 300)
# --------------------------------------------------------------------------------------------------

# ------------------------------------------ github optimizer ------------------------------------------
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
    # githubのAdaBoundをimport
    # https://github.com/titu1994/keras-adabound
    sys.path.append( str(current_dir) + '/../Git/keras-adabound' )
    from adabound import AdaBound
    if 'adabound' in sys.modules.keys():
        return AdaBound(lr=lr, final_lr=final_lr, beta_1=beta_1, beta_2=beta_2, gamma=gamma, epsilon=epsilon, decay=decay, amsbound=amsbound, weight_decay=weight_decay)
    else:
        print("It can not be imported because there is no adabound library.Please obtain the adabound library from 'https://github.com/titu1994/keras-adabound'.")

def get_radam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
              epsilon=None, decay=0., weight_decay=0., amsgrad=False,
              total_steps=0, warmup_proportion=0.1, min_lr=0.):
    """
    githubのRAdamをimport
    https://github.com/CyberZHG/keras-radam
    ※Rectified Adam（RAdam）: Adamの改良版optimaizer. Adam 学習開始時の LR 調整（ウォームアップ）を自動化する
                              初期学習率の低いウォームアップ（最初の小さいlrからはじめて数epochはlr上げる手法）を適用
                              入力バッチの最初の数セットのmomentumをオフにする
                              参考：https://nykergoto.hatenablog.jp/entry/2019/08/16/Adam_%E3%81%AE%E5%AD%A6%E7%BF%92%E4%BF%82%E6%95%B0%E3%81%AE%E5%88%86%E6%95%A3%E3%82%92%E8%80%83%E3%81%88%E3%81%9F%E8%AB%96%E6%96%87_RAdam_%E3%82%92%E8%AA%AD%E3%82%93%E3%81%A0%E3%82%88%21

    コンパイル済みモデルでロードする場合はcustom_objectsが必要
    from keras_radam import RAdam
    model = keras.models.load_model(os.path.join(output_dir, 'finetuning.h5'), custom_objects={'RAdam':RAdam})
    """
    # githubのRAdamをimport
    # https://github.com/CyberZHG/keras-radam
    sys.path.append( str(current_dir) + '/../Git/keras-radam' )
    from keras_radam import RAdam
    return RAdam(learning_rate=learning_rate, decay=decay, weight_decay=weight_decay, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, total_steps=total_steps, warmup_proportion=warmup_proportion, min_lr=min_lr)
# --------------------------------------------------------------------------------------------------

def get_optimizers(choice_optim='sgd', lr=0.0, decay=0.0
                   , momentum=0.9, nesterov=True # SGD
                   , rmsprop_rho=0.9#, epsilon=None # RMSprop
                   , adadelta_rho=0.95 # Adadelta
                   , beta_1=0.9, beta_2=0.999, amsgrad=False # Adam, Adamax, Nadam
                   , total_steps=0, warmup_proportion=0.1, min_lr=0 # RAdam
                   , *args, **kwargs
                  ):
    """
    オプティマイザを取得する
    引数のchoice_optim は 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam', 'adabound', 'radam' のいずれかを指定する
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
    elif choice_optim == 'radam':
        if lr == 0.0:
            lr=0.001
        optim = get_radam(learning_rate=lr, decay=decay, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, total_steps=total_steps, warmup_proportion=warmup_proportion, min_lr=min_lr)
        print('radam_lr radam_decay beta_1 beta_2, amsgrad total_steps warmup_proportion min_lr =', lr, decay, beta_1, beta_2, amsgrad, total_steps, warmup_proportion, min_lr)
    return optim

def FC_batch_drop(x, activation='relu', dence=1024, dropout_rate=0.5, is_add_batchnorm=False, kernel_initializer='he_normal', l2_rate=1e-4, name=None):
    """
    中間層の全結合1層（BatchNormalizationとDropout追加可能）
    Args:
        x: model.output
        activation: 活性化関数。デフォルトは'relu'
        dence, dropout_rate: 各層のnode数。デフォルトは1024としてる
        dropout_rate: dropout_rate(0<dropout_rate<1 の範囲出ないとエラーになる)。デフォルトは0.5としてる
        is_add_batchnorm: BatchNormalization いれるか。デフォルトは入れない
        kernel_initializer:Denseレイヤーの重みの初期化 デフォルトをHe の正規分布('he_normal')としてる。
                           Denseのデフォルトglorot_uniform（Glorot（Xavier）の一様分布）にしたい場合は 'glorot_uniform' とすること
        l2_rate:l2正則化によるWeight decay入れるか。デフォルトは無し(1e-4)。kerasのマニュアルでは0.01とか使ってる。Deeptoxの4層では1e-04, 1e-05, 1e-06, 0にしてる
        name: 層の名前
    Returns:
        全結合1層(x)
    """
    x = keras.layers.Dense(dence, kernel_initializer=kernel_initializer, kernel_regularizer=keras.regularizers.l2(l2_rate), name=name+'_dence')(x)
    if is_add_batchnorm == True:
        x = keras.layers.BatchNormalization(name=name+'_batchNormalization')(x)
    x = keras.layers.Activation(activation, name=name+'_act')(x) # BatchNormalizationはDenseと活性化の間
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name=name+'_dropout')(x)
    print('dence dropout is_add_batchnorm kernel_initializer l2_rate =', dence, dropout_rate, is_add_batchnorm, kernel_initializer, l2_rate)
    return x

def get_fine_tuning_model(output_dir, img_rows, img_cols, channels, num_classes, choice_model
                            , trainable='all'
                            , fcpool='GlobalAveragePooling2D'
                            , fcs=[], drop=0.5, is_add_batchnorm=None, kernel_init='he_normal', l2_rate=1e-4
                            , pred_kernel_initializer='zeros', pred_l2_rate=1e-4
                            , activation='softmax'#'sigmoid'
                            , gpu_count=1
                            , skip_bn=True
                            , seresnet_num=154 # SEResNet の種類指定 18,34,50,101,154 のいずれかしかだめ
                            , sedensenet_num=169 # SEDenseNet の種類指定 121,161,169,201,264 のいずれかしかだめ
                            , seresnext_num=50 # SEResNext の種類指定 50,101 のいずれかしかだめ
                            , wrn_N=4, wrn_k=10 # WideResNetの引数
                            , oct_conv_alpha=0.25 # OctConv_WideResNet の低周波と高周波のチャンネル数の割り振り具合であるα
                            , efficientnet_num=3 # EfficientNet の種類指定 0,1,2,3,4,5,6,7 のいずれかしかだめ
                            , is_keras=True # EfficientNet keras版を使うか
                            , is_base_model_trainable=True # attentionモデルのベースモデルの重み更新するか
                            , n_multitask=1, multitask_pred_n_node=2
                            , is_imagenet_model_save=True
                            , weights='imagenet'
                            , *args, **kwargs
                            ):
    """
    fine-tuningなど設定したモデルを返す
    オプティマイザの引数入れたくないのでコンパイルはしない
    マルチGPU対応あり gpu_count>1 なら return でマルチじゃないオリジナルのモデルとマルチGPUモデルを返す
    Args:
        output_dir:出力ディレクトリ
        img_rows, img_cols, channels:モデルの入力サイズ
        num_classes:クラス数
        choice_model:fine-tuningするVGG16などimagenetのモデル名
        trainable:どの層番号までパラメータフリーズするか。'all'なら全層更新
        fcpool:全結合層の直前に入れるpooling
        fcs:全結合層のnode数リスト
        drop:全結合層のdropout rate
        is_add_batchnorm:全結合層にBatchNormalization いれるか
        kernel_init:全結合層の重み初期値
        l2_rate:全結合層のl2正則化
        pred_kernel_initializer:出力層の重み初期値
        pred_l2_rate:出力層のl2正則化
        activation:出力層の活性化関数
        gpu_count:使うGPUの数。2以上ならマルチGPUでtrain
        skip_bn:BatchNormalizationだけは必ず重み更新するか
        n_multitask:タスク数。2以上なら全結合層を分岐させてマルチタスクにする
        multitask_pred_n_node:マルチタスクの各タスクの出力層のnode数
        is_imagenet_model_save:imagenetのmodelファイル保存するか
    Returns
        model:マルチGPU用モデルオブジェクト。引数のgpu_count=1ならorig_modelと同じもの
        orig_model:シングルGPU用モデルオブジェクト
    """
    print('----- model_param -----')
    print('output_dir =', output_dir)
    print('img_rows img_cols channels =', img_rows, img_cols, channels)
    print('num_classes =', num_classes)
    print('choice_model trainable =', choice_model, trainable)
    print('fcs =', str(fcs))
    print('fcpool =', fcpool)
    print('pred_kernel_initializer pred_l2_rate =', pred_kernel_initializer, pred_l2_rate)
    print('activation =', activation)
    print('gpu_count =', gpu_count)
    print('skip_bn =', skip_bn)
    print('n_multitask =', n_multitask)

    # imagenetモデル
    if choice_model in ['VGG16', 'ResNet50', 'ResNet152V2', 'InceptionV3', 'Xception', 'DenseNet121', 'InceptionResNetV2', 'MobileNet', 'MobileNetV2', 'NASNetLarge']:
        trained_model = get_imagenet_model(output_dir, choice_model, img_rows, img_cols, weights=weights, channels=channels
                                            , is_imagenet_model_save=is_imagenet_model_save)
    elif choice_model == 'SEResNet':
        trained_model = get_SENet_model(output_dir, choice_model, img_rows=img_rows, img_cols=img_cols, weights=weights, channels=channels
                                        , seresnet_num=seresnet_num, sedensenet_num=sedensenet_num, seresnext_num=seresnext_num
                                        , is_model_save=is_imagenet_model_save)
    elif choice_model == 'WideResNet':
        trained_model = get_WideResNet_model(img_rows=img_rows, img_cols=img_cols, channels=channels
                                            , num_classes=num_classes
                                            , wrn_N=wrn_N, wrn_k=wrn_k, include_top=False)
    elif choice_model == 'OctConv_WideResNet':
        trained_model = get_OctConv_WideResNet_model(alpha=oct_conv_alpha, img_rows=img_rows, img_cols=img_cols, channels=channels
                                                    , wrn_N=wrn_N, wrn_k=wrn_k)
    elif choice_model == 'EfficientNet':
        if (img_rows is None) and (img_cols is None) and (channels is None):
            trained_model = get_EfficientNet_model(output_dir, input_shape=None, weights=weights
                                                    , efficientnet_num=efficientnet_num, is_imagenet_model_save=is_imagenet_model_save)
        else:
            trained_model = get_EfficientNet_model(output_dir, input_shape=(img_rows,img_cols,channels)
                                                    , efficientnet_num=efficientnet_num, is_imagenet_model_save=is_imagenet_model_save)
    elif choice_model == 'PeleeNet':
            trained_model = get_Pelee_net(input_shape=(img_rows,img_cols,channels), include_top=False)
    #print(trained_model.summary())

    # attensionレイヤー付けるor全結合多層にするか
    if fcpool=='attention':
        model = get_attention_ptmodel(num_classes, activation, base_pretrained_model=trained_model, base_model_trainable=is_base_model_trainable)
    else:
        x = trained_model.output
        # 学習済みモデルのpooling指定
        if fcpool=='GlobalAveragePooling2D':
            x = keras.layers.GlobalAveragePooling2D(name='FC_avg')(x)
        elif fcpool=='GlobalMaxPooling2D':
            x = keras.layers.GlobalMaxPooling2D(name='FC_max')(x)
        print('----- FC_layers -----')
        if n_multitask == 1:
            # マルチクラス/マルチラベルの全結合層+出力層
            # 全結合層
            for i, den in enumerate(fcs):
                x = FC_batch_drop(x, activation='relu'
                                , dence=den
                                , dropout_rate=drop
                                , is_add_batchnorm=is_add_batchnorm
                                , kernel_initializer=kernel_init
                                , l2_rate=l2_rate
                                , name='FC'+str(i))
            # 出力層
            predictions = keras.layers.Dense(num_classes, activation=activation
                                , kernel_initializer=pred_kernel_initializer
                                , kernel_regularizer=keras.regularizers.l2(pred_l2_rate)
                                , name='pred')(x)
        else:
            # マルチタスクの全結合層+出力層
            predictions = []
            for n_task in range(n_multitask):
                task_x = x
                # 全結合層
                for i, den in enumerate(fcs):
                    task_x = FC_batch_drop(task_x, activation='relu'
                                            , dence=den
                                            , dropout_rate=drop
                                            , is_add_batchnorm=is_add_batchnorm
                                            , kernel_initializer=kernel_init
                                            , l2_rate=l2_rate
                                            , name='task'+str(n_task)+'_FC'+str(i))
                # 出力層
                task_x = keras.layers.Dense(multitask_pred_n_node, activation=activation
                                , kernel_initializer=pred_kernel_initializer
                                , kernel_regularizer=keras.regularizers.l2(pred_l2_rate)
                                , name='task'+str(n_task)+'_pred')(task_x)
                predictions.append(task_x)

        model = keras.models.Model(inputs=trained_model.input, outputs=predictions)

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
    #keras.utils.plot_model(model, to_file = os.path.join(output_dir, 'model.svg'), show_shapes=True, show_layer_names=True)

    # マルチじゃないオリジナルのモデル確保 https://github.com/keras-team/keras/issues/8649
    orig_model = model
    if gpu_count > 1:
        # マルチGPU http://tech.wonderpla.net/entry/2018/01/09/110000
        model = keras.utils.multi_gpu_model(model, gpus=gpu_count)

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
    x = model.layers[-2].output
    x = keras.layers.Lambda(lambda x: alpha*keras.backend.l2_normalize(x, axis=-1), name='l2_soft')(x) # L2ノルムで割って定数倍
    predictions = model.layers[-1](x)
    model = keras.models.Model(inputs=model.input, outputs=predictions)
    return model

def load_json_weight(weight_file, architecture_file):
    """
    ファイルからモデルのネットワークと重みをロード
    オプティマイザの引数入れたくないのでコンパイルはしない
    """
    # モデルのネットワークロード
    model = keras.models.model_from_json(open(architecture_file).read())
    # モデルの重みをロード
    model.load_weights(weight_file)
    return model

def load_model_file(weight_file, compile=False):
    """
    ファイルからモデルをロード
    オプティマイザの引数入れたくないのでコンパイルはしない
    """
    return keras.models.load_model(weight_file, compile=compile)

def print_model_summary(model):
    """
    modelのサマリーとmodelのレイヤー名とid番号をprintする
    Args:
        model:モデルオブジェクト
    """
    model.summary()
    # 各レイヤーのid, 名前, 重み更新するか をprint
    print('<id> <layer.name> <layer.trainable>' )
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
        w = model.layers[i].get_weights()
        # cnnカーネル（フィルター）のサイズもprint
        if len(w) == 2:
            weights, bias = w
            print('    weights.shape:'+str(weights.shape))
