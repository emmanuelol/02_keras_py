# -*- coding: utf-8 -*-
"""
パラメータを引数に持たせて optuna(ハイパーパラメータ自動最適化ツール)でパラメータチューニングを実行する
(aaa00162ユーザのpy36環境でしか optuna 実行できない)

以下でパラメータチューニング実行。引数のn_trials は試行回数
    import optuna
    import optuna_train
    objective = optuna_train.Objective(out_dir, d_cls)# d_cls: データ管理クラス get_train_valid_test.LabeledDataset
    study = optuna.create_study()
    study.optimize(objective, n_trials=1)

参考:
optunaのドキュメント: https://optuna.readthedocs.io/en/stable/index.html
optunaの関数メモ：
    trial.suggest_categorical('name', [a,b,c]): リストのa,b,cのいずれかを選択（リストの値は固定値でない（trialで出した値とか）だとエラーになることがある）
        ↓このようなのはエラーになる
        den_2 = int(trial.suggest_categorical('Dence_2', [den, den//2, den//4, den//8]))
    trial.suggest_discrete_uniform('name', 0.1, 1.0, 0.1): 0.1-1.0までの範囲で0.1刻みで値選択（最小値を先に設定しないと機能しない!!!）
    trial.suggest_int('name', 3, 7): 3-7のいずれかをint型で選択
    trial.suggest_uniform('name', 0, 100): 0-100のいずれかを選択
"""
import os, sys
import keras
import keras.backend as K
import copy
import uuid # UUIDはPython組み込みのUUIDモジュール. ランダムなUUIDは簡単に発生させることができる
import shutil

# 自作モジュール
#current_dir = os.path.dirname(os.path.abspath("__file__"))
#path = os.path.join(current_dir, '../')
#sys.path.append(path)
sys.path.append(r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py\transformer')
sys.path.append(r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py\model')
from transformer import get_train_valid_test
from model import define_model, multi_loss, my_callback

# Random Erasing+mixup
#sys.path.append(r'/home/tmp10014/jupyterhub/notebook/other/lib_DL/mixup-generator')
sys.path.append(r'C:\Users\shingo\Git\mixup-generator')
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser

# optuna
#sys.path.append(r'/home/aaa00162/jupyterhub/notebook/other/lib_DL/optuna-master')
sys.path.append(r'C:\Users\shingo\Git\optuna')
import optuna


class OptunaCallback(keras.callbacks.Callback):
    """
    Kerasでの枝刈り（最終的な結果がどのぐらいうまくいきそうかを大まかに予測し、良い結果を残すことが見込まれない試行は、最後まで行うことなく早期終了）
    https://qiita.com/koshian2/items/107c386f81c9bb7f8df3
    """
    def __init__(self, trial):
        self.trial = trial

    def on_epoch_end(self, epoch, logs):
        current_val_error = logs["val_loss"]# 1.0 - logs["val_acc"]
        self.trial.report(current_val_error, step=epoch)
        # 打ち切り判定
        if self.trial.should_prune(epoch):
            raise optuna.structs.TrialPruned()


class Objective(object):
    """
    学習パラメータを引数に持たせてoptunaの目的関数を定義するクラス
    参考：https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments

    ### チューニング可能パラメータ ###
    ■ モデル（ニューラルネットワーク）
        - Imagenetの学習済みモデル
        - fine-tuning
        - 全結合0-5層（重みの初期値はhe_normal(He の正規分布)で固定）
            - ユニット数 (同じ値か層ごとに減らす)
            - dropout_rate (全層同じ値になる)
            - Batch_Normalization (全層同じ値になる)
            - l2正則化(weight decay) (全層同じ値になる)
    ■ オプティマイザ
    ■ 学習率
        - 学習率変更なし
        - cosine_annealing(factor=0.01, epochs=None)
        - LearningRateScheduler(lr* 1/4 を3回する)
    ■ データ水増し( keras.preprocessing.image.ImageDataGenerator )
        - 画像の剪断(shear)
        - 拡大縮小(zoom)
        - 回転(rotation)
        - 上下反転(vertical_flip)
        - 左右反転(horizontal_flip)
        - ランダムに画素値に値を足す(channel_shift_range)
        - ランダムに明度を変更(brightness_range)
        - 画像の一部矩形領域を隠す(random_erasing)
        - 画像混ぜる(mix_up)
    """
    def __init__(self
                  , out_dir # 出力ディレクトリ
                  , d_cls # データ管理クラス get_train_valid_test.LabeledDataset
                  , train_data_dir, validation_data_dir=None, test_dir=None # train/valid/test set の画像ディレクトリ
                  , shape=[331,331,3] # 入力層のサイズ
                  , num_classes=12 # クラス数
                  , class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'] # クラス名
                  , epochs=30 # エポック数
                  , gpu_count=1 # GPUの数
                  , loss=multi_loss.build_masked_loss(K.binary_crossentropy) # 損失関数
                  , metrics=['accuracy', 'binary_accuracy', multi_loss.masked_accuracy] # model.fit_generator()で使うメトリック
                  , verbose=0 # model.fit_generator()でログ出すか.0なら出さない.2ならエポックごとにログ出す.1はstepごとに出すためログが膨大になるので使わない
                  , activation='sigmoid' # 出力層の活性化関数
                  , pred_kernel_initializer='zeros' # 出力層の初期値
                  , pred_l2_rate=1e-4 # 出力層のl2
                  , FCpool='GlobalAveragePooling2D' # FC層のpooling
                  , return_val_loss=True # best_parameter をval_loss でとるか.False にしたらval_metrics[0] の指標でbest_parameter を返す
                  , callbacks=[] # コールバックのリスト
                  , callback_save_model=False # モデル保存するcallback つけるか. False なら保存しない
                  , choice_model=['VGG16','ResNet50','InceptionV3','Xception','InceptionResNetV2','NASNetLarge','SEResNet','SEInceptionV3','SEInceptionResNetV2','SEDenseNet' ,'SEResNext'] # Fine-tuning する学習済みモデル
                  , trainable=['all', 249] # 重み全層学習させるか（重みunfreeze開始レイヤーを番号で指定できる）
                  , FCnum=[0,1,2,3,4,5] # FC層の数
                  , Dence=[1024, 512, 256, 128] # FC層のユニット数
                  , Dropout=[0.0, 0.5, 0.7] # FC層のDropout
                  , addBatchNorm=[None, 'add'] # FC層のBatchNorm
                  , l2=[0.0, 1e-6, 1e-4, 1e-2] # FC層のl2
                  , choice_optim=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'] # optimizer
                  , lr=[0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0] # （初期）学習率
                  , callback_lr=[None, 'cosine_annealing', 'LearningRateScheduler'] # 学習率変更するcallback
                  , cosine_annealing_epochs=None # コサインアニーリングの半周期のエポック数. None なら学習率下がるだけ
                  , rescale=1.0/255.0 # 画像の前処理
                  , horizontal_flip=[True, False] # 画像の左右反転
                  , vertical_flip=[True, False] # 画像の上下反転
                  , rotation_range_min=0, rotation_range_max=30, rotation_unit=10 # 画像の回転の下限角、回転の上限角、回転角の刻み幅. 上限下限はAutoaugment の論文の値. 上限下限はAutoaugment の論文の値(-30-30)
                  , zoom_range_low_min=0.1, zoom_range_low_max=1.0, zoom_range_high_min=1.0, zoom_range_high_max=1.9, zoom_range_unit=0.1 # 画像の縮小の最少-最大倍率、拡大の最少-最大倍率、縮小拡大倍率の刻み幅. 上限下限はAutoaugment の論文の値(0.1-1.9)
                  , shear_range_min=0.0, shear_range_max=0.3, shear_range_unit=0.1 # 画像のせん断の最少倍率、せん断の最大倍率、せん断倍率の刻み幅. 上限下限はAutoaugment の論文の値(-0.3-0.3)
                  , channel_shift_range_min=0.0, channel_shift_range_max=10.0, channel_shift_range_unit=0.1 # channel_shift_range=5. とした場合、[-5.0, 5.0] の範囲でランダムに画素値に値を足す
                  , brightness_range_low_min=1.0, brightness_range_low_max=1.0, brightness_range_high_min=1.0, brightness_range_high_max=1.0, brightness_range_unit=0.1 # ランダムに明度を変更 brightness_range=[0.3, 1.0]みたいなの
                  , branch_Tox21_12task=[False] # Tox21用.12出力（task）をだすgeneratorにするか. False なら出力層12taskに分ける分岐なし
                  , seresnet_num=[154] # SEResNet の種類指定 18,34,50,101,154 のいずれかしかだめ
                  , sedensenet_num=[169]  # SEDenseNet の種類指定 121,161,169,201,264 のいずれかしかだめ
                  , seresnext_num=[50] # SEResNext の種類指定 50,101 のいずれかしかだめ
                  , add_se=[False] # FC層の前にSE block つけるか
                  , random_crop=None # 画像のrandom_crop.付ける場合は[224,224]とかにする
                  , random_erasing_prob=[0.0] # random_erasing の確率. 使わない場合は0.0にする
                  , random_erasing_maxpixel=255.0 # random_erasing で消す領域の画素の最大値
                  , mix_up_alpha=[0.0] # mixup 含めるか. 使わない場合は0.0にする。使う場合は0.2とか
                 ):
        # define_model.get_fine_tuning_model(), model.fit_generator() で使う引数
        self.out_dir = out_dir
        self.d_cls = d_cls
        self.shape = shape
        self.num_classes = num_classes
        self.class_name = class_name
        self.epochs = epochs
        self.gpu_count = gpu_count
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose
        self.return_val_loss = return_val_loss
        self.train_data_dir=train_data_dir
        self.validation_data_dir = validation_data_dir
        self.test_dir = test_dir
        # ハイパーパラメータの引数
        self.callbacks = callbacks
        self.callback_save_model = callback_save_model
        self.activation = activation
        self.pred_kernel_initializer = pred_kernel_initializer
        self.pred_l2_rate = pred_l2_rate
        self.FCpool = FCpool
        self.choice_model = choice_model
        self.trainable = trainable
        self.FCnum = FCnum
        self.Dence = Dence
        self.Dropout = Dropout
        self.addBatchNorm = addBatchNorm
        self.l2 = l2
        self.choice_optim = choice_optim
        self.lr = lr
        self.callback_lr = callback_lr
        self.cosine_annealing_epochs = cosine_annealing_epochs
        self.rescale = rescale
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range_min = rotation_range_min
        self.rotation_range_max = rotation_range_max
        self.rotation_unit = rotation_unit # 0.0 にしたら rotation_range_max で固定
        self.zoom_range_low_min = zoom_range_low_min
        self.zoom_range_low_max = zoom_range_low_max
        self.zoom_range_high_min = zoom_range_high_min
        self.zoom_range_high_max = zoom_range_high_max
        self.zoom_range_unit = zoom_range_unit # 0.0 にしたら zoom_range_low_min, zoom_range_high_max で固定
        self.shear_range_min = shear_range_min
        self.shear_range_max = shear_range_max
        self.shear_range_unit = shear_range_unit # 0.0 にしたら shear_range_max で固定
        self.channel_shift_range_min = channel_shift_range_min
        self.channel_shift_range_max = channel_shift_range_max
        self.channel_shift_range_unit = channel_shift_range_unit
        self.brightness_range_low_min = brightness_range_low_min
        self.brightness_range_low_max = brightness_range_low_max
        self.brightness_range_high_min = brightness_range_high_min
        self.brightness_range_high_max = brightness_range_high_max
        self.brightness_range_unit = brightness_range_unit # 0.0 にしたら brightness_range_low_min, brightness_range_high_max で固定
        self.random_crop = random_crop
        self.random_erasing_prob = random_erasing_prob
        self.random_erasing_maxpixel = random_erasing_maxpixel
        self.mix_up_alpha = mix_up_alpha
        self.branch_Tox21_12task = branch_Tox21_12task
        self.seresnet_num = seresnet_num
        self.sedensenet_num = sedensenet_num
        self.seresnext_num = seresnext_num
        self.add_se = add_se

    # ============================================================ model ============================================================
    def _optuna_model(self, trial, branch_Tox21_12task):
        """
        trialでmodelにパラメータセットする
        """
        # Fine-tuning する学習済みモデル
        choice_model = trial.suggest_categorical('choice_model', self.choice_model)
        # 重み全層学習させるか（重みunfreeze開始レイヤーを番号で指定できる）
        trainable = trial.suggest_categorical('trainable', self.trainable)
        # FC層の数
        FCnum = int(trial.suggest_categorical('FCnum', self.FCnum))
        # FC層のユニット数、Dropout、BatchNorm、l2
        if FCnum == 0:
            den_1, den_2, den_3, den_4, den_5 = 0, 0, 0, 0, 0
            drop_1, drop_2, drop_3, drop_4, drop_5 = 0, 0, 0, 0, 0
            batchnorm_1, batchnorm_2, batchnorm_3, batchnorm_4, batchnorm_5 = 0, 0, 0, 0, 0
            l2_1, l2_2, l2_3, l2_4, l2_5 = 0, 0, 0, 0, 0
        else:
            den_1 = int(trial.suggest_categorical('Dence_1', self.Dence))
            drop_1 = float(trial.suggest_categorical('Dropout', self.Dropout))
            batchnorm_1 = trial.suggest_categorical('addBatchNorm', self.addBatchNorm)
            l2_1 = float(trial.suggest_categorical('l2', self.l2))
            if FCnum == 1:
                den_2, den_3, den_4, den_5 = 0, 0, 0, 0
                drop_2, drop_3, drop_4, drop_5 = 0, 0, 0, 0
                batchnorm_2, batchnorm_3, batchnorm_4, batchnorm_5 = None, None, None, None
                l2_2, l2_3, l2_4, l2_5 = 0, 0, 0, 0
            elif FCnum == 2:
                den_2 = int(trial.suggest_discrete_uniform("Dence_2", den_1//2, den_1, den_1//4))
                den_3, den_4, den_5 = 0, 0, 0
                drop_2, drop_3, drop_4, drop_5 = drop_1, 0, 0, 0
                batchnorm_2, batchnorm_3, batchnorm_4, batchnorm_5 = batchnorm_1, None, None, None
                l2_2, l2_3, l2_4, l2_5 = l2_1, 0, 0, 0
            elif FCnum == 3:
                den_2 = int(trial.suggest_discrete_uniform("Dence_2", den_1//2, den_1, den_1//4))
                den_3 = int(trial.suggest_discrete_uniform("Dence_3", den_2//2, den_2, den_2//4))
                den_4, den_5 = 0, 0
                drop_2, drop_3, drop_4, drop_5 = drop_1, drop_1, 0, 0
                batchnorm_2, batchnorm_3, batchnorm_4, batchnorm_5 = batchnorm_1, batchnorm_1, None, None
                l2_2, l2_3, l2_4, l2_5 = l2_1, l2_1, 0, 0
            elif FCnum == 4:
                den_2 = int(trial.suggest_discrete_uniform("Dence_2", den_1//2, den_1, den_1//4))
                den_3 = int(trial.suggest_discrete_uniform("Dence_3", den_2//2, den_2, den_2//4))
                den_4 = int(trial.suggest_discrete_uniform("Dence_4", den_3//2, den_3, den_3//4))
                den_5 = 0
                drop_2, drop_3, drop_4, drop_5 = drop_1, drop_1, drop_1, 0
                batchnorm_2, batchnorm_3, batchnorm_4, batchnorm_5 = batchnorm_1, batchnorm_1, batchnorm_1, None
                l2_2, l2_3, l2_4, l2_5 = l2_1, l2_1, l2_1, 0
            elif FCnum == 5:
                den_2 = int(trial.suggest_discrete_uniform("Dence_2", den_1//2, den_1, den_1//4))
                den_3 = int(trial.suggest_discrete_uniform("Dence_3", den_2//2, den_2, den_2//4))
                den_4 = int(trial.suggest_discrete_uniform("Dence_4", den_3//2, den_3, den_3//4))
                den_5 = int(trial.suggest_discrete_uniform("Dence_5", den_4//2, den_4, den_4//4))
                drop_2, drop_3, drop_4, drop_5 = drop_1, drop_1, drop_1, drop_1
                batchnorm_2, batchnorm_3, batchnorm_4, batchnorm_5 = batchnorm_1, batchnorm_1, batchnorm_1, batchnorm_1
                l2_2, l2_3, l2_4, l2_5 = l2_1, l2_1, l2_1, l2_1
        # モデル定義
        if branch_Tox21_12task == True:
            # 全結合層12個に分岐し、fine-tuningなど設定したモデルを返す
            model, orig_model = define_model.get_12branch_fine_tuning_model(self.out_dir, self.shape[0], self.shape[1], self.shape[2], self.num_classes
                                                                           , choice_model, trainable
                                                                           , FCnum=FCnum
                                                                           , FCpool=self.FCpool
                                                                           , Dence_1=den_1, Dropout_1=drop_1, addBatchNorm_1=batchnorm_1, l2_rate_1=l2_1
                                                                           , Dence_2=den_2, Dropout_2=drop_2, addBatchNorm_2=batchnorm_2, l2_rate_2=l2_2
                                                                           , Dence_3=den_3, Dropout_3=drop_3, addBatchNorm_3=batchnorm_3, l2_rate_3=l2_3
                                                                           , Dence_4=den_4, Dropout_4=drop_4, addBatchNorm_4=batchnorm_4, l2_rate_4=l2_4
                                                                           , Dence_5=den_5, Dropout_5=drop_5, addBatchNorm_5=batchnorm_5, l2_rate_5=l2_5
                                                                           , pred_kernel_initializer=self.pred_kernel_initializer, pred_l2_rate=self.pred_l2_rate
                                                                           , activation=self.activation
                                                                           , gpu_count=self.gpu_count
                                                                          )
        else:
            # 分岐なしのfine-tuningなど設定したモデルを返す
            model, orig_model = define_model.get_fine_tuning_model(self.out_dir, self.shape[0], self.shape[1], self.shape[2], self.num_classes
                                                                   , choice_model, trainable
                                                                   , FCnum=FCnum
                                                                   , FCpool=self.FCpool
                                                                   , Dence_1=den_1, Dropout_1=drop_1, addBatchNorm_1=batchnorm_1, l2_rate_1=l2_1
                                                                   , Dence_2=den_2, Dropout_2=drop_2, addBatchNorm_2=batchnorm_2, l2_rate_2=l2_2
                                                                   , Dence_3=den_3, Dropout_3=drop_3, addBatchNorm_3=batchnorm_3, l2_rate_3=l2_3
                                                                   , Dence_4=den_4, Dropout_4=drop_4, addBatchNorm_4=batchnorm_4, l2_rate_4=l2_4
                                                                   , Dence_5=den_5, Dropout_5=drop_5, addBatchNorm_5=batchnorm_5, l2_rate_5=l2_5
                                                                   , pred_kernel_initializer=self.pred_kernel_initializer, pred_l2_rate=self.pred_l2_rate
                                                                   , activation=self.activation
                                                                   , gpu_count=self.gpu_count
                                                                   , seresnet_num=trial.suggest_categorical('seresnet_num', self.seresnet_num) # SEResNet の種類指定 18,34,50,101,154 のいずれかしかだめ
                                                                   , sedensenet_num=trial.suggest_categorical('sedensenet_num', self.sedensenet_num) # SEDenseNet の種類指定 121,161,169,201,264 のいずれかしかだめ
                                                                   , seresnext_num=trial.suggest_categorical('seresnext_num', self.seresnext_num) # SEResNext の種類指定 50,101 のいずれかしかだめ
                                                                   , add_se=trial.suggest_categorical('add_se', self.add_se) # FC層の前にSE block つけるか
                                                                  )
        return model, orig_model


    # ============================================================ d_cls_generator ============================================================
    def _optuna_d_cls_generator(self, trial, branch_Tox21_12task):
        """
        trialでself.d_clsのImageDataGeneratorにパラメータセットする
        """
        # ImageDataGenerator
        horizontal_flip = trial.suggest_categorical('horizontal_flip', self.horizontal_flip)
        vertical_flip = trial.suggest_categorical('vertical_flip', self.vertical_flip)
        if self.rotation_unit != 0.0:
            rotation_range = trial.suggest_discrete_uniform('rotation_range', self.rotation_range_min, self.rotation_range_max, self.rotation_unit)
        else:
            # 回転の刻み幅(self.rotation_unit)=0.0 なら回転角は固定値(self.rotation_range_max)
            rotation_range = self.rotation_range_max
            trial.set_user_attr("rotation_range", str(rotation_range))
        if self.zoom_range_unit != 0.0:
            zoom_range_low = trial.suggest_discrete_uniform('zoom_range_low', self.zoom_range_low_min, self.zoom_range_low_max, self.zoom_range_unit)
            zoom_range_high = trial.suggest_discrete_uniform('zoom_range_high', self.zoom_range_high_min, self.zoom_range_high_max, self.zoom_range_unit)
        else:
            # 拡大縮小の刻み幅(self.zoom_range_unit)=0.0 なら拡大縮小の大きさは固定値(self.zoom_range_low_min, self.zoom_range_high_max)
            zoom_range_low = self.zoom_range_low_min
            zoom_range_high = self.zoom_range_high_max
            trial.set_user_attr("zoom_range_low", str(zoom_range_low))
            trial.set_user_attr("zoom_range_high", str(zoom_range_high))
        if self.shear_range_unit != 0.0:
            shear_range = trial.suggest_discrete_uniform('shear_range', self.shear_range_min, self.shear_range_max, self.shear_range_unit)
        else:
            # せん断の刻み幅(self.shear_range_unit)=0.0 ならせん断の倍率は固定値(self.shear_range_max)
            shear_range = self.shear_range_max
            trial.set_user_attr("shear_range", str(shear_range))
        if self.channel_shift_range_unit != 0.0:
            channel_shift_range = trial.suggest_discrete_uniform('channel_shift_range_unit', self.channel_shift_range_min, self.channel_shift_range_max, self.channel_shift_range_unit)
        else:
            # ランダムに画素値に値を足す幅(self.channel_shift_range_unit)=0.0 ならランダムに画素値に値を足す値は固定値(self.channel_shift_range_max)
            channel_shift_range = self.channel_shift_range_max
            trial.set_user_attr("channel_shift_range", str(channel_shift_range))
        if self.brightness_range_unit != 0.0:
            brightness_range_low = trial.suggest_discrete_uniform('brightness_range_low', self.brightness_range_low_min, self.brightness_range_low_max, self.brightness_range_unit)
            brightness_range_high = trial.suggest_discrete_uniform('brightness_range_high', self.brightness_range_high_min, self.brightness_range_high_max, self.brightness_range_unit)
        else:
            # ランダムに明度変更の幅(self.brightness_range_unit)=0.0 ならランダムに明度変更倍率は固定値(self.brightness_range_low_min, self.brightness_range_high_max)
            brightness_range_low = self.brightness_range_low_min
            brightness_range_high = self.brightness_range_high_max
            trial.set_user_attr("brightness_range_low", str(brightness_range_low))
            trial.set_user_attr("brightness_range_high", str(brightness_range_high))
        # mixup 含めるか
        mix_up_alpha = trial.suggest_categorical('mix_up_alpha', self.mix_up_alpha)
        # Random Erasing 含めるか
        random_erasing_prob = trial.suggest_categorical('random_erasing_prob', self.random_erasing_prob)
        random_erasing_maxpixel = self.random_erasing_maxpixel
        # MyImageDataGenerator のパラメータセット
        my_IDG_options={'rescale': self.rescale
                        , 'horizontal_flip': horizontal_flip
                        , 'vertical_flip': vertical_flip
                        , 'rotation_range': rotation_range
                        , 'zoom_range': [zoom_range_low, zoom_range_high]
                        , 'shear_range': shear_range
                        , 'channel_shift_range': channel_shift_range
                        , 'brightness_range': [brightness_range_low, brightness_range_high]
                        , 'random_crop': self.random_crop # random_crop 含めるか
                        , 'mix_up_alpha': mix_up_alpha
                        , 'random_erasing_prob': random_erasing_prob
                        , 'random_erasing_maxpixel': random_erasing_maxpixel
                        }

        # d_cls.train_gen, d_cls.valid_gen d_cls.test_gen 作成
        self.d_cls.create_my_generator_flow_from_directory(self.train_data_dir
                                                            , self.class_name
                                                            , valid_data_dir=self.validation_data_dir
                                                            , test_data_dir=self.test_dir
                                                            , color_mode='rgb'
                                                            , class_mode='categorical'
                                                            , my_IDG_options=my_IDG_options)

        # Tox21用.12出力（task）をだすgeneratorにするか
        if branch_Tox21_12task == True:
            train_gen = get_train_valid_test.generator_12output(self.d_cls.train_gen)
            valid_gen = get_train_valid_test.generator_12output(self.d_cls.valid_gen)
        else:
            train_gen = self.d_cls.train_gen
            valid_gen = self.d_cls.valid_gen
        #print(type(train_gen))
        return train_gen, valid_gen


    # ============================================================ callback ============================================================
    def _optuna_callbacks(self, trial, lr, orig_model):
        """
        trialでcallbackにパラメータセットする
        """
        # 学習率変更するcallback
        callback_lr = trial.suggest_categorical('callback_lr', self.callback_lr)
        cb = copy.copy(self.callbacks)# コピー後に一方の値を変更しても、もう一方には影響しないようにする
        if callback_lr == 'cosine_annealing':
            cb.append(my_callback.cosine_annealing(factor=0.01, epochs=self.cosine_annealing_epochs))
        elif callback_lr == 'LearningRateScheduler':
            base_lr = lr  # adamとかなら1e-3くらい。SGDなら例えば 0.1 * batch_size / 128 とかくらい。nadamなら0.002*10 ?
            lr_decay_rate = 1 / 3
            lr_steps = 4
            cb.append(keras.callbacks.LearningRateScheduler(lambda ep: float(base_lr * lr_decay_rate ** (ep * lr_steps // self.epochs))))

        # モデル保存するcallback つけるか
        if self.callback_save_model == True:
            weight_dir = os.path.join(self.out_dir, 'model_weight_optuna')
            os.makedirs(weight_dir, exist_ok=True)
            if self.gpu_count == 1:
                if self.return_val_loss:
                    cb.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(weight_dir, str(trial.trial_id)+'.h5'), monitor='val_loss', save_best_only=True, verbose=1))
                else:
                    cb.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(weight_dir, str(trial.trial_id)+'.h5'), monitor='val_'+self.metrics[0], save_best_only=True, verbose=1))
            else:
                # multigpuだとkeras.callbacks.ModelCheckpoint はつかえない
                cb.append(my_callback.MyCheckPoint(orig_model, weight_dir, snapshots_epoch=self.epochs-1, filename=str(trial.trial_id)))

        # 学習ログのtsvファイル出力するcallback
        log_dir = os.path.join(self.out_dir, 'tsv_logger')
        os.makedirs(log_dir, exist_ok=True)
        cb.append(my_callback.tsv_logger(os.path.join(log_dir, str(trial.trial_id)+'.tsv')))

        # 枝刈り（最終的な結果がどのぐらいうまくいきそうかを大まかに予測し、良い結果を残すことが見込まれない試行は、最後まで行うことなく早期終了）
        cb.append(OptunaCallback(trial))

        return cb

    # ============================================================ __call__(main) ============================================================
    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        """optunaの目的関数定義（チューニングするパラメータをこの関数に書いて、学習結果のhistoryをreturnする）"""
        #セッションのクリア
        K.clear_session()

        # 試行にUUIDを設定
        # Optunaではtrial.set_user_attr()を使うことで、試行ごとにユーザーが設定した値を記録することができます。この値にはチューニングには使われません
        # https://qiita.com/koshian2/items/ef9c0c74fe38739599d5
        #trial_uuid = str(uuid.uuid4())
        #trial.set_user_attr("uuid", trial_uuid)

        # 固定値のパラメータもtrial.set_user_attr() にセットする
        trial.set_user_attr("out_dir", str(self.out_dir))
        trial.set_user_attr("shape", str(self.shape))
        trial.set_user_attr("num_classes", str(self.num_classes))
        trial.set_user_attr("gpu_count", str(self.gpu_count))
        trial.set_user_attr("loss", str(self.loss))
        trial.set_user_attr("metrics", str(self.metrics))
        trial.set_user_attr("callbacks", str(self.callbacks))
        trial.set_user_attr("activation", str(self.activation))
        trial.set_user_attr("pred_kernel_initializer", str(self.pred_kernel_initializer))
        trial.set_user_attr("pred_l2_rate", str(self.pred_l2_rate))
        trial.set_user_attr("FCpool", str(self.FCpool))

        # 分岐有りタスクにするか
        branch_Tox21_12task = trial.suggest_categorical('branch_Tox21_12task', self.branch_Tox21_12task)

        # trialでmodelにパラメータセットする
        model, orig_model = self._optuna_model(trial, branch_Tox21_12task)

        # （初期）学習率
        lr = float(trial.suggest_categorical('lr', self.lr))

        # optimizer
        choice_optim = trial.suggest_categorical('choice_optim', self.choice_optim)
        optim = define_model.get_optimizers(choice_optim, lr=lr)

        # モデルコンパイル
        model.compile(loss=multi_loss.build_masked_loss(K.binary_crossentropy),
                      optimizer=optim,
                      metrics=self.metrics)

        # trialでself.d_clsのImageDataGeneratorにパラメータセットする
        train_gen, valid_gen = self._optuna_d_cls_generator(trial, branch_Tox21_12task)

        # trialでcallbackにパラメータセットする
        cb = self._optuna_callbacks(trial, lr, orig_model)

        # 学習実行
        history = model.fit_generator(train_gen
                                      , steps_per_epoch=int(self.d_cls.init_train_steps_per_epoch)
                                      , validation_data=valid_gen
                                      , validation_steps=int(self.d_cls.init_valid_steps_per_epoch)
                                      , epochs=self.epochs
                                      , callbacks=cb
                                      , verbose=self.verbose # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
                                     )

        # acc とloss の記録
        # https://qiita.com/koshian2/items/ef9c0c74fe38739599d5
        #print(history.history)
        trial.set_user_attr('loss', history.history['loss'][-1])
        trial.set_user_attr('val_loss', history.history['val_loss'][-1])
        # 分岐有りにすると binary_accuracy とは出ずに、task3_pred_binary_accuracy のように接頭語がついてしまうためacc = None とする
        if branch_Tox21_12task == True:
            trial.set_user_attr(self.metrics[0], None)
            trial.set_user_attr('val_'+self.metrics[0], None)
        else:
            trial.set_user_attr(self.metrics[0], 1 - history.history[self.metrics[0]][-1])
            trial.set_user_attr('val_'+self.metrics[0], 1 - history.history['val_'+self.metrics[0]][-1])

        if self.return_val_loss:
            # 検証用データに対するlossが最小となるハイパーパラメータ を返す
            return history.history['val_loss'][-1]
        else:
            # 検証用データに対するaccが最大となるハイパーパラメータ を返す
            return 1 - history.history['val_'+self.metrics[0]][-1]

if __name__ == '__main__':
    print('optuna_train_base.py: loaded as script file')
else:
    print('optuna_train_base.py: loaded as module file')
