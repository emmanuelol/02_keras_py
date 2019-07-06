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
current_dir = os.path.dirname(os.path.abspath("__file__"))
path = os.path.join(current_dir, '../')
sys.path.append(path)
from transformer import get_train_valid_test
from model import define_model, multi_loss, my_callback

# Random Erasing+mixup
sys.path.append(r'/home/tmp10014/jupyterhub/notebook/other/lib_DL/mixup-generator')
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser

# optuna
sys.path.append(r'/home/aaa00162/jupyterhub/notebook/other/lib_DL/optuna-master')
import optuna


class OptunaCallback(keras.callbacks.Callback):
    """
    Kerasでの枝刈り
    https://qiita.com/koshian2/items/107c386f81c9bb7f8df3
    """
    def __init__(self, trial):
        self.trial = trial

    def on_epoch_end(self, epoch, logs):
        current_val_error = 1.0 - logs["val_loss"]
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
        - 画像の一部矩形領域を隠す（random_erasing)
        - 画像混ぜる(mix_up)
    """
    def __init__(self
                  , out_dir # 出力ディレクトリ
                  , d_cls # データ管理クラス get_train_valid_test.LabeledDataset
                  , shape=[331,331,3] # 入力層のサイズ
                  , num_classes=12 # クラス数
                  , epochs=30 # エポック数
                  , gpu_count=1 # GPUの数
                  , loss=multi_loss.build_masked_loss(K.binary_crossentropy) # 損失関数
                  , metrics=['binary_accuracy', multi_loss.masked_accuracy] # model.fit_generator()で使うメトリック
                  , verbose=0 # model.fit_generator()でログ出すか.0なら出さない.2ならエポックごとにログ出す.1はstepごとに出すためログが膨大になるので使わない
                  , activation='sigmoid' # 出力層の活性化関数
                  , pred_kernel_initializer='zeros' # 出力層の初期値
                  , pred_l2_rate=1e-4 # 出力層のl2
                  , FCpool='GlobalAveragePooling2D' # FC層のpooling
                  , return_val_loss=True # best_parameter をval_loss でとるか.False にしたらval_metrics[0] の指標でbest_parameter を返す
                  , callbacks=[] # コールバックのリスト
                  , callback_save_model=False # モデル保存するcallback つけるか. False なら保存しない
                  , choice_model=['VGG16','ResNet50','InceptionV3','Xception','InceptionResNetV2','NASNetLarge','SEResNet154','SEInceptionV3','SEInceptionResNetV2'] # Fine-tuning する学習済みモデル
                  , trainable=['all', 249] # 重み全層学習させるか（重みunfreeze開始レイヤーを番号で指定できる）
                  , FCnum=[0,1,2,3,4,5] # FC層の数
                  , Dence=[1024, 512, 256, 128] # FC層のユニット数
                  , Dropout=[0.0, 0.5, 0.7] # FC層のDropout
                  , addBatchNorm=[None, 'add'] # FC層のBatchNorm
                  , l2=[0.0, 1e-6, 1e-4, 1e-2] # FC層のl2
                  , choice_optim=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'] # optimizer
                  , lr=[0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0] # （初期）学習率
                  , callback_lr=[None, 'cosine_annealing', 'LearningRateScheduler'] # 学習率変更するcallback
                  , horizontal_flip=[True, False] # 画像の左右反転
                  , vertical_flip=[True, False] # 画像の上下反転
                  , rotation_range_min=0, rotation_range_max=30, rotation_unit=10 # 画像の回転の下限角、回転の上限角、回転角の刻み幅. 上限下限はAutoaugment の論文の値
                  , zoom_range_low_min=0.1, zoom_range_low_max=1.0, zoom_range_high_min=1.0, zoom_range_high_max=1.9, zoom_range_unit=0.1 # 画像の縮小の最少-最大倍率、拡大の最少-最大倍率、縮小拡大倍率の刻み幅. 上限下限はAutoaugment の論文の値
                  , shear_range_min=-0.3, shear_range_max=0.3, shear_range_unit=0.1 # 画像のせん断の最少倍率、せん断の最大倍率、せん断倍率の刻み幅
                  , random_eraser_flg=[True, False] # Random Erasing 含めるか
                  , pixel_min=0.0 # Random Erasing で使う画素数の最小値
                  , pixel_max=1.0 # Random Erasing で使う画像数の最大値（1/255で割ってるはずだから基本1.0）
                  , use_mixup=[True, False] # mixup 含めるか
                  , branch_Tox21_12task=[False] # Tox21用.12出力（task）をだすgeneratorにするか. False なら出力層12taskに分ける分岐なし（20181227_うまくいかない）
                 ):
        # define_model.get_fine_tuning_model(), model.fit_generator() で使う引数
        self.out_dir = out_dir
        self.d_cls = d_cls
        self.shape = shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.gpu_count = gpu_count
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose
        self.return_val_loss = return_val_loss
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
        self.random_eraser_flg = random_eraser_flg
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max
        self.use_mixup = use_mixup
        self.branch_Tox21_12task = branch_Tox21_12task

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        """optunaの目的関数定義（チューニングするパラメータをこの関数に書いて、学習結果のhistoryをreturnする）"""
        #セッションのクリア
        K.clear_session()

        # 試行にUUIDを設定
        # Optunaではtrial.set_user_attr()を使うことで、試行ごとにユーザーが設定した値を記録することができます。この値にはチューニングには使われません
        # https://qiita.com/koshian2/items/ef9c0c74fe38739599d5
        trial_uuid = str(uuid.uuid4())
        trial.set_user_attr("uuid", trial_uuid)

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
                                                              )

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
            trial.set_user_attr("zoom_range_low", str(zoom_range_low))
        if self.shear_range_unit != 0.0:
            shear_range = trial.suggest_discrete_uniform('shear_range', self.shear_range_min, self.shear_range_max, self.shear_range_unit)
        else:
            # せん断の刻み幅(self.shear_range_unit)=0.0 ならせん断の倍率は固定値(self.shear_range_max)
            shear_range = self.shear_range_max
            trial.set_user_attr("shear_range", str(shear_range))
        # ImageDataGenerator のパラメータセット
        IDG_options={'horizontal_flip': horizontal_flip
                     , 'vertical_flip': vertical_flip
                     , 'rotation_range': rotation_range
                     , 'zoom_range': [zoom_range_low, zoom_range_high]
                     , 'shear_range': shear_range
                    }
        # Random Erasing 含めるか
        random_eraser_flg = trial.suggest_categorical('random_eraser_flg', self.random_eraser_flg)
        if random_eraser_flg:
            IDG_options['preprocessing_function'] = get_random_eraser(v_l=self.pixel_min, v_h=self.pixel_max)
        # mixup 含めるか
        use_mixup = trial.suggest_categorical('use_mixup', self.use_mixup)
        # d_cls.train_gen, d_cls.valid_gen 作成
        self.d_cls.create_generator(use_mixup=use_mixup, IDG_options=IDG_options)

        # Tox21用.12出力（task）をだすgeneratorにするか
        branch_Tox21_12task = trial.suggest_categorical('branch_Tox21_12task', self.branch_Tox21_12task)
        if branch_Tox21_12task == True:
            self.d_cls.train_gen = get_train_valid_test.generator_12output(self.d_cls.train_gen)
            self.d_cls.valid_gen = get_train_valid_test.generator_12output(self.d_cls.valid_gen)

        # （初期）学習率
        lr = float(trial.suggest_categorical('lr', self.lr))
        # 学習率変更するcallback
        callback_lr = trial.suggest_categorical('callback_lr', self.callback_lr)
        cb = copy.copy(self.callbacks)# コピー後に一方の値を変更しても、もう一方には影響しないようにする
        if callback_lr == 'cosine_annealing':
            cb.append(my_callback.cosine_annealing(factor=0.01, epochs=None))
        elif callback_lr == 'LearningRateScheduler':
            base_lr = lr  # adamとかなら1e-3くらい。SGDなら例えば 0.1 * batch_size / 128 とかくらい。nadamなら0.002*10 ?
            lr_decay_rate = 1 / 3
            lr_steps = 4
            cb.append(keras.callbacks.LearningRateScheduler(lambda ep: float(base_lr * lr_decay_rate ** (ep * lr_steps // self.epochs))))

        # モデル保存するcallback つけるか
        # multigpuだとkeras.callbacks.ModelCheckpoint はつかえない
        if self.callback_save_model == True and self.gpu_count == 1:
            model_dir = os.path.join(self.out_dir, 'model_weight_optuna')
            os.makedirs(model_dir, exist_ok=True)
            if self.return_val_loss:
                #cb.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, trial_uuid+'.h5'), monitor='val_loss', save_best_only=True, verbose=1))
                cb.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, str(trial.trial_id)+'.h5'), monitor='val_loss', save_best_only=True, verbose=1))
            else:
                #cb.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, trial_uuid+'.h5'), monitor='val_'+self.metrics[0], save_best_only=True, verbose=1))
                cb.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, str(trial.trial_id)+'.h5'), monitor='val_'+self.metrics[0], save_best_only=True, verbose=1))

        # 学習ログのtsvファイル出力するcallback
        log_dir = os.path.join(self.out_dir, 'tsv_logger')
        os.makedirs(log_dir, exist_ok=True)
        #cb.append(my_callback.tsv_logger(os.path.join(log_dir, trial_uuid+'.tsv')))
        cb.append(my_callback.tsv_logger(os.path.join(log_dir, str(trial.trial_id)+'.tsv')))

        # optimizer
        choice_optim = trial.suggest_categorical('choice_optim', self.choice_optim)
        optim = define_model.get_optimizers(choice_optim, lr=lr)

        # モデルコンパイル
        model.compile(loss=multi_loss.build_masked_loss(K.binary_crossentropy),
                      optimizer=optim,
                      metrics=self.metrics)
        # 学習実行
        history = model.fit_generator(self.d_cls.train_gen
                                      , steps_per_epoch=self.d_cls.train_steps_per_epoch()
                                      , validation_data=self.d_cls.valid_gen
                                      , validation_steps=self.d_cls.valid_steps_per_epoch()
                                      , epochs=self.epochs
                                      , callbacks=cb
                                      , verbose=self.verbose # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
                                     )

        # acc とloss の記録
        # https://qiita.com/koshian2/items/ef9c0c74fe38739599d5
        trial.set_user_attr('loss', history.history['loss'][-1])
        trial.set_user_attr(self.metrics[0], 1 - history.history[self.metrics[0]][-1])
        trial.set_user_attr('val_loss', history.history['val_loss'][-1])
        trial.set_user_attr('val_'+self.metrics[0], 1 - history.history['val_'+self.metrics[0]][-1])

        if self.return_val_loss:
            # 検証用データに対するlossが最小となるハイパーパラメータ を返す
            return history.history['val_loss'][-1]
        else:
            # 検証用データに対するaccが最大となるハイパーパラメータ を返す
            return 1 - history.history['val_'+self.metrics[0]][-1]

if __name__ == '__main__':
    print('optuna_train.py: loaded as script file')
else:
    print('optuna_train.py: loaded as module file')
