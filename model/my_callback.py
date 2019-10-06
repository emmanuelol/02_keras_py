# -*- coding: utf-8 -*-
"""
自作コールバック
"""
import os
import keras
import math
import time
import numpy as np
import pandas as pd
import pathlib
import warnings
from sklearn.metrics import roc_auc_score#, average_precision_score
from collections import OrderedDict

class MyCheckPoint(keras.callbacks.Callback):
    """
    https://github.com/keras-team/keras/issues/8649 より
    各epochのモデル保存するためのコールバック（マルチGPU用）
    Args:
        snapshots_epoch: 指定したepoch 倍のepoch 時だけ保存する（コサインアニーリング+アンサンブルやるSnapshot Ensemble用）
        デフォルトのsnapshots_epoch=None なら全エポック保存
    """
    def __init__(self, model, weight_dir, snapshots_epoch=None, filename='model_at_epoch', is_best_val_loss=True):
         self.model_to_save = model
         self.weight_dir = weight_dir
         self.snapshots_epoch = snapshots_epoch
         self.filename = filename
         self.is_best_val_loss = is_best_val_loss
         self.val_loss = 100000.0

    def on_epoch_end(self, epoch, logs=None):
        # best_val_lossのモデルの重み保存場合
        if self.is_best_val_loss == True and self.val_loss > log['val_loss']:
            self.val_loss = log['val_loss']
            self.model_to_save.save(os.path.join(self.weight_dir, self.filename+'best_val_loss.h5' % epoch))

        # 指定したepoch 倍のepoch 時だけ保存する（コサインアニーリング+アンサンブルやるSnapshot Ensemble用）
        if self.snapshots_epoch is not None:
            amari = (epoch+1) % self.snapshots_epoch
            if epoch+1 >= self.snapshots_epoch and amari == 0:
                self.model_to_save.save(os.path.join(self.weight_dir, self.filename+'_%d.h5' % epoch))
        # 全エポック重み保存
        else:
            self.model_to_save.save(os.path.join(self.weight_dir, self.filename+'_%d.h5' % epoch))



class LossHistory(keras.callbacks.Callback):
    """
    lrとloss
    ※この関数は毎エポックの終わりにlrが表示されるだけ
    　keras.callbacks.CSVLogger にlr出力しない
    Usage:
        loss_history = LossHistory()
        callbacks_list = [loss_history]
    """
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('\nlr:', step_decay(len(self.losses)))

def step_decay(epoch, initial_lrate = 0.1, drop = 0.5, epochs_drop = 10.0):
    """
    LearningRateScheduler用lr段階的に下げるための関数
    epochs_drop のエポック刻みでinitial_lrate をdrop 倍する
    Usage:
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate]
    """
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate


def cosine_annealing(factor=0.01, epochs=None):
    """Cosine Annealing (without restart)。
    ■SGDR: Stochastic Gradient Descent with Warm Restarts
    https://arxiv.org/abs/1608.03983

    学習率をエポック増やすごとにコサインカーブで上げ下げする
    home/aaa00162/jupyterhub/notebook/other/04.HSC_AI_tool_code/detectAI_tool/pytoolkit/pytoolkit/dl/
    callbacks.py
    より

    Args:
        factor: lr下げる倍率。lr*factor が最小のlrに設定される
        epochs: コサインカーブのほぼ半周期になるエポック数。デフォルトのNoneだと半周期だけ（lr下がるだけ）
    """
    import keras
    import keras.backend as K
    assert factor < 1
    class _CosineAnnealing(keras.callbacks.Callback):
        def __init__(self, factor, epochs):
            self.factor = factor
            self.period = None
            self.start_lr = None
            self.epochs = epochs
            super().__init__()
        def on_train_begin(self, logs=None):
            self.start_lr = float(K.get_value(self.model.optimizer.lr))
        def on_epoch_begin(self, epoch, logs=None):
            lr_max = self.start_lr
            lr_min = self.start_lr * self.factor
            r = (epoch + 1) / (self.epochs or self.params['epochs'])
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * r))
            K.set_value(self.model.optimizer.lr, float(lr))
    return _CosineAnnealing(factor=factor, epochs=epochs)

# 下山さんのコードより
# callbacks.py
def learning_curve_plot(filename, metric='loss'):
    """Learning Curvesの描画を行う。
    # 引数
    - filename: 保存先ファイル名。「{metric}」はmetricの値に置換される。str or pathlib.Path
    - metric: 対象とするmetric名。lossとかaccとか。
    """
    import keras
    import matplotlib.pyplot as plt

    class _LearningCurvePlotter(keras.callbacks.Callback):

        def __init__(self, filename, metric='loss'):
            self.filename = pathlib.Path(filename).resolve()
            self.metric = metric
            self.met_list = []
            self.val_met_list = []
            super().__init__()

        def on_epoch_end(self, epoch, logs=None):
            try:
                self._plot(logs)
            except BaseException:
                import traceback
                warnings.warn(traceback.format_exc(), RuntimeWarning)

        def _save(self, ax, file, dpi=None, facecolor='w', edgecolor='w',
                 orientation='portrait', papertype=None, format=None,  # pylint: disable=W0622
                 transparent=False, bbox_inches=None, pad_inches=0.1,
                 frameon=None, **kwargs):
            """保存。"""
            # ディレクトリ作成
            if isinstance(file, (str, pathlib.Path)):
                file = pathlib.Path(file)
                file.resolve().parent.mkdir(parents=True, exist_ok=True)
                if format is None:
                    format = file.suffix[1:]
            else:
                if format is None:
                    format = 'png'
            # 保存
            ax.get_figure().savefig(
                file, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor,
                orientation=orientation, papertype=papertype, format=format,
                transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches,
                frameon=frameon, **kwargs)

        def _close(self, ax):
            """後始末。"""
            # …これで正しいかは不明…
            import matplotlib.pyplot as plt
            plt.close(ax.get_figure())

        def _plot(self, logs):
            met = logs.get(self.metric)
            if met is None:
                warnings.warn(f'LearningCurvePlotter requires {self.metric} available!', RuntimeWarning)
            val_met = logs.get(f'val_{self.metric}')

            self.met_list.append(met)
            self.val_met_list.append(val_met)

            if len(self.met_list) > 1:
                import pandas as pd
                df = pd.DataFrame()
                df[self.metric] = self.met_list
                if val_met is not None:
                    df[f'val_{self.metric}'] = self.val_met_list

                #with draw.get_lock():
                ax = df.plot()
                self._save(ax, self.filename.parent / self.filename.name.format(metric=self.metric))
                self._close(ax)

    return _LearningCurvePlotter(filename=filename, metric=metric)

# 下山さんのコードより
# callbacks.py
def tsv_logger(filename, append=False):
    """ログを保存するコールバック
    # 引数
    - filename: 保存先ファイル名。「{metric}」はmetricの値に置換される。str or pathlib.Path
    - append: 追記するのか否か。
    Usage:
        cd = []
        cb.append(my_callback.tsv_logger(os.path.join(out_dir, 'tsv_logger.tsv')))
    """
    import keras
    import keras.backend as K

    import csv
    import pathlib
    #enabled = hvd.is_master()
    class _TSVLogger(keras.callbacks.Callback):
        def __init__(self, filename, append):#, enabled):
            self.filename = pathlib.Path(filename)
            self.append = append
            #self.enabled = enabled
            self.log_file = None
            self.log_writer = None
            self.epoch_start_time = None
            super().__init__()
        def on_train_begin(self, logs=None):
            #if self.enabled:
            #    self.filename.parent.mkdir(parents=True, exist_ok=True)
            #    self.log_file = self.filename.open('a' if self.append else 'w', buffering=65536)
            #else:
            #    self.log_file = io.open_devnull('w', buffering=65536)
            self.log_file = self.filename.open('a' if self.append else 'w', buffering=65536)
            self.log_writer = csv.writer(self.log_file, delimiter='\t', lineterminator='\n')
            self.log_writer.writerow(['epoch', 'lr'] + self.params['metrics'] + ['time'])
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs['lr'] = K.get_value(self.model.optimizer.lr)
            elapsed_time = time.time() - self.epoch_start_time
            def _format_metric(logs, k):
                value = logs.get(k)
                if value is None:
                    return '<none>'
                return format(value, '.4f')
            metrics = [_format_metric(logs, k) for k in self.params['metrics']]
            self.log_writer.writerow([epoch + 1, format(logs['lr'], '.1e')] + metrics +
                                     [str(int(np.ceil(elapsed_time)))])
            self.log_file.flush()
        def on_train_end(self, logs=None):
            self.log_file.close()
            self.append = True  # 同じインスタンスの再利用時は自動的に追記にする
    return _TSVLogger(filename=filename, append=append)#, enabled=enabled)


def get_base_cb(output_dir, epochs, cosine_annealing_num_epoch=None, early_stopping=50):
    """ 基本のcallbackの組み合わせ """
    cb = []
    # 学習率をエポック増やすごとにコサインカーブで上げ下げする. epochsはコサインカーブのほぼ半周期になるエポック数
    cb.append(cosine_annealing(epochs=epochs))
    # ログを保存するカスタムコールバック
    cb.append(tsv_logger(os.path.join(output_dir, 'tsv_logger.tsv')))
    # epochごとに学習曲線保存する自作callback
    cb.append(learning_curve_plot(os.path.join(output_dir, 'learning_curve.png')))
    # 各エポックでval_lossが最小となるモデル保存
    cb.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'best_val_loss.h5'), monitor='val_loss', save_best_only=True, verbose=1))
    # 各エポックでval_accが最大となるモデル保存
    cb.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'best_val_acc.h5'), monitor='val_acc', save_best_only=True, verbose=1, mode='max'))
    # 過学習の抑制 <early_stopping_pati>step続けてval_loss減らなかったら打ち切る
    cb.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping, verbose=1))
    return cb

class RocAucCallbackGenerator(keras.callbacks.Callback):
    """
    二値分類のとき使うroc_aucを計算するコールバック
    generatorを使う場合
    マルチラベル+欠損値の対応有り
    https://github.com/keras-team/keras/issues/3230
    Usage:
        cb = []
        cb.append(my_callback.RocAucCallbackGenerator(d_cls.valid_gen, steps=len(d_cls.valid_gen), mask=-1.0))
    """
    def __init__(self, val_gen, steps=None, mask=-1.0):
        self.val_gen = val_gen
        self.steps = steps
        if steps is None:
            self.steps = len(self.val_gen)
        self.mask = mask
        self.val_roc = []
        self.val_pr = []

    def on_epoch_end(self, epoch, logs={}):
        # generator をreset() せずにpredict_generator 実行すると順番がグチャグチャでpredict してしまうらしい
        # https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
        # 自作generatorだとreset()が無い場合があった。。。その場合エラーになる
        self.val_gen.reset()
        y_pred = self.model.predict_generator(self.val_gen, steps=self.steps)
        y_true = self.val_gen.y
        #print('y_pred =', y_pred)
        #print('y_pred.shape = ', y_pred.shape)
        #print('y_true =', y_true)
        #print('y_true.shape = ', y_true.shape)

        # 多クラス（マルチラベル）の場合
        if y_true.ndim == 2:
            # クラスごとに分ける
            #print('')
            val_roc = []
            val_pr = []
            for i in range(y_true.shape[1]):
                y_true_i= y_true[:,i]
                y_pred_i = y_pred[:,i]
                #print(y_true_i)
                #print(y_pred_i)
                #print(y_true_i.shape)
                #print(y_pred_i.shape)
                val_roc_i, val_pr_i = self._del_mask_label_roc_auc_score(y_true_i, y_pred_i)
                val_roc.append(val_roc_i)
                val_pr.append(val_pr_i)
                #print('class:', i, 'val_roc_auc:', str(round(val_roc_i, 4)))
            self.val_roc.append(val_roc)
            self.val_pr.append(val_pr)
            print('val_roc_auc:', [str(round(val_roc_i, 4)) for val_roc_i in val_roc])
            #print('val_pr_auc:', [str(round(val_pr_i, 4)) for val_pr_i in val_pr])
        else:
            val_roc, val_pr = self._del_mask_label_roc_auc_score(y_true, y_pred)
            self.val_roc.append(val_roc)
            self.val_pr.append(val_pr)
            print('val_roc_auc:', str(round(val_roc, 4)))
            #print('val_pr_auc:', str(round(val_pr, 4)))

    def _del_mask_label_roc_auc_score(self, y_true, y_pred):
        """ metrics.roc_curveはy_trueが2種類でないとダメなので、欠損ラベルのレコードは削除してroc_auc計算 """
        #print(y_true.shape, y_pred.shape)
        roc, pr = 0.0, 0.0
        if len(np.unique(y_true)) != 2:
            y_df = pd.DataFrame( {'y_true': y_true, 'y_pred': y_pred}, index=list(range(0, len(y_true))) ) # index指定しないとエラーになる
            #print('y_df:', y_df)
            y_df = y_df[y_df['y_true'] != self.mask]# 欠損ラベル=self.mask 以外の行だけにする
            y_true = np.array(y_df['y_true'])
            y_pred = np.array(y_df['y_pred'])
            #print(y_true.shape)
            # 欠損ラベルのレコードは削除したから、posi/negaの2種ラベルだけになってるはず
            # ラベルが1種類しかないときはauc計算できないのではじく
            if len(np.unique(y_true)) == 2:
                roc = roc_auc_score(y_true, y_pred)
                #pr = average_precision_score(y_true, y_pred)
        return roc, pr

class HeteroGeniousCallbacks(keras.callbacks.Callback):
    """
    重み付きヘテロジニアスラーニング
    重み付きヘテロジニアスラーニングはマルチタスクラーニングに適用される手法で、事前学習し、タスクごとのclass_weightを導出した後に重みを設定して再度学習する手法
    https://www.kabuku.co.jp/developers/multiple_improve

    学習の終了時に検証データのロスに対する平均と標準偏差を計算して、タスクごとのclass_weightを計算
    マルチタスクでは機能する（若干精度上がる）が、マルチクラスでは機能しなかった
    Usage:
        hetero_genious_callbacks = my_callback.HeteroGeniousCallbacks()
        callbacks_list = [hetero_genious_callbacks]
    """

    def __init__(self,
                 variable_number: int = 2,
                 ):
        super(HeteroGeniousCallbacks, self).__init__()

        self.current_val_loss = {}
        self.class_weight = {}
        self.variable_number = variable_number

    def on_epoch_end(self, epoch, logs=None):
        """ Epoch後の検証データのロスデータを貯める """
        sort_logs = OrderedDict(
            sorted(logs.items(), key=lambda x: x[0]))
        for each_label, each_values in sort_logs.items():
            if 'val' in each_label and 'loss' in each_label:
                if each_label not in self.class_weight:
                    self.class_weight[each_label] = each_values
                else:
                    each_values_tmp = self.class_weight[each_label]
                    each_value_list = np.vstack((each_values,
                                                 each_values_tmp))
                    self.class_weight[each_label] = each_value_list

    def on_train_end(self, logs=None):
        """ 学習終了後に基準タスクとの安定度の比から重みを導出 """
        self.class_weight = OrderedDict(
            sorted(self.class_weight.items(), key=lambda x: x[0]))

        val_stable_dict = {}
        for each_label, each_value_lsit in self.class_weight.items():
            val_stable_dict[each_label] = \
                np.average(each_value_lsit) + 3.0 * np.std(each_value_lsit)

        val_stable_dict = OrderedDict(
            sorted(val_stable_dict.items(), key=lambda x: x[1]))

        most_stable_value = [value for value in val_stable_dict.values()][0]

        index = 0
        tmp_class_weight = {}

        for each_label in self.class_weight.keys():
            tmp_class_weight[index] = \
                val_stable_dict[each_label] / most_stable_value
            if index >= self.variable_number - 1:
                break
            index += 1

        self.class_weight = tmp_class_weight
        print('hetero genious class weight {}'.format(self.class_weight))

# -------------- snapshot ensembleのcallback --------------
class SnapshotModelCheckpoint(keras.callbacks.Callback):
    """
    https://raw.githubusercontent.com/titu1994/Snapshot-Ensembles/master/snapshot.py

    モデルのスナップショットウェイトを保存するコールバック
    特定の時期にモデルの重みを保存する
    （そのエポックのモデルのスナップショットと見なすことができます）
    学習率が急激に増加する直前にウェイトを節約するために、コサインアニーリング学習率スケジュールと共に使用する必要があります

    # Arguments:
        nb_epochs: total number of epochs that the model will be trained for.
        nb_snapshots: number of times the weights of the model will be saved.
        fn_prefix: prefix for the filename of the weights.
    """

    def __init__(self, nb_epochs, nb_snapshots, fn_prefix='Model'):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = nb_epochs // nb_snapshots
        self.fn_prefix = fn_prefix

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:
            filepath = self.fn_prefix + "-%d.h5" % ((epoch + 1) // self.check)
            #self.model.save_weights(filepath, overwrite=True) # model.load_weights でうまくロードできなかったので、model.saveにする
            self.model.save(filepath)
            #print("Saved snapshot at weights/%s_%d.h5" % (self.fn_prefix, epoch))

class SnapshotCallbackBuilder:
    """
    https://raw.githubusercontent.com/titu1994/Snapshot-Ensembles/master/snapshot.py

    モデルのスナップショットアンサンブルトレーニングのためのコールバックビルダー
    コールバックのリストを作成します
    これは、モデルの重みを特定のエポックで保存して学習率を急激に高めるためにモデルをトレーニングするときに提供されます

    Usage:
        from snapshot import SnapshotCallbackBuilder
        M = 5 # number of snapshots
        nb_epoch = T = 200 # number of epochs
        alpha_zero = 0.1 # initial learning rate
        model_prefix = 'Model_'
        snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)
        ...
        model = Sequential() OR model = Model(ip, output) # Some model that has been compiled
        model.fit(trainX, trainY, callbacks=snapshot.get_callbacks(model_prefix=model_prefix))
    """

    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        """
        Initialize a snapshot callback builder.

        # Arguments:
            nb_epochs: total number of epochs that the model will be trained for.
            nb_snapshots: number of times the weights of the model will be saved.
            init_lr: initial learning rate
        """
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model', monitor="val_acc", out_dir='weights'):
        """
        Creates a list of callbacks that can be used during training to create a
        snapshot ensemble of the model.

        Args:
            model_prefix: prefix for the filename of the weights.
            monitor: ModelCheckpointで保存する指標
            out_dir: 重みファイルの出力ディレクトリ

        Returns: list of 3 callbacks [ModelCheckpoint, LearningRateScheduler,
                 SnapshotModelCheckpoint] which can be provided to the 'fit' function
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        callback_list = [keras.callbacks.ModelCheckpoint(out_dir+"/%s-Best.h5" % model_prefix,
                                                            monitor=monitor,
                                                            save_best_only=True,
                                                            save_weights_only=False),
                         keras.callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule),
                         SnapshotModelCheckpoint(self.T, self.M, fn_prefix=out_dir+'/%s' % model_prefix)]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out) # init_lr*0.25ぐらいまでコサインカーブで下がる


if __name__ == '__main__':
    print('my_callback.py: loaded as script file')
else:
    print('my_callback.py: loaded as module file')
