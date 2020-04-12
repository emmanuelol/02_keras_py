# -*- coding: utf-8 -*-
"""
Kerasを使用してワイン品質データセットのニューラルネットワークリグレッサーを最適化し、MLflowを使用してハイパーパラメーターとメトリックを記録するOptunaの例
参考:
https://github.com/optuna/optuna/tree/master/examples/mlflow
https://analytics-note.xyz/machine-learning/optuna-keras/
Usage:
    # モデル作成
    $ python keras_mlflow.py

    # mlflow ui起動
    $ cd ./output
    $ mlflow ui # ローカルで起動する場合 -> http://127.0.0.1:5000
    $ mlflow ui --port 5000 --host 0.0.0.0 # docker上で実行する場合は--host必要 -> http://127.0.0.1:5000
    $ mlflow server -p 6055 -h moltes24 # サーバで起動する場合 -> http://xceedsys:6055

    # jupyterでOptunaの学習曲線可視化
    %matplotlib inline
    import optuna
    study_name = 'study'
    storage = storage=f"sqlite:///output/study_history.db"
    study = optuna.study.load_study(study_name, storage)
    optuna.visualization.plot_intermediate_values(study)
"""

import os
gpu_num = "0" # "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

import argparse
import datetime
import numpy as np

# tensorflowのINFOレベルのログを出さないようにする
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

from tensorflow import keras
import mlflow
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.integration import KerasPruningCallback
import traceback

TEST_SIZE = 0.25
BATCHSIZE = 16
EPOCHS = 100

class MlflowWriter():
    """ https://ymym3412.hatenablog.com/entry/2020/02/09/034644 """
    def __init__(self, experiment_name, **kwargs):
        self.client = MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parent_name}.{k}', v)
                else:
                    self.client.log_param(self.run_id, f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f'{parent_name}.{i}', v)

    def log_torch_model(self, model):
        with mlflow.start_run(self.run_id):
            pytorch.log_model(model, 'models')

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)


class OptunaCallback(keras.callbacks.Callback):
    """
    Optunaでの枝刈り（最終的な結果がどのぐらいうまくいきそうかを大まかに予測し、良い結果を残すことが見込まれない試行は、最後まで行うことなく早期終了）
    https://qiita.com/koshian2/items/107c386f81c9bb7f8df3
    """
    def __init__(self, trial, prune):
        self.trial = trial
        self.prune = prune

    def on_epoch_end(self, epoch, logs):
        current_val_error = logs["val_loss"]# 1.0 - logs["val_acc"]
        # epochごとの値記録（intermediate_values）
        self.trial.report(current_val_error, step=epoch)
        if self.prune == True:
            # 打ち切り判定
            if self.trial.should_prune(epoch):
                # optuna.prunersのデフォルトの設定で、最初の5trialをたたき台して使って、以降のtrialで打ち切っていく
                #raise optuna.structs.TrialPruned()
                raise optuna.exceptions.TrialPruned()

def get_keras_callbacks(output_dir:str, num_epoch:int) -> list:
    """ trainでつかうkerasのcallback取得 """
    cbs = []
    # 学習率をエポック増やすごとにコサインカーブで上げ下げする. epochsはコサインカーブのほぼ半周期になるエポック数
    #cbs.append(my_callback.cosine_annealing(factor=lr_factor, epochs=num_epoch))
    # ログを保存するカスタムコールバック
    #cbs.append(my_callback.tsv_logger(os.path.join(output_dir, 'tsv_logger.tsv')))
    # epochごとに学習曲線保存する自作callback
    #cbs.append(my_callback.learning_curve_plot(os.path.join(output_dir, 'learning_curve.png')))
    # 各エポックでval_lossが最小となるモデル保存
    cbs.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'val_loss_best.h5'), monitor='val_loss', save_best_only=True, verbose=False))
    #cbs.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'val_binary_accuracy_best.h5'), monitor='val_binary_accuracy', save_best_only=True, verbose=False, mode='max'))
    # 過学習の抑制 <early_stopping_pati>step続けてval_loss減らなかったら打ち切る
    cbs.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=num_epoch//3, verbose=1))
    return cbs

class Objective(object):
    def __init__(self, args:dict):
        self.args = args
        # mlflowの出力先ディレクトリ指定。直前のディレクトリはmlrunsでないとだめ
        mlflow.set_tracking_uri(os.path.join(args["output_dir"], 'mlruns'))
        # get_experiment_by_nameを指定すると、mlflow.start_run(experiment_id=experiment.experiment_idによりmlruns/1とかにデータ出力される
        mlflow.set_experiment(args["study_name"])
        self.experiment = mlflow.tracking.MlflowClient().get_experiment_by_name(args["study_name"])
        print("INFO: mlflow save dir. [{}]".format(os.path.join(args["output_dir"], 'mlruns', str(self.experiment.experiment_id))))

    def get_trial_params(self, trial) -> dict:
        """
        Get trial parameter
        Args:
            trial(trial.Trial):
        Returns:
            dict: parameter sample generated by trial object
        """
        trial.set_user_attr('data', 'sklearn.datasets.load_wine')
        trial.set_user_attr('batch_size', BATCHSIZE)
        trial.set_user_attr('test_size', TEST_SIZE)
        trial.set_user_attr('optimizer', 'SGD')
        trial.set_user_attr('normalize', 'StandardScaler')
        return {
            'lr': trial.suggest_loguniform("lr", 1e-5, 1e-1),
            'momentum': trial.suggest_uniform("momentum", 0.0, 1.0)
        }

    def mlflow_callback(self, study, trial):
        """ Optunaのobjectiveの最後に実行するcallback """
        trial_value = trial.value if trial.value is not None else float("nan")
        with mlflow.start_run(experiment_id=self.experiment.experiment_id, run_name=study.study_name):
            mlflow.log_params(trial.user_attrs)# mlflowで属性情報記録
            mlflow.log_params(trial.params)# mlflowでパラメータ記録
            mlflow.log_metrics({"mean_squared_error": trial_value})# mlflowでmetric記録

    def create_model(self, num_features, params, trial):
        model = keras.models.Sequential()
        model.add(
            keras.layers.Dense(
                num_features,
                activation="relu",
                kernel_initializer="normal",
                input_shape=(num_features,),
            )
        ),
        model.add(keras.layers.Dense(16, activation="relu", kernel_initializer="normal"))
        model.add(keras.layers.Dense(16, activation="relu", kernel_initializer="normal"))
        model.add(keras.layers.Dense(1, kernel_initializer="normal", activation="linear"))
        optimizer = keras.optimizers.SGD(**params)
        model.compile(loss="mean_squared_error", optimizer=optimizer)
        return model

    def objective(self, trial):
        # Clear clutter from previous Keras session graphs.
        keras.backend.clear_session()

        params = self.get_trial_params(trial)

        X, y = load_wine(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

        cbs = get_keras_callbacks(self.args["model_output_dir"], EPOCHS)
        if self.args['is_not_prune']:
            pass
        else:
            cbs.append(KerasPruningCallback(trial, "val_loss"))
            #cbs.append(OptunaCallback(trial, (not self.args['is_not_prune'])))

        model = self.create_model(X.shape[1], params, trial)
        history = model.fit(X_train, y_train
                            , shuffle=True
                            , batch_size=BATCHSIZE
                            , epochs=EPOCHS
                            , validation_data=(X_test, y_test)
                            , verbose=False
                            , callbacks=cbs)
        #test_loss = model.evaluate(X_test, y_test, verbose=0) # model.compileでmetricの指定が無いからlossが返される
        return np.min(history.history['val_loss'])

    def __call__(self, trial):
        """ Objective function for optuna """
        try: # optuna v0.18以上だとtryで囲まないとエラーでtrial落ちる
            return self.objective(trial)
        except Exception as e:
            #traceback.print_exc() # Exceptionが発生した際に表示される全スタックトレース表示
            return e # 例外を返さないとstudy.csvにエラー内容が記載されない

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default='/home/aaa00162/jupyterhub/notebook/other/tfgpu114_work/mlflow_test/output'
                    , help="output dir path.")
    ap.add_argument("-n_t", "--n_trials", type=int, default=100
                    , help="Optuna num trials")
    ap.add_argument("-s_n", "--study_name", type=str, default='study'
                    , help="Optuna trials study name")
    ap.add_argument("-d_n", "--distribute_name", type=str, default='model_output'
                    , help="Optuna distributing run name")
    ap.add_argument("--is_not_prune", action='store_const', const=True, default=False, help="optuna not prune flag.")
    return vars(ap.parse_args())

if __name__ == "__main__":
    args = get_args()

    os.makedirs(args['output_dir'], exist_ok=True)
    study = optuna.create_study(study_name=args["study_name"]
                                , storage=f"sqlite:///{args['output_dir']}/study_history.db"
                                , load_if_exists=True
                                , pruner=optuna.pruners.MedianPruner(n_warmup_steps=10) # MedianPrunerは、あるepochのときで計算したこれまでのtrialの中央値よりも今のepochの値が悪い場合は剪定する。n_warmup_steps=10で最低でも10epochは実行する
                                )

    # 分散実行用にモデルファイルの出力ディレクトリ作成
    args["model_output_dir"] = os.path.join(args['output_dir'], args['distribute_name'])
    os.makedirs(args['model_output_dir'], exist_ok=True)

    objective = Objective(args)
    study.optimize(objective
                    , n_trials=args["n_trials"]
                    , timeout=600# 600sec=10分経ったらstudy強制終了
                    , callbacks=[objective.mlflow_callback]) #
    study.trials_dataframe().to_csv(f"{args['output_dir']}/study_history.csv", index=False)
    print("Number of finished trials: {}".format(len(study.trials)))
    print(f"\nstudy.best_params:\n{study.best_params}")
    print(f"\nstudy.best_trial:\n{study.best_trial}")
