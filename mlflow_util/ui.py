# -*- coding: utf-8 -*-
""" mlflow uiで使う関数 """
import os
import mlflow
import mlflow.keras

def keras_mlflow_param(output_dir, dict_log_metrics, model=None, exp_name='experiment'):#, my_IDG_options
    """
    モデルの学習実行後、MLflowにパラメータを記録する
    https://qiita.com/sumita_v09/items/174977f48c44244f9719
    """
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(nested=True):
        # 辞書でパラメータをまとめて記録
        # log_metricsに入れる値は数値にすること！！！数値以外いれるとエラーになる。bool型もダメ
        mlflow.log_metrics(dict_log_metrics)
        #mlflow.log_metrics(my_IDG_options)

        # その他に保存したいファイルなどを記録する
        # ./mlrunsのartifactsに指定したファイルがコピーされる。Dドライブのパスでもいけた
        mlflow.log_artifact(os.path.join(output_dir, 'tsv_logger.tsv'))
        mlflow.log_artifact(os.path.join(output_dir, 'val_acc.png'))
        mlflow.log_artifact(os.path.join(output_dir, 'val_loss.png'))
        mlflow.log_artifact(os.path.join(output_dir, 'lr.png'))
        #mlflow.log_artifact(os.path.join(output_dir, 'CM_without_normalize.png'))

        if model is not None:
            # kerasの場合、モデルをmlflow.kerasで保存できる
            # ./mlrunsのartifactsにconda.yamlやmodel.h5などができる。
            os.makedirs(os.path.join('models', 'data'), exist_ok=True)
            mlflow.keras.log_model(model, 'models')

    mlflow.search_runs()

def mlflow_callback(self, study, trial):
    """
    Optunaのobjectiveの最後に実行するcallback
    引数にtrialあるからObjectiveクラスの中に置かないと使えない
    """
    trial_value = trial.value if trial.value is not None else float("nan")
    with mlflow.start_run(experiment_id=self.experiment.experiment_id, run_name=study.study_name):
        mlflow.log_params(trial.user_attrs)  # mlflowで属性情報記録
        mlflow.log_params(trial.params)  # mlflowでパラメータ記録
        mlflow.log_metrics({"mean_squared_error": trial_value})  # mlflowでmetric記録