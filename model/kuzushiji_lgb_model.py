# -*- coding: utf-8 -*-
"""自作metric"""

%%time

import lightgbm as lgb
import optuna, os, uuid, pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
# optunw v 0.7.0
from optuna.visualization import plot_intermediate_values

def train_optuna(X_train, y_train, X_test, y_test,
                 out_dir='lgb_output',
                 is_save_model=False,
                 n_trials=100):

    #data = load_breast_cancer()
    #X_train, X_test, y_train, y_test = train_test_split(data["data"], data["target"], test_size=0.3, random_state=19)

    def objectives(trial):
        ## 試行にUUIDを設定
        #trial_uuid = str(uuid.uuid4())

        # boosting_typeについて
        # gbdt：Gradient boosting decision tree（いわゆる勾配ブースティング）
        # dart：Dropouts meet Multiple Additive Regression Trees
        # goss：Gradient-based One-Side Sampling
        params = {
            'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
            'objective': 'multiclass',#'binary',
            'num_class': 10, # ターゲットクラスは10種類の平仮名ですのでnum_classは10
            'metric': {'multi_logloss', 'multi_error'},#{'binary', 'binary_error', 'auc'},
            'num_leaves': trial.suggest_int("num_leaves", 10, 500),
            'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1),
            #'feature_fraction': trial.suggest_uniform("feature_fraction", 0.0, 1.0), # 特徴量サンプリングの割合
            #'device' : 'gpu', # gpu版は使えない。そもそもLightGBMはGPUにしても早くならない場合もあるみたい https://qiita.com/TomokIshii/items/95c49d5e86f2eae61720
            'verbose' : 2 #0
        }
        if params['boosting_type'] == 'dart':
            params['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
            params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
        if params['boosting_type'] == 'goss':
            params['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
            params['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - params['top_rate'])

        # 枝刈りありの訓練
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_error") # "multi_logloss" # 正式名で呼ばないとダメなので注意
        #pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "binary_logloss") # 正式名で呼ばないとダメなので注意
        gbm = lgb.train(params,
                        lgb.Dataset(X_train, y_train),
                        num_boost_round=200, # 500,  # number of boosting rounds
                        valid_sets=lgb.Dataset(X_test, y_test),
                        callbacks=[pruning_callback],
                        early_stopping_rounds=50, # early_stoppingで途中で止める場合 50 ラウンド経過しても性能が向上しないときは学習を打ち切る
                        verbose_eval=100 # 100roundごとに評価表示
                       )

        # 訓練、テスト誤差
        y_pred_train = list(map(lambda x: np.argmax(x), gbm.predict(X_train)))
        #y_pred_train = np.rint(gbm.predict(X_train)) # np.rint()は四捨五入 2クラス分類用
        y_pred_test = list(map(lambda x: np.argmax(x), gbm.predict(X_test)))
        #y_pred_test = np.rint(gbm.predict(X_test)) # np.rint()は四捨五入 2クラス分類用

        error_train = 1.0 - accuracy_score(y_train, y_pred_train)
        error_test = 1.0 - accuracy_score(y_test, y_pred_test)

        # エラー率の記録
        trial.set_user_attr("train_error", error_train)
        trial.set_user_attr("test_error", error_test)

        # 最も良いスコアのときのroundをoptimum_boost_roundsに格納
        trial.set_user_attr("best_iteration", gbm.best_iteration)

        if is_save_model == True:
            # モデルの保存 LightGBMはモデルをPickleとして書き出すことで、モデルの保存／読み込みができる
            #with open(out_dir+"/"+f"{trial_uuid}.pkl", "wb") as fp:
            with open(out_dir+"/"+str(trial.trial_id-1)+".pkl", "wb") as fp:
                pickle.dump(gbm, fp)

        return error_test


    #study = optuna.create_study()
    # SQLiteに記録する場合は、ディスクアクセスが遅いとボトルネックになることもある
    #study = optuna.create_study(storage="sqlite:///brestcancer_lgb.db", study_name="brestcancer_lgb")
    # sqlite 使って履歴ファイル作る
    os.makedirs(out_dir, exist_ok=True)
    sqllite_path = out_dir+'/lgb.db'
    if os.path.exists(sqllite_path) == True:
        os.remove(sqllite_path) # sqllite_pathすでにあれば一旦削除
    study = optuna.create_study(study_name='lgb', storage='sqlite:///'+sqllite_path)

    # Optuna exe
    study.optimize(objectives, n_trials=n_trials)

    # 一番良い試行結果を表示
    print(study.best_params)
    print(study.best_value)

    # best_paramsにはuser_attrは表示されないのでtrialから呼ぶ（dict形式で記録されている）
    print(study.best_trial.user_attrs)

    # 試行履歴csvに保存
    df = study.trials_dataframe()
    df.to_csv(out_dir+'/optuna_lgb.csv')
    df.to_excel(os.path.join(out_dir, 'optuna_lgb.tsv.xlsx'))

    # 試行履歴plot
    plot_intermediate_values(study)

if __name__ == '__main__':
    train_optuna(X_train, y_train, X_test, y_test,
                 out_dir=r'D:\work\tmp\LigtGBM_test\lgb_output',
                 is_save_model=False,
                 n_trials=100)
