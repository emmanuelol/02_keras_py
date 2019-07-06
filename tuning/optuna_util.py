# -*- coding: utf-8 -*-
"""
Optunaに関連するutil関数
"""

def trial_plot(out_dir, result_df, epochs=150, val_name="val_loss", trial_id=None):
    """
    optunaのtrial結果plot
    →optuna_lgb.tsvの列名が「intermediate_values.n」じゃないときがあったのでうまく機能しないときあり。。。
    
    以下のモジュールで代用できるのでいらなくなった（20190314）
    # optunw v 0.7.0
    from optuna.visualization import plot_intermediate_values
    plot_intermediate_values(study)

    Arges:
        out_dir: plot画像出力先ディレクトリ
        result_df: optunaのtrial結果のデータフレーム
        epochs: optunaで回したepoch数
        val_name: optunaで評価した評価指標  "val_loss" や "val_acc"など
        trial_id: 特定のtrial_id のログだけ見たい場合つかう。trial_id の番号をint or listで指定する (例. trial_id=2 trial_id=[2,45,46,47,48,49])
    Return:
        なし（plot表示し、trial_log.png 出力）
    """
    import os
    import matplotlib.pyplot as plt

    # 学習履歴の列名（'intermediate_values.n'）取得
    col_list = []
    for i in range(epochs):
        if i == 0:
            col_list.append('intermediate_values')
        else:
            col_list.append('intermediate_values.'+str(i))
    log_df = result_df[col_list]# 学習履歴の列（'intermediate_values.n'）だけ抽出
    #log_df = log_df.fillna(0)# 枝刈りしたepochはNaNになるので0で置換
    log_df_T = log_df.T# trial 単位でplotしやすいように転置する

    # trial結果 plot
    plt.figure(figsize=(10, 10),dpi=100)# 図大きくする
    if trial_id is not None:
        # 特定のtrial_id の結果だけplot
        if isinstance(trial_id, int):
            plt.plot(log_df_T[0], log_df_T[trial_id], label='trial_id_'+str(trial_id), marker='.')
        # 複数のtrial_id の結果plot
        elif isinstance(trial_id, list):
            for i in trial_id:
                plt.plot(log_df_T[0], log_df_T[i], label='trial_id_'+str(i), marker='.')
    else:
        # 全trial結果 plot
        for i in range(1,len(log_df_T.columns)):
            plt.plot(log_df_T[0], log_df_T[i], label='trial_id_'+str(i-1))
    plt.xlabel("epoch")
    plt.ylabel(val_name)
    plt.ylim([0, 1.3])# y軸の値していないとき変になることがあった
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)# 凡例枠外に書く
    plt.grid()
    plt.savefig(os.path.join(out_dir, 'trial_log.png'), bbox_inches="tight")# label見切れ対策 bbox_inches="tight"
    plt.show()
    plt.clf()

if __name__ == '__main__':
    print('optuna_util.py: loaded as script file')
else:
    print('optuna_util.py: loaded as module file')
