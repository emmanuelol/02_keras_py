# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile

from tensorflow import keras

def run(model, gen, batch_size:int, steps_per_epoch:int, output_dir='./'):
    """
    modelとトレーニングデータからLearningRateFinder実行する
    引数のgenがトレーニングデータ（NumPy配列またはflow済みImageDataGenerator）
    """
    lrf = LearningRateFinder(model)
    lrf.find(gen
            , 1e-10, 1e+1 # 最小最大学習率
            , stepsPerEpoch=steps_per_epoch
            , batchSize=batch_size)
    lrs, losses = lrf.plot_loss()
    plt.savefig(os.path.join(output_dir,'lrfind_plot.jpg'))
    df_lrs = pd.DataFrame({'lrs':lrs, 'losses':losses})
    df_lrs.to_csv(os.path.join(output_dir,'lrfinder.tsv'), sep='\t', index=False)

class LearningRateFinder:
    """
    kerasのモデルで最適な学習率探索するクラス（1batchごとにで学習率を上げてlossの増減をplotする）
    https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/

    Usage:
        from lr_finder import LearningRateFinder

        # モデル準備
        opt = SGD(lr=config.MIN_LR, momentum=0.9)
        model = get_VGG16_model(img_rows=32, img_cols=32, channels=1, weights=None)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        # ImageDataGenerator準備
        aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

        # kerasのモデルで最適な学習率探索するクラスのインスタンス
        lrf = LearningRateFinder(model)

        # kerasのモデルで最適な学習率探索（1e-10から1e + 1の範囲の学習率でトレーニング）
        lrf.find(
            aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
            1e-10, 1e+1, # 最小最大学習率
            stepsPerEpoch=np.ceil((len(trainX) / float(config.BATCH_SIZE))),
            batchSize=config.BATCH_SIZE)

        # 学習率の損失プロット結果をディスクに保存
        lrs, losses = lrf.plot_loss()
        plt.savefig(config.LRFIND_PLOT_PATH)

        # 学習率の損失tsvファイルに保存
        df_lrs = pd.DataFrame({'lrs':lrs, 'losses':losses})
        display(df_lrs.head())
        df_lrs.to_csv('lr_finder_EfficientNet4.tsv', sep='\t', index=False)
    """
    def __init__(self, model, stopFactor=4, beta=0.98):
        # モデル、停止係数、ベータ値を保存（平滑化された平均損失を計算するため）
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta

        # 学習率と損失のリストをそれぞれ初期化
        self.lrs = []
        self.losses = []

        # lrMult：学習率の乗算係数。
        # avgLoss：経時的な平均損失。
        # bestLoss：トレーニング時に発見した最高の損失。
        # batchNum：現在のバッチ番号の更新。
        # weightsFile：初期モデルの重みへのパス（したがって、学習率が見つかった後、トレーニングの前に重みを再ロードして初期値に設定できます）。
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def reset(self):
        # コンストラクターからすべての変数を再初期化します
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def is_data_iter(self, data):
        # チェックするクラスタイプのセットを定義する
        iterClasses = ["NumpyArrayIterator", "DirectoryIterator", "DataFrameIterator", "Iterator", "Sequence", "generator"]

        # データがイテレータかどうかを返します
        return data.__class__.__name__ in iterClasses

    def on_batch_end(self, batch, logs):
        # 現在の学習率を取得し、試行した学習率のリストにログを追加します
        lr = keras.backend.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # このバッチの最後に損失を取得し、
        # 処理されたバッチの総数を増やし、
        # 平均損失の平均を計算し、平滑化し、平滑化された値で損失リストを更新します
        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)

        # 最大損失停止係数値を計算する
        stopLoss = self.stopFactor * self.bestLoss

        # 損失が大きくなりすぎていないか確認
        if self.batchNum > 1 and smooth > stopLoss:
            # 戻りを停止し、メソッドから戻ります
            self.model.stop_training = True
            return

        # 最高の損失を更新する必要があるかどうかを確認
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth

        # 学習率を上げる
        lr *= self.lrMult
        keras.backend.set_value(self.model.optimizer.lr, lr)

    def find(self, trainData, startLR, endLR
             , epochs=None, stepsPerEpoch=None, batchSize=32, sampleSize=2048, verbose=1):
        """
        学習率を探索する
        Args:
            trainData：トレーニングデータ（NumPy配列またはデータジェネレーター）。
            startLR：初期の開始学習率。
            epochs：トレーニングするエポックの数（値が指定されていない場合、エポックの数を計算します）。
            stepsPerEpoch：各エポックごとのバッチ更新ステップの総数。
            batchSize：オプティマイザーのバッチサイズ。
            sampleSize： trainDataから最適な学習率を見つけるときに使用するサンプルの数 。
            verbose：fit, fit_generatorのverbose
        """

        # クラス固有の変数をリセットします
        self.reset()

        # データジェネレータを使用しているかどうかを判断する
        useGen = self.is_data_iter(trainData)
        print('useGen:', useGen)

        # ジェネレータを使用していて、エポックごとのステップが指定されていない場合、エラーが発生します
        if useGen and stepsPerEpoch is None:
            msg = "Using generator without supplying stepsPerEpoch"
            raise Exception(msg)

        # ジェネレータを使用していない場合は、データセット全体が既にメモリ内にある必要があります
        elif not useGen:
            # トレーニングデータのサンプル数を取得し、エポックごとのステップ数を導出します
            numSamples = len(trainData[0])
            stepsPerEpoch = np.ceil(numSamples / float(batchSize))

        # トレーニングエポックの数が指定されていない場合、
        # デフォルトのサンプルサイズに基づいてトレーニングエポックを計算します
        if epochs is None:
            epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))

        # 適切な開始学習率を見つけようとしている間に行われるバッチ更新の総数を計算する
        numBatchUpdates = epochs * stepsPerEpoch

        # 終了学習率、開始学習率、およびバッチ更新の総数に基づいて、学習率の乗数を導き出します
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)

        # モデルの重みの一時ファイルパスを作成し、重みを保存します（完了したら重みをリセットできます）
        self.weightsFile = tempfile.mkstemp()[1]
        #print(self.weightsFile)
        #print(self.model)
        self.model.save_weights(self.weightsFile)

        # （元の）学習率を取得し（後でリセットできるようにします）、次に*開始*学習率を設定します
        origLR = keras.backend.get_value(self.model.optimizer.lr)
        keras.backend.set_value(self.model.optimizer.lr, startLR)

        # 各バッチの終わりに呼び出されるコールバックを作成し、トレーニングが進行するにつれて学習率を上げることができるようにします
        callback = keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs:
            self.on_batch_end(batch, logs))

        # データ反復子を使用しているかどうかを確認します
        if useGen:
            self.model.fit_generator(
                trainData,
                steps_per_epoch=stepsPerEpoch,
                epochs=epochs,
                verbose=verbose,
                callbacks=[callback])

        # それ以外の場合、トレーニングデータ全体が既にメモリにあります
        else:
            # train our model using Keras' fit method
            self.model.fit(
                trainData[0], trainData[1],
                batch_size=batchSize,
                epochs=epochs,
                callbacks=[callback],
                verbose=verbose)

        # 元のモデルの重みと学習率を復元する
        self.model.load_weights(self.weightsFile)
        keras.backend.set_value(self.model.optimizer.lr, origLR)

    def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
        # 学習率と損失値を取得してプロットする。lrs, lossesも返す
        lrs = self.lrs[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]

        # 学習率と損失のプロット
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")

        # タイトルが空ではない場合、プロットにタイトル追加します
        if title != "":
            plt.title(title)
        return lrs, losses
