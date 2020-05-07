# -*- coding: utf-8 -*-
"""
google Colabに書いてるtensorflow2のチュートリアル-MNIST
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/advanced.ipynb?linkId=85251566&utm_campaign=piqcy&utm_medium=email&utm_source=Revue%20newsletter#scrollTo=DUNzJc4jTj6G
Usage:
    $ python ./tf2_quickstart_MNIST.py
"""
import tensorflow as tf
from tensorflow import keras


class MyModel(keras.Model):
    """
    Build the tf.keras model using the Keras model subclassing API
    __init__()で呼び出すレイヤー決めてcall()でレイヤー組み合わせる
    """
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(128, activation='relu')
        self.d2 = keras.layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


def load_mnist():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Use tf.data to batch and shuffle the dataset:
    # tf.dataはTensorFlowが提供する入力パイプライン設計用のモジュール
    # ### tf.data使用例 ### #
    # https://qiita.com/S-aiueo32/items/c7e86ef6c339dfb013ba より
    # filenames = glob.glob(DATA_DIR)  # filenameのリスト取得 (CSVとかから読んでも良い)
    # dataset = tf.data.Dataset.from_tensor_slices(filenames)  # filenamesからデータセット作成
    # dataset = dataset.batch(10)  # ミニバッチ化（指定したサイズのミニバッチに分割）
    # dataset = dataset.repeat(1)  # 指定した（ここでは1回だけ）回数データセットをリピート(指定しないと無限リピート)
    # dataset = dataset.shuffle(1000)  # ランダムシード指定してdatasetシャッフル
    # ##################### #
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    return train_ds, test_ds


if __name__ == '__main__':
    train_ds, test_ds = load_mnist()

    # Create an instance of the model
    model = MyModel()

    # Choose an optimizer and loss function for training:
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam()

    # Select metrics to measure the loss and the accuracy of the model.
    # These metrics accumulate the values over epochs and then print the overall result.
    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Use tf.GradientTape to train the model:
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {0}, Loss: {1:.2f}, Accuracy: {2:.2f}, Test Loss: {3:.2f}, Test Accuracy: {4:.2f}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))