

"""
Randomized Image Sampling for Explanations (RISE) 実行モジュール
https://github.com/eclique/RISE

RISE: RIMEみたいに入力画像マスクしてCNNの重要な領域可視化する手法
論文ではGrad-CAMより対象物をはっきり可視化してたケースあり
Grad-CAMはGlobal Average Poolig(GAP)で1次元化してるNNでないと使えない（はず）だが、
RISEは画像隠してモデルの重要領域探してるだけなので、GAPの制限なしで使えるはず

RISEの制限
- RGBの3チャネル持つCNNである必要っぽい
- マスクする割合のパラメータ(N,s,p1)に依存しそう

このモジュールは
https://github.com/eclique/RISE/blob/master/Easy_start.ipynb を参考にしたもの

Usage:
    $ python tf_rise.py  # imagenetのResnet50で犬猫画像についてRISE実行
"""
import os
import sys
import pathlib
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm

# tensorflowのINFOレベルのログを出さないようにする
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K


class Model():
    def __init__(self, model=None, classes=None):
        if model is None:
            self.model = ResNet50()
            self.input_size = (224, 224)
            self.classes = None
        else:
            self.model = model
            self.input_size = (model.input_shape[1], model.input_shape[2])  # モデルオブジェクトの入力層のサイズ取得
            self.classes = classes

    def run_on_batch(self, x):
        return self.model.predict(x)

    def class_name(self, idx):
        if self.model.name == 'resnet50' and self.classes is None:
            # imagenetのResNet50()の場合
            return decode_predictions(np.eye(1, 1000, idx))[0][0][1]
        else:
            return self.classes[idx] if self.classes is not None else str(idx)

    def load_img(self, path):
        img = image.load_img(path, target_size=self.input_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if self.model.name == 'resnet50' and self.classes is None:
            # imagenetのResNet50()の場合
            x = preprocess_input(x)
        else:
            # 1/255.0の前処理
            x = x / 255.0
        return img, x


class Rise():
    def __init__(self, model, N=2000, s=8, p1=0.5):
        self.model = model
        self.N = N
        self.s = s
        self.p1 = p1

    def generate_masks(self):
        cell_size = np.ceil(np.array(self.model.input_size) / self.s)
        up_size = (self.s + 1) * cell_size

        grid = np.random.rand(self.N, self.s, self.s) < self.p1
        grid = grid.astype('float32')

        masks = np.empty((self.N, *self.model.input_size))

        for i in tqdm(range(self.N), desc='Generating masks'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + self.model.input_size[0], y:y + self.model.input_size[1]]
        masks = masks.reshape(-1, *self.model.input_size, 1)
        return masks

    def explain(self, inp, masks, batch_size=100):
        preds = []
        # 正しい軸に対して乗算が行われていることを確認する
        masked = inp * masks
        for i in tqdm(range(0, self.N, batch_size), desc='Explaining'):
            preds.append(self.model.run_on_batch(masked[i:min(i + batch_size, self.N)]))
        preds = np.concatenate(preds)
        sal = preds.T.dot(masks.reshape(self.N, -1)).reshape(-1, *self.model.input_size)
        sal = sal / self.N / self.p1
        return sal


def main(image_path=r'C:\Users\81908\jupyter_notebook\tfgpu_py36_work\02_keras_py\experiment\01_code_test\cat_dog.png',
         class_idx=None,  # 243,
         model_path=None,
         classes=None,
         output_dir=None):
    """
    RISE画像可視化
    Args:
        image_path: RISE実行する画像
        class_idx: 可視化するクラスid。Noneなら確率最大のクラスidにする
        model_path: ロードするモデルファイル(*.h5)。NoneならimagenetのResnet50で実行する
        classes: クラス名リスト。指定したら可視化画像のタイトルに指定クラス名が入る。Noneでも良い
        output_dir: RISE画像保存先ディレクトリ
    """
    # model load
    K.clear_session()
    K.set_learning_phase(0)
    model = Model(model=keras.models.load_model(model_path, compile=False), classes=classes) if model_path is not None else Model()

    # 画像前処理
    img, x = model.load_img(image_path)

    # class_idx指定なければ予測スコア最大クラスにする
    class_idx = np.argmax(model.run_on_batch(x)[0]) if class_idx is None else class_idx

    # RISE
    rise = Rise(model)
    masks = rise.generate_masks()
    sal = rise.explain(x, masks)

    # 可視化
    class_name = model.class_name(class_idx)
    plt.title('Explanation for `{}`'.format(class_name))
    plt.axis('off')
    plt.imshow(img)
    plt.imshow(sal[class_idx], cmap='jet', alpha=0.5)
    # plt.colorbar()
    if output_dir is not None:
        out_jpg = os.path.join(output_dir, str(pathlib.Path(image_path).stem) + f'_{class_name}_rise.jpg')
        plt.savefig(out_jpg, bbox_inches='tight', pad_inches=0)  # bbox_inchesなどは余白削除オプション
    plt.show()


if __name__ == '__main__':
    # test
    matplotlib.use('Agg')
    main()
