# 02_keras_py
keras2.1.5+tensorflow1.8+lightgbm2.2.3+optuna0.7.0  
My Library for creating image classification models for Windows(GeForce 1080)  

## Setup
- Anaconda 4.4.10: https://www.anaconda.com/distribution/
- NVIDIA Driver for GeForce 1080: http://www.nvidia.co.jp/Download/index.aspx?lang=jp
- Visual Studio 2015 Update 3(Visual Studio Community 2015 with Update 3): https://my.visualstudio.com/Downloads  
	- Custom installation of Visual Studio 2015
- Visual C ++ Redistributable Package for Visual Studio 2015: https://www.microsoft.com/ja-JP/download/details.aspx?id=48145
- CUDA Toolkit 9.0: https://developer.nvidia.com/cuda-90-download-archive  
	- Installation execution except "Visual studio integration"
- cuDNN v7.0.5: https://developer.nvidia.com/rdp/cudnn-download
```bash
$ conda create -n tfgpu_py36_v3
$ activate tfgpu_py36_v3
$ conda install python=3.6
$ conda install -c anaconda tensorflow-gpu=1.8 
$ conda install -c conda-forge keras=2.1.5 
$ conda install -c conda-forge lightgbm=2.2.3 scikit-learn=0.20.3 opencv=4.1.0 grpcio=1.16 numba=0.38.1 pandas jupyter Cython Protobuf Pillow lxml Matplotlib tqdm future graphviz pydot pytest pyperclip networkx selenium beautifulsoup4 cssselect openpyxl pypdf2 python-docx requests tweepy textblob seaborn scikit-image imbalanced-learn colorlog sqlalchemy papermill shapely imageio git shap eli5 umap-learn plotly ipysheet bqplot rise bokeh jupyter_contrib_nbextensions yapf flask joblib xgboost alembic dill xlrd nose xlsxwriter lime
```

## Usage
```bash
import os, sys
sys.path.append(r'02_keras_py')
from dataset import plot_log, prepare_data, util, plot_12task_log, util, set_split
from transformer import get_train_valid_test, my_generator
from model import define_model, multi_loss, my_callback, my_metric, my_class_weight 
from predicter import roc_curve, conf_matrix, multi_predict, grad_cam, ensemble_predict, base_predict, grad_cam_util, visualize_keras_predict
from tuning import optuna_train_base

batch_size = 6
img_rows, img_cols, channels = 331, 331, 3

# data generator
data_dir = r'D:\work\kaggle_aptos2019-blindness-detection\OrigData\aptos2019-blindness-detection\train_images'
d_cls = get_train_valid_test.LabeledDataset([img_rows, img_cols, channels]
                                            , batch_size
                                            , valid_batch_size=batch_size
                                            , train_samples=train_df.shape[0]
                                            , valid_samples=valid_df.shape[0]
                                           )
d_cls.create_my_generator_flow_from_dataframe('x_paths'
                                              , 'y_ids'
                                              , train_df
                                              , data_dir
                                              , valid_df=valid_df
                                              , valid_data_dir=data_dir
                                              , color_mode='rgb'
                                              , class_mode='categorical'
                                              , my_IDG_options=my_IDG_options)
class_name = train_df['y_ids'].unique()

# model
model, orig_model = define_model.get_fine_tuning_model(output_dir
                                                       , img_rows, img_cols, channels
                                                       , len(class_name)
                                                       , choice_model, trainable
                                                       , FCnum=FCnum
                                                       , activation=activation
                                                       , efficientnet_num=efficientnet_num
                                                      )
# compile the model
optim = define_model.get_optimizers(choice_optim=choice_optim, lr=lr, momentum=momentum, nesterov=True)#, decay=decay)
model.compile(loss='categorical_crossentropy'
              , optimizer=optim
              , metrics=['acc'])

history = model.fit_generator(
    d_cls.train_gen
    , steps_per_epoch = d_cls.init_train_steps_per_epoch
    , epochs = num_epoch
    , validation_data = d_cls.valid_gen
    , validation_steps = d_cls.init_valid_steps_per_epoch
    , verbose = 2
    , callbacks = get_cb(output_dir, cosine_annealing_num_epoch=None)
    #, class_weight = class_weight 
    )

# validation generator predict TTA
load_model = keras.models.load_model(os.path.join(output_dir, 'best_val_acc.h5'))
pred_tta = base_predict.predict_tta_generator(load_model
                                              , d_cls.valid_gen
                                              , TTA='flip'
                                              #, TTA_rotate_deg=20
                                              #, TTA_crop_num=4, TTA_crop_size=[70,70]
                                              , resize_size=[img_rows, img_cols])
pred_tta_df = base_predict.get_predict_generator_results(pred_tta
                                                         , d_cls.valid_gen
                                                         , classes_list=classes_list)

# gradcam
grad_cam_img = grad_cam_util.gradcam_from_img_path(load_model
                                                   , data_paths[50]
                                                   , output_dir
                                                   , classes_list
                                                   , img_rows, img_cols
                                                   , layer_name=layer_name
                                                   , is_gradcam_plus=True)

# test files predict TTA
img_paths = util.find_img_files(base_image_dir+'/test_images')
pred_tta_df = base_predict.pred_tta_from_paths(load_model, img_paths, img_rows, img_cols
                                           , classes=None, show_img=False
                                           , TTA='flip'
                                           #, TTA_rotate_deg=0
                                           #, TTA_crop_num=0, TTA_crop_size=[224, 224]
                                           , preprocess=1.0/255.0)

```

<!-- 
## License
This software is released under the MIT License, see LICENSE.
-->

## Author
- Github: [riron1206](https://github.com/riron1206)