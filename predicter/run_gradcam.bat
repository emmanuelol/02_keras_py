@rem �쐬��2020/3/14 Grad-CAM���s����

call activate tfgpu113
cd C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py\predicter

@rem --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@rem ���s�e�X�g
call python grad_cam.py -i C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py\experiment\01_code_test\creative_commons_elephant.png
call python grad_cam.py -i C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py\experiment\01_code_test\cat_dog.png
call python grad_cam.py -i C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py\experiment\01_code_test\horse_gray.jpg

call python grad_cam.py ^
-i C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py\experiment\01_code_test\creative_commons_elephant.png ^
-m D:\work\keras_iPhone_pictures\01_classes_results_tfgpu_py36\20190529\train_all\finetuning.h5 ^
-o_j D:\work\kaggle_data\tf_Cats_VS._Dogs\results\tf_Cats_VS._Dogs_test\grad_cam\_test\grad_creative_commons_elephant.jpg

@rem --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

pause