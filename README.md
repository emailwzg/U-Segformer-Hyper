[![DOI](https://zenodo.org/badge/543462554.svg)](https://zenodo.org/badge/latestdoi/543462554)

# U-Segformer-Hyper

Training and testing can run four files directly (patch_train.py，patch_test.py，section_train.py，section_test.py).
code file:
core/models/USegformerHyper.py：Network structure of U-Segformer-Hyper、U-Segformer、Segformer
core/models/patch_deconvnet.py：Network structure of patch-based Benchmark model
core/models/section_deconvnet.py：Network structure of section-based Benchmark model

patch_train.py：U-Segformer-Hyper,U-Segformer and Segformer are trained by patch-based mode
patch_test.py：U-Segformer-Hyper,U-Segformer and Segformer are tested by patch-based mode

section_train.py：U-Segformer-Hyper,U-Segformer and Segformer are trained by section-based mode
section_test.py：U-Segformer-Hyper,U-Segformer and Segformer are tested by section-based mode

inline200-xline300.ipynb：Used to draw prediction sections
metricget-5-5.ipynb：Used to draw confusion matrix 

data file:
runs-section：Saves the predicted data of section-based mode
runs-patch：Saves the predicted data of patch-based mode

data：Training set and test set

inline：Predicted inline sections
xline：Predicted xline sections

confusion_55：Confusion matrix for the test set

================================================================Chinese version 中文说明=========


训练和测试直接运行patch_train.py，patch_test.py，section_train.py，section_test.py四个文件即可。
代码说明：
core/models/USegformerHyper.py：U-Segformer-Hyper、U-Segformer、Segformer的网络结构
core/models/patch_deconvnet.py：基于patch模式的Benchmark网络
core/models/section_deconvnet.py：基于section模式的Benchmark网络

patch_train.py：基于patch模式训练的U-Segformer-Hyper、U-Segformer、Segformer网络
patch_test.py：用基于patch模式训练的U-Segformer-Hyper、U-Segformer、Segformer网络进行测试

section_train.py：基于section模式训练的U-Segformer-Hyper、U-Segformer、Segformer网络
section_test.py：用基于section模式训练的U-Segformer-Hyper、U-Segformer、Segformer网络进行测试

inline200-xline300.ipynb：用于绘制预测剖面
metricget-5-5.ipynb：绘制测试集的混淆矩阵


数据：
runs-section：保存基于section模式的预测数据
runs-patch：保存基于patch模式的预测数据

data：训练集和测试集

inline：预测的inline剖面
xline：预测的xline剖面

confusion_55：测试集的混淆矩阵
