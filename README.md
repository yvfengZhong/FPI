# FPI

本项目是论文《CME Arrival Time Prediction via Fusion of Physical Parameters and Image Features》的代码实现。

## 安装

本项目使用anaconda作为包管理器，基于python==3.7.10、scikit-learn==0.19.2和torch==1.7.1进行开发。更多依赖库请查看requirements.txt文件，请确保在运行代码前本地已安装这些依赖库。

```sh
conda create -n cme python==3.7.10
conda activate cme
pip install -r requirements.txt
```

## 模型

模型存放在[阿里云盘](https://www.alipan.com/s/iztrcn7UScm)，提取码: j70w。下载后解压到FPI文件夹下。

## 使用说明

### PFE模型实验
#### 测试svr模型。
```sh
sh test_svr.sh
```
测试结果保存在checkpoints/cat_puma/log_test.txt文件中。

#### 训练mlp模型。
使用scripts/mlp.ipynb文件进行训练。

#### 测试mlp模型。
```sh
sh test_mlp.sh
```
测试结果保存在checkpoints/mlp/log_test.txt文件中。

### IFE模型实验
#### 训练resnet18模型。
```sh
sh train_ife.sh
```
使用train_ife.sh文件进行训练。

#### 测试resnet18模型。
```sh
sh test_img.sh
```
测试结果保存在checkpoints/resnet18/log_test.txt文件中。

#### 训练vgg11模型。
```sh
sh train_ife_vgg.sh
```
使用train_ife_vgg.sh文件进行训练。

#### 测试vgg11模型。
```sh
sh test_vgg11.sh
```
测试结果保存在checkpoints/vgg11/log_test.txt文件中。

#### 训练densenet121模型。
```sh
sh train_ife_densenet.sh
```
使用train_ife_densenet.sh文件进行训练。

#### 测试densenet121模型。
```sh
sh test_densenet121.sh
```
测试结果保存在checkpoints/densenet121/log_test.txt文件中。

#### 训练mobilenet模型。
```sh
sh train_ife_mobilenet.sh
```
使用train_ife_mobilenet.sh文件进行训练。

#### 测试mobilenet模型。
```sh
sh test_mobilenet.sh
```
测试结果保存在checkpoints/mobilenet/log_test.txt文件中。

### FF模型实验
#### 训练attention模型。
```sh
sh train_ff.sh
```
使用train_ife_mobilenet.sh文件进行训练。

#### 测试attention模型。
```sh
sh test_ff.sh
```
测试结果保存在checkpoints/attention/log_test.txt文件中。