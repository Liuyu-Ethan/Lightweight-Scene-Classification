# Lightweight-Scene-Classification
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
Remote sensing scene classification model of only 171KB size with 99.86% accuracy，base on RSI-CB128

### Remote Sensing Scene Classification with MobileNetV2
## 简介
这是一个基于 **MobileNetV2** 架构实现的轻量化卷积神经网络（CNN）模型，用于对 **RSI-CB128** 数据集中的 20 类遥感场景进行高效分类。该项目专注于在保证高精度的同时，将模型大小压缩到极致，以适应资源受限的部署环境。

---

#### 特性

* **准确率**：在 RSI-CB128 数据集上达到了 **99.86%** 的惊人准确率。
* **轻量化**：最终模型大小仅为 **171 KB**，非常适合移动设备、边缘计算等对存储和计算资源有严格要求的应用场景。
* **架构设计**：项目基于 MobileNetV2，利用其**倒残差块**（Inverted Residuals）和**线性瓶颈层**（Linear Bottlenecks）等核心设计，实现了高效的特征提取。

---

#### 数据集

本项目使用了由西北工业大学提供的 **RSI-CB128** 数据集。
* **数据集简介**：RSI-CB128 是一个包含 20 类遥感场景的图像数据集，广泛用于遥感图像分类任务的研究。
* **数据集链接**：[https://github.com/lehaifeng/RSI-CB](https://github.com/lehaifeng/RSI-CB)

---



## 安装

1.  克隆本仓库到本地：
    ```bash
    git clone [https://github.com/Liuyu-Ethan/Lightweight-Scene-Classification.git](https://github.com/Liuyu-Ethan/Lightweight-Scene-Classification.git)
    cd Lightweight-Scene-Classification
    ```

2.  创建并激活 Python 虚拟环境（推荐）：
    ```bash
    python -m venv venv
    source venv/bin/activate  # on Windows, use `venv\Scripts\activate`
    ```

3.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

## 数据集准备

请在此处详细说明所使用的数据集，以及如何准备数据集。例如：

1.  下载 `[数据集名称]` 数据集。
2.  将数据集解压至 `data/` 目录下，目录结构如下：
    ```
    data/
    ├── train/
    │   ├── class1/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── class2/
    └── val/
        ├── class1/
        └── class2/
    ```

## 使用

### 训练

您可以使用以下命令来训练模型（示例）：
```bash
python train.py --model_type custom_cnn --data_dir RSI-CB128 --batch_size 32 --epochs 10 --lr 0.001
```

### 评估

使用以下命令来评估已训练好的模型：

```bash
python eval.py --model_type custom_cnn --weights_path ./custom_cnn_final.pth
```
同时代码中还包含VGG16的对比测试，可根据命令行设置自行训练、评估

## 许可

本项目采用 [MIT](LICENSE) 许可。

## 如何引用

如果您在研究中使用了本项目，请考虑引用：

```bibtex
@misc{LightweightSceneClassification,
  author = {Yu, Liu},
  title = {Lightweight Scene Classification},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/Liuyu-Ethan/Lightweight-Scene-Classification](https://github.com/Liuyu-Ethan/Lightweight-Scene-Classification)}}
}
```


#### 贡献与鸣谢

* 感谢 **lehaifeng** 提供的 RSI-CB 数据集。
* [PyTorch](https://pytorch.org/)
* [timm](https://github.com/rwightman/pytorch-image-models)
* 欢迎通过提交 Issue 或 Pull Request 来改进本项目。
