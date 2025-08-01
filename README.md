# Lightweight-Scene-Classification
Remote sensing scene classification model of only 171KB size with 99.86% accuracy，base on RSI-CB128

### Remote Sensing Scene Classification with MobileNetV2

这是一个基于 **MobileNetV2** 架构实现的轻量化卷积神经网络（CNN）模型，用于对 **RSI-CB128** 数据集中的 20 类遥感场景进行高效分类。该项目专注于在保证高精度的同时，将模型大小压缩到极致，以适应资源受限的部署环境。

---

#### 核心成果

* **卓越的分类精度**：在 RSI-CB128 数据集上达到了 **99.86%** 的惊人准确率。
* **极致的模型压缩**：最终模型大小仅为 **171 KB**，非常适合移动设备、边缘计算等对存储和计算资源有严格要求的应用场景。
* **轻量化架构**：项目基于 MobileNetV2，利用其**倒残差块**（Inverted Residuals）和**线性瓶颈层**（Linear Bottlenecks）等核心设计，实现了高效的特征提取。

---

#### 数据集

本项目使用了由西北工业大学提供的 **RSI-CB128** 数据集。
* **数据集简介**：RSI-CB128 是一个包含 20 类遥感场景的图像数据集，广泛用于遥感图像分类任务的研究。
* **数据集链接**：[https://github.com/lehaifeng/RSI-CB](https://github.com/lehaifeng/RSI-CB)

---

#### 项目结构

本项目包含以下主要文件：

* `notebook_name.ipynb`：核心 Jupyter Notebook，包含了数据预处理、模型构建、训练、评估和模型保存的完整流程。
* `model_weights.h5` 或 `.pth`：训练好的模型文件，可直接用于推理。

---

#### 如何使用

1.  **克隆仓库**：
    ```bash
    git clone [https://github.com/YourUsername/YourRepositoryName.git](https://github.com/YourUsername/YourRepositoryName.git)
    cd YourRepositoryName
    ```
2.  **安装依赖**：
    ```bash
    pip install -r requirements.txt
    ```
3.  **运行 Notebook**：
    在 Jupyter 环境中打开 `notebook_name.ipynb` 文件，按照步骤运行代码即可复现模型的训练和评估过程。

---

#### 贡献与鸣谢

* 感谢 **lehaifeng** 提供的 RSI-CB 数据集。
* 欢迎通过提交 Issue 或 Pull Request 来改进本项目。
