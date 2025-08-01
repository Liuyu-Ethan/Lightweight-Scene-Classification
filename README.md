# Lightweight-Scene-Classification
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
Remote sensing scene classification model of only 171KB size with 99.86% accuracy，base on RSI-CB128

### Remote Sensing Scene Classification with MobileNetV2
## Introduction
This is a lightweight convolutional neural network (CNN) model based on the **MobileNetV2** architecture, designed for efficient classification of 20 remote sensing scene classes in the **RSI-CB128** dataset. The project focuses on minimizing model size to the extreme while maintaining high accuracy, making it suitable for deployment in resource-constrained environments.

---

#### Features

* **Accuracy**: Achieves an accuracy of **99.86%** on the RSI-CB128 dataset.
* **Lightweight**: The final model size is only **171 KB**, making it ideal for applications on mobile devices, edge computing, and other scenarios with strict storage and computational resource requirements.
* **Architecture Design**: The project is built upon MobileNetV2, utilizing its core design principles such as **Inverted Residuals** and **Linear Bottlenecks** to achieve efficient feature extraction.

---

#### Dataset

This project uses the **RSI-CB128** dataset provided by Northwestern Polytechnical University.
* **Dataset Description**: RSI-CB128 is an image dataset containing 20 classes of remote sensing scenes, widely used for research in remote sensing image classification tasks.
* **Dataset Link**: [https://github.com/lehaifeng/RSI-CB](https://github.com/lehaifeng/RSI-CB)

---

## Installation

1.  Clone this repository to your local machine:
    ```bash
    git clone [https://github.com/Liuyu-Ethan/Lightweight-Scene-Classification.git](https://github.com/Liuyu-Ethan/Lightweight-Scene-Classification.git)
    cd Lightweight-Scene-Classification
    ```

2.  Create and activate a Python virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate # on Windows, use `venv\Scripts\activate`
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

You can use the following command to train the model (example):
```bash
python train.py --model_type custom_cnn --data_dir RSI-CB128 --batch_size 32 --epochs 10 --lr 0.001
````

### Evaluation

Use the following command to evaluate a trained model:

```bash
python eval.py --model_type custom_cnn --weights_path ./custom_cnn_final.pth
```

The code also includes comparative tests for VGG16, which can be trained and evaluated by setting command-line options.

## License

This project is licensed under the [MIT](LICENSE) .

## How to Cite

If you use this project in your research, please consider citing:

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

#### Contributions and Acknowledgments

  * Thanks to **lehaifeng** for providing the RSI-CB dataset.
  * [PyTorch](https://pytorch.org/)
  * [timm](https://github.com/rwightman/pytorch-image-models)
  * Contributions are welcome via Issues or Pull Requests to improve this project.

<!-- end list -->

```
```
