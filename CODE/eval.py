import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import get_dataloaders
from model import get_model
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Evaluate a CNN model on RSI-CB128 dataset')
    parser.add_argument('--model_type', type=str, required=True, choices=['custom_cnn', 'vgg16'],
                        help='Model type: custom_cnn or vgg16')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to trained model weights (e.g., custom_cnn_final.pth or vgg16_final.pth)')
    parser.add_argument('--data_dir', type=str, default='RSI-CB128',
                        help='Path to RSI-CB128 dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    return parser.parse_args()

def evaluate_model(model, test_loader, device, class_names, batch_size):
    """
    评估模型在测试集上的性能，计算各项指标
    Args:
        model: 待评估的模型
        test_loader: 测试数据加载器
        device: 设备 (CPU or GPU)
        class_names: 类别名称列表
        batch_size: 批次大小，用于计算吞吐量
    Returns:
        metrics: 包含各类别和整体指标的字典
        y_true: 真实标签
        y_pred: 预测标签
        y_scores: 预测概率
        avg_inference_time: 平均推理时间
        memory_increment: 平均内存增量
    """
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    total_inference_time = 0
    memory_increment = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            total_inference_time += inference_time

            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            process = psutil.Process(os.getpid())
            memory_increment += process.memory_info().rss / 1024 / 1024

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    avg_inference_time = total_inference_time / len(test_loader)
    throughput = batch_size / avg_inference_time
    avg_memory_increment = memory_increment / len(test_loader)

    metrics = {
        'class_precision': precision, 'class_recall': recall, 'class_f1': f1,
        'class_support': support, 'overall_precision': overall_precision,
        'overall_recall': overall_recall, 'overall_f1': overall_f1,
        'overall_accuracy': overall_accuracy, 'macro_precision': macro_precision,
        'macro_recall': macro_recall, 'macro_f1': macro_f1
    }

    print("\nPer-class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"Class {class_name}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}, Support: {support[i]}")

    print("\nOverall Metrics:")
    print(f"Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}, Accuracy: {overall_accuracy:.4f}")

    logger.info(f"验证损失: {loss.item():.4f}")
    logger.info(f"总体准确率: {overall_accuracy:.4f}")
    logger.info(f"宏平均精确率: {macro_precision:.4f}")
    logger.info(f"宏平均召回率: {macro_recall:.4f}")
    logger.info(f"宏平均F1分数: {macro_f1:.4f}")
    logger.info(f"平均每批次推理时间: {avg_inference_time:.4f} 秒")
    logger.info(f"吞吐量: {throughput:.2f} 样本/秒")
    logger.info(f"平均内存增量: {avg_memory_increment:.2f} MB")

    return metrics, y_true, y_pred, y_scores, avg_inference_time, avg_memory_increment

def plot_pr_roc_curves(y_true, y_scores, class_names, model_type, weights_path):
    """
    绘制P-R曲线和ROC曲线
    Args:
        y_true: 真实标签
        y_scores: 预测概率
        class_names: 类别名称列表
        model_type: 模型类型，用于保存文件名
        weights_path: 权重文件路径，用于生成唯一文件名
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    weights_name = os.path.basename(weights_path).split('.')[0]

    # P-R曲线
    plt.figure(figsize=(12, 8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_type}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tight_layout()
    pr_path = f'pr_curve_{model_type}_{weights_name}.png'
    plt.savefig(pr_path, bbox_inches='tight')
    plt.close()

    # ROC曲线
    plt.figure(figsize=(12, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_type}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tight_layout()
    roc_path = f'roc_curve_{model_type}_{weights_name}.png'
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()

    return pr_path, roc_path

def plot_confusion_matrix(y_true, y_pred, class_names, model_type, weights_path):
    """
    绘制混淆矩阵
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        model_type: 模型类型，用于保存文件名
        weights_path: 权重文件路径，用于生成唯一文件名
    Returns:
        cm_path: 混淆矩阵图像保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    weights_name = os.path.basename(weights_path).split('.')[0]

    # 绘制混淆矩阵
    plt.figure(figsize=(15, 12), dpi=100)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_type}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = f'confusion_matrix_{model_type}_{weights_name}.png'
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    return cm_path

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)
    test_loader = data['test_loader']
    class_names = data['class_names']
    print(f"Number of classes: {len(class_names)}")
    print(f"Testing samples: {len(test_loader.dataset)}")

    model = get_model(model_type=args.model_type, num_classes=len(class_names), pretrained=(args.model_type == 'vgg16'))
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"权重文件 {args.weights_path} 不存在")
    print(f"Loading weights from {args.weights_path}")

    state_dict = torch.load(args.weights_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    model = model.to(device)
    metrics, y_true, y_pred, y_scores, avg_inference_time, avg_memory_increment = evaluate_model(model, test_loader, device, class_names, args.batch_size)
    pr_path, roc_path = plot_pr_roc_curves(y_true, y_scores, class_names, args.model_type, args.weights_path)
    cm_path = plot_confusion_matrix(y_true, y_pred, class_names, args.model_type, args.weights_path)
    print(f"P-R curve saved as {pr_path}")
    print(f"ROC curve saved as {roc_path}")
    print(f"Confusion matrix saved as {cm_path}")

if __name__ == "__main__":
    main()