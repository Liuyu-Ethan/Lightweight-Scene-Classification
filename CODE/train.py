import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model import get_model
import numpy as np


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train a CNN model on RSI-CB128 dataset')
    parser.add_argument('--model_type', type=str, required=True, choices=['custom_cnn', 'vgg16'],
                        help='Model type: custom_cnn or vgg16')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to pre-trained model weights (optional)')
    parser.add_argument('--data_dir', type=str, default='RSI-CB128',
                        help='Path to RSI-CB128 dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate for optimizer')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Finetune VGG16 convolutional layers (only for vgg16)')
    return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """在测试集上评估模型性能"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = 100 * correct / total
    return test_loss, test_acc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集（训练集和测试集）
    data = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)
    train_loader = data['train_loader']
    test_loader = data['test_loader']
    class_names = data['class_names']
    print(f"Number of classes: {len(class_names)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")

    # 加载模型
    model = get_model(model_type=args.model_type, num_classes=20, pretrained=(args.model_type == 'vgg16'), finetune=args.finetune)
    model = model.to(device)

    # 加载预训练权重（如果提供）
    if args.weights_path and os.path.exists(args.weights_path):
        print(f"Loading pre-trained weights from {args.weights_path}")
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
    else:
        print("No pre-trained weights found, training from scratch")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环（每轮只在训练集上训练，不做验证）
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # 保存最终模型
    output_path = f"{args.model_type}_final.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Final model saved to {output_path}")

    # 在测试集上进行一次性评估
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Evaluation on Test Set:")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
