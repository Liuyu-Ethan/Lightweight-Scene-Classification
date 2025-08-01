import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import numpy as np



# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 数据集类
class RSIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        RSI-CB128数据集加载器，支持嵌套子文件夹，识别20个小类
        Args:
            root_dir (string): 数据集根目录
            transform (callable, optional): 可选的图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"数据集目录 {root_dir} 不存在")

        # 递归获取所有小类文件夹
        self.classes = []
        self.class_to_idx = {}
        self.class_paths = {}  # 存储类名到实际路径的映射
        class_idx = 0
        for root, dirs, _ in sorted(os.walk(root_dir)):
            # 过滤隐藏文件夹（如 .ipynb_checkpoints）
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for d in sorted(dirs):
                class_path = os.path.join(root, d)
                # 仅将包含图像的文件夹视为小类
                if any(f.lower().endswith(('.jpg', '.png', '.tif', '.jpeg')) for f in os.listdir(class_path)):
                    class_name = os.path.relpath(class_path, root_dir).replace(os.sep, '_')
                    self.classes.append(class_name)
                    self.class_to_idx[class_name] = class_idx
                    self.class_paths[class_name] = class_path  # 记录实际路径
                    class_idx += 1

        # if len(self.classes) != 45:
        #     raise ValueError(f"预期45个小类，实际找到 {len(self.classes)} 个：{self.classes}")
        print(f"检测到的类别数量: {len(self.classes)}")
        print(f"类别: {self.classes}")

        print(f"找到的类别数: {len(self.classes)}")
        print(f"类别: {self.classes}")

        self.samples = []
        for class_name in self.classes:
            class_dir = self.class_paths[class_name]  # 使用实际路径
            img_count = 0
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.tif', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                    img_count += 1
            print(f"类别 {class_name}: 找到 {img_count} 张图像")

        print(f"总样本数: {len(self.samples)}")
        if len(self.samples) == 0:
            raise ValueError("未找到任何有效的图像文件")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法打开图像 {img_path}: {e}")
            raise
        if self.transform:
            image = self.transform(image)
        return image, label
# 获取数据变换
def get_transforms():
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transforms = [
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(p=0.375),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

        return {
        'train': transforms.Compose(train_transforms),
        'test': test_transform
    }

# 获取数据加载器
def get_dataloaders(data_dir='RSI-CB128', batch_size=32):
    set_seed(42)
    transforms_dict = get_transforms()

    full_dataset = RSIDataset(
        root_dir=data_dir,
        transform=None
    )

    class_names = full_dataset.classes

    class_indices = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_indices = []
    test_indices = []
    for label, indices in class_indices.items():
        split = int(np.floor(0.8 * len(indices)))
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])

    print(f"训练集样本数: {len(train_indices)}")
    print(f"测试集样本数: {len(test_indices)}")

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    train_dataset.dataset.transform = transforms_dict['train']
    test_dataset.dataset.transform = transforms_dict['test']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'class_names': class_names
    }

if __name__ == "__main__":
#需要手动更改数据集路径，这里我使用了绝对路径
    # data_dir = r"E:\Sofw_E\Anaconda\envs\pytorch_env\pythonProject1\designed_CNN\RSI-CB128"
    data_dir = r"RSI-CB128"
    print(f"正在加载数据集目录: {data_dir}")
    data = get_dataloaders(data_dir=data_dir, batch_size=4)
    train_loader = data['train_loader']
    class_names = data['class_names']

    print(f"类别数量: {len(class_names)}")
    print(f"类别名称: {class_names}")

    images, labels = next(iter(train_loader))
    print(f"图像张量形状: {images.shape}")
    print(f"标签: {labels}")