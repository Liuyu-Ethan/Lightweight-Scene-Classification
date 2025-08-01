import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu6(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu6(x)
        return x


class CustomInvertedResidualBlock(nn.Module):
    """
    倒残差块，参考MobileNetV2：1x1升维 -> 3x3深度卷积 -> 1x1降维
    """
    def __init__(self, in_channels, out_channels, stride, expansion_factor=4):
        super(CustomInvertedResidualBlock, self).__init__()
        # 步幅，用于控制降采样
        self.stride = stride
        # 是否使用残差连接：仅当步幅为1且输入输出通道相同时
        self.use_residual = (stride == 1 and in_channels == out_channels)
        # 扩展后的隐藏层通道数
        hidden_dim = in_channels * expansion_factor

        # 1x1升维
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ) if expansion_factor > 1 else None
        # 3x3深度卷积
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim if expansion_factor > 1 else in_channels,
                      hidden_dim if expansion_factor > 1 else in_channels,
                      kernel_size=3, stride=stride, padding=1,
                      groups=hidden_dim if expansion_factor > 1 else in_channels, bias=False),
            nn.BatchNorm2d(hidden_dim if expansion_factor > 1 else in_channels),
            nn.ReLU6(inplace=True)
        )
        # 1x1降维
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim if expansion_factor > 1 else in_channels,
                      out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.expand(x) if self.expand is not None else x
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_residual:
            out = out + identity
        return out


class CustomCNN(nn.Module):
    """
    优化后的自定义CNN模型，参考MobileNetV2：
    - 轻量设计，适配RSI-CB128数据集
    """
    def __init__(self, num_classes=20):
        super(CustomCNN, self).__init__()
        # 初始卷积层：提取低级特征，步幅2降采样
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        # 倒残差块序列，精简结构
        self.features2 = nn.Sequential(
            CustomInvertedResidualBlock(in_channels=16, out_channels=12, stride=1, expansion_factor=1),
            CustomInvertedResidualBlock(in_channels=12, out_channels=16, stride=2, expansion_factor=4),
            CustomInvertedResidualBlock(in_channels=16, out_channels=24, stride=2, expansion_factor=4),
            CustomInvertedResidualBlock(in_channels=24, out_channels=24, stride=1, expansion_factor=4),
            CustomInvertedResidualBlock(in_channels=24, out_channels=32, stride=2, expansion_factor=4),
            CustomInvertedResidualBlock(in_channels=32, out_channels=48, stride=2, expansion_factor=4),
        )

        # 全局平均池化：将特征图压缩为1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出: 48 x 1 x 1

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # Dropout防止过拟合，降低p值减少计算
            nn.Linear(48, num_classes)  # 输出: 45类
        )

    def forward(self, x):
        x = self.features(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(model_type='custom_cnn', num_classes=None, pretrained=False, finetune=False):
    """
    获取指定类型的模型
    Args:
        model_type (str): 模型类型，'custom_cnn' 或 'vgg16'
        num_classes (int): 分类数量，RSI-CB128为45
        pretrained (bool): 是否加载预训练权重
        finetune (bool): 是否微调，仅对预训练模型有效
    Returns:
        nn.Module: 模型实例
    """
    if model_type == 'custom_cnn':
        # 返回优化后的自定义CNN模型
        model = CustomCNN(num_classes=num_classes)
    elif model_type == 'vgg16':
        # 加载VGG16模型
        model = vgg16(pretrained=pretrained)
        if finetune:
            # 允许微调所有层
            for param in model.parameters():
                param.requires_grad = True
        else:
            # 冻结特征提取层，仅微调分类器
            for param in model.parameters():
                param.requires_grad = False
        # 修改分类器输出层以匹配类别数
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        if finetune:
            # 仅分类器参数可训练
            for param in model.classifier[6].parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    return model


def load_model_weights(model, weights_path, device, model_type):
    """
    加载模型权重，适配可能的前缀差异
    Args:
        model (nn.Module): 待加载权重的模型
        weights_path (str): 权重文件路径
        device (torch.device): 设备 (CPU 或 GPU)
        model_type (str): 模型类型，'custom_cnn' 或 'vgg16'
    Returns:
        nn.Module: 加载权重后的模型
    """
    state_dict = torch.load(weights_path, map_location=device)
    # 检查权重键名前缀
    state_dict_keys = list(state_dict.keys())
    expected_prefix = 'features' if model_type == 'vgg16' else 'features'
    if any(key.startswith('model.') for key in state_dict_keys):
        # 如果权重包含 'model.' 前缀，移除它
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    # 测试模型
    model = CustomCNN(num_classes=20)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")

    # 计算参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数量: {count_parameters(model)}")
