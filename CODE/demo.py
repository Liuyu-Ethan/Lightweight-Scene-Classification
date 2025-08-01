import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from data import get_transforms, get_dataloaders
from model import get_model
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
import logging
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 抑制 log4cplus 警告
os.environ["LOG4CPLUS_LOGLEVEL"] = "0"
logging.getLogger("AdSyncNamespace").setLevel(logging.CRITICAL)

# 解决 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

matplotlib.use('TkAgg')  # 使用 TkAgg 后端以嵌入 Matplotlib 到 tkinter


class PredictionApp:
    """
    遥感图像分类演示程序的 GUI 界面，支持加载训练好的模型并预测 RSI-CB128 数据集中的任意图像
    包含模型选择、权重加载、图像选择/随机抽取和结果可视化
    """

    def __init__(self, root):
        self.root = root
        self.root.title("遥感图像分类演示")
        self.root.geometry("900x600")  # 设置窗口大小，适应 sidebar 和可视化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.transforms_dict = get_transforms()
        self.test_transform = self.transforms_dict['test']
        data_dir = r"E:\Sofw_E\Anaconda\envs\pytorch_env\pythonProject1\designed_CNN\RSI-CB128"
        self.data = get_dataloaders(data_dir=data_dir, batch_size=4)
        self.class_names = self.data['class_names']
        self.test_dataset = self.data['test_loader'].dataset

        self.current_image_path = None
        self.current_model = None  # 存储模型实例
        self.current_model_type = None  # 存储当前模型类型
        self.current_weights_path = None  # 存储当前权重路径

        self.setup_ui()

    def setup_ui(self):

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(self.main_frame, width=300, relief="sunken", borderwidth=2)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.content_frame = tk.Frame(self.main_frame)
        self.content_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        tk.Label(self.sidebar, text="选择模型类型:", font=("Arial", 12)).pack(pady=5, anchor="w")
        self.model_var = tk.StringVar(value="custom_cnn")
        model_menu = ttk.Combobox(self.sidebar, textvariable=self.model_var,
                                  values=["custom_cnn", "vgg16"], state="readonly", width=25)
        model_menu.pack(pady=5, anchor="w")
        model_menu.bind('<<ComboboxSelected>>', self.update_model)  # 绑定模型类型变化

        # 权重文件选择
        tk.Label(self.sidebar, text="模型权重文件:", font=("Arial", 12)).pack(pady=5, anchor="w")
        self.weights_entry = tk.Entry(self.sidebar, width=30)
        self.weights_entry.pack(pady=5, anchor="w")
        tk.Button(self.sidebar, text="浏览", command=self.browse_weights).pack(pady=5, anchor="w")
        self.weights_entry.bind('<FocusOut>', self.update_model)  # 绑定权重路径变化

        # 图像选择
        tk.Label(self.sidebar, text="选择图像或随机抽取:", font=("Arial", 12)).pack(pady=5, anchor="w")
        tk.Button(self.sidebar, text="选择图像", command=self.select_image).pack(pady=5, anchor="w")
        tk.Button(self.sidebar, text="随机选择图像", command=self.random_select_image).pack(pady=5, anchor="w")

        # 预测按钮
        tk.Button(self.sidebar, text="进行预测", command=self.predict).pack(pady=10, anchor="w")

        # 结果文本显示
        tk.Label(self.sidebar, text="预测结果:", font=("Arial", 12)).pack(pady=5, anchor="w")
        self.result_text = tk.Text(self.sidebar, height=10, width=35, font=("Arial", 10))
        self.result_text.pack(pady=10, anchor="w")

        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.content_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def browse_weights(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch 权重文件", "*.pth")]
        )
        if file_path:
            self.weights_entry.delete(0, tk.END)
            self.weights_entry.insert(0, file_path)
            self.update_model()  # 选择新权重后立即更新模型

    def select_image(self):
        """从文件系统中选择一张图像"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg *.png *.tif *.jpeg")]
        )
        if file_path:
            self.current_image_path = file_path
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"选择的图像: {os.path.basename(file_path)}\n")

    def random_select_image(self):
        """从测试集中随机选择一张图像"""
        try:
            idx = random.randint(0, len(self.test_dataset) - 1)
            self.current_image_path = self.test_dataset.dataset.samples[idx][0]
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"随机选择的图像: {os.path.basename(self.current_image_path)}\n")
        except Exception as e:
            messagebox.showerror("错误", f"随机选择图像失败: {str(e)}")

    def update_model(self, event=None):
        """当模型类型或权重路径变化时，重新加载模型"""
        try:
            model_type = self.model_var.get()
            weights_path = self.weights_entry.get()
            # 检查权重文件和模型类型是否有效且有变化
            if not weights_path or not os.path.exists(weights_path):
                self.current_model = None
                self.current_model_type = None
                self.current_weights_path = None
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "请提供有效的模型权重文件路径\n")
                return
            if model_type == self.current_model_type and weights_path == self.current_weights_path:
                return  # 无变化，不重新加载
            # 加载模型
            model = get_model(model_type=model_type, num_classes=len(self.class_names),
                              pretrained=(model_type == 'vgg16'), finetune=False)
            state_dict = torch.load(weights_path, map_location=self.device)
            new_state_dict = {}
            if model_type == 'vgg16':
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
            else:
                new_state_dict = state_dict
            model.load_state_dict(new_state_dict)
            model = model.to(self.device)
            # 更新存储的模型和状态
            self.current_model = model
            self.current_model_type = model_type
            self.current_weights_path = weights_path
            print(f"已加载模型: {model_type}, 权重: {weights_path}")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")
            self.current_model = None
            self.current_model_type = None
            self.current_weights_path = None

    def load_image(self, image_path):
        """
        加载并预处理单张图像
        Args:
            image_path (str): 图像文件路径
        Returns:
            tuple: 预处理后的图像张量, 原始图像
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件 {image_path} 不存在")
        original_image = Image.open(image_path).convert('RGB')
        image_tensor = self.test_transform(original_image)
        return image_tensor.unsqueeze(0), original_image

    def predict_image(self, model, image_tensor):
        model.eval()
        with torch.no_grad():
            start_time = time.time()  # 记录开始时间
            image_tensor = image_tensor.to(self.device)
            output = model(image_tensor)
            inference_time = time.time() - start_time  # 计算推理时间
            probabilities = F.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probabilities, dim=1)
            pred_class = self.class_names[pred_idx.item()]
            top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)
            top5_results = [(self.class_names[idx.item()], prob.item())
                            for idx, prob in zip(top5_idx[0], top5_prob[0])]
            print(f"单张图像推理时间: {inference_time:.4f} 秒")  # 打印时间
        return pred_idx.item(), pred_class, confidence.item(), top5_results

    def visualize_prediction(self, image_path, pred_class, confidence, top5_results, original_image):
        """
        可视化预测结果，嵌入到 GUI 中
        Args:
            image_path (str): 图像文件路径
            pred_class (str): 预测类别名称
            confidence (float): 预测置信度
            top5_results (list): Top-5 预测结果
            original_image (PIL.Image): 原始图像
        """
        # 清除之前的绘图
        self.ax.clear()

        # 显示原始图像
        self.ax.imshow(original_image)
        self.ax.set_title(f"图像: {os.path.basename(image_path)}")
        self.ax.axis('off')

        # 调整布局并刷新画布
        self.figure.tight_layout()
        self.canvas.draw()

        # 更新文本结果
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"选择的图像: {os.path.basename(self.current_image_path)}\n")
        self.result_text.insert(tk.END, f"预测结果: 类别={pred_class}, 置信度={confidence:.4f}\n")
        self.result_text.insert(tk.END, "Top-5 预测:\n")
        for class_name, prob in top5_results:
            self.result_text.insert(tk.END, f"  {class_name}: {prob:.4f}\n")

    def predict(self):
        """执行预测并显示结果"""
        try:
            # 验证输入
            if not self.current_image_path:
                messagebox.showerror("错误", "请先选择或随机抽取一张图像")
                return
            if self.current_model is None:
                messagebox.showerror("错误", "请提供有效的模型权重文件路径并确保模型加载成功")
                return

            # 加载图像
            image_tensor, original_image = self.load_image(self.current_image_path)
            print(f"输入张量形状: {image_tensor.shape}")

            # 进行预测
            pred_idx, pred_class, confidence, top5_results = self.predict_image(self.current_model, image_tensor)

            # 可视化结果
            self.visualize_prediction(self.current_image_path, pred_class, confidence, top5_results, original_image)

        except Exception as e:
            messagebox.showerror("错误", f"预测过程中发生错误: {str(e)}")

def main():
    """
    主函数：启动 GUI 应用程序
    """
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    # 如果通过命令行运行，提示使用 GUI
    parser = argparse.ArgumentParser(description="遥感图像分类演示程序（GUI 模式）")
    args = parser.parse_args()
    main()