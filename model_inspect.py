import torch
import matplotlib.pyplot as plt
from model_train import build_model, num_classes, full_dataset

def visualize_first_layer_output(model, image, device):
    model.to(device)
    model.eval()  # 设置模型为评估模式

    # 将图像移动到设备
    image = image.to(device).unsqueeze(0)  # 添加 batch 维度

    # 提取第一层
    first_layer = list(model.children())[0]
    with torch.no_grad():
        output = first_layer(image)

    # 可视化输出
    output = output.squeeze(0).cpu()  # 移除 batch 维度并移动到 CPU
    num_filters = output.size(0)  # 获取特征图数量

    plt.figure(figsize=(12, 6))
    for i in range(num_filters):
        plt.subplot(4, 8, i + 1)  # 4 行 8 列，共 32 个特征图
        plt.imshow(output[i].numpy(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 加载模型
model = build_model(num_classes)
model.load_state_dict(torch.load("models/model_linear.pth"))

# 加载图像
image, _ = full_dataset[0]  # 假设从数据集中获取图像

# 可视化第一层输出
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
visualize_first_layer_output(model, image, device)