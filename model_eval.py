import torch
from torch.utils.data import DataLoader
from models import build_model_old
from datasets import JAFFEDataset
from torchvision import transforms
from model_old_jaffe_train import val_dataloader, num_classes



def evaluate_model(model, dataloader, device):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)  # 获取预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

# 构建模型框架并加载参数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model_old(num_classes)
model.load_state_dict(torch.load("models/model_jaffe.pth"))
model.to(device)  # 将模型移动到 GPU
model.eval()  # 确保模型处于评估模式
evaluate_model(model, val_dataloader, device)