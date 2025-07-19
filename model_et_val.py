import torch
from torch.utils.data import DataLoader, random_split
from models import build_model, MiniExpressionTransformNet, CombinedModel
from datasets import CKPlusDataset, JAFFEDataset
from torchvision import transforms

# 设置随机种子以确保可重复性
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# dataset_path = "data/CK+2"
# full_dataset = CKPlusDataset(dataset_path, transform=transform)
# _, val_dataset = full_dataset.split_dataset()

# val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# num_classes = full_dataset.get_num_classes()
dataset_path = "data/jaffe_crop"
full_dataset = JAFFEDataset(dataset_path, transform=transform)
train_dataset, val_dataset = random_split(full_dataset, [len(full_dataset) - len(full_dataset) // 5, len(full_dataset) // 5])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
num_classes = full_dataset.get_num_classes()
print("Number of classes:", num_classes)

# 加载预处理模型
preprocessing_model = MiniExpressionTransformNet()
preprocessing_model.load_state_dict(torch.load('models/model_et.pth'))
preprocessing_model.eval()  # 设置为评估模式
for param in preprocessing_model.parameters():
    param.requires_grad = False  # 冻结参数

# 加载分类模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model = build_model(num_classes)
classification_model.load_state_dict(torch.load("models/model_et_deform_conv.pth"))

model = CombinedModel(preprocessing_model, classification_model)
model.to(device)

# 模型评估
def evaluate_model():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()