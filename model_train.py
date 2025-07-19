"""
This script contains training-related code for the CK+ dataset.
Some parts of the code are adapted or inspired by publicly available resources:
- The dataset handling and model training structure are based on common PyTorch practices.
- The FocalLoss implementation is inspired by the original paper: "Focal Loss for Dense Object Detection" by Lin et al. (https://arxiv.org/abs/1708.02002).
- The Deformable Convolutional Network is inspired by the paper: "Deformable Convolutional Networks" by Dai et al. (https://arxiv.org/abs/1703.06211).
"""

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from models import build_model
from loss_func import FocalLoss
from datasets import CKPlusDataset

# 设置随机种子以确保可重复性
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

dataset_path = "data/CK+2"
full_dataset = CKPlusDataset(dataset_path, transform=transform)
train_dataset, val_dataset = full_dataset.split_dataset()

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = full_dataset.get_num_classes()
print("Number of classes:", num_classes)

if __name__ == "__main__":

    model = build_model(num_classes)
    optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
    criterion = FocalLoss(alpha=0.5, gamma=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader)}")

    torch.save(model.state_dict(), "models/model_deform_conv(dropout2d).pth")