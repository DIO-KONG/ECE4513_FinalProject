from datasets import CKPlusDataset_et, JAFFEDataset_et
from torch.utils.data import Dataset, DataLoader
from models import MiniExpressionTransformNet
from torchvision import transforms
import torch
from torch.optim import Adam
from loss_func import etLoss

# 超参数设置
batch_size = 32
epochs = 50
learning_rate = 0.001

# # 数据集加载
# train_set = CKPlusDataset_et(root_dir='data/', phase='train')
# val_set = CKPlusDataset_et(root_dir='data/', phase='val')
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

train_set = JAFFEDataset_et(root_dir='data/jaffe_crop', phase='train')
val_set = JAFFEDataset_et(root_dir='data/jaffe_crop', phase='val')
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# 模型、损失函数和优化器
model = MiniExpressionTransformNet()  # 根据需要调整参数
# criterion = MSELoss()
criterion = etLoss()  # 使用自定义损失函数
optimizer = Adam(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 训练过程
def train():
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()
            output = model(source)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 验证过程
def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)

            output = model(source)
            loss = criterion(output, target)
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss/len(val_loader):.4f}")

if __name__ == "__main__":
    train()
    validate()
    # 保存模型
    torch.save(model.state_dict(), 'models/model_etjaffe.pth')
    print("Model saved to models/model_et_jaffe.pth")

