import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

# 导入自定义模块
from RotationModel import RotationModel
from GetRotatedDataset import RotatedDataset

def set_seed(seed=42):
    """设置随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 记录训练过程
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch_idx, (images, angles) in enumerate(self.train_loader):
            images = images.to(self.device)
            angles = angles.to(self.device).float().unsqueeze(1)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, angles)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 计算指标
            total_loss += loss.item()
            mae = torch.mean(torch.abs(outputs - angles)).item()
            total_mae += mae
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'  Batch {batch_idx}, Loss: {loss.item():.4f}, MAE: {mae:.4f}')
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        return avg_loss, avg_mae
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, angles in self.val_loader:
                images = images.to(self.device)
                angles = angles.to(self.device).float().unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, angles)
                
                total_loss += loss.item()
                mae = torch.mean(torch.abs(outputs - angles)).item()
                total_mae += mae
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        return avg_loss, avg_mae
    
    def train(self, num_epochs=100, save_path="rotation_model.pth"):
        """训练模型"""
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        print(f"开始训练，共 {num_epochs} 个epochs")
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            # 训练
            train_loss, train_mae = self.train_epoch()
            
            # 验证
            val_loss, val_mae = self.validate()
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录指标
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_maes.append(train_mae)
            self.val_maes.append(val_mae)
            
            print(f'Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, save_path)
                print(f'保存最佳模型到 {save_path}')
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f'早停: 验证损失在 {max_patience} 个epochs内没有改善')
                break
        
        print(f'\n训练完成! 最佳验证损失: {best_val_loss:.4f}')
        return best_val_loss
    
    def plot_training_history(self, save_path="training_history.png"):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE曲线
        ax2.plot(self.train_maes, label='Training MAE')
        ax2.plot(self.val_maes, label='Validation MAE')
        ax2.set_title('Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (degrees)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"训练历史图保存到: {save_path}")

def create_data_loaders(data_root, batch_size=32, num_workers=4):
    """创建数据加载器"""
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((112, 96)),  # 调整到模型期望的尺寸
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]
    ])
    
    # 创建数据集
    dataset = RotatedDataset(data_root, max_images_per_folder=3, transform=transform)
    
    # 计算训练集和验证集大小 (8:2)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    print(f"数据集总大小: {total_size}")
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")
    
    # 使用固定种子分割数据集
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset

def main():
    """主函数"""
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据集路径
    data_root = "/home/wenhao/CUHK/ECE4513/FinalProject/Data/CASIA_Rotated"
    
    # 超参数
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, dataset = create_data_loaders(
        data_root, batch_size=batch_size, num_workers=4
    )
    
    # 打印数据集统计信息
    stats = dataset.get_statistics()
    print("\n数据集统计信息:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 创建模型
    print("\n创建模型...")
    model = RotationModel(image_zise=(112, 96))
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建训练器
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate
    )
    
    # 开始训练
    print("\n开始训练...")
    best_val_loss = trainer.train(
        num_epochs=num_epochs,
        save_path="best_rotation_model.pth"
    )
    
    # 绘制训练历史
    trainer.plot_training_history("rotation_training_history.png")
    
    print(f"\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
