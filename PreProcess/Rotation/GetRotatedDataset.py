import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class RotatedDataset(Dataset):
    """
    数据集类用于读取CASIA_Rotated数据集
    每个子文件夹只读取前3张图片
    图片名称格式：XXX_角度.jpg，其中角度作为标签
    """
    
    def __init__(self, data_root, max_images_per_folder=3, transform=None):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录路径 (Data/CASIA_Rotated)
            max_images_per_folder: 每个子文件夹最多读取的图片数量
            transform: 图像变换函数
        """
        self.data_root = data_root
        self.max_images_per_folder = max_images_per_folder
        self.transform = transform
        
        # 存储图片路径和对应的角度标签
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """加载数据集，从每个子文件夹读取前3张图片"""
        
        # 获取所有子文件夹
        subfolders = [f for f in os.listdir(self.data_root) 
                     if os.path.isdir(os.path.join(self.data_root, f))]
        
        print(f"找到 {len(subfolders)} 个子文件夹")
        
        for subfolder in sorted(subfolders):
            subfolder_path = os.path.join(self.data_root, subfolder)
            
            # 获取该子文件夹下的所有jpg图片
            image_files = glob.glob(os.path.join(subfolder_path, "*.jpg"))
            image_files.sort()  # 按文件名排序
            
            # 只取前max_images_per_folder张图片
            selected_images = image_files[:self.max_images_per_folder]
            
            for image_path in selected_images:
                # 从文件名中提取角度标签
                filename = os.path.basename(image_path)
                # 格式：XXX_角度.jpg
                try:
                    # 去掉.jpg后缀，然后按'_'分割，取最后一部分作为角度
                    angle_str = filename.replace('.jpg', '').split('_')[-1]
                    angle = float(angle_str)
                    
                    self.image_paths.append(image_path)
                    self.labels.append(angle)
                    
                except (ValueError, IndexError) as e:
                    print(f"无法解析文件名 {filename} 中的角度: {e}")
                    continue
        
        print(f"总共加载了 {len(self.image_paths)} 张图片")
        print(f"角度范围: {min(self.labels):.3f} 到 {max(self.labels):.3f}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            image: PIL图像或经过transform的tensor
            label: 角度标签 (float)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")
            # 返回一个空白图像
            image = Image.new('RGB', (224, 224), color='white')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_statistics(self):
        """获取数据集统计信息"""
        if not self.labels:
            return {}
        
        labels_array = np.array(self.labels)
        
        stats = {
            'total_images': len(self.labels),
            'angle_min': float(labels_array.min()),
            'angle_max': float(labels_array.max()),
            'angle_mean': float(labels_array.mean()),
            'angle_std': float(labels_array.std())
        }
        
        return stats

def main():
    """测试函数"""
    # 数据集路径
    data_root = "/home/wenhao/CUHK/ECE4513/FinalProject/Data/CASIA_Rotated"
    
    # 创建数据集实例
    dataset = RotatedDataset(data_root, max_images_per_folder=3)
    
    # 打印统计信息
    stats = dataset.get_statistics()
    print("\n数据集统计信息:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 显示前几个样本的信息
    print("\n前10个样本:")
    for i in range(min(10, len(dataset))):
        image, label = dataset[i]
        image_path = dataset.image_paths[i]
        filename = os.path.basename(image_path)
        print(f"样本 {i}: {filename} -> 角度: {label:.3f}°")

if __name__ == "__main__":
    main()