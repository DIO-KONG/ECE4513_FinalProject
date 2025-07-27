import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class FirstImageDataset(Dataset):
    """
    数据集类用于读取CASIA数据集中每个人物的第一张照片
    每个子文件夹只读取第一张图片（按文件名排序）
    """
    
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        
        Args:
            root_dir: 数据根目录路径 (例如: Data/CASIA)
            transform: 图像变换函数
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 存储图片路径和对应的人物ID
        self.image_paths = []
        self.person_ids = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """加载数据集，从每个子文件夹读取第一张图片"""
        
        # 检查数据根目录是否存在
        if not os.path.exists(self.root_dir):
            raise ValueError(f"数据根目录不存在: {self.root_dir}")
        
        # 获取所有子文件夹
        subfolders = [f for f in os.listdir(self.root_dir) 
                     if os.path.isdir(os.path.join(self.root_dir, f))]
        
        print(f"找到 {len(subfolders)} 个人物文件夹")
        
        for subfolder in sorted(subfolders):
            subfolder_path = os.path.join(self.root_dir, subfolder)
            
            # 获取该子文件夹下的所有jpg图片
            image_files = glob.glob(os.path.join(subfolder_path, "*.jpg"))
            
            if not image_files:
                # 如果没有jpg文件，尝试其他格式
                image_files = glob.glob(os.path.join(subfolder_path, "*.png"))
                if not image_files:
                    image_files = glob.glob(os.path.join(subfolder_path, "*.jpeg"))
            
            if image_files:
                # 按文件名排序，选择第一张图片
                image_files.sort()
                first_image = image_files[0]
                
                self.image_paths.append(first_image)
                self.person_ids.append(subfolder)
            else:
                print(f"警告: 文件夹 {subfolder} 中没有找到图片文件")
        
        print(f"总共加载了 {len(self.image_paths)} 张图片")
        print(f"人物ID范围: {min(self.person_ids) if self.person_ids else 'None'} 到 {max(self.person_ids) if self.person_ids else 'None'}")
    
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
            person_id: 人物ID (string)
        """
        image_path = self.image_paths[idx]
        person_id = self.person_ids[idx]
        
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
        
        return image, person_id
    
    def get_statistics(self):
        """获取数据集统计信息"""
        if not self.person_ids:
            return {}
        
        stats = {
            'total_images': len(self.person_ids),
            'total_persons': len(set(self.person_ids)),
            'person_ids_sample': sorted(self.person_ids)[:10] if len(self.person_ids) >= 10 else sorted(self.person_ids)
        }
        
        return stats
    
    def get_person_list(self):
        """获取所有人物ID列表"""
        return sorted(list(set(self.person_ids)))

def main():
    """测试函数"""
    # 数据集路径 - 允许传递root_dir参数
    import sys
    
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "/home/wenhao/CUHK/ECE4513/FinalProject/Data/CASIA"
    
    print(f"使用数据根目录: {root_dir}")
    
    # 创建数据集实例
    try:
        dataset = FirstImageDataset(root_dir)
        
        # 打印统计信息
        stats = dataset.get_statistics()
        print("\n数据集统计信息:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # 显示前几个样本的信息
        print("\n前10个样本:")
        for i in range(min(10, len(dataset))):
            image, person_id = dataset[i]
            image_path = dataset.image_paths[i]
            filename = os.path.basename(image_path)
            print(f"样本 {i}: 人物ID {person_id} -> 文件: {filename}")
        
        # 获取人物列表
        person_list = dataset.get_person_list()
        print(f"\n总共有 {len(person_list)} 个不同的人物")
        if len(person_list) <= 20:
            print(f"人物ID列表: {person_list}")
        else:
            print(f"人物ID示例: {person_list[:10]} ... {person_list[-5:]}")
            
    except Exception as e:
        print(f"创建数据集时发生错误: {e}")

if __name__ == "__main__":
    main()