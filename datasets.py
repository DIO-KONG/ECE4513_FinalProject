"""
This module contains the CKPlusDataset class for handling the CK+ dataset.
References:
- The dataset handling structure is inspired by common PyTorch Dataset implementations.
- The dataset splitting logic is adapted from PyTorch's random_split function.
"""

import os
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import numpy as np

class CKPlusDataset(Dataset):
    def __init__(self, root_dir, transform=None, train_ratio=0.8):
        self.samples = []
        self.transform = transform
        self.train_ratio = train_ratio
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    self.samples.append((img_path, label))
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(label for _, label in self.samples)))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label_idx = self.label_to_idx[label]
        return image, label_idx

    def get_num_classes(self):
        return len(self.label_to_idx)

    def split_dataset(self):
        train_size = int(self.train_ratio * len(self))
        val_size = len(self) - train_size
        # generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(self, [train_size, val_size])
        return train_dataset, val_dataset

class CKPlusDataset_et(Dataset):
    def __init__(self, root_dir='data/', phase='train', transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        self.samples = []

        # 收集样本路径
        for emotion_idx, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(root_dir, f"CK+1/{emotion}")
            if not os.path.exists(emotion_dir):
                continue

            # 每个表情目录下的所有图像
            for img_name in os.listdir(emotion_dir):
                if img_name.endswith('.png'):
                    img_path = os.path.join(emotion_dir, img_name)
                    if self._find_happy_pair(img_path) is not None:  # 仅保留有匹配happy样本的样本
                        self.samples.append((img_path, emotion_idx))

        # 划分训练验证集 (8:2)
        split_idx = int(0.8 * len(self.samples))
        if phase == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        # # 数据增强
        # if phase == 'train':
        #     self.transform = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomRotation(10),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5], [0.5])
        #     ])
        # else:
        #     self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5], [0.5])
        #     ])
    
    def __len__(self):
        return len(self.samples)
    
    def _find_happy_pair(self, img_path):
        """找到同一subject的happy表情作为目标"""
        # 解析路径格式: data/CK+1/emotion/subject_sequence.png
        parts = img_path.split(os.sep)
        subject = parts[-1].split('_')[0]

        # 搜索happy目录
        happy_dir = os.path.join(self.root_dir, "CK+1/happy")
        for happy_img in os.listdir(happy_dir):
            if happy_img.startswith(subject):
                return Image.open(os.path.join(happy_dir, happy_img)).convert('L')

        # 如果找不到匹配的happy样本，返回None
        return None

    def __getitem__(self, idx):
        img_path, emotion = self.samples[idx]
        img = Image.open(img_path)  # 转为灰度

        # 找到对应的happy样本
        happy_img = self._find_happy_pair(img_path)
        if happy_img is None:
            return None  # 跳过没有匹配的样本

        return {
            'source': self.transform(img),
            'target': self.transform(happy_img),
            'emotion': emotion
        }

class JAFFEDataset(Dataset):
    def __init__(self, root_dir='data/jaffe_crop', transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.samples = []
        self.posers = set()

        # 收集样本路径和人名
        for img_name in os.listdir(root_dir):
            if img_name.endswith('.tiff'):
                poser = img_name.split('.')[0]  # 人名（如 KA, KM）
                img_path = os.path.join(root_dir, img_name)
                self.samples.append((img_path, poser))
                self.posers.add(poser)

        # 建立人名到索引的映射
        self.poser_to_idx = {poser: idx for idx, poser in enumerate(sorted(self.posers))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, poser = self.samples[idx]
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        label = self.poser_to_idx[poser]
        return img, label

    def get_num_classes(self):
        return len(self.poser_to_idx)

class JAFFEDataset_et(Dataset):
    def __init__(self, root_dir='data/jaffe_crop', phase='train', transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.emotions = ['NE', 'HA', 'SA', 'SU', 'AN', 'DI', 'FE']
        self.samples = []

        # 收集样本路径
        for img_name in os.listdir(root_dir):
            if img_name.endswith('.tiff'):
                parts = img_name.split('.')
                poser = parts[0]  # Poser ID (e.g., KA, KM)
                emotion = parts[1][:-1]  # Emotion label (e.g., NE, HA)
                if emotion in self.emotions:
                    img_path = os.path.join(root_dir, img_name)
                    try:
                        neutral_img = self._find_neutral_pair(poser)
                        if neutral_img is not None:
                            emotion_idx = self.emotions.index(emotion)
                            self.samples.append((img_path, emotion_idx))
                    except FileNotFoundError as e:
                        print(f"Warning: {e}")

        # 划分训练验证集 (8:2)
        split_idx = int(0.8 * len(self.samples))
        if phase == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def _find_neutral_pair(self, poser):
        """找到同一poser的NE表情作为目标"""
        for img_name in os.listdir(self.root_dir):
            if img_name.startswith(poser) and '.NE' in img_name:
                return Image.open(os.path.join(self.root_dir, img_name)).convert('L')

        # 如果找不到匹配的NE样本，抛出异常
        raise FileNotFoundError(f"Neutral (NE) image for poser {poser} not found in {self.root_dir}")

    def __getitem__(self, idx):
        img_path, emotion = self.samples[idx]
        img = Image.open(img_path).convert('L')

        # 找到对应的NE样本
        parts = os.path.basename(img_path).split('.')
        poser = parts[0]  # Poser ID (e.g., KA, KM)
        neutral_img = self._find_neutral_pair(poser)

        return {
            'source': self.transform(img),
            'target': self.transform(neutral_img),
            'emotion': emotion
        }



train_set = CKPlusDataset_et(root_dir='data/', phase='train')
val_set = CKPlusDataset_et(root_dir='data/', phase='val')

# 测试数据集是否正常加载
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random

    # 隐去CK+数据集的测试部分
    # print(f"Number of training samples: {len(train_set)}")
    # print(f"Number of validation samples: {len(val_set)}")

    # # 随机选择5个样本进行可视化
    # for _ in range(5):
    #     idx = random.randint(0, len(train_set) - 1)
    #     sample = train_set[idx]
    #     if sample is None:
    #         continue  # 跳过无效样本

    #     source = sample['source'].squeeze(0).numpy()  # 去掉通道维度
    #     target = sample['target'].squeeze(0).numpy()

    #     plt.figure(figsize=(8, 4))
    #     plt.subplot(1, 2, 1)
    #     plt.title("Source Image")
    #     plt.imshow(source, cmap='gray')
    #     plt.axis('off')

    #     plt.subplot(1, 2, 2)
    #     plt.title("Target (Happy) Image")
    #     plt.imshow(target, cmap='gray')
    #     plt.axis('off')

    #     plt.show()

    # # 测试数据加载
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    # for batch in train_loader:
    #     images = batch['source']
    #     labels = batch['emotion']
    #     print(f"Batch size: {images.size()}, Labels: {labels.size()}")
    #     break  # 只测试一个batch

    # 测试JAFFE数据集
    jaffe_set = JAFFEDataset_et(root_dir='data/jaffe_crop', phase='train')
    print(f"Number of JAFFE training samples: {len(jaffe_set)}")
    for i in range(3):
        sample = jaffe_set[i]
        source = sample['source'].squeeze(0).numpy()
        target = sample['target'].squeeze(0).numpy()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Source Image (JAFFE)")
        plt.imshow(source, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Target (NE) Image (JAFFE)")
        plt.imshow(target, cmap='gray')
        plt.axis('off')

        plt.show()
