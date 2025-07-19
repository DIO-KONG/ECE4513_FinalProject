"""
This module contains model-related code, including the Deformable Convolutional Network and model architecture.
References:
- The Deformable Convolutional Network is inspired by the paper: "Deformable Convolutional Networks" by Dai et al. (https://arxiv.org/abs/1703.06211).
- The model architecture is based on common practices in convolutional neural networks for image classification.
"""

from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, ReLU, Dropout, Linear, Module, InstanceNorm2d, Tanh, LeakyReLU, ConvTranspose2d, Dropout2d
from torchvision.ops import deform_conv
import torch
import torch.nn.functional as F

class DeformConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)  # 原卷积

        # offset channels: 2 * kernel_size * kernel_size
        offset_channels = 2 * kernel_size * kernel_size
        self.conv_offset = Conv2d(in_channels, offset_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        init_offset = torch.zeros(offset_channels, in_channels, kernel_size, kernel_size)
        self.conv_offset.weight = torch.nn.Parameter(init_offset)  # 初始化为0

        # mask channels: kernel_size * kernel_size
        mask_channels = kernel_size * kernel_size
        self.conv_mask = Conv2d(in_channels, mask_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        init_mask = torch.zeros(mask_channels, in_channels, kernel_size, kernel_size) + 0.5
        self.conv_mask.weight = torch.nn.Parameter(init_mask)  # 初始化为0.5
 
    def forward(self, x, offset):
        mask = torch.sigmoid(self.conv_mask(x)) #保证在0到1之间
        out = deform_conv.deform_conv2d(input=x, offset=offset, 
                                            weight=self.conv.weight, 
                                             mask=mask, padding=(1, 1))
        return out
        
class DeformableBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 偏移量生成网络
        self.offset_conv = Sequential(
            Conv2d(in_channels, in_channels//2, 3, padding=1),
            LeakyReLU(0.2),
            Conv2d(in_channels//2, 2*3*3, 3, padding=1),
            Tanh()  # 限制偏移范围
        )
        # 可变形卷积
        self.deform_conv = DeformConv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = InstanceNorm2d(out_channels)
    
    def forward(self, x):
        offset = self.offset_conv(x) * 3  # 限制偏移幅度
        return F.leaky_relu(self.norm(self.deform_conv(x, offset)), 0.2)

def build_model(nb_classes):
    class ModelWithOffset(Module):
        def __init__(self):
            super().__init__()
            self.conv1 = DeformConv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
            self.offset1 = Conv2d(in_channels=1, out_channels=18, kernel_size=3, padding=1)  # Offset generator
            self.relu1 = ReLU()

            self.conv2 = DeformConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.offset2 = Conv2d(in_channels=32, out_channels=18, kernel_size=3, padding=1)  # Offset generator
            self.relu2 = ReLU()

            self.pool1 = MaxPool2d(kernel_size=2, stride=2)
            self.dropout1 = Dropout2d(0.25)

            self.conv3 = DeformConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            self.offset3 = Conv2d(in_channels=64, out_channels=18, kernel_size=3, padding=1)  # Offset generator
            self.relu3 = ReLU()

            self.conv4 = DeformConv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.offset4 = Conv2d(in_channels=128, out_channels=18, kernel_size=3, padding=1)  # Offset generator
            self.relu4 = ReLU()

            self.pool2 = MaxPool2d(kernel_size=2, stride=2)
            self.dropout2 = Dropout2d(0.5)

            self.flatten = Flatten()

            self.fc1 = Linear(in_features=256 * 32 * 32, out_features=512)
            self.relu5 = ReLU()
            self.dropout3 = Dropout(0.5)

            self.fc2 = Linear(in_features=512, out_features=nb_classes)

        def forward(self, x):
            offset1 = self.offset1(x)
            x = self.conv1(x, offset1)
            x = self.relu1(x)

            offset2 = self.offset2(x)
            x = self.conv2(x, offset2)
            x = self.relu2(x)

            x = self.pool1(x)
            x = self.dropout1(x)

            offset3 = self.offset3(x)
            x = self.conv3(x, offset3)
            x = self.relu3(x)

            offset4 = self.offset4(x)
            x = self.conv4(x, offset4)
            x = self.relu4(x)

            x = self.pool2(x)
            x = self.dropout2(x)

            x = self.flatten(x)

            x = self.fc1(x)
            x = self.relu5(x)
            x = self.dropout3(x)

            x = self.fc2(x)
            return x

    return ModelWithOffset()

def build_model_old(nb_classes):
    model = Sequential(
        Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        ReLU(),

        Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        ReLU(),

        MaxPool2d(kernel_size=2, stride=2),
        Dropout2d(0.25),

        Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        ReLU(),

        Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        ReLU(),

        MaxPool2d(kernel_size=2, stride=2),
        Dropout2d(0.5),

        Flatten(),

        Linear(in_features=256 * 32 * 32, out_features=512),
        ReLU(),
        Dropout(0.5),

        Linear(in_features=512, out_features=nb_classes)
    )
    return model

class MiniExpressionTransformNet(Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # 编码器
        self.enc1 = Sequential(
            Conv2d(in_channels, 32, 3, padding=1),
            InstanceNorm2d(32),
            LeakyReLU(0.2),
            MaxPool2d(2)
        )

        self.enc_dropout = Dropout2d(0.25)

        self.enc2 = Sequential(
            Conv2d(32, 64, 3, padding=1),
            InstanceNorm2d(64),
            LeakyReLU(0.2),
            MaxPool2d(2)
        )
        
        # 可变形处理
        # self.deform_block = DeformConv2d(64, 64)
        self.deform_block = DeformableBlock(64, 64)
        
        # 解码器
        self.dec1 = Sequential(
            ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            InstanceNorm2d(32),
            LeakyReLU(0.2)
        )
        self.dec2 = Sequential(
            ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            InstanceNorm2d(16),
            LeakyReLU(0.2)
        )
        
        # 输出层
        self.final = Sequential(
            Conv2d(16, 1, 3, padding=1),
            Tanh())

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)  # 24x24
        e1 = self.enc_dropout(e1)
        e2 = self.enc2(e1) # 12x12
        
        # 可变形变换
        d = self.deform_block(e2)
        
        # 解码
        x = self.dec1(d) + e1  # 跳跃连接
        x = self.dec2(x)
        
        return self.final(x)

class CombinedModel(Module):
    def __init__(self, preprocessing_model, classification_model):
        super().__init__()
        self.preprocessing_model = preprocessing_model
        self.classification_model = classification_model

        # Freeze preprocessing model parameters
        for param in self.preprocessing_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.preprocessing_model(x)  # Apply preprocessing
        x = self.classification_model(x)  # Apply classification
        return x
