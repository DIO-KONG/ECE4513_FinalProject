import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class TinyFreqNet(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        
        # 输入图像特征提取器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # [B,32,112,96]
            nn.ReLU(),
            nn.MaxPool2d(2),                 # [B,32,56,48]
            
            nn.Conv2d(32, 64, 3, padding=1), # [B,64,56,48]
            nn.ReLU(),
            nn.MaxPool2d(2),                 # [B,64,28,24]
            
            nn.Conv2d(64, 128, 3, padding=1),# [B,128,28,24]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 6))     # [B,128,7,6] 保持一定的空间信息
        )
        
        # 变换学习网络：从图像特征映射到卷积核变换
        self.transform_net = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), # [B,256,7,6]
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1), # [B,512,7,6]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),           # [B,512,1,1]
            nn.Flatten(),                      # [B,512]
        )
        
        # 变换参数预测头
        self.transform_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, kernel_size * kernel_size)  # 预测变换权重
        )
        
        # 可学习的基础卷积核模板库
        self.register_buffer('base_kernels', self._create_base_kernels())
        
        # 用于调制基础卷积核的权重网络
        num_base_kernels = self.base_kernels.shape[0]
        self.kernel_mixer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_base_kernels),
            nn.Softmax(dim=1)  # 归一化权重
        )

    def _create_base_kernels(self):
        """创建多种基础卷积核模板"""
        import math
        kernels = []
        
        # 1. 水平运动模糊核
        h_kernel = torch.zeros(self.kernel_size, self.kernel_size)
        center = self.kernel_size // 2
        h_kernel[center, :] = 1.0 / self.kernel_size
        kernels.append(h_kernel)
        
        # 2. 垂直运动模糊核
        v_kernel = torch.zeros(self.kernel_size, self.kernel_size)
        v_kernel[:, center] = 1.0 / self.kernel_size
        kernels.append(v_kernel)
        
        # 3. 对角线运动模糊核
        d1_kernel = torch.zeros(self.kernel_size, self.kernel_size)
        for i in range(self.kernel_size):
            if i < self.kernel_size:
                d1_kernel[i, i] = 1.0
        d1_kernel = d1_kernel / torch.sum(d1_kernel)
        kernels.append(d1_kernel)
        
        # 4. 反对角线运动模糊核
        d2_kernel = torch.zeros(self.kernel_size, self.kernel_size)
        for i in range(self.kernel_size):
            if self.kernel_size - 1 - i >= 0:
                d2_kernel[i, self.kernel_size - 1 - i] = 1.0
        d2_kernel = d2_kernel / torch.sum(d2_kernel)
        kernels.append(d2_kernel)
        
        # 5. 高斯模糊核
        gaussian_kernel = torch.zeros(self.kernel_size, self.kernel_size)
        sigma = self.kernel_size / 6.0
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                x, y = i - center, j - center
                gaussian_kernel[i, j] = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        kernels.append(gaussian_kernel)
        
        # 6. 圆形运动模糊核
        circular_kernel = torch.zeros(self.kernel_size, self.kernel_size)
        radius = self.kernel_size // 4
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                x, y = i - center, j - center
                if x*x + y*y <= radius*radius:
                    circular_kernel[i, j] = 1.0
        circular_kernel = circular_kernel / torch.sum(circular_kernel)
        kernels.append(circular_kernel)
        
        return torch.stack(kernels, dim=0)  # [num_kernels, kernel_size, kernel_size]

    def forward(self, x):
        # 1. 图像特征提取
        img_features = self.image_encoder(x)  # [B,128,7,6]
        
        # 2. 学习变换参数
        transform_features = self.transform_net(img_features)  # [B,512]
        
        # 3. 预测空间变换权重
        transform_weights = self.transform_head(transform_features)  # [B, kernel_size*kernel_size]
        transform_weights = transform_weights.view(-1, self.kernel_size, self.kernel_size)  # [B, kernel_size, kernel_size]
        
        # 4. 预测基础卷积核的混合权重
        mixing_weights = self.kernel_mixer(transform_features)  # [B, num_base_kernels]
        
        # 5. 通过加权组合基础卷积核
        batch_size = x.shape[0]
        num_base_kernels = self.base_kernels.shape[0]
        
        # 扩展基础卷积核到batch维度
        base_kernels_expanded = self.base_kernels.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, num_base_kernels, kernel_size, kernel_size]
        
        # 加权组合
        mixed_kernel = torch.sum(
            base_kernels_expanded * mixing_weights.unsqueeze(-1).unsqueeze(-1), 
            dim=1
        )  # [B, kernel_size, kernel_size]
        
        # 6. 应用学习到的空间变换
        # 使用逐元素乘法作为变换操作
        final_kernel = mixed_kernel * torch.sigmoid(transform_weights)  # sigmoid确保权重在[0,1]范围
        
        # 7. 归一化卷积核
        kernel_sum = torch.sum(final_kernel.view(batch_size, -1), dim=1, keepdim=True)
        kernel_sum = torch.clamp(kernel_sum, min=1e-8)  # 避免除零
        final_kernel = final_kernel / kernel_sum.view(batch_size, 1, 1)
        
        # 8. 转换到频域返回
        final_kernel_fft = torch.fft.fft2(final_kernel)
        
        return final_kernel_fft  # [B, kernel_size, kernel_size] complex
    

class CompactFreqLoss(nn.Module):
    def __init__(self, magnitude_weight=1.0, phase_weight=0.1, scale_factor=1000.0):
        super().__init__()
        self.eps = 1e-6
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight
        self.scale_factor = scale_factor  # 缩放因子来增加损失数值
        
    def forward(self, pred, target):
        """ 计算幅度谱和相位损失，并进行适当缩放 """
        # 创建低频权重矩阵 (高斯加权)
        h, w = pred.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=pred.device), 
            torch.linspace(-1, 1, w, device=pred.device),
            indexing='ij'
        )
        weight = torch.exp(-(x**2 + y**2)/0.2)  # 中心区域权重高
        
        # 计算幅度和相位
        pred_mag = pred.abs() + self.eps
        target_mag = target.abs() + self.eps
        pred_phase = pred.angle()
        target_phase = target.angle()
        
        # 幅度谱加权MSE损失
        mag_loss = F.mse_loss(
            pred_mag * weight,
            target_mag * weight
        )
        
        # 相位损失 (角度差异)
        phase_diff = torch.angle(torch.exp(1j * (pred_phase - target_phase)))
        phase_loss = F.mse_loss(
            phase_diff * weight,
            torch.zeros_like(phase_diff)
        )
        
        # 总损失，增加缩放因子
        total_loss = self.scale_factor * (
            self.magnitude_weight * mag_loss + 
            self.phase_weight * phase_loss
        )
        
        return total_loss