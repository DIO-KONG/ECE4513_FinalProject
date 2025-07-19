"""
This module contains the implementation of the Focal Loss function.
Reference:
- The Focal Loss is based on the paper: "Focal Loss for Dense Object Detection" by Lin et al. (https://arxiv.org/abs/1708.02002).
"""

import torch
import torch.nn.functional as F
from torch.nn import Module

class FocalLoss(Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class etLoss(Module):
    def __init__(self):
        super().__init__()
        # 预定义梯度检测核
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('sobel_y', torch.tensor([[-1,-2,-1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3))
        
    def gradient_loss(self, pred, target):
        # 确保 sobel_x 和 sobel_y 在与输入相同的设备上
        self.sobel_x = self.sobel_x.to(pred.device)
        self.sobel_y = self.sobel_y.to(pred.device)

        # 计算梯度
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        
        # 梯度差异损失
        return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
    
    def forward(self, pred, target):
        # 像素级损失
        pixel_loss = F.mse_loss(pred, target)
        
        # 结构相似性损失
        ssim_loss = 1 - self.ssim(pred, target)
        
        # 梯度一致性损失
        grad_loss = self.gradient_loss(pred, target)
        
        return 0.4*pixel_loss + 0.4*ssim_loss + 0.2*grad_loss
    
    def ssim(self, x, y, window_size=7):
        # 简化版SSIM计算
        C1 = 0.01**2
        C2 = 0.03**2
        
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size//2)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size//2)
        
        sigma_x = F.avg_pool2d(x**2, window_size, stride=1, padding=window_size//2) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, window_size, stride=1, padding=window_size//2) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, window_size, stride=1, padding=window_size//2) - mu_x*mu_y
        
        ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / \
                   ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))
        
        return ssim_map.mean()