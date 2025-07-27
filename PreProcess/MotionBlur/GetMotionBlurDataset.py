import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
from typing import Tuple, Optional, Union, List
from scipy import interpolate
import matplotlib
matplotlib.use('TkAgg')  # 设置matplotlib后端为TkAgg，确保图像能够显示
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter


class MotionBlurKernelEstimator:
    """
    运动模糊卷积核估计器类
    从模糊图像频谱中检测OTF零点位置，应用相位一致性约束修复零点区域，
    通过迭代优化求解精确卷积核
    支持GPU加速计算
    """
    
    def __init__(self, 
                 max_iterations: int = 50,
                 tolerance: float = 1e-6,
                 regularization: float = 0.01,
                 use_gpu: bool = True,
                 visualize_convergence: bool = False):
        """
        初始化卷积核估计器
        
        Args:
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            regularization: 正则化参数
            use_gpu: 是否使用GPU加速
            visualize_convergence: 是否可视化收敛过程
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.visualize_convergence = visualize_convergence
        
        # 存储收敛过程数据
        self.convergence_errors = []
        self.kernel_evolution = []
        
        # GPU设置
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.device = torch.device('cuda')
            print(f"Using GPU acceleration: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU computation")
        
    def preprocess_images(self, 
                         sharp_img: np.ndarray, 
                         blurred_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理图像，转换为灰度并归一化
        
        Args:
            sharp_img: 原始清晰图像 (H, W, 3) 或 (H, W)
            blurred_img: 模糊图像 (H, W, 3) 或 (H, W)
            
        Returns:
            sharp_gray: 灰度清晰图像 (H, W)
            blurred_gray: 灰度模糊图像 (H, W)
        """
        # 转换为灰度图像
        if len(sharp_img.shape) == 3:
            sharp_gray = cv2.cvtColor(sharp_img, cv2.COLOR_RGB2GRAY)
        else:
            sharp_gray = sharp_img.copy()
            
        if len(blurred_img.shape) == 3:
            blurred_gray = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2GRAY)
        else:
            blurred_gray = blurred_img.copy()
        
        # 归一化到 [0, 1]
        sharp_gray = sharp_gray.astype(np.float64) / 255.0
        blurred_gray = blurred_gray.astype(np.float64) / 255.0
        
        return sharp_gray, blurred_gray
    
    def detect_otf_zeros_gpu(self, 
                            sharp_fft: torch.Tensor, 
                            blurred_fft: torch.Tensor,
                            threshold: float = 0.01) -> torch.Tensor:
        """
        GPU版本：检测光学传递函数(OTF)的零点位置
        
        Args:
            sharp_fft: 清晰图像的傅里叶变换 (torch.Tensor)
            blurred_fft: 模糊图像的傅里叶变换 (torch.Tensor)
            threshold: 零点检测阈值
            
        Returns:
            zero_mask: 零点位置掩膜 (torch.Tensor, bool)
        """
        # 计算幅度谱
        sharp_magnitude = torch.abs(sharp_fft)
        blurred_magnitude = torch.abs(blurred_fft)
        
        # 避免除零，添加小的正则化项
        epsilon = 1e-10
        sharp_magnitude = torch.clamp(sharp_magnitude, min=epsilon)
        
        # 计算传递函数的幅度
        transfer_magnitude = blurred_magnitude / sharp_magnitude
        
        # 检测零点：传递函数幅度接近零的位置
        zero_mask = transfer_magnitude < threshold
        
        return zero_mask
    
    def apply_phase_consistency_constraint_gpu(self, 
                                              otf: torch.Tensor, 
                                              zero_mask: torch.Tensor) -> torch.Tensor:
        """
        GPU版本：应用相位一致性约束修复零点区域
        
        Args:
            otf: 光学传递函数 (torch.Tensor)
            zero_mask: 零点位置掩膜 (torch.Tensor)
            
        Returns:
            repaired_otf: 修复后的光学传递函数 (torch.Tensor)
        """
        repaired_otf = otf.clone()
        
        # 对零点区域进行简单的修复（使用邻近值的平均）
        if torch.any(zero_mask):
            # 使用高斯滤波进行局部平均修复
            from scipy.ndimage import gaussian_filter
            
            # 转换为numpy进行处理（避免复杂的GPU张量类型问题）
            otf_np = otf.cpu().numpy()
            zero_mask_np = zero_mask.cpu().numpy()
            
            # 修复实部和虚部
            real_part = np.real(otf_np)
            imag_part = np.imag(otf_np)
            
            # 使用高斯滤波平滑
            real_smoothed = gaussian_filter(real_part, sigma=1.0)
            imag_smoothed = gaussian_filter(imag_part, sigma=1.0)
            
            # 在零点位置使用修复后的值
            repaired_real = np.where(zero_mask_np, real_smoothed, real_part)
            repaired_imag = np.where(zero_mask_np, imag_smoothed, imag_part)
            
            # 重构复数并转换回GPU张量
            repaired_otf_np = repaired_real + 1j * repaired_imag
            repaired_otf = torch.from_numpy(repaired_otf_np).to(self.device, dtype=otf.dtype)
        
        return repaired_otf
    
    def estimate_kernel_iterative(self, 
                                sharp_img: np.ndarray, 
                                blurred_img: np.ndarray,
                                kernel_size: int = 15,
                                image_index: int = None) -> np.ndarray:
        """
        通过迭代优化求解精确卷积核（支持GPU加速）
        
        Args:
            sharp_img: 清晰图像
            blurred_img: 模糊图像
            kernel_size: 卷积核大小
            image_index: 图像索引（用于显示进度信息）
            
        Returns:
            estimated_kernel: 估计的卷积核
        """
        # 预处理图像
        sharp_gray, blurred_gray = self.preprocess_images(sharp_img, blurred_img)
        
        # 确保图像大小一致
        h, w = min(sharp_gray.shape[0], blurred_gray.shape[0]), min(sharp_gray.shape[1], blurred_gray.shape[1])
        sharp_gray = sharp_gray[:h, :w]
        blurred_gray = blurred_gray[:h, :w]
        
        if self.use_gpu:
            # GPU加速版本
            return self._estimate_kernel_gpu(sharp_gray, blurred_gray, kernel_size, image_index)
        else:
            # CPU版本
            return self._estimate_kernel_cpu(sharp_gray, blurred_gray, kernel_size, image_index)
    
    def _estimate_kernel_gpu(self, 
                           sharp_gray: np.ndarray, 
                           blurred_gray: np.ndarray,
                           kernel_size: int,
                           image_index: int = None) -> np.ndarray:
        """
        GPU加速的卷积核估计（简化稳定版本）
        """
        try:
            # 转换为PyTorch张量并移动到GPU
            sharp_tensor = torch.from_numpy(sharp_gray.astype(np.float32)).to(self.device)
            blurred_tensor = torch.from_numpy(blurred_gray.astype(np.float32)).to(self.device)
            
            # 计算傅里叶变换
            sharp_fft = torch.fft.fft2(sharp_tensor)
            blurred_fft = torch.fft.fft2(blurred_tensor)
            
            # 简化的卷积核估计：直接频域除法
            epsilon = 1e-8
            
            # 添加正则化避免除零
            sharp_magnitude = torch.abs(sharp_fft)
            regularized_sharp_fft = sharp_fft * (sharp_magnitude / (sharp_magnitude + epsilon))
            
            # 估计传递函数
            transfer_function = blurred_fft / (regularized_sharp_fft + epsilon)
            
            # 逆FFT得到空间域的卷积核
            kernel_full = torch.real(torch.fft.ifft2(transfer_function))
            
            # 提取中心区域作为卷积核
            h, w = kernel_full.shape
            center_h, center_w = h // 2, w // 2
            half_size = kernel_size // 2
            
            kernel = kernel_full[
                center_h - half_size:center_h - half_size + kernel_size,
                center_w - half_size:center_w - half_size + kernel_size
            ]
            
            # 归一化处理
            kernel = kernel - torch.min(kernel)  # 确保非负
            kernel_sum = torch.sum(kernel)
            if kernel_sum > 0:
                kernel = kernel / kernel_sum
            else:
                # 如果和为0，创建默认的中心卷积核
                kernel = torch.zeros_like(kernel)
                kernel[kernel_size//2, kernel_size//2] = 1.0
            
            if image_index is not None:
                # print(f"Image {image_index + 1}: Kernel estimation completed (GPU simplified version)")
                pass
            
            # 转换回numpy并返回
            return kernel.cpu().numpy().astype(np.float32)
            
        except Exception as e:
            # 如果GPU处理失败，回退到CPU处理
            print(f"GPU processing failed, falling back to CPU: {e}")
            return self._estimate_kernel_cpu(sharp_gray, blurred_gray, kernel_size, image_index)
    
    def _estimate_kernel_cpu(self, 
                           sharp_gray: np.ndarray, 
                           blurred_gray: np.ndarray,
                           kernel_size: int,
                           image_index: int = None) -> np.ndarray:
        """
        CPU版本的卷积核估计（原始实现）
        """
        h, w = sharp_gray.shape
        
        # 重置收敛过程数据
        self.convergence_errors = []
        self.kernel_evolution = []
        
        # 计算傅里叶变换
        sharp_fft = np.fft.fft2(sharp_gray)
        blurred_fft = np.fft.fft2(blurred_gray)
        
        # 检测零点（使用原始方法）
        sharp_magnitude = np.abs(sharp_fft)
        blurred_magnitude = np.abs(blurred_fft)
        epsilon = 1e-10
        sharp_magnitude = np.maximum(sharp_magnitude, epsilon)
        transfer_magnitude = blurred_magnitude / sharp_magnitude
        zero_mask = transfer_magnitude < 0.01
        
        # 初始化卷积核（高斯核作为初始估计）
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        kernel[center, center] = 1.0
        kernel = gaussian_filter(kernel, sigma=1.0)
        kernel = kernel / np.sum(kernel)
        
        # 存储初始卷积核
        if self.visualize_convergence:
            self.kernel_evolution.append(kernel.copy())
        
        prev_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # 计算当前卷积核的傅里叶变换
            kernel_padded = np.zeros_like(sharp_gray)
            kh, kw = kernel.shape
            start_h, start_w = (h - kh) // 2, (w - kw) // 2
            kernel_padded[start_h:start_h + kh, start_w:start_w + kw] = kernel
            kernel_fft = np.fft.fft2(kernel_padded)
            
            # 计算光学传递函数
            epsilon = 1e-10
            otf = kernel_fft / (np.abs(sharp_fft) + epsilon)
            
            # 应用相位一致性约束修复零点（简化版本）
            # 在第一次迭代时可视化零点修复过程
            visualize_zero_repair = (self.visualize_convergence and iteration == 0)
            save_path_zero = f"zero_repair_iter_{iteration}.png" if visualize_zero_repair else None
            
            otf_repaired = self.apply_phase_consistency_constraint(
                otf, zero_mask, visualize=visualize_zero_repair, save_path=save_path_zero
            )
            
            # 从修复的OTF估计新的卷积核
            estimated_blurred = otf_repaired * sharp_fft
            
            # 计算误差
            error = np.mean(np.abs(estimated_blurred - blurred_fft) ** 2)
            
            # 存储收敛数据
            if self.visualize_convergence:
                self.convergence_errors.append(error)
            
            # 检查收敛
            if abs(prev_error - error) < self.tolerance:
                if image_index is not None:
                    print(f"Image {image_index + 1}: Kernel converged at iteration {iteration + 1}")
                else:
                    print(f"Kernel converged at iteration {iteration + 1}")
                break
                
            prev_error = error
            
            # 更新卷积核估计（通过逆傅里叶变换）
            kernel_update_fft = blurred_fft / (sharp_fft + self.regularization)
            kernel_update = np.real(np.fft.ifft2(kernel_update_fft))
            
            # 提取中心区域作为新的卷积核
            center_h, center_w = h // 2, w // 2
            half_size = kernel_size // 2
            new_kernel = kernel_update[
                center_h - half_size:center_h - half_size + kernel_size,
                center_w - half_size:center_w - half_size + kernel_size
            ]
            
            # 归一化
            if np.sum(np.abs(new_kernel)) > 0:
                new_kernel = new_kernel / np.sum(np.abs(new_kernel))
                kernel = 0.7 * kernel + 0.3 * new_kernel  # 平滑更新
            
            # 存储卷积核演化
            if self.visualize_convergence:
                self.kernel_evolution.append(kernel.copy())
        
        return kernel
    
    def apply_phase_consistency_constraint(self, 
                                         otf: np.ndarray, 
                                         zero_mask: np.ndarray,
                                         visualize: bool = False,
                                         save_path: Optional[str] = None) -> np.ndarray:
        """
        CPU版本：应用相位一致性约束修复零点区域（简化版本）
        
        Args:
            otf: 光学传递函数
            zero_mask: 零点位置掩膜
            visualize: 是否可视化修复过程
            save_path: 可视化结果保存路径
        """
        repaired_otf = otf.copy()
        
        # 简化的修复：使用邻近值的平均
        if np.any(zero_mask):
            from scipy.ndimage import gaussian_filter
            
            # 对实部和虚部分别进行高斯滤波
            real_part = np.real(otf)
            imag_part = np.imag(otf)
            
            real_smoothed = gaussian_filter(real_part, sigma=1.0)
            imag_smoothed = gaussian_filter(imag_part, sigma=1.0)
            
            # 在零点位置使用平滑后的值
            repaired_otf = np.where(
                zero_mask,
                real_smoothed + 1j * imag_smoothed,
                otf
            )
            
            # 可视化零点修复过程
            if visualize:
                self._visualize_zero_point_repair(
                    otf, repaired_otf, zero_mask, real_part, imag_part, 
                    real_smoothed, imag_smoothed, save_path
                )
        
        return repaired_otf
    
    def _visualize_zero_point_repair(self, 
                                   original_otf: np.ndarray,
                                   repaired_otf: np.ndarray,
                                   zero_mask: np.ndarray,
                                   real_part: np.ndarray,
                                   imag_part: np.ndarray,
                                   real_smoothed: np.ndarray,
                                   imag_smoothed: np.ndarray,
                                   save_path: Optional[str] = None):
        """可视化零点修复过程"""
        print("=== 开始可视化零点修复过程 ===")
        
        # 启用交互模式
        plt.ion()
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Zero Point Repair Process', fontsize=16)
        
        # 原始OTF的实部和虚部
        im1 = axes[0, 0].imshow(real_part, cmap='RdBu_r')
        axes[0, 0].set_title('Original OTF Real Part')
        axes[0, 0].axis('on')  # 显示坐标轴
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        
        im2 = axes[0, 1].imshow(imag_part, cmap='RdBu_r')
        axes[0, 1].set_title('Original OTF Imaginary Part')
        axes[0, 1].axis('on')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        
        # 零点掩膜 - 增强显示
        im_mask = axes[0, 2].imshow(zero_mask.astype(float), cmap='Reds', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Zero Points Mask\n({np.sum(zero_mask)} zeros found)')
        axes[0, 2].axis('on')
        plt.colorbar(im_mask, ax=axes[0, 2], shrink=0.8)
        
        # 原始OTF幅度
        original_magnitude = np.abs(original_otf)
        im3 = axes[0, 3].imshow(np.log(original_magnitude + 1e-10), cmap='hot')
        axes[0, 3].set_title('Original OTF Magnitude (log)')
        axes[0, 3].axis('on')
        plt.colorbar(im3, ax=axes[0, 3], shrink=0.8)
        
        # 平滑后的实部和虚部
        im4 = axes[1, 0].imshow(real_smoothed, cmap='RdBu_r')
        axes[1, 0].set_title('Smoothed Real Part')
        axes[1, 0].axis('on')
        plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
        
        im5 = axes[1, 1].imshow(imag_smoothed, cmap='RdBu_r')
        axes[1, 1].set_title('Smoothed Imaginary Part')
        axes[1, 1].axis('on')
        plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
        
        # 修复后的OTF幅度
        repaired_magnitude = np.abs(repaired_otf)
        im6 = axes[1, 2].imshow(np.log(repaired_magnitude + 1e-10), cmap='hot')
        axes[1, 2].set_title('Repaired OTF Magnitude (log)')
        axes[1, 2].axis('on')
        plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
        
        # 修复前后的差异
        magnitude_diff = repaired_magnitude - original_magnitude
        im7 = axes[1, 3].imshow(magnitude_diff, cmap='RdBu_r')
        axes[1, 3].set_title('Magnitude Difference')
        axes[1, 3].axis('on')
        plt.colorbar(im7, ax=axes[1, 3], shrink=0.8)
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            zero_repair_path = save_path.replace('.png', '_zero_repair.png')
            plt.savefig(zero_repair_path, dpi=300, bbox_inches='tight')
            print(f"零点修复可视化已保存到: {zero_repair_path}")
        
        # 强制显示图像
        plt.draw()
        plt.pause(0.1)  # 短暂暂停确保图像显示
        
        print("零点修复可视化已显示，按任意键继续...")
        input()  # 等待用户输入
        
        plt.close(fig)
        plt.ioff()  # 关闭交互模式
    
    def visualize_convergence_process(self, save_path: Optional[str] = None):
        """
        可视化迭代收敛过程
        
        Args:
            save_path: 保存路径（可选）
        """
        if not self.visualize_convergence or not self.convergence_errors:
            print("No convergence data available. Set visualize_convergence=True when initializing.")
            return
        
        print("=== 开始可视化收敛过程 ===")
        
        # 启用交互模式
        plt.ion()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Kernel Estimation Convergence Process', fontsize=16)
        
        # 1. 误差收敛曲线
        axes[0, 0].plot(self.convergence_errors, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Reconstruction Error')
        axes[0, 0].set_title(f'Convergence Curve\n(Final Error: {self.convergence_errors[-1]:.2e})')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # 2. 误差变化率
        if len(self.convergence_errors) > 1:
            error_rates = np.diff(self.convergence_errors)
            axes[0, 1].plot(error_rates, 'r-', linewidth=2, marker='s', markersize=4)
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Error Change Rate')
            axes[0, 1].set_title('Error Change Rate')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        else:
            axes[0, 1].text(0.5, 0.5, 'Need >1 iteration\nfor change rate', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Error Change Rate (N/A)')
        
        # 3. 最终卷积核
        if self.kernel_evolution:
            final_kernel = self.kernel_evolution[-1]
            im_final = axes[0, 2].imshow(final_kernel, cmap='hot', interpolation='nearest')
            axes[0, 2].set_title('Final Estimated Kernel')
            axes[0, 2].axis('on')
            plt.colorbar(im_final, ax=axes[0, 2], shrink=0.8)
        else:
            axes[0, 2].text(0.5, 0.5, 'No kernel\nevolution data', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Final Kernel (N/A)')
        
        # 4-6. 卷积核演化过程（显示关键时刻）
        if self.kernel_evolution and len(self.kernel_evolution) > 1:
            num_kernels = len(self.kernel_evolution)
            # 选择显示的迭代点：开始、1/4、1/2、3/4、结束（但只显示3个）
            if num_kernels <= 3:
                show_indices = list(range(num_kernels))
            else:
                show_indices = [0, num_kernels//2, num_kernels-1]
            
            iteration_labels = ['Initial', 'Middle', 'Final']
            
            for i, idx in enumerate(show_indices):
                if i < 3:  # 显示在第二行的三个位置
                    kernel = self.kernel_evolution[idx]
                    im = axes[1, i].imshow(kernel, cmap='hot', interpolation='nearest')
                    axes[1, i].set_title(f'{iteration_labels[i]} Kernel\n(Iteration {idx})')
                    axes[1, i].axis('on')
                    plt.colorbar(im, ax=axes[1, i], shrink=0.8)
        else:
            # 如果没有足够的演化数据，显示提示信息
            for i in range(3):
                axes[1, i].text(0.5, 0.5, f'Kernel evolution\ndata unavailable\n(Position {i+1})', 
                               ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'Kernel Evolution {i+1} (N/A)')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            convergence_path = save_path.replace('.png', '_convergence.png')
            plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
            print(f"收敛过程可视化已保存到: {convergence_path}")
        
        # 强制显示图像
        plt.draw()
        plt.pause(0.1)
        
        print("收敛过程可视化已显示，按任意键继续...")
        input()  # 等待用户输入
        
        plt.close(fig)
        plt.ioff()  # 关闭交互模式
    
    def visualize_kernel_evolution_animation(self, save_path: Optional[str] = None):
        """
        创建卷积核演化的动画可视化
        
        Args:
            save_path: 保存路径（可选）
        """
        if not self.kernel_evolution:
            print("No kernel evolution data available.")
            return
        
        print("=== 开始可视化核演化动画 ===")
        
        # 启用交互模式
        plt.ion()
        
        fig = plt.figure(figsize=(18, 6))
        fig.suptitle('Kernel Evolution Animation', fontsize=16)
        
        # 计算全局颜色范围
        all_kernels = np.array(self.kernel_evolution)
        vmin, vmax = np.min(all_kernels), np.max(all_kernels)
        
        # 只显示关键帧而不是所有帧（避免动画过长）
        num_kernels = len(self.kernel_evolution)
        if num_kernels > 10:
            # 如果帧数太多，采样显示
            step = max(1, num_kernels // 10)
            display_indices = list(range(0, num_kernels, step))
            if display_indices[-1] != num_kernels - 1:
                display_indices.append(num_kernels - 1)
        else:
            display_indices = list(range(num_kernels))
        
        print(f"将显示 {len(display_indices)} 帧核演化过程...")
        
        for frame_idx, kernel_idx in enumerate(display_indices):
            kernel = self.kernel_evolution[kernel_idx]
            
            # 清除之前的图像
            fig.clear()
            
            # 创建子图
            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2, projection='3d') 
            ax3 = fig.add_subplot(1, 3, 3)
            
            # 1. 显示当前卷积核（2D热图）
            im1 = ax1.imshow(kernel, cmap='hot', vmin=vmin, vmax=vmax, interpolation='nearest')
            ax1.set_title(f'Kernel at Iteration {kernel_idx}\n(Frame {frame_idx+1}/{len(display_indices)})')
            ax1.axis('on')
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # 2. 显示卷积核的3D视图
            x, y = np.meshgrid(range(kernel.shape[1]), range(kernel.shape[0]))
            surf = ax2.plot_surface(x, y, kernel, cmap='hot', alpha=0.8, 
                                  vmin=vmin, vmax=vmax, linewidth=0)
            ax2.set_title(f'3D View - Iteration {kernel_idx}')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Kernel Value')
            ax2.set_zlim(vmin, vmax)
            
            # 3. 显示收敛曲线（到当前迭代）
            current_errors = self.convergence_errors[:kernel_idx+1] if kernel_idx < len(self.convergence_errors) else self.convergence_errors
            if current_errors:
                ax3.plot(current_errors, 'b-', linewidth=2, marker='o', markersize=4)
                ax3.axvline(x=kernel_idx, color='r', linestyle='--', alpha=0.7, label=f'Current Iter')
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Reconstruction Error')
                ax3.set_title('Convergence Progress')
                ax3.grid(True, alpha=0.3)
                ax3.set_yscale('log')
                ax3.legend()
                
                if len(current_errors) > 1:
                    ax3.set_xlim(0, len(self.convergence_errors)-1)
                    ax3.set_ylim(min(self.convergence_errors)*0.9, max(self.convergence_errors)*1.1)
            else:
                ax3.text(0.5, 0.5, 'No convergence\ndata available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Convergence (N/A)')
            
            plt.tight_layout()
            
            # 保存最终帧
            if save_path and frame_idx == len(display_indices) - 1:
                animation_path = save_path.replace('.png', f'_evolution_final.png')
                plt.savefig(animation_path, dpi=300, bbox_inches='tight')
                print(f"核演化最终帧已保存到: {animation_path}")
            
            # 显示当前帧
            plt.draw()
            plt.pause(0.5)  # 暂停0.5秒观察每一帧
            
            print(f"显示第 {frame_idx+1}/{len(display_indices)} 帧 (迭代 {kernel_idx})")
        
        print("核演化动画播放完毕，按任意键继续...")
        input()  # 等待用户输入
        
        plt.close(fig)
        plt.ioff()  # 关闭交互模式
    
    def visualize_results(self, 
                         sharp_img: np.ndarray,
                         blurred_img: np.ndarray, 
                         estimated_kernel: np.ndarray,
                         save_path: Optional[str] = None):
        """
        可视化傅里叶变换与最终卷积核
        
        Args:
            sharp_img: 清晰图像
            blurred_img: 模糊图像
            estimated_kernel: 估计的卷积核
            save_path: 保存路径（可选）
        """
        print("=== 开始可视化基础结果 ===")
        
        # 启用交互模式
        plt.ion()
        
        # 预处理图像
        sharp_gray, blurred_gray = self.preprocess_images(sharp_img, blurred_img)
        
        # 计算傅里叶变换
        sharp_fft = np.fft.fft2(sharp_gray)
        blurred_fft = np.fft.fft2(blurred_gray)
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Kernel Estimation Results Analysis', fontsize=16)
        
        # 原始图像
        axes[0, 0].imshow(sharp_gray, cmap='gray')
        axes[0, 0].set_title('Sharp Image')
        axes[0, 0].axis('on')
        
        axes[0, 1].imshow(blurred_gray, cmap='gray')
        axes[0, 1].set_title('Blurred Image')
        axes[0, 1].axis('on')
        
        # 估计的卷积核
        im_kernel = axes[0, 2].imshow(estimated_kernel, cmap='hot', interpolation='nearest')
        axes[0, 2].set_title(f'Estimated Kernel\n(Size: {estimated_kernel.shape})')
        axes[0, 2].axis('on')
        plt.colorbar(im_kernel, ax=axes[0, 2], shrink=0.8)
        
        # 傅里叶变换幅度谱（对数尺度）
        sharp_magnitude = np.log(np.abs(np.fft.fftshift(sharp_fft)) + 1)
        blurred_magnitude = np.log(np.abs(np.fft.fftshift(blurred_fft)) + 1)
        
        im1 = axes[1, 0].imshow(sharp_magnitude, cmap='hot')
        axes[1, 0].set_title('Sharp Image Spectrum (log)')
        axes[1, 0].axis('on')
        plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)
        
        im2 = axes[1, 1].imshow(blurred_magnitude, cmap='hot')
        axes[1, 1].set_title('Blurred Image Spectrum (log)')
        axes[1, 1].axis('on')
        plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)
        
        # 卷积核的傅里叶变换
        kernel_padded = np.zeros_like(sharp_gray)
        kh, kw = estimated_kernel.shape
        start_h = (kernel_padded.shape[0] - kh) // 2
        start_w = (kernel_padded.shape[1] - kw) // 2
        kernel_padded[start_h:start_h + kh, start_w:start_w + kw] = estimated_kernel
        
        kernel_fft = np.fft.fft2(kernel_padded)
        kernel_magnitude = np.log(np.abs(np.fft.fftshift(kernel_fft)) + 1)
        
        im3 = axes[1, 2].imshow(kernel_magnitude, cmap='hot')
        axes[1, 2].set_title('Kernel Spectrum (log)')
        axes[1, 2].axis('on')
        plt.colorbar(im3, ax=axes[1, 2], shrink=0.8)
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"基础结果可视化已保存到: {save_path}")
        
        # 强制显示图像
        plt.draw()
        plt.pause(0.1)
        
        print("基础结果可视化已显示，按任意键继续...")
        input()  # 等待用户输入
        
        plt.close(fig)
        plt.ioff()  # 关闭交互模式

class MotionBlurDataset(Dataset):
    """
    运动模糊数据集类
    用于读取图片对：原图和运动模糊后的图片
    原图来自Data/CASIA，模糊图来自Data/CASIA_MotionBlurred
    每个人只读取前3张图片
    """
    
    def __init__(self, 
                 original_root: str, 
                 blurred_root: str, 
                 max_images_per_folder: int = 3, 
                 transform=None,
                 estimate_kernels: bool = False):
        """
        初始化数据集
        
        Args:
            original_root: 原图数据根目录路径 (Data/CASIA)
            blurred_root: 模糊图数据根目录路径 (Data/CASIA_MotionBlurred)
            max_images_per_folder: 每个子文件夹最多读取的图片数量
            transform: 图像变换函数
            estimate_kernels: 是否估计卷积核
        """
        self.original_root = original_root
        self.blurred_root = blurred_root
        self.max_images_per_folder = max_images_per_folder
        self.transform = transform
        self.estimate_kernels = estimate_kernels
        
        # 存储图片路径对
        self.original_paths = []
        self.blurred_paths = []
        self.person_ids = []  # 存储人员ID
        
        # 如果需要估计卷积核，初始化估计器
        if self.estimate_kernels:
            # 使用更稳定的简化版本避免GPU数据类型问题
            self.kernel_estimator = MotionBlurKernelEstimator(
                max_iterations=20,  # 减少迭代次数加速
                tolerance=1e-5,
                use_gpu=True,
                visualize_convergence=True  # 启用收敛过程可视化
            )
            self.estimated_kernels = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """加载数据集，从每个子文件夹读取前3张图片对"""
        
        # 获取原图文件夹中的所有子文件夹
        original_subfolders = [f for f in os.listdir(self.original_root) 
                              if os.path.isdir(os.path.join(self.original_root, f))]
        
        # 获取模糊图文件夹中的所有子文件夹
        blurred_subfolders = [f for f in os.listdir(self.blurred_root) 
                             if os.path.isdir(os.path.join(self.blurred_root, f))]
        
        # 找到两个文件夹都存在的人员ID
        common_subfolders = sorted(list(set(original_subfolders) & set(blurred_subfolders)))
        
        print(f"Found {len(original_subfolders)} subfolders in original image directory")
        print(f"Found {len(blurred_subfolders)} subfolders in blurred image directory")
        print(f"Common subfolders: {len(common_subfolders)}")
        
        for subfolder in common_subfolders:
            original_subfolder_path = os.path.join(self.original_root, subfolder)
            blurred_subfolder_path = os.path.join(self.blurred_root, subfolder)
            
            # 获取原图文件夹下的所有jpg图片
            original_image_files = glob.glob(os.path.join(original_subfolder_path, "*.jpg"))
            original_image_files.sort()  # 按文件名排序
            
            # 只取前max_images_per_folder张图片
            selected_original_images = original_image_files[:self.max_images_per_folder]
            
            for original_path in selected_original_images:
                # 从原图文件名构造对应的模糊图文件名
                original_filename = os.path.basename(original_path)
                # 原图格式：001.jpg，模糊图格式：001_blurred.jpg
                base_name = original_filename.replace('.jpg', '')
                blurred_filename = f"{base_name}_blurred.jpg"
                blurred_path = os.path.join(blurred_subfolder_path, blurred_filename)
                
                # 检查对应的模糊图是否存在
                if os.path.exists(blurred_path):
                    self.original_paths.append(original_path)
                    self.blurred_paths.append(blurred_path)
                    self.person_ids.append(subfolder)
                else:
                    print(f"Warning: Cannot find corresponding blurred image {blurred_path}")
        
        print(f"Total loaded image pairs: {len(self.original_paths)}")
        
        # 如果需要估计卷积核，为每对图片估计卷积核
        if self.estimate_kernels and len(self.original_paths) > 0:
            print("Starting kernel estimation...")
            self._estimate_all_kernels()
    
    def _estimate_all_kernels(self):
        """为所有图片对估计卷积核"""
        self.estimated_kernels = []
        
        for i, (original_path, blurred_path) in enumerate(zip(self.original_paths, self.blurred_paths)):
            try:
                # 加载图片
                original_img = np.array(Image.open(original_path).convert('RGB'))
                blurred_img = np.array(Image.open(blurred_path).convert('RGB'))
                
                # 估计卷积核
                kernel = self.kernel_estimator.estimate_kernel_iterative(
                    original_img, blurred_img, kernel_size=15, image_index=i
                )
                
                self.estimated_kernels.append(kernel)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(self.original_paths)} image pairs")
                    
            except Exception as e:
                print(f"Error estimating kernel for image pair {i}: {e}")
                # 添加一个默认的卷积核
                default_kernel = np.zeros((15, 15))
                default_kernel[7, 7] = 1.0
                self.estimated_kernels.append(default_kernel)
        
        print("Kernel estimation completed!")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.original_paths)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            如果estimate_kernels=False:
                original_image: 原图 (PIL图像或经过transform的tensor)
                blurred_image: 模糊图 (PIL图像或经过transform的tensor)
                person_id: 人员ID (string)
            如果estimate_kernels=True:
                original_image: 原图
                blurred_image: 模糊图
                estimated_kernel: 估计的卷积核 (numpy array)
                person_id: 人员ID (string)
        """
        original_path = self.original_paths[idx]
        blurred_path = self.blurred_paths[idx]
        person_id = self.person_ids[idx]
        
        # 加载原图
        try:
            original_image = Image.open(original_path).convert('RGB')
        except Exception as e:
            print(f"Cannot load original image {original_path}: {e}")
            # 返回一个空白图像
            original_image = Image.new('RGB', (224, 224), color='white')
        
        # 加载模糊图
        try:
            blurred_image = Image.open(blurred_path).convert('RGB')
        except Exception as e:
            print(f"Cannot load blurred image {blurred_path}: {e}")
            # 返回一个空白图像
            blurred_image = Image.new('RGB', (224, 224), color='white')
        
        # 应用变换
        if self.transform:
            original_image = self.transform(original_image)
            blurred_image = self.transform(blurred_image)
        
        if self.estimate_kernels:
            estimated_kernel = self.estimated_kernels[idx] if idx < len(self.estimated_kernels) else None
            return original_image, blurred_image, estimated_kernel, person_id
        else:
            return original_image, blurred_image, person_id
    
    def get_statistics(self):
        """获取数据集统计信息"""
        if not self.person_ids:
            return {}
        
        # 统计每个人的图片数量
        person_count = {}
        for person_id in self.person_ids:
            person_count[person_id] = person_count.get(person_id, 0) + 1
        
        stats = {
            'total_image_pairs': len(self.original_paths),
            'unique_persons': len(person_count),
            'avg_images_per_person': np.mean(list(person_count.values())),
            'min_images_per_person': min(person_count.values()),
            'max_images_per_person': max(person_count.values())
        }
        
        if self.estimate_kernels and self.estimated_kernels:
            # 添加卷积核统计信息
            kernel_sizes = [kernel.shape for kernel in self.estimated_kernels]
            stats['estimated_kernels_count'] = len(self.estimated_kernels)
            stats['kernel_size'] = kernel_sizes[0] if kernel_sizes else None
        
        return stats
    
    def visualize_sample_kernel_estimation(self, 
                                         idx: int, 
                                         save_path: Optional[str] = None,
                                         show_convergence: bool = True,
                                         show_animation: bool = False):
        """
        可视化指定样本的卷积核估计结果
        
        Args:
            idx: 样本索引
            save_path: 保存路径（可选）
            show_convergence: 是否显示收敛过程
            show_animation: 是否显示核演化动画
        """
        if not self.estimate_kernels:
            print("This dataset does not have kernel estimation enabled")
            return
        
        if idx >= len(self.original_paths):
            print(f"Index {idx} out of range")
            return
        
        # 加载图片
        original_img = np.array(Image.open(self.original_paths[idx]).convert('RGB'))
        blurred_img = np.array(Image.open(self.blurred_paths[idx]).convert('RGB'))
        
        if idx < len(self.estimated_kernels):
            estimated_kernel = self.estimated_kernels[idx]
            # 如果已有估计结果，重新运行以获取可视化数据
            print("Re-running estimation for visualization...")
            temp_estimator = MotionBlurKernelEstimator(
                max_iterations=20,
                tolerance=1e-5,
                use_gpu=True,
                visualize_convergence=True
            )
            estimated_kernel = temp_estimator.estimate_kernel_iterative(
                original_img, blurred_img, kernel_size=15, image_index=idx
            )
        else:
            # 实时估计
            self.kernel_estimator.visualize_convergence = True
            estimated_kernel = self.kernel_estimator.estimate_kernel_iterative(
                original_img, blurred_img, kernel_size=15, image_index=idx
            )
            temp_estimator = self.kernel_estimator
        
        # 基础可视化
        temp_estimator.visualize_results(
            original_img, blurred_img, estimated_kernel, save_path
        )
        
        # 显示收敛过程
        if show_convergence:
            temp_estimator.visualize_convergence_process(save_path)
        
        # 显示核演化动画
        if show_animation:
            temp_estimator.visualize_kernel_evolution_animation(save_path)


def main():
    """测试函数"""
    # 数据集路径
    original_root = "/home/wenhao/CUHK/ECE4513/FinalProject/Data/CASIA"
    blurred_root = "/home/wenhao/CUHK/ECE4513/FinalProject/Data/CASIA_MotionBlurred"
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"Detected GPU: {torch.cuda.get_device_name()}")
        use_gpu = True
    else:
        print("No GPU detected, using CPU")
        use_gpu = False
    
    print("Creating dataset instance (without kernel estimation)...")
    # 创建数据集实例（不估计卷积核，速度更快）
    dataset = MotionBlurDataset(
        original_root, 
        blurred_root, 
        max_images_per_folder=3,
        estimate_kernels=False
    )
    
    # 打印统计信息
    stats = dataset.get_statistics()
    print("\nDataset statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 显示前几个样本的信息
    print("\nFirst 5 samples:")
    for i in range(min(5, len(dataset))):
        original_image, blurred_image, person_id = dataset[i]
        original_path = dataset.original_paths[i]
        blurred_path = dataset.blurred_paths[i]
        original_filename = os.path.basename(original_path)
        blurred_filename = os.path.basename(blurred_path)
        print(f"Sample {i}: Person {person_id} - Original: {original_filename}, Blurred: {blurred_filename}")
        print(f"  Original size: {original_image.size if hasattr(original_image, 'size') else 'transformed'}")
        print(f"  Blurred size: {blurred_image.size if hasattr(blurred_image, 'size') else 'transformed'}")
    
    # 测试卷积核估计功能（仅对第一个样本）
    if len(dataset) > 0:
        print(f"\nTesting kernel estimation ({'GPU' if use_gpu else 'CPU'})...")
        
        # 创建卷积核估计器类，指定GPU使用
        class CustomMotionBlurDataset(MotionBlurDataset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if self.estimate_kernels:
                    self.kernel_estimator = MotionBlurKernelEstimator(
                        use_gpu=use_gpu, 
                        visualize_convergence=True,  # 启用可视化
                        max_iterations=15  # 减少迭代次数用于演示
                    )
        
        # 创建带卷积核估计的数据集（仅处理一个样本用于演示）
        dataset_with_kernels = CustomMotionBlurDataset(
            original_root, 
            blurred_root, 
            max_images_per_folder=1,  # 只处理第一张图片
            estimate_kernels=True
        )
        
        if len(dataset_with_kernels) > 0:
            print("\n=== 开始完整的可视化演示 ===")
            
            # 可视化第一个样本的卷积核估计结果（包含所有可视化功能）
            dataset_with_kernels.visualize_sample_kernel_estimation(
                0, 
                save_path="/home/wenhao/CUHK/ECE4513/FinalProject/PreProcess/MotionBlur/complete_kernel_analysis.png",
                show_convergence=True,   # 显示收敛过程
                show_animation=True      # 显示核演化动画
            )
            
            print("\n=== 可视化演示完成 ===")
            print("生成的可视化文件:")
            print("1. complete_kernel_analysis.png - 基础分析结果")
            print("2. complete_kernel_analysis_zero_repair.png - 零点修复过程")
            print("3. complete_kernel_analysis_convergence.png - 收敛过程分析")
            print("4. complete_kernel_analysis_evolution_final.png - 核演化最终帧")


if __name__ == "__main__":
    main()