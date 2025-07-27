import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
from typing import Tuple, Optional, Union, List
from scipy import interpolate, ndimage
import matplotlib
matplotlib.use('TkAgg')  # 设置matplotlib后端确保图像能够显示
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MotionBlurKernelSolver:
    """
    基于零点检测和相位约束的运动模糊核精确求解器
    
    实现功能：
    1. 从模糊图像频谱中检测OTF零点位置
    2. 应用相位一致性约束修复零点区域
    3. 通过迭代优化求解精确卷积核
    """
    
    def __init__(self, 
                 kernel_size: int = 25,
                 max_iterations: int = 30,
                 tolerance: float = 1e-6,
                 regularization: float = 0.01,
                 zero_threshold: float = 0.05,
                 visualize: bool = True):
        """
        初始化运动模糊核求解器
        
        Args:
            kernel_size: 卷积核大小
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            regularization: 正则化参数
            zero_threshold: 零点检测阈值
            visualize: 是否启用可视化
        """
        self.kernel_size = kernel_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.zero_threshold = zero_threshold
        self.visualize = visualize
        
        # 存储中间结果用于可视化
        self.convergence_errors = []
        self.kernel_evolution = []
        self.zero_points_history = []
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像：转换为灰度并归一化
        
        Args:
            image: 输入图像 (H, W, 3) 或 (H, W)
            
        Returns:
            processed_image: 预处理后的灰度图像 (H, W)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 归一化到 [0, 1]
        normalized = gray.astype(np.float64) / 255.0
        return normalized
    
    def detect_otf_zeros(self, 
                        sharp_fft: np.ndarray, 
                        blurred_fft: np.ndarray) -> np.ndarray:
        """
        检测光学传递函数(OTF)的零点位置
        
        Args:
            sharp_fft: 清晰图像的傅里叶变换
            blurred_fft: 模糊图像的傅里叶变换
            
        Returns:
            zero_mask: 零点位置掩膜 (bool数组)
        """
        # 计算幅度谱
        sharp_magnitude = np.abs(sharp_fft)
        blurred_magnitude = np.abs(blurred_fft)
        
        # 避免除零，添加小的正则化项
        epsilon = 1e-10
        sharp_magnitude = np.maximum(sharp_magnitude, epsilon)
        
        # 计算传递函数的幅度比
        transfer_ratio = blurred_magnitude / sharp_magnitude
        
        # 检测零点：传递函数幅度接近零的位置
        zero_mask = transfer_ratio < self.zero_threshold
        
        # 存储零点信息用于可视化
        if self.visualize:
            self.zero_points_history.append({
                'zero_mask': zero_mask.copy(),
                'transfer_ratio': transfer_ratio.copy(),
                'zero_count': np.sum(zero_mask)
            })
        
        return zero_mask
    
    def apply_phase_consistency_constraint(self, 
                                         otf: np.ndarray, 
                                         zero_mask: np.ndarray) -> np.ndarray:
        """
        应用相位一致性约束修复零点区域
        
        Args:
            otf: 光学传递函数 (复数数组)
            zero_mask: 零点位置掩膜
            
        Returns:
            repaired_otf: 修复后的光学传递函数
        """
        repaired_otf = otf.copy()
        
        if np.any(zero_mask):
            # 分离实部和虚部
            real_part = np.real(otf)
            imag_part = np.imag(otf)
            
            # 使用高斯滤波进行局部平滑
            sigma = 1.5  # 高斯核标准差
            real_smoothed = ndimage.gaussian_filter(real_part, sigma=sigma)
            imag_smoothed = ndimage.gaussian_filter(imag_part, sigma=sigma)
            
            # 在零点位置使用双线性插值修复
            real_repaired = self._interpolate_zeros(real_part, zero_mask)
            imag_repaired = self._interpolate_zeros(imag_part, zero_mask)
            
            # 结合平滑和插值结果
            alpha = 0.7  # 插值权重
            real_final = alpha * real_repaired + (1 - alpha) * real_smoothed
            imag_final = alpha * imag_repaired + (1 - alpha) * imag_smoothed
            
            # 在零点位置应用修复
            repaired_otf = np.where(
                zero_mask,
                real_final + 1j * imag_final,
                otf
            )
            
            # 应用相位一致性约束
            repaired_otf = self._enforce_phase_consistency(repaired_otf, zero_mask)
        
        return repaired_otf
    
    def _interpolate_zeros(self, data: np.ndarray, zero_mask: np.ndarray) -> np.ndarray:
        """
        使用双线性插值修复零点区域
        
        Args:
            data: 输入数据
            zero_mask: 零点掩膜
            
        Returns:
            interpolated_data: 插值修复后的数据
        """
        # 获取非零点的坐标和值
        valid_mask = ~zero_mask
        valid_coords = np.where(valid_mask)
        valid_values = data[valid_coords]
        
        if len(valid_values) == 0:
            return data
        
        # 获取需要插值的点坐标
        zero_coords = np.where(zero_mask)
        
        if len(zero_coords[0]) == 0:
            return data
        
        # 使用scipy进行插值
        try:
            from scipy.interpolate import griddata
            interpolated_values = griddata(
                np.column_stack(valid_coords),
                valid_values,
                np.column_stack(zero_coords),
                method='linear',
                fill_value=0.0
            )
            
            # 创建修复后的数组
            repaired_data = data.copy()
            repaired_data[zero_coords] = interpolated_values
            
        except Exception as e:
            print(f"Interpolation failed, using mean value: {e}")
            mean_value = np.mean(valid_values)
            repaired_data = data.copy()
            repaired_data[zero_mask] = mean_value
        
        return repaired_data
    
    def _enforce_phase_consistency(self, otf: np.ndarray, zero_mask: np.ndarray) -> np.ndarray:
        """
        强制应用相位一致性约束
        
        Args:
            otf: 光学传递函数
            zero_mask: 零点掩膜
            
        Returns:
            consistent_otf: 相位一致的传递函数
        """
        # 计算相位
        phase = np.angle(otf)
        magnitude = np.abs(otf)
        
        # 对相位进行平滑处理，确保一致性
        phase_smoothed = ndimage.gaussian_filter(phase, sigma=0.8)
        
        # 在零点区域应用平滑相位
        phase_consistent = np.where(zero_mask, phase_smoothed, phase)
        
        # 重构复数传递函数
        consistent_otf = magnitude * np.exp(1j * phase_consistent)
        
        return consistent_otf
    
    def estimate_kernel_iterative(self, 
                                sharp_img: np.ndarray, 
                                blurred_img: np.ndarray) -> np.ndarray:
        """
        通过迭代优化求解精确卷积核
        
        Args:
            sharp_img: 清晰图像
            blurred_img: 模糊图像
            
        Returns:
            estimated_kernel: 估计的卷积核
        """
        # 预处理图像
        sharp_gray = self.preprocess_image(sharp_img)
        blurred_gray = self.preprocess_image(blurred_img)
        
        # 确保图像大小一致
        h = min(sharp_gray.shape[0], blurred_gray.shape[0])
        w = min(sharp_gray.shape[1], blurred_gray.shape[1])
        sharp_gray = sharp_gray[:h, :w]
        blurred_gray = blurred_gray[:h, :w]
        
        # 计算傅里叶变换
        sharp_fft = np.fft.fft2(sharp_gray)
        blurred_fft = np.fft.fft2(blurred_gray)
        
        # 初始化卷积核（使用高斯核）
        kernel = self._initialize_kernel()
        
        # 重置收敛历史
        self.convergence_errors = []
        self.kernel_evolution = []
        self.zero_points_history = []
        
        prev_error = float('inf')
        
        print(f"Starting iterative optimization for {self.kernel_size}x{self.kernel_size} kernel...")
        
        for iteration in range(self.max_iterations):
            # Detect zero points
            zero_mask = self.detect_otf_zeros(sharp_fft, blurred_fft)
            
            # Estimate current transfer function
            epsilon = self.regularization
            otf_estimate = blurred_fft / (sharp_fft + epsilon)
            
            # Apply phase consistency constraint to repair zero points
            otf_repaired = self.apply_phase_consistency_constraint(otf_estimate, zero_mask)
            
            # Update kernel estimation
            kernel_new = self._update_kernel_from_otf(otf_repaired, h, w)
            
            # Calculate reconstruction error
            reconstructed_fft = otf_repaired * sharp_fft
            error = np.mean(np.abs(reconstructed_fft - blurred_fft) ** 2)
            
            # Store convergence information
            self.convergence_errors.append(error)
            if self.visualize:
                self.kernel_evolution.append(kernel.copy())
            
            # Check convergence
            if abs(prev_error - error) < self.tolerance:
                print(f"Algorithm converged at iteration {iteration + 1}")
                break
            
            # Smooth kernel update
            alpha = 0.6  # Update step size
            kernel = alpha * kernel_new + (1 - alpha) * kernel
            kernel = self._normalize_kernel(kernel)
            
            prev_error = error
            
            if (iteration + 1) % 5 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Error: {error:.2e}, Zeros: {np.sum(zero_mask)}")
        
        return self._normalize_kernel(kernel)
    
    def _initialize_kernel(self) -> np.ndarray:
        """
        初始化卷积核
        
        Returns:
            初始卷积核
        """
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        
        # 使用高斯分布初始化
        sigma = self.kernel_size / 6.0
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                distance_sq = (i - center) ** 2 + (j - center) ** 2
                kernel[i, j] = np.exp(-distance_sq / (2 * sigma ** 2))
        
        return self._normalize_kernel(kernel)
    
    def _update_kernel_from_otf(self, otf: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        从传递函数更新卷积核
        
        Args:
            otf: 传递函数
            h, w: 图像尺寸
            
        Returns:
            更新后的卷积核
        """
        # 逆FFT得到空间域核
        kernel_full = np.real(np.fft.ifft2(otf))
        
        # 提取中心区域
        center_h, center_w = h // 2, w // 2
        half_size = self.kernel_size // 2
        
        start_h = max(0, center_h - half_size)
        end_h = min(h, center_h - half_size + self.kernel_size)
        start_w = max(0, center_w - half_size)
        end_w = min(w, center_w - half_size + self.kernel_size)
        
        kernel = kernel_full[start_h:end_h, start_w:end_w]
        
        # 确保核大小正确
        if kernel.shape != (self.kernel_size, self.kernel_size):
            kernel = cv2.resize(kernel, (self.kernel_size, self.kernel_size))
        
        return kernel
    
    def _normalize_kernel(self, kernel: np.ndarray) -> np.ndarray:
        """
        归一化卷积核
        
        Args:
            kernel: 输入卷积核
            
        Returns:
            归一化后的卷积核
        """
        # 确保非负
        kernel = np.maximum(kernel, 0)
        
        # 归一化
        total = np.sum(kernel)
        if total > 1e-10:
            kernel = kernel / total
        else:
            # 如果和为0，创建默认的中心卷积核
            kernel = np.zeros_like(kernel)
            center = kernel.shape[0] // 2
            kernel[center, center] = 1.0
        
        return kernel
    
    def visualize_results(self, 
                         sharp_img: np.ndarray,
                         blurred_img: np.ndarray,
                         estimated_kernel: np.ndarray,
                         save_path: Optional[str] = None):
        """
        可视化求解结果
        
        Args:
            sharp_img: 清晰图像
            blurred_img: 模糊图像
            estimated_kernel: 估计的卷积核
            save_path: 保存路径
        """
        if not self.visualize:
            return
        
        plt.ion()  # 启用交互模式
        
        # 创建综合结果图
        fig = plt.figure(figsize=(20, 15))
        
        # 预处理图像
        sharp_gray = self.preprocess_image(sharp_img)
        blurred_gray = self.preprocess_image(blurred_img)
        
        # 1. Original image comparison (first row)
        ax1 = plt.subplot(4, 4, 1)
        plt.imshow(sharp_gray, cmap='gray')
        plt.title('Sharp Image')
        plt.axis('off')
        
        ax2 = plt.subplot(4, 4, 2)
        plt.imshow(blurred_gray, cmap='gray')
        plt.title('Blurred Image')
        plt.axis('off')
        
        # 2. Estimated kernel (first row)
        ax3 = plt.subplot(4, 4, 3)
        im3 = plt.imshow(estimated_kernel, cmap='hot', interpolation='nearest')
        plt.title(f'Estimated Kernel ({self.kernel_size}x{self.kernel_size})')
        plt.colorbar(im3, shrink=0.8)
        
        # 3. Kernel 3D view (first row)
        ax4 = plt.subplot(4, 4, 4, projection='3d')
        x, y = np.meshgrid(range(self.kernel_size), range(self.kernel_size))
        ax4.plot_surface(x, y, estimated_kernel, cmap='hot', alpha=0.8)
        ax4.set_title('Kernel 3D View')
        
        # 4. Frequency domain analysis (second row)
        sharp_fft = np.fft.fft2(sharp_gray)
        blurred_fft = np.fft.fft2(blurred_gray)
        
        ax5 = plt.subplot(4, 4, 5)
        sharp_spectrum = np.log(np.abs(np.fft.fftshift(sharp_fft)) + 1)
        plt.imshow(sharp_spectrum, cmap='hot')
        plt.title('Sharp Image Spectrum')
        plt.axis('off')
        
        ax6 = plt.subplot(4, 4, 6)
        blurred_spectrum = np.log(np.abs(np.fft.fftshift(blurred_fft)) + 1)
        plt.imshow(blurred_spectrum, cmap='hot')
        plt.title('Blurred Image Spectrum')
        plt.axis('off')
        
        # 5. Zero point detection results (second row)
        if self.zero_points_history:
            zero_info = self.zero_points_history[-1]
            ax7 = plt.subplot(4, 4, 7)
            plt.imshow(zero_info['zero_mask'], cmap='Reds')
            plt.title(f'Detected Zeros\n(Total: {zero_info["zero_count"]})')
            plt.axis('off')
            
            ax8 = plt.subplot(4, 4, 8)
            plt.imshow(np.log(zero_info['transfer_ratio'] + 1e-10), cmap='viridis')
            plt.title('Transfer Function Ratio')
            plt.axis('off')
        
        # 6. Convergence process (third row)
        if self.convergence_errors:
            ax9 = plt.subplot(4, 4, 9)
            plt.plot(self.convergence_errors, 'b-', linewidth=2, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('Reconstruction Error')
            plt.title('Convergence Curve')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            # Error change rate
            if len(self.convergence_errors) > 1:
                ax10 = plt.subplot(4, 4, 10)
                error_rates = np.diff(self.convergence_errors)
                plt.plot(error_rates, 'r-', linewidth=2, marker='s')
                plt.xlabel('Iteration')
                plt.ylabel('Error Change Rate')
                plt.title('Error Change Rate')
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 7. Kernel evolution (third row last two positions)
        if self.kernel_evolution and len(self.kernel_evolution) >= 3:
            positions = [11, 12]  # Third row, 3rd and 4th positions
            indices = [0, len(self.kernel_evolution)//2, -1]
            labels = ['Initial Kernel', 'Middle Kernel', 'Final Kernel']
            
            for i, (pos, idx, label) in enumerate(zip(positions, indices[:2], labels[:2])):
                if i < len(positions) and idx < len(self.kernel_evolution):
                    ax = plt.subplot(4, 4, pos)
                    im = plt.imshow(self.kernel_evolution[idx], cmap='hot', interpolation='nearest')
                    plt.title(f'{label}')
                    plt.colorbar(im, shrink=0.6)
                    plt.axis('off')
        
        # 8. Zero point repair process (fourth row)
        if len(self.zero_points_history) >= 2:
            ax13 = plt.subplot(4, 4, 13)
            zero_evolution = [info['zero_count'] for info in self.zero_points_history]
            plt.plot(zero_evolution, 'g-', linewidth=2, marker='d')
            plt.xlabel('Iteration')
            plt.ylabel('Zero Count')
            plt.title('Zero Repair Process')
            plt.grid(True, alpha=0.3)
        
        # 9. Algorithm performance statistics (fourth row last position)
        ax16 = plt.subplot(4, 4, 16)
        plt.axis('off')
        
        # Display statistics
        stats_text = f"""Algorithm Statistics:
Iterations: {len(self.convergence_errors)}
Final Error: {self.convergence_errors[-1]:.2e}
Initial Zeros: {self.zero_points_history[0]['zero_count'] if self.zero_points_history else 'N/A'}
Final Zeros: {self.zero_points_history[-1]['zero_count'] if self.zero_points_history else 'N/A'}
Kernel Size: {self.kernel_size}x{self.kernel_size}
Tolerance: {self.tolerance:.0e}"""
        
        plt.text(0.05, 0.95, stats_text, transform=ax16.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save results
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization results saved to: {save_path}")
        
        plt.draw()
        plt.pause(0.1)
        print("Visualization displayed, press Enter to continue...")
        input()
        
        plt.close(fig)
        plt.ioff()


class MotionBlurDataset(Dataset):
    """
    运动模糊数据集类
    
    仿照GetRotatedDataset.py的结构，用于读取CASIA_MotionBlurred数据集
    每个人读取前3张图片，并可选择性地估计运动模糊核
    """
    
    def __init__(self, 
                 data_root: str = "Data/CASIA_MotionBlurred",
                 max_images_per_person: int = 3,
                 transform=None,
                 estimate_kernels: bool = False,
                 kernel_size: int = 25):
        """
        初始化运动模糊数据集
        
        Args:
            data_root: 数据集根目录
            max_images_per_person: 每个人最多读取的图片数量
            transform: 图像变换函数
            estimate_kernels: 是否估计运动模糊核
            kernel_size: 卷积核大小
        """
        self.data_root = data_root
        self.max_images_per_person = max_images_per_person
        self.transform = transform
        self.estimate_kernels = estimate_kernels
        self.kernel_size = kernel_size
        
        # 存储图片路径和标签
        self.image_paths = []
        self.person_ids = []
        self.estimated_kernels = []
        
        # 如果需要估计卷积核，初始化求解器
        if self.estimate_kernels:
            self.kernel_solver = MotionBlurKernelSolver(
                kernel_size=kernel_size,
                max_iterations=20,  # 减少迭代次数加速
                visualize=True
            )
        
        self._load_dataset()
    
    def _load_dataset(self):
        """加载数据集"""
        print("Loading motion blur dataset:", self.data_root)
        
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_root}")
        
        # Get all person folders
        person_folders = [f for f in os.listdir(self.data_root) 
                         if os.path.isdir(os.path.join(self.data_root, f))]
        person_folders.sort()
        
        print(f"Found {len(person_folders)} person folders")
        
        total_images = 0
        for person_id in person_folders:
            person_path = os.path.join(self.data_root, person_id)
            
            # Get all images for this person
            image_files = glob.glob(os.path.join(person_path, "*.jpg"))
            image_files.sort()
            
            # Take only the first max_images_per_person images
            selected_images = image_files[:self.max_images_per_person]
            
            for image_path in selected_images:
                self.image_paths.append(image_path)
                self.person_ids.append(person_id)
                total_images += 1
        
        print(f"Total loaded images: {total_images}")
        
        # If kernel estimation is needed
        if self.estimate_kernels and total_images > 0:
            print("Starting motion blur kernel estimation...")
            self._estimate_all_kernels()
    
    def _estimate_all_kernels(self):
        """Estimate motion blur kernels for all images (requires corresponding sharp images)"""
        print("Note: Motion blur kernel estimation requires corresponding sharp images")
        print("Currently processing only blurred images, using simplified kernel estimation")
        
        # Here you can implement blind deconvolution methods, or use full methods if clear image correspondence is available
        for i, blurred_path in enumerate(self.image_paths):
            try:
                # Load blurred image
                blurred_img = np.array(Image.open(blurred_path).convert('RGB'))
                
                # Simplified kernel estimation (assuming clear version exists, actual project needs to establish correspondence)
                # Create a default kernel as example
                default_kernel = self._create_default_kernel()
                self.estimated_kernels.append(default_kernel)
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(self.image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                default_kernel = self._create_default_kernel()
                self.estimated_kernels.append(default_kernel)
        
        print("Motion blur kernel estimation completed!")
    
    def _create_default_kernel(self) -> np.ndarray:
        """Create default motion blur kernel"""
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        
        # Create horizontal motion blur kernel
        length = self.kernel_size // 3
        start = center - length // 2
        end = center + length // 2
        kernel[center, start:end] = 1.0 / length
        
        return kernel
    
    def estimate_kernel_for_pair(self, 
                                sharp_img: np.ndarray, 
                                blurred_img: np.ndarray,
                                visualize: bool = True) -> np.ndarray:
        """
        为特定的清晰-模糊图片对估计运动模糊核
        
        Args:
            sharp_img: 清晰图像
            blurred_img: 模糊图像
            visualize: 是否可视化结果
            
        Returns:
            estimated_kernel: 估计的卷积核
        """
        if not hasattr(self, 'kernel_solver'):
            self.kernel_solver = MotionBlurKernelSolver(
                kernel_size=self.kernel_size,
                visualize=visualize
            )
        
        # Set visualization options
        self.kernel_solver.visualize = visualize
        
        print("Starting motion blur kernel estimation...")
        estimated_kernel = self.kernel_solver.estimate_kernel_iterative(sharp_img, blurred_img)
        
        if visualize:
            # Visualize results
            save_path = f"kernel_estimation_result_{self.kernel_size}x{self.kernel_size}.png"
            self.kernel_solver.visualize_results(
                sharp_img, blurred_img, estimated_kernel, save_path
            )
        
        return estimated_kernel
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            根据estimate_kernels参数返回不同的数据组合
        """
        image_path = self.image_paths[idx]
        person_id = self.person_ids[idx]
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Cannot load image {image_path}: {e}")
            # Return blank image
            image = Image.new('RGB', (224, 224), color='white')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        if self.estimate_kernels and idx < len(self.estimated_kernels):
            estimated_kernel = self.estimated_kernels[idx]
            return image, estimated_kernel, person_id, image_path
        else:
            return image, person_id, image_path
    
    def get_statistics(self):
        """Get dataset statistics"""
        # Count images per person
        person_count = {}
        for person_id in self.person_ids:
            person_count[person_id] = person_count.get(person_id, 0) + 1
        
        stats = {
            'total_images': len(self.image_paths),
            'unique_persons': len(person_count),
            'avg_images_per_person': np.mean(list(person_count.values())),
            'min_images_per_person': min(person_count.values()) if person_count else 0,
            'max_images_per_person': max(person_count.values()) if person_count else 0,
            'data_root': self.data_root
        }
        
        if self.estimate_kernels:
            stats['estimated_kernels_count'] = len(self.estimated_kernels)
            stats['kernel_size'] = f"{self.kernel_size}x{self.kernel_size}"
        
        return stats
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """
        Visualize specified sample
        
        Args:
            idx: Sample index
            save_path: Save path
        """
        if idx >= len(self):
            print(f"Index {idx} out of range")
            return
        
        # Get sample data
        data = self[idx]
        if self.estimate_kernels:
            image, kernel, person_id, image_path = data
        else:
            image, person_id, image_path = data
            kernel = None
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Display image
        plt.subplot(2, 2, 1)
        if hasattr(image, 'numpy'):  # If torch tensor
            img_np = image.numpy().transpose(1, 2, 0)
        else:
            img_np = np.array(image)
        plt.imshow(img_np)
        plt.title(f'Person {person_id}')
        plt.axis('off')
        
        # Display kernel (if available)
        if kernel is not None:
            plt.subplot(2, 2, 2)
            plt.imshow(kernel, cmap='hot', interpolation='nearest')
            plt.title('Estimated Motion Blur Kernel')
            plt.colorbar()
            
            # 3D view
            plt.subplot(2, 2, 3, projection='3d')
            x, y = np.meshgrid(range(kernel.shape[1]), range(kernel.shape[0]))
            plt.gca().plot_surface(x, y, kernel, cmap='hot', alpha=0.8)
            plt.title('Motion Blur Kernel 3D View')
        
        # Display file information
        plt.subplot(2, 2, 4)
        plt.axis('off')
        info_text = f"""Sample Information:
Index: {idx}
Person ID: {person_id}
File Path: {os.path.basename(image_path)}
Image Size: {img_np.shape}"""
        
        if kernel is not None:
            info_text += f"\nKernel Size: {kernel.shape}"
        
        plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample visualization saved to: {save_path}")
        
        plt.show()


def main():
    """Test function"""
    # Dataset path
    data_root = "/home/wenhao/CUHK/ECE4513/FinalProject/Data/CASIA_MotionBlurred"
    
    print("=== Motion Blur Dataset Loading Test ===")
    
    # Create dataset instance
    dataset = MotionBlurDataset(
        data_root=data_root,
        max_images_per_person=3,
        estimate_kernels=False  # First don't estimate kernels for faster testing
    )
    
    # Print statistics
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Show first few sample information
    print(f"\nFirst 5 sample information:")
    for i in range(min(5, len(dataset))):
        data = dataset[i]
        image, person_id, image_path = data
        print(f"Sample {i}: Person {person_id}, File: {os.path.basename(image_path)}")
    
    # Test motion blur kernel estimation (using simulated data)
    if len(dataset) > 0:
        print(f"\n=== Testing Motion Blur Kernel Estimation Algorithm ===")
        
        # Create test image pair
        print("Creating test image pair...")
        sharp_test = np.random.rand(128, 128) * 255
        
        # Create simple motion blur
        kernel_true = np.zeros((15, 15))
        kernel_true[7, 3:12] = 1/9  # Horizontal motion blur
        
        # Apply convolution to create blurred image
        blurred_test = cv2.filter2D(sharp_test, -1, kernel_true)
        blurred_test = np.clip(blurred_test, 0, 255)
        
        # Estimate kernel
        estimated_kernel = dataset.estimate_kernel_for_pair(
            sharp_test, blurred_test, visualize=True
        )
        
        print(f"True kernel size: {kernel_true.shape}")
        print(f"Estimated kernel size: {estimated_kernel.shape}")
        print(f"True kernel sum: {np.sum(kernel_true):.4f}")
        print(f"Estimated kernel sum: {np.sum(estimated_kernel):.4f}")
        
    print("\n=== Test Completed ===")


if __name__ == "__main__":
    main()
