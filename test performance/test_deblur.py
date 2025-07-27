import numpy as np
import cv2
import pickle
import os
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis
import sys
from torch import nn
sys.path.append('/home/wenhao/CUHK/ECE4513/FinalProject/PreProcess/MotionBlur')
from DeConvModel import TinyFreqNet

# 初始化 InsightFace 模型（使用 GPU）
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 加载逆卷积模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deblur_model = TinyFreqNet(kernel_size=25)  # 输出25x25卷积核
checkpoint = torch.load('/home/wenhao/CUHK/ECE4513/FinalProject/training_output/models/best_model.pth', map_location=device)
deblur_model.load_state_dict(checkpoint['model_state_dict'])
deblur_model.to(device)
deblur_model.eval()

# 更激进的检测阈值优化
for model in app.models.values():
    if hasattr(model, 'det_thresh'):
        model.det_thresh = 0.15  # 进一步降低到0.15
    if hasattr(model, 'nms_thresh'):
        model.nms_thresh = 0.2   # 更低的NMS阈值


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def wiener_deconvolution(blurred_img, kernel, noise_var=0.01):
    """
    使用维纳滤波进行逆卷积去模糊
    
    Args:
        blurred_img: 模糊图像 (H, W) 或 (H, W, C)
        kernel: 运动模糊卷积核 (25, 25)
        noise_var: 噪声方差
    
    Returns:
        restored_img: 去模糊后的图像
    """
    if len(blurred_img.shape) == 3:
        # 彩色图像，分别处理每个通道
        restored_channels = []
        for c in range(blurred_img.shape[2]):
            channel = blurred_img[:, :, c]
            restored_channel = wiener_deconvolution_single_channel(channel, kernel, noise_var)
            restored_channels.append(restored_channel)
        return np.stack(restored_channels, axis=2)
    else:
        # 灰度图像
        return wiener_deconvolution_single_channel(blurred_img, kernel, noise_var)

def wiener_deconvolution_single_channel(blurred_channel, kernel, noise_var):
    """
    单通道维纳滤波去卷积
    """
    # 图像和卷积核的傅里叶变换
    blurred_fft = np.fft.fft2(blurred_channel)
    
    # 将25x25卷积核扩展到图像大小
    h, w = blurred_channel.shape
    kernel_padded = np.zeros((h, w))
    kh, kw = kernel.shape
    start_h, start_w = (h - kh) // 2, (w - kw) // 2
    kernel_padded[start_h:start_h + kh, start_w:start_w + kw] = kernel
    
    kernel_fft = np.fft.fft2(kernel_padded)
    
    # 维纳滤波
    kernel_conj = np.conj(kernel_fft)
    kernel_mag_sq = np.abs(kernel_fft) ** 2
    
    # 维纳滤波器
    wiener_filter = kernel_conj / (kernel_mag_sq + noise_var)
    
    # 应用滤波器
    restored_fft = blurred_fft * wiener_filter
    restored_img = np.real(np.fft.ifft2(restored_fft))
    
    # 裁剪到合理范围
    restored_img = np.clip(restored_img, 0, 255)
    
    return restored_img

def predict_kernel_and_deblur(img):
    """
    使用TinyFreqNet预测卷积核并进行去模糊
    
    Args:
        img: 输入的模糊图像 (BGR格式)
    
    Returns:
        deblurred_img: 去模糊后的图像
        predicted_kernel: 预测的25x25卷积核
    """
    # 转换为灰度图并归一化
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 调整到模型输入尺寸 (112, 96)
    resized = cv2.resize(gray, (96, 112))
    normalized = resized.astype(np.float32) / 255.0
    
    # 转换为张量
    input_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    # 预测卷积核（频域）
    with torch.no_grad():
        kernel_fft_pred = deblur_model(input_tensor)  # [1, 25, 25] complex
    
    # 转换到空间域
    # 新的网络直接输出频域的卷积核，需要转换到空间域
    kernel_spatial = torch.real(torch.fft.ifft2(kernel_fft_pred)).squeeze().cpu().numpy()
    
    # 检查卷积核是否有效
    kernel_sum = np.sum(np.abs(kernel_spatial))
    if kernel_sum > 1e-8:
        # 归一化卷积核
        kernel_spatial = kernel_spatial / kernel_sum
    else:
        # 如果预测失败，使用默认卷积核
        print("Warning: Predicted kernel is invalid, using default kernel")
        kernel_spatial = np.zeros((25, 25))
        kernel_spatial[12, 12] = 1.0
    
    # 确保卷积核为正值（物理约束）
    kernel_spatial = np.abs(kernel_spatial)
    kernel_spatial = kernel_spatial / (np.sum(kernel_spatial) + 1e-8)
    
    # 使用维纳滤波进行去模糊
    img_float = img.astype(np.float32)
    deblurred_img = wiener_deconvolution(img_float, kernel_spatial, noise_var=0.01)
    
    # 转换回uint8
    deblurred_img = np.clip(deblurred_img, 0, 255).astype(np.uint8)
    
    return deblurred_img, kernel_spatial

def evaluate(test_dir='Data/CASIA', gallery_path='gallery.pkl', threshold=0.35):
    with open(gallery_path, 'rb') as f:
        gallery = pickle.load(f)

    correct = 0
    total = 0
    for person in os.listdir(test_dir):
        person_path = os.path.join(test_dir, person)
        if not os.path.isdir(person_path):
            continue
        test_img_path = os.path.join(person_path, '001.jpg')
        if not os.path.exists(test_img_path):
            if 1:
                # 如果旋转数据没有001.jpg，尝试读取第一个可用的图片
                img_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not img_files:
                    print(f"[!] {person}: 没有可用图片")
                    continue
                test_img_path = os.path.join(person_path, img_files[0])
        img = cv2.imread(test_img_path)
        if img is None:
            print(f"[!] {person}: 无法读取图像")
            continue
        
        # 先进行逆卷积去模糊处理
        deblurred_img, predicted_kernel = predict_kernel_and_deblur(img)
        
        # 多尺度检测策略（使用去模糊后的图像）
        faces = []
        
        # 策略1: 对于小图像，先放大到合适的尺寸
        if deblurred_img.shape[0] < 224 or deblurred_img.shape[1] < 224:
            # 尝试多种放大倍数
            scale_factors = [
                max(300 / deblurred_img.shape[0], 300 / deblurred_img.shape[1]),  # 原策略
                max(400 / deblurred_img.shape[0], 400 / deblurred_img.shape[1]),  # 更大放大
                max(224 / deblurred_img.shape[0], 224 / deblurred_img.shape[1])   # 保守放大
            ]
            
            for scale_factor in scale_factors:
                new_h = int(deblurred_img.shape[0] * scale_factor)
                new_w = int(deblurred_img.shape[1] * scale_factor)
                resized_img = cv2.resize(deblurred_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                detected_faces = app.get(resized_img)
                if len(detected_faces) > 0:
                    faces = detected_faces
                    break
        
        # 策略2: 如果放大后仍然检测不到，尝试原图
        if len(faces) == 0:
            faces = app.get(deblurred_img)
        
        # 策略3: 如果还是检测不到，尝试图像增强
        if len(faces) == 0:
            # 增强对比度
            enhanced_img = cv2.convertScaleAbs(deblurred_img, alpha=1.5, beta=30)
            if enhanced_img.shape[0] < 224 or enhanced_img.shape[1] < 224:
                scale_factor = max(300 / enhanced_img.shape[0], 300 / enhanced_img.shape[1])
                new_h = int(enhanced_img.shape[0] * scale_factor)
                new_w = int(enhanced_img.shape[1] * scale_factor)
                enhanced_img = cv2.resize(enhanced_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            faces = app.get(enhanced_img)
        
        if len(faces) == 0:
            print(f"[!] {person}: 未检测到人脸 (卷积核范数: {np.linalg.norm(predicted_kernel):.4f})")
            continue
        query_emb = faces[0].normed_embedding

        best_name, best_score = None, -1
        for name, emb_list in gallery.items():
            sims = [cosine_similarity(query_emb, emb) for emb in emb_list]
            max_sim = max(sims)
            if max_sim > best_score:
                best_score = max_sim
                best_name = name

        is_correct = (best_name == person) and (best_score >= threshold)
        result = "✓" if is_correct else "✗"
        kernel_norm = np.linalg.norm(predicted_kernel)
        print(f"[{result}] 预测: {best_name:10} 分数: {best_score:.3f} → 真实: {person} (核范数: {kernel_norm:.4f})")
        if is_correct:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"\n🎯 识别准确率: {acc*100:.2f}% （{correct}/{total}）")

if __name__ == "__main__":
    evaluate(test_dir='Data/CASIA_MotionBlurred', gallery_path='gallery.pkl', threshold=0.35)
