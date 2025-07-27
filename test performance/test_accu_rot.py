import numpy as np
import cv2
import pickle
import os
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis
import sys
sys.path.append('/home/wenhao/CUHK/ECE4513/FinalProject/PreProcess/Rotation')
from RotationModel import RotationModel

# 初始化 InsightFace 模型（使用 GPU）
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 加载旋转预测模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rotation_model = RotationModel(image_zise=(112, 96))
checkpoint = torch.load('/home/wenhao/CUHK/ECE4513/FinalProject/best_rotation_model.pth', map_location=device)
rotation_model.load_state_dict(checkpoint['model_state_dict'])
rotation_model.to(device)
rotation_model.eval()

# 更激进的检测阈值优化
for model in app.models.values():
    if hasattr(model, 'det_thresh'):
        model.det_thresh = 0.15  # 进一步降低到0.15
    if hasattr(model, 'nms_thresh'):
        model.nms_thresh = 0.2   # 更低的NMS阈值


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def crop_black_borders(img, tolerance=10):
    """裁剪图像的黑边部分"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 找到非黑色像素的边界
    coords = np.column_stack(np.where(gray > tolerance))
    if len(coords) == 0:
        return img  # 如果全是黑色，返回原图
    
    # 获取边界框
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 裁剪图像
    if len(img.shape) == 3:
        cropped = img[y_min:y_max+1, x_min:x_max+1, :]
    else:
        cropped = img[y_min:y_max+1, x_min:x_max+1]
    
    return cropped

def predict_rotation_and_correct(img):
    """预测图像旋转角度并进行旋转校正"""
    # 转换为灰度图像
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 调整图像大小到模型输入尺寸
    resized = cv2.resize(gray, (96, 112))
    
    # 归一化
    normalized = resized.astype(np.float32) / 255.0
    
    # 转换为张量
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    # 预测旋转角度
    with torch.no_grad():
        predicted_angle = rotation_model(tensor).item()
    
    # 如果预测角度接近0，直接返回原图
    if abs(predicted_angle) < 1.0:
        return img, predicted_angle
    
    # 旋转校正
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -predicted_angle, 1.0)
    corrected_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # 裁剪黑边
    corrected_img = crop_black_borders(corrected_img, tolerance=10)
    
    return corrected_img, predicted_angle

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
            if test_dir == 'Data/CASIA_Rotated':
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
        
        # 先进行旋转复原
        corrected_img, predicted_angle = predict_rotation_and_correct(img)
        
        # 多尺度检测策略（使用旋转校正后的图像）
        faces = []
        
        # 策略1: 对于小图像，先放大到合适的尺寸
        if corrected_img.shape[0] < 224 or corrected_img.shape[1] < 224:
            # 尝试多种放大倍数
            scale_factors = [
                max(300 / corrected_img.shape[0], 300 / corrected_img.shape[1]),  # 原策略
                max(400 / corrected_img.shape[0], 400 / corrected_img.shape[1]),  # 更大放大
                max(224 / corrected_img.shape[0], 224 / corrected_img.shape[1])   # 保守放大
            ]
            
            for scale_factor in scale_factors:
                new_h = int(corrected_img.shape[0] * scale_factor)
                new_w = int(corrected_img.shape[1] * scale_factor)
                resized_img = cv2.resize(corrected_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                detected_faces = app.get(resized_img)
                if len(detected_faces) > 0:
                    faces = detected_faces
                    break
        
        # 策略2: 如果放大后仍然检测不到，尝试原图
        if len(faces) == 0:
            faces = app.get(corrected_img)
        
        # 策略3: 如果还是检测不到，尝试图像增强
        if len(faces) == 0:
            # 增强对比度
            enhanced_img = cv2.convertScaleAbs(corrected_img, alpha=1.5, beta=30)
            if enhanced_img.shape[0] < 224 or enhanced_img.shape[1] < 224:
                scale_factor = max(300 / enhanced_img.shape[0], 300 / enhanced_img.shape[1])
                new_h = int(enhanced_img.shape[0] * scale_factor)
                new_w = int(enhanced_img.shape[1] * scale_factor)
                enhanced_img = cv2.resize(enhanced_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            faces = app.get(enhanced_img)
        
        if len(faces) == 0:
            print(f"[!] {person}: 未检测到人脸 (旋转角度: {predicted_angle:.1f}°)")
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
        print(f"[{result}] 预测: {best_name:10} 分数: {best_score:.3f} → 真实: {person} (旋转: {predicted_angle:.1f}°)")
        if is_correct:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"\n🎯 识别准确率: {acc*100:.2f}% （{correct}/{total}）")

if __name__ == "__main__":
    evaluate(test_dir='Data/CASIA_Rotated', gallery_path='gallery.pkl', threshold=0.35)
