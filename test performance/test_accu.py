import numpy as np
import cv2
import pickle
import os
import cv2
from insightface.app import FaceAnalysis

# 初始化 InsightFace 模型（使用 GPU）
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 更激进的检测阈值优化
for model in app.models.values():
    if hasattr(model, 'det_thresh'):
        model.det_thresh = 0.15  # 进一步降低到0.15
    if hasattr(model, 'nms_thresh'):
        model.nms_thresh = 0.2   # 更低的NMS阈值


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
        
        # 多尺度检测策略
        faces = []
        
        # 策略1: 对于小图像，先放大到合适的尺寸
        if img.shape[0] < 224 or img.shape[1] < 224:
            # 尝试多种放大倍数
            scale_factors = [
                max(300 / img.shape[0], 300 / img.shape[1]),  # 原策略
                max(400 / img.shape[0], 400 / img.shape[1]),  # 更大放大
                max(224 / img.shape[0], 224 / img.shape[1])   # 保守放大
            ]
            
            for scale_factor in scale_factors:
                new_h = int(img.shape[0] * scale_factor)
                new_w = int(img.shape[1] * scale_factor)
                resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                detected_faces = app.get(resized_img)
                if len(detected_faces) > 0:
                    faces = detected_faces
                    break
        
        # 策略2: 如果放大后仍然检测不到，尝试原图
        if len(faces) == 0:
            faces = app.get(img)
        
        # 策略3: 如果还是检测不到，尝试图像增强
        if len(faces) == 0:
            # 增强对比度
            enhanced_img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
            if enhanced_img.shape[0] < 224 or enhanced_img.shape[1] < 224:
                scale_factor = max(300 / enhanced_img.shape[0], 300 / enhanced_img.shape[1])
                new_h = int(enhanced_img.shape[0] * scale_factor)
                new_w = int(enhanced_img.shape[1] * scale_factor)
                enhanced_img = cv2.resize(enhanced_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            faces = app.get(enhanced_img)
        
        if len(faces) == 0:
            print(f"[!] {person}: 未检测到人脸")
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
        print(f"[{result}] 预测: {best_name:10} 分数: {best_score:.3f} → 真实: {person}")
        if is_correct:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"\n🎯 识别准确率: {acc*100:.2f}% （{correct}/{total}）")

if __name__ == "__main__":
    evaluate(test_dir='Data/CASIA_MotionBlurred', gallery_path='gallery.pkl', threshold=0.35)
