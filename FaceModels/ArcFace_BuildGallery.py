import os
import cv2
import numpy as np
import pickle
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

def build_gallery(data_dir='Data/CASIA', save_path='gallery.pkl'):
    gallery = {}  # person_name -> list of embeddings
    person_list = [p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))]
    total_persons = len(person_list)
    successful_persons = 0
    total_images_processed = 0
    total_faces_extracted = 0
    
    print(f"开始处理 {total_persons} 个人员的人脸库...")
    
    for idx, person in enumerate(person_list):
        person_path = os.path.join(data_dir, person)
        img_names = sorted(os.listdir(person_path))
        if len(img_names) <= 1:
            continue  # 没有足够数据

        embeddings = []
        # 跳过第一张（通常是 001.jpg），用于测试
        for img_name in img_names[1:]:
            total_images_processed += 1
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
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
                continue
            emb = faces[0].normed_embedding
            embeddings.append(emb)
            total_faces_extracted += 1
        
        if len(embeddings) > 0:
            gallery[person] = embeddings
            successful_persons += 1
            success_rate = len(embeddings) / (len(img_names) - 1) * 100
            print(f"[✓] {person}: {len(embeddings)} 张图已建库 (成功率: {success_rate:.1f}%) [{idx+1}/{total_persons}]")
        else:
            print(f"[!] {person}: 无有效人脸 [{idx+1}/{total_persons}]")

    # 显示最终统计
    print(f"\n📊 构建完成统计:")
    print(f"   成功人员: {successful_persons}/{total_persons} ({successful_persons/total_persons*100:.1f}%)")
    print(f"   处理图像: {total_images_processed} 张")
    print(f"   提取特征: {total_faces_extracted} 个")
    print(f"   整体成功率: {total_faces_extracted/total_images_processed*100:.1f}%")

    # 保存
    with open(save_path, 'wb') as f:
        pickle.dump(gallery, f)
    print(f"\n✅ 人脸库已保存到 {save_path}")

# 构建并保存
build_gallery()
