import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 初始化 InsightFace 模型
app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 降低检测阈值以提高对小图像的敏感度
for model in app.models.values():
    if hasattr(model, 'det_thresh'):
        model.det_thresh = 0.1  # 默认通常是0.5，降低到0.1

def debug_face_detection(person_id='0000570'):
    person_path = os.path.join('Data/CASIA', person_id)
    print(f"调试人员: {person_id}")
    print(f"文件夹路径: {person_path}")
    
    if not os.path.exists(person_path):
        print(f"文件夹不存在: {person_path}")
        return
    
    img_names = sorted(os.listdir(person_path))
    print(f"图像文件数量: {len(img_names)}")
    print(f"文件列表: {img_names}")
    
    # 测试前几张图像
    for i, img_name in enumerate(img_names[1:6]):  # 跳过第一张，测试接下来的5张
        img_path = os.path.join(person_path, img_name)
        print(f"\n--- 测试图像 {i+1}: {img_name} ---")
        
        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            continue
            
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
            
        print(f"图像尺寸: {img.shape}")
        print(f"图像数据类型: {img.dtype}")
        
        # 检查图像是否为空或损坏
        if img.size == 0:
            print("图像为空")
            continue
        
        # 对于小图像，先放大到合适的尺寸
        original_img = img.copy()
        if img.shape[0] < 224 or img.shape[1] < 224:
            # 计算放大倍数，保持纵横比
            scale_factor = max(224 / img.shape[0], 224 / img.shape[1])
            new_h = int(img.shape[0] * scale_factor)
            new_w = int(img.shape[1] * scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"图像放大后尺寸: {img.shape}")
            
        # 尝试人脸检测
        try:
            faces = app.get(img)
            print(f"检测到的人脸数量: {len(faces)}")
            
            if len(faces) > 0:
                face = faces[0]
                print(f"人脸置信度: {face.det_score if hasattr(face, 'det_score') else 'N/A'}")
                print(f"人脸边界框: {face.bbox if hasattr(face, 'bbox') else 'N/A'}")
                print(f"特征向量维度: {face.normed_embedding.shape}")
                print("✅ 成功提取人脸特征")
            else:
                print("❌ 未检测到人脸")
                
                # 尝试原图
                faces_original = app.get(original_img)
                print(f"原图检测到的人脸数量: {len(faces_original)}")
                
        except Exception as e:
            print(f"检测过程中出错: {e}")

if __name__ == "__main__":
    # 测试几个失败的案例
    test_cases = ['0000570', '0353546', '1024093']
    
    for case in test_cases:
        debug_face_detection(case)
        print("\n" + "="*50 + "\n")
