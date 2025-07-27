import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 初始化 InsightFace 模型
app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 进一步降低检测阈值
for model in app.models.values():
    if hasattr(model, 'det_thresh'):
        model.det_thresh = 0.2
    if hasattr(model, 'nms_thresh'):
        model.nms_thresh = 0.3

def test_single_case(person_id='0000570'):
    person_path = os.path.join('Data/CASIA', person_id)
    img_names = sorted(os.listdir(person_path))
    success_count = 0
    total_count = 0
    
    print(f"测试人员: {person_id}")
    
    for img_name in img_names[1:]:  # 跳过第一张
        total_count += 1
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 放大图像
        if img.shape[0] < 224 or img.shape[1] < 224:
            scale_factor = max(300 / img.shape[0], 300 / img.shape[1])
            new_h = int(img.shape[0] * scale_factor)
            new_w = int(img.shape[1] * scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        faces = app.get(img)
        if len(faces) > 0:
            success_count += 1
    
    print(f"成功率: {success_count}/{total_count} = {success_count/total_count*100:.1f}%")
    return success_count, total_count

if __name__ == "__main__":
    test_cases = ['0000570', '0353546', '1024093']
    
    for case in test_cases:
        success, total = test_single_case(case)
        print(f"  --> {case}: {success}/{total} 成功\n")
