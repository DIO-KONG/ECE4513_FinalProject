import os
from random import random
import cv2
import numpy as np

root_dir = "Data/CASIA"
def rotate_image(image_path, angle):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 随机选择填充方式
    pad_modes = [cv2.BORDER_REFLECT, cv2.BORDER_REPLICATE, cv2.BORDER_CONSTANT]
    pad_mode = pad_modes[np.random.randint(0, len(pad_modes))]
    if pad_mode == cv2.BORDER_CONSTANT:
        # 随机噪声填充
        value = np.random.randint(0, 256, size=3).tolist()
    else:
        value = None
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=pad_mode, borderValue=value)
    return rotated

for subdir, dirs, _ in os.walk(root_dir):
    for dir_name in dirs:
        dir_path = os.path.join(subdir, dir_name)
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.jpg'):
                file_path = os.path.join(dir_path, file_name)
                angle = random() * 360  # Random angle between 0 and 360
                rotated_image = rotate_image(file_path, angle)
                # Create corresponding directory in Data/CASIA_Rotated
                rotated_dir = dir_path.replace("Data/CASIA", "Data/CASIA_Rotated")
                os.makedirs(rotated_dir, exist_ok=True)
                rotated_file_path = os.path.join(rotated_dir, file_name[:-4] + '_' + str(round(angle, 3)) + '.jpg')
                cv2.imwrite(rotated_file_path, rotated_image)
                print(f"Rotated image saved to: {rotated_file_path}")
print("All images have been rotated and saved.")