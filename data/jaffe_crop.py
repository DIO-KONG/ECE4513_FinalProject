import os
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt

# 定义输入和输出文件夹
input_dir = 'data/jaffe'
output_dir = 'data/jaffe_crop'
os.makedirs(output_dir, exist_ok=True)

# 初始化MTCNN检测器
detector = MTCNN()

# 遍历输入文件夹中的所有图像
for img_name in os.listdir(input_dir):
    if img_name.endswith('.tiff'):
        img_path = os.path.join(input_dir, img_name)
        try:
            # 打开图像并转换为RGB
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)

            # 检测人脸
            detections = detector.detect_faces(img_array)
            if len(detections) == 0:
                print(f"No face detected in {img_name}")
                continue

            # 假设每张图片只有一个人脸，裁剪人脸
            detection = detections[0]
            x, y, width, height = detection['box']
            cropped_face = img_array[y:y+height, x:x+width]

            # 转换为PIL图像并调整大小为 [168, 124]
            cropped_img = Image.fromarray(cropped_face).convert('L')  # 转为灰度图像
            resized_img = cropped_img.resize((128,128))  # 调整大小

            # 保存调整后的图像
            output_path = os.path.join(output_dir, img_name)
            resized_img.save(output_path)
            print(f"Processed and saved: {output_path}")

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# 可视化前三张裁剪效果
cropped_images = []
for idx, img_name in enumerate(os.listdir(output_dir)):
    if idx >= 3:
        break
    img_path = os.path.join(output_dir, img_name)
    cropped_images.append(Image.open(img_path))

# 显示裁剪后的图像
plt.figure(figsize=(12, 4))
for i, cropped_img in enumerate(cropped_images):
    plt.subplot(1, 3, i + 1)
    plt.imshow(np.array(cropped_img), cmap='gray')
    plt.title(f"Cropped Image {i + 1}")
    plt.axis('off')
plt.tight_layout()
plt.show()