import os
import shutil

def organize_images(source_dir, target_dir):
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有文件
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # 过滤图片文件
                label = file.split('_')[0]  # 提取标签，例如 S010
                label_dir = os.path.join(target_dir, label)

                # 如果标签文件夹不存在，则创建
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)

                # 复制文件到目标文件夹
                source_path = os.path.join(root, file)
                target_path = os.path.join(label_dir, file)
                shutil.copy2(source_path, target_path)
                print(f"Copied {source_path} to {target_path}")

if __name__ == "__main__":
    source_directory = "data/CK+1"  # 源文件夹路径
    target_directory = "data/CK+2"  # 目标文件夹路径
    organize_images(source_directory, target_directory)