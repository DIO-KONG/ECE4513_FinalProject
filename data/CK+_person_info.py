import os

base_dir = 'data/CK+2'
subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

file_counts = []
for folder in subfolders:
    count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    file_counts.append(count)

if file_counts:
    max_files = max(file_counts)
    min_files = min(file_counts)
    print(f"子文件夹数量: {len(subfolders)}")
    print(f"单个子文件夹下最大文件数: {max_files}")
    print(f"单个子文件夹下最小文件数: {min_files}")
    print(f"总文件数: {sum(file_counts)}")
else:
    print("没有找到子文件夹。")