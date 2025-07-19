import os

base_dir = 'data/CK+2'
subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
subfolders.sort()  # 按字母顺序排序

for idx, folder in enumerate(subfolders):
    old_path = os.path.join(base_dir, folder)
    new_path = os.path.join(base_dir, str(idx))
    os.rename(old_path, new_path)
    print(f'Renamed "{folder}" to "{idx}"')