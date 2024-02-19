import os
from glob import glob
import time

rootpath = r'D:/Python_file/HJX_file/DCGAN_hjx/data/faces/faces'
start_time = time.time()  # 计时开始

files = glob(os.path.join(rootpath, '*.jpg'))

# 首先，将所有文件重命名为临时名称
temp_files = []
for i, file_path in enumerate(files):
    temp_file_path = os.path.join(rootpath, f"temp_{i}.jpg")
    os.rename(file_path, temp_file_path)
    temp_files.append(temp_file_path)

# 然后，从这些临时文件名重命名到最终的文件名
for i, temp_file_path in enumerate(temp_files):
    final_file_path = os.path.join(rootpath, f"{i+1}.jpg")
    os.rename(temp_file_path, final_file_path)
    end_time = time.time()  # 更新结束时间
    print(f"Renamed to {final_file_path}. Time elapsed: {end_time - start_time} seconds")  # 打印每次重命名消耗的时间

# 注意：这段代码假设在重命名开始前，目录中不存在以 "temp_" 开头的文件名。

