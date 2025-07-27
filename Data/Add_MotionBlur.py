from MotionBlurKernel import Kernel
import os
from random import random

def add_motion_blur(image_path, kernel_size=(3, 3), nonlinearity=0.5):
    kernel = Kernel(size=kernel_size, intensity=nonlinearity)
    blurred_image = kernel.applyTo(image_path, keep_image_dim=True)
    return blurred_image
    # blurred_image.save("./output2.jpg", "JPEG")

root_dir = "Data/CASIA"
for subdir, dirs, _ in os.walk(root_dir):
    for dir_name in dirs:
        dir_path = os.path.join(subdir, dir_name)
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.jpg'):
                file_path = os.path.join(dir_path, file_name)
                blurred_image = add_motion_blur(file_path, kernel_size=(9, 9), nonlinearity=random()/2)
                # Create corresponding directory in Data/CASIA_MotionBlurred
                blurred_dir = dir_path.replace("Data/CASIA", "Data/CASIA_MotionBlurred")
                os.makedirs(blurred_dir, exist_ok=True)
                blurred_file_path = os.path.join(blurred_dir, file_name[:-4] + '_blurred.jpg')
                blurred_image.save(blurred_file_path, "JPEG")
                print(f"Blurred image saved to: {blurred_file_path}")
print("All images have been blurred and saved.")